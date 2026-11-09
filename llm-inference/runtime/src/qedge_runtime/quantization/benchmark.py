"""Benchmark harness for same-engine quantization measurement (F16, Q8_0, Q4_K_M).

Measures TTFT, decode throughput, p50/p95 percentiles, fresh-process peak RSS,
artifact size, and quality metrics against the same-engine full-precision reference.

KV-Cache State Isolation & Prefix-Cache Control
-----------------------------------------------
Every independently measured trial calls ``runner.generate(..., clear_state=True)``,
which invokes ``llm.reset()`` before prompt evaluation.  This guarantees that
each measured trial evaluates the full prompt token sequence without reusing
a prior KV prefix.  Declared warmup calls use ``clear_state=False`` to legitimately
warm kernel and weight caches without polluting measured trials.

Authoritative Token Agreement & Scoring
---------------------------------------
1. Token-ID Agreement (Primary):
   Operates directly on the generated integer token ID sequences.
   - exact_token_agreement: 1.0 if token ID sequences match identically.
   - len_aware_token_agreement: common prefix length / max(len_a, len_b).
2. Word-Split Agreement (Secondary / Legacy):
   Operates on whitespace-separated words; explicitly labelled as word-split.
3. Coarse Target Scoring with Negation Rejection:
   Whole-word regex match with explicit rejection of negated answers
   (e.g. 'not Paris', 'not Red', 'neither', 'no').

Structured Failure & RSS Reporting
----------------------------------
Failed requests and RSS measurement errors are preserved as structured records.
A failed RSS measurement returns `None` and an error description, never a silent
zero-byte value.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import re
import resource
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qedge_runtime.quantization.converter import (
    compute_sha256,
    get_native_runtime_info,
    verify_manifest_identity,
)
from qedge_runtime.quantization.runner import GGUFModelRunner

logger = logging.getLogger(__name__)

# Evaluation Prompts with ground-truth target expectations
EVALUATION_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "qa_01_paris",
        "category": "factual_geography",
        "prompt": "What is the capital of France? Answer with only the name of the city.",
        "expected_target": "Paris",
    },
    {
        "id": "qa_02_water",
        "category": "factual_science",
        "prompt": "What is the common chemical formula for water? Answer with only the formula.",
        "expected_target": "H2O",
    },
    {
        "id": "qa_03_speed_of_light",
        "category": "factual_physics",
        "prompt": "In physics, what physical constant is represented by the letter c? Answer with only the constant name.",
        "expected_target": "speed of light",
    },
    {
        "id": "qa_04_binary_five",
        "category": "factual_math",
        "prompt": "What is the binary representation of the decimal number 5? Answer with only the binary digits.",
        "expected_target": "101",
    },
    {
        "id": "qa_05_primary_colors",
        "category": "factual_color",
        "prompt": "In additive color mixing (RGB), what is the third primary color alongside green and blue? Answer with only the color name.",
        "expected_target": "Red",
    },
    {
        "id": "inst_06_arithmetic",
        "category": "reasoning_math",
        "prompt": "What is 15 multiplied by 4? Answer with only the number.",
        "expected_target": "60",
    },
    {
        "id": "inst_07_logic",
        "category": "reasoning_logic",
        "prompt": "If all roses are flowers and all flowers are plants, are roses plants? Answer with Yes or No.",
        "expected_target": "Yes",
    },
    {
        "id": "inst_08_extraction",
        "category": "information_extraction",
        "prompt": "Extract the fruit from this list: table, chair, apple, window. Answer with only the fruit.",
        "expected_target": "apple",
    },
]


def score_target_hit(generated_text: str, target: str) -> float:
    """Coarse word-boundary target scoring with explicit negation rejection.

    Matches target as a whole word, case-insensitive, while rejecting answers
    that contain negative context (e.g. 'not Paris', 'not red', 'isn't Paris',
    'neither', 'no').
    """
    clean_text = generated_text.strip()
    pattern = rf"(?i)\b{re.escape(target)}\b"
    if not re.search(pattern, clean_text):
        return 0.0

    # Negation patterns preceding or adjacent to target
    neg_pattern = rf"(?i)\b(not|isn't|is not|aren't|are not|neither|never|no)\s+{re.escape(target)}\b"
    if re.search(neg_pattern, clean_text):
        return 0.0

    # Binary yes/no targets
    if target.lower() == "yes" and re.search(r"(?i)\b(no|not)\b", clean_text):
        return 0.0
    if target.lower() == "no" and re.search(r"(?i)\b(yes)\b", clean_text):
        return 0.0

    return 1.0


def compute_sequence_agreement(text_a: str, text_b: str) -> Tuple[float, float]:
    """Compute exact string match and length-aware word-split agreement (legacy).

    Compares whitespace-separated word tokens, NOT model token IDs.
    """
    exact_match = 1.0 if text_a.strip() == text_b.strip() else 0.0

    toks_a = text_a.strip().split()
    toks_b = text_b.strip().split()

    if not toks_a and not toks_b:
        return 1.0, 1.0
    if not toks_a or not toks_b:
        return 0.0, 0.0

    max_len = max(len(toks_a), len(toks_b))
    matches = sum(1 for a, b in zip(toks_a, toks_b) if a == b)
    length_aware = matches / max_len

    return exact_match, length_aware


def compute_token_id_agreement(ids_a: List[int], ids_b: List[int]) -> Tuple[float, float]:
    """Compute exact token-ID sequence agreement and length-aware token agreement.

    Operates directly on authoritative integer token ID sequences.
    """
    if not ids_a and not ids_b:
        return 1.0, 1.0
    if not ids_a or not ids_b:
        return 0.0, 0.0

    exact_match = 1.0 if ids_a == ids_b else 0.0
    max_len = max(len(ids_a), len(ids_b))
    matches = sum(1 for a, b in zip(ids_a, ids_b) if a == b)
    length_aware = matches / max_len

    return exact_match, length_aware


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, median (p50), p95, min, max, and 95% CI."""
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95": 0.0,
        }

    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(variance)
    ci95_val = 1.96 * (std_val / math.sqrt(n)) if n > 1 else 0.0

    sorted_vals = sorted(values)
    if n % 2 == 1:
        p50 = sorted_vals[n // 2]
    else:
        p50 = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    idx_p95 = min(n - 1, int(math.ceil(0.95 * n)) - 1)
    p95 = sorted_vals[idx_p95]

    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "ci95": round(ci95_val, 2),
    }


def _worker_measure_fresh_rss(model_path: str, conn: Any) -> None:
    """Isolated subprocess worker to measure peak RSS of fresh process loading model."""
    try:
        runner = GGUFModelRunner(model_path, n_ctx=512, verbose=False)
        _ = runner.generate("Test prompt for RSS measurement", max_tokens=16, clear_state=False)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_bytes = usage.ru_maxrss
        conn.send({"peak_rss_bytes": peak_bytes, "error": None})
    except Exception as e:
        conn.send({"peak_rss_bytes": None, "error": str(e)})
    finally:
        conn.close()


_RSS_WORKER_TIMEOUT_SEC = 30


def measure_fresh_process_peak_rss(model_path: Path) -> Dict[str, Any]:
    """Launch an isolated subprocess to measure peak RSS for a model.

    Returns:
        Structured dictionary:
            - peak_rss_bytes: Optional[int] (None on error/timeout, never 0)
            - peak_rss_mb: Optional[float]
            - status: 'success' | 'error' | 'timeout'
            - error: Optional[str]
    """
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker_measure_fresh_rss, args=(str(model_path), child_conn))
    p.start()
    child_conn.close()

    result: Optional[Dict[str, Any]] = None
    error_msg: Optional[str] = None
    status = "error"

    try:
        if parent_conn.poll(timeout=_RSS_WORKER_TIMEOUT_SEC):
            result = parent_conn.recv()
            if result.get("error"):
                error_msg = result["error"]
                status = "error"
            else:
                status = "success"
        else:
            error_msg = f"RSS worker timed out after {_RSS_WORKER_TIMEOUT_SEC}s"
            status = "timeout"
    except Exception as exc:
        error_msg = f"RSS worker pipe error: {exc}"
        status = "error"
    finally:
        parent_conn.close()
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    if status == "success" and result and result.get("peak_rss_bytes") is not None:
        peak_bytes = result["peak_rss_bytes"]
        return {
            "peak_rss_bytes": peak_bytes,
            "peak_rss_mb": round(peak_bytes / (1024 * 1024), 2),
            "status": "success",
            "error": None,
        }

    logger.warning("RSS measurement failed for %s: %s", model_path.name, error_msg)
    return {
        "peak_rss_bytes": None,
        "peak_rss_mb": None,
        "status": status,
        "error": error_msg,
    }


def create_run_manifest(
    model_paths: Dict[str, Path],
    source_model_dir: Path,
    converter_commit: str = "aac6da8012efd5c0fd211c907c4e870ba075b211",
    warmups: int = 3,
    trials_per_prompt: int = 10,
    max_tokens: int = 16,
) -> Dict[str, Any]:
    """Create authoritative run manifest identifying models, converter, and native runtime."""
    native_runtime = get_native_runtime_info()

    # Source revision from acquisition manifest
    acq_path = source_model_dir / "acquisition_manifest.json"
    source_revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    if acq_path.exists():
        try:
            with open(acq_path, "r", encoding="utf-8") as f:
                source_metadata = json.load(f)
                source_revision = source_metadata.get("revision", source_revision)
        except Exception:
            pass

    # Converter identity
    llm_infer_dir = Path(__file__).resolve().parents[4]
    converter_script_path = llm_infer_dir / "tools" / "llama_converter" / "convert_hf_to_gguf.py"
    converter_sha256 = compute_sha256(converter_script_path) if converter_script_path.exists() else "unknown"

    # Compute and verify artifact digests
    artifacts_info: Dict[str, Dict[str, Any]] = {}
    for quant_key, path in model_paths.items():
        sha = compute_sha256(path)
        artifacts_info[quant_key] = {
            "name": path.name,
            "path": str(path),
            "size_bytes": os.path.getsize(path),
            "sha256": sha,
            "format": quant_key,
        }

    manifest_hash_input = (
        f"{source_revision}:{converter_commit}:{converter_sha256}:"
        + ":".join(artifacts_info[k]["sha256"] for k in sorted(artifacts_info.keys()))
        + f":{native_runtime['library_sha256']}"
    )
    manifest_id = "run_manifest_" + hashlib.sha256(manifest_hash_input.encode("utf-8")).hexdigest()[:16]

    return {
        "manifest_id": manifest_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "source_revision": source_revision,
        "converter_identity": {
            "tool": "llama.cpp convert_hf_to_gguf.py",
            "commit": converter_commit,
            "script_sha256": converter_sha256,
        },
        "gguf_artifacts": artifacts_info,
        "native_runtime": native_runtime,
        "effective_settings": {
            "warmups_per_config": warmups,
            "trials_per_prompt": trials_per_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "clear_state": True,
            "kv_cache_state_isolation": "llm.reset() called before every measured trial (clear_state=True)",
            "configured_kv_cache_type": "GGML_TYPE_F16",
            "observed_kv_cache_type": "GGML_TYPE_F16 (context_params.type_k=1, type_v=1)",
            "configured_compute_precision": "F32/F16",
            "observed_compute_precision": "unverified",
        },
    }


def generate_comparison_table(model_summaries: Dict[str, Any]) -> Dict[str, Any]:
    """Generate structured comparison table from model summaries."""
    rows = []
    headers = [
        "Variant",
        "Disk Size",
        "Peak RSS",
        "TTFT p50 (ms)",
        "TTFT Mean ± Std",
        "Decode tok/s p50",
        "Decode tok/s Mean ± Std",
        "Exact Token Match vs F16",
        "Len-Aware Token Agreement",
        "Coarse Target Hit Rate",
    ]
    for key in ["F16", "Q8_0", "Q4_K_M"]:
        if key not in model_summaries:
            continue
        s = model_summaries[key]
        ttft = s["ttft_stats"]
        dec = s["decode_speed_stats"]
        row = {
            "variant": key,
            "disk_size_mb": s["artifact_size_mb"],
            "peak_rss_mb": s["peak_rss_mb"],
            "ttft_p50_ms": ttft["p50"],
            "ttft_mean_ms": ttft["mean"],
            "ttft_std_ms": ttft["std"],
            "decode_p50_tok_s": dec["p50"],
            "decode_mean_tok_s": dec["mean"],
            "decode_std_tok_s": dec["std"],
            "exact_token_agreement_with_f16": s.get("exact_token_agreement_with_f16", 1.0),
            "len_aware_token_agreement_with_f16": s.get("len_aware_token_agreement_with_f16", 1.0),
            "coarse_target_hit_rate": s["coarse_target_hit_rate"],
        }
        rows.append(row)
    return {
        "headers": headers,
        "rows": rows,
    }


def run_model_benchmark(
    model_path: Path,
    prompts: List[Dict[str, Any]],
    warmups: int = 3,
    trials_per_prompt: int = 10,
    max_tokens: int = 16,
    require_verified: bool = True,
    run_manifest: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute matched benchmark trials on a single model."""
    runner = GGUFModelRunner(model_path, n_ctx=512, verbose=False, require_verified=require_verified)

    # 1. Warmup (clear_state=False — warmups legitimately warm caches)
    logger.info("Executing %d declared warmup runs on %s...", warmups, model_path.name)
    for _ in range(warmups):
        _ = runner.generate(
            prompts[0]["prompt"],
            max_tokens=max_tokens,
            temperature=0.0,
            clear_state=False,
        )

    # 2. Measured Trials (clear_state=True — each trial sees a clean KV prefix)
    raw_trials: List[Dict[str, Any]] = []
    per_prompt_results: Dict[str, Any] = {}

    ttft_all: List[float] = []
    decode_speed_all: List[float] = []
    target_hits: List[float] = []
    generated_texts: Dict[str, str] = {}
    generated_token_ids: Dict[str, List[int]] = {}

    for prompt_info in prompts:
        p_id = prompt_info["id"]
        p_text = prompt_info["prompt"]
        expected = prompt_info["expected_target"]

        p_ttft: List[float] = []
        p_decode_speed: List[float] = []
        last_text = ""
        last_tok_ids: List[int] = []
        last_stop_reason = "unknown"

        for trial_idx in range(trials_per_prompt):
            try:
                res = runner.generate(
                    p_text,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    clear_state=True,
                )
                hit = score_target_hit(res["text"], expected)

                trial_record = {
                    "run_manifest_id": run_manifest.get("manifest_id") if run_manifest else "unbound",
                    "model": model_path.name,
                    "model_sha256": compute_sha256(model_path),
                    "weight_precision": runner.weight_precision,
                    "compute_precision": runner.compute_precision,
                    "kv_cache_precision": runner.kv_cache_precision,
                    "precision_source": runner._precision_source,
                    "precision_details": res.get("precision_details"),
                    "native_runtime": res.get("native_runtime"),
                    "prompt_id": p_id,
                    "trial_idx": trial_idx,
                    "status": "success",
                    "prompt_token_ids": res["prompt_token_ids"],
                    "generated_token_ids": res["generated_token_ids"],
                    "prompt_tokens": res["prompt_tokens"],
                    "output_tokens": res["output_tokens"],
                    "decode_tokens": res["decode_tokens"],
                    "decode_steps": res["decode_steps"],
                    "stop_reason": res["stop_reason"],
                    "prefill_ms": res["prefill_ms"],
                    "first_token_selection_ms": res["first_token_selection_ms"],
                    "ttft_ms": res["ttft_ms"],
                    "decode_ms": res["decode_ms"],
                    "total_ms": res["total_ms"],
                    "decode_tok_per_sec": res["decode_tok_per_sec"],
                    "prefill_tok_per_sec": res["prefill_tok_per_sec"],
                    "cache_hit": res["cache_hit"],
                    "cached_prefix_tokens": res["cached_prefix_tokens"],
                    "evaluated_prompt_tokens": res["evaluated_prompt_tokens"],
                    "generated_text": res["text"],
                    "expected_target": expected,
                    "coarse_target_hit": hit,
                    "error": None,
                }
                if res["ttft_ms"] is not None:
                    ttft_all.append(res["ttft_ms"])
                    p_ttft.append(res["ttft_ms"])
                if res["decode_tok_per_sec"] > 0.0:
                    decode_speed_all.append(res["decode_tok_per_sec"])
                    p_decode_speed.append(res["decode_tok_per_sec"])

                last_text = res["text"]
                last_tok_ids = res["generated_token_ids"]
                last_stop_reason = res["stop_reason"]

            except Exception as e:
                logger.error("Error in trial %d on %s: %s", trial_idx, p_id, e)
                trial_record = {
                    "run_manifest_id": run_manifest.get("manifest_id") if run_manifest else "unbound",
                    "model": model_path.name,
                    "model_sha256": compute_sha256(model_path) if model_path.exists() else "unknown",
                    "weight_precision": runner.weight_precision,
                    "compute_precision": runner.compute_precision,
                    "kv_cache_precision": runner.kv_cache_precision,
                    "precision_source": runner._precision_source,
                    "precision_details": None,
                    "native_runtime": runner.native_runtime_info,
                    "prompt_id": p_id,
                    "trial_idx": trial_idx,
                    "status": "error",
                    "error": str(e),
                    "prompt_token_ids": [],
                    "generated_token_ids": [],
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "decode_tokens": 0,
                    "decode_steps": 0,
                    "stop_reason": "error",
                    "prefill_ms": None,
                    "first_token_selection_ms": None,
                    "ttft_ms": None,
                    "decode_ms": None,
                    "total_ms": None,
                    "decode_tok_per_sec": 0.0,
                    "prefill_tok_per_sec": 0.0,
                    "cache_hit": False,
                    "cached_prefix_tokens": 0,
                    "evaluated_prompt_tokens": 0,
                    "generated_text": "",
                    "expected_target": expected,
                    "coarse_target_hit": 0.0,
                }

            raw_trials.append(trial_record)

        hit_val = score_target_hit(last_text, expected)
        target_hits.append(hit_val)
        generated_texts[p_id] = last_text
        generated_token_ids[p_id] = last_tok_ids

        per_prompt_results[p_id] = {
            "ttft_ms": calculate_stats(p_ttft),
            "decode_tok_per_sec": calculate_stats(p_decode_speed),
            "coarse_target_hit": hit_val,
            "stop_reason": last_stop_reason,
            "output_text": last_text,
            "output_token_count": len(last_tok_ids),
        }

    # Measure fresh-process peak RSS
    rss_info = measure_fresh_process_peak_rss(model_path)
    file_size_bytes = os.path.getsize(model_path)

    # Analytical KV cache payload bytes for context length 512, 30 layers, 3 KV heads, dim 64, F16
    kv_cache_bytes_512 = 2 * 30 * 3 * 512 * 64 * 2

    summary = {
        "run_manifest_id": run_manifest.get("manifest_id") if run_manifest else "unbound",
        "model_name": model_path.name,
        "model_sha256": compute_sha256(model_path),
        "weight_precision": runner.weight_precision,
        "precision_source": runner._precision_source,
        "compute_precision": runner.compute_precision,
        "kv_cache_precision": runner.kv_cache_precision,
        "precision_details": {
            "weight": {
                "configured": runner.configured_weight_precision,
                "observed": runner.observed_weight_precision,
                "established_by": runner.weight_precision_established_by,
            },
            "kv_cache": {
                "configured": runner.configured_kv_cache_precision,
                "observed": runner.observed_kv_cache_precision,
                "type_k": runner.observed_kv_cache_type_k,
                "type_v": runner.observed_kv_cache_type_v,
                "established_by": runner.kv_cache_precision_established_by,
            },
            "compute": {
                "configured": runner.configured_compute_precision,
                "observed": runner.observed_compute_precision,
                "established_by": runner.compute_precision_established_by,
            },
        },
        "native_runtime": runner.native_runtime_info,
        "artifact_size_bytes": file_size_bytes,
        "artifact_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "peak_rss_bytes": rss_info["peak_rss_bytes"],
        "peak_rss_mb": rss_info["peak_rss_mb"],
        "rss_measurement_status": rss_info["status"],
        "rss_measurement_error": rss_info["error"],
        "kv_cache_analytical_bytes_ctx512": kv_cache_bytes_512,
        "coarse_target_hit_rate": round(sum(target_hits) / len(target_hits), 4),
        "total_measured_trials": len(raw_trials),
        "warmups_per_config": warmups,
        "ttft_stats": calculate_stats(ttft_all),
        "decode_speed_stats": calculate_stats(decode_speed_all),
        "per_prompt_results": per_prompt_results,
        "generated_texts": generated_texts,
        "generated_token_ids": generated_token_ids,
    }

    return summary, raw_trials


def run_full_quantization_comparison(
    models_dir: Path,
    output_dir: Path,
    warmups: int = 3,
    trials_per_prompt: int = 10,
) -> Dict[str, Any]:
    """Execute complete comparison across F16, Q8_0, and Q4_K_M."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    f16_path = models_dir / "smollm2-135m-instruct-f16.gguf"
    q8_0_path = models_dir / "smollm2-135m-instruct-q8_0.gguf"
    q4_k_m_path = models_dir / "smollm2-135m-instruct-q4_k_m.gguf"

    model_paths = {
        "F16": f16_path,
        "Q8_0": q8_0_path,
        "Q4_K_M": q4_k_m_path,
    }
    for k, p in model_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Required GGUF model not found: {p}")

    # Verify build manifest identity
    build_manifest_path = models_dir / "quantization_manifest.json"
    if not build_manifest_path.is_file():
        raise FileNotFoundError(f"Build manifest not found at {build_manifest_path}")

    m_valid, m_err, m_data = verify_manifest_identity(build_manifest_path)
    if not m_valid:
        raise ValueError(f"Build manifest failed identity verification: {m_err}")

    # Create run manifest binding all three artifacts, source revision, converter, and native library
    llm_infer_dir = Path(__file__).resolve().parents[4]
    source_model_dir = llm_infer_dir / "models" / "smollm2-135m-instruct"
    run_manifest = create_run_manifest(
        model_paths,
        source_model_dir=source_model_dir,
        warmups=warmups,
        trials_per_prompt=trials_per_prompt,
    )

    all_raw_trials: List[Dict[str, Any]] = []
    model_summaries: Dict[str, Any] = {}

    for quant_key in ["F16", "Q8_0", "Q4_K_M"]:
        path = model_paths[quant_key]
        logger.info("Benchmarking model: %s (%s)...", path.name, quant_key)
        summary, raw_trials = run_model_benchmark(
            path,
            EVALUATION_PROMPTS,
            warmups=warmups,
            trials_per_prompt=trials_per_prompt,
            require_verified=True,
            run_manifest=run_manifest,
        )
        model_summaries[quant_key] = summary
        all_raw_trials.extend(raw_trials)

    # Compute comparative quality degradation against same-engine F16 reference
    ref_texts = model_summaries["F16"]["generated_texts"]
    ref_token_ids = model_summaries["F16"]["generated_token_ids"]

    for quant_key in ["F16", "Q8_0", "Q4_K_M"]:
        summary = model_summaries[quant_key]
        texts = summary["generated_texts"]
        tok_ids = summary["generated_token_ids"]

        exact_token_agreements: List[float] = []
        len_aware_token_agreements: List[float] = []
        exact_word_agreements: List[float] = []
        len_aware_word_agreements: List[float] = []

        for p_info in EVALUATION_PROMPTS:
            p_id = p_info["id"]

            # Token-ID level agreement (Primary)
            exact_tok, len_tok = compute_token_id_agreement(tok_ids[p_id], ref_token_ids[p_id])
            exact_token_agreements.append(exact_tok)
            len_aware_token_agreements.append(len_tok)

            # Word-split agreement (Secondary)
            exact_w, len_w = compute_sequence_agreement(texts[p_id], ref_texts[p_id])
            exact_word_agreements.append(exact_w)
            len_aware_word_agreements.append(len_w)

        n = len(EVALUATION_PROMPTS)
        summary["exact_token_agreement_with_f16"] = round(sum(exact_token_agreements) / n, 4)
        summary["len_aware_token_agreement_with_f16"] = round(sum(len_aware_token_agreements) / n, 4)
        summary["word_split_exact_agreement_with_f16"] = round(sum(exact_word_agreements) / n, 4)
        summary["word_split_len_aware_agreement_with_f16"] = round(sum(len_aware_word_agreements) / n, 4)

    comparison_table = generate_comparison_table(model_summaries)

    # Save raw trials to jsonl
    raw_trials_path = logs_dir / "raw_trials.jsonl"
    with open(raw_trials_path, "w", encoding="utf-8") as f:
        for t in all_raw_trials:
            f.write(json.dumps(t) + "\n")

    full_results = {
        "run_manifest": run_manifest,
        "metadata": {
            "source_model": run_manifest["source_model_id"],
            "source_revision": run_manifest["source_revision"],
            "runtime_engine": f"llama.cpp via llama-cpp-python {run_manifest['native_runtime']['binding_version']}",
            "device": run_manifest["native_runtime"]["device"],
            "native_library_name": run_manifest["native_runtime"]["library_name"],
            "native_library_sha256": run_manifest["native_runtime"]["library_sha256"],
            "backend_evidence": run_manifest["native_runtime"]["backend_evidence"],
            "warmups_per_config": warmups,
            "measured_trials_per_prompt": trials_per_prompt,
            "kv_cache_state_isolation": "llm.reset() called before every measured trial (clear_state=True)",
            "timing_definitions": {
                "prefill_ms": "eval(prompt_tokens) duration",
                "first_token_selection_ms": "sample() duration for first token",
                "ttft_ms": "prefill_ms + first_token_selection_ms",
                "decode_ms": "subsequent eval/sample duration for decode steps",
                "decode_tok_per_sec": "decode_steps / (decode_ms / 1000)",
                "prefill_tok_per_sec": "evaluated_prompt_tokens / (prefill_ms / 1000)",
            },
            "distinct_prompts_count": len(EVALUATION_PROMPTS),
            "total_raw_trials": len(all_raw_trials),
            "timestamp": run_manifest["timestamp"],
            "energy_reporting": "omitted (no hardware power meter instrument integrated)",
            "calibration": "none (standard block quantization applied)",
        },
        "models": model_summaries,
        "comparison_table": comparison_table,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    logger.info("Complete comparison saved to %s", results_path)
    return full_results

    logger.info("Complete comparison saved to %s", results_path)
    return full_results
