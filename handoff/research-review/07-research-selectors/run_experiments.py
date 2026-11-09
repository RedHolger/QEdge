#!/usr/bin/env python3
"""Stage 07 Authoritative Benchmark Experiments Runner.

Evaluates matched, reproducible configurations holding model revision, precision,
prompts, seeds, and hardware constant:
1. Dynamic Provenance: reads model revision, device, and actual dtype directly from runtime.
2. Permanent Eviction Policies: Full vs FIFO vs KeyDiff vs CentroidNeighbor across fixed
   token capacities C in {Full, C=30, C=20, C=10} (distinct from attention ratios).
3. Temporary Attention Selection: Dense vs QUOKA vs AttentionValueDecay across attention
   ratios in {1.0 (100%), 0.75 (75%), 0.50 (50%), 0.25 (25%)}.
4. Correct Evaluation Metric Semantics:
   - exact_sequence_agreement: Strict binary indicator of identical full sequence generation.
   - length_aware_token_agreement: Token-level agreement normalized by max(len(gen), len(ref)),
     penalizing premature EOS and truncation.
   - sequence_agreement_diagnostic: Detailed diagnostics with prefix match and length delta.
   - coarse_authored_target_hit: Boundary-aware and negation-aware factual target matching.
   - teacher_forced_prefix_logit_mse: Pure numerical approximation error on identical conditioning prefixes.
   - trajectory_divergence_logit_mse: Step-by-step logit comparison across free-running diverging trajectories.
5. Statistical Rigor and Uncertainty Separation:
   - Timing estimand: 1 warmup run + 3 repeated timing trials per configuration (N_timing = 15
     across 5 prompts) reporting sample counts, means, standard deviations, and 95% confidence intervals.
   - Quality estimand: Macro-averaged across 5 distinct authored QA prompts (N_prompts = 5).
     Deterministic greedy decoding (temperature=0.0) yields identical tokens on repeated runs.
6. Conservative Selector Memory Accounting:
   - peak_selector_intermediates_bytes: Documented conservative accounting of coexisting selector buffers.
   - peak_gathered_active_bytes, peak_gqa_expanded_bytes, peak_attention_mask_bytes, and resident_payload_bytes
     are maintained as strictly distinct telemetry categories.
7. Chunked Prefill Workload: Exercises S_prompt = 125 with chunk size 64.

Outputs:
- handoff/research-review/07-research-selectors/results.json
- handoff/research-review/07-research-selectors/logs/raw_trials.jsonl
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

repo_root = Path(__file__).resolve().parents[3]
runtime_src = repo_root / "llm-inference" / "runtime" / "src"
if str(runtime_src) not in sys.path:
    sys.path.insert(0, str(runtime_src))

from qedge_runtime.runner import ModelRunner, GenerationResult
from qedge_runtime.config import get_default_model_dir
from qedge_runtime.selectors import SparseAttentionContext, calculate_selector_intermediates_bytes
from qedge_runtime.cache import BoundedKVCache, QEdgeTransformersCacheAdapter
from qedge_runtime.evaluation import (
    exact_sequence_agreement,
    length_aware_token_agreement,
    sequence_agreement_diagnostic,
    trajectory_divergence_logit_mse,
    coarse_authored_target_hit,
    compute_teacher_forced_step_logits,
    compute_teacher_forced_logit_mse,
)

QA_BENCHMARK_PROMPTS = [
    {
        "id": "qa_paris",
        "prompt": "What is the capital city of France? Answer with just the name of the city.",
        "target": "Paris",
    },
    {
        "id": "qa_water",
        "prompt": "What common chemical substance has the formula H2O? Answer with just the name.",
        "target": "Water",
    },
    {
        "id": "qa_light",
        "prompt": "In physics, what physical constant is denoted by the letter c? Answer with the exact speed in meters per second.",
        "target": "299,792,458",
    },
    {
        "id": "qa_binary",
        "prompt": "What is the binary representation of the decimal number 5? Answer with just the binary digits.",
        "target": "101",
    },
    {
        "id": "qa_color",
        "prompt": "In additive color synthesis (RGB), which color alongside green and blue forms the primary triad? Answer with just the color.",
        "target": "Red",
    },
]

OUTPUT_DIR = repo_root / "handoff" / "research-review" / "07-research-selectors"
LOGS_DIR = OUTPUT_DIR / "logs"


def compute_ci95(values: List[float]) -> Tuple[float, float, float, float]:
    """Compute sample count, mean, sample standard deviation, and 95% CI delta."""
    n = len(values)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return 1, m, 0.0, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * (std / math.sqrt(n))
    return n, m, std, ci95


def run_benchmark():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    raw_logs_path = LOGS_DIR / "raw_trials.jsonl"

    print("Initializing ModelRunner with pinned SmolLM2...")
    model_dir = get_default_model_dir()
    runner = ModelRunner(model_dir)

    print(f"Loaded model: {runner.model_name}")
    print(f"Dynamically derived revision: {runner.revision}")
    print(f"Dynamically derived device: {runner.device}")
    print(f"Dynamically derived actual dtype: {runner.actual_dtype}")

    trials_data: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # STEP 0: Compute Dense Unselected References for all Prompts
    # -------------------------------------------------------------------------
    print("\nComputing Dense References for Quality Baseline...")
    dense_references: Dict[str, Dict[str, Any]] = {}
    for p_info in QA_BENCHMARK_PROMPTS:
        p_id = p_info["id"]
        res = runner.generate(
            prompt=p_info["prompt"],
            max_new_tokens=16,
            use_chat_template=True,
            record_logits=True,
            cache_policy="full",
            seed=42,
        )
        hit = coarse_authored_target_hit(res.generated_text, p_info["target"])
        dense_references[p_id] = {
            "token_ids": res.generated_token_ids,
            "text": res.generated_text,
            "step_logits": res.step_logits,
            "prompt_tokens_count": len(res.prompt_token_ids),
            "generated_tokens_count": len(res.generated_token_ids),
            "target_hit": hit,
        }
        print(f"Dense baseline {p_id}: tokens={len(res.generated_token_ids)}, text={res.generated_text!r}, target_hit={hit}")

    # -------------------------------------------------------------------------
    # PHASE 1: PERMANENT EVICTION EXPERIMENTS (RESIDENT MEMORY BUDGETS)
    # Fixed capacities C in {Full, 30, 20, 10}
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1: PERMANENT EVICTION EXPERIMENTS (RESIDENT MEMORY BUDGETS)")
    print("=" * 80)

    eviction_configs = [
        ("Full", None, "full"),
        ("C=30", 30, "fifo"),
        ("C=30", 30, "keydiff"),
        ("C=30", 30, "centroid_neighbor"),
        ("C=20", 20, "fifo"),
        ("C=20", 20, "keydiff"),
        ("C=20", 20, "centroid_neighbor"),
        ("C=10", 10, "fifo"),
        ("C=10", 10, "keydiff"),
        ("C=10", 10, "centroid_neighbor"),
    ]

    for cap_label, cap_val, policy in eviction_configs:
        print(f"Eviction Config: {policy:<18} | Capacity: {cap_label}")
        for p_info in QA_BENCHMARK_PROMPTS:
            p_id = p_info["id"]
            p_text = p_info["prompt"]
            p_target = p_info["target"]
            ref = dense_references[p_id]

            # 1 Warmup run (unmeasured)
            runner.generate(
                prompt=p_text,
                max_new_tokens=16,
                use_chat_template=True,
                cache_policy=policy,
                cache_capacity=cap_val,
                seed=42,
            )

            # Compute Teacher-Forced Logit MSE (identical prefix)
            tf_logit_mse = compute_teacher_forced_logit_mse(
                runner=runner,
                prompt=p_text,
                ref_tokens=ref["token_ids"],
                cache_policy=policy,
                cache_capacity=cap_val,
                use_chat_template=True,
            )

            # 3 Measured Trials
            for trial_idx in range(3):
                res = runner.generate(
                    prompt=p_text,
                    max_new_tokens=16,
                    use_chat_template=True,
                    cache_policy=policy,
                    cache_capacity=cap_val,
                    record_logits=True,
                    seed=42,
                )

                hit = coarse_authored_target_hit(res.generated_text, p_target)
                exact_agree = exact_sequence_agreement(res.generated_token_ids, ref["token_ids"])
                len_agree = length_aware_token_agreement(res.generated_token_ids, ref["token_ids"])
                diag = sequence_agreement_diagnostic(
                    res.generated_token_ids,
                    ref["token_ids"],
                    eos_token_id=runner.tokenizer.eos_token_id,
                )
                traj_logit_mse = trajectory_divergence_logit_mse(res.step_logits, ref["step_logits"])

                rec = {
                    "experiment_family": "permanent_eviction",
                    "prompt_id": p_id,
                    "policy": policy,
                    "capacity_label": cap_label,
                    "capacity_tokens": cap_val,
                    "trial_idx": trial_idx,
                    "prompt_tokens_count": len(res.prompt_token_ids),
                    "generated_tokens_count": len(res.generated_token_ids),
                    "generated_text": res.generated_text,
                    "coarse_authored_target_hit": hit,
                    "exact_sequence_agreement": exact_agree,
                    "length_aware_token_agreement": round(len_agree, 4),
                    "sequence_agreement_diagnostic": diag,
                    "teacher_forced_prefix_logit_mse": round(tf_logit_mse, 6),
                    "trajectory_divergence_logit_mse": round(traj_logit_mse, 6) if traj_logit_mse is not None else 0.0,
                    "ttft_ms": res.ttft_ms,
                    "decode_ms": res.decode_ms,
                    "decode_tok_per_sec": res.decode_tok_per_sec,
                    "prefill_ms": res.prefill_ms,
                    "total_ms": res.total_ms,
                    "resident_payload_bytes": res.cache_stats.get("resident_payload_bytes", 0),
                    "peak_temporary_bytes": res.cache_stats.get("peak_temporary_bytes", 0),
                    "total_evictions": res.cache_stats.get("total_evictions", 0),
                    "reproducibility": res.reproducibility,
                }
                trials_data.append(rec)

    # -------------------------------------------------------------------------
    # PHASE 2: TEMPORARY ATTENTION SELECTION EXPERIMENTS (ATTENTION-PAIR BUDGETS)
    # Attention ratios in {1.0 (100%), 0.75 (75%), 0.50 (50%), 0.25 (25%)}
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: TEMPORARY ATTENTION SELECTION EXPERIMENTS (ATTENTION-PAIR BUDGETS)")
    print("=" * 80)

    selection_configs = [
        ("100%", 1.0, "dense"),
        ("100%", 1.0, "quoka"),
        ("75%", 0.75, "quoka"),
        ("50%", 0.50, "quoka"),
        ("25%", 0.25, "quoka"),
        ("100%", 1.0, "attention_value_decay"),
        ("75%", 0.75, "attention_value_decay"),
        ("50%", 0.50, "attention_value_decay"),
        ("25%", 0.25, "attention_value_decay"),
    ]

    for ratio_label, ratio_val, selector in selection_configs:
        print(f"Selection Config: {selector:<22} | Ratio: {ratio_label}")
        for p_info in QA_BENCHMARK_PROMPTS:
            p_id = p_info["id"]
            p_text = p_info["prompt"]
            p_target = p_info["target"]
            ref = dense_references[p_id]

            sel_arg = None if selector == "dense" else selector
            ratio_arg = None if selector == "dense" else ratio_val

            # 1 Warmup run (unmeasured)
            runner.generate(
                prompt=p_text,
                max_new_tokens=16,
                use_chat_template=True,
                cache_policy="full",
                attention_selector=sel_arg,
                attention_ratio=ratio_arg,
                seed=42,
            )

            # Compute Teacher-Forced Logit MSE (identical prefix)
            tf_logit_mse = compute_teacher_forced_logit_mse(
                runner=runner,
                prompt=p_text,
                ref_tokens=ref["token_ids"],
                cache_policy="full",
                attention_selector=sel_arg,
                attention_ratio=ratio_arg,
                use_chat_template=True,
            )

            # 3 Measured Trials
            for trial_idx in range(3):
                res = runner.generate(
                    prompt=p_text,
                    max_new_tokens=16,
                    use_chat_template=True,
                    cache_policy="full",
                    attention_selector=sel_arg,
                    attention_ratio=ratio_arg,
                    record_logits=True,
                    seed=42,
                )

                hit = coarse_authored_target_hit(res.generated_text, p_target)
                exact_agree = exact_sequence_agreement(res.generated_token_ids, ref["token_ids"])
                len_agree = length_aware_token_agreement(res.generated_token_ids, ref["token_ids"])
                diag = sequence_agreement_diagnostic(
                    res.generated_token_ids,
                    ref["token_ids"],
                    eos_token_id=runner.tokenizer.eos_token_id,
                )
                traj_logit_mse = trajectory_divergence_logit_mse(res.step_logits, ref["step_logits"])

                sparse_stats = res.cache_stats.get("sparse_attention", {})
                overhead = sparse_stats.get("total_selection_overhead_ms", 0.0)
                sparsity = sparse_stats.get("sparsity_ratio", 0.0)
                inter_peak = sparse_stats.get("peak_selector_intermediates_bytes", 0)
                gather_peak = sparse_stats.get("peak_gathered_active_bytes", 0)
                gqa_peak = sparse_stats.get("peak_gqa_expanded_bytes", 0)
                mask_peak = sparse_stats.get("peak_attention_mask_bytes", 0)

                rec = {
                    "experiment_family": "temporary_attention_selection",
                    "prompt_id": p_id,
                    "selector": selector,
                    "ratio_label": ratio_label,
                    "attention_ratio": ratio_val,
                    "trial_idx": trial_idx,
                    "prompt_tokens_count": len(res.prompt_token_ids),
                    "generated_tokens_count": len(res.generated_token_ids),
                    "generated_text": res.generated_text,
                    "coarse_authored_target_hit": hit,
                    "exact_sequence_agreement": exact_agree,
                    "length_aware_token_agreement": round(len_agree, 4),
                    "sequence_agreement_diagnostic": diag,
                    "teacher_forced_prefix_logit_mse": round(tf_logit_mse, 6),
                    "trajectory_divergence_logit_mse": round(traj_logit_mse, 6) if traj_logit_mse is not None else 0.0,
                    "ttft_ms": res.ttft_ms,
                    "decode_ms": res.decode_ms,
                    "decode_tok_per_sec": res.decode_tok_per_sec,
                    "prefill_ms": res.prefill_ms,
                    "total_ms": res.total_ms,
                    "selection_overhead_ms": overhead,
                    "sparsity_ratio": sparsity,
                    "peak_selector_intermediates_bytes": inter_peak,
                    "peak_gathered_active_bytes": gather_peak,
                    "peak_gqa_expanded_bytes": gqa_peak,
                    "peak_attention_mask_bytes": mask_peak,
                    "resident_payload_bytes": res.cache_stats.get("resident_payload_bytes", 0),
                    "peak_temporary_bytes": res.cache_stats.get("peak_temporary_bytes", 0),
                    "sparse_attention_stats": sparse_stats,
                    "reproducibility": res.reproducibility,
                }
                trials_data.append(rec)

    # -------------------------------------------------------------------------
    # PHASE 3: CHUNKED PREFILL WORKLOAD EXPERIMENT (S_prompt = 125, chunk = 64)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3: CHUNKED PREFILL WORKLOAD EXPERIMENT (S_prompt = 125, chunk = 64)")
    print("=" * 80)

    base_text = (
        "Qualcomm Snapdragon mobile platforms combine high-performance CPU cores, "
        "Adreno graphics processing, and dedicated Hexagon neural processing units "
        "for efficient on-device artificial intelligence inference. "
    )
    tokens_base = runner.tokenizer.encode(base_text)
    repeated_tokens = (tokens_base * 5)[:125]
    chunked_prompt_125 = runner.tokenizer.decode(repeated_tokens)

    # Reference full prefill dense baseline
    res_dense_125 = runner.generate(
        prompt=chunked_prompt_125,
        max_new_tokens=16,
        use_chat_template=False,
        cache_policy="full",
        record_logits=True,
        seed=42,
    )
    ref_125_tokens = res_dense_125.generated_token_ids
    ref_125_logits = res_dense_125.step_logits

    chunk_configs = [
        ("full_prefill_dense", None, None, None),
        ("chunked_dense_c64", 64, None, None),
        ("chunked_quoka_75pct_c64", 64, "quoka", 0.75),
        ("chunked_quoka_50pct_c64", 64, "quoka", 0.50),
    ]

    for name, chunk_sz, selector, ratio in chunk_configs:
        print(f"Chunk Workload: {name:<26} | ChunkSize: {chunk_sz} | Selector: {selector} | Ratio: {ratio}")
        # Warmup
        runner.generate(
            prompt=chunked_prompt_125,
            max_new_tokens=16,
            use_chat_template=False,
            chunk_size=chunk_sz,
            attention_selector=selector,
            attention_ratio=ratio,
            seed=42,
        )

        for trial_idx in range(3):
            res = runner.generate(
                prompt=chunked_prompt_125,
                max_new_tokens=16,
                use_chat_template=False,
                chunk_size=chunk_sz,
                attention_selector=selector,
                attention_ratio=ratio,
                record_logits=True,
                seed=42,
            )

            exact_agree = exact_sequence_agreement(res.generated_token_ids, ref_125_tokens)
            len_agree = length_aware_token_agreement(res.generated_token_ids, ref_125_tokens)
            traj_logit_mse = trajectory_divergence_logit_mse(res.step_logits, ref_125_logits)
            sparse_stats = res.cache_stats.get("sparse_attention", {})
            overhead = sparse_stats.get("total_selection_overhead_ms", 0.0)

            rec = {
                "experiment_family": "chunked_prefill_workload",
                "workload_name": name,
                "chunk_size": chunk_sz,
                "selector": selector or "dense",
                "attention_ratio": ratio or 1.0,
                "trial_idx": trial_idx,
                "prompt_tokens_count": len(res.prompt_token_ids),
                "generated_tokens_count": len(res.generated_token_ids),
                "exact_sequence_agreement": exact_agree,
                "length_aware_token_agreement": round(len_agree, 4),
                "trajectory_divergence_logit_mse": round(traj_logit_mse, 6) if traj_logit_mse is not None else 0.0,
                "ttft_ms": res.ttft_ms,
                "decode_ms": res.decode_ms,
                "decode_tok_per_sec": res.decode_tok_per_sec,
                "prefill_ms": res.prefill_ms,
                "total_ms": res.total_ms,
                "selection_overhead_ms": overhead,
                "reproducibility": res.reproducibility,
            }
            trials_data.append(rec)

    # -------------------------------------------------------------------------
    # STEP 4: Write Raw JSONL Trials
    # -------------------------------------------------------------------------
    print(f"\nWriting {len(trials_data)} raw trials to {raw_logs_path}...")
    with open(raw_logs_path, "w", encoding="utf-8") as f:
        for t in trials_data:
            f.write(json.dumps(t) + "\n")

    # -------------------------------------------------------------------------
    # STEP 5: Statistical Aggregation with 95% Confidence Intervals
    # -------------------------------------------------------------------------
    print("Aggregating structured summary with 95% confidence intervals...")

    # Permanent Eviction Summary
    eviction_summary: Dict[str, Dict[str, Any]] = {}
    for policy in ["full", "fifo", "keydiff", "centroid_neighbor"]:
        eviction_summary[policy] = {}
        for cap_label in ["Full", "C=30", "C=20", "C=10"]:
            matches = [
                t for t in trials_data
                if t["experiment_family"] == "permanent_eviction"
                and t["policy"] == policy
                and t["capacity_label"] == cap_label
            ]
            if not matches:
                continue

            # Timing statistics over N_timing = 15 trials
            n_timing, ttft_m, ttft_s, ttft_ci = compute_ci95([m["ttft_ms"] for m in matches])
            _, dec_m, dec_s, dec_ci = compute_ci95([m["decode_ms"] for m in matches])
            _, tps_m, tps_s, tps_ci = compute_ci95([m["decode_tok_per_sec"] for m in matches])

            # Quality metrics macro-averaged across N_prompts = 5 distinct prompts (trial_idx == 0)
            prompt_matches = [m for m in matches if m["trial_idx"] == 0]
            n_prompts = len(prompt_matches)
            hit_rate = sum(1.0 if m["coarse_authored_target_hit"] else 0.0 for m in prompt_matches) / n_prompts
            exact_seq_agree = sum(m["exact_sequence_agreement"] for m in prompt_matches) / n_prompts
            len_tok_agree = sum(m["length_aware_token_agreement"] for m in prompt_matches) / n_prompts
            tf_mse = sum(m["teacher_forced_prefix_logit_mse"] for m in prompt_matches) / n_prompts
            traj_mse = sum(m["trajectory_divergence_logit_mse"] for m in prompt_matches) / n_prompts

            mem_bytes = matches[0]["resident_payload_bytes"]

            eviction_summary[policy][cap_label] = {
                "timing_sample_count": n_timing,
                "distinct_prompts_count": n_prompts,
                "prompt_tokens_actual": matches[0]["prompt_tokens_count"],
                "generated_tokens_actual": matches[0]["generated_tokens_count"],
                "coarse_authored_target_hit_rate": round(hit_rate, 4),
                "exact_sequence_agreement": round(exact_seq_agree, 4),
                "length_aware_token_agreement": round(len_tok_agree, 4),
                "teacher_forced_prefix_logit_mse": round(tf_mse, 6),
                "trajectory_divergence_logit_mse_mean": round(traj_mse, 6),
                "ttft_ms_mean": round(ttft_m, 2),
                "ttft_ms_std": round(ttft_s, 2),
                "ttft_ms_ci95": round(ttft_ci, 2),
                "decode_ms_mean": round(dec_m, 2),
                "decode_ms_std": round(dec_s, 2),
                "decode_ms_ci95": round(dec_ci, 2),
                "decode_tok_per_sec_mean": round(tps_m, 2),
                "decode_tok_per_sec_std": round(tps_s, 2),
                "decode_tok_per_sec_ci95": round(tps_ci, 2),
                "resident_payload_bytes": mem_bytes,
            }

    # Temporary Attention Selection Summary
    selection_summary: Dict[str, Dict[str, Any]] = {}
    for sel in ["dense", "quoka", "attention_value_decay"]:
        selection_summary[sel] = {}
        for r_label in ["100%", "75%", "50%", "25%"]:
            matches = [
                t for t in trials_data
                if t["experiment_family"] == "temporary_attention_selection"
                and t["selector"] == sel
                and t["ratio_label"] == r_label
            ]
            if not matches:
                continue

            # Timing statistics over N_timing = 15 trials
            n_timing, ttft_m, ttft_s, ttft_ci = compute_ci95([m["ttft_ms"] for m in matches])
            _, dec_m, dec_s, dec_ci = compute_ci95([m["decode_ms"] for m in matches])
            _, tps_m, tps_s, tps_ci = compute_ci95([m["decode_tok_per_sec"] for m in matches])
            _, ovh_m, ovh_s, ovh_ci = compute_ci95([m["selection_overhead_ms"] for m in matches])
            _, spar_m, _, _ = compute_ci95([m["sparsity_ratio"] for m in matches])

            # Quality metrics macro-averaged across N_prompts = 5 distinct prompts (trial_idx == 0)
            prompt_matches = [m for m in matches if m["trial_idx"] == 0]
            n_prompts = len(prompt_matches)
            hit_rate = sum(1.0 if m["coarse_authored_target_hit"] else 0.0 for m in prompt_matches) / n_prompts
            exact_seq_agree = sum(m["exact_sequence_agreement"] for m in prompt_matches) / n_prompts
            len_tok_agree = sum(m["length_aware_token_agreement"] for m in prompt_matches) / n_prompts
            tf_mse = sum(m["teacher_forced_prefix_logit_mse"] for m in prompt_matches) / n_prompts
            traj_mse = sum(m["trajectory_divergence_logit_mse"] for m in prompt_matches) / n_prompts

            mem_bytes = matches[0]["resident_payload_bytes"]
            inter_bytes = matches[0]["peak_selector_intermediates_bytes"]
            gather_bytes = matches[0]["peak_gathered_active_bytes"]
            gqa_bytes = matches[0]["peak_gqa_expanded_bytes"]
            mask_bytes = matches[0]["peak_attention_mask_bytes"]

            selection_summary[sel][r_label] = {
                "timing_sample_count": n_timing,
                "distinct_prompts_count": n_prompts,
                "prompt_tokens_actual": matches[0]["prompt_tokens_count"],
                "generated_tokens_actual": matches[0]["generated_tokens_count"],
                "coarse_authored_target_hit_rate": round(hit_rate, 4),
                "exact_sequence_agreement": round(exact_seq_agree, 4),
                "length_aware_token_agreement": round(len_tok_agree, 4),
                "teacher_forced_prefix_logit_mse": round(tf_mse, 6),
                "trajectory_divergence_logit_mse_mean": round(traj_mse, 6),
                "ttft_ms_mean": round(ttft_m, 2),
                "ttft_ms_std": round(ttft_s, 2),
                "ttft_ms_ci95": round(ttft_ci, 2),
                "decode_ms_mean": round(dec_m, 2),
                "decode_ms_std": round(dec_s, 2),
                "decode_ms_ci95": round(dec_ci, 2),
                "decode_tok_per_sec_mean": round(tps_m, 2),
                "decode_tok_per_sec_std": round(tps_s, 2),
                "decode_tok_per_sec_ci95": round(tps_ci, 2),
                "selection_overhead_ms_mean": round(ovh_m, 2),
                "selection_overhead_ms_std": round(ovh_s, 2),
                "selection_overhead_ms_ci95": round(ovh_ci, 2),
                "sparsity_ratio_mean": round(spar_m, 4),
                "resident_payload_bytes": mem_bytes,
                "peak_selector_intermediates_bytes": inter_bytes,
                "peak_gathered_active_bytes": gather_bytes,
                "peak_gqa_expanded_bytes": gqa_bytes,
                "peak_attention_mask_bytes": mask_bytes,
            }

    # Chunked Prefill Workload Summary
    chunk_summary: Dict[str, Any] = {}
    for name, _, _, _ in chunk_configs:
        matches = [
            t for t in trials_data
            if t["experiment_family"] == "chunked_prefill_workload"
            and t["workload_name"] == name
        ]
        if not matches:
            continue
        n, ttft_m, ttft_s, ttft_ci = compute_ci95([m["ttft_ms"] for m in matches])
        _, dec_m, dec_s, dec_ci = compute_ci95([m["decode_ms"] for m in matches])
        _, tps_m, tps_s, tps_ci = compute_ci95([m["decode_tok_per_sec"] for m in matches])
        exact_agree = matches[0]["exact_sequence_agreement"]
        len_agree = matches[0]["length_aware_token_agreement"]
        traj_mse = matches[0]["trajectory_divergence_logit_mse"]
        _, ovh_m, _, _ = compute_ci95([m["selection_overhead_ms"] for m in matches])

        chunk_summary[name] = {
            "timing_sample_count": n,
            "prompt_tokens_actual": matches[0]["prompt_tokens_count"],
            "generated_tokens_actual": matches[0]["generated_tokens_count"],
            "exact_sequence_agreement": exact_agree,
            "length_aware_token_agreement": len_agree,
            "trajectory_divergence_logit_mse": round(traj_mse, 6),
            "ttft_ms_mean": round(ttft_m, 2),
            "ttft_ms_std": round(ttft_s, 2),
            "ttft_ms_ci95": round(ttft_ci, 2),
            "decode_ms_mean": round(dec_m, 2),
            "decode_ms_std": round(dec_s, 2),
            "decode_ms_ci95": round(dec_ci, 2),
            "decode_tok_per_sec_mean": round(tps_m, 2),
            "decode_tok_per_sec_std": round(tps_s, 2),
            "decode_tok_per_sec_ci95": round(tps_ci, 2),
            "selection_overhead_ms_mean": round(ovh_m, 2),
        }

    final_results = {
        "benchmark_metadata": {
            "model": runner.model_name,
            "model_revision": runner.revision,
            "device": runner.device,
            "actual_dtype": runner.actual_dtype,
            "requested_dtype": runner.dtype_str,
            "max_new_tokens": 16,
            "warmup_runs_per_config": 1,
            "measured_trials_per_config": 3,
            "timing_sample_count_per_config": 15,
            "distinct_quality_prompts_count": len(QA_BENCHMARK_PROMPTS),
            "total_raw_trials": len(trials_data),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "superseded_evidence_archive": "handoff/research-review/07-research-selectors/evidence/superseded/",
            "estimand_documentation": {
                "timing_metrics": "Mean, std, and 95% CI computed over 15 repeated hardware executions (5 prompts x 3 runs) to quantify silicon/system execution variance.",
                "quality_metrics": "Macro-averaged across 5 distinct authored QA prompts. Repeated runs with greedy decoding (temperature=0.0) produce identical tokens and do not constitute independent semantic samples."
            },
        },
        "permanent_eviction_summary": eviction_summary,
        "temporary_attention_selection_summary": selection_summary,
        "chunked_prefill_workload_summary": chunk_summary,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nFinal structured results written to {results_path} successfully!")
    print(f"Total benchmark trials executed: {len(trials_data)}")


if __name__ == "__main__":
    run_benchmark()
