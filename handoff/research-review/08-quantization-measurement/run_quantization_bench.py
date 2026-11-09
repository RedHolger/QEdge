#!/usr/bin/env python3
"""Run Stage 08 Quantization Benchmark and Regenerate Artifacts.

Executes matched measurement trials across F16, Q8_0, and Q4_K_M GGUF models:
1. Validates hardware status against schema v2, marking Qualcomm NPU as BLOCKED.
2. Runs 3 declared warmups + 10 measured trials per prompt across 8 evaluation prompts.
3. Records TTFT (p50, p95, mean, std, 95% CI) and decode throughput (tok/s).
4. Measures fresh-process peak RSS via isolated subprocess.
5. Scores task quality (factual/reasoning target hits) and sequence agreement against F16.
6. Emits results.json, logs/raw_trials.jsonl, and formatted markdown comparison tables.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root and runtime source to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RUNTIME_SRC = REPO_ROOT / "llm-inference" / "runtime" / "src"
COMMON_SRC = REPO_ROOT / "qedge-common" / "src"

sys.path.insert(0, str(RUNTIME_SRC))
sys.path.insert(0, str(COMMON_SRC))

from qedge_common.schema import validate as schema_validate
from qedge_runtime.quantization.benchmark import run_full_quantization_comparison
from qedge_runtime.quantization.qnn_status import generate_hardware_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("quantization_bench")


def main() -> int:
    models_dir = REPO_ROOT / "llm-inference" / "models" / "smollm2-135m-instruct-gguf"
    output_dir = SCRIPT_DIR

    logger.info("======================================================================")
    logger.info("STAGE 08: QUANTIZATION MEASUREMENT BENCHMARK")
    logger.info("======================================================================")

    # 1. Generate & Validate Hardware Status
    hw_status_path = output_dir / "hardware_status.json"
    hw_status = generate_hardware_status(hw_status_path)
    schema_validate("hardware_status", hw_status)
    logger.info("Hardware status validated (schema v2). Qualcomm NPU status: %s", hw_status["npu_status"])

    # 2. Verify / Build Quantized Artifacts with Full Provenance
    from qedge_runtime.quantization.converter import build_quantized_artifacts
    source_model_dir = REPO_ROOT / "llm-inference" / "models" / "smollm2-135m-instruct"
    logger.info("Verifying source checkpoint and GGUF artifacts with full provenance...")
    build_quantized_artifacts(source_model_dir, models_dir, force=False)

    # 3. Run Full Quantization Comparison
    logger.info("Starting matched benchmark across F16, Q8_0, Q4_K_M...")
    results = run_full_quantization_comparison(
        models_dir=models_dir,
        output_dir=output_dir,
        warmups=3,
        trials_per_prompt=10,
    )

    models_data = results["models"]

    # 4. Print Markdown Comparison Table
    print("\n" + "=" * 80)
    print("STAGE 08 QUANTIZATION MEASUREMENT RESULTS")
    print("=" * 80 + "\n")

    run_manifest = results.get("run_manifest", {})
    print(f"**Run Manifest ID:** `{run_manifest.get('manifest_id', 'unknown')}`")
    native_rt = run_manifest.get("native_runtime", {})
    print(f"**Native Library:** `{native_rt.get('library_name', 'unknown')}` ({native_rt.get('library_path', 'unknown')})")
    print(f"**Native Library SHA256:** `{native_rt.get('library_sha256', 'unknown')}`")
    print(f"**Binding Version:** `{native_rt.get('binding_version', 'unknown')}`")
    print(f"**Execution Device:** `{native_rt.get('device', 'unknown')}`")
    print(f"**Backend Evidence:** `{native_rt.get('backend_evidence', 'unknown')}`\n")

    print("### Precision Dimension & Artifact Size Summary\n")
    print("| Quantization Variant | Weight Precision | Compute Precision | KV Cache Precision | Artifact Size (MB) | Artifact SHA256 | Peak RSS (MB) | RSS Status |")
    print("|---|---|---|---|---|---|---|---|")

    manifest_path = models_dir / "quantization_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    for k in ["F16", "Q8_0", "Q4_K_M"]:
        m = models_data[k]
        sha = manifest.get("artifacts", {}).get(m["model_name"], {}).get("sha256", "N/A")
        sha_short = sha[:12] + "..." if len(sha) > 12 else sha
        rss_val = f"{m['peak_rss_mb']} MB" if m["peak_rss_mb"] is not None else "Failed"
        print(f"| **{k}** | {m['weight_precision']} | {m['compute_precision']} | {m['kv_cache_precision']} | {m['artifact_size_mb']} MB | `{sha_short}` | {rss_val} | {m['rss_measurement_status']} |")

    print("\n### Latency & Throughput Benchmark (3 Warmups, 10 Trials/Prompt x 8 Prompts = 80 Trials/Model)\n")
    print("| Variant | TTFT p50 (ms) | TTFT p95 (ms) | TTFT Mean ± Std (ms) | TTFT 95% CI | Decode tok/s p50 | Decode tok/s p95 | Decode tok/s Mean ± Std |")
    print("|---|---|---|---|---|---|---|---|")

    for k in ["F16", "Q8_0", "Q4_K_M"]:
        m = models_data[k]
        t = m["ttft_stats"]
        d = m["decode_speed_stats"]
        print(
            f"| **{k}** | {t['p50']:.1f} | {t['p95']:.1f} | {t['mean']:.1f} ± {t['std']:.1f} | ±{t['ci95']:.2f} | "
            f"{d['p50']:.1f} | {d['p95']:.1f} | {d['mean']:.1f} ± {d['std']:.1f} |"
        )

    print("\n### Task Quality & Reference Degradation (Same-Engine F16 Baseline)\n")
    print("| Variant | Target Hit Rate (8 QA items) | Exact Token Match vs F16 | Len-Aware Token Agreement | Word-Split Exact vs F16 | Word-Split Len-Aware |")
    print("|---|---|---|---|---|---|")

    for k in ["F16", "Q8_0", "Q4_K_M"]:
        m = models_data[k]
        print(
            f"| **{k}** | {m['coarse_target_hit_rate'] * 100:.1f}% | "
            f"{m['exact_token_agreement_with_f16'] * 100:.1f}% | {m['len_aware_token_agreement_with_f16'] * 100:.1f}% | "
            f"{m['word_split_exact_agreement_with_f16'] * 100:.1f}% | {m['word_split_len_aware_agreement_with_f16'] * 100:.1f}% |"
        )

    print("\n### Generated Completions per Prompt\n")
    prompts_list = [
        ("qa_01_paris", "Capital of France"),
        ("qa_02_water", "Chemical formula for water"),
        ("qa_03_speed_of_light", "Physics constant c"),
        ("qa_04_binary_five", "Binary of 5"),
        ("qa_05_primary_colors", "RGB 3rd primary color"),
        ("inst_06_arithmetic", "15 * 4"),
        ("inst_07_logic", "Roses are plants?"),
        ("inst_08_extraction", "Extract fruit"),
    ]

    print("| Prompt ID | Description | F16 Output | Q8_0 Output | Q4_K_M Output | Target Match? |")
    print("|---|---|---|---|---|---|")

    f16_texts = models_data["F16"]["generated_texts"]
    q8_texts = models_data["Q8_0"]["generated_texts"]
    q4_texts = models_data["Q4_K_M"]["generated_texts"]

    for pid, desc in prompts_list:
        t_f16 = f16_texts.get(pid, "").replace("\n", " ")[:35]
        t_q8 = q8_texts.get(pid, "").replace("\n", " ")[:35]
        t_q4 = q4_texts.get(pid, "").replace("\n", " ")[:35]
        h_f16 = models_data["F16"]["per_prompt_results"][pid]["coarse_target_hit"] == 1.0
        h_q8 = models_data["Q8_0"]["per_prompt_results"][pid]["coarse_target_hit"] == 1.0
        h_q4 = models_data["Q4_K_M"]["per_prompt_results"][pid]["coarse_target_hit"] == 1.0
        match_str = f"F16:{'Y' if h_f16 else 'N'} Q8:{'Y' if h_q8 else 'N'} Q4:{'Y' if h_q4 else 'N'}"
        print(f"| `{pid}` | {desc} | `{t_f16}` | `{t_q8}` | `{t_q4}` | {match_str} |")

    print("\nExecution completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
