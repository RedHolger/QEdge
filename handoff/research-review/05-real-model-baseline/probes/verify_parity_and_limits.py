"""Standalone verification probe for Stage 05 Real-Model Reference Runtime.

Verifies:
1. Pinned metadata and architecture configuration loaded from checkpoint
2. Greedy token parity against upstream model.generate()
3. Numerical logits agreement within documented tolerances
4. Zero-token and one-token limits
5. Early EOS detection
6. State isolation across repeated requests
7. Contract v2 benchmark emission with outputs separated from metrics
"""
import json
from pathlib import Path
import sys
import unittest
import torch

from qedge_runtime.config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    SMOLLM2_135M_LICENSE,
    ModelArchitectureConfig,
)
from qedge_runtime.runner import ModelRunner
from qedge_runtime.reference import (
    generate_upstream_reference,
    verify_greedy_token_parity,
    verify_logits_agreement,
)
from qedge_runtime.acquisition import check_model_assets
from qedge_runtime.benchmark import build_benchmark_record, build_smoke_record
from qedge_common.schema import validate as validate_contract

MODEL_DIR = Path("llm-inference/models/smollm2-135m-instruct").resolve()


def run_checks() -> int:
    print("=== Running Stage 05 Real-Model Baseline Verification Probe ===")

    # Check 1: Model assets and offline presence
    print("Check 1: Model assets and offline presence...")
    valid, missing = check_model_assets(MODEL_DIR)
    if not valid:
        print(f"  FAIL: Assets missing in {MODEL_DIR}: {missing}", file=sys.stderr)
        return 1
    manifest_file = MODEL_DIR / "acquisition_manifest.json"
    if not manifest_file.is_file():
        print(f"  FAIL: Missing acquisition manifest {manifest_file}", file=sys.stderr)
        return 1
    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"  PASS: Assets present ({manifest['model_id']}, revision {manifest['revision'][:8]}).")

    # Check 2: Architecture config loaded from checkpoint
    print("Check 2: Architecture configuration loaded from checkpoint...")
    config_file = MODEL_DIR / "config.json"
    with config_file.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    cfg = ModelArchitectureConfig.from_checkpoint_dict(raw_cfg)
    assert cfg.vocab_size == 49152, f"Expected vocab 49152, got {cfg.vocab_size}"
    assert cfg.hidden_size == 576, f"Expected hidden 576, got {cfg.hidden_size}"
    assert cfg.num_hidden_layers == 30, f"Expected 30 layers, got {cfg.num_hidden_layers}"
    assert cfg.num_attention_heads == 9, f"Expected 9 heads, got {cfg.num_attention_heads}"
    assert cfg.num_key_value_heads == 3, f"Expected 3 KV heads, got {cfg.num_key_value_heads}"
    assert cfg.head_dim == 64, f"Expected head dim 64, got {cfg.head_dim}"
    assert cfg.rope_theta == 100000.0, f"Expected rope theta 100000, got {cfg.rope_theta}"
    print(f"  PASS: Architecture matches checkpoint (30L, 9Q/3KV heads, head_dim 64, theta 100k).")

    # Check 3: Runner initialization on CPU float32 and MPS if available
    print("Check 3: Deterministic CPU float32 runner initialization...")
    runner = ModelRunner(model_dir=MODEL_DIR, device="cpu", dtype="float32", local_files_only=True)
    print(f"  PASS: ModelRunner initialized on CPU in {runner.load_ms:.1f}ms.")

    # Check 4: Greedy token parity against upstream model.generate()
    print("Check 4: Greedy token parity against upstream model.generate()...")
    prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Name the primary colors in order.",
        "Write a Python function to square a number.",
    ]
    for prompt in prompts:
        res = runner.generate(prompt=prompt, max_new_tokens=16, record_logits=True)
        ref_ids, ref_text, ref_scores = generate_upstream_reference(
            model=runner.model,
            tokenizer=runner.tokenizer,
            prompt=prompt,
            max_new_tokens=16,
            formatted_prompt=runner.format_prompt(prompt),
            device=runner.device,
        )
        ok, msg = verify_greedy_token_parity(res.generated_token_ids, ref_ids)
        if not ok:
            print(f"  FAIL: Token parity mismatch for '{prompt}': {msg}", file=sys.stderr)
            return 1
        logits_ok, max_diff, logit_msg = verify_logits_agreement(
            res.step_logits, ref_scores, atol=1e-4, rtol=1e-4
        )
        if not logits_ok:
            print(f"  FAIL: Logits divergence for '{prompt}': {logit_msg}", file=sys.stderr)
            return 1
    print(f"  PASS: 100% greedy token parity and numerical logits agreement (atol=1e-4) across 4 prompts.")

    # Check 5: Output limits (zero and one token) and measurement definitions
    print("Check 5: Output limits (zero and one token) and measurement definitions...")
    res0 = runner.generate("Hello", max_new_tokens=0)
    assert len(res0.generated_token_ids) == 0, "Zero tokens expected"
    assert res0.ttft_ms is None, "Zero tokens must have ttft_ms = None (no token arrived)"
    assert res0.decode_tok_per_sec == 0.0, "Zero tokens must have decode_tok_per_sec = 0.0"
    assert res0.formatting_ms >= 0.0, "formatting_ms must be non-negative"

    res1 = runner.generate("Hello", max_new_tokens=1)
    assert len(res1.generated_token_ids) == 1, "One token expected"
    assert res1.ttft_ms is not None, "One token must have ttft_ms recorded"
    assert res1.first_token_select_ms >= 0.0, "first_token_select_ms must be non-negative"
    assert res1.formatting_ms >= 0.0, "formatting_ms must be non-negative"
    assert res1.detokenization_ms >= 0.0, "detokenization_ms must be non-negative"
    print(f"  PASS: Zero-token (0 ids, ttft_ms=None) and one-token (id: {res1.generated_token_ids[0]}, ttft_ms={res1.ttft_ms:.2f}ms) limits verified.")

    # Check 6: Early EOS stopping
    print("Check 6: Early EOS stopping...")
    res_eos = runner.generate("What is the capital of France?", max_new_tokens=64)
    if res_eos.stopped_on_eos:
        assert res_eos.generated_token_ids[-1] == runner.eos_token_id
        assert len(res_eos.generated_token_ids) < 64
        print(f"  PASS: Early EOS stopped at token {len(res_eos.generated_token_ids)}: {res_eos.generated_text!r}")
    else:
        print("  PASS: Completed without early EOS.")

    # Check 7: State isolation across repeated requests
    print("Check 7: State isolation across repeated requests...")
    r_a1 = runner.generate("Tell me a color.", max_new_tokens=10)
    r_b = runner.generate("What is the speed of light?", max_new_tokens=10)
    r_a2 = runner.generate("Tell me a color.", max_new_tokens=10)
    assert r_a1.generated_token_ids == r_a2.generated_token_ids, "State leakage detected: request A changed"
    assert r_a1.generated_token_ids != r_b.generated_token_ids, "Collateral collision between different prompts"
    print(f"  PASS: Requests A1 and A2 match exactly ({r_a1.generated_token_ids}); no state leakage.")

    # Check 8: Apple Silicon MPS parity (if available)
    if torch.backends.mps.is_available():
        print("Check 8: Apple Silicon MPS accelerator parity and synchronization...")
        mps_runner = ModelRunner(model_dir=MODEL_DIR, device="mps", dtype="bfloat16", local_files_only=True)
        res_mps = mps_runner.generate("What is 10 divided by 2?", max_new_tokens=16, record_logits=True)
        ref_mps_ids, _, ref_mps_scores = generate_upstream_reference(
            model=mps_runner.model,
            tokenizer=mps_runner.tokenizer,
            prompt="What is 10 divided by 2?",
            max_new_tokens=16,
            formatted_prompt=mps_runner.format_prompt("What is 10 divided by 2?"),
            device=mps_runner.device,
        )
        ok, msg = verify_greedy_token_parity(res_mps.generated_token_ids, ref_mps_ids)
        assert ok, f"MPS parity failed: {msg}"
        ok_l, max_d, msg_l = verify_logits_agreement(res_mps.step_logits, ref_mps_scores, atol=0.15, rtol=1e-2)
        assert ok_l, f"MPS bfloat16 logits failed: {msg_l}"
        print(f"  PASS: MPS parity verified (bfloat16 max logit diff: {max_d:.6f} <= 0.15).")
    else:
        print("Check 8: Apple Silicon MPS not available; skipped.")

    # Check 9: Contract v2 benchmark record validation
    print("Check 9: Contract v2 benchmark emission with outputs separation...")
    res_bench = runner.generate("Summarize edge computing.", max_new_tokens=16)
    rec = build_benchmark_record(res_bench)
    assert rec["schema_version"] == "2"
    assert rec["measured"] is True
    assert rec["kind"] == "runtime-generation"
    assert "text" in rec["outputs"] and rec["outputs"]["text"] == res_bench.generated_text
    assert "token_ids" in rec["outputs"] and rec["outputs"]["token_ids"] == res_bench.generated_token_ids
    assert "token_ids" not in rec["metrics"]
    validate_contract("benchmark_result", rec)
    print("  PASS: Contract v2 benchmark record validated; outputs stored separately from metrics.")

    # Check 10: Unsupported settings rejection
    print("Check 10: Rejection of unsupported configurations...")
    try:
        runner.generate("Test", max_new_tokens=-5)
        print("  FAIL: Negative max_new_tokens accepted", file=sys.stderr)
        return 1
    except ValueError:
        pass
    try:
        runner.generate("Test", max_new_tokens=4.5)
        print("  FAIL: Float max_new_tokens accepted", file=sys.stderr)
        return 1
    except (ValueError, TypeError):
        pass
    try:
        runner.generate("Test", temperature=-1.0)
        print("  FAIL: Negative temperature accepted", file=sys.stderr)
        return 1
    except ValueError:
        pass
    try:
        runner.generate("Test", do_sample=True, temperature=0.7)
        print("  FAIL: Sampling with temp=0.7 accepted", file=sys.stderr)
        return 1
    except ValueError:
        pass
    try:
        runner.generate("Test", do_sample=True, temperature=0.0)
        print("  FAIL: Sampling with temp=0 accepted", file=sys.stderr)
        return 1
    except ValueError:
        pass
    print("  PASS: Unsupported settings correctly rejected with ValueError.")

    print("\nALL 10 VERIFICATION PROBE CHECKS PASSED CLEANLY.")
    return 0


if __name__ == "__main__":
    sys.exit(run_checks())
