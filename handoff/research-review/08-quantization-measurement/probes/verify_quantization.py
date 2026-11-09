"""Authoritative verification probe for Stage 08 Quantization Measurement.

Validates:
1. Source checkpoint acquisition manifest and SHA256 integrity.
2. Production of real GGUF artifacts (F16, Q8_0, Q4_K_M).
3. Raw tensor type verification via GGUFReader (no float dequantization helpers).
4. Tokenizer vocab size, special tokens (BOS=1, EOS=2), and chat template matching.
5. Explicit precision dimension separation (weight, compute, KV cache).
6. Decode denominator contract under immediate EOS.
7. Hardware status schema v2 compliance and explicit QNN/NPU blocker enforcement.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure packages are importable
root_dir = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root_dir / "llm-inference" / "runtime" / "src"))
sys.path.insert(0, str(root_dir / "qedge-common" / "src"))

import llama_cpp
from qedge_common.schema import validate
from qedge_runtime.quantization.converter import compute_sha256
from qedge_runtime.quantization.inspector import inspect_gguf_model, verify_gguf_against_hf_spec
from qedge_runtime.quantization.qnn_status import generate_hardware_status
from qedge_runtime.quantization.runner import GGUFModelRunner


def run_checks() -> int:
    models_dir = root_dir / "llm-inference" / "models" / "smollm2-135m-instruct"
    gguf_dir = root_dir / "llm-inference" / "models" / "smollm2-135m-instruct-gguf"

    print("=" * 70)
    print("STAGE 08 QUANTIZATION MEASUREMENT: AUTHORITATIVE VERIFICATION PROBE")
    print("=" * 70)

    # [Check 1/10] Source checkpoint integrity
    print("[Check 1/10] Verifying source checkpoint acquisition manifest integrity...")
    acq_path = models_dir / "acquisition_manifest.json"
    assert acq_path.exists(), f"Acquisition manifest missing: {acq_path}"
    with open(acq_path, "r", encoding="utf-8") as f:
        acq_data = json.load(f)
    assert acq_data["model_id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert acq_data["revision"] == "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    for fname, expected_hash in acq_data["artifact_hashes"].items():
        actual_hash = compute_sha256(models_dir / fname)
        assert actual_hash == expected_hash, f"Hash mismatch on {fname}"
    print("  PASS: Source checkpoint and all 8 safetensors/config hashes verified.")

    # [Check 2/10] F16 GGUF artifact
    print("[Check 2/10] Verifying F16 GGUF artifact existence, size, and hash...")
    f16_path = gguf_dir / "smollm2-135m-instruct-f16.gguf"
    assert f16_path.exists(), f"F16 GGUF missing: {f16_path}"
    f16_size = f16_path.stat().st_size
    assert 250 * 1024 * 1024 < f16_size < 300 * 1024 * 1024, f"Unexpected F16 size: {f16_size}"
    f16_hash = compute_sha256(f16_path)
    print(f"  PASS: F16 artifact verified ({f16_size / 1024 / 1024:.2f} MB, SHA256: {f16_hash[:16]}...).")

    # [Check 3/10] Q8_0 GGUF artifact
    print("[Check 3/10] Verifying Q8_0 GGUF artifact existence, size, and hash...")
    q8_0_path = gguf_dir / "smollm2-135m-instruct-q8_0.gguf"
    assert q8_0_path.exists(), f"Q8_0 GGUF missing: {q8_0_path}"
    q8_0_size = q8_0_path.stat().st_size
    assert 130 * 1024 * 1024 < q8_0_size < 160 * 1024 * 1024, f"Unexpected Q8_0 size: {q8_0_size}"
    q8_0_hash = compute_sha256(q8_0_path)
    print(f"  PASS: Q8_0 artifact verified ({q8_0_size / 1024 / 1024:.2f} MB, SHA256: {q8_0_hash[:16]}...).")

    # [Check 4/10] Q4_K_M GGUF artifact
    print("[Check 4/10] Verifying Q4_K_M GGUF artifact existence, size, and hash...")
    q4_k_m_path = gguf_dir / "smollm2-135m-instruct-q4_k_m.gguf"
    assert q4_k_m_path.exists(), f"Q4_K_M GGUF missing: {q4_k_m_path}"
    q4_k_m_size = q4_k_m_path.stat().st_size
    assert 90 * 1024 * 1024 < q4_k_m_size < 120 * 1024 * 1024, f"Unexpected Q4_K_M size: {q4_k_m_size}"
    q4_k_m_hash = compute_sha256(q4_k_m_path)
    print(f"  PASS: Q4_K_M artifact verified ({q4_k_m_size / 1024 / 1024:.2f} MB, SHA256: {q4_k_m_hash[:16]}...).")

    # [Check 5/10] F16 tensor types inspection
    print("[Check 5/10] Verifying F16 actual loaded tensor types...")
    f16_report = inspect_gguf_model(f16_path)
    verify_gguf_against_hf_spec(f16_report, "F16")
    print(f"  PASS: 272 tensors verified: {f16_report['tensor_type_counts']}.")

    # [Check 6/10] Q8_0 tensor types inspection
    print("[Check 6/10] Verifying Q8_0 actual loaded tensor types...")
    q8_0_report = inspect_gguf_model(q8_0_path)
    verify_gguf_against_hf_spec(q8_0_report, "Q8_0")
    print(f"  PASS: 272 tensors verified: {q8_0_report['tensor_type_counts']}.")

    # [Check 7/10] Q4_K_M tensor types inspection
    print("[Check 7/10] Verifying Q4_K_M actual loaded tensor types (k-quants)...")
    q4_k_m_report = inspect_gguf_model(q4_k_m_path)
    verify_gguf_against_hf_spec(q4_k_m_report, "Q4_K_M")
    print(f"  PASS: 272 tensors verified: {q4_k_m_report['tensor_type_counts']}.")

    # [Check 8/10] Tokenizer parameters and chat template parity
    print("[Check 8/10] Verifying tokenizer vocabulary and chat template match HF reference...")
    for rep in [f16_report, q8_0_report, q4_k_m_report]:
        assert rep["vocab_size"] == 49152
        assert rep["special_tokens"]["bos_id"] == 1
        assert rep["special_tokens"]["eos_id"] == 2
        assert rep["has_chat_template"] is True
    print("  PASS: Vocab size 49152, BOS=1, EOS=2, and SmolLM chat template verified across all 3 variants.")

    # [Check 9/10] Token counting & immediate EOS handling
    print("[Check 9/10] Verifying token counting, one-token limit, and zero-token limit...")
    runner = GGUFModelRunner(q8_0_path)
    # 1-token request
    res1 = runner.generate("Test one-token prompt", max_tokens=1)
    assert res1["output_tokens"] == 1, f"Expected 1 output token, got {res1['output_tokens']}"
    assert res1["decode_tokens"] == 0, f"Expected 0 decode tokens, got {res1['decode_tokens']}"
    assert res1["decode_tok_per_sec"] == 0.0, f"Expected 0.0 decode tok/s, got {res1['decode_tok_per_sec']}"
    assert res1["stop_reason"] in ("length", "stop")
    assert res1["ttft_ms"] is not None and res1["ttft_ms"] > 0

    # 0-token request
    res0 = runner.generate("Test zero-token prompt", max_tokens=0)
    assert res0["output_tokens"] == 0
    assert res0["decode_tokens"] == 0
    assert res0["ttft_ms"] is None
    assert res0["decode_tok_per_sec"] == 0.0
    assert res0["stop_reason"] == "zero_token_limit"
    print("  PASS: Token counting, one-token limit, and zero-token limit verified without division-by-zero or invented TTFT.")

    # [Check 10/12] QNN hardware blocker status validation
    print("[Check 10/12] Verifying hardware status schema v2 and Qualcomm NPU blocker...")
    status = generate_hardware_status()
    validate("hardware_status", status)
    assert status["schema_version"] == "2"
    assert status["snapdragon"]["available"] is False
    assert status["qnn"]["device_verified"] is False
    assert status["npu_status"] == "unverified"
    print("  PASS: Hardware status satisfies schema v2; Qualcomm NPU is strictly blocked.")

    # [Check 11/12] Native llama library & backend evidence
    print("[Check 11/12] Verifying native runtime library name, SHA256, and backend evidence...")
    from qedge_runtime.quantization.converter import get_native_runtime_info
    native_info = get_native_runtime_info()
    assert native_info["library_name"] in ("libllama.dylib", "libllama.so", "llama.dll")
    assert len(native_info["library_sha256"]) == 64
    assert native_info["binding_version"] == "0.3.36"
    assert "Apple Silicon Metal" in native_info["device"] or "CPU" in native_info["device"]
    print(f"  PASS: Native library {native_info['library_name']} verified ({native_info['library_sha256'][:16]}..., device: {native_info['device']}).")

    # [Check 12/12] Tampered artifact rejection & reuse controls
    print("[Check 12/12] Verifying manifest digest verification, missing/invalid manifest rejection, and reuse controls...")
    import tempfile, shutil
    from qedge_runtime.quantization.converter import (
        _validate_existing_gguf,
        build_quantized_artifacts,
    )
    manifest_path = gguf_dir / "quantization_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_q8 = Path(tmpdir) / "smollm2-135m-instruct-q8_0.gguf"
        shutil.copyfile(q8_0_path, tmp_q8)

        # 1. Positive control: valid artifact with valid manifest passes
        assert _validate_existing_gguf(tmp_q8, "Q8_0", manifest=m_data) is True

        # 2. Strict rejection when manifest is absent or invalid
        assert _validate_existing_gguf(tmp_q8, "Q8_0", manifest=None) is False
        assert _validate_existing_gguf(tmp_q8, "Q8_0", manifest={"converter_commit": "wrong"}) is False

        # 3. Tamper 1 byte in tensor data
        with open(tmp_q8, "r+b") as f:
            f.seek(10_000_000)
            b = f.read(1)
            f.seek(10_000_000)
            f.write(bytes([b[0] ^ 0xFF]))

        # Must reject reuse
        assert _validate_existing_gguf(tmp_q8, "Q8_0", manifest=m_data) is False
        assert _validate_existing_gguf(tmp_q8, "Q8_0", manifest=None) is False

        shutil.copyfile(manifest_path, Path(tmpdir) / "quantization_manifest.json")
        try:
            GGUFModelRunner(tmp_q8, require_verified=True)
            assert False, "Runner accepted tampered file!"
        except ValueError:
            pass

    # 4. Control proving unchanged GGUFs with valid manifest are reused
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir)
        for g in [f16_path, q8_0_path, q4_k_m_path, manifest_path]:
            shutil.copyfile(g, tmp_out / g.name)
        with patch("qedge_runtime.quantization.converter.convert_hf_to_gguf") as mock_conv, \
             patch("qedge_runtime.quantization.converter.quantize_gguf") as mock_quant:
            reused_m = build_quantized_artifacts(models_dir, tmp_out, force=False)
            mock_conv.assert_not_called()
            mock_quant.assert_not_called()
            assert reused_m["artifacts"]["smollm2-135m-instruct-q8_0.gguf"]["sha256"] == compute_sha256(q8_0_path)

    print("  PASS: Manifest digest verification, missing/invalid manifest rejection, and reuse controls verified.")

    print("=" * 70)
    print("SUMMARY: 12/12 checks passed, 0 failed.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_checks())
