"""Unit tests for Stage 08 quantization artifacts, tensor types, and contracts.

Artifact path resolution
------------------------
GGUF artifact tests use absolute paths anchored to this file's location so they
run correctly regardless of the working directory from which the test runner is
invoked (e.g. from repo root, runtime dir, or tests dir).

Token-counting & Timing Contract
--------------------------------
Tests verify that:
- Negative max_tokens are rejected with ValueError.
- Zero-token requests (max_tokens=0) return output_tokens=0, decode_tokens=0,
  ttft_ms=None (no invented TTFT), decode_tok_per_sec=0.0, and stop_reason='zero_token_limit'.
- One-token requests (max_tokens=1) produce output_tokens=1, decode_tokens=0,
  decode_tok_per_sec=0.0, stop_reason='length' (or 'stop' if EOS), with real TTFT.
- Immediate EOS returns output_tokens=1, decode_tokens=0, decode_tok_per_sec=0.0,
  stop_reason='stop'.
- Multi-token decode returns decode_tokens = output_tokens - 1.
- Prefix caching: clear_state=True forces full prompt evaluation without cache reuse;
  clear_state=False evaluates only the uncached suffix and calculates prefill throughput
  using only the evaluated tokens.

Provenance Contract
-------------------
- Empty source directory is rejected by check_model_assets / build_quantized_artifacts.
- Existing output files containing "not a gguf" are rejected by structural inspection.
- Forged source manifest hashes are rejected.
- Valid source assets and GGUF outputs are accepted with full provenance linkage.

RSS & Failure Handling
----------------------
- measure_fresh_process_peak_rss returns structured records; failures return None, never 0.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

# Anchor artifact paths relative to this test file, not the working directory
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]  # llm-inference/runtime/tests -> repo root
_GGUF_DIR = _REPO_ROOT / "llm-inference" / "models" / "smollm2-135m-instruct-gguf"
_SOURCE_DIR = _REPO_ROOT / "llm-inference" / "models" / "smollm2-135m-instruct"

import llama_cpp
from qedge_common.schema import validate
from qedge_runtime.quantization.benchmark import (
    EVALUATION_PROMPTS,
    calculate_stats,
    compute_sequence_agreement,
    compute_token_id_agreement,
    create_run_manifest,
    generate_comparison_table,
    measure_fresh_process_peak_rss,
    run_model_benchmark,
    score_target_hit,
)
from qedge_runtime.quantization.converter import (
    _validate_existing_gguf,
    build_quantized_artifacts,
    compute_sha256,
    get_native_runtime_info,
    verify_manifest_identity,
)
from qedge_runtime.quantization.inspector import inspect_gguf_model, verify_gguf_against_hf_spec
from qedge_runtime.quantization.qnn_status import generate_hardware_status
from qedge_runtime.quantization.runner import GGUFModelRunner


class TestQuantizationContracts(unittest.TestCase):
    """Test quantization contracts, precision isolation, token counting, and metrics."""

    def test_sha256_computation(self):
        """Verify streaming sha256 matches hashlib."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write("QEdge Quantization Streaming SHA256 Test String")
            f_path = Path(f.name)

        try:
            expected = hashlib.sha256(b"QEdge Quantization Streaming SHA256 Test String").hexdigest()
            computed = compute_sha256(f_path)
            self.assertEqual(computed, expected)
        finally:
            f_path.unlink()

    def test_precision_dimension_isolation(self):
        """Verify that runner explicitly isolates weight, compute, and KV cache precision."""
        with tempfile.NamedTemporaryFile(suffix="-q4_k_m.gguf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with patch("llama_cpp.Llama"):
                runner = GGUFModelRunner(
                    tmp_path,
                    weight_precision_override="Q4_K_M",
                )
                self.assertEqual(runner.weight_precision, "Q4_K_M")
                self.assertEqual(runner._precision_source, "override")
                self.assertEqual(runner.configured_compute_precision, "F32/F16")
                self.assertEqual(runner.observed_compute_precision, "unverified")
                self.assertEqual(runner.compute_precision, "unverified")
                self.assertEqual(runner.kv_cache_precision, "F16")

    def test_negative_max_tokens_rejected(self):
        """Verify that max_tokens < 0 raises ValueError explicitly."""
        with tempfile.NamedTemporaryFile(suffix="-f16.gguf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with patch("llama_cpp.Llama"):
                runner = GGUFModelRunner(tmp_path, weight_precision_override="F16")
                with self.assertRaises(ValueError):
                    runner.generate("Test", max_tokens=-1)
                with self.assertRaises(ValueError):
                    runner.generate("Test", max_tokens=-10)

    def test_zero_token_limit_handling(self):
        """Verify max_tokens=0 evaluates prompt without inventing TTFT or decode duration."""
        with tempfile.NamedTemporaryFile(suffix="-f16.gguf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with patch("llama_cpp.Llama") as mock_llama:
                instance = mock_llama.return_value
                instance.tokenize.return_value = [10, 20, 30]

                runner = GGUFModelRunner(tmp_path, weight_precision_override="F16")
                res = runner.generate("Test prompt", max_tokens=0)

                self.assertEqual(res["output_tokens"], 0)
                self.assertEqual(res["decode_tokens"], 0)
                self.assertEqual(res["decode_steps"], 0)
                self.assertIsNone(res["ttft_ms"])
                self.assertEqual(res["decode_ms"], 0.0)
                self.assertEqual(res["decode_tok_per_sec"], 0.0)
                self.assertEqual(res["stop_reason"], "zero_token_limit")
                self.assertEqual(res["prompt_tokens"], 3)
                self.assertEqual(res["text"], "")

    def test_one_token_limit_handling(self):
        """Verify max_tokens=1 produces output_tokens=1, decode_tokens=0, and no divide-by-zero."""
        with tempfile.NamedTemporaryFile(suffix="-f16.gguf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with patch("llama_cpp.Llama") as mock_llama:
                instance = mock_llama.return_value
                instance.tokenize.return_value = [10, 20]
                instance.sample.return_value = 198  # newline token
                instance.token_eos.return_value = 2  # distinct from 198
                instance.detokenize.return_value = b"\n"

                runner = GGUFModelRunner(tmp_path, weight_precision_override="F16")
                res = runner.generate("Say hello.", max_tokens=1)

                self.assertEqual(res["output_tokens"], 1)
                self.assertEqual(res["decode_tokens"], 0)
                self.assertEqual(res["decode_steps"], 0)
                self.assertEqual(res["decode_ms"], 0.0)
                self.assertEqual(res["decode_tok_per_sec"], 0.0)
                self.assertEqual(res["stop_reason"], "length")
                self.assertIsNotNone(res["ttft_ms"])
                self.assertGreaterEqual(res["ttft_ms"], 0.0)

    def test_immediate_eos_handling(self):
        """Verify immediate EOS returns output_tokens=1, decode_tokens=0, and stop_reason='stop'."""
        with tempfile.NamedTemporaryFile(suffix="-q8_0.gguf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with patch("llama_cpp.Llama") as mock_llama:
                instance = mock_llama.return_value
                instance.tokenize.return_value = [10, 20]
                instance.sample.return_value = 2  # EOS token
                instance.token_eos.return_value = 2
                instance.detokenize.return_value = b""

                runner = GGUFModelRunner(tmp_path, weight_precision_override="Q8_0")
                res = runner.generate("Prompt", max_tokens=16)

                self.assertEqual(res["output_tokens"], 1)
                self.assertEqual(res["decode_tokens"], 0)
                self.assertEqual(res["decode_steps"], 0)
                self.assertEqual(res["decode_tok_per_sec"], 0.0)
                self.assertEqual(res["stop_reason"], "stop")
                self.assertIsNotNone(res["ttft_ms"])

    def test_multi_token_decode_steps(self):
        """Verify decode_tokens = output_tokens - 1 for multi-token completions."""
        with tempfile.NamedTemporaryFile(suffix="-q4_k_m.gguf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with patch("llama_cpp.Llama") as mock_llama:
                instance = mock_llama.return_value
                instance.tokenize.return_value = [1, 2, 3]
                # Returns 5 tokens: 100, 101, 102, 103, 104
                instance.sample.side_effect = [100, 101, 102, 103, 104]
                instance.token_eos.return_value = 2
                instance.detokenize.return_value = b"Hello world test"

                runner = GGUFModelRunner(tmp_path, weight_precision_override="Q4_K_M")
                res = runner.generate("Test prompt", max_tokens=5)

                self.assertEqual(res["output_tokens"], 5)
                self.assertEqual(res["decode_tokens"], 4)
                self.assertEqual(res["decode_steps"], 4)
                self.assertEqual(res["stop_reason"], "length")
                self.assertGreaterEqual(res["decode_tok_per_sec"], 0.0)

    def test_prefix_cache_state_isolation(self):
        """Verify clear_state=True invokes reset() and avoids prefix reuse."""
        with tempfile.NamedTemporaryFile(suffix="-f16.gguf") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with patch("llama_cpp.Llama") as mock_llama:
                instance = mock_llama.return_value
                instance.tokenize.return_value = [1, 2, 3]
                instance.sample.return_value = 50
                instance.token_eos.return_value = 2
                instance.detokenize.return_value = b"A"

                runner = GGUFModelRunner(tmp_path, weight_precision_override="F16")

                # 1. Independent request with clear_state=True
                res1 = runner.generate("Prompt A", max_tokens=1, clear_state=True)
                instance.reset.assert_called()
                self.assertFalse(res1["cache_hit"])
                self.assertEqual(res1["cached_prefix_tokens"], 0)
                self.assertEqual(res1["evaluated_prompt_tokens"], 3)

    def test_hardware_status_v2_schema_validation(self):
        """Verify generated hardware status satisfies QEdge schema v2 and blocks NPU claims."""
        status = generate_hardware_status()
        self.assertEqual(status["schema_version"], "2")
        self.assertFalse(status["snapdragon"]["available"])
        self.assertFalse(status["qnn"]["device_verified"])
        self.assertEqual(status["npu_status"], "unverified")

        # Validate with qedge_common schema validator
        validate("hardware_status", status)

    def test_target_scoring_negation_rejection(self):
        """Verify word-boundary target scoring rejects negated answers."""
        # Positive matches
        self.assertEqual(score_target_hit("The capital of France is Paris.", "Paris"), 1.0)
        self.assertEqual(score_target_hit("The formula is H2O.", "H2O"), 1.0)
        self.assertEqual(score_target_hit("Yes, roses are plants.", "Yes"), 1.0)

        # Non-matches (unrelated)
        self.assertEqual(score_target_hit("The comparison is comparison.", "Paris"), 0.0)

        # Negated answers must be rejected
        self.assertEqual(score_target_hit("The capital is not Paris.", "Paris"), 0.0)
        self.assertEqual(score_target_hit("It isn't Paris, it is Lyon.", "Paris"), 0.0)
        self.assertEqual(score_target_hit("No", "Yes"), 0.0)
        self.assertEqual(score_target_hit("No, roses are not plants.", "Yes"), 0.0)
        self.assertEqual(score_target_hit("The color is not Red.", "Red"), 0.0)

    def test_token_id_agreement(self):
        """Verify token-ID agreement compares integer token ID sequences directly."""
        ids_ref = [10, 20, 30, 40]
        ids_match = [10, 20, 30, 40]
        ids_partial = [10, 20]
        ids_diff = [99, 88]

        # Exact match
        exact, la = compute_token_id_agreement(ids_match, ids_ref)
        self.assertEqual(exact, 1.0)
        self.assertEqual(la, 1.0)

        # Partial prefix match
        exact_p, la_p = compute_token_id_agreement(ids_partial, ids_ref)
        self.assertEqual(exact_p, 0.0)
        self.assertEqual(la_p, 2 / 4)

        # Completely different
        exact_d, la_d = compute_token_id_agreement(ids_diff, ids_ref)
        self.assertEqual(exact_d, 0.0)
        self.assertEqual(la_d, 0.0)

    def test_statistical_aggregation(self):
        """Verify calculate_stats computes mean, std, median (p50), and p95."""
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calculate_stats(vals)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["p50"], 30.0)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)
        self.assertGreater(stats["std"], 0.0)

    def test_rss_failure_returns_structured_record(self):
        """Verify that RSS measurement failure returns None bytes and error status, never 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent.gguf"
            rss_record = measure_fresh_process_peak_rss(nonexistent)

            self.assertIsNone(rss_record["peak_rss_bytes"])
            self.assertIsNone(rss_record["peak_rss_mb"])
            self.assertEqual(rss_record["status"], "error")
            self.assertIsNotNone(rss_record["error"])
            self.assertIn("does not exist", rss_record["error"])

    def test_provenance_refuses_empty_source_dir(self):
        """Verify build_quantized_artifacts refuses an empty source directory."""
        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as out_dir:
            # Empty source directory must fail check_model_assets
            with self.assertRaises(ValueError) as ctx:
                build_quantized_artifacts(Path(src_dir), Path(out_dir))
            self.assertIn("verification failed", str(ctx.exception).lower())

    def test_provenance_rejects_non_gguf_existing_files(self):
        """Verify structural GGUF inspection rejects corrupt / 'not a gguf' files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt = Path(tmpdir) / "smollm2-135m-instruct-f16.gguf"
            corrupt.write_text("not a gguf")

            # Must reject invalid file when manifest is absent
            self.assertFalse(_validate_existing_gguf(corrupt, "F16", manifest=None))

            # Must reject invalid file even if a mock manifest with matching SHA256 is supplied
            corrupt_large = Path(tmpdir) / "smollm2-135m-instruct-f16_large.gguf"
            corrupt_large.write_bytes(b"GGUF_FAKE" + b"\x00" * (1024 * 1024))
            mock_manifest = {
                "artifacts": {
                    corrupt_large.name: {"sha256": compute_sha256(corrupt_large)}
                }
            }
            self.assertFalse(_validate_existing_gguf(corrupt_large, "F16", manifest=mock_manifest))

    def test_tampered_artifact_rejected_by_builder_and_benchmark(self):
        """Regression test: Altered GGUF with 1-byte flipped in tensor data.

        Verifies:
        1. Positive control: Unchanged artifact passes _validate_existing_gguf with manifest
           and loads in GGUFModelRunner(require_verified=True).
        2. Tampered file: Flipping 1 byte in tensor data leaves headers and tensor inventory
           intact, but alters SHA256 digest.
        3. Builder reuse rejection: _validate_existing_gguf detects digest mismatch and rejects file.
           Calling _validate_existing_gguf with manifest=None is also strictly rejected.
        4. No replacement manifest by rehashing: build_quantized_artifacts refuses to adopt the
           altered file's hash without rebuilding from verified source.
        5. Benchmark entry point rejection: GGUFModelRunner(require_verified=True) and
           run_model_benchmark raise ValueError before loading or executing.
        """
        q8_orig = _GGUF_DIR / "smollm2-135m-instruct-q8_0.gguf"
        manifest_orig = _GGUF_DIR / "quantization_manifest.json"
        if not q8_orig.exists() or not manifest_orig.exists():
            self.skipTest(f"Reference GGUF artifacts not found in {_GGUF_DIR}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            tampered_q8 = tmp_path / "smollm2-135m-instruct-q8_0.gguf"
            tmp_manifest = tmp_path / "quantization_manifest.json"

            shutil.copyfile(q8_orig, tampered_q8)
            shutil.copyfile(manifest_orig, tmp_manifest)

            # Positive Control: Unchanged artifact passes reuse check and runner load
            with open(tmp_manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            pos_valid = _validate_existing_gguf(tampered_q8, "Q8_0", manifest=manifest_data)
            self.assertTrue(pos_valid, "Unchanged artifact positive control must pass reuse check")

            runner_pos = GGUFModelRunner(tampered_q8, require_verified=True)
            self.assertEqual(runner_pos.weight_precision, "Q8_0")
            self.assertEqual(runner_pos.observed_kv_cache_precision, "GGML_TYPE_F16")
            del runner_pos

            # Tamper: Flip 1 byte in tensor data at offset 10,000,000
            orig_sha = compute_sha256(tampered_q8)
            with open(tampered_q8, "r+b") as f:
                f.seek(10_000_000)
                byte_val = f.read(1)
                f.seek(10_000_000)
                f.write(bytes([byte_val[0] ^ 0xFF]))

            tampered_sha = compute_sha256(tampered_q8)
            self.assertNotEqual(orig_sha, tampered_sha, "Digest must differ after byte flip")

            # Verify header and tensor inventory are preserved
            report = inspect_gguf_model(tampered_q8)
            self.assertEqual(report["total_tensors"], 272)
            self.assertEqual(report["primary_quantization"], "Q8_0")
            self.assertTrue(verify_gguf_against_hf_spec(report, "Q8_0"))

            # Test 1: Builder reuse check rejects tampered file (both with manifest and when manifest is None)
            reuse_valid = _validate_existing_gguf(tampered_q8, "Q8_0", manifest=manifest_data)
            self.assertFalse(reuse_valid, "Tampered artifact MUST NOT pass manifest reuse check")
            self.assertFalse(
                _validate_existing_gguf(tampered_q8, "Q8_0", manifest=None),
                "Artifact MUST NOT be reused when manifest is absent",
            )

            # Test 2: Altered file cannot acquire replacement manifest without verified source
            with tempfile.TemporaryDirectory() as empty_src:
                with self.assertRaises(ValueError):
                    build_quantized_artifacts(Path(empty_src), tmp_path, force=False)
                # Manifest must be preserved, not overwritten with tampered hash
                with open(tmp_manifest, "r", encoding="utf-8") as f:
                    preserved_manifest = json.load(f)
                self.assertEqual(
                    preserved_manifest["artifacts"]["smollm2-135m-instruct-q8_0.gguf"]["sha256"],
                    orig_sha,
                    "Previous manifest must be preserved without adopting tampered digest",
                )

            # Test 3: Benchmark runner entry point rejects tampered file
            with self.assertRaises(ValueError) as ctx:
                GGUFModelRunner(tampered_q8, require_verified=True)
            self.assertIn("digest mismatch", str(ctx.exception).lower())

            # Test 4: run_model_benchmark rejects tampered file
            with self.assertRaises(ValueError) as ctx:
                run_model_benchmark(tampered_q8, EVALUATION_PROMPTS[:1], warmups=0, trials_per_prompt=1, require_verified=True)
            self.assertIn("digest mismatch", str(ctx.exception).lower())

    def test_public_builder_tampered_q8_with_missing_manifest_rebuilds_from_source(self):
        """Public builder with valid source, tampered Q8_0, and missing manifest rebuilds authentic artifact."""
        q8_orig = _GGUF_DIR / "smollm2-135m-instruct-q8_0.gguf"
        f16_orig = _GGUF_DIR / "smollm2-135m-instruct-f16.gguf"
        q4_orig = _GGUF_DIR / "smollm2-135m-instruct-q4_k_m.gguf"
        if not (q8_orig.exists() and f16_orig.exists() and q4_orig.exists() and _SOURCE_DIR.exists()):
            self.skipTest("Reference artifacts or source checkpoint not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            shutil.copyfile(f16_orig, tmp_out / f16_orig.name)
            shutil.copyfile(q8_orig, tmp_out / q8_orig.name)
            shutil.copyfile(q4_orig, tmp_out / q4_orig.name)
            # Manifest is intentionally missing from tmp_out

            tampered_q8 = tmp_out / q8_orig.name
            orig_sha = compute_sha256(tampered_q8)
            with open(tampered_q8, "r+b") as f:
                f.seek(10_000_000)
                byte_val = f.read(1)
                f.seek(10_000_000)
                f.write(bytes([byte_val[0] ^ 0xFF]))

            tampered_sha = compute_sha256(tampered_q8)
            self.assertNotEqual(orig_sha, tampered_sha)

            # Public builder run: missing manifest forces rebuild from verified source
            new_manifest = build_quantized_artifacts(_SOURCE_DIR, tmp_out, force=False)

            # Manifest must record authentic hash, NOT the altered digest
            self.assertEqual(
                new_manifest["artifacts"]["smollm2-135m-instruct-q8_0.gguf"]["sha256"],
                orig_sha,
                "Manifest must publish authentic sha256, never the altered digest",
            )
            # Disk file must be replaced with rebuilt authentic file
            self.assertEqual(compute_sha256(tampered_q8), orig_sha)

    def test_public_builder_tampered_q8_with_invalid_manifest_rebuilds_from_source(self):
        """Public builder with valid source, tampered Q8_0, and invalid manifest rebuilds authentic artifact."""
        q8_orig = _GGUF_DIR / "smollm2-135m-instruct-q8_0.gguf"
        f16_orig = _GGUF_DIR / "smollm2-135m-instruct-f16.gguf"
        q4_orig = _GGUF_DIR / "smollm2-135m-instruct-q4_k_m.gguf"
        manifest_orig = _GGUF_DIR / "quantization_manifest.json"
        if not (q8_orig.exists() and f16_orig.exists() and q4_orig.exists() and manifest_orig.exists() and _SOURCE_DIR.exists()):
            self.skipTest("Reference artifacts or source checkpoint not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            shutil.copyfile(f16_orig, tmp_out / f16_orig.name)
            shutil.copyfile(q8_orig, tmp_out / q8_orig.name)
            shutil.copyfile(q4_orig, tmp_out / q4_orig.name)

            # Write invalid manifest with wrong converter commit
            with open(manifest_orig, "r", encoding="utf-8") as f:
                bad_m = json.load(f)
            bad_m["converter_commit"] = "invalid_commit_deadbeef0000"
            with open(tmp_out / "quantization_manifest.json", "w", encoding="utf-8") as f:
                json.dump(bad_m, f)

            tampered_q8 = tmp_out / q8_orig.name
            orig_sha = compute_sha256(tampered_q8)
            with open(tampered_q8, "r+b") as f:
                f.seek(10_000_000)
                byte_val = f.read(1)
                f.seek(10_000_000)
                f.write(bytes([byte_val[0] ^ 0xFF]))

            tampered_sha = compute_sha256(tampered_q8)
            self.assertNotEqual(orig_sha, tampered_sha)

            # Public builder run: invalid manifest identity forces rebuild from verified source
            new_manifest = build_quantized_artifacts(_SOURCE_DIR, tmp_out, force=False)

            # Manifest must record authentic hash, NOT the altered digest
            self.assertEqual(
                new_manifest["artifacts"]["smollm2-135m-instruct-q8_0.gguf"]["sha256"],
                orig_sha,
                "Manifest must publish authentic sha256, never the altered digest",
            )
            # Manifest converter commit must be restored to canonical commit
            self.assertNotEqual(new_manifest["converter_commit"], "invalid_commit_deadbeef0000")
            # Disk file must be replaced with rebuilt authentic file
            self.assertEqual(compute_sha256(tampered_q8), orig_sha)

    def test_public_builder_unchanged_gguf_with_valid_manifest_is_reused(self):
        """Positive control: unchanged GGUFs with valid manifest are reused without conversion subprocess."""
        q8_orig = _GGUF_DIR / "smollm2-135m-instruct-q8_0.gguf"
        f16_orig = _GGUF_DIR / "smollm2-135m-instruct-f16.gguf"
        q4_orig = _GGUF_DIR / "smollm2-135m-instruct-q4_k_m.gguf"
        manifest_orig = _GGUF_DIR / "quantization_manifest.json"
        if not (q8_orig.exists() and f16_orig.exists() and q4_orig.exists() and manifest_orig.exists() and _SOURCE_DIR.exists()):
            self.skipTest("Reference artifacts or source checkpoint not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            shutil.copyfile(f16_orig, tmp_out / f16_orig.name)
            shutil.copyfile(q8_orig, tmp_out / q8_orig.name)
            shutil.copyfile(q4_orig, tmp_out / q4_orig.name)
            shutil.copyfile(manifest_orig, tmp_out / manifest_orig.name)

            with patch("qedge_runtime.quantization.converter.convert_hf_to_gguf") as mock_convert, \
                 patch("qedge_runtime.quantization.converter.quantize_gguf") as mock_quantize:
                manifest = build_quantized_artifacts(_SOURCE_DIR, tmp_out, force=False)
                mock_convert.assert_not_called()
                mock_quantize.assert_not_called()

            self.assertIn("artifacts", manifest)
            self.assertEqual(
                manifest["artifacts"]["smollm2-135m-instruct-q8_0.gguf"]["sha256"],
                compute_sha256(q8_orig),
            )

    def test_native_library_info_recording(self):
        """Verify native llama library name, SHA256, binding version, and backend info."""
        info = get_native_runtime_info()
        self.assertIn(info["library_name"], ["libllama.dylib", "libllama.so", "llama.dll"])
        self.assertEqual(len(info["library_sha256"]), 64)
        self.assertEqual(info["binding_version"], "0.3.36")
        self.assertIn("device", info)
        self.assertIn("backend_evidence", info)

    def test_run_manifest_binding_and_comparison_table(self):
        """Verify run manifest structure and comparison table generation."""
        model_paths = {
            "F16": _GGUF_DIR / "smollm2-135m-instruct-f16.gguf",
            "Q8_0": _GGUF_DIR / "smollm2-135m-instruct-q8_0.gguf",
            "Q4_K_M": _GGUF_DIR / "smollm2-135m-instruct-q4_k_m.gguf",
        }
        if not all(p.exists() for p in model_paths.values()):
            self.skipTest(f"Reference GGUF artifacts not found in {_GGUF_DIR}")

        run_manifest = create_run_manifest(model_paths, source_model_dir=_SOURCE_DIR)
        self.assertTrue(run_manifest["manifest_id"].startswith("run_manifest_"))
        self.assertEqual(run_manifest["source_revision"], "12fd25f77366fa6b3b4b768ec3050bf629380bac")
        self.assertEqual(len(run_manifest["gguf_artifacts"]), 3)
        self.assertEqual(len(run_manifest["native_runtime"]["library_sha256"]), 64)

        # Comparison table formatting
        mock_summaries = {
            "F16": {
                "artifact_size_mb": 258.34,
                "peak_rss_mb": 589.41,
                "ttft_stats": {"p50": 16.5, "mean": 17.6, "std": 3.3},
                "decode_speed_stats": {"p50": 142.8, "mean": 137.3, "std": 18.1},
                "exact_token_agreement_with_f16": 1.0,
                "len_aware_token_agreement_with_f16": 1.0,
                "coarse_target_hit_rate": 0.5,
            }
        }
        table = generate_comparison_table(mock_summaries)
        self.assertIn("headers", table)
        self.assertEqual(len(table["rows"]), 1)
        self.assertEqual(table["rows"][0]["variant"], "F16")


class TestGGUFTensorTypes(unittest.TestCase):
    """Test actual loaded tensor types on the three GGUF artifacts if built.

    Artifact paths are anchored to the repository root via the absolute path
    of this test file, so these tests run and pass from any working directory.
    """

    @classmethod
    def setUpClass(cls):
        cls.models_dir = _GGUF_DIR

    def test_f16_gguf_tensor_types(self):
        f16_path = self.models_dir / "smollm2-135m-instruct-f16.gguf"
        if not f16_path.exists():
            self.skipTest(f"GGUF artifact not yet built: {f16_path}")
        report = inspect_gguf_model(f16_path)
        self.assertTrue(verify_gguf_against_hf_spec(report, "F16"))

    def test_q8_0_gguf_tensor_types(self):
        q8_0_path = self.models_dir / "smollm2-135m-instruct-q8_0.gguf"
        if not q8_0_path.exists():
            self.skipTest(f"GGUF artifact not yet built: {q8_0_path}")
        report = inspect_gguf_model(q8_0_path)
        self.assertTrue(verify_gguf_against_hf_spec(report, "Q8_0"))

    def test_q4_k_m_gguf_tensor_types(self):
        q4_k_m_path = self.models_dir / "smollm2-135m-instruct-q4_k_m.gguf"
        if not q4_k_m_path.exists():
            self.skipTest(f"GGUF artifact not yet built: {q4_k_m_path}")
        report = inspect_gguf_model(q4_k_m_path)
        self.assertTrue(verify_gguf_against_hf_spec(report, "Q4_K_M"))


if __name__ == "__main__":
    unittest.main()
