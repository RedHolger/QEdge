"""Unit tests for contract v2 benchmark emission and schema validation."""
import unittest
from pathlib import Path
import tempfile

from qedge_common.schema import validate as validate_contract, ContractValidationError
from qedge_runtime.benchmark import (
    build_smoke_record,
    build_benchmark_record,
    main as bench_main,
)
from qedge_runtime.runner import GenerationResult
from qedge_runtime.config import SMOLLM2_135M_ID, SMOLLM2_135M_REVISION


class TestBenchmarkContracts(unittest.TestCase):
    def test_smoke_record_validates(self):
        rec = build_smoke_record()
        self.assertEqual(rec["schema_version"], "2")
        self.assertEqual(rec["kind"], "fixture-smoke")
        self.assertFalse(rec["measured"])
        validate_contract("benchmark_result", rec)

    def test_real_benchmark_record_validates(self):
        dummy_res = GenerationResult(
            prompt="What is edge AI?",
            prompt_token_ids=[1, 2, 3],
            generated_token_ids=[4, 5, 6, 7],
            generated_text="Edge AI executes models locally.",
            stopped_on_eos=True,
            model_name=SMOLLM2_135M_ID,
            model_revision=SMOLLM2_135M_REVISION,
            device="cpu",
            dtype="float32",
            load_ms=100.0,
            formatting_ms=1.0,
            tokenization_ms=2.0,
            prefill_ms=15.0,
            first_token_select_ms=0.5,
            ttft_ms=17.5,
            decode_ms=30.0,
            detokenization_ms=1.5,
            total_ms=50.0,
            prefill_tok_per_sec=200.0,
            decode_tok_per_sec=133.3,
        )

        rec = build_benchmark_record(dummy_res, run_id="test-run-123")
        self.assertEqual(rec["schema_version"], "2")
        self.assertEqual(rec["kind"], "runtime-generation")
        self.assertTrue(rec["measured"])
        self.assertEqual(rec["model"]["name"], SMOLLM2_135M_ID)
        self.assertEqual(rec["outputs"]["text"], "Edge AI executes models locally.")
        self.assertEqual(rec["outputs"]["token_ids"], [4, 5, 6, 7])
        self.assertNotIn("text", rec["metrics"])
        self.assertNotIn("token_ids", rec["metrics"])

        # Contract validation
        validate_contract("benchmark_result", rec)

    def test_zero_output_benchmark_record_validates(self):
        """Zero tokens generated: ttft_ms is not present in metrics."""
        zero_res = GenerationResult(
            prompt="Hello",
            prompt_token_ids=[1, 2],
            generated_token_ids=[],
            generated_text="",
            stopped_on_eos=False,
            model_name=SMOLLM2_135M_ID,
            model_revision=SMOLLM2_135M_REVISION,
            device="cpu",
            dtype="float32",
            load_ms=100.0,
            formatting_ms=1.0,
            tokenization_ms=2.0,
            prefill_ms=0.0,
            first_token_select_ms=0.0,
            ttft_ms=None,
            decode_ms=0.0,
            detokenization_ms=0.0,
            total_ms=3.0,
        )
        rec = build_benchmark_record(zero_res, run_id="test-zero-tokens")
        self.assertNotIn("ttft_ms", rec["metrics"])
        validate_contract("benchmark_result", rec)

    def test_measured_run_without_model_rejected(self):
        rec = build_smoke_record()
        rec["measured"] = True
        rec["kind"] = "runtime-generation"
        # Missing model and log_path
        with self.assertRaises(ContractValidationError):
            validate_contract("benchmark_result", rec)

    def test_missing_assets_reports_error_without_smoke_fallback(self):
        """A requested real benchmark without assets must report a missing dependency (not silently emit smoke)."""
        with tempfile.TemporaryDirectory() as td:
            exit_code = bench_main(["--model-dir", td])
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
