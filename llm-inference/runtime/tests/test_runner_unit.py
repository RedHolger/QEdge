"""Unit tests for ModelRunner settings, bounds, sampling rejection, and telemetry."""
import math
import tempfile
import unittest
from pathlib import Path

from qedge_runtime.runner import ModelRunner
from qedge_runtime.acquisition import check_model_assets
from qedge_runtime.config import get_default_model_dir


class TestRunnerUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = get_default_model_dir()
        valid, missing = check_model_assets(cls.model_dir)
        if not valid:
            raise unittest.SkipTest(f"Model assets not found in {cls.model_dir}; run 'make acquire' first")
        cls.runner = ModelRunner(model_dir=cls.model_dir, device="cpu", dtype="float32", local_files_only=True)

    def test_sampling_explicitly_rejected(self):
        """Review defect 2: sampling must be rejected in greedy baseline."""
        with self.assertRaises(ValueError):
            self.runner.generate("Test", do_sample=True, temperature=0.7)

        with self.assertRaises(ValueError):
            self.runner.generate("Test", do_sample=True, temperature=0.0)

    def test_non_zero_temperature_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.generate("Test", temperature=0.5)

        with self.assertRaises(ValueError):
            self.runner.generate("Test", temperature=-1.0)

        with self.assertRaises(ValueError):
            self.runner.generate("Test", temperature=float("inf"))

        with self.assertRaises(ValueError):
            self.runner.generate("Test", temperature=float("nan"))

    def test_invalid_max_tokens_rejected(self):
        with self.assertRaises(TypeError):
            self.runner.generate("Test", max_new_tokens=4.5)

        with self.assertRaises(TypeError):
            self.runner.generate("Test", max_new_tokens=True)

        with self.assertRaises(TypeError):
            self.runner.generate("Test", max_new_tokens="10")

        with self.assertRaises(ValueError):
            self.runner.generate("Test", max_new_tokens=-1)

    def test_unsupported_top_p_rejected(self):
        with self.assertRaises(ValueError):
            self.runner.generate("Test", top_p=0.9)

    def test_zero_token_limit_and_timing_definition(self):
        """Zero-token limit: returns immediately without implying a token arrived."""
        res = self.runner.generate("Hello", max_new_tokens=0)
        self.assertEqual(len(res.generated_token_ids), 0)
        self.assertEqual(res.generated_text, "")
        self.assertFalse(res.stopped_on_eos)
        self.assertIsNone(res.ttft_ms)
        self.assertEqual(res.prefill_ms, 0.0)
        self.assertEqual(res.decode_ms, 0.0)
        self.assertGreater(res.total_ms, 0.0)

    def test_one_token_limit_and_ttft_measurement(self):
        """One-token limit: TTFT measures through first-token selection."""
        res = self.runner.generate("Hello", max_new_tokens=1)
        self.assertEqual(len(res.generated_token_ids), 1)
        self.assertIsNotNone(res.ttft_ms)
        self.assertGreater(res.ttft_ms, 0.0)
        self.assertGreater(res.first_token_select_ms, 0.0)
        self.assertGreater(res.prefill_ms, 0.0)
        self.assertEqual(res.decode_ms, 0.0)

    def test_measurement_phases_and_breakdown(self):
        """Verify explicit phase timings and total duration accounting."""
        res = self.runner.generate("What is 2 + 2?", max_new_tokens=8)
        self.assertGreater(res.formatting_ms, 0.0)
        self.assertGreater(res.tokenization_ms, 0.0)
        self.assertGreater(res.prefill_ms, 0.0)
        self.assertGreater(res.first_token_select_ms, 0.0)
        self.assertGreater(res.decode_ms, 0.0)
        self.assertGreater(res.detokenization_ms, 0.0)
        self.assertGreater(res.total_ms, 0.0)
        # TTFT encompasses tokenization + prefill + selection
        self.assertAlmostEqual(
            res.ttft_ms,
            res.tokenization_ms + res.prefill_ms + res.first_token_select_ms,
            places=4,
        )

    def test_state_isolation_repeated_requests(self):
        res1 = self.runner.generate("What is 3 + 3?", max_new_tokens=12)
        res_other = self.runner.generate("Explain gravity briefly.", max_new_tokens=12)
        res2 = self.runner.generate("What is 3 + 3?", max_new_tokens=12)

        self.assertEqual(res1.generated_token_ids, res2.generated_token_ids)
        self.assertEqual(res1.generated_text, res2.generated_text)
        self.assertNotEqual(res1.generated_token_ids, res_other.generated_token_ids)

    def test_early_eos_behavior(self):
        res = self.runner.generate("What is the capital of France?", max_new_tokens=64)
        if res.stopped_on_eos:
            self.assertIn(res.generated_token_ids[-1], self.runner.eos_token_ids)
            self.assertLess(len(res.generated_token_ids), 64)

    def test_unverified_local_model_rejected(self):
        """Negative test for review defect 1: arbitrary local dir cannot inherit SmolLM identity."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                ModelRunner(model_dir=Path(td), local_files_only=True)
            self.assertIn("Unsupported or unverified model assets", str(ctx.exception))

    def test_reproducibility_metadata(self):
        res = self.runner.generate("Test prompt", max_new_tokens=4)
        data = res.to_dict()
        self.assertIn("reproducibility", data)
        repro = data["reproducibility"]
        self.assertIn("torch", repro["dependencies"])
        self.assertIn("transformers", repro["dependencies"])
        self.assertIn("frozen", repro["lockfile"])
        self.assertEqual(repro["generation_settings"]["greedy"], True)
        self.assertEqual(repro["generation_settings"]["do_sample"], False)
        # Factual seed policy: greedy decoding default has no seed
        self.assertEqual(repro["seed_policy"], "none_deterministic_greedy")
        self.assertIsNone(repro["seed"])
        # Factual template identity: derived from effective template digest
        self.assertIn("template", repro)
        self.assertFalse(repro["template"]["raw_prompt_mode"])
        self.assertIn("sha256:872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788", repro["template"]["template_identity"])
        # Factual actual_dtype: matches model parameter dtype
        self.assertEqual(repro["actual_dtype"], "float32")
        self.assertEqual(res.dtype, "float32")

    def test_explicit_seed_setting_and_factual_metadata(self):
        """Verify that passing an explicit seed sets seed policy and records it factually."""
        res = self.runner.generate("Test prompt", max_new_tokens=4, seed=123)
        repro = res.reproducibility
        self.assertEqual(repro["seed_policy"], "explicit")
        self.assertEqual(repro["seed"], 123)

        # Invalid seed types rejected
        with self.assertRaises(TypeError):
            self.runner.generate("Test", seed="123")
        with self.assertRaises(TypeError):
            self.runner.generate("Test", seed=True)

    def test_factual_template_identity_and_raw_prompt_mode(self):
        """Verify raw_prompt_mode records raw_prompt identity and skips chat template."""
        res = self.runner.generate("Test prompt", max_new_tokens=4, use_chat_template=False)
        repro = res.reproducibility
        self.assertTrue(repro["template"]["raw_prompt_mode"])
        self.assertEqual(repro["template"]["template_identity"], "raw_prompt")
        self.assertIsNone(repro["template"]["effective_template_digest"])

    def test_loader_override_cannot_change_formatted_prompt_while_verified(self):
        """Demonstrate that an extra template cannot change formatted input while provenance is verified."""
        import shutil
        from qedge_runtime.config import REQUIRED_ASSET_FILES

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for req in REQUIRED_ASSET_FILES:
                src = self.model_dir / req
                if src.is_file():
                    shutil.copy(src, p / req)
            manifest_src = self.model_dir / "acquisition_manifest.json"
            if manifest_src.is_file():
                shutil.copy(manifest_src, p / "acquisition_manifest.json")

            # Control: verified directory formats with canonical SmolLM template
            verified_runner = ModelRunner(model_dir=p, device="cpu", dtype="float32", local_files_only=True)
            formatted = verified_runner.format_prompt("hello")
            self.assertIn("<|im_start|>user\nhello<|im_end|>", formatted)
            del verified_runner

            # Inject unverified chat_template.jinja
            (p / "chat_template.jinja").write_text("UNVERIFIED OVERRIDE {{ messages[0]['content'] }}", encoding="utf-8")

            # ModelRunner refuses to load because provenance verification rejects the override asset
            with self.assertRaises(ValueError) as ctx:
                ModelRunner(model_dir=p, device="cpu", dtype="float32", local_files_only=True)
            self.assertIn("Unexpected unverified loader asset: 'chat_template.jinja'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
