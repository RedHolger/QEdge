"""Parity and logits agreement tests against unmodified upstream generation."""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from qedge_runtime.runner import ModelRunner
from qedge_runtime.reference import (
    generate_upstream_reference,
    verify_greedy_token_parity,
    verify_logits_agreement,
)
from qedge_runtime.acquisition import check_model_assets
from qedge_runtime.config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    get_default_model_dir,
)


class TestParityReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = get_default_model_dir()
        valid, missing = check_model_assets(cls.model_dir)
        if not valid:
            raise unittest.SkipTest(f"Model assets not found in {cls.model_dir}; run 'make acquire' first")
        # CPU float32 ensures deterministic numerical precision across all platforms
        cls.runner = ModelRunner(model_dir=cls.model_dir, device="cpu", dtype="float32", local_files_only=True)

    def test_greedy_token_parity_and_logits_agreement(self):
        prompts = [
            "What is 2 + 2?",
            "What is the capital of France?",
            "Name the primary colors in order.",
            "Write a Python function to square a number.",
        ]

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                # Custom prefill and decode
                res = self.runner.generate(
                    prompt=prompt,
                    max_new_tokens=16,
                    record_logits=True,
                )

                # Upstream reference model.generate()
                ref_ids, ref_text, ref_scores = generate_upstream_reference(
                    model=self.runner.model,
                    tokenizer=self.runner.tokenizer,
                    prompt=prompt,
                    max_new_tokens=16,
                    formatted_prompt=self.runner.format_prompt(prompt),
                    device=self.runner.device,
                )

                # Verify token parity
                parity_ok, parity_msg = verify_greedy_token_parity(res.generated_token_ids, ref_ids)
                self.assertTrue(parity_ok, f"Token parity failed for prompt '{prompt}': {parity_msg}")

                # Verify logits numerical agreement (float32 tolerance: atol=1e-4)
                logits_ok, max_diff, logit_msg = verify_logits_agreement(
                    res.step_logits,
                    ref_scores,
                    atol=1e-4,
                    rtol=1e-4,
                )
                self.assertTrue(logits_ok, f"Logits agreement failed for prompt '{prompt}': {logit_msg}")

    def test_zero_and_one_token_upstream_parity(self):
        prompt = "Hello SmolLM"
        # Zero tokens
        res0 = self.runner.generate(prompt=prompt, max_new_tokens=0)
        ref0_ids, _, _ = generate_upstream_reference(
            self.runner.model, self.runner.tokenizer, prompt, max_new_tokens=0,
            formatted_prompt=self.runner.format_prompt(prompt), device=self.runner.device
        )
        self.assertEqual(res0.generated_token_ids, ref0_ids)

        # One token
        res1 = self.runner.generate(prompt=prompt, max_new_tokens=1)
        ref1_ids, _, _ = generate_upstream_reference(
            self.runner.model, self.runner.tokenizer, prompt, max_new_tokens=1,
            formatted_prompt=self.runner.format_prompt(prompt), device=self.runner.device
        )
        self.assertEqual(res1.generated_token_ids, ref1_ids)

    def test_mps_greedy_parity_if_available(self):
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("Apple Silicon MPS not available")

        mps_runner = ModelRunner(model_dir=self.model_dir, device="mps", dtype="bfloat16", local_files_only=True)
        prompt = "What is 10 divided by 2?"

        res = mps_runner.generate(prompt, max_new_tokens=16, record_logits=True)
        ref_ids, ref_text, ref_scores = generate_upstream_reference(
            mps_runner.model, mps_runner.tokenizer, prompt, max_new_tokens=16,
            formatted_prompt=mps_runner.format_prompt(prompt), device=mps_runner.device
        )

        parity_ok, msg = verify_greedy_token_parity(res.generated_token_ids, ref_ids)
        self.assertTrue(parity_ok, f"MPS token parity failed: {msg}")

        # Documented bfloat16 tolerance: atol=0.15 (reflecting 7-bit mantissa at magnitude ~30)
        logits_ok, max_diff, logit_msg = verify_logits_agreement(
            res.step_logits, ref_scores, atol=0.15, rtol=1e-2
        )
        self.assertTrue(logits_ok, f"MPS bfloat16 logits agreement failed: {logit_msg}")

    def test_hub_revision_propagation_without_network_downloads(self):
        """Verify that when loading from Hub ID, the pinned revision is passed explicitly to both loaders."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
             patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
            mock_tok_inst = MagicMock()
            mock_tok_inst.eos_token_id = 2
            mock_tok_inst.pad_token_id = 2
            mock_tok.return_value = mock_tok_inst

            mock_model_inst = MagicMock()
            mock_model_inst.config.eos_token_id = 2
            mock_model_inst.config.pad_token_id = 2
            mock_model_inst.parameters.return_value = [torch.zeros(1)]
            mock_model.return_value.to.return_value = mock_model_inst

            runner = ModelRunner(
                model_dir_or_id=SMOLLM2_135M_ID,
                device="cpu",
                dtype="float32",
                local_files_only=False,
            )

            # Check that revision was explicitly propagated to AutoTokenizer
            mock_tok.assert_called_once()
            _, tok_kwargs = mock_tok.call_args
            self.assertEqual(tok_kwargs.get("revision"), SMOLLM2_135M_REVISION)

            # Check that revision was explicitly propagated to AutoModelForCausalLM
            mock_model.assert_called_once()
            _, model_kwargs = mock_model.call_args
            self.assertEqual(model_kwargs.get("revision"), SMOLLM2_135M_REVISION)


if __name__ == "__main__":
    unittest.main()
