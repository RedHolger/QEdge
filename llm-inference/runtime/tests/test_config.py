"""Unit tests for runtime configuration, pinned revisions, and templates."""
import unittest
from pathlib import Path

from qedge_runtime.config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    SMOLLM2_135M_LICENSE,
    SMOLLM2_135M_TOTAL_ESTIMATED_BYTES,
    ModelArchitectureConfig,
    format_smollm_chat_prompt,
    compute_file_sha256,
)


class TestConfig(unittest.TestCase):
    def test_pinned_metadata(self):
        self.assertEqual(SMOLLM2_135M_ID, "HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertEqual(SMOLLM2_135M_REVISION, "12fd25f77366fa6b3b4b768ec3050bf629380bac")
        self.assertEqual(SMOLLM2_135M_LICENSE, "Apache-2.0")
        self.assertGreater(SMOLLM2_135M_TOTAL_ESTIMATED_BYTES, 200_000_000)

    def test_architecture_defaults(self):
        cfg = ModelArchitectureConfig()
        self.assertEqual(cfg.vocab_size, 49152)
        self.assertEqual(cfg.hidden_size, 576)
        self.assertEqual(cfg.intermediate_size, 1536)
        self.assertEqual(cfg.num_hidden_layers, 30)
        self.assertEqual(cfg.num_attention_heads, 9)
        self.assertEqual(cfg.num_key_value_heads, 3)
        self.assertEqual(cfg.head_dim, 64)
        self.assertEqual(cfg.max_position_embeddings, 8192)
        self.assertEqual(cfg.rope_theta, 100000.0)
        self.assertEqual(cfg.bos_token_id, 1)
        self.assertEqual(cfg.eos_token_id, 2)
        self.assertEqual(cfg.pad_token_id, 2)

    def test_from_checkpoint_dict(self):
        raw = {
            "model_type": "llama",
            "vocab_size": 49152,
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "max_position_embeddings": 8192,
            "rope_theta": 100000.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 2,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
        }
        cfg = ModelArchitectureConfig.from_checkpoint_dict(raw)
        self.assertEqual(cfg.hidden_size, 576)
        self.assertEqual(cfg.num_hidden_layers, 30)
        self.assertEqual(cfg.num_attention_heads, 9)
        self.assertEqual(cfg.num_key_value_heads, 3)
        self.assertEqual(cfg.head_dim, 64)

    def test_format_smollm_chat_prompt(self):
        p = format_smollm_chat_prompt("Hello assistant!")
        self.assertIn("<|im_start|>system", p)
        self.assertIn("You are a helpful AI assistant named SmolLM", p)
        self.assertIn("<|im_start|>user\nHello assistant!<|im_end|>", p)
        self.assertTrue(p.endswith("<|im_start|>assistant\n"))

    def test_compute_file_sha256(self):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"QEdge real model runtime\n")
            tpath = Path(f.name)
        try:
            h = compute_file_sha256(tpath)
            import hashlib
            expected = hashlib.sha256(b"QEdge real model runtime\n").hexdigest()
            self.assertEqual(h, expected)
        finally:
            tpath.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
