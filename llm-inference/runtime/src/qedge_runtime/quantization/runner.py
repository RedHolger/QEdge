"""GGUF Model Runner for reproducible quantization evaluation.

Wraps llama_cpp.Llama with authoritative token-level generation, explicit
prefill vs first-token selection timing boundaries, rigorous prefix-cache
state clearing, immediate-EOS zero-decode guarantees, and input validation.

Timing Boundaries & Definitions
-------------------------------
1. Prefill Latency (prefill_ms):
   Measured time for evaluating prompt tokens via `llm.eval(prompt_tokens)`.
2. First-Token Selection Latency (first_token_selection_ms):
   Measured time for sampling the first token via `llm.sample(...)`.
3. Time To First Token (ttft_ms):
   prefill_ms + first_token_selection_ms.
   For zero-token requests (max_tokens=0), ttft_ms is None (not invented).
4. Decode Latency (decode_ms):
   Measured elapsed time across all subsequent autoregressive decode steps
   (excluding the first token). If output_tokens <= 1, decode_ms is 0.0.
5. Decode Throughput (decode_tok_per_sec):
   decode_steps / (decode_ms / 1000). Strictly 0.0 when decode_steps == 0.
6. Prefill Throughput (prefill_tok_per_sec):
   evaluated_prompt_tokens / (prefill_ms / 1000). Evaluated against only
   the prompt tokens that were actually evaluated (excluding cached prefixes).

Prefix-Cache State Isolation
----------------------------
Every independent measured trial calls generate with `clear_state=True` (default),
which resets the KV-cache via `llm.reset()` before evaluating.
Declared warmup calls use `clear_state=False` to warm kernel/weight caches.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SMOLLM_CHAT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
    "<|im_start|>user\n"
    "{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class GGUFModelRunner:
    """Inference runner for GGUF models via llama.cpp."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 512,
        n_threads: Optional[int] = 4,
        n_gpu_layers: int = -1,
        seed: int = 42,
        verbose: bool = False,
        weight_precision_override: Optional[str] = None,
        require_verified: bool = False,
        manifest_path: Optional[str | Path] = None,
    ):
        try:
            import llama_cpp
        except ImportError as e:
            raise ImportError("llama-cpp-python is required for GGUFModelRunner") from e

        from qedge_runtime.quantization.converter import (
            compute_sha256,
            get_native_runtime_info,
            verify_manifest_identity,
        )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.verbose = verbose

        # Retrieve loaded native llama library and backend evidence
        self.native_runtime_info = get_native_runtime_info()

        # Build manifest verification: required for measured runs
        manifest_file = Path(manifest_path) if manifest_path else (self.model_path.parent / "quantization_manifest.json")
        if require_verified and not manifest_file.is_file():
            raise ValueError(
                f"Build manifest quantization_manifest.json not found in {self.model_path.parent}. "
                f"Artifact verification required before loading for measured runs."
            )

        if manifest_file.is_file() and (require_verified or weight_precision_override is None):
            m_valid, m_err, m_data = verify_manifest_identity(manifest_file)
            if not m_valid:
                raise ValueError(f"Build manifest failed identity verification ({manifest_file}): {m_err}")
            if self.model_path.name not in m_data.get("artifacts", {}):
                raise ValueError(
                    f"Artifact {self.model_path.name} is not registered in build manifest {manifest_file}"
                )
            expected_sha = m_data["artifacts"][self.model_path.name]["sha256"]
            actual_sha = compute_sha256(self.model_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"GGUF artifact byte digest mismatch for {self.model_path.name}: "
                    f"expected {expected_sha} from manifest, got {actual_sha} (tampered or modified). "
                    f"Refusing to load for execution."
                )

        # Determine weight precision via structural GGUF inspection
        if weight_precision_override is not None:
            self.weight_precision = weight_precision_override
            self._precision_source = "override"
            self.weight_precision_established_by = "override"
        else:
            from qedge_runtime.quantization.inspector import (
                inspect_gguf_model,
                verify_gguf_against_hf_spec,
            )
            report = inspect_gguf_model(self.model_path)
            primary_quant = report["primary_quantization"]
            verify_gguf_against_hf_spec(report, primary_quant)
            self.weight_precision = primary_quant
            self._precision_source = "gguf_inspection"
            self.weight_precision_established_by = "structural_gguf_tensor_inspection"
            self.inspection_report = report

        # Precision dimension accounting:
        # Configured vs observed precision distinction
        self.configured_weight_precision = self.weight_precision
        self.observed_weight_precision = self.weight_precision

        # Compute precision: internal GEMM accumulator precision is not exposed by llama.cpp runtime API
        self.configured_compute_precision = "F32/F16"
        self.observed_compute_precision = "unverified"
        self.compute_precision = "unverified"
        self.compute_precision_established_by = (
            "unverified (internal kernel accumulation precision not exposed by llama.cpp runtime API)"
        )

        # KV cache precision: configured F16, read back from runtime context_params
        self.configured_kv_cache_precision = "F16"
        self.kv_cache_precision = "F16"

        self.llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            seed=self.seed,
            verbose=self.verbose,
            type_k=getattr(llama_cpp, "GGML_TYPE_F16", 1),
            type_v=getattr(llama_cpp, "GGML_TYPE_F16", 1),
        )

        # Read back KV cache types from context params where supported
        raw_k = getattr(getattr(self.llm, "context_params", None), "type_k", None)
        raw_v = getattr(getattr(self.llm, "context_params", None), "type_v", None)
        ggml_type_map = {0: "GGML_TYPE_F32", 1: "GGML_TYPE_F16", 2: "GGML_TYPE_Q4_0", 3: "GGML_TYPE_Q4_1", 8: "GGML_TYPE_Q8_0"}
        self.observed_kv_cache_type_k = ggml_type_map.get(raw_k, f"TYPE_{raw_k}") if raw_k is not None else "unknown"
        self.observed_kv_cache_type_v = ggml_type_map.get(raw_v, f"TYPE_{raw_v}") if raw_v is not None else "unknown"
        self.observed_kv_cache_precision = "GGML_TYPE_F16" if raw_k == 1 and raw_v == 1 else f"k={self.observed_kv_cache_type_k}, v={self.observed_kv_cache_type_v}"
        self.kv_cache_precision_established_by = "context_params.type_k/type_v readback from llama_context"

    def format_prompt(self, user_prompt: str) -> str:
        """Format user prompt using SmolLM chat template."""
        return SMOLLM_CHAT_TEMPLATE.format(prompt=user_prompt)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 16,
        temperature: float = 0.0,
        use_chat_template: bool = True,
        clear_state: bool = True,
    ) -> Dict[str, Any]:
        """Generate text and return authoritative token IDs, counts, and timings.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum tokens to generate (must be >= 0).
            temperature: Sampling temperature (0.0 for greedy argmax).
            use_chat_template: Whether to wrap prompt in SmolLM chat format.
            clear_state: Whether to reset KV cache state before prompt evaluation.
                         Default True for independent requests.

        Returns:
            Dict containing authoritative token IDs, token counts, TTFT,
            decode throughput, stop reason, and precision metadata.

        Raises:
            ValueError: If max_tokens < 0.
        """
        if max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative, got {max_tokens}")

        formatted_prompt = self.format_prompt(prompt) if use_chat_template else prompt
        prompt_token_ids = list(self.llm.tokenize(formatted_prompt.encode("utf-8"), special=True))
        prompt_tokens_count = len(prompt_token_ids)

        # 1. Prefix Cache Control
        if clear_state:
            self.llm.reset()
            cached_prefix_len = 0
            tokens_to_eval = prompt_token_ids
            cache_hit = False
        else:
            # Check longest prefix matching current resident input IDs
            n_past = self.llm.n_tokens
            longest = 0
            for a, b in zip(self.llm._input_ids[:n_past], prompt_token_ids):
                if a == b:
                    longest += 1
                else:
                    break
            cached_prefix_len = longest
            cache_hit = (cached_prefix_len > 0)
            if cache_hit:
                self.llm.n_tokens = cached_prefix_len
                self.llm._ctx.kv_cache_seq_rm(-1, cached_prefix_len, -1)
                tokens_to_eval = prompt_token_ids[cached_prefix_len:]
            else:
                self.llm.reset()
                tokens_to_eval = prompt_token_ids

        eval_prompt_count = len(tokens_to_eval)

        # 2. Prefill Timing Boundary
        t0 = time.perf_counter()
        if eval_prompt_count > 0:
            self.llm.eval(tokens_to_eval)
        t_prefill = time.perf_counter()
        prefill_sec = t_prefill - t0
        prefill_ms = prefill_sec * 1000.0

        # Handle zero-token request (no first-token selection, no invented TTFT)
        if max_tokens == 0:
            return {
                "text": "",
                "prompt_token_ids": prompt_token_ids,
                "generated_token_ids": [],
                "prompt_tokens": prompt_tokens_count,
                "output_tokens": 0,
                "decode_tokens": 0,
                "decode_steps": 0,
                "stop_reason": "zero_token_limit",
                "prefill_ms": round(prefill_ms, 3),
                "first_token_selection_ms": 0.0,
                "ttft_ms": None,
                "decode_ms": 0.0,
                "total_ms": round(prefill_ms, 3),
                "decode_tok_per_sec": 0.0,
                "prefill_tok_per_sec": round(eval_prompt_count / prefill_sec, 2) if prefill_sec > 1e-9 and eval_prompt_count > 0 else 0.0,
                "cache_hit": cache_hit,
                "cached_prefix_tokens": cached_prefix_len,
                "evaluated_prompt_tokens": eval_prompt_count,
                "weight_precision": self.weight_precision,
                "compute_precision": self.compute_precision,
                "kv_cache_precision": self.kv_cache_precision,
                "precision_source": self._precision_source,
                "precision_details": {
                    "weight": {
                        "configured": self.configured_weight_precision,
                        "observed": self.observed_weight_precision,
                        "established_by": self.weight_precision_established_by,
                    },
                    "kv_cache": {
                        "configured": self.configured_kv_cache_precision,
                        "observed": self.observed_kv_cache_precision,
                        "type_k": self.observed_kv_cache_type_k,
                        "type_v": self.observed_kv_cache_type_v,
                        "established_by": self.kv_cache_precision_established_by,
                    },
                    "compute": {
                        "configured": self.configured_compute_precision,
                        "observed": self.observed_compute_precision,
                        "established_by": self.compute_precision_established_by,
                    },
                },
                "native_runtime": {
                    "library_name": self.native_runtime_info["library_name"],
                    "library_sha256": self.native_runtime_info["library_sha256"],
                    "binding_version": self.native_runtime_info["binding_version"],
                    "device": self.native_runtime_info["device"],
                },
            }

        # 3. First-Token Selection Timing Boundary
        t_sample_start = time.perf_counter()
        first_token = self.llm.sample(temp=temperature)
        t_sample_end = time.perf_counter()
        first_token_selection_ms = (t_sample_end - t_sample_start) * 1000.0
        ttft_ms = prefill_ms + first_token_selection_ms

        generated_token_ids = [first_token]
        eos_id = self.llm.token_eos()

        # Immediate EOS check
        if first_token == eos_id:
            stop_reason = "stop"
        elif max_tokens == 1:
            stop_reason = "length"
        else:
            stop_reason = None

        # 4. Subsequent Autoregressive Decode Steps
        decode_steps = 0
        t_decode_start = time.perf_counter()
        if len(generated_token_ids) < max_tokens and first_token != eos_id:
            for _ in range(max_tokens - 1):
                self.llm.eval([generated_token_ids[-1]])
                next_tok = self.llm.sample(temp=temperature)
                generated_token_ids.append(next_tok)
                decode_steps += 1
                if next_tok == eos_id:
                    stop_reason = "stop"
                    break
            if stop_reason is None:
                stop_reason = "length"
        t_decode_end = time.perf_counter()

        decode_sec = max(0.0, t_decode_end - t_decode_start) if decode_steps > 0 else 0.0
        decode_ms = decode_sec * 1000.0
        decode_tok_per_sec = decode_steps / decode_sec if decode_steps > 0 and decode_sec > 1e-9 else 0.0
        prefill_tok_per_sec = eval_prompt_count / prefill_sec if prefill_sec > 1e-9 and eval_prompt_count > 0 else 0.0

        # Detokenize content tokens (excluding trailing EOS if present)
        content_token_ids = [t for t in generated_token_ids if t != eos_id]
        output_text = self.llm.detokenize(content_token_ids).decode("utf-8", errors="replace").strip()

        return {
            "text": output_text,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "prompt_tokens": prompt_tokens_count,
            "output_tokens": len(generated_token_ids),
            "decode_tokens": decode_steps,
            "decode_steps": decode_steps,
            "stop_reason": stop_reason,
            "prefill_ms": round(prefill_ms, 3),
            "first_token_selection_ms": round(first_token_selection_ms, 3),
            "ttft_ms": round(ttft_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "total_ms": round(ttft_ms + decode_ms, 3),
            "decode_tok_per_sec": round(decode_tok_per_sec, 2),
            "prefill_tok_per_sec": round(prefill_tok_per_sec, 2),
            "cache_hit": cache_hit,
            "cached_prefix_tokens": cached_prefix_len,
            "evaluated_prompt_tokens": eval_prompt_count,
            "weight_precision": self.weight_precision,
            "compute_precision": self.compute_precision,
            "kv_cache_precision": self.kv_cache_precision,
            "precision_source": self._precision_source,
            "precision_details": {
                "weight": {
                    "configured": self.configured_weight_precision,
                    "observed": self.observed_weight_precision,
                    "established_by": self.weight_precision_established_by,
                },
                "kv_cache": {
                    "configured": self.configured_kv_cache_precision,
                    "observed": self.observed_kv_cache_precision,
                    "type_k": self.observed_kv_cache_type_k,
                    "type_v": self.observed_kv_cache_type_v,
                    "established_by": self.kv_cache_precision_established_by,
                },
                "compute": {
                    "configured": self.configured_compute_precision,
                    "observed": self.observed_compute_precision,
                    "established_by": self.compute_precision_established_by,
                },
            },
            "native_runtime": {
                "library_name": self.native_runtime_info["library_name"],
                "library_sha256": self.native_runtime_info["library_sha256"],
                "binding_version": self.native_runtime_info["binding_version"],
                "device": self.native_runtime_info["device"],
            },
        }
