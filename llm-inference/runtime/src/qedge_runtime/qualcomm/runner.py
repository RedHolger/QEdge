"""ONNX Runtime model runner for exported graphs on CPU/QNN (Stage 11 - Repaired).

Provides:
- Tokenization via canonical SmolLM2-135M-Instruct tokenizer with template digest checking.
- Mandatory full export provenance and graph interface validation before session initialization.
- Autoregressive greedy token generation with full-prefix recomputation (no KV cache).
- Strict hardware status governance: distinguishes requested providers, registered providers,
  backend, session type, executed placement, and fallback.
- Hardware status remains UNVERIFIED under all circumstances in the local environment;
  caller-supplied evidence is recorded strictly as unvalidated metadata.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from qedge_runtime.acquisition import check_model_assets
from qedge_runtime.config import (
    SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST,
    compute_file_sha256,
    get_default_model_dir,
)
from qedge_runtime.qualcomm.provenance import (
    ProvenanceValidationError,
    validate_onnx_export_provenance,
)


class ONNXModelRunner:
    """Inference runner executing exported SmolLM2 ONNX models via ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str | Path,
        model_dir: Optional[str | Path] = None,
        providers: Optional[list[str]] = None,
        provider_options: Optional[list[dict[str, Any]]] = None,
        require_verified: bool = True,
    ):
        self.onnx_path = Path(onnx_path).resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

        self.model_dir = Path(model_dir).resolve() if model_dir else get_default_model_dir()

        # Compute onnx artifact digest
        self.onnx_sha256 = compute_file_sha256(self.onnx_path)

        if require_verified:
            # 1. Checkpoint source assets validation
            is_valid, errors = check_model_assets(self.model_dir)
            if not is_valid:
                raise ValueError(f"Model directory assets unverified: {errors}")

            # 2. Strict export manifest and graph interface validation before session creation
            try:
                self.manifest = validate_onnx_export_provenance(
                    onnx_path=self.onnx_path,
                    model_dir=self.model_dir,
                )
            except ProvenanceValidationError as exc:
                raise ValueError(f"Export provenance validation failed: {exc}") from exc
        else:
            self.manifest = None

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)

        # Verify chat template digest
        chat_template = getattr(self.tokenizer, "chat_template", None) or ""
        import hashlib
        tmpl_digest = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if require_verified and tmpl_digest != SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST:
            raise ValueError(
                f"Tokenizer chat_template digest mismatch: expected {SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST}, got {tmpl_digest}"
            )
        self.effective_template_digest = tmpl_digest

        # Provider configuration
        self.requested_providers = list(providers) if providers else ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Error only

        # Extract backend if specified in provider options
        self.backend: Optional[str] = None
        if provider_options:
            for opt in provider_options:
                if isinstance(opt, dict) and "backend_path" in opt:
                    self.backend = opt["backend_path"]
            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=self.requested_providers,
                provider_options=provider_options,
            )
        else:
            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=self.requested_providers,
            )

        self.registered_providers = self.session.get_providers()
        self.active_providers = self.registered_providers
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Classify session type based on registered providers
        is_cpu_only = all(p == "CPUExecutionProvider" for p in self.registered_providers)
        if is_cpu_only:
            self.session_type = "cpu_only"
        elif any("QNN" in p for p in self.registered_providers):
            self.session_type = "unverified_qnn"
        else:
            self.session_type = "mixed_or_accelerator"

    def close(self) -> None:
        """Explicitly release ONNX Runtime session and native resources."""
        if hasattr(self, "session") and self.session is not None:
            self.session = None

    def __enter__(self) -> ONNXModelRunner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        use_chat_template: bool = True,
        timeout_s: Optional[float] = None,
        hardware_run_evidence: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate output tokens autoregressively using full-prefix recomputation (no KV cache)."""
        start_time = time.perf_counter()

        # Format input
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
        current_tokens = list(input_ids)
        prompt_len = len(current_tokens)

        eos_id = self.tokenizer.eos_token_id

        # 1. Prefill step (full prompt forward pass)
        t0 = time.perf_counter()
        inp_arr = np.array([current_tokens], dtype=np.int64)
        out = self.session.run([self.output_name], {self.input_name: inp_arr})[0]
        ttft_s = time.perf_counter() - t0

        first_token = int(np.argmax(out[0, -1, :]))
        current_tokens.append(first_token)
        generated_tokens = [first_token]

        # 2. Autoregressive decode loop: full prefix recomputed on each step (use_cache=False)
        t_decode_start = time.perf_counter()
        decode_steps = 0

        while len(generated_tokens) < max_tokens and generated_tokens[-1] != eos_id:
            if timeout_s is not None and (time.perf_counter() - start_time) > timeout_s:
                break

            inp_arr = np.array([current_tokens], dtype=np.int64)
            out = self.session.run([self.output_name], {self.input_name: inp_arr})[0]
            next_token = int(np.argmax(out[0, -1, :]))
            current_tokens.append(next_token)
            generated_tokens.append(next_token)
            decode_steps += 1

        decode_time_s = time.perf_counter() - t_decode_start
        total_time_s = time.perf_counter() - start_time

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Hardware governance:
        # A caller-supplied boolean and model checksum are NOT execution proof.
        # Until an actual evidence-validation mechanism exists, hardware remains UNVERIFIED.
        # Any supplied evidence is recorded strictly as unvalidated metadata.
        hardware_status = "UNVERIFIED"

        if self.session_type == "cpu_only":
            executed_placement = "CPU (CPUExecutionProvider)"
            fallback_detected = False
        else:
            # In mixed/QNN sessions without verified placement traces, do NOT infer CPU or HTP placement,
            # and do NOT infer absence of fallback when placement is unknown.
            executed_placement = "UNKNOWN (placement logs not verified)"
            fallback_detected = "UNKNOWN (node placement unverified)"

        return {
            "text": text,
            "generated_tokens": generated_tokens,
            "prompt_tokens_count": prompt_len,
            "output_tokens_count": len(generated_tokens),
            "decode_steps": decode_steps,
            "ttft_ms": round(ttft_s * 1000.0, 3),
            "decode_ms": round(decode_time_s * 1000.0, 3),
            "total_ms": round(total_time_s * 1000.0, 3),
            "requested_providers": self.requested_providers,
            "registered_providers": self.registered_providers,
            "active_providers": self.active_providers,
            "session_type": self.session_type,
            "backend": self.backend or "none",
            "executed_placement": executed_placement,
            "fallback_detected": fallback_detected,
            "hardware_status": hardware_status,
            "model_onnx_sha256": self.onnx_sha256,
            "graph_execution_mode": "full_prefix_recomputation_no_kv_cache",
            "unvalidated_hardware_evidence": hardware_run_evidence,
        }
