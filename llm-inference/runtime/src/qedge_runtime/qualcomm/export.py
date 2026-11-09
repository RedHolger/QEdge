"""Export SmolLM2-135M-Instruct to ONNX and verify numerical parity on CPU (Stage 11 - Repaired).

Enforces:
- Full manifest contract and graph interface validation via validate_onnx_export_provenance.
- Strict reuse checking: rejects mismatched opsets, tampered digests, wrong revisions, or incompatible interfaces.
- Active parity verification under both standard and optimized Python (python -O) using ParityVerificationError.
- Explicit interface specification: full-prefix recomputation without KV cache (use_cache=False),
  dynamic batch/seq_len on CPU, internal causal masking and positional indexing.
- Multi-prompt, multi-step autoregressive parity verification against PyTorch CPU with active token match counting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from qedge_runtime.acquisition import check_model_assets
from qedge_runtime.config import compute_file_sha256, get_default_model_dir
from qedge_runtime.qualcomm.provenance import (
    EXPECTED_MODEL_ID,
    EXPECTED_SOURCE_REVISION,
    ParityVerificationError,
    ProvenanceValidationError,
    get_actual_graph_default_opset,
    validate_onnx_export_provenance,
)


# ---------------------------------------------------------------------------
# Custom Symbolic Operator Registration for PyTorch Legacy ONNX Exporter
# ---------------------------------------------------------------------------

_SYMBOLIC_OPS_REGISTERED = False


def register_export_symbolic_ops() -> None:
    """Register custom symbolic implementations for ops unhandled in opset 17/18."""
    global _SYMBOLIC_OPS_REGISTERED
    if _SYMBOLIC_OPS_REGISTERED:
        return

    from torch.onnx import register_custom_op_symbolic
    import torch.onnx.symbolic_helper as sym_help
    import torch.onnx.symbolic_opset11 as sym11

    def symbolic_diff(g, self, n, dim, prepend, append):
        """Translate torch.diff(..., prepend=...) into Concat + Slices + Sub."""
        if prepend is not None and not prepend.node().mustBeNone():
            full = g.op("Concat", prepend, self, axis_i=-1)
        else:
            full = self
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        max_val = g.op("Constant", value_t=torch.tensor([9223372036854775807], dtype=torch.long))
        axis = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        s1 = g.op("Slice", full, one, max_val, axis, one)
        neg_one = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        s0 = g.op("Slice", full, zero, neg_one, axis, one)
        return g.op("Sub", s1, s0)

    def symbolic_cumsum(g, self, dim, dtype=None):
        """Translate torch.cumsum on boolean tensors by inserting explicit Cast to Int64."""
        type_kind = self.type().scalarType() if hasattr(self.type(), "scalarType") else None
        if type_kind == "Bool":
            self = g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        return sym11.cumsum(g, self, dim, dtype)

    register_custom_op_symbolic("::diff", symbolic_diff, opset_version=17)
    register_custom_op_symbolic("::cumsum", symbolic_cumsum, opset_version=17)
    register_custom_op_symbolic("::diff", symbolic_diff, opset_version=18)
    register_custom_op_symbolic("::cumsum", symbolic_cumsum, opset_version=18)
    _SYMBOLIC_OPS_REGISTERED = True


# ---------------------------------------------------------------------------
# Wrapper for Export
# ---------------------------------------------------------------------------


class SmolLM2ONNXWrapper(nn.Module):
    """Encapsulates SmolLM2 forward pass emitting raw logits for full prefix recomputation."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with input_ids -> logits [batch_size, seq_len, vocab_size]."""
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits


@dataclass
class ONNXExportResult:
    """Artifact and verification metadata for exported ONNX model."""

    model_id: str
    source_revision: str
    onnx_file_path: str
    sha256: str
    file_size_bytes: int
    opset_version: int
    actual_graph_opset: int
    graph_execution_mode: str
    use_cache: bool
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, list[Any]]
    output_shapes: dict[str, list[Any]]
    manifest_path: str
    reused_existing: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_smollm2_to_onnx(
    model_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    opset_version: int = 17,
    force_rebuild: bool = False,
) -> ONNXExportResult:
    """Export canonical SmolLM2-135M-Instruct to ONNX with verified provenance.

    Validates requested configuration against manifest and actual graph before reuse.
    Rebuilds if manifest is missing, tampered, mismatched, or if graph fails verification.
    """
    src_dir = Path(model_dir) if model_dir else get_default_model_dir()
    is_valid, errors = check_model_assets(src_dir)
    if not is_valid:
        raise ValueError(f"Cannot export unverified checkpoint: {errors}")

    out_root = (
        Path(output_dir)
        if output_dir
        else src_dir.parent / "smollm2-135m-instruct-onnx"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    onnx_path = out_root / "smollm2-135m-instruct.onnx"
    manifest_path = out_root / "export_manifest.json"

    # Strict reuse validation via shared validate_onnx_export_provenance
    if not force_rebuild and onnx_path.is_file() and manifest_path.is_file():
        try:
            manifest_data = validate_onnx_export_provenance(
                onnx_path=onnx_path,
                manifest_path=manifest_path,
                model_dir=src_dir,
                requested_opset=opset_version,
            )
            # Confirmed valid: reuse existing artifact
            actual_hash = compute_file_sha256(onnx_path)
            actual_size = onnx_path.stat().st_size
            onnx_model = onnx.load(str(onnx_path), load_external_data=False)
            actual_graph_opset = get_actual_graph_default_opset(onnx_model)

            return ONNXExportResult(
                model_id=manifest_data["model_id"],
                source_revision=manifest_data["source_revision"],
                onnx_file_path=str(onnx_path),
                sha256=actual_hash,
                file_size_bytes=actual_size,
                opset_version=opset_version,
                actual_graph_opset=actual_graph_opset,
                graph_execution_mode="full_prefix_recomputation_no_kv_cache",
                use_cache=False,
                input_names=["input_ids"],
                output_names=["logits"],
                input_shapes={"input_ids": ["batch_size", "seq_len"]},
                output_shapes={"logits": ["batch_size", "seq_len", 49152]},
                manifest_path=str(manifest_path),
                reused_existing=True,
            )
        except ProvenanceValidationError:
            # Rebuild on any validation failure
            pass

    register_export_symbolic_ops()

    # Load with eager attention to bypass PyTorch SDPA GQA conversion assertion
    model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    wrapper = SmolLM2ONNXWrapper(model)
    dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    actual_hash = compute_file_sha256(onnx_path)
    file_size = onnx_path.stat().st_size

    # Validate exported model on disk
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    actual_graph_opset = get_actual_graph_default_opset(onnx_model)

    manifest_payload = {
        "model_id": EXPECTED_MODEL_ID,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "export_format": "ONNX",
        "opset_version": opset_version,
        "onnx_file": onnx_path.name,
        "sha256": actual_hash,
        "file_size_bytes": file_size,
        "graph_execution_mode": "full_prefix_recomputation_no_kv_cache",
        "use_cache": False,
        "input_schema": {"input_ids": {"shape": ["batch_size", "seq_len"], "dtype": "int64"}},
        "output_schema": {"logits": {"shape": ["batch_size", "seq_len", 49152], "dtype": "float32"}},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    # Validate written manifest
    validate_onnx_export_provenance(
        onnx_path=onnx_path,
        manifest_path=manifest_path,
        model_dir=src_dir,
        requested_opset=opset_version,
    )

    return ONNXExportResult(
        model_id=EXPECTED_MODEL_ID,
        source_revision=EXPECTED_SOURCE_REVISION,
        onnx_file_path=str(onnx_path),
        sha256=actual_hash,
        file_size_bytes=file_size,
        opset_version=opset_version,
        actual_graph_opset=actual_graph_opset,
        graph_execution_mode="full_prefix_recomputation_no_kv_cache",
        use_cache=False,
        input_names=["input_ids"],
        output_names=["logits"],
        input_shapes={"input_ids": ["batch_size", "seq_len"]},
        output_shapes={"logits": ["batch_size", "seq_len", 49152]},
        manifest_path=str(manifest_path),
        reused_existing=False,
    )


# ---------------------------------------------------------------------------
# Active Numerical Parity Verification Functions (Active under python -O)
# ---------------------------------------------------------------------------


def verify_onnx_cpu_numerical_parity(
    onnx_path: str | Path,
    model_dir: Optional[str | Path] = None,
    test_input_ids: Optional[list[int]] = None,
    max_tolerance: float = 1.0e-3,
    mean_tolerance: float = 2.0e-4,
) -> dict[str, Any]:
    """Single forward pass per-position argmax comparison on CPU.

    Enforces max absolute diff <= 1.0e-3, mean absolute diff <= 2.0e-4, and matching argmax tokens.
    Uses active ParityVerificationError checks that remain fully active under python -O.
    """
    import onnxruntime as ort

    src_dir = Path(model_dir) if model_dir else get_default_model_dir()
    tokens = test_input_ids or [1, 2, 3, 4]
    if not tokens:
        raise ParityVerificationError("test_input_ids cannot be empty")

    input_tensor = torch.tensor([tokens], dtype=torch.long)

    # 1. PyTorch CPU baseline forward pass
    model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    with torch.no_grad():
        pt_logits = model(input_ids=input_tensor, use_cache=False).logits.numpy()

    # 2. ONNX Runtime CPUExecutionProvider forward pass
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {"input_ids": input_tensor.numpy()})[0]

    # 3. Numerical comparisons
    abs_diff = np.abs(pt_logits - ort_logits)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Reject non-finite metrics
    if not np.isfinite(max_abs_diff) or not np.isfinite(mean_abs_diff):
        raise ParityVerificationError(
            f"Non-finite logit difference detected: max={max_abs_diff}, mean={mean_abs_diff}"
        )

    pt_greedy = np.argmax(pt_logits, axis=-1).tolist()[0]
    ort_greedy = np.argmax(ort_logits, axis=-1).tolist()[0]
    token_parity = pt_greedy == ort_greedy

    # Active threshold checks (not suppressed by python -O)
    if max_abs_diff > max_tolerance:
        raise ParityVerificationError(
            f"Max absolute logit difference {max_abs_diff:.6e} exceeded tolerance {max_tolerance:.6e}"
        )
    if mean_abs_diff > mean_tolerance:
        raise ParityVerificationError(
            f"Mean absolute logit difference {mean_abs_diff:.6e} exceeded tolerance {mean_tolerance:.6e}"
        )
    if not token_parity:
        raise ParityVerificationError(
            f"Greedy tokens mismatched: PyTorch {pt_greedy} vs ONNX {ort_greedy}"
        )

    return {
        "verified": True,
        "verification_scope": "single_forward_pass_per_position_argmax",
        "execution_provider": "CPUExecutionProvider",
        "graph_execution_mode": "full_prefix_recomputation_no_kv_cache",
        "input_tokens": tokens,
        "logits_shape": list(ort_logits.shape),
        "max_absolute_difference": max_abs_diff,
        "mean_absolute_difference": mean_abs_diff,
        "max_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
        "pytorch_greedy_tokens": pt_greedy,
        "onnx_greedy_tokens": ort_greedy,
        "greedy_token_parity": token_parity,
        "numerical_tolerance_status": "PASS",
    }


def verify_onnx_cpu_autoregressive_parity(
    onnx_path: str | Path,
    model_dir: Optional[str | Path] = None,
    test_prompts: Optional[list[str]] = None,
    max_new_tokens: int = 8,
    max_tolerance: float = 1.0e-3,
    mean_tolerance: float = 2.0e-4,
) -> dict[str, Any]:
    """Multi-prompt, multi-step autoregressive generation comparison between PyTorch CPU and ONNX CPU.

    Verifies step-by-step greedy token generation, actively checking:
    - Max logit difference <= 1.0e-3 at every decode step.
    - Mean logit difference <= 2.0e-4 at every decode step.
    - Finite metric validation and conditional token match counting.
    - Rejects empty evaluations or zero decode steps.
    """
    import onnxruntime as ort

    src_dir = Path(model_dir) if model_dir else get_default_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), local_files_only=True)

    prompts = test_prompts if test_prompts is not None else [
        "Hello, my name is",
        "The capital of France is",
        "Qualcomm Snapdragon NPU inference",
    ]

    # Precondition validation
    if not prompts:
        raise ParityVerificationError("Cannot verify autoregressive parity with empty prompts list")
    if max_new_tokens <= 0:
        raise ParityVerificationError(f"max_new_tokens must be positive, got {max_new_tokens}")

    # Load PyTorch CPU model
    pt_model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        dtype=torch.float32,
        attn_implementation="eager",
    )
    pt_model.eval()

    # Load ONNX Runtime CPU session
    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_input_name = ort_sess.get_inputs()[0].name
    ort_output_name = ort_sess.get_outputs()[0].name

    prompt_results = []
    overall_max_diff = 0.0
    overall_mean_diffs = []
    total_steps_evaluated = 0
    total_matching_tokens = 0

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if not input_ids:
            raise ParityVerificationError(f"Tokenizer produced empty tokens for prompt '{prompt}'")

        pt_tokens = list(input_ids)
        ort_tokens = list(input_ids)

        step_records = []

        for step in range(max_new_tokens):
            # PyTorch step (full prefix, use_cache=False)
            pt_inp = torch.tensor([pt_tokens], dtype=torch.long)
            with torch.no_grad():
                pt_out = pt_model(input_ids=pt_inp, use_cache=False).logits.numpy()
            pt_step_logits = pt_out[0, -1, :]
            pt_next = int(np.argmax(pt_step_logits))

            # ONNX Runtime step (full prefix, use_cache=False)
            ort_inp = np.array([ort_tokens], dtype=np.int64)
            ort_out = ort_sess.run([ort_output_name], {ort_input_name: ort_inp})[0]
            ort_step_logits = ort_out[0, -1, :]
            ort_next = int(np.argmax(ort_step_logits))

            # Compare logits
            step_abs_diff = np.abs(pt_step_logits - ort_step_logits)
            step_max_diff = float(np.max(step_abs_diff))
            step_mean_diff = float(np.mean(step_abs_diff))

            # Non-finite rejection
            if not np.isfinite(step_max_diff) or not np.isfinite(step_mean_diff):
                raise ParityVerificationError(
                    f"Prompt '{prompt}' step {step}: non-finite difference detected: max={step_max_diff}, mean={step_mean_diff}"
                )

            overall_max_diff = max(overall_max_diff, step_max_diff)
            overall_mean_diffs.append(step_mean_diff)
            total_steps_evaluated += 1

            # Active threshold checks (active under python -O)
            if step_max_diff > max_tolerance:
                raise ParityVerificationError(
                    f"Prompt '{prompt}' step {step}: max logit diff {step_max_diff:.6e} > {max_tolerance:.6e}"
                )
            if step_mean_diff > mean_tolerance:
                raise ParityVerificationError(
                    f"Prompt '{prompt}' step {step}: mean logit diff {step_mean_diff:.6e} > {mean_tolerance:.6e}"
                )

            # Conditional matching check: ONLY increment on actual match
            if pt_next == ort_next:
                total_matching_tokens += 1
            else:
                raise ParityVerificationError(
                    f"Prompt '{prompt}' step {step}: token mismatch: PyTorch {pt_next} vs ONNX {ort_next}"
                )

            step_records.append({
                "step": step,
                "token_id": pt_next,
                "token_text": tokenizer.decode([pt_next]),
                "max_diff": step_max_diff,
                "mean_diff": step_mean_diff,
                "tokens_match": True,
            })

            pt_tokens.append(pt_next)
            ort_tokens.append(ort_next)

            if pt_next == tokenizer.eos_token_id:
                break

        pt_generated = pt_tokens[len(input_ids):]
        ort_generated = ort_tokens[len(input_ids):]

        prompt_results.append({
            "prompt": prompt,
            "prompt_length": len(input_ids),
            "generated_tokens": pt_generated,
            "generated_text": tokenizer.decode(pt_generated, skip_special_tokens=True),
            "steps_evaluated": len(step_records),
            "tokens_match_all": pt_generated == ort_generated,
            "step_records": step_records,
        })

    # Reject zero-step evaluation
    if total_steps_evaluated == 0:
        raise ParityVerificationError("Zero decode steps evaluated across all prompts")

    avg_mean_diff = float(np.mean(overall_mean_diffs))

    return {
        "verified": True,
        "verification_scope": "multi_prompt_multi_step_autoregressive_generation",
        "execution_provider": "CPUExecutionProvider",
        "graph_execution_mode": "full_prefix_recomputation_no_kv_cache",
        "num_prompts_evaluated": len(prompts),
        "total_steps_evaluated": total_steps_evaluated,
        "matching_steps_count": total_matching_tokens,
        "token_agreement_rate": total_matching_tokens / total_steps_evaluated,
        "overall_max_absolute_difference": overall_max_diff,
        "overall_average_mean_absolute_difference": avg_mean_diff,
        "max_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
        "prompt_evaluations": prompt_results,
        "numerical_tolerance_status": "PASS",
    }
