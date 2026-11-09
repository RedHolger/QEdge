"""GGUF Model Tensor and Metadata Inspector.

Inspects raw GGUF files to prove actual loaded tensor quantization types
and ensure tokenizer and model parameters match the reference HuggingFace model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import gguf

logger = logging.getLogger(__name__)


def inspect_gguf_model(gguf_path: Path) -> Dict[str, Any]:
    """Inspect tensor types, shapes, and metadata of a GGUF model file.

    Args:
        gguf_path: Path to the GGUF file.

    Returns:
        Structured inspection report dictionary.
    """
    if not Path(gguf_path).exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    reader = gguf.GGUFReader(str(gguf_path))

    # 1. Tensors analysis
    tensor_type_counts: Dict[str, int] = {}
    tensor_bytes_by_type: Dict[str, int] = {}
    total_tensor_bytes = 0

    tensors_detail: List[Dict[str, Any]] = []
    for tensor in reader.tensors:
        t_type = tensor.tensor_type.name
        t_bytes = tensor.n_bytes
        tensor_type_counts[t_type] = tensor_type_counts.get(t_type, 0) + 1
        tensor_bytes_by_type[t_type] = tensor_bytes_by_type.get(t_type, 0) + t_bytes
        total_tensor_bytes += t_bytes
        tensors_detail.append({
            "name": tensor.name,
            "type": t_type,
            "shape": list(tensor.shape),
            "n_bytes": t_bytes,
        })

    # 2. Extract metadata fields
    fields: Dict[str, Any] = {}
    for name, field in reader.fields.items():
        # Handle string or array parts
        try:
            if field.types == [gguf.GGUFValueType.STRING]:
                val = bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
            elif field.types == [gguf.GGUFValueType.UINT32] or field.types == [gguf.GGUFValueType.INT32]:
                val = int(field.parts[field.data[0]][0])
            elif field.types == [gguf.GGUFValueType.FLOAT32]:
                val = float(field.parts[field.data[0]][0])
            elif field.types == [gguf.GGUFValueType.BOOL]:
                val = bool(field.parts[field.data[0]][0])
            else:
                val = str(field.types)
            fields[name] = val
        except Exception:
            fields[name] = "<unparsed>"

    # 3. Model specifications
    arch = fields.get("general.architecture", "")
    block_count = fields.get(f"{arch}.block_count", 0)
    embedding_length = fields.get(f"{arch}.embedding_length", 0)
    head_count = fields.get(f"{arch}.attention.head_count", 0)
    head_count_kv = fields.get(f"{arch}.attention.head_count_kv", 0)
    vocab_size = fields.get(f"{arch}.vocab_size", 0)
    bos_id = fields.get("tokenizer.ggml.bos_token_id", -1)
    eos_id = fields.get("tokenizer.ggml.eos_token_id", -1)
    chat_template = fields.get("tokenizer.chat_template", "")

    # 4. Determine verified quantization category
    primary_quant = "UNKNOWN"
    if "Q8_0" in tensor_type_counts and tensor_type_counts["Q8_0"] > 100:
        primary_quant = "Q8_0"
    elif "Q4_K" in tensor_type_counts or "Q5_0" in tensor_type_counts:
        primary_quant = "Q4_K_M"
    elif "F16" in tensor_type_counts and tensor_type_counts["F16"] > 100:
        primary_quant = "F16"
    elif "BF16" in tensor_type_counts and tensor_type_counts["BF16"] > 100:
        primary_quant = "BF16"

    report = {
        "path": str(gguf_path),
        "total_tensors": len(reader.tensors),
        "total_tensor_bytes": total_tensor_bytes,
        "primary_quantization": primary_quant,
        "tensor_type_counts": tensor_type_counts,
        "tensor_bytes_by_type": tensor_bytes_by_type,
        "architecture": arch,
        "block_count": block_count,
        "embedding_length": embedding_length,
        "head_count": head_count,
        "head_count_kv": head_count_kv,
        "vocab_size": vocab_size,
        "special_tokens": {
            "bos_id": bos_id,
            "eos_id": eos_id,
        },
        "has_chat_template": bool(chat_template and "<|im_start|>" in chat_template),
    }

    return report


def verify_gguf_against_hf_spec(
    report: Dict[str, Any],
    expected_quant: str,
) -> bool:
    """Verify that a GGUF report strictly matches SmolLM2-135M-Instruct specification.

    Args:
        report: Report from inspect_gguf_model.
        expected_quant: 'F16', 'Q8_0', or 'Q4_K_M'.

    Returns:
        True if all architecture, parameter, tokenizer, and tensor invariants hold.
    """
    assert report["architecture"] == "llama", f"Expected llama architecture, got {report['architecture']}"
    assert report["block_count"] == 30, f"Expected 30 layers, got {report['block_count']}"
    assert report["embedding_length"] == 576, f"Expected hidden size 576, got {report['embedding_length']}"
    assert report["head_count"] == 9, f"Expected 9 Q heads, got {report['head_count']}"
    assert report["head_count_kv"] == 3, f"Expected 3 KV heads, got {report['head_count_kv']}"
    assert report["total_tensors"] == 272, f"Expected 272 tensors, got {report['total_tensors']}"
    assert report["special_tokens"]["bos_id"] == 1, "Expected bos=1"
    assert report["special_tokens"]["eos_id"] == 2, "Expected eos=2"
    assert report["has_chat_template"] is True, "Expected chat template with <|im_start|>"

    counts = report["tensor_type_counts"]
    if expected_quant == "F16":
        assert counts.get("F16", 0) == 211, f"F16 model must have 211 F16 tensors, got {counts}"
        assert counts.get("F32", 0) == 61, f"F16 model must have 61 F32 norm tensors, got {counts}"
    elif expected_quant == "Q8_0":
        assert counts.get("Q8_0", 0) == 211, f"Q8_0 model must have 211 Q8_0 tensors, got {counts}"
        assert counts.get("F32", 0) == 61, f"Q8_0 model must have 61 F32 norm tensors, got {counts}"
    elif expected_quant == "Q4_K_M":
        # Mixed k-quants: Q4_K, Q5_0, Q6_K, Q8_0 + 61 F32 norm tensors
        k_quant_total = sum(counts.get(k, 0) for k in ["Q4_K", "Q5_0", "Q6_K", "Q8_0"])
        assert k_quant_total == 211, f"Q4_K_M must have 211 quantized weight tensors, got {counts}"
        assert counts.get("F32", 0) == 61, f"Q4_K_M model must have 61 F32 norm tensors, got {counts}"
    else:
        raise ValueError(f"Unknown expected quantization: {expected_quant}")

    return True
