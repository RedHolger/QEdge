"""Export provenance and artifact validation for Qualcomm deployment (Stage 11 - Repaired).

Enforces:
- Full manifest contract validation (model identity, source revision, export format,
  opset version, onnx filename match, artifact sha256, file size, input schema, output schema).
- Actual ONNX graph inspection on disk (default-domain opset import, input/output interface,
  tensor data types, dimension ranks).
- Deep comparison between manifest declarations (input_schema, output_schema, onnx_file,
  graph_execution_mode, use_cache) and actual physical graph on disk.
- Rejection of contradictory schemas, tampered manifests, wrong revisions, missing opsets,
  and unrelated identity graphs.
- Shared validation between export reuse decisions and verified runner instantiation.
- Active parity verification exceptions not suppressed by python -O.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import onnx

from qedge_runtime.acquisition import check_model_assets
from qedge_runtime.config import compute_file_sha256, get_default_model_dir


EXPECTED_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
EXPECTED_SOURCE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
EXPECTED_INPUT_NAME = "input_ids"
EXPECTED_OUTPUT_NAME = "logits"
EXPECTED_VOCAB_SIZE = 49152

ONNX_TENSOR_TYPE_NAMES: dict[int, str] = {
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.DOUBLE: "float64",
    onnx.TensorProto.BOOL: "bool",
}


class ProvenanceValidationError(ValueError):
    """Raised when an ONNX model or manifest fails provenance validation."""
    pass


class ParityVerificationError(RuntimeError):
    """Raised when numerical parity or token agreement tolerances are violated."""
    pass


def get_actual_graph_default_opset(onnx_model: onnx.ModelProto) -> int:
    """Extract default-domain opset version from loaded ONNX ModelProto."""
    if not onnx_model.opset_import:
        raise ProvenanceValidationError("ONNX model has no opset_import entries")

    for op in onnx_model.opset_import:
        if op.domain == "" or op.domain == "ai.onnx":
            return op.version

    raise ProvenanceValidationError("ONNX model has no default-domain (ai.onnx) opset import")


def validate_dimensions(declared: list, actual: Any, label: str) -> None:
    """Require exact fixed/symbolic axes; unknown graph axes are unverifiable.

    This export contract permits positive fixed sizes and nonempty, whitespace-
    free ONNX symbols. A manifest cannot specialize a dynamic graph.
    """
    for axis, (value, dimension) in enumerate(zip(declared, actual)):
        location = f"{label} dimension {axis}"
        if type(value) is int:
            if value <= 0:
                raise ProvenanceValidationError(f"{location}: size must be positive")
            if not dimension.HasField("dim_value") or dimension.dim_value != value:
                raise ProvenanceValidationError(f"{location}: fixed dimension mismatch")
        elif isinstance(value, str) and value and not any(c.isspace() for c in value):
            if not dimension.HasField("dim_param") or dimension.dim_param != value:
                raise ProvenanceValidationError(f"{location}: symbolic dimension mismatch")
        else:
            raise ProvenanceValidationError(f"{location}: malformed dimension {value!r}")


def graph_has_kv_cache_tensors(onnx_model: onnx.ModelProto) -> bool:
    """Check if an ONNX model graph contains KV cache input or output tensors."""
    kv_markers = ("past_key_values", "present", "key_cache", "value_cache", "kv_cache")
    for inp in onnx_model.graph.input:
        inp_lower = inp.name.lower()
        if any(marker in inp_lower for marker in kv_markers):
            return True
    for out in onnx_model.graph.output:
        out_lower = out.name.lower()
        if any(marker in out_lower for marker in kv_markers):
            return True
    return False


def validate_onnx_export_provenance(
    onnx_path: str | Path,
    manifest_path: Optional[str | Path] = None,
    model_dir: Optional[str | Path] = None,
    requested_opset: Optional[int] = None,
    expected_model_id: str = EXPECTED_MODEL_ID,
    expected_revision: str = EXPECTED_SOURCE_REVISION,
    expected_execution_mode: Optional[str] = "full_prefix_recomputation_no_kv_cache",
    expected_use_cache: Optional[bool] = False,
) -> dict[str, Any]:
    """Validate full export provenance contract for an exported ONNX model.

    Validates:
    1. Manifest existence, valid JSON, and all required schema fields.
    2. Model identity and pinned source revision.
    3. Export format ("ONNX") and artifact onnx_file name match against resolved file.
    4. Artifact file presence, file size, and SHA256 digest match against disk.
    5. Requested opset vs manifest opset vs actual graph default-domain opset header.
    6. ONNX model structure integrity via onnx.checker.check_model.
    7. Graph interface on disk: input named 'input_ids' (int64, rank 2), output named 'logits' (float32, rank 3).
    8. Deep comparison of manifest input_schema against actual graph inputs (names, dtypes, ranks).
    9. Deep comparison of manifest output_schema against actual graph outputs (names, dtypes, ranks, vocab dim).
    10. KV cache consistency: rejects contradictory graph_execution_mode (e.g. 'cached_decode')
        or use_cache=True for graphs without KV cache tensors.
    11. Source checkpoint asset integrity if model_dir is provided.

    Shared between export reuse and runner verification.
    """
    resolved_onnx = Path(onnx_path).resolve()
    if not resolved_onnx.is_file():
        raise ProvenanceValidationError(f"ONNX model file not found: {resolved_onnx}")

    resolved_manifest = (
        Path(manifest_path).resolve()
        if manifest_path
        else resolved_onnx.parent / "export_manifest.json"
    )
    if not resolved_manifest.is_file():
        raise ProvenanceValidationError(f"Export manifest not found: {resolved_manifest}")

    try:
        with open(resolved_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        raise ProvenanceValidationError(f"Failed to parse export manifest JSON: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ProvenanceValidationError("Export manifest root must be a JSON object")

    # 1. Required fields check
    required_fields = [
        "model_id",
        "source_revision",
        "export_format",
        "opset_version",
        "onnx_file",
        "sha256",
        "file_size_bytes",
        "input_schema",
        "output_schema",
    ]
    for field in required_fields:
        if field not in manifest or manifest[field] is None:
            raise ProvenanceValidationError(f"Export manifest missing required field: '{field}'")

    # 2. Model identity and revision
    manifest_model_id = manifest["model_id"]
    if manifest_model_id != expected_model_id:
        raise ProvenanceValidationError(
            f"Model identity mismatch: expected '{expected_model_id}', got '{manifest_model_id}'"
        )

    manifest_rev = manifest["source_revision"]
    if manifest_rev != expected_revision:
        raise ProvenanceValidationError(
            f"Source revision mismatch: expected '{expected_revision}', got '{manifest_rev}'"
        )

    manifest_fmt = manifest["export_format"]
    if manifest_fmt != "ONNX":
        raise ProvenanceValidationError(
            f"Export format mismatch: expected 'ONNX', got '{manifest_fmt}'"
        )

    # 3. ONNX filename match against resolved file on disk
    manifest_onnx_file = manifest["onnx_file"]
    if not isinstance(manifest_onnx_file, str) or not manifest_onnx_file.strip():
        raise ProvenanceValidationError("Export manifest onnx_file must be a non-empty string")
    if manifest_onnx_file != resolved_onnx.name:
        raise ProvenanceValidationError(
            f"ONNX filename mismatch: manifest declares '{manifest_onnx_file}', but resolved file is '{resolved_onnx.name}'"
        )

    # 4. File size and SHA256 verification
    actual_size = resolved_onnx.stat().st_size
    manifest_size = manifest["file_size_bytes"]
    if not isinstance(manifest_size, int) or manifest_size != actual_size:
        raise ProvenanceValidationError(
            f"File size mismatch: manifest records {manifest_size} bytes, actual is {actual_size} bytes"
        )

    actual_sha256 = compute_file_sha256(resolved_onnx)
    manifest_sha256 = manifest["sha256"]
    if not isinstance(manifest_sha256, str) or manifest_sha256.lower() != actual_sha256.lower():
        raise ProvenanceValidationError(
            f"Artifact digest mismatch: manifest records {manifest_sha256}, actual is {actual_sha256}"
        )

    # 5. Manifest opset validation
    manifest_opset = manifest["opset_version"]
    if not isinstance(manifest_opset, int):
        raise ProvenanceValidationError(
            f"Invalid opset_version in manifest: expected int, got {type(manifest_opset).__name__}"
        )

    if requested_opset is not None and manifest_opset != requested_opset:
        raise ProvenanceValidationError(
            f"Requested opset {requested_opset} does not match manifest opset {manifest_opset}"
        )

    # 6. Graph inspection on disk
    try:
        onnx_model = onnx.load(str(resolved_onnx), load_external_data=False)
        onnx.checker.check_model(onnx_model)
    except Exception as exc:
        raise ProvenanceValidationError(f"ONNX model structural verification failed: {exc}") from exc

    graph_default_opset = get_actual_graph_default_opset(onnx_model)
    if graph_default_opset != manifest_opset:
        raise ProvenanceValidationError(
            f"ONNX opset mismatch: manifest declares opset {manifest_opset}, but actual graph default opset is {graph_default_opset}"
        )
    if requested_opset is not None and graph_default_opset != requested_opset:
        raise ProvenanceValidationError(
            f"Requested opset {requested_opset} does not match actual graph opset {graph_default_opset}"
        )

    # 7. Graph interface validation on disk
    # Validate input: single input named 'input_ids', INT64, rank 2
    if len(onnx_model.graph.input) != 1:
        input_names = [i.name for i in onnx_model.graph.input]
        raise ProvenanceValidationError(
            f"Graph interface mismatch: expected exactly 1 input ('{EXPECTED_INPUT_NAME}'), got {len(onnx_model.graph.input)}: {input_names}"
        )

    graph_input = onnx_model.graph.input[0]
    if graph_input.name != EXPECTED_INPUT_NAME:
        raise ProvenanceValidationError(
            f"Graph input name mismatch: expected '{EXPECTED_INPUT_NAME}', got '{graph_input.name}'"
        )

    input_elem_type = graph_input.type.tensor_type.elem_type
    if input_elem_type != onnx.TensorProto.INT64:
        raise ProvenanceValidationError(
            f"Graph input '{EXPECTED_INPUT_NAME}' type mismatch: expected INT64 (7), got {input_elem_type}"
        )

    input_dims = graph_input.type.tensor_type.shape.dim
    if len(input_dims) != 2:
        raise ProvenanceValidationError(
            f"Graph input '{EXPECTED_INPUT_NAME}' rank mismatch: expected rank 2 [batch, seq], got {len(input_dims)}"
        )

    # Validate output: single output named 'logits', FLOAT (float32), rank 3
    if len(onnx_model.graph.output) != 1:
        output_names = [o.name for o in onnx_model.graph.output]
        raise ProvenanceValidationError(
            f"Graph interface mismatch: expected exactly 1 output ('{EXPECTED_OUTPUT_NAME}'), got {len(onnx_model.graph.output)}: {output_names}"
        )

    graph_output = onnx_model.graph.output[0]
    if graph_output.name != EXPECTED_OUTPUT_NAME:
        raise ProvenanceValidationError(
            f"Graph output name mismatch: expected '{EXPECTED_OUTPUT_NAME}', got '{graph_output.name}'"
        )

    output_elem_type = graph_output.type.tensor_type.elem_type
    if output_elem_type != onnx.TensorProto.FLOAT:
        raise ProvenanceValidationError(
            f"Graph output '{EXPECTED_OUTPUT_NAME}' type mismatch: expected FLOAT32 (1), got {output_elem_type}"
        )

    output_dims = graph_output.type.tensor_type.shape.dim
    if len(output_dims) != 3:
        raise ProvenanceValidationError(
            f"Graph output '{EXPECTED_OUTPUT_NAME}' rank mismatch: expected rank 3 [batch, seq, vocab], got {len(output_dims)}"
        )

    # Check vocab dimension if fixed in graph
    if output_dims[2].dim_value and output_dims[2].dim_value != EXPECTED_VOCAB_SIZE:
        raise ProvenanceValidationError(
            f"Graph output vocab dimension mismatch: expected {EXPECTED_VOCAB_SIZE}, got {output_dims[2].dim_value}"
        )

    # 8. Compare manifest input_schema declarations against graph interface
    manifest_input_schema = manifest["input_schema"]
    if not isinstance(manifest_input_schema, dict) or not manifest_input_schema:
        raise ProvenanceValidationError("Export manifest input_schema must be a non-empty dictionary")

    graph_inputs_by_name = {inp.name: inp for inp in onnx_model.graph.input}
    manifest_input_keys = set(manifest_input_schema.keys())
    graph_input_keys = set(graph_inputs_by_name.keys())
    if manifest_input_keys != graph_input_keys:
        raise ProvenanceValidationError(
            f"Manifest input_schema keys {sorted(manifest_input_keys)} do not match graph inputs {sorted(graph_input_keys)}"
        )

    for inp_name, inp_decl in manifest_input_schema.items():
        if not isinstance(inp_decl, dict):
            raise ProvenanceValidationError(f"Manifest input_schema['{inp_name}'] must be a dictionary")
        if "dtype" not in inp_decl or "shape" not in inp_decl:
            raise ProvenanceValidationError(
                f"Manifest input_schema['{inp_name}'] missing required 'dtype' or 'shape' declaration"
            )

        # Validate declared shape
        decl_shape = inp_decl["shape"]
        if not isinstance(decl_shape, (list, tuple)):
            raise ProvenanceValidationError(
                f"Manifest input_schema['{inp_name}']['shape'] must be a list or tuple"
            )
        g_inp = graph_inputs_by_name[inp_name]
        g_dims = g_inp.type.tensor_type.shape.dim
        if len(decl_shape) != len(g_dims):
            raise ProvenanceValidationError(
                f"Manifest input_schema['{inp_name}'] shape rank mismatch: manifest declares rank {len(decl_shape)} ({decl_shape}), actual graph has rank {len(g_dims)}"
            )

        # For EXPECTED_INPUT_NAME, strictly enforce rank 2
        if inp_name == EXPECTED_INPUT_NAME and len(decl_shape) != 2:
            raise ProvenanceValidationError(
                f"Manifest input_schema['{inp_name}'] shape rank mismatch: expected rank 2, got {len(decl_shape)} ({decl_shape})"
            )

        validate_dimensions(decl_shape, g_dims, f"input_schema[{inp_name}]")

        # Validate declared dtype
        decl_dtype = str(inp_decl["dtype"]).lower().strip()
        g_elem_type = g_inp.type.tensor_type.elem_type
        expected_dtype_str = ONNX_TENSOR_TYPE_NAMES.get(g_elem_type)
        if expected_dtype_str and decl_dtype != expected_dtype_str:
            raise ProvenanceValidationError(
                f"Manifest input_schema['{inp_name}'] dtype mismatch: manifest declares '{inp_decl['dtype']}', actual graph requires '{expected_dtype_str}'"
            )

        # For EXPECTED_INPUT_NAME, strictly enforce int64
        if inp_name == EXPECTED_INPUT_NAME and decl_dtype != "int64":
            raise ProvenanceValidationError(
                f"Manifest input_schema['{inp_name}'] dtype mismatch: expected 'int64', got '{inp_decl['dtype']}'"
            )

    # 9. Compare manifest output_schema declarations against graph interface
    manifest_output_schema = manifest["output_schema"]
    if not isinstance(manifest_output_schema, dict) or not manifest_output_schema:
        raise ProvenanceValidationError("Export manifest output_schema must be a non-empty dictionary")

    graph_outputs_by_name = {out.name: out for out in onnx_model.graph.output}
    manifest_output_keys = set(manifest_output_schema.keys())
    graph_output_keys = set(graph_outputs_by_name.keys())
    if manifest_output_keys != graph_output_keys:
        raise ProvenanceValidationError(
            f"Manifest output_schema keys {sorted(manifest_output_keys)} do not match graph outputs {sorted(graph_output_keys)}"
        )

    for out_name, out_decl in manifest_output_schema.items():
        if not isinstance(out_decl, dict):
            raise ProvenanceValidationError(f"Manifest output_schema['{out_name}'] must be a dictionary")
        if "dtype" not in out_decl or "shape" not in out_decl:
            raise ProvenanceValidationError(
                f"Manifest output_schema['{out_name}'] missing required 'dtype' or 'shape' declaration"
            )

        # Validate declared shape
        decl_shape = out_decl["shape"]
        if not isinstance(decl_shape, (list, tuple)):
            raise ProvenanceValidationError(
                f"Manifest output_schema['{out_name}']['shape'] must be a list or tuple"
            )
        g_out = graph_outputs_by_name[out_name]
        g_dims = g_out.type.tensor_type.shape.dim
        if len(decl_shape) != len(g_dims):
            raise ProvenanceValidationError(
                f"Manifest output_schema['{out_name}'] shape rank mismatch: manifest declares rank {len(decl_shape)} ({decl_shape}), actual graph has rank {len(g_dims)}"
            )

        # For EXPECTED_OUTPUT_NAME, strictly enforce rank 3
        if out_name == EXPECTED_OUTPUT_NAME and len(decl_shape) != 3:
            raise ProvenanceValidationError(
                f"Manifest output_schema['{out_name}'] shape rank mismatch: expected rank 3, got {len(decl_shape)} ({decl_shape})"
            )

        # For EXPECTED_OUTPUT_NAME, validate vocab dimension if integer
        if out_name == EXPECTED_OUTPUT_NAME and len(decl_shape) == 3:
            vocab_dim = decl_shape[2]
            if isinstance(vocab_dim, int) and vocab_dim != EXPECTED_VOCAB_SIZE:
                raise ProvenanceValidationError(
                    f"Manifest output_schema['{out_name}'] vocab dimension mismatch: expected {EXPECTED_VOCAB_SIZE}, got {vocab_dim}"
                )

        validate_dimensions(decl_shape, g_dims, f"output_schema[{out_name}]")

        # Validate declared dtype
        decl_dtype = str(out_decl["dtype"]).lower().strip()
        g_elem_type = g_out.type.tensor_type.elem_type
        expected_dtype_str = ONNX_TENSOR_TYPE_NAMES.get(g_elem_type)
        if expected_dtype_str and decl_dtype != expected_dtype_str:
            raise ProvenanceValidationError(
                f"Manifest output_schema['{out_name}'] dtype mismatch: manifest declares '{out_decl['dtype']}', actual graph requires '{expected_dtype_str}'"
            )

        # For EXPECTED_OUTPUT_NAME, strictly enforce float32
        if out_name == EXPECTED_OUTPUT_NAME and decl_dtype not in ("float32", "float"):
            raise ProvenanceValidationError(
                f"Manifest output_schema['{out_name}'] dtype mismatch: expected 'float32', got '{out_decl['dtype']}'"
            )

    # 10. KV cache and execution mode consistency against actual graph
    has_kv = graph_has_kv_cache_tensors(onnx_model)

    # Check graph_execution_mode if declared in manifest
    if "graph_execution_mode" in manifest and manifest["graph_execution_mode"] is not None:
        decl_mode = manifest["graph_execution_mode"]
        if not isinstance(decl_mode, str):
            raise ProvenanceValidationError("Manifest graph_execution_mode must be a string")
        if not has_kv:
            if decl_mode != "full_prefix_recomputation_no_kv_cache":
                raise ProvenanceValidationError(
                    f"Contradictory graph_execution_mode: manifest declares '{decl_mode}', but graph does not contain KV cache tensors (expected 'full_prefix_recomputation_no_kv_cache')"
                )
        if expected_execution_mode is not None and decl_mode != expected_execution_mode:
            raise ProvenanceValidationError(
                f"Manifest graph_execution_mode '{decl_mode}' does not match expected '{expected_execution_mode}'"
            )

    # Check use_cache if declared in manifest
    if "use_cache" in manifest and manifest["use_cache"] is not None:
        decl_use_cache = manifest["use_cache"]
        if not isinstance(decl_use_cache, bool):
            raise ProvenanceValidationError("Manifest use_cache must be a boolean")
        if not has_kv and decl_use_cache is True:
            raise ProvenanceValidationError(
                "Contradictory use_cache: manifest declares use_cache=True, but graph does not contain KV cache tensors (no-cache graph)"
            )
        if expected_use_cache is not None and decl_use_cache != expected_use_cache:
            raise ProvenanceValidationError(
                f"Manifest use_cache={decl_use_cache} does not match expected use_cache={expected_use_cache}"
            )

    # 11. Check source model assets if model_dir provided
    if model_dir:
        src_path = Path(model_dir).resolve()
        is_valid, errors = check_model_assets(src_path)
        if not is_valid:
            raise ProvenanceValidationError(f"Source model checkpoint assets invalid: {errors}")

    return manifest
