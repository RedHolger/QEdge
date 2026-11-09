"""Unit tests for Stage 11 Qualcomm deployment, inventory, export provenance, and parity (Repaired)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import onnx
from onnx import helper, TensorProto

from qedge_runtime.config import compute_file_sha256, get_default_model_dir
from qedge_runtime.qualcomm.cache_analysis import (
    analyze_cache_surgery_in_serialized_graphs,
    calculate_kv_cache_payload,
)
from qedge_runtime.qualcomm.compatibility import get_qualcomm_compatibility_matrix
from qedge_runtime.qualcomm.export import (
    export_smollm2_to_onnx,
    verify_onnx_cpu_autoregressive_parity,
    verify_onnx_cpu_numerical_parity,
)
from qedge_runtime.qualcomm.inventory import inventory_qualcomm_hardware_and_software
from qedge_runtime.qualcomm.provenance import (
    EXPECTED_MODEL_ID,
    EXPECTED_SOURCE_REVISION,
    ParityVerificationError,
    ProvenanceValidationError,
    validate_onnx_export_provenance,
)
from qedge_runtime.qualcomm.runner import ONNXModelRunner


def create_temporary_dummy_onnx(
    path: Path,
    input_name: str = "input_ids",
    output_name: str = "logits",
    opset: int = 17,
    input_type: int = TensorProto.INT64,
    output_type: int = TensorProto.FLOAT,
    input_shape: tuple = (1, 4),
    output_shape: tuple = (1, 4, 49152),
    ir_version: int = 10,
) -> None:
    """Helper creating a minimal valid ONNX model file for negative provenance testing."""
    X = helper.make_tensor_value_info(input_name, input_type, list(input_shape))
    Y = helper.make_tensor_value_info(output_name, output_type, list(output_shape))

    # Minimal node converting or reshaping input to output
    if input_type == output_type and input_shape == output_shape:
        node = helper.make_node("Identity", [input_name], [output_name])
    else:
        # Cast then expand or identity
        node = helper.make_node("Cast", [input_name], [output_name], to=output_type)

    graph = helper.make_graph([node], "test_graph", [X], [Y])
    opset_import = helper.make_operatorsetid("", opset)
    model = helper.make_model(graph, opset_imports=[opset_import], ir_version=ir_version)
    onnx.save(model, str(path))


class QualcommDeploymentTests(unittest.TestCase):
    """Test suite verifying Qualcomm deployment inventory, compatibility, export provenance, and CPU parity."""

    def test_inventory_behavioral_classification(self):
        """Verify inventory correctly distinguishes installed, accessible, missing, and unverified."""
        inv = inventory_qualcomm_hardware_and_software()
        data = inv.to_dict()

        self.assertIn("host_os", data)
        self.assertIn("host_arch", data)
        self.assertIn("host_cpu_brand", data)

        # Behavioral checks based on device connectivity
        if not data["device_access"]:
            self.assertEqual(data["hardware_status"], "UNVERIFIED")
            self.assertEqual(data["device_execution_status"], "BLOCKED")
            self.assertEqual(data["device_performance_status"], "BLOCKED")
            self.assertTrue(any("hardware target" in m for m in data["missing_components"]))
        else:
            self.assertIsNotNone(data["device_soc"])

        # Check installed libraries and missing providers
        self.assertTrue(data["onnx_installed"])
        self.assertTrue(data["onnxruntime_installed"])
        self.assertIn("CPUExecutionProvider", data["onnxruntime_providers"])

        # Unverified claims must be clearly tracked
        self.assertTrue(any("NPU offload" in u for u in data["unverified_claims"]))
        self.assertTrue(any("VTCM" in u for u in data["unverified_claims"]))

    def test_compatibility_matrix_and_preparation_checklist(self):
        """Verify single deployment route, provisional target profile, and pre-staging preparation checklist."""
        matrix = get_qualcomm_compatibility_matrix()

        # Single deployment route selected
        route = matrix["selected_route"]
        self.assertIn("ONNX Runtime with QNN Execution Provider", route["name"])
        self.assertIn("HTP", route["target_backend"])
        self.assertGreaterEqual(len(route["primary_documentation"]), 2)

        # Provisional target profile clearly labeled
        prov = matrix["provisional_target_profile"]
        self.assertIn("PROVISIONAL", prov["status"])
        self.assertIn("SM8650", prov["soc"])

        # Multi-SoC matrix
        hw = matrix["target_hardware_matrix"]
        soc_names = [item["soc"] for item in hw]
        self.assertTrue(any("8 Gen 2" in s for s in soc_names))
        self.assertTrue(any("8 Gen 3" in s for s in soc_names))
        self.assertTrue(any("8 Elite" in s for s in soc_names))
        self.assertTrue(any("Snapdragon X Elite" in s for s in soc_names))

        # Pre-staging checklist mandates static shapes and KV cache
        checklist = matrix["pre_staging_preparation_checklist"]
        steps = [item["step"] for item in checklist]
        self.assertTrue(any("Shape Specialization" in s for s in steps))
        self.assertTrue(any("KV Cache" in s for s in steps))
        self.assertTrue(any("Quantization" in s for s in steps))

        # Dynamic CPU graph is explicitly marked not ready for direct HTP staging
        readiness = matrix["current_export_readiness_status"]
        self.assertFalse(readiness["is_ready_for_direct_htp_staging"])
        self.assertIn("NOT READY FOR DIRECT HTP STAGING", readiness["readiness_verdict"])

    def test_cache_surgery_analysis_derived_bytes_and_unmeasured_latencies(self):
        """Verify cache analysis derives byte counts from config, distinguishes round-trip traffic, and marks physical latencies unmeasured."""
        payload = calculate_kv_cache_payload(layers=30, kv_heads=3, head_dim=64, resident_tokens=512)

        # Exact formula: 2 * 30 * 3 * 512 * 64 * 2 = 11,796,480 bytes (11.25 MiB)
        self.assertEqual(payload.fp16_one_way_payload_bytes, 11_796_480)
        self.assertEqual(payload.fp16_one_way_payload_mib, 11.25)
        # Round-trip traffic: 2 * 11,796,480 = 23,592,960 bytes (22.50 MiB)
        self.assertEqual(payload.fp16_round_trip_traffic_bytes, 23_592_960)
        self.assertEqual(payload.fp16_round_trip_traffic_mib, 22.50)

        # INT8 one-way: 5.625 MiB, round-trip: 11.25 MiB
        self.assertEqual(payload.int8_one_way_payload_bytes, 5_898_240)
        self.assertEqual(payload.int8_one_way_payload_mib, 5.625)
        self.assertEqual(payload.int8_round_trip_traffic_bytes, 11_796_480)
        self.assertEqual(payload.int8_round_trip_traffic_mib, 11.25)

        # Analysis module reports physical device measurements as None
        analysis = analyze_cache_surgery_in_serialized_graphs()
        data = analysis.to_dict()

        self.assertIsNone(data["measured_device_transfer_ms"])
        self.assertIsNone(data["measured_npu_decode_compute_ms"])
        self.assertIsNone(data["measured_transfer_to_compute_ratio"])
        self.assertFalse(data["cache_selection_supported_in_implemented_graph"])

        # Analytical estimates expose explicit formulas and assumptions
        self.assertGreaterEqual(len(data["analytical_transfer_estimates"]), 2)
        for est in data["analytical_transfer_estimates"]:
            self.assertFalse(est["is_measured"])
            self.assertIn("formula", est)
            self.assertGreater(est["assumed_bus_bandwidth_gb_s"], 0)

    def test_onnx_export_reuse_matching_configuration(self):
        """Verify exported ONNX model artifact exists, matches manifest, and is reused when configuration matches."""
        res = export_smollm2_to_onnx(opset_version=17)

        self.assertEqual(res.model_id, EXPECTED_MODEL_ID)
        self.assertEqual(res.source_revision, EXPECTED_SOURCE_REVISION)
        self.assertEqual(res.opset_version, 17)
        self.assertEqual(res.actual_graph_opset, 17)
        self.assertEqual(res.graph_execution_mode, "full_prefix_recomputation_no_kv_cache")
        self.assertFalse(res.use_cache)
        self.assertTrue(Path(res.onnx_file_path).is_file())
        self.assertTrue(Path(res.manifest_path).is_file())
        self.assertTrue(res.reused_existing)

        # Validate manifest content
        with open(res.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        self.assertEqual(manifest_data["sha256"], res.sha256)
        self.assertEqual(manifest_data["opset_version"], 17)
        self.assertEqual(manifest_data["file_size_bytes"], res.file_size_bytes)
        self.assertEqual(manifest_data["model_id"], res.model_id)

    def test_export_reuse_rejection_on_opset_mismatch(self):
        """Verify requesting opset 18 does NOT reuse opset 17 artifact and report 18."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            # 1. First export at opset 17
            res17 = export_smollm2_to_onnx(output_dir=tmp_root, opset_version=17)
            self.assertEqual(res17.opset_version, 17)
            self.assertEqual(res17.actual_graph_opset, 17)
            self.assertFalse(res17.reused_existing)

            # 2. Re-export requesting opset 18 from same directory
            # Must NOT reuse opset 17 file or report opset 18 with opset 17 graph
            res18 = export_smollm2_to_onnx(output_dir=tmp_root, opset_version=18)
            self.assertEqual(res18.opset_version, 18)
            self.assertEqual(res18.actual_graph_opset, 18)
            self.assertFalse(res18.reused_existing)

            # Check that manifest on disk was updated to opset 18
            manifest_path = tmp_root / "export_manifest.json"
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            self.assertEqual(manifest_data["opset_version"], 18)

    def test_provenance_validation_negative_probes(self):
        """Test negative provenance probes using small temporary graphs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            onnx_file = tmp_root / "model.onnx"
            man_file = tmp_root / "export_manifest.json"

            # Create an unrelated identity graph (inputs 'X', outputs 'Y')
            create_temporary_dummy_onnx(onnx_file, input_name="X", output_name="Y", opset=17)
            from qedge_runtime.config import compute_file_sha256
            actual_hash = compute_file_sha256(onnx_file)
            actual_size = onnx_file.stat().st_size

            # Probe 1: Matching checksum and wrong revision
            man_payload = {
                "model_id": EXPECTED_MODEL_ID,
                "source_revision": "wrong-revision",
                "export_format": "ONNX",
                "opset_version": 17,
                "onnx_file": onnx_file.name,
                "sha256": actual_hash,
                "file_size_bytes": actual_size,
                "input_schema": {"input_ids": {"shape": [1, 4], "dtype": "int64"}},
                "output_schema": {"logits": {"shape": [1, 4, 49152], "dtype": "float32"}},
            }
            man_file.write_text(json.dumps(man_payload), encoding="utf-8")

            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("Source revision mismatch", str(ctx.exception))

            # Probe 2: Missing opset_version
            man_payload["source_revision"] = EXPECTED_SOURCE_REVISION
            del man_payload["opset_version"]
            man_file.write_text(json.dumps(man_payload), encoding="utf-8")

            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("missing required field: 'opset_version'", str(ctx.exception))

            # Probe 3: Incompatible graph interface (input name 'X' instead of 'input_ids')
            man_payload["opset_version"] = 17
            man_file.write_text(json.dumps(man_payload), encoding="utf-8")

            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("Graph input name mismatch", str(ctx.exception))

            # Probe 4: Wrong model_id
            man_payload["model_id"] = "wrong/model"
            man_file.write_text(json.dumps(man_payload), encoding="utf-8")

            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("Model identity mismatch", str(ctx.exception))

    def test_provenance_validation_rejects_contradictory_schemas_and_cache_modes(self):
        """Verify validator rejects contradictory input/output schemas, filename mismatch, cached_decode, and use_cache=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            onnx_file = tmp_root / "smollm2-135m-instruct.onnx"
            man_file = tmp_root / "export_manifest.json"

            # Create a valid minimal SmolLM2 graph: input_ids [1, 4] INT64, logits [1, 4, 49152] FLOAT32
            create_temporary_dummy_onnx(
                onnx_file,
                input_name="input_ids",
                output_name="logits",
                opset=17,
                input_type=TensorProto.INT64,
                output_type=TensorProto.FLOAT,
                input_shape=(1, 4),
                output_shape=(1, 4, 49152),
            )
            from qedge_runtime.config import compute_file_sha256
            actual_hash = compute_file_sha256(onnx_file)
            actual_size = onnx_file.stat().st_size

            base_payload = {
                "model_id": EXPECTED_MODEL_ID,
                "source_revision": EXPECTED_SOURCE_REVISION,
                "export_format": "ONNX",
                "opset_version": 17,
                "onnx_file": "smollm2-135m-instruct.onnx",
                "sha256": actual_hash,
                "file_size_bytes": actual_size,
                "input_schema": {"input_ids": {"shape": [1, 4], "dtype": "int64"}},
                "output_schema": {"logits": {"shape": [1, 4, 49152], "dtype": "float32"}},
            }

            # Baseline check: valid base payload succeeds
            man_file.write_text(json.dumps(base_payload), encoding="utf-8")
            validated = validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertEqual(validated["model_id"], EXPECTED_MODEL_ID)

            # Probe 1: Contradictory onnx_file name
            p1 = dict(base_payload)
            p1["onnx_file"] = "different_model.onnx"
            man_file.write_text(json.dumps(p1), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("ONNX filename mismatch", str(ctx.exception))

            # Probe 2: Contradictory input_schema with wrong input name 'wrong_input', dtype 'float16', shape [99]
            p2 = dict(base_payload)
            p2["input_schema"] = {"wrong_input": {"shape": [99], "dtype": "float16"}}
            man_file.write_text(json.dumps(p2), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("input_schema keys", str(ctx.exception))

            # Probe 3: Contradictory input_schema dtype (float16 instead of int64)
            p3 = dict(base_payload)
            p3["input_schema"] = {"input_ids": {"shape": [1, 4], "dtype": "float16"}}
            man_file.write_text(json.dumps(p3), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("dtype mismatch", str(ctx.exception))

            # Probe 4: Contradictory input_schema rank ([99] rank 1 instead of rank 2)
            p4 = dict(base_payload)
            p4["input_schema"] = {"input_ids": {"shape": [99], "dtype": "int64"}}
            man_file.write_text(json.dumps(p4), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("shape rank mismatch", str(ctx.exception))

            # Probe 5: Empty output_schema ({})
            p5 = dict(base_payload)
            p5["output_schema"] = {}
            man_file.write_text(json.dumps(p5), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("output_schema must be a non-empty dictionary", str(ctx.exception))

            # Probe 6: Contradictory output_schema key ('wrong_output')
            p6 = dict(base_payload)
            p6["output_schema"] = {"wrong_output": {"shape": [1, 4, 49152], "dtype": "float32"}}
            man_file.write_text(json.dumps(p6), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("output_schema keys", str(ctx.exception))

            # Probe 7: Contradictory output_schema dtype (int64 instead of float32)
            p7 = dict(base_payload)
            p7["output_schema"] = {"logits": {"shape": [1, 4, 49152], "dtype": "int64"}}
            man_file.write_text(json.dumps(p7), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("dtype mismatch", str(ctx.exception))

            # Probe 8: Contradictory output_schema rank ([1, 4] rank 2 instead of rank 3)
            p8 = dict(base_payload)
            p8["output_schema"] = {"logits": {"shape": [1, 4], "dtype": "float32"}}
            man_file.write_text(json.dumps(p8), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("shape rank mismatch", str(ctx.exception))

            # Probe 9: Contradictory output_schema vocab size (32000 instead of 49152)
            p9 = dict(base_payload)
            p9["output_schema"] = {"logits": {"shape": [1, 4, 32000], "dtype": "float32"}}
            man_file.write_text(json.dumps(p9), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("vocab dimension mismatch", str(ctx.exception))

            # Probe 10: Contradictory graph_execution_mode ('cached_decode') for no-cache graph
            p10 = dict(base_payload)
            p10["graph_execution_mode"] = "cached_decode"
            man_file.write_text(json.dumps(p10), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("Contradictory graph_execution_mode", str(ctx.exception))

            # Probe 11: Contradictory use_cache=True for no-cache graph
            p11 = dict(base_payload)
            p11["use_cache"] = True
            man_file.write_text(json.dumps(p11), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError) as ctx:
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)
            self.assertIn("Contradictory use_cache", str(ctx.exception))

            # Probe 12: Combined contradictory manifest matching exact reviewer probe:
            # - wrong_input, float16, [99]
            # - empty output_schema
            # - different onnx_file
            # - cached_decode and use_cache=True
            p12 = {
                "model_id": EXPECTED_MODEL_ID,
                "source_revision": EXPECTED_SOURCE_REVISION,
                "export_format": "ONNX",
                "opset_version": 17,
                "onnx_file": "different_model.onnx",
                "sha256": actual_hash,
                "file_size_bytes": actual_size,
                "input_schema": {"wrong_input": {"shape": [99], "dtype": "float16"}},
                "output_schema": {},
                "graph_execution_mode": "cached_decode",
                "use_cache": True,
            }
            man_file.write_text(json.dumps(p12), encoding="utf-8")
            with self.assertRaises(ProvenanceValidationError):
                validate_onnx_export_provenance(onnx_file, manifest_path=man_file)

            # Probe 13: Runner instantiation with require_verified=True rejects contradictory manifest
            # (must raise ValueError indicating export provenance validation failed)
            with self.assertRaises(ValueError) as ctx:
                ONNXModelRunner(
                    onnx_path=onnx_file,
                    model_dir=get_default_model_dir(),
                    require_verified=True,
                )
            self.assertIn("Export provenance validation failed", str(ctx.exception))

    def test_runner_rejects_hardware_bypass_and_separates_sessions(self):
        """Verify runner rejects fabricated evidence and separates CPU from unverified QNN sessions."""
        res = export_smollm2_to_onnx()
        runner = ONNXModelRunner(res.onnx_file_path)

        # 1. CPU-only session with fabricated evidence: MUST remain UNVERIFIED and CPU placement
        fabricated_evidence = {
            "verified_htp_placement": True,
            "artifact_sha256": res.sha256,
            "notes": "fabricated claim",
        }
        out_cpu = runner.generate(
            "Test prompt", max_tokens=2, use_chat_template=False, hardware_run_evidence=fabricated_evidence
        )

        self.assertEqual(out_cpu["hardware_status"], "UNVERIFIED")
        self.assertEqual(out_cpu["session_type"], "cpu_only")
        self.assertEqual(out_cpu["executed_placement"], "CPU (CPUExecutionProvider)")
        self.assertFalse(out_cpu["fallback_detected"])
        self.assertEqual(out_cpu["unvalidated_hardware_evidence"], fabricated_evidence)

        # 2. Mock a QNN session: MUST report session_type="unverified_qnn", executed_placement="UNKNOWN", fallback="UNKNOWN"
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["QNNExecutionProvider", "CPUExecutionProvider"]
        mock_input = MagicMock()
        mock_input.name = "input_ids"
        mock_output = MagicMock()
        mock_output.name = "logits"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]

        dummy_logits = np.zeros((1, 5, 49152), dtype=np.float32)
        mock_session.run.return_value = [dummy_logits]

        runner.session = mock_session
        runner.registered_providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
        runner.active_providers = runner.registered_providers
        runner.session_type = "unverified_qnn"

        out_qnn = runner.generate(
            "Test prompt", max_tokens=2, use_chat_template=False, hardware_run_evidence=fabricated_evidence
        )
        self.assertEqual(out_qnn["hardware_status"], "UNVERIFIED")
        self.assertEqual(out_qnn["session_type"], "unverified_qnn")
        self.assertIn("UNKNOWN", out_qnn["executed_placement"])
        self.assertIn("UNKNOWN", str(out_qnn["fallback_detected"]))
        self.assertEqual(out_qnn["unvalidated_hardware_evidence"], fabricated_evidence)
        runner.close()

    def test_parity_checks_active_under_normal_and_optimized_python(self):
        """Verify parity checks use explicit ParityVerificationError active under python -O."""
        res = export_smollm2_to_onnx()

        # 1. Deliberate failure under normal Python (unrealistically tight tolerance)
        with self.assertRaises(ParityVerificationError) as ctx:
            verify_onnx_cpu_numerical_parity(
                res.onnx_file_path,
                test_input_ids=[1, 2, 3, 4],
                max_tolerance=1.0e-7,
            )
        self.assertIn("exceeded tolerance", str(ctx.exception))

        # 2. Deliberate failure under python -O (assert would be suppressed, ParityVerificationError must NOT be)
        cmd = [
            sys.executable,
            "-O",
            "-c",
            (
                "import sys\n"
                "from qedge_runtime.qualcomm.export import verify_onnx_cpu_numerical_parity\n"
                "from qedge_runtime.qualcomm.provenance import ParityVerificationError\n"
                "try:\n"
                f"    verify_onnx_cpu_numerical_parity(r'{res.onnx_file_path}', max_tolerance=1.0e-7)\n"
                "    sys.exit(0)\n"
                "except ParityVerificationError:\n"
                "    sys.exit(42)\n"
                "except Exception:\n"
                "    sys.exit(1)\n"
            ),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            proc.returncode,
            42,
            f"Expected exit code 42 (ParityVerificationError active under -O), got {proc.returncode}. Stderr: {proc.stderr}",
        )

        # 3. Autoregressive check rejects empty prompts
        with self.assertRaises(ParityVerificationError) as ctx:
            verify_onnx_cpu_autoregressive_parity(res.onnx_file_path, test_prompts=[])
        self.assertIn("empty", str(ctx.exception).lower())

    def test_onnx_cpu_single_forward_pass_parity(self):
        """Verify accurately scoped single forward pass per-position argmax comparison for [1, 2, 3, 4]."""
        res = export_smollm2_to_onnx()
        parity = verify_onnx_cpu_numerical_parity(res.onnx_file_path, test_input_ids=[1, 2, 3, 4])

        self.assertTrue(parity["verified"])
        self.assertEqual(parity["verification_scope"], "single_forward_pass_per_position_argmax")
        self.assertEqual(parity["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(parity["numerical_tolerance_status"], "PASS")
        self.assertLessEqual(parity["max_absolute_difference"], 1.0e-3)
        self.assertLessEqual(parity["mean_absolute_difference"], 2.0e-4)
        self.assertTrue(parity["greedy_token_parity"])
        self.assertEqual(parity["pytorch_greedy_tokens"], parity["onnx_greedy_tokens"])

    def test_onnx_cpu_multi_prompt_autoregressive_parity(self):
        """Verify multi-prompt, multi-step autoregressive generation parity between PyTorch CPU and ONNX CPU."""
        res = export_smollm2_to_onnx()
        prompts = [
            "Hello, my name is",
            "The capital of France is",
        ]
        parity = verify_onnx_cpu_autoregressive_parity(
            res.onnx_file_path,
            test_prompts=prompts,
            max_new_tokens=4,
        )

        self.assertTrue(parity["verified"])
        self.assertEqual(parity["verification_scope"], "multi_prompt_multi_step_autoregressive_generation")
        self.assertEqual(parity["numerical_tolerance_status"], "PASS")
        self.assertEqual(parity["num_prompts_evaluated"], 2)
        self.assertEqual(parity["token_agreement_rate"], 1.0)
        self.assertLessEqual(parity["overall_max_absolute_difference"], 1.0e-3)
        self.assertLessEqual(parity["overall_average_mean_absolute_difference"], 2.0e-4)

        for p_eval in parity["prompt_evaluations"]:
            self.assertTrue(p_eval["tokens_match_all"])
            self.assertGreater(p_eval["steps_evaluated"], 0)

    def test_device_runbook_standards_and_provisional_labeling(self):
        """Verify device runbook exists, is marked provisional, and details pre-staging preparation requirements."""
        runbook_path = Path(__file__).resolve().parent.parent / "src" / "qedge_runtime" / "qualcomm" / "device_runbook.md"
        self.assertTrue(runbook_path.is_file(), f"Runbook not found at {runbook_path}")
        text = runbook_path.read_text(encoding="utf-8")

        self.assertIn("PROVISIONAL RUNBOOK TEMPLATE", text)
        self.assertIn("Do NOT Stage Dynamic CPU Graph", text)
        self.assertIn("Shape Specialization", text)
        self.assertIn("Stateful / I/O KV Cache Interface", text)
        self.assertIn("libQnnHtp.so", text)
        self.assertIn("soc_model:0", text)
        self.assertIn("VTCM SRAM Residency Verification", text)
        self.assertIn("NPU Status:** `UNVERIFIED`", text)


if __name__ == "__main__":
    unittest.main()
