"""Adversarial semantic tests for the v2 contracts and v1 compatibility."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from jsonschema import Draft202012Validator

from qedge_common import cli
from qedge_common.records import fixture_smoke_benchmark
from qedge_common.schema import (
    ContractValidationError,
    example_path,
    load_json,
    load_schema,
    schema_versions,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]


def load_example(name: str, version: str = "2") -> dict:
    return json.loads(example_path(name, version).read_text(encoding="utf-8"))


class ContractSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decision = load_example("decision")
        self.claim = load_example("evidence_claim")
        self.benchmark = load_example("benchmark_result")
        self.hardware = load_example("hardware_status")

    def assert_invalid(self, name: str, payload: dict, fragment: str | None = None) -> None:
        with self.assertRaises(ContractValidationError) as caught:
            validate(name, payload)
        if fragment is not None:
            self.assertIn(fragment, str(caught.exception))

    # Historical reader -------------------------------------------------

    def test_v1_examples_remain_valid(self) -> None:
        for name in ("product", "retrieval_request", "retrieval_result", "evidence_claim", "decision", "benchmark_result", "hardware_status"):
            with self.subTest(name=name):
                validate(name, load_example(name, "1"), version="1")

    def test_v1_and_v2_versions_are_explicit(self) -> None:
        self.assertEqual(schema_versions("decision"), ("1", "2"))
        self.assertEqual(schema_versions("product"), ("1",))
        with self.assertRaises(Exception):
            load_schema("decision", "9")

    # Decision semantics ------------------------------------------------

    def test_v2_positive_example_is_valid(self) -> None:
        validate("decision", self.decision)

    def test_empty_supported_decision_is_rejected(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"] = []
        self.assert_invalid("decision", payload, "UNKNOWN")

    def test_empty_supported_decision_is_structurally_valid(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"] = []
        structural_errors = list(
            Draft202012Validator(load_schema("decision", "2")).iter_errors(payload)
        )
        self.assertEqual(structural_errors, [])
        self.assert_invalid("decision", payload, "UNKNOWN")

    def test_contradicted_child_blocks_positive_parent(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"][0]["observed_value"] = 75.0
        payload["claims"][0]["status"] = "CONTRADICTED"
        payload["claims"][0]["evidence_text"] = "width: 75.0 cm"
        self.assert_invalid("decision", payload, "CONTRADICTED")

    def test_contradicted_child_with_contradicted_parent_is_valid(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"][0]["observed_value"] = 75.0
        payload["claims"][0]["status"] = "CONTRADICTED"
        payload["claims"][0]["evidence_text"] = "width: 75.0 cm"
        payload["status"] = "CONTRADICTED"
        payload["action"] = "DO_NOT_RECOMMEND"
        validate("decision", payload)

    def test_missing_mandatory_coverage_is_rejected(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"] = [payload["claims"][0]]
        self.assert_invalid("decision", payload, "UNKNOWN")

    def test_missing_coverage_can_be_explicitly_unknown(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"] = [payload["claims"][0]]
        payload["status"] = "UNKNOWN"
        payload["action"] = "ABSTAIN"
        validate("decision", payload)

    def test_mismatched_child_item_id_is_rejected(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"][0]["item_id"] = "other-item"
        self.assert_invalid("decision", payload, "must match decision item_id")

    def test_duplicate_requirement_and_claim_ids_are_rejected(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["requirements"].append(copy.deepcopy(payload["requirements"][0]))
        self.assert_invalid("decision", payload, "duplicate requirement_id")

        payload = copy.deepcopy(self.decision)
        payload["claims"].append(copy.deepcopy(payload["claims"][0]))
        self.assert_invalid("decision", payload, "duplicate claim_id")

    def test_claim_must_match_its_requirement(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"][0]["required_value"] = 61.0
        self.assert_invalid("decision", payload, "does not match requirement")

    def test_unparsed_and_ambiguity_block_positive_response(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["unparsed_clauses"] = ["not oak"]
        self.assert_invalid("decision", payload, "ABSTAIN")

        payload = copy.deepcopy(self.decision)
        payload["ambiguities"] = ["under 60 could mean width or height"]
        self.assert_invalid("decision", payload, "ABSTAIN")

    def test_conflicting_evidence_abstains(self) -> None:
        payload = copy.deepcopy(self.decision)
        contradiction = copy.deepcopy(payload["claims"][0])
        contradiction.update(
            claim_id="claim-v2-width-conflict",
            observed_value=75.0,
            status="CONTRADICTED",
            evidence_text="width: 75.0 cm",
            source_revision="fixture-v2-conflict",
        )
        payload["claims"].append(contradiction)
        payload["status"] = "ABSTAIN"
        payload["action"] = "ABSTAIN"
        validate("decision", payload)

        payload["status"] = "SUPPORTED"
        payload["action"] = "RECOMMEND"
        self.assert_invalid("decision", payload, "ABSTAIN")

    def test_no_candidate_response_is_explicit(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "qedge-common"
                / "tests"
                / "fixtures"
                / "valid"
                / "v2"
                / "decision_no_candidate.json"
            ).read_text()
        )
        validate("decision", payload)
        self.assertIsNone(payload["item_id"])
        self.assertEqual(payload["claims"], [])

    def test_no_candidate_cannot_invent_an_item_id_or_claims(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "qedge-common"
                / "tests"
                / "fixtures"
                / "valid"
                / "v2"
                / "decision_no_candidate.json"
            ).read_text()
        )
        payload["item_id"] = "invented-no-result"
        self.assert_invalid("decision", payload)

        payload = json.loads(
            (
                ROOT
                / "qedge-common"
                / "tests"
                / "fixtures"
                / "valid"
                / "v2"
                / "decision_no_candidate.json"
            ).read_text()
        )
        payload["claims"] = [copy.deepcopy(self.decision["claims"][0])]
        self.assert_invalid("decision", payload)

    # Dates and numbers -------------------------------------------------

    def test_invalid_timestamps_are_rejected(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["created_at"] = "not-a-date"
        self.assert_invalid("decision", payload, "date-time")

        payload = copy.deepcopy(self.decision)
        payload["claims"][0]["retrieved_at"] = "2026-10-27T00:00:00"
        self.assert_invalid("decision", payload, "date-time")

        payload = copy.deepcopy(self.decision)
        payload["created_at"] = "2026-13-01T00:00:00Z"
        self.assert_invalid("decision", payload, "date-time")

    def test_null_supported_observations_are_rejected(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"][0]["observed_value"] = None
        self.assert_invalid("decision", payload)

        payload = copy.deepcopy(self.claim)
        payload["observed_value"] = None
        self.assert_invalid("evidence_claim", payload)

    def test_nonfinite_python_values_are_rejected(self) -> None:
        payload = copy.deepcopy(self.decision)
        payload["claims"][0]["observed_value"] = math.nan
        self.assert_invalid("decision", payload, "non-finite")

        payload = copy.deepcopy(self.benchmark)
        payload["metrics"]["ratio"] = math.inf
        self.assert_invalid("benchmark_result", payload, "non-finite")

    def test_strict_json_loader_rejects_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            path.write_text('{"schema_version": "2", "metrics": {"x": NaN}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_json(path)
            path.write_text('{"schema_version": "2", "metrics": {"x": 1e309}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_json(path)

    def test_strict_json_loader_accepts_valid_example(self) -> None:
        self.assertEqual(load_json(example_path("decision", "2"))["schema_version"], "2")

    # Predicate semantics ------------------------------------------------

    def test_strict_and_inclusive_comparisons(self) -> None:
        payload = copy.deepcopy(self.claim)
        payload["operator"] = "<"
        payload["observed_value"] = 60.0
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "does not match predicate")

        payload = copy.deepcopy(self.claim)
        payload["operator"] = "<"
        payload["observed_value"] = 60.0
        payload["status"] = "CONTRADICTED"
        validate("evidence_claim", payload)

        payload = copy.deepcopy(self.claim)
        payload["operator"] = "<="
        payload["observed_value"] = 60.0
        payload["status"] = "SUPPORTED"
        validate("evidence_claim", payload)

    def test_equality_inequality_and_containment(self) -> None:
        payload = copy.deepcopy(self.claim)
        payload.update(
            attribute="material",
            operator="!=",
            required_value="oak",
            required_unit=None,
            observed_value="aluminum",
            observed_unit=None,
            status="SUPPORTED",
            evidence_text="material: aluminum",
        )
        validate("evidence_claim", payload)
        payload["observed_value"] = "oak"
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "does not match predicate")

        payload = copy.deepcopy(self.claim)
        payload.update(
            attribute="material",
            operator="contains",
            required_value="oak",
            required_unit=None,
            observed_value="Solid OAK seat",
            observed_unit=None,
            status="SUPPORTED",
            evidence_text="material: solid OAK seat",
        )
        validate("evidence_claim", payload)
        payload["operator"] = "not_contains"
        self.assert_invalid("evidence_claim", payload, "does not match predicate")

        # Material taxonomy derivation: oak satisfies wood requirement under ==
        payload = copy.deepcopy(self.claim)
        payload.update(
            attribute="material",
            operator="==",
            required_value="wood",
            required_unit=None,
            observed_value="oak",
            observed_unit=None,
            status="SUPPORTED",
            evidence_text="material: oak (source: material); taxonomy: oak satisfies wood",
        )
        validate("evidence_claim", payload)

        # Brand equality requires complete normalized identity; substring is rejected
        payload = copy.deepcopy(self.claim)
        payload.update(
            attribute="brand",
            operator="==",
            required_value="Fixture",
            required_unit=None,
            observed_value="FixtureCo",
            observed_unit=None,
            status="SUPPORTED",
            evidence_text="brand: FixtureCo",
        )
        self.assert_invalid("evidence_claim", payload, "does not match predicate")

        # Brand inequality: Fixture != FixtureCo is SUPPORTED
        payload["operator"] = "!="
        validate("evidence_claim", payload)

    def test_semantic_boundary_color_material_and_brand(self) -> None:
        # 1. Brand whitespace and case normalization
        payload = copy.deepcopy(self.claim)
        payload.update(
            attribute="brand",
            operator="==",
            required_value="FixtureCo",
            required_unit=None,
            observed_value="  FixtureCo  ",
            observed_unit=None,
            status="SUPPORTED",
            evidence_text="brand:  FixtureCo  ",
        )
        validate("evidence_claim", payload)

        # Distinct brands on == must not be SUPPORTED
        payload["observed_value"] = "OtherBrand"
        self.assert_invalid("evidence_claim", payload, "does not match predicate")
        payload["status"] = "CONTRADICTED"
        validate("evidence_claim", payload)

        # 2. Color derivation: compound shade observation satisfies base requirement
        payload = copy.deepcopy(self.claim)
        payload.update(
            attribute="color",
            operator="==",
            required_value="red",
            required_unit=None,
            observed_value="dark red",
            observed_unit=None,
            status="SUPPORTED",
            evidence_text="color: dark red",
        )
        validate("evidence_claim", payload)

        # Negative predicate: dark red CONTRADICTS 'not red'
        payload["operator"] = "!="
        payload["status"] = "CONTRADICTED"
        validate("evidence_claim", payload)

        # Disjoint colors: blue contradicts red on ==, satisfies !=
        payload["observed_value"] = "blue"
        payload["operator"] = "=="
        payload["status"] = "CONTRADICTED"
        validate("evidence_claim", payload)
        payload["operator"] = "!="
        payload["status"] = "SUPPORTED"
        validate("evidence_claim", payload)

        # Asymmetric base observation cannot prove specific compound requirement
        payload["required_value"] = "dark red"
        payload["observed_value"] = "red"
        payload["operator"] = "=="
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "not evaluable")

        # Bounded affirmative color grammar: non-affirmative descriptions cannot certify color
        # Negated color observation
        payload["required_value"] = "red"
        payload["observed_value"] = "not red"
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "not evaluable")
        payload["operator"] = "!="
        self.assert_invalid("evidence_claim", payload, "not evaluable")
        payload["status"] = "UNKNOWN"
        payload["observed_value"] = None
        validate("evidence_claim", payload)

        # Alternative color observation
        payload["observed_value"] = "red or blue"
        payload["operator"] = "=="
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "not evaluable")
        payload["operator"] = "!="
        self.assert_invalid("evidence_claim", payload, "not evaluable")
        payload["status"] = "UNKNOWN"
        payload["observed_value"] = None
        validate("evidence_claim", payload)

        # Uncertain color observation
        payload["observed_value"] = "maybe red"
        payload["operator"] = "=="
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "not evaluable")

        # Unrecognized color modifier
        payload["observed_value"] = "sparkly red"
        payload["operator"] = "=="
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "not evaluable")

        # 3. Numeric exact equality without implicit tolerance
        payload_num = copy.deepcopy(self.claim)
        payload_num.update(
            attribute="width",
            operator="==",
            required_value=48.0,
            required_unit="cm",
            observed_value=48.0,
            observed_unit="cm",
            status="SUPPORTED",
            evidence_text="width: 48.0 cm",
        )
        validate("evidence_claim", payload_num)

        # Value just inside former tolerance (diff = 5e-7 < 1e-6): must NOT match on ==
        payload_num["observed_value"] = 48.0000005
        self.assert_invalid("evidence_claim", payload_num, "does not match predicate")
        payload_num["status"] = "CONTRADICTED"
        validate("evidence_claim", payload_num)
        payload_num["operator"] = "!="
        payload_num["status"] = "SUPPORTED"
        validate("evidence_claim", payload_num)

        # Value just outside former tolerance (diff = 2e-6 > 1e-6): must NOT match on ==
        payload_num["observed_value"] = 48.000002
        payload_num["operator"] = "=="
        self.assert_invalid("evidence_claim", payload_num, "does not match predicate")
        payload_num["status"] = "CONTRADICTED"
        validate("evidence_claim", payload_num)

        # Unit conversion exact equality (24 in == 60.96 cm)
        payload_num["required_value"] = 60.96
        payload_num["required_unit"] = "cm"
        payload_num["observed_value"] = 60.96
        payload_num["observed_unit"] = "cm"
        payload_num["operator"] = "=="
        payload_num["status"] = "SUPPORTED"
        validate("evidence_claim", payload_num)

        # 4. Material: unrecognized composition ('oak veneer') cannot establish solid oak
        payload = copy.deepcopy(self.claim)
        payload.update(
            attribute="material",
            operator="==",
            required_value="oak",
            required_unit=None,
            observed_value="oak veneer",
            observed_unit=None,
            status="SUPPORTED",
            evidence_text="material: oak veneer",
        )
        self.assert_invalid("evidence_claim", payload, "not evaluable")
        # Nor can it establish an unsupported negation
        payload["operator"] = "!="
        self.assert_invalid("evidence_claim", payload, "not evaluable")

        # But explicitly UNKNOWN claim is valid
        payload["status"] = "UNKNOWN"
        payload["observed_value"] = None
        validate("evidence_claim", payload)

    def test_incompatible_types_and_units_are_rejected(self) -> None:
        payload = copy.deepcopy(self.claim)
        payload["observed_unit"] = "mm"
        self.assert_invalid("evidence_claim", payload, "not evaluable")

        payload = copy.deepcopy(self.claim)
        payload["operator"] = "=="
        payload["required_value"] = "oak"
        payload["required_unit"] = "cm"
        payload["observed_value"] = "oak"
        payload["observed_unit"] = "cm"
        payload["status"] = "SUPPORTED"
        self.assert_invalid("evidence_claim", payload, "not evaluable")

    def test_status_must_agree_with_observed_value(self) -> None:
        payload = copy.deepcopy(self.claim)
        payload["observed_value"] = 75.0
        self.assert_invalid("evidence_claim", payload, "does not match predicate")

    # Measurement invariants -------------------------------------------

    def test_synthetic_benchmark_cannot_be_measured(self) -> None:
        payload = copy.deepcopy(self.benchmark)
        payload["measured"] = True
        self.assert_invalid("benchmark_result", payload, "synthetic")

    def test_measured_benchmark_needs_evidence_and_model(self) -> None:
        payload = copy.deepcopy(self.benchmark)
        payload["kind"] = "runtime-generation"
        payload["measured"] = True
        self.assert_invalid("benchmark_result", payload, "model")
        self.assert_invalid("benchmark_result", payload, "log_path")

        payload["model"] = {
            "name": "fixture-model",
            "revision": "r1",
            "quantization": "int8",
        }
        payload["provenance"]["log_path"] = "logs/fixture.log"
        validate("benchmark_result", payload)

    def test_measured_npu_requires_execution_identity(self) -> None:
        payload = copy.deepcopy(self.benchmark)
        payload.update(
            kind="runtime-generation",
            measured=True,
            model={"name": "fixture-model", "revision": "r1", "quantization": "int8"},
            backend="qnn-htp",
        )
        payload["hardware"] = {
            "cpu": "fixture-host",
            "device": "Snapdragon fixture",
            "npu_status": "measured",
            "execution_evidence": "logs/fixture-npu.log",
            "execution_backend": "qnn-htp",
        }
        payload["provenance"]["log_path"] = "logs/fixture-npu.log"
        validate("benchmark_result", payload)

        missing_evidence = copy.deepcopy(payload)
        del missing_evidence["hardware"]["execution_evidence"]
        self.assert_invalid("benchmark_result", missing_evidence, "execution evidence")

        missing_device = copy.deepcopy(payload)
        del missing_device["hardware"]["device"]
        self.assert_invalid("benchmark_result", missing_device, "device identity")

        mismatched_backend = copy.deepcopy(payload)
        mismatched_backend["backend"] = "cpu"
        self.assert_invalid("benchmark_result", mismatched_backend, "match top-level backend")

        cpu_backend = copy.deepcopy(payload)
        cpu_backend["backend"] = "cpu"
        cpu_backend["hardware"]["execution_backend"] = "cpu"
        self.assert_invalid("benchmark_result", cpu_backend, "CPU")

        unrelated_backend = copy.deepcopy(payload)
        unrelated_backend["backend"] = "fixture-backend"
        unrelated_backend["hardware"]["execution_backend"] = "fixture-backend"
        self.assert_invalid("benchmark_result", unrelated_backend, "QNN/NPU-style")

    def test_cache_hit_rate_is_optional(self) -> None:
        payload = copy.deepcopy(self.benchmark)
        payload["cache"] = {"policy": "keydiff", "capacity_tokens": 128}
        validate("benchmark_result", payload)

    def test_hardware_contradictions_are_rejected(self) -> None:
        payload = copy.deepcopy(self.hardware)
        payload["npu_status"] = "measured"
        self.assert_invalid("hardware_status", payload, "measured NPU")

        payload = copy.deepcopy(self.hardware)
        payload["metal"]["used"] = True
        self.assert_invalid("hardware_status", payload, "used=true")

        payload = copy.deepcopy(self.hardware)
        payload["snapdragon"]["model"] = "Snapdragon fixture"
        self.assert_invalid("hardware_status", payload, "absent Snapdragon")

        payload = copy.deepcopy(self.hardware)
        payload["qnn"]["device_verified"] = True
        self.assert_invalid("hardware_status", payload, "device_verified")

        payload = copy.deepcopy(self.hardware)
        payload["npu_status"] = "not-present"
        payload["snapdragon"]["available"] = True
        payload["snapdragon"]["model"] = "Snapdragon fixture"
        self.assert_invalid("hardware_status", payload, "not-present")

    def test_measured_npu_hardware_requires_device_identity_and_evidence(self) -> None:
        payload = copy.deepcopy(self.hardware)
        payload.update(
            npu_status="measured",
            snapdragon={"available": True, "model": "Snapdragon fixture"},
            qnn={
                "sdk_available": True,
                "device_verified": True,
                "device": "Snapdragon fixture",
                "execution_evidence": "logs/fixture-npu.log",
                "execution_backend": "qnn-htp",
            },
        )
        validate("hardware_status", payload)

        missing_evidence = copy.deepcopy(payload)
        del missing_evidence["qnn"]["execution_evidence"]
        self.assert_invalid("hardware_status", missing_evidence, "execution evidence")

        missing_device = copy.deepcopy(payload)
        del missing_device["qnn"]["device"]
        self.assert_invalid("hardware_status", missing_device, "qnn.device")

        mismatched_device = copy.deepcopy(payload)
        mismatched_device["qnn"]["device"] = "Different Snapdragon"
        self.assert_invalid("hardware_status", mismatched_device, "must match snapdragon.model")

    def test_measured_npu_hardware_cannot_use_cpu_backend(self) -> None:
        payload = copy.deepcopy(self.hardware)
        payload.update(
            npu_status="measured",
            snapdragon={"available": True, "model": "Snapdragon fixture"},
            qnn={
                "sdk_available": True,
                "device_verified": True,
                "device": "Snapdragon fixture",
                "execution_evidence": "logs/fixture-npu.log",
                "execution_backend": "cpu",
            },
        )
        self.assert_invalid("hardware_status", payload, "CPU")

        unrelated_backend = copy.deepcopy(payload)
        unrelated_backend["qnn"]["execution_backend"] = "fixture-backend"
        self.assert_invalid("hardware_status", unrelated_backend, "QNN/NPU-style")

    # Producer/consumer boundaries -------------------------------------

    def test_records_producer_calls_validator(self) -> None:
        record = fixture_smoke_benchmark(run_id="producer-test", iterations=1)
        self.assertEqual(record["schema_version"], "2")
        self.assertFalse(record["measured"])
        validate("benchmark_result", record)

    def test_cli_uses_semantic_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.json"
            invalid_path = Path(directory) / "invalid.json"
            valid_payload = copy.deepcopy(self.decision)
            valid_path.write_text(json.dumps(valid_payload), encoding="utf-8")
            invalid_payload = copy.deepcopy(self.decision)
            invalid_payload["claims"] = []
            invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["decision", str(valid_path)]), 0)
                self.assertEqual(cli.main(["decision", str(invalid_path)]), 1)
                self.assertEqual(cli.main(["decision", str(valid_path), "--schema-version", "1"]), 1)

    def test_edge_bench_producer_validates_when_qedge_common_is_available(self) -> None:
        edge_src = ROOT / "edge-rag-demo" / "src"
        fixtures = ROOT / "edge-rag-demo" / "tests" / "fixtures"
        if not (edge_src / "edge_rag" / "bench.py").is_file():
            self.skipTest("edge-rag-demo checkout is not present")
        sys.path.insert(0, str(edge_src))
        try:
            from edge_rag.bench import run_smoke

            with mock.patch.dict(os.environ, {"QEDGE_CATALOG_DIR": str(fixtures)}):
                record = run_smoke(iterations=1)
            self.assertEqual(record["schema_version"], "2")
            self.assertFalse(record["measured"])
            validate("benchmark_result", record)
        finally:
            sys.path.remove(str(edge_src))


if __name__ == "__main__":
    unittest.main()
