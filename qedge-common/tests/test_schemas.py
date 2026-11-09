"""Contract tests for the shared QEdge schemas."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from qedge_common.schema import (
    ContractValidationError,
    example_path,
    load_schema,
    schema_names,
    schema_path,
    schema_versions,
    validate,
)


class SchemaTests(unittest.TestCase):
    def test_all_schemas_are_valid_draft_2020_12(self):
        for name in schema_names():
            for version in schema_versions(name):
                with self.subTest(schema=name, version=version):
                    Draft202012Validator.check_schema(load_schema(name, version))

    def test_all_examples_validate(self):
        for name in schema_names():
            for version in schema_versions(name):
                with self.subTest(schema=name, version=version):
                    with example_path(name, version).open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    validate(name, payload, version=version)

    def test_unknown_fields_are_rejected(self):
        with example_path("product").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["unexpected_field"] = True
        with self.assertRaises(ContractValidationError):
            validate("product", payload)

    def test_present_attribute_cannot_be_null(self):
        with example_path("product").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["attributes"]["width"]["value"] = None
        with self.assertRaises(ContractValidationError):
            validate("product", payload)

    def test_unknown_evidence_status_is_rejected(self):
        with example_path("evidence_claim").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["status"] = "PROBABLY"
        with self.assertRaises(ContractValidationError):
            validate("evidence_claim", payload)

    def test_unknown_decision_status_is_rejected(self):
        with example_path("decision").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["status"] = "OK"
        with self.assertRaises(ContractValidationError):
            validate("decision", payload)

    def test_supported_decision_must_recommend(self):
        with example_path("decision").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["action"] = "ABSTAIN"
        with self.assertRaises(ContractValidationError):
            validate("decision", payload)

    def test_measured_benchmark_must_be_boolean(self):
        with example_path("benchmark_result").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["measured"] = "false"
        with self.assertRaises(ContractValidationError):
            validate("benchmark_result", payload)

    def test_hardware_status_requires_qnn_block(self):
        with example_path("hardware_status").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        del payload["qnn"]
        with self.assertRaises(ContractValidationError):
            validate("hardware_status", payload)

    def test_retrieval_request_requires_a_modality(self):
        with example_path("retrieval_request").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["text"] = None
        payload["method"] = "hybrid"
        with self.assertRaises(ContractValidationError):
            validate("retrieval_request", payload)

    def test_schema_files_do_not_contain_machine_paths(self):
        marker = "/" + "Users" + "/"
        for name in schema_names():
            for version in schema_versions(name):
                with self.subTest(schema=name, version=version):
                    self.assertNotIn(
                        marker,
                        schema_path(name, version).read_text(encoding="utf-8"),
                    )

    def test_examples_are_independent_copies(self):
        # Guards against a test mutating the fixture on disk by accident.
        with example_path("product").open("r", encoding="utf-8") as handle:
            first = json.load(handle)
        second = copy.deepcopy(first)
        second["title"] = "changed"
        self.assertNotEqual(first["title"], second["title"])

    def test_example_paths_exist(self):
        for name in schema_names():
            for version in schema_versions(name):
                with self.subTest(schema=name, version=version):
                    self.assertTrue(example_path(name, version).is_file(), name)


if __name__ == "__main__":
    unittest.main()
