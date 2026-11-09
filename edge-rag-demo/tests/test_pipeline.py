"""Portable unit tests for the QEdge Assist pipeline.

The default tests use a hand-written fixture catalog. They never import another
checkout, download a model, touch the network, or require a Snapdragon device.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from edge_rag.cli import format_cli_output, run_pipeline
from edge_rag.generator import Generator
from edge_rag.paths import resolve_catalog_file
from edge_rag.profiler import PipelineProfiler
from edge_rag.retriever import Retriever
from edge_rag.verifier import Verifier

EDGE_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"
SCHEMA_ROOT = EDGE_ROOT.parent / "qedge-common" / "schemas"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(catalog_path=FIXTURE_CATALOG)
        cls.verifier = Verifier(catalog_path=FIXTURE_CATALOG)
        cls.generator = Generator()

    def test_pipeline_returns_fixture_results(self):
        result = run_pipeline(
            "wooden dining chair under 80cm",
            retriever=self.retriever,
            verifier=self.verifier,
            generator=self.generator,
            k=5,
        )
        self.assertGreater(len(result["candidates"]), 0)
        self.assertGreater(len(result["verified_results"]), 0)
        self.assertIn("Abstain for", result["generated_response"])
        self.assertNotIn("Stage 01 template only", result["generated_response"])
        self.assertNotIn("matches your requirements", result["generated_response"])

    def test_pipeline_returns_empty_for_unmatched_query(self):
        result = run_pipeline(
            "quantum flux capacitor",
            retriever=self.retriever,
            verifier=self.verifier,
            generator=self.generator,
        )
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["verified_results"], [])
        self.assertIn("No catalog items", result["generated_response"])

    def test_verify_supported_when_attribute_matches(self):
        self.verifier.store.upload(
            "test_doc_supported",
            "system",
            ["width: 48.0 cm (source: item_dimensions)", "material: oak (source: material)"],
        )
        status_width, detail_width = self.verifier.verify_attribute(
            "test_doc_supported", "width", 60.0
        )
        self.assertEqual(status_width, "SUPPORTED")
        self.assertEqual(detail_width, "48cm")

        status_material, detail_material = self.verifier.verify_attribute(
            "test_doc_supported", "material", "oak"
        )
        self.assertEqual(status_material, "SUPPORTED")
        self.assertEqual(detail_material, "oak")

    def test_verify_contradicted_when_attribute_outside_range(self):
        self.verifier.store.upload(
            "test_doc_contradicted",
            "system",
            ["width: 75.0 cm (source: item_dimensions)", "material: aluminum (source: material)"],
        )
        status_width, detail_width = self.verifier.verify_attribute(
            "test_doc_contradicted", "width", 60.0
        )
        self.assertEqual(status_width, "CONTRADICTED")
        self.assertEqual(detail_width, "75cm")

        status_material, detail_material = self.verifier.verify_attribute(
            "test_doc_contradicted", "material", "oak"
        )
        self.assertEqual(status_material, "CONTRADICTED")
        self.assertEqual(detail_material, "aluminum")

    def test_verify_unknown_when_attribute_missing(self):
        self.verifier.store.upload(
            "test_doc_missing",
            "system",
            ["title: Minimalist Stool (source: catalog)"],
        )
        status_width, detail_width = self.verifier.verify_attribute(
            "test_doc_missing", "width", 60.0
        )
        self.assertEqual(status_width, "UNKNOWN")
        self.assertIsNone(detail_width)

        status_material, detail_material = self.verifier.verify_attribute(
            "test_doc_missing", "material", "oak"
        )
        self.assertEqual(status_material, "UNKNOWN")
        self.assertIsNone(detail_material)

    def test_profiler_records_all_stages(self):
        profiler = PipelineProfiler()
        result = run_pipeline(
            "dining chair under 60cm",
            retriever=self.retriever,
            verifier=self.verifier,
            generator=self.generator,
            profiler=profiler,
        )
        summary = result["profiler_summary"]
        for stage in ("RETRIEVE", "VERIFY", "GENERATE"):
            self.assertIn(stage, summary["stages"])
            self.assertGreater(summary["stages"][stage]["time_ms"], 0.0)
        self.assertGreater(summary["total_ms"], 0.0)

    def test_no_npu_offload_or_snapdragon_projection_is_printed(self):
        result = run_pipeline(
            "dining chair under 60cm",
            retriever=self.retriever,
            verifier=self.verifier,
            generator=self.generator,
        )
        summary = result["profiler_summary"]
        self.assertIsNone(summary["npu_offloadable_pct"])
        self.assertIsNone(summary["est_npu_total_ms"])
        self.assertEqual(summary["npu_status"], "unverified")

        rendered = format_cli_output(result)
        self.assertIn("NPU status: unverified", rendered)
        self.assertNotIn("100%", rendered)
        self.assertNotIn("< 3 ms", rendered)
        self.assertNotIn("Snapdragon 8 Elite (45 TOPS)", rendered)

    def test_footer_has_no_hardcoded_energy_or_hit_rate(self):
        footer = self.generator.format_footer()
        self.assertNotIn("6368", footer)
        self.assertNotIn("79%", footer)
        self.assertIn("not measured", footer)
        self.assertIn("QNN/NPU status: unverified", footer)

    def test_sqlite_evidence_store_is_local(self):
        self.assertEqual(self.verifier.store.backend, "sqlite")
        self.assertIn("qedge_mem_", self.verifier.store._conn_str)

    def test_product_fixture_uses_shared_contract_field_names(self):
        schema_path = SCHEMA_ROOT / "product.schema.json"
        if not schema_path.is_file():
            self.skipTest("qedge-common schemas are not present in this checkout")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        row = json.loads(FIXTURE_CATALOG.read_text(encoding="utf-8").splitlines()[0])
        for field in schema["required"]:
            self.assertIn(field, row)

    def test_missing_catalog_fails_with_an_actionable_error(self):
        with self.assertRaises(Exception) as caught:
            Retriever(catalog_path="/definitely/not/a/catalog")
        self.assertIn("No catalog found", str(caught.exception))

    def test_bench_smoke_emits_validated_contract_record(self):
        from edge_rag.bench import run_smoke
        from unittest import mock

        fixtures_dir = EDGE_ROOT / "tests" / "fixtures"
        with mock.patch.dict(os.environ, {"QEDGE_CATALOG_DIR": str(fixtures_dir)}):
            record = run_smoke(iterations=1)
        self.assertEqual(record["schema_version"], "2")
        self.assertFalse(record["measured"])
        self.assertIn("outputs", record)
        self.assertNotIn("hit_rate", record.get("metrics", {}))

    def test_bench_fails_clearly_when_validator_unavailable(self):
        from edge_rag.bench import run_smoke
        from unittest import mock

        fixtures_dir = EDGE_ROOT / "tests" / "fixtures"
        with mock.patch.dict(os.environ, {"QEDGE_CATALOG_DIR": str(fixtures_dir)}):
            with mock.patch.dict("sys.modules", {"qedge_common.schema": None}):
                with self.assertRaises(RuntimeError) as caught:
                    run_smoke(iterations=1)
                self.assertIn("qedge-common is required", str(caught.exception))


@unittest.skipUnless(
    os.environ.get("QEDGE_RUN_CATALOG_INTEGRATION") == "1",
    "set QEDGE_RUN_CATALOG_INTEGRATION=1 with QEDGE_CATALOG_DIR to test the licensed catalog",
)
class RealCatalogIntegrationTests(unittest.TestCase):
    def test_catalog_loads(self):
        path = resolve_catalog_file()
        retriever = Retriever(catalog_path=path)
        self.assertGreater(len(retriever.rows), 0)


if __name__ == "__main__":
    unittest.main()
