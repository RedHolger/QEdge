"""Tests for Stage 09 provenance-governed catalog retrieval and evaluation.

Covers:
- BM25 baseline determinism and zero-overlap handling.
- Feature flag enforcement (enable_neural=False defaults to BM25 baseline).
- Persisted index key verification and stale/tampered index invalidation.
- Deterministic weighted RRF with 1-based ranks and stable tie-breaking.
- Product family deduplication.
- Verifier separation: high retrieval score != factual certification.
- Evaluation metrics hand-checks, coverage calculation, and bootstrap CI determinism.
- Pilot evaluation execution and blocked quality gate on 120-query evaluation plan.
- Schema compliance with retrieval_result.schema.json.
- Independent failure probes:
  - Neural index integrity bypass closed (files: {}, invented IDs, reordered IDs, unnormalized vectors).
  - Schema validation failure propagation.
  - Evaluation gate input validation (empty labels, empty runs, insufficient depth).
  - Explicit judgment coverage (30/685, 20/30 positive, 10 without positive, 655 in review queue).
"""
from __future__ import annotations

import copy
import os
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from edge_rag.eval import (
    EvidenceValidationError,
    bootstrap_ci,
    build_review_queue,
    compute_judgment_coverage,
    compute_known_positive_diagnostics,
    evaluate_pilot_runs,
    load_human_labels,
    load_human_qrels,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    run_evaluation,
    validate_pilot_evidence,
)
from edge_rag.paths import (
    resolve_annotation_dir,
    resolve_catalog_dir,
    resolve_index_dir,
    resolve_models_dir,
)
from edge_rag.retriever import (
    PREPROCESS,
    REQUIRED_INDEX_FILES,
    Retriever,
    compute_index_key,
    load_persisted_index,
    rrf,
    sha256_file,
)
from edge_rag.verifier import Verifier
import qedge_common.schema as qs

FIXTURE_CATALOG = Path(__file__).parent / "fixtures" / "catalog.jsonl"


RUN_CATALOG_INTEGRATION = os.environ.get("QEDGE_RUN_CATALOG_INTEGRATION") == "1"
CATALOG_SKIP = "Requires separately acquired catalog/index/annotation assets; set QEDGE_RUN_CATALOG_INTEGRATION=1"

class TestBM25Baseline(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(catalog_path=FIXTURE_CATALOG, enable_neural=False)

    def test_bm25_retrieval_determinism(self):
        res1 = self.retriever.retrieve("wooden chair", k=3)
        res2 = self.retriever.retrieve("wooden chair", k=3)
        self.assertEqual(res1, res2)
        self.assertTrue(len(res1) > 0)
        self.assertEqual(res1[0]["rank"], 1)
        self.assertGreater(res1[0]["score"], 0.0)

    def test_zero_overlap_returns_empty(self):
        res = self.retriever.retrieve("quantum flux capacitor xyz123", k=5)
        self.assertEqual(res, [])

    def test_describe_reflects_bm25_baseline(self):
        self.assertIn("local BM25", self.retriever.describe())
        self.assertIn("not neural", self.retriever.describe())


class TestFeatureFlagEnforcement(unittest.TestCase):
    def test_enable_neural_false_rejects_or_falls_back_with_warning(self):
        r = Retriever(catalog_path=FIXTURE_CATALOG, enable_neural=False)
        self.assertFalse(r.enable_neural)
        req = {
            "schema_version": "1",
            "query_id": "test-flag",
            "text": "chair",
            "k": 3,
            "catalog_version": r.catalog_version,
            "method": "text",
        }
        res = r.search_request(req)
        # Res falls back to bm25 and issues explicit warning
        self.assertTrue(any("neural retrieval is disabled" in w.lower() for w in res["warnings"]))
        self.assertEqual(r.model_version, "local-bm25-v0")


@unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
class TestIndexKeyAndInvalidation(unittest.TestCase):
    def setUp(self):
        self.catalog_dir = resolve_catalog_dir()
        self.models_dir = resolve_models_dir()
        self.index_dir = resolve_index_dir()

    def test_index_key_calculation(self):
        cat_man = self.catalog_dir / "manifest.json"
        mod_man = self.models_dir / "manifest.json"
        expected = compute_index_key(cat_man, mod_man)
        index_man = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(expected, index_man["key"])

    def test_stale_index_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            load_persisted_index(self.index_dir, "wrong_digest_key_1234567890abcdef")
        self.assertIn("stale catalog/model/preprocessing index", str(ctx.exception))

    def test_corrupted_file_checksum_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            # Copy manifest and ids.json, but tamper with text.npy
            for fname in ["manifest.json", "ids.json", "image.npy"]:
                (tmp / fname).write_bytes((self.index_dir / fname).read_bytes())
            (tmp / "text.npy").write_bytes(b"corrupted bytes")

            cat_man = self.catalog_dir / "manifest.json"
            mod_man = self.models_dir / "manifest.json"
            expected = compute_index_key(cat_man, mod_man)

            with self.assertRaises(ValueError) as ctx:
                load_persisted_index(tmp, expected)
            self.assertIn("index file checksum mismatch", str(ctx.exception))


class TestWeightedRRFAndDeduplication(unittest.TestCase):
    def test_rrf_one_based_ranks(self):
        rankings = {
            "bm25": [("item_A", 10.0), ("item_B", 5.0)],
            "text": [("item_B", 0.9), ("item_A", 0.8)],
        }
        fused = rrf(rankings, k=60)
        score_A = 1.0 / 61 + 1.0 / 62
        score_B = 1.0 / 62 + 1.0 / 61
        self.assertAlmostEqual(fused[0]["score"], score_A, places=7)
        self.assertAlmostEqual(fused[1]["score"], score_B, places=7)
        # Tie-breaking by item_id: "item_A" before "item_B"
        self.assertEqual(fused[0]["item_id"], "item_A")
        self.assertEqual(fused[1]["item_id"], "item_B")

    def test_weighted_rrf(self):
        rankings = {
            "bm25": [("item_A", 10.0)],
            "text": [("item_B", 0.9)],
        }
        weights = {"bm25": 1.0, "text": 2.0}
        fused = rrf(rankings, weights=weights, k=60)
        self.assertEqual(fused[0]["item_id"], "item_B")
        self.assertEqual(fused[1]["item_id"], "item_A")
        self.assertAlmostEqual(fused[0]["score"], 2.0 / 61, places=7)
        self.assertAlmostEqual(fused[1]["score"], 1.0 / 61, places=7)

    def test_duplicate_family_deduplication(self):
        r = Retriever(catalog_path=self._create_mock_catalog_with_families(), dedup_families=True)
        r.family_map = {
            "item_1": "fam_alpha",
            "item_2": "fam_alpha",
            "item_3": "fam_beta",
        }
        res = r.search_request({
            "schema_version": "1",
            "query_id": "test-dedup",
            "text": "chair",
            "k": 5,
            "catalog_version": r.catalog_version,
            "method": "hybrid",
        })
        item_ids = [c["item_id"] for c in res["candidates"]]
        fam_counts = {}
        for iid in item_ids:
            fam = r.family_map.get(iid, iid)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        for fam, cnt in fam_counts.items():
            self.assertLessEqual(cnt, 1)

    def _create_mock_catalog_with_families(self) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        rows = [
            {"item_id": "item_1", "catalog_version": "test-cat", "title": "Chair Alpha 1", "text": "wooden chair"},
            {"item_id": "item_2", "catalog_version": "test-cat", "title": "Chair Alpha 2", "text": "wooden chair"},
            {"item_id": "item_3", "catalog_version": "test-cat", "title": "Chair Beta 1", "text": "wooden chair"},
        ]
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        return Path(tmp.name)


class TestVerifierSeparation(unittest.TestCase):
    def test_retrieved_candidate_missing_attributes_is_not_supported(self):
        catalog_path = FIXTURE_CATALOG
        retriever = Retriever(catalog_path=catalog_path, enable_neural=False)
        verifier = Verifier(catalog_path=catalog_path)

        query = "solid oak chair width under 40cm"
        candidates = retriever.retrieve(query, k=3)
        self.assertTrue(len(candidates) > 0)
        top = candidates[0]
        self.assertGreater(top["score"], 0.0)

        parse_res = verifier.parse_requirements(query)
        decision = verifier.verify_candidate(top["item_id"], parse_res, query)

        self.assertIn(decision["status"], ("REJECTED", "ABSTAIN", "INSUFFICIENT_EVIDENCE"))
        self.assertNotEqual(decision["status"], "SUPPORTED")


class TestEvaluationMetrics(unittest.TestCase):
    def test_recall_at_k(self):
        ranked = ["a", "b", "c", "d"]
        rel = {"b", "d"}
        self.assertEqual(recall_at_k(ranked, rel, 1), 0.0)
        self.assertEqual(recall_at_k(ranked, rel, 2), 0.5)
        self.assertEqual(recall_at_k(ranked, rel, 4), 1.0)
        self.assertEqual(recall_at_k(ranked, set(), 5), 0.0)

    def test_mrr_at_k(self):
        ranked = ["a", "b", "c"]
        self.assertEqual(mrr_at_k(ranked, {"b"}, 3), 0.5)
        self.assertEqual(mrr_at_k(ranked, {"a"}, 3), 1.0)
        self.assertEqual(mrr_at_k(ranked, {"z"}, 3), 0.0)

    def test_ndcg_at_k(self):
        ranked = ["a", "b", "c"]
        rel = {"a", "c"}
        idcg = 1.0 / 1.0 + 1.0 / 1.5849625
        dcg = 1.0 / 1.0 + 1.0 / 2.0
        expected = dcg / idcg
        self.assertAlmostEqual(ndcg_at_k(ranked, rel, 3), expected, places=5)

    def test_bootstrap_ci(self):
        vals = [1.0, 0.0, 1.0, 0.0, 1.0]
        lo, hi = bootstrap_ci(vals, iterations=500, seed=42)
        self.assertLessEqual(lo, hi)
        self.assertTrue(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0)

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_human_qrels_loaded_from_sqlite(self):
        ann_dir = resolve_annotation_dir()
        db_path = ann_dir / "labels.sqlite3"
        labels = load_human_labels(db_path)
        qrels = load_human_qrels(labels, min_grade=1)
        self.assertEqual(len(qrels), 20)
        self.assertTrue(all(len(items) > 0 for items in qrels.values()))

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_run_evaluation_produces_blocked_quality_gate(self):
        report = run_evaluation()
        self.assertEqual(report["stage"], "09-catalog-retrieval")
        self.assertEqual(report["catalog_version"], "abo-v2-pilot-500")
        self.assertEqual(report["overall_status"], "PILOT_VERIFIED_FULL_EVAL_BLOCKED")
        plan = report["evaluation_plan_120"]
        self.assertEqual(plan["quality_gate"]["status"], "BLOCKED")
        self.assertIn("queries 31-120", plan["quality_gate"]["reason"])
        self.assertIn("pending annotation", plan["quality_gate"]["reason"])


class TestSchemaValidation(unittest.TestCase):
    def test_search_request_validates_against_schema(self):
        catalog_path = FIXTURE_CATALOG
        r = Retriever(catalog_path=catalog_path, enable_neural=False)
        req = {
            "schema_version": "1",
            "query_id": "test-schema-01",
            "text": "dining chair",
            "k": 5,
            "catalog_version": r.catalog_version,
            "method": "bm25",
        }
        res = r.search_request(req)
        qs.validate("retrieval_result", res)
        self.assertEqual(res["schema_version"], "1")
        self.assertIn("timing", res)
        self.assertIn("candidates", res)


@unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
class TestNeuralIndexIntegrityProbes(unittest.TestCase):
    """Independent probe tests verifying resolution of neural index integrity gaps."""

    def setUp(self):
        self.catalog_dir = resolve_catalog_dir()
        self.models_dir = resolve_models_dir()
        self.index_dir = resolve_index_dir()
        self.expected_key = compute_index_key(
            self.catalog_dir / "manifest.json",
            self.models_dir / "manifest.json",
        )
        catalog_rows = Retriever._load_rows(self.catalog_dir / "catalog.jsonl")
        self.canonical_ids = [str(r["item_id"]) for r in catalog_rows]

    def test_probe_empty_files_dict_rejected(self):
        """Probe: manifest containing files: {} must be rejected before loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            for fname in ["ids.json", "text.npy", "image.npy"]:
                (tmp / fname).write_bytes((self.index_dir / fname).read_bytes())
            # Write manifest with valid key but files: {}
            bad_manifest = {
                "key": self.expected_key,
                "preprocessing": PREPROCESS,
                "files": {},
            }
            (tmp / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_persisted_index(tmp, self.expected_key, catalog_ids=self.canonical_ids)
            self.assertIn("missing required file digest in index manifest", str(ctx.exception))

    def test_probe_invented_item_id_rejected(self):
        """Probe: index containing invented item ID not in catalog is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            tampered_ids = list(self.canonical_ids)
            tampered_ids[0] = "INVENTED_ITEM_ID_99999"
            (tmp / "ids.json").write_text(json.dumps(tampered_ids), encoding="utf-8")
            (tmp / "text.npy").write_bytes((self.index_dir / "text.npy").read_bytes())
            (tmp / "image.npy").write_bytes((self.index_dir / "image.npy").read_bytes())

            manifest = {
                "key": self.expected_key,
                "preprocessing": PREPROCESS,
                "files": {
                    "ids.json": sha256_file(tmp / "ids.json"),
                    "text.npy": sha256_file(tmp / "text.npy"),
                    "image.npy": sha256_file(tmp / "image.npy"),
                },
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_persisted_index(tmp, self.expected_key, catalog_ids=self.canonical_ids)
            self.assertIn("index item set does not match catalog items", str(ctx.exception))

    def test_probe_reordered_ids_rejected(self):
        """Probe: index with valid IDs in swapped order violates row alignment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            reordered = list(self.canonical_ids)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            (tmp / "ids.json").write_text(json.dumps(reordered), encoding="utf-8")
            (tmp / "text.npy").write_bytes((self.index_dir / "text.npy").read_bytes())
            (tmp / "image.npy").write_bytes((self.index_dir / "image.npy").read_bytes())

            manifest = {
                "key": self.expected_key,
                "preprocessing": PREPROCESS,
                "files": {
                    "ids.json": sha256_file(tmp / "ids.json"),
                    "text.npy": sha256_file(tmp / "text.npy"),
                    "image.npy": sha256_file(tmp / "image.npy"),
                },
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_persisted_index(tmp, self.expected_key, catalog_ids=self.canonical_ids)
            self.assertIn("index item alignment mismatch with catalog", str(ctx.exception))

    def test_probe_unnormalized_vectors_rejected(self):
        """Probe: unnormalized embedding matrix must be rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "ids.json").write_bytes((self.index_dir / "ids.json").read_bytes())
            (tmp / "image.npy").write_bytes((self.index_dir / "image.npy").read_bytes())
            raw_text = np.load(self.index_dir / "text.npy", allow_pickle=False)
            unnorm_text = raw_text * 5.0
            np.save(tmp / "text.npy", unnorm_text)

            manifest = {
                "key": self.expected_key,
                "preprocessing": PREPROCESS,
                "files": {
                    "ids.json": sha256_file(tmp / "ids.json"),
                    "text.npy": sha256_file(tmp / "text.npy"),
                    "image.npy": sha256_file(tmp / "image.npy"),
                },
            }
            (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_persisted_index(tmp, self.expected_key, catalog_ids=self.canonical_ids)
            self.assertIn("unnormalized index", str(ctx.exception))

    def test_probe_schema_validation_failures_propagate_as_errors(self):
        """Probe: invalid schema payload in search_request raises rather than logs warning."""
        catalog_path = self.catalog_dir / "catalog.jsonl"
        r = Retriever(catalog_path=catalog_path, enable_neural=False)

        try:
            bad_req = {
                "schema_version": "1",
                "query_id": "test-invalid-schema",
                "text": "chair",
                "k": 5,
                "catalog_version": r.catalog_version,
                "method": "bm25",
            }
            # Return item_id with spaces and exclamation mark, violating candidate item_id pattern
            def broken_bm25_search(*args, **kwargs):
                return [("bad item with spaces!", 1.0)]
            r._index.search = broken_bm25_search

            with self.assertRaises(qs.ContractValidationError) as ctx:
                r.search_request(bad_req)
            self.assertIn("does not match", str(ctx.exception))
        finally:
            r._index = Retriever(catalog_path=catalog_path, enable_neural=False)._index


class TestEvaluationGateAndCoverageProbes(unittest.TestCase):
    """Independent probe tests for evaluation gate, evidence validation, and coverage."""

    def setUp(self):
        self.ann_dir = resolve_annotation_dir()

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_empty_labels_db_produces_incomplete_status(self):
        """Probe: empty labels database must return PILOT_EVAL_INCOMPLETE, not PILOT_VERIFIED."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            report = run_evaluation(annotation_dir=tmp)
            self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
            self.assertEqual(report["status"], "EVIDENCE_INVALID")
            self.assertIn("tasks list is empty", report["error"])

    def test_probe_insufficient_ranking_depth_rejected(self):
        """Probe: candidate runs with depth 10 cannot satisfy Recall@20 depth requirement."""
        cat_ver = "abo-v2-pilot-500"
        idx_ver = "idx-v1"
        mod_ver = "MiniLM+OpenCLIP-frozen"
        cands_10 = [
            {"item_id": f"item_{i:02d}", "rank": i, "score": 1.0 / i}
            for i in range(1, 11)
        ]
        tasks = [
            {
                "query_id": "pilot-01",
                "catalog_version": cat_ver,
                "index_version": idx_ver,
                "candidate_ids": [f"item_{i:02d}" for i in range(1, 11)],
            }
        ]
        runs = [
            {
                "query_id": "pilot-01",
                "method": m,
                "status": "completed",
                "catalog_version": cat_ver,
                "index_version": idx_ver,
                "model_version": mod_ver,
                "response": {
                    "schema_version": "1",
                    "request_id": f"req-{m}",
                    "catalog_version": cat_ver,
                    "index_version": idx_ver,
                    "model_version": mod_ver,
                    "candidates": copy.deepcopy(cands_10),
                    "timing": {
                        "encode_ms": 1.0,
                        "retrieve_ms": 1.0,
                        "rerank_ms": 1.0,
                        "total_ms": 3.0,
                    },
                    "warnings": [],
                },
            }
            for m in ("bm25", "text", "hybrid")
        ]
        runs.append(
            {
                "query_id": "pilot-01",
                "method": "image",
                "status": "unavailable",
                "catalog_version": cat_ver,
                "index_version": idx_ver,
                "model_version": mod_ver,
                "reason": "unavailable for test",
            }
        )
        labels = [
            {
                "query_id": "pilot-01",
                "item_id": "item_01",
                "source": "human",
                "grade": 3,
                "catalog_version": cat_ver,
                "index_version": idx_ver,
            }
        ]

        with self.assertRaises(EvidenceValidationError) as ctx:
            validate_pilot_evidence(tasks, runs, labels, min_depth=20)
        self.assertIn("depth 10 is less than required 20", str(ctx.exception))

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_pilot_judgment_coverage_exact_numbers(self):
        """Probe: exact coverage verification across historical, fresh, and union pools."""
        tasks_file = self.ann_dir / "pilot-v1" / "tasks.json"
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))
        labels = load_human_labels(self.ann_dir / "labels.sqlite3")

        cov = compute_judgment_coverage(tasks, runs, labels)
        self.assertEqual(cov["historical_pool"]["total_pairs"], 685)
        self.assertEqual(cov["historical_pool"]["judged_pairs"], 30)
        self.assertEqual(cov["historical_pool"]["unjudged_pairs"], 655)
        self.assertEqual(cov["historical_pool"]["coverage_pct"], 4.38)

        self.assertEqual(cov["fresh_pool"]["total_pairs"], 1032)
        self.assertEqual(cov["fresh_pool"]["judged_pairs"], 23)
        self.assertEqual(cov["fresh_pool"]["unjudged_pairs"], 1009)
        self.assertEqual(cov["fresh_pool"]["coverage_pct"], 2.23)

        self.assertEqual(cov["union_pool"]["total_pairs"], 1172)
        self.assertEqual(cov["union_pool"]["judged_pairs"], 30)
        self.assertEqual(cov["union_pool"]["unjudged_pairs"], 1142)
        self.assertEqual(cov["union_pool"]["coverage_pct"], 2.56)

        self.assertEqual(cov["newly_retrieved_pairs_count"], 487)
        self.assertEqual(cov["queries_with_known_positive_count"], 20)
        self.assertEqual(cov["queries_without_known_positive_count"], 10)
        expected_no_pos = [
            "pilot-02", "pilot-08", "pilot-12", "pilot-13", "pilot-14",
            "pilot-18", "pilot-20", "pilot-25", "pilot-27", "pilot-30",
        ]
        self.assertEqual(cov["queries_without_known_positive"], expected_no_pos)

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_review_queue_contains_1009_unjudged_pairs(self):
        """Probe: active review queue covers fresh pool (1009 unjudged) and union queue covers 1142."""
        tasks_file = self.ann_dir / "pilot-v1" / "tasks.json"
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))
        labels = load_human_labels(self.ann_dir / "labels.sqlite3")

        cov = compute_judgment_coverage(tasks, runs, labels)

        queue_file = self.ann_dir / "pilot-v1" / "review_queue.json"
        self.assertTrue(queue_file.is_file())
        queue = json.loads(queue_file.read_text(encoding="utf-8"))
        self.assertEqual(len(queue), 1009)
        self.assertEqual(len(queue), cov["fresh_pool"]["unjudged_pairs"])
        self.assertTrue(all(item["status"] == "pending_human_review" for item in queue))
        self.assertTrue(all(item.get("candidate_id") and item.get("item_id") for item in queue))
        self.assertTrue(all(len(item.get("contributing_runs", [])) > 0 for item in queue))

        # Union queue assertion
        union_queue = build_review_queue(tasks, runs, labels, scope="union")
        self.assertEqual(len(union_queue), 1142)
        self.assertEqual(len(union_queue), cov["union_pool"]["unjudged_pairs"])

        # Historical queue assertion
        hist_queue = build_review_queue(tasks, runs, labels, scope="historical")
        self.assertEqual(len(hist_queue), 655)
        self.assertEqual(len(hist_queue), cov["historical_pool"]["unjudged_pairs"])

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_tasks_status_is_reconciled(self):
        """Probe: query-review status between tasks and labels is reconciled."""
        tasks_file = self.ann_dir / "pilot-v1" / "tasks.json"
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        self.assertTrue(all(t["status"] == "query_reviewed" for t in tasks))

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_known_positive_diagnostics_denominator(self):
        """Probe: known-positive rank diagnostics use explicit denominator N_pos=20."""
        tasks_file = self.ann_dir / "pilot-v1" / "tasks.json"
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))
        labels = load_human_labels(self.ann_dir / "labels.sqlite3")

        cov = compute_judgment_coverage(tasks, runs, labels)
        diag = compute_known_positive_diagnostics(runs, cov["known_positives_by_query"])

        for method in ("bm25", "text", "hybrid"):
            self.assertIn(method, diag)
            self.assertEqual(diag[method]["denominator_known_positive_queries"], 20)
            self.assertGreaterEqual(diag[method]["recall@20_known_positives"], 0.0)
            self.assertLessEqual(diag[method]["recall@20_known_positives"], 1.0)

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_negative_wrong_index_revision_rejected_by_run_evaluation(self):
        """Probe: in-memory copy of BM25 run with wrong index revision must fail run_evaluation."""
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))

        tampered = copy.deepcopy(runs)
        for r in tampered:
            if r.get("query_id") == "pilot-01" and r.get("method") == "bm25":
                r["index_version"] = "wrong-revision"
                r["response"]["index_version"] = "wrong-revision"
                break

        report = run_evaluation(runs=tampered)
        self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
        self.assertEqual(report["status"], "EVIDENCE_INVALID")
        self.assertEqual(report["offending_run"], "pilot-01/bm25")
        self.assertIn("inconsistent index_version 'wrong-revision'", report["error"])

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_negative_invented_item_id_rejected_by_run_evaluation(self):
        """Probe: in-memory copy of BM25 run with invented item ID must fail run_evaluation."""
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))

        tampered = copy.deepcopy(runs)
        for r in tampered:
            if r.get("query_id") == "pilot-01" and r.get("method") == "bm25":
                r["response"]["candidates"][0]["item_id"] = "invented-item"
                break

        report = run_evaluation(runs=tampered)
        self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
        self.assertEqual(report["status"], "EVIDENCE_INVALID")
        self.assertEqual(report["offending_run"], "pilot-01/bm25")
        self.assertIn("candidate item_id 'invented-item' is not in catalog", report["error"])

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_negative_nonnumeric_candidate_score_rejected_by_run_evaluation(self):
        """Probe: candidate score set to 'not-a-number' fails schema validation and run_evaluation."""
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))

        tampered = copy.deepcopy(runs)
        for r in tampered:
            if r.get("query_id") == "pilot-01" and r.get("method") == "bm25":
                r["response"]["candidates"][0]["score"] = "not-a-number"
                break

        report = run_evaluation(runs=tampered)
        self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
        self.assertEqual(report["status"], "EVIDENCE_INVALID")
        self.assertEqual(report["offending_run"], "pilot-01/bm25")
        self.assertIn("retrieval_result schema validation failed", report["error"])
        self.assertIn("is not of type 'number'", report["error"])

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_negative_missing_schema_required_response_field_rejected_by_run_evaluation(self):
        """Probe: response missing a schema-required field fails schema validation and run_evaluation."""
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))

        tampered = copy.deepcopy(runs)
        for r in tampered:
            if r.get("query_id") == "pilot-01" and r.get("method") == "bm25":
                del r["response"]["timing"]
                break

        report = run_evaluation(runs=tampered)
        self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
        self.assertEqual(report["status"], "EVIDENCE_INVALID")
        self.assertEqual(report["offending_run"], "pilot-01/bm25")
        self.assertIn("retrieval_result schema validation failed", report["error"])
        self.assertIn("'timing' is a required property", report["error"])

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_negative_duplicate_image_request_and_missing_query_rejected_by_run_evaluation(self):
        """Probe: duplicating an unavailable image request and omitting another query is rejected."""
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))

        tampered = copy.deepcopy(runs)
        img_01 = next(r for r in tampered if r.get("query_id") == "pilot-01" and r.get("method") == "image")
        img_02_idx = next(i for i, r in enumerate(tampered) if r.get("query_id") == "pilot-02" and r.get("method") == "image")
        tampered[img_02_idx] = copy.deepcopy(img_01)

        report = run_evaluation(runs=tampered)
        self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
        self.assertEqual(report["status"], "EVIDENCE_INVALID")
        self.assertEqual(report["offending_run"], "pilot-01/image")
        self.assertIn("duplicate run found for query 'pilot-01' and method 'image'", report["error"])

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_negative_missing_or_blank_unavailable_reason_rejected_by_run_evaluation(self):
        """Probe: unavailable image request with missing or blank reason is rejected."""
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))

        tampered = copy.deepcopy(runs)
        for r in tampered:
            if r.get("query_id") == "pilot-01" and r.get("method") == "image":
                r["reason"] = "   "
                break

        report = run_evaluation(runs=tampered)
        self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
        self.assertEqual(report["status"], "EVIDENCE_INVALID")
        self.assertEqual(report["offending_run"], "pilot-01/image")
        self.assertIn("missing required nonempty reason", report["error"])

    @unittest.skipUnless(RUN_CATALOG_INTEGRATION, CATALOG_SKIP)
    def test_probe_negative_unknown_method_rejected_by_run_evaluation(self):
        """Probe: run with unknown method is explicitly rejected by run_evaluation."""
        runs_file = self.ann_dir / "pilot-v1" / "candidate_runs.json"
        runs = json.loads(runs_file.read_text(encoding="utf-8"))

        tampered = copy.deepcopy(runs)
        for r in tampered:
            if r.get("query_id") == "pilot-01" and r.get("method") == "bm25":
                r["method"] = "unknown-method"
                break

        report = run_evaluation(runs=tampered)
        self.assertEqual(report["overall_status"], "PILOT_EVAL_INCOMPLETE")
        self.assertEqual(report["status"], "EVIDENCE_INVALID")
        self.assertEqual(report["offending_run"], "pilot-01/unknown-method")
        self.assertIn("unknown or missing method 'unknown-method'", report["error"])
