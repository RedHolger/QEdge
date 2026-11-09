"""Comprehensive Stage 10 tests for the Local Assistant integration.

Verifies:
- Find and check modes end-to-end.
- Deterministic extractive path without model downloads.
- Safety invariant: no recommendation when mandatory predicate is unresolved.
- Tenant/request isolation under declared boundary.
- Document revision updating verdict and immutable historical citation resolution.
- Timeout / cancellation enforcement with structured TIMEOUT error.
- Contract-guarded generation fallback and reason recording.
- Image query refusal without visual fabrication.
- CLI execution in both modes with structured JSON output.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import time
import unittest

from edge_rag.assistant import (
    AssistantAuthorizationError,
    AssistantResult,
    AssistantTimeoutError,
    LocalAssistant,
)
from edge_rag.cli import main as cli_main
from edge_rag.integration_eval import evaluate_10_scenarios
from edge_rag.retriever import Retriever
from edge_rag.store import AuthorizationError, CitationError, EvidenceStore, Fact, Principal

EDGE_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"


class IntegrationAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = LocalAssistant(catalog_path=FIXTURE_CATALOG)

    def test_find_mode_supported_recommend(self):
        res = self.assistant.find("dining chair with oak material under 60cm wide")
        self.assertEqual(res.status, "OK")
        self.assertEqual(res.action, "RECOMMEND")
        self.assertEqual(res.decision_status, "SUPPORTED")
        self.assertEqual(res.item_id, "fixture-chair-oak")
        self.assertGreater(len(res.citations), 0)
        self.assertTrue(all(c["valid"] for c in res.citations))
        self.assertGreater(res.timing_ms["parse_ms"], 0.0)
        self.assertGreater(res.timing_ms["retrieve_ms"], 0.0)
        self.assertGreater(res.timing_ms["verify_ms"], 0.0)
        self.assertIn("Recommended:", res.answer)

    def test_find_mode_unmatched_query_abstain(self):
        res = self.assistant.find("quantum flux capacitor 9999")
        self.assertEqual(res.status, "OK")
        self.assertEqual(res.action, "ABSTAIN")
        self.assertEqual(res.decision_status, "ABSTAIN")
        self.assertIsNone(res.item_id)
        self.assertEqual(res.candidates, [])
        self.assertIn("No candidate item matched", res.reason)

    def test_check_mode_supported_skips_retrieval(self):
        res = self.assistant.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.status, "OK")
        self.assertEqual(res.action, "RECOMMEND")
        self.assertEqual(res.decision_status, "SUPPORTED")
        self.assertEqual(res.timing_ms["retrieve_ms"], 0.0, "Check mode must skip retrieval stage")
        self.assertGreater(len(res.citations), 0)
        self.assertTrue(all(c["valid"] for c in res.citations))

    def test_check_mode_contradicted(self):
        res = self.assistant.check("fixture-chair-wide", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.status, "OK")
        self.assertEqual(res.action, "DO_NOT_RECOMMEND")
        self.assertEqual(res.decision_status, "CONTRADICTED")
        self.assertIn("Do not recommend:", res.answer)

    def test_mandatory_predicate_safety_never_recommends_unknown_or_unparsed(self):
        # 1. Unknown attribute
        res_unk = self.assistant.check("fixture-chair-unknown", "chair width <= 60cm")
        self.assertEqual(res_unk.action, "ABSTAIN")
        self.assertEqual(res_unk.decision_status, "UNKNOWN")

        # 2. Unparsed clause
        res_unp = self.assistant.check("fixture-chair-oak", "dining chair with built-in quantum charger")
        self.assertEqual(res_unp.action, "ABSTAIN")
        self.assertGreater(len(res_unp.decision.get("unparsed_clauses", [])), 0)

        # 3. Ambiguous units
        res_amb = self.assistant.check("fixture-chair-oak", "dining chair under 60")
        self.assertEqual(res_amb.action, "ABSTAIN")
        self.assertGreater(len(res_amb.decision.get("ambiguities", [])), 0)

    def test_tenant_request_isolation(self):
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-tenant-secret",
            Principal.user("tenant-alpha"),
            ["width: 44.0 cm (source: internal_spec)"],
            is_public=False,
            facts=[Fact(attribute="width", value=44.0, unit="cm", status="present", source_location="internal_spec", source_revision="v1")],
        )

        tenant_assistant = LocalAssistant(store=custom_store)

        # Tenant Alpha (owner) can verify candidate
        res_alpha = tenant_assistant.check("doc-tenant-secret", "chair width <= 50cm", requester="tenant-alpha")
        self.assertEqual(res_alpha.status, "OK")
        self.assertEqual(res_alpha.action, "RECOMMEND")

        # Tenant Beta (unauthorized caller) is blocked with UNAUTHORIZED error
        res_beta = tenant_assistant.check("doc-tenant-secret", "chair width <= 50cm", requester="tenant-beta")
        self.assertEqual(res_beta.status, "ERROR")
        self.assertEqual(res_beta.error["code"], "UNAUTHORIZED")
        self.assertIn("denied", res_beta.reason.lower())

        # Tenant Beta cannot resolve Alpha's citation
        cite_str = res_alpha.citations[0]["citation"]
        with self.assertRaises(AuthorizationError):
            custom_store.resolve_citation(cite_str, requester="tenant-beta")

    def test_document_revision_update_and_historical_citation_immutability(self):
        custom_store = EvidenceStore(":memory:")
        # Version 1: 75cm -> CONTRADICTED
        custom_store.upload(
            "doc-chair-rev",
            Principal.system(),
            ["width: 75.0 cm (source: rev1)"],
            is_public=True,
            facts=[Fact(attribute="width", value=75.0, unit="cm", status="present", source_location="rev1", source_revision="v1")],
        )
        asst = LocalAssistant(store=custom_store)
        res_v1 = asst.check("doc-chair-rev", "chair width <= 60cm")
        self.assertEqual(res_v1.action, "DO_NOT_RECOMMEND")
        cite_v1 = res_v1.citations[0]["citation"]

        # Version 2: updated to 50cm -> RECOMMEND
        custom_store.upload(
            "doc-chair-rev",
            Principal.system(),
            ["width: 50.0 cm (source: rev2)"],
            is_public=True,
            expected_version=1,
            facts=[Fact(attribute="width", value=50.0, unit="cm", status="present", source_location="rev2", source_revision="v2")],
        )
        res_v2 = asst.check("doc-chair-rev", "chair width <= 60cm")
        self.assertEqual(res_v2.action, "RECOMMEND")
        cite_v2 = res_v2.citations[0]["citation"]

        self.assertNotEqual(cite_v1, cite_v2)
        # Historical v1 citation resolves to original 75.0cm fact
        res_hist = custom_store.resolve_citation(cite_v1)
        self.assertEqual(res_hist["fact"]["value"], 75.0)
        self.assertEqual(res_hist["version"], 1)

    def test_timeout_enforcement(self):
        # Extremely small timeout ensures timeout triggers cleanly
        res = self.assistant.find("dining chair with oak material under 60cm wide", timeout_s=0.0000001)
        self.assertEqual(res.status, "ERROR")
        self.assertEqual(res.error["code"], "TIMEOUT")
        self.assertIn("timed out", res.reason.lower())

    def test_contract_guarded_generation_fallback(self):
        # Mock runner attempting to recommend an adverse candidate
        class MaliciousRunner:
            def generate(self, prompt: str, max_tokens: int = 50):
                return {"text": "I highly recommend buying this chair, it is fantastic!"}

        adv_assistant = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=MaliciousRunner(),
            enable_generation=True,
        )
        res = adv_assistant.check("fixture-chair-wide", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "DO_NOT_RECOMMEND")
        self.assertIsNotNone(res.generation_fallback_reason)
        self.assertIn("contract_violation", res.generation_fallback_reason)
        self.assertIn("Do not recommend:", res.answer)

    def test_contract_guarded_generation_citation_hallucination_fallback(self):
        # Mock runner inventing an unverified citation
        class HallucinatingRunner:
            def generate(self, prompt: str, max_tokens: int = 50):
                return {"text": "The chair is oak. [cite:fake-doc:v99:c99:deadbeef]"}

        hall_assistant = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=HallucinatingRunner(),
            enable_generation=True,
        )
        res = hall_assistant.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "RECOMMEND")
        self.assertIsNotNone(res.generation_fallback_reason)
        self.assertIn("contract_violation", res.generation_fallback_reason)
        self.assertIn("unverified citation", res.generation_fallback_reason)

    def test_image_query_refusal_without_visual_fabrication(self):
        res = self.assistant.find("--image chair.png")
        self.assertEqual(res.action, "ABSTAIN")
        self.assertIn("refused", res.reason.lower())
        self.assertIn("Image assets unavailable", res.answer)

    def test_all_ten_scenarios(self):
        eval_res = evaluate_10_scenarios(self.assistant)
        self.assertTrue(eval_res["all_passed"], f"Not all scenarios passed: {eval_res}")
        self.assertEqual(eval_res["passed_scenarios"], 10)

    def test_cli_execution_find_and_check(self):
        # Test CLI find mode with --json
        stdout_buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_buf
            code = cli_main([
                "--catalog", str(FIXTURE_CATALOG),
                "--mode", "find",
                "dining chair with oak material under 60cm wide",
                "--json",
            ])
        finally:
            sys.stdout = old_stdout

        self.assertEqual(code, 0)
        cli_dict = json.loads(stdout_buf.getvalue())
        self.assertEqual(cli_dict["mode"], "find")
        self.assertEqual(cli_dict["action"], "RECOMMEND")

        # Test CLI check mode with --json
        stdout_buf = io.StringIO()
        try:
            sys.stdout = stdout_buf
            code = cli_main([
                "--catalog", str(FIXTURE_CATALOG),
                "--mode", "check",
                "--item-id", "fixture-chair-wide",
                "dining chair with oak material under 60cm wide",
                "--json",
            ])
        finally:
            sys.stdout = old_stdout

        self.assertEqual(code, 0)
        cli_dict = json.loads(stdout_buf.getvalue())
        self.assertEqual(cli_dict["mode"], "check")
        self.assertEqual(cli_dict["action"], "DO_NOT_RECOMMEND")

    def test_find_mode_filters_unauthorized_candidates_completely(self):
        """Verify that find mode strictly excludes private candidates from all outputs."""
        store = EvidenceStore(":memory:")
        store.upload(
            "public-chair",
            Principal.system(),
            ["width: 48.0 cm (source: pub)"],
            is_public=True,
            facts=[Fact(attribute="width", value=48.0, unit="cm", status="present", source_location="pub", source_revision="v1")],
        )
        store.upload(
            "secret-chair-alpha",
            Principal.user("alpha"),
            ["width: 45.0 cm (source: priv)"],
            is_public=False,
            facts=[Fact(attribute="width", value=45.0, unit="cm", status="present", source_location="priv", source_revision="v1")],
        )

        class MixedRetriever:
            def retrieve(self, q, k):
                return [
                    {"item_id": "secret-chair-alpha", "title": "Confidential Prototype Alpha", "score": 0.99, "rank": 1},
                    {"item_id": "public-chair", "title": "Standard Public Chair", "score": 0.85, "rank": 2},
                ]

        asst = LocalAssistant(store=store, retriever=Retriever(engine=MixedRetriever()))

        # Requester 'alpha' (owner) sees both
        res_alpha = asst.find("chair width <= 50cm", requester="alpha")
        self.assertEqual(len(res_alpha.candidates), 2)
        self.assertEqual(res_alpha.candidates[0]["item_id"], "secret-chair-alpha")

        # Requester 'beta' (unauthorized) MUST NOT receive secret-chair-alpha anywhere
        res_beta = asst.find("chair width <= 50cm", requester="beta")
        self.assertEqual(len(res_beta.candidates), 1)
        self.assertEqual(res_beta.candidates[0]["item_id"], "public-chair")
        # Ensure no private candidate leaked in verified_candidates, answer, or citations
        all_res_str = json.dumps(res_beta.to_dict())
        self.assertNotIn("secret-chair-alpha", all_res_str)
        self.assertNotIn("Confidential Prototype Alpha", all_res_str)

    def test_timeout_enforced_after_blocking_generation(self):
        """A runner taking ~100 ms under timeout_s=0.05 must return TIMEOUT error, not OK."""
        class SlowRunner:
            def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                time.sleep(0.08)
                return {"text": '{"style": "standard", "selected_claims": []}'}

        slow_asst = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=SlowRunner(),
            enable_generation=True,
        )
        res = slow_asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide", timeout_s=0.04)
        self.assertEqual(res.status, "ERROR")
        self.assertEqual(res.error["code"], "TIMEOUT")
        self.assertIn("timed out", res.reason.lower())

    def test_contract_guard_rejects_invented_facts(self):
        """Supported candidate with runner hallucinating new facts must fall back to deterministic response."""
        class InventedFactsRunner:
            def generate(self, prompt: str, max_tokens: int = 50):
                return {"text": "This chair is certified fireproof and costs 10 euros."}

        inv_asst = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=InventedFactsRunner(),
            enable_generation=True,
        )
        res = inv_asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "RECOMMEND")
        self.assertIsNotNone(res.generation_fallback_reason)
        self.assertIn("contract_violation", res.generation_fallback_reason)
        self.assertNotIn("fireproof", res.answer)
        self.assertNotIn("10 euros", res.answer)
        self.assertIn("Recommended:", res.answer)

    def test_contract_guard_rejects_phrased_recommendation_on_contradicted(self):
        """Contradicted candidate with runner outputting recommendation must fall back to deterministic answer."""
        class ContradictionPraisingRunner:
            def generate(self, prompt: str, max_tokens: int = 50):
                return {"text": "Buy this chair. It meets every requirement."}

        praise_asst = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=ContradictionPraisingRunner(),
            enable_generation=True,
        )
        res = praise_asst.check("fixture-chair-wide", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "DO_NOT_RECOMMEND")
        self.assertIsNotNone(res.generation_fallback_reason)
        self.assertIn("contract_violation", res.generation_fallback_reason)
        self.assertNotIn("Buy this chair", res.answer)
        self.assertIn("Do not recommend:", res.answer)

    def test_constrained_generation_successful_selection(self):
        """Runner outputting valid constrained JSON renders approved presentation deterministically."""
        class ConstrainedJsonRunner:
            def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                return {"text": '{"style": "compact", "selected_claims": [0]}'}

        valid_asst = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=ConstrainedJsonRunner(),
            enable_generation=True,
        )
        res = valid_asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "RECOMMEND")
        self.assertIsNone(res.generation_fallback_reason)
        self.assertIn("RECOMMEND", res.answer)
        self.assertIn("fixture-chair-oak", res.answer)

    def test_citation_verification_failure_on_contradicted_candidate_check_mode(self):
        """Independent reproduction: CitationError on fixture-chair-wide fails closed to ABSTAIN.

        Asserts:
        - Decision action and status fail closed to ABSTAIN.
        - Failed claims and invalid citations are excluded from decision claims, answer, and user output.
        - Answer does NOT mention width/material facts or invalid citations under DO_NOT_RECOMMEND.
        - Failed citations are kept separate under citation diagnostics.
        """
        asst = LocalAssistant(catalog_path=FIXTURE_CATALOG)
        orig_resolve = asst.store.resolve_citation

        def mock_resolve(citation: str, requester=None):
            if "fixture-chair-wide" in citation:
                raise CitationError(f"Integrity check failed: tampered citation {citation}")
            return orig_resolve(citation, requester)

        asst.store.resolve_citation = mock_resolve

        res = asst.check("fixture-chair-wide", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "ABSTAIN", "Decision must fail closed to ABSTAIN on citation failure")
        self.assertEqual(res.decision_status, "ABSTAIN")
        self.assertEqual(len(res.decision.get("claims", [])), 0, "Failed claims must be excluded from decision")
        self.assertEqual(len(res.citations), 0, "citations must contain ONLY successfully resolved evidence")
        self.assertTrue(all(c.get("valid", False) for c in res.citations))
        self.assertEqual(len(res.citation_diagnostics), 2, "Diagnostics must capture the failed citations")
        self.assertTrue(all(not c.get("valid", True) for c in res.citation_diagnostics))
        self.assertIn("Evidence verification failed", res.reason)

        # Verified candidate entries maintain strict separation
        self.assertEqual(len(res.verified_candidates), 1)
        cand = res.verified_candidates[0]
        self.assertEqual(len(cand["citations"]), 0, "Candidate citations must contain ONLY valid evidence")
        self.assertEqual(len(cand["citation_diagnostics"]), 2, "Candidate diagnostics must retain failed evidence")
        self.assertTrue(all(c.get("valid", False) for c in cand["citations"]))
        self.assertTrue(all(not c.get("valid", True) for c in cand["citation_diagnostics"]))

        # JSON dictionary output maintains strict separation
        d = res.to_dict()
        self.assertEqual(len(d["citations"]), 0)
        self.assertEqual(len(d["citation_diagnostics"]), 2)

        # Exclude failed evidence from deterministic answers
        self.assertNotIn("75.0cm", res.answer)
        self.assertNotIn("material=aluminum", res.answer)
        self.assertNotIn("cite:fixture-chair-wide", res.answer)
        self.assertNotIn("Do not recommend:", res.answer)
        self.assertIn("Abstain for fixture-chair-wide", res.answer)

        # Format output verification
        formatted = asst.format_output(res)
        self.assertIn("(No verified claims available)", formatted)
        self.assertIn("Citation Diagnostics (Failed Verification):", formatted)
        self.assertNotIn("75.0cm", formatted)

    def test_citation_verification_failure_on_contradicted_candidate_find_mode(self):
        """Find mode: candidate with failing citations cannot be recommended and fails closed to ABSTAIN."""
        asst = LocalAssistant(catalog_path=FIXTURE_CATALOG)
        orig_resolve = asst.store.resolve_citation

        def mock_resolve(citation: str, requester=None):
            if "fixture-chair-wide" in citation:
                raise CitationError(f"Integrity check failed: tampered citation {citation}")
            return orig_resolve(citation, requester)

        asst.store.resolve_citation = mock_resolve

        # Query that retrieves fixture-chair-wide
        res = asst.find("dining chair width under 60cm")
        # Ensure any candidate with citation error was downgraded to ABSTAIN, claims stripped,
        # and citations strictly separated from citation_diagnostics
        for cand in res.verified_candidates:
            if cand["item_id"] == "fixture-chair-wide":
                self.assertEqual(cand["decision"]["action"], "ABSTAIN")
                self.assertEqual(len(cand["decision"]["claims"]), 0)
                self.assertEqual(len(cand["citations"]), 0, "Candidate citations must contain ONLY valid evidence")
                self.assertEqual(len(cand["citation_diagnostics"]), 1, "Candidate diagnostics must retain failed evidence")
                self.assertTrue(all(c.get("valid", False) for c in cand["citations"]))
                self.assertTrue(all(not c.get("valid", True) for c in cand["citation_diagnostics"]))

        # If fixture-chair-wide was chosen as primary candidate, its answer must exclude failed evidence
        # and result collections must strictly separate valid citations from diagnostics
        if res.item_id == "fixture-chair-wide":
            self.assertEqual(res.action, "ABSTAIN")
            self.assertEqual(len(res.citations), 0)
            self.assertEqual(len(res.citation_diagnostics), 1)
            self.assertNotIn("75.0cm", res.answer)
            self.assertNotIn("cite:fixture-chair-wide", res.answer)

    def test_citation_verification_failure_on_supporting_candidate_check_and_find(self):
        """Supporting candidate with citation failure fails closed to ABSTAIN and excludes failed evidence."""
        asst = LocalAssistant(catalog_path=FIXTURE_CATALOG)
        orig_resolve = asst.store.resolve_citation

        def mock_resolve(citation: str, requester=None):
            if "fixture-chair-oak" in citation:
                raise CitationError(f"Integrity check failed: corrupted chunk for {citation}")
            return orig_resolve(citation, requester)

        asst.store.resolve_citation = mock_resolve

        # Check mode on oak chair
        res_check = asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertEqual(res_check.action, "ABSTAIN")
        self.assertEqual(res_check.decision_status, "ABSTAIN")
        self.assertEqual(len(res_check.decision.get("claims", [])), 0)
        self.assertEqual(len(res_check.citations), 0, "citations collection must be empty on complete failure")
        self.assertEqual(len(res_check.citation_diagnostics), 2, "diagnostics must retain failed records")
        self.assertNotIn("Recommended:", res_check.answer)
        self.assertIn("Evidence verification failed", res_check.reason)

        # Find mode
        res_find = asst.find("dining chair with oak material under 60cm wide")
        self.assertNotEqual(res_find.action, "RECOMMEND", "Oak chair with failed citation must NOT be recommended")
        oak_entry = next((c for c in res_find.verified_candidates if c["item_id"] == "fixture-chair-oak"), None)
        self.assertIsNotNone(oak_entry)
        self.assertEqual(oak_entry["decision"]["action"], "ABSTAIN")
        self.assertEqual(len(oak_entry["citations"]), 0)
        self.assertEqual(len(oak_entry["citation_diagnostics"]), 2)

    def test_partial_citation_failure_check_mode(self):
        """Partial citation failure in check mode:
        - Exactly 1 citation succeeds, 1 citation fails.
        - Decision fails closed to ABSTAIN.
        - res.citations contains ONLY the 1 successfully resolved citation (valid: True).
        - res.citation_diagnostics contains ONLY the 1 failed citation (valid: False).
        - verified_candidates[0] maintains identical separation.
        - Failed claim is excluded from decision claims and answer text.
        - to_dict() maintains strict separation without mixing.
        """
        asst = LocalAssistant(catalog_path=FIXTURE_CATALOG)
        orig_resolve = asst.store.resolve_citation

        # fixture-chair-oak has 2 checked attributes for this query: width (c0) and material (c1)
        # Mock so c0 succeeds, but c1 raises CitationError
        def mock_resolve(citation: str, requester=None):
            if "fixture-chair-oak" in citation and ":c1:" in citation:
                raise CitationError(f"Integrity check failed: corrupt chunk {citation}")
            return orig_resolve(citation, requester)

        asst.store.resolve_citation = mock_resolve

        res = asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "ABSTAIN", "Partial citation failure must fail closed to ABSTAIN")
        self.assertEqual(res.decision_status, "ABSTAIN")
        self.assertIn("Evidence verification failed", res.reason)

        # Citations collection contains ONLY successfully resolved evidence
        self.assertEqual(len(res.citations), 1)
        self.assertTrue(res.citations[0]["valid"])
        self.assertIn(":c0:", res.citations[0]["citation"])

        # Citation diagnostics contains ONLY failed-resolution diagnostics
        self.assertEqual(len(res.citation_diagnostics), 1)
        self.assertFalse(res.citation_diagnostics[0]["valid"])
        self.assertEqual(res.citation_diagnostics[0]["status"], "FAILED")
        self.assertIn(":c1:", res.citation_diagnostics[0]["citation"])

        # Verified candidate entries maintain identical strict separation
        self.assertEqual(len(res.verified_candidates), 1)
        cand = res.verified_candidates[0]
        self.assertEqual(len(cand["citations"]), 1)
        self.assertTrue(cand["citations"][0]["valid"])
        self.assertEqual(len(cand["citation_diagnostics"]), 1)
        self.assertFalse(cand["citation_diagnostics"][0]["valid"])

        # Failed claim (material) is excluded from decision claims
        self.assertEqual(len(res.decision.get("claims", [])), 1)
        self.assertEqual(res.decision["claims"][0]["attribute"], "width")

        # Answer text excludes failed claim and failed citation
        self.assertNotIn(":c1:", res.answer)
        self.assertNotIn("Recommended:", res.answer)
        self.assertIn("Abstain for fixture-chair-oak", res.answer)

        # JSON output maintains strict separation
        d = res.to_dict()
        self.assertEqual(len(d["citations"]), 1)
        self.assertTrue(d["citations"][0]["valid"])
        self.assertEqual(len(d["citation_diagnostics"]), 1)
        self.assertFalse(d["citation_diagnostics"][0]["valid"])

    def test_partial_citation_failure_find_mode(self):
        """Partial citation failure in find mode:
        - Candidate with partial citation failure cannot be recommended and fails closed to ABSTAIN.
        - Candidate's verified_candidates entry contains only valid citations in citations,
          and failed record in citation_diagnostics.
        - Output JSON preserves strict separation.
        """
        asst = LocalAssistant(catalog_path=FIXTURE_CATALOG)
        orig_resolve = asst.store.resolve_citation

        # Inject partial citation failure on fixture-chair-oak (:c1: fails)
        def mock_resolve(citation: str, requester=None):
            if "fixture-chair-oak" in citation and ":c1:" in citation:
                raise CitationError(f"Integrity check failed: corrupt chunk {citation}")
            return orig_resolve(citation, requester)

        asst.store.resolve_citation = mock_resolve

        res = asst.find("dining chair with oak material under 60cm wide")
        # Oak chair must NOT be recommended due to partial citation failure
        self.assertNotEqual(res.action, "RECOMMEND")

        # Check candidate separation in verified_candidates
        oak_entry = next((c for c in res.verified_candidates if c["item_id"] == "fixture-chair-oak"), None)
        self.assertIsNotNone(oak_entry)
        self.assertEqual(oak_entry["decision"]["action"], "ABSTAIN")
        self.assertEqual(len(oak_entry["citations"]), 1)
        self.assertTrue(all(c["valid"] for c in oak_entry["citations"]))
        self.assertEqual(len(oak_entry["citation_diagnostics"]), 1)
        self.assertTrue(all(not c["valid"] for c in oak_entry["citation_diagnostics"]))

        # Check overall result separation
        self.assertTrue(all(c["valid"] for c in res.citations), "All returned citations must be valid")
        self.assertTrue(all(not c.get("valid", True) for c in res.citation_diagnostics), "All diagnostics must be failed")

        # JSON dict check
        d = res.to_dict()
        self.assertTrue(all(c["valid"] for c in d["citations"]))
        self.assertTrue(all(not c.get("valid", True) for c in d["citation_diagnostics"]))

    def test_citation_integrity_hash_mismatch(self):
        """A citation with tampered hash triggers CitationError and fails closed to ABSTAIN."""
        store = EvidenceStore(":memory:")
        store.upload(
            "doc-tampered",
            Principal.system(),
            ["width: 45.0 cm (source: spec)"],
            is_public=True,
            facts=[Fact(attribute="width", value=45.0, unit="cm", status="present", source_location="spec", source_revision="v1")],
        )
        asst = LocalAssistant(store=store)
        # Verify valid baseline
        res_valid = asst.check("doc-tampered", "chair width <= 50cm")
        self.assertEqual(res_valid.action, "RECOMMEND")
        valid_cite = res_valid.citations[0]["citation"]

        # Tamper the hash prefix in the citation
        tampered_cite = valid_cite[:-8] + "00000000"
        with self.assertRaises(CitationError):
            store.resolve_citation(tampered_cite)

    def test_presentation_contract_rejects_permissive_payload(self):
        """Exact reproduction: undeclared extra keys and boolean indices must be rejected with fallback."""
        class PermissiveRunner:
            def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                return {"text": '{"style":"unsupported-style","selected_claims":[true], "invented_fact":"fireproof"}'}

        asst = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=PermissiveRunner(),
            enable_generation=True,
        )
        res = asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertEqual(res.action, "RECOMMEND")
        self.assertIsNotNone(res.generation_fallback_reason)
        self.assertIn("contract_violation", res.generation_fallback_reason)
        self.assertIn("extra keys", res.generation_fallback_reason)
        self.assertNotIn("fireproof", res.answer)

    def test_presentation_contract_rejects_unapproved_style(self):
        """Unapproved style must not be silently normalized; must trigger contract violation."""
        class BadStyleRunner:
            def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                return {"text": '{"style":"unsupported-style","selected_claims":[0]}'}

        asst = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=BadStyleRunner(),
            enable_generation=True,
        )
        res = asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertIsNotNone(res.generation_fallback_reason)
        self.assertIn("contract_violation", res.generation_fallback_reason)
        self.assertIn("unapproved style", res.generation_fallback_reason)

    def test_presentation_contract_rejects_boolean_and_out_of_range_indices(self):
        """Boolean indices and out-of-range indices must trigger contract violation fallback."""
        class BoolIndexRunner:
            def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                return {"text": '{"style":"standard","selected_claims":[true]}'}

        asst_bool = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=BoolIndexRunner(),
            enable_generation=True,
        )
        res_bool = asst_bool.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertIsNotNone(res_bool.generation_fallback_reason)
        self.assertIn("contract_violation", res_bool.generation_fallback_reason)
        self.assertIn("boolean or non-int rejected", res_bool.generation_fallback_reason)

        class OutOfRangeRunner:
            def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                return {"text": '{"style":"standard","selected_claims":[99]}'}

        asst_range = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=OutOfRangeRunner(),
            enable_generation=True,
        )
        res_range = asst_range.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertIsNotNone(res_range.generation_fallback_reason)
        self.assertIn("contract_violation", res_range.generation_fallback_reason)
        self.assertIn("out of range", res_range.generation_fallback_reason)

    def test_presentation_contract_rejects_missing_keys(self):
        """Payloads missing required declared keys must trigger contract violation fallback."""
        class MissingKeyRunner:
            def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                return {"text": '{"style":"compact"}'}

        asst = LocalAssistant(
            catalog_path=FIXTURE_CATALOG,
            llm_runner=MissingKeyRunner(),
            enable_generation=True,
        )
        res = asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
        self.assertIsNotNone(res.generation_fallback_reason)
        self.assertIn("contract_violation", res.generation_fallback_reason)
        self.assertIn("missing required declared keys", res.generation_fallback_reason)

    def test_presentation_contract_accepts_all_approved_styles(self):
        """Approved styles 'standard', 'compact', 'bullet' with valid indices must succeed without fallback."""
        for approved_style in ("compact", "bullet", "standard"):
            class ApprovedRunner:
                def __init__(self, st):
                    self.st = st
                def generate(self, prompt: str, max_tokens: int = 50, timeout_s: Optional[float] = None):
                    return {"text": f'{{"style":"{self.st}","selected_claims":[0]}}'}

            asst = LocalAssistant(
                catalog_path=FIXTURE_CATALOG,
                llm_runner=ApprovedRunner(approved_style),
                enable_generation=True,
            )
            res = asst.check("fixture-chair-oak", "dining chair with oak material under 60cm wide")
            self.assertEqual(res.action, "RECOMMEND")
            self.assertIsNone(res.generation_fallback_reason, f"Style {approved_style} should not have fallen back")
            if approved_style == "bullet":
                self.assertIn("Verdict: RECOMMEND", res.answer)
            elif approved_style == "compact":
                self.assertIn("RECOMMEND fixture-chair-oak", res.answer)
            else:
                self.assertIn("Recommended: fixture-chair-oak", res.answer)


if __name__ == "__main__":
    unittest.main()
