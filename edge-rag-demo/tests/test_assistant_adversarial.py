"""Table-driven adversarial tests for QEdge Assist Stage 04 correctness.

Tests all required adversarial scenarios:
1. red chair (avoiding substring matches in words like 'shredded' or 'reduced'; color contradiction)
2. under 60 without units (ambiguity, prevents positive decision)
3. not oak (negation: oak contradicts, metal/aluminum supports, generic wood is unknown)
4. oak versus generic wood (asymmetric taxonomy: oak certifies wood; generic wood cannot certify oak)
5. 24 inches versus 60.96cm (exact conversion without rounding; strict boundary checks)
6. strict boundary equality (< vs <=, > vs >=)
7. multiple dimensions (product dimensions vs package dimensions, multiple bounds)
8. unknown units (e.g. 'kg' for width evaluates to UNKNOWN)
9. contradictory sources (conflicting facts in store lead to ABSTAIN)
10. unsupported features (e.g. 'built-in quantum charger' preserved as unparsed clause -> ABSTAIN)
11. no constraints (query without constraints -> ABSTAIN)
12. no candidates (unmatched query -> ABSTAIN with item_id=None)
13. revisions changing a verdict (document update alters active verdict; old citation resolves original)
14. catalog-directory path CLI test (CLI accepts directory path for --catalog)
"""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

from edge_rag.cli import format_cli_output, main as cli_main, run_pipeline
from edge_rag.generator import Generator
from edge_rag.requirements import parse_query_requirements
from edge_rag.retriever import Retriever
from edge_rag.store import EvidenceStore, Fact, Principal
from edge_rag.verifier import Verifier

EDGE_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"
FIXTURES_DIR = EDGE_ROOT / "tests" / "fixtures"


class AssistantAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = Verifier(catalog_path=FIXTURE_CATALOG)
        cls.retriever = Retriever(catalog_path=FIXTURE_CATALOG)
        cls.generator = Generator()

    def test_red_chair_and_substring_avoidance(self):
        # 1. Negative match: fixture-chair-oak has color 'natural'. A query for 'red' must CONTRADICT.
        res = self.verifier.parse_requirements("red chair")
        dec = self.verifier.verify_candidate("fixture-chair-oak", res, "red chair")
        self.assertEqual(dec["status"], "CONTRADICTED")
        self.assertEqual(dec["action"], "DO_NOT_RECOMMEND")

        # 2. Substring avoidance: word 'shredded' or 'reduced' in text must not certify 'red'
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-shredded",
            Principal.system(),
            ["material: shredded wood (source: material)", "text: reduced price clearance"],
            is_public=True,
        )
        custom_v = Verifier(store=custom_store, auto_load=False)
        claims = custom_v.evaluate_requirement_against_chunks(
            res.requirements[0], "doc-shredded", custom_store.retrieve("doc-shredded", Principal.system()), "2026-10-30T00:00:00Z"
        )
        self.assertEqual(claims[0]["status"], "UNKNOWN")

    def test_under_60_without_units_abstains(self):
        res = self.verifier.parse_requirements("dining chair under 60")
        self.assertTrue(len(res.ambiguities) > 0)
        dec = self.verifier.verify_candidate("fixture-chair-oak", res, "dining chair under 60")
        self.assertEqual(dec["status"], "ABSTAIN")
        self.assertEqual(dec["action"], "ABSTAIN")
        self.assertIn("ambiguities", dec["reason"].lower())

    def test_not_oak_negation(self):
        res = self.verifier.parse_requirements("dining chair not oak")
        self.assertEqual(res.requirements[0].operator, "!=")
        self.assertEqual(res.requirements[0].required_value, "oak")

        # Oak chair contradicts 'not oak'
        dec_oak = self.verifier.verify_candidate("fixture-chair-oak", res, "dining chair not oak")
        self.assertEqual(dec_oak["status"], "CONTRADICTED")
        self.assertEqual(dec_oak["action"], "DO_NOT_RECOMMEND")

        # Aluminum chair supports 'not oak'
        dec_al = self.verifier.verify_candidate("fixture-chair-wide", res, "dining chair not oak")
        self.assertEqual(dec_al["status"], "SUPPORTED")
        self.assertEqual(dec_al["action"], "RECOMMEND")

    def test_oak_versus_generic_wood_asymmetry(self):
        # A: Query specifies general 'wood', item is 'oak' -> SUPPORTED (oak is-a wood)
        res_wood = self.verifier.parse_requirements("wooden dining chair")
        dec_wood = self.verifier.verify_candidate("fixture-chair-oak", res_wood, "wooden dining chair")
        self.assertEqual(dec_wood["status"], "SUPPORTED")
        self.assertEqual(dec_wood["action"], "RECOMMEND")

        # B: Query specifies specific 'oak', item has generic 'wood' -> UNKNOWN (wood cannot prove oak)
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-generic-wood",
            Principal.system(),
            ["material: wood (source: material)"],
            is_public=True,
            facts=[Fact(attribute="material", value="wood", status="present", source_location="material", source_revision="rev1")],
        )
        custom_v = Verifier(store=custom_store, auto_load=False)
        res_oak = self.verifier.parse_requirements("dining chair with oak material")
        dec_oak = custom_v.verify_candidate("doc-generic-wood", res_oak, "dining chair with oak material")
        self.assertEqual(dec_oak["status"], "UNKNOWN")
        self.assertEqual(dec_oak["action"], "ABSTAIN")

        # C: Query specifies 'not oak', item has generic 'wood' -> UNKNOWN (generic wood cannot prove not-oak)
        res_not_oak = self.verifier.parse_requirements("dining chair not oak")
        dec_not_oak = custom_v.verify_candidate("doc-generic-wood", res_not_oak, "dining chair not oak")
        self.assertEqual(dec_not_oak["status"], "UNKNOWN")
        self.assertEqual(dec_not_oak["action"], "ABSTAIN")

    def test_24_inches_versus_60_96_cm_and_strict_boundary(self):
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-24in",
            Principal.system(),
            ["width: 24.0 in (source: item_dimensions)"],
            is_public=True,
            facts=[Fact(attribute="width", value=24.0, unit="in", status="present", source_location="item_dimensions", source_revision="rev1")],
        )
        custom_v = Verifier(store=custom_store, auto_load=False)

        # 24 inches == exactly 60.96 cm
        # Strict < 60.96cm must be CONTRADICTED (60.96 is not < 60.96)
        res_strict = self.verifier.parse_requirements("dining chair width < 60.96cm")
        dec_strict = custom_v.verify_candidate("doc-24in", res_strict, "dining chair width < 60.96cm")
        self.assertEqual(dec_strict["status"], "CONTRADICTED")
        self.assertEqual(dec_strict["action"], "DO_NOT_RECOMMEND")

        # Inclusive <= 60.96cm must be SUPPORTED (60.96 <= 60.96)
        res_incl = self.verifier.parse_requirements("dining chair width <= 60.96cm")
        dec_incl = custom_v.verify_candidate("doc-24in", res_incl, "dining chair width <= 60.96cm")
        self.assertEqual(dec_incl["status"], "SUPPORTED")
        self.assertEqual(dec_incl["action"], "RECOMMEND")

    def test_strict_boundary_equality(self):
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-50cm",
            Principal.system(),
            ["width: 50.0 cm (source: item_dimensions)"],
            is_public=True,
            facts=[Fact(attribute="width", value=50.0, unit="cm", status="present", source_location="item_dimensions", source_revision="rev1")],
        )
        custom_v = Verifier(store=custom_store, auto_load=False)

        # < 50.0cm -> CONTRADICTED
        dec1 = custom_v.verify_candidate("doc-50cm", self.verifier.parse_requirements("chair width < 50.0cm"), "q")
        self.assertEqual(dec1["status"], "CONTRADICTED")

        # <= 50.0cm -> SUPPORTED
        dec2 = custom_v.verify_candidate("doc-50cm", self.verifier.parse_requirements("chair width <= 50.0cm"), "q")
        self.assertEqual(dec2["status"], "SUPPORTED")

        # > 50.0cm -> CONTRADICTED
        dec3 = custom_v.verify_candidate("doc-50cm", self.verifier.parse_requirements("chair width > 50.0cm"), "q")
        self.assertEqual(dec3["status"], "CONTRADICTED")

        # >= 50.0cm -> SUPPORTED
        dec4 = custom_v.verify_candidate("doc-50cm", self.verifier.parse_requirements("chair width >= 50.0cm"), "q")
        self.assertEqual(dec4["status"], "SUPPORTED")

        # == 50.0cm -> SUPPORTED
        dec5 = custom_v.verify_candidate("doc-50cm", self.verifier.parse_requirements("chair width 50.0cm"), "q")
        self.assertEqual(dec5["status"], "SUPPORTED")

    def test_multiple_dimensions_and_package_distinction(self):
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-dims",
            Principal.system(),
            [
                "width: 45.0 cm (source: item_dimensions)",
                "height: 90.0 cm (source: item_dimensions)",
                "package_width: 55.0 cm (source: package_dimensions)",
            ],
            is_public=True,
            facts=[
                Fact(attribute="width", value=45.0, unit="cm", status="present", source_location="item_dimensions", source_revision="rev1"),
                Fact(attribute="height", value=90.0, unit="cm", status="present", source_location="item_dimensions", source_revision="rev1"),
                Fact(attribute="package_width", value=55.0, unit="cm", status="present", source_location="package_dimensions", source_revision="rev1"),
            ],
        )
        custom_v = Verifier(store=custom_store, auto_load=False)

        # Both product dimensions satisfied: width <= 50cm and height <= 100cm
        res_both = custom_v.parse_requirements("chair width <= 50cm and height <= 100cm")
        dec_both = custom_v.verify_candidate("doc-dims", res_both, "q")
        self.assertEqual(dec_both["status"], "SUPPORTED")

        # One satisfied (width 45 <= 50), one contradicted (height 90 <= 80) -> CONTRADICTED
        res_fail = custom_v.parse_requirements("chair width <= 50cm and height <= 80cm")
        dec_fail = custom_v.verify_candidate("doc-dims", res_fail, "q")
        self.assertEqual(dec_fail["status"], "CONTRADICTED")

        # Package dimension requested: package_width <= 50cm (package_width is 55cm, item width is 45cm)
        # Must examine package_width (55cm) and CONTRADICT, not substitute item width (45cm)
        res_pkg = custom_v.parse_requirements("chair package width <= 50cm")
        dec_pkg = custom_v.verify_candidate("doc-dims", res_pkg, "q")
        self.assertEqual(dec_pkg["status"], "CONTRADICTED")

    def test_unknown_units_evaluate_to_unknown(self):
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-kg-width",
            Principal.system(),
            ["width: 50.0 kg (source: bad_source)"],
            is_public=True,
            facts=[Fact(attribute="width", value=50.0, unit="kg", status="present", source_location="bad_source", source_revision="rev1")],
        )
        custom_v = Verifier(store=custom_store, auto_load=False)
        res = custom_v.parse_requirements("chair width <= 60cm")
        dec = custom_v.verify_candidate("doc-kg-width", res, "q")
        self.assertEqual(dec["status"], "UNKNOWN")
        self.assertEqual(dec["action"], "ABSTAIN")

    def test_contradictory_sources_produce_abstain(self):
        custom_store = EvidenceStore(":memory:")
        custom_store.upload(
            "doc-conflict",
            Principal.system(),
            [
                "width: 45.0 cm (source: spec_a)",
                "width: 75.0 cm (source: spec_b)",
            ],
            is_public=True,
            facts=[
                Fact(attribute="width", value=45.0, unit="cm", status="present", source_location="spec_a", source_revision="rev1"),
                Fact(attribute="width", value=75.0, unit="cm", status="present", source_location="spec_b", source_revision="rev2"),
            ],
        )
        custom_v = Verifier(store=custom_store, auto_load=False)
        res = custom_v.parse_requirements("chair width <= 60cm")
        dec = custom_v.verify_candidate("doc-conflict", res, "q")
        # Conflicting evidence for same requirement must lead to ABSTAIN
        self.assertEqual(dec["status"], "ABSTAIN")
        self.assertEqual(dec["action"], "ABSTAIN")
        self.assertIn("conflicting", dec["reason"].lower())

    def test_unsupported_features_produce_abstain(self):
        # Query contains an unsupported capability: 'built-in quantum charger'
        res = self.verifier.parse_requirements("dining chair with built-in quantum charger")
        self.assertIn("built-in quantum charger", res.unparsed_clauses)
        dec = self.verifier.verify_candidate("fixture-chair-oak", res, "dining chair with built-in quantum charger")
        self.assertEqual(dec["status"], "ABSTAIN")
        self.assertEqual(dec["action"], "ABSTAIN")
        self.assertIn("unresolved", dec["reason"].lower())

    def test_no_constraints_produce_abstain(self):
        res = self.verifier.parse_requirements("comfortable dining chair")
        self.assertEqual(len(res.requirements), 0)
        dec = self.verifier.verify_candidate("fixture-chair-oak", res, "comfortable dining chair")
        self.assertEqual(dec["status"], "ABSTAIN")
        self.assertEqual(dec["action"], "ABSTAIN")
        self.assertIsNone(dec["item_id"])

    def test_no_candidates_retrieved_produces_abstain(self):
        res = self.verifier.parse_requirements("quantum flux capacitor")
        dec = self.verifier.verify_candidate(None, res, "quantum flux capacitor")
        self.assertEqual(dec["status"], "ABSTAIN")
        self.assertEqual(dec["action"], "ABSTAIN")
        self.assertIsNone(dec["item_id"])
        self.assertEqual(dec["claims"], [])

    def test_revisions_changing_verdict_and_historical_citation(self):
        custom_store = EvidenceStore(":memory:")
        # Version 1: width 75cm -> CONTRADICTED for width <= 60cm
        custom_store.upload(
            "doc-rev",
            Principal.system(),
            ["width: 75.0 cm (source: rev1)"],
            is_public=True,
            facts=[Fact(attribute="width", value=75.0, unit="cm", status="present", source_location="rev1", source_revision="v1")],
        )
        custom_v = Verifier(store=custom_store, auto_load=False)
        res = custom_v.parse_requirements("chair width <= 60cm")

        dec_v1 = custom_v.verify_candidate("doc-rev", res, "q")
        self.assertEqual(dec_v1["status"], "CONTRADICTED")
        citation_v1 = dec_v1["claims"][0]["evidence_text"]
        self.assertIn("cite:doc-rev:v1:c0", citation_v1)

        # Version 2: updated to width 50cm -> SUPPORTED
        custom_store.upload(
            "doc-rev",
            Principal.system(),
            ["width: 50.0 cm (source: rev2)"],
            is_public=True,
            expected_version=1,
            facts=[Fact(attribute="width", value=50.0, unit="cm", status="present", source_location="rev2", source_revision="v2")],
        )
        dec_v2 = custom_v.verify_candidate("doc-rev", res, "q")
        self.assertEqual(dec_v2["status"], "SUPPORTED")

        # Resolving old citation v1 from store still returns original 75.0cm fact
        chunk_v1 = custom_store.resolve_citation("cite:doc-rev:v1:c0", Principal.system())
        self.assertEqual(chunk_v1["fact"]["value"], 75.0)

    def test_catalog_directory_path_cli(self):
        # Passing directory path tests/fixtures instead of catalog.jsonl file
        result = run_pipeline(
            "I need a dining chair with oak material under 60cm wide",
            catalog_path=FIXTURES_DIR,
            k=2,
        )
        self.assertGreater(len(result["candidates"]), 0)
        self.assertGreater(len(result["verified_results"]), 0)
        top = result["verified_results"][0]
        self.assertIn("decision", top)
        self.assertEqual(top["decision"]["status"], "SUPPORTED")
        self.assertEqual(top["decision"]["action"], "RECOMMEND")

    def test_mixed_unit_ranges_and_reversed_endpoints(self):
        # fixture-chair-oak width is 48cm; fixture-chair-wide width is 75cm
        # 1. 'between 60cm and 1m wide': range [60cm, 100cm]
        res1 = self.verifier.parse_requirements("chair between 60cm and 1m wide")
        dec1_oak = self.verifier.verify_candidate("fixture-chair-oak", res1, "q")
        self.assertEqual(dec1_oak["status"], "CONTRADICTED")
        self.assertEqual(dec1_oak["action"], "DO_NOT_RECOMMEND")
        dec1_wide = self.verifier.verify_candidate("fixture-chair-wide", res1, "q")
        self.assertEqual(dec1_wide["status"], "SUPPORTED")
        self.assertEqual(dec1_wide["action"], "RECOMMEND")

        # 2. Reversed range: 'between 1m and 60cm wide' must normalize to [60cm, 100cm]
        res2 = self.verifier.parse_requirements("chair between 1m and 60cm wide")
        dec2_oak = self.verifier.verify_candidate("fixture-chair-oak", res2, "q")
        self.assertEqual(dec2_oak["status"], "CONTRADICTED")
        self.assertEqual(dec2_oak["action"], "DO_NOT_RECOMMEND")
        dec2_wide = self.verifier.verify_candidate("fixture-chair-wide", res2, "q")
        self.assertEqual(dec2_wide["status"], "SUPPORTED")
        self.assertEqual(dec2_wide["action"], "RECOMMEND")

        # 3. Units across mm, m, in, ft:
        # 'between 400mm and 0.5m wide' -> [40cm, 50cm]
        res_mm_m = self.verifier.parse_requirements("chair between 400mm and 0.5m wide")
        dec_mm_m = self.verifier.verify_candidate("fixture-chair-oak", res_mm_m, "q")
        self.assertEqual(dec_mm_m["status"], "SUPPORTED")
        self.assertEqual(dec_mm_m["action"], "RECOMMEND")

        # 'between 18in and 2ft wide' -> [45.72cm, 60.96cm]
        res_in_ft = self.verifier.parse_requirements("chair between 18in and 2ft wide")
        dec_in_ft = self.verifier.verify_candidate("fixture-chair-oak", res_in_ft, "q")
        self.assertEqual(dec_in_ft["status"], "SUPPORTED")
        self.assertEqual(dec_in_ft["action"], "RECOMMEND")

        # 4. Exact boundaries:
        # 48cm chair with 'between 48cm and 60cm wide' -> SUPPORTED
        res_bound_low = self.verifier.parse_requirements("chair between 48cm and 60cm wide")
        dec_bound_low = self.verifier.verify_candidate("fixture-chair-oak", res_bound_low, "q")
        self.assertEqual(dec_bound_low["status"], "SUPPORTED")

        # 48cm chair with 'between 40cm and 48cm wide' -> SUPPORTED
        res_bound_high = self.verifier.parse_requirements("chair between 40cm and 48cm wide")
        dec_bound_high = self.verifier.verify_candidate("fixture-chair-oak", res_bound_high, "q")
        self.assertEqual(dec_bound_high["status"], "SUPPORTED")

        # 5. Contradictory mixed-unit bounds (e.g. conjoined width > 1m and width < 60cm)
        res_contra = self.verifier.parse_requirements("chair width > 1m and width < 60cm")
        self.assertTrue(len(res_contra.ambiguities) > 0)
        dec_contra = self.verifier.verify_candidate("fixture-chair-oak", res_contra, "q")
        self.assertEqual(dec_contra["status"], "ABSTAIN")
        self.assertEqual(dec_contra["action"], "ABSTAIN")

    def test_missing_dimension_in_range_produces_abstain(self):
        # 'between 40cm and 50cm' lacks dimension; must not guess 'width'
        res = self.verifier.parse_requirements("oak chair between 40cm and 50cm")
        self.assertTrue(any("unspecified dimension" in a for a in res.ambiguities))
        # Ensure no width requirement was invented
        self.assertFalse(any(r.attribute == "width" for r in res.requirements))
        dec = self.verifier.verify_candidate("fixture-chair-oak", res, "oak chair between 40cm and 50cm")
        self.assertEqual(dec["status"], "ABSTAIN")
        self.assertEqual(dec["action"], "ABSTAIN")

    def test_subjective_qualifiers_and_unsupported_disjunctions(self):
        # 1. Subjective requirement: 'that must be comfortable'
        res_comf = self.verifier.parse_requirements("oak chair that must be comfortable")
        self.assertTrue(len(res_comf.unparsed_clauses) > 0)
        self.assertIn("comfortable", " ".join(res_comf.unparsed_clauses))
        dec_comf = self.verifier.verify_candidate("fixture-chair-oak", res_comf, "q")
        self.assertEqual(dec_comf["status"], "ABSTAIN")
        self.assertEqual(dec_comf["action"], "ABSTAIN")

        # 2. Disjunction: 'without red or blue'
        res_or = self.verifier.parse_requirements("oak chair without red or blue")
        self.assertTrue(any("unsupported disjunction" in a for a in res_or.ambiguities))
        dec_or = self.verifier.verify_candidate("fixture-chair-oak", res_or, "q")
        self.assertEqual(dec_or["status"], "ABSTAIN")
        self.assertEqual(dec_or["action"], "ABSTAIN")

        # 3. Dimensional disjunction: 'under 50cm or over 100cm wide'
        res_dim_or = self.verifier.parse_requirements("chair under 50cm or over 100cm wide")
        self.assertTrue(any("unsupported disjunction" in a for a in res_dim_or.ambiguities))
        dec_dim_or = self.verifier.verify_candidate("fixture-chair-oak", res_dim_or, "q")
        self.assertEqual(dec_dim_or["status"], "ABSTAIN")
        self.assertEqual(dec_dim_or["action"], "ABSTAIN")

    def test_brand_grammar_and_negation(self):
        # 1. 'brand FixtureCo' matching fixture-chair-oak (brand: FixtureCo)
        res1 = self.verifier.parse_requirements("oak chair brand FixtureCo")
        brand_reqs1 = [r for r in res1.requirements if r.attribute == "brand"]
        self.assertEqual(len(brand_reqs1), 1)
        self.assertEqual(brand_reqs1[0].operator, "==")
        self.assertEqual(brand_reqs1[0].required_value, "FixtureCo")
        dec1 = self.verifier.verify_candidate("fixture-chair-oak", res1, "q")
        self.assertEqual(dec1["status"], "SUPPORTED")
        self.assertEqual(dec1["action"], "RECOMMEND")
        brand_claim1 = next(c for c in dec1["claims"] if c["attribute"] == "brand")
        self.assertEqual(brand_claim1["observed_value"], "FixtureCo")

        # 2. Case-insensitivity without substring equality: 'brand fixtureco'
        res2 = self.verifier.parse_requirements("oak chair brand fixtureco")
        dec2 = self.verifier.verify_candidate("fixture-chair-oak", res2, "q")
        self.assertEqual(dec2["status"], "SUPPORTED")
        self.assertEqual(dec2["action"], "RECOMMEND")
        brand_claim2 = next(c for c in dec2["claims"] if c["attribute"] == "brand")
        self.assertEqual(brand_claim2["observed_value"], "FixtureCo")

        # 3. Substring non-match: 'brand Fixture' does NOT equal 'FixtureCo'
        res_sub = self.verifier.parse_requirements("oak chair brand Fixture")
        dec_sub = self.verifier.verify_candidate("fixture-chair-oak", res_sub, "q")
        self.assertEqual(dec_sub["status"], "CONTRADICTED")
        self.assertEqual(dec_sub["action"], "DO_NOT_RECOMMEND")
        brand_claim_sub = next(c for c in dec_sub["claims"] if c["attribute"] == "brand")
        # Observation fidelity: must retain actual observed 'FixtureCo', never rewrite to 'Fixture'
        self.assertEqual(brand_claim_sub["observed_value"], "FixtureCo")

        # 4. Negated brand with substring prefix: 'not brand Fixture' against 'FixtureCo' -> SUPPORTED
        res_neg_sub = self.verifier.parse_requirements("oak chair not brand Fixture")
        dec_neg_sub = self.verifier.verify_candidate("fixture-chair-oak", res_neg_sub, "q")
        self.assertEqual(dec_neg_sub["status"], "SUPPORTED")
        self.assertEqual(dec_neg_sub["action"], "RECOMMEND")
        brand_claim_neg_sub = next(c for c in dec_neg_sub["claims"] if c["attribute"] == "brand")
        self.assertEqual(brand_claim_neg_sub["observed_value"], "FixtureCo")

        # 5. Negated brand exact match: 'not brand FixtureCo' contradicts fixture-chair-oak
        res3 = self.verifier.parse_requirements("oak chair not brand FixtureCo")
        brand_reqs3 = [r for r in res3.requirements if r.attribute == "brand"]
        self.assertEqual(len(brand_reqs3), 1)
        self.assertEqual(brand_reqs3[0].operator, "!=")
        dec3 = self.verifier.verify_candidate("fixture-chair-oak", res3, "q")
        self.assertEqual(dec3["status"], "CONTRADICTED")
        self.assertEqual(dec3["action"], "DO_NOT_RECOMMEND")
        brand_claim3 = next(c for c in dec3["claims"] if c["attribute"] == "brand")
        self.assertEqual(brand_claim3["observed_value"], "FixtureCo")

        # 6. Multiword brand: 'made by Herman Miller'
        res4 = self.verifier.parse_requirements("desk lamp made by Herman Miller")
        brand_reqs4 = [r for r in res4.requirements if r.attribute == "brand"]
        self.assertEqual(len(brand_reqs4), 1)
        self.assertEqual(brand_reqs4[0].required_value, "Herman Miller")

        # 7. Non-brand query containing 'brand': 'brand new oak chair'
        res5 = self.verifier.parse_requirements("brand new oak chair")
        brand_reqs5 = [r for r in res5.requirements if r.attribute == "brand"]
        self.assertEqual(len(brand_reqs5), 0, "Should not parse 'new' as a brand name")
        # 'brand new' is preserved as unparsed clause
        self.assertTrue(len(res5.unparsed_clauses) > 0)

    def test_structured_evidence_status_and_non_present_facts(self):
        # 1. Non-present numeric fact with status="unknown"
        store = EvidenceStore(":memory:")
        store.upload(
            "doc-unknown-width",
            Principal.system(),
            ["width: 48.0 cm (source: test)"],
            is_public=True,
            item_id="doc-unknown-width",
            variant_id="doc-unknown-width",
            facts=[
                Fact(
                    attribute="width",
                    value=48.0,
                    unit="cm",
                    status="unknown",
                    source_location="test",
                    source_revision="r1",
                )
            ],
        )
        custom_v = Verifier(store=store, auto_load=False)
        res_width = custom_v.parse_requirements("chair width < 50cm")
        dec_unk = custom_v.verify_candidate("doc-unknown-width", res_width, "q")
        self.assertEqual(dec_unk["status"], "UNKNOWN")
        self.assertEqual(dec_unk["action"], "ABSTAIN")
        claim_unk = dec_unk["claims"][0]
        self.assertIsNone(claim_unk["observed_value"])
        self.assertEqual(claim_unk["status"], "UNKNOWN")

        # 2. Non-present numeric fact with status="missing"
        store.upload(
            "doc-missing-width",
            Principal.system(),
            ["width: 48.0 cm (source: test)"],
            is_public=True,
            item_id="doc-missing-width",
            variant_id="doc-missing-width",
            facts=[
                Fact(
                    attribute="width",
                    value=48.0,
                    unit="cm",
                    status="missing",
                    source_location="test",
                    source_revision="r1",
                )
            ],
        )
        dec_miss = custom_v.verify_candidate("doc-missing-width", res_width, "q")
        self.assertEqual(dec_miss["status"], "UNKNOWN")
        self.assertEqual(dec_miss["action"], "ABSTAIN")

        # 3. Text fallback must not bypass structured non-present status
        store.upload(
            "doc-bypass-attempt",
            Principal.system(),
            ["width: 45.0 cm (source: display_text)"],
            is_public=True,
            item_id="doc-bypass-attempt",
            variant_id="doc-bypass-attempt",
            facts=[
                Fact(
                    attribute="width",
                    value=None,
                    unit=None,
                    status="missing",
                    source_location="spec",
                    source_revision="r1",
                )
            ],
        )
        dec_bypass = custom_v.verify_candidate("doc-bypass-attempt", res_width, "q")
        self.assertEqual(dec_bypass["status"], "UNKNOWN")
        self.assertEqual(dec_bypass["action"], "ABSTAIN")
        self.assertIsNone(dec_bypass["claims"][0]["observed_value"])

        # 4. Non-present categorical fact: status="missing" on material
        store.upload(
            "doc-missing-mat",
            Principal.system(),
            ["material: oak (source: test)"],
            is_public=True,
            item_id="doc-missing-mat",
            variant_id="doc-missing-mat",
            facts=[
                Fact(
                    attribute="material",
                    value="oak",
                    status="missing",
                    source_location="test",
                    source_revision="r1",
                )
            ],
        )
        res_mat = custom_v.parse_requirements("chair with oak material")
        dec_mat_miss = custom_v.verify_candidate("doc-missing-mat", res_mat, "q")
        self.assertEqual(dec_mat_miss["status"], "UNKNOWN")
        self.assertEqual(dec_mat_miss["action"], "ABSTAIN")

        # 5. Negated predicate against non-present fact must ABSTAIN, not SUPPORT
        res_not_oak = custom_v.parse_requirements("chair not oak")
        dec_neg_miss = custom_v.verify_candidate("doc-missing-mat", res_not_oak, "q")
        self.assertEqual(dec_neg_miss["status"], "UNKNOWN")
        self.assertEqual(dec_neg_miss["action"], "ABSTAIN")

        # 6. Present-fact control: status="present" correctly supports
        store.upload(
            "doc-present-control",
            Principal.system(),
            ["width: 48.0 cm (source: spec)"],
            is_public=True,
            item_id="doc-present-control",
            variant_id="doc-present-control",
            facts=[
                Fact(
                    attribute="width",
                    value=48.0,
                    unit="cm",
                    status="present",
                    source_location="spec",
                    source_revision="r1",
                )
            ],
        )
        dec_ctrl = custom_v.verify_candidate("doc-present-control", res_width, "q")
        self.assertEqual(dec_ctrl["status"], "SUPPORTED")
        self.assertEqual(dec_ctrl["action"], "RECOMMEND")
        self.assertEqual(dec_ctrl["claims"][0]["observed_value"], 48.0)

        # 7. Categorical taxonomy observation fidelity: oak preserves observed_value="oak"
        res_wooden = self.verifier.parse_requirements("wooden dining chair")
        dec_wooden = self.verifier.verify_candidate("fixture-chair-oak", res_wooden, "q")
        self.assertEqual(dec_wooden["status"], "SUPPORTED")
        self.assertEqual(dec_wooden["action"], "RECOMMEND")
        mat_claim = next(c for c in dec_wooden["claims"] if c["attribute"] == "material")
        self.assertEqual(mat_claim["observed_value"], "oak")
        self.assertEqual(mat_claim["required_value"], "wood")

    def test_decision_answer_boundary_rendering(self):
        # 1. Pipeline result produces verified_results with decisions
        res_supp = run_pipeline("dining chair with oak material under 60cm wide", catalog_path=FIXTURE_CATALOG, k=1)
        top_supp = res_supp["verified_results"][0]
        self.assertEqual(top_supp["decision"]["action"], "RECOMMEND")
        self.assertEqual(top_supp["decision"]["status"], "SUPPORTED")
        # Render answer via pipeline and generator
        self.assertIn("Recommended: Fixture Oak Dining Chair", res_supp["generated_response"])
        self.assertIn("cite:fixture-chair-oak-01", res_supp["generated_response"])

        # 2. Contradicted decision renders 'Do not recommend'
        res_contra = run_pipeline("dining chair with oak material between 60cm and 1m wide", catalog_path=FIXTURE_CATALOG, k=1)
        top_contra = res_contra["verified_results"][0]
        self.assertEqual(top_contra["decision"]["action"], "DO_NOT_RECOMMEND")
        self.assertEqual(top_contra["decision"]["status"], "CONTRADICTED")
        self.assertIn("Do not recommend: Fixture Oak Dining Chair", res_contra["generated_response"])
        self.assertIn("Contradicted requirements:", res_contra["generated_response"])

        # 3. Abstain decision renders 'Abstain for'
        res_abstain = run_pipeline("dining chair that must be comfortable", catalog_path=FIXTURE_CATALOG, k=1)
        top_abstain = res_abstain["verified_results"][0]
        self.assertEqual(top_abstain["decision"]["action"], "ABSTAIN")
        self.assertIn("Abstain for Fixture Oak Dining Chair", res_abstain["generated_response"])
        self.assertIn("unresolved clauses", res_abstain["generated_response"])

        # 4. Formatted CLI output contains action, status, and citations
        cli_out = format_cli_output(res_supp)
        self.assertIn("[Decision: SUPPORTED / RECOMMEND]", cli_out)
        self.assertIn("cite:fixture-chair-oak-01", cli_out)

    def test_semantic_cross_layer_decision_boundary(self):
        """Cross-layer table-driven tests for brand, color, material, and evidence status.

        Verifies:
        - Exact and case/whitespace-normalized brands, distinct brands, containment.
        - Single and compound colors, positive and negative predicates.
        - Material subtypes, supertypes, siblings, and unrecognized compositions.
        - Present and non-present evidence.
        - Source-observation fidelity, intended predicate status, and successful contract validation.
        """
        cases = [
            # --- Brands ---
            {
                "name": "brand_exact",
                "attr": "brand",
                "val": "FixtureCo",
                "status_ev": "present",
                "query": "chair brand FixtureCo",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "FixtureCo",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "brand_whitespace_case",
                "attr": "brand",
                "val": "  fixtureco  ",
                "status_ev": "present",
                "query": "chair brand FixtureCo",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "  fixtureco  ",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "brand_distinct",
                "attr": "brand",
                "val": "AcmeCorp",
                "status_ev": "present",
                "query": "chair brand FixtureCo",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": "AcmeCorp",
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            {
                "name": "brand_substring_equality_rejected",
                "attr": "brand",
                "val": "FixtureCo",
                "status_ev": "present",
                "query": "chair brand Fixture",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": "FixtureCo",
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            {
                "name": "brand_negation_distinct",
                "attr": "brand",
                "val": "FixtureCo",
                "status_ev": "present",
                "query": "chair not brand Fixture",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "FixtureCo",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            # --- Colors ---
            {
                "name": "color_single_exact",
                "attr": "color",
                "val": "red",
                "status_ev": "present",
                "query": "red chair",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "red",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "color_compound_observation_positive",
                "attr": "color",
                "val": "dark red",
                "status_ev": "present",
                "query": "red chair",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "dark red",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "color_compound_observation_negative",
                "attr": "color",
                "val": "dark red",
                "status_ev": "present",
                "query": "chair not red",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": "dark red",
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            {
                "name": "color_disjoint_positive",
                "attr": "color",
                "val": "dark blue",
                "status_ev": "present",
                "query": "red chair",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": "dark blue",
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            {
                "name": "color_disjoint_negative",
                "attr": "color",
                "val": "blue",
                "status_ev": "present",
                "query": "chair not red",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "blue",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "color_negated_observation",
                "attr": "color",
                "val": "not red",
                "status_ev": "present",
                "query": "red chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "color_negated_observation_negative_query",
                "attr": "color",
                "val": "not red",
                "status_ev": "present",
                "query": "chair not red",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "color_alternative_observation",
                "attr": "color",
                "val": "red or blue",
                "status_ev": "present",
                "query": "red chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "color_uncertain_observation",
                "attr": "color",
                "val": "maybe red",
                "status_ev": "present",
                "query": "red chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "color_unrecognized_modifier",
                "attr": "color",
                "val": "sparkly red",
                "status_ev": "present",
                "query": "red chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "color_generic_to_specific_asymmetry",
                "attr": "color",
                "val": "red",
                "status_ev": "present",
                "query": "dark red chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "color_distinct_shades_same_family",
                "attr": "color",
                "val": "light red",
                "status_ev": "present",
                "query": "dark red chair",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": "light red",
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            # --- Numeric Exact Equality ---
            {
                "name": "numeric_exact_equality",
                "attr": "width",
                "val": 48.0,
                "unit": "cm",
                "status_ev": "present",
                "query": "chair width = 48cm",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": 48.0,
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "numeric_inside_former_tolerance_equality",
                "attr": "width",
                "val": 48.0000005,
                "unit": "cm",
                "status_ev": "present",
                "query": "chair width = 48cm",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": 48.0000005,
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            {
                "name": "numeric_inside_former_tolerance_inequality",
                "attr": "width",
                "val": 48.0000005,
                "unit": "cm",
                "status_ev": "present",
                "query": "chair width != 48cm",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": 48.0000005,
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "numeric_outside_former_tolerance_equality",
                "attr": "width",
                "val": 48.000002,
                "unit": "cm",
                "status_ev": "present",
                "query": "chair width = 48cm",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": 48.000002,
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            {
                "name": "numeric_unit_conversion_exact",
                "attr": "width",
                "val": 24.0,
                "unit": "in",
                "status_ev": "present",
                "query": "chair width = 60.96cm",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": 60.96,
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            # --- Materials ---
            {
                "name": "material_subtype_satisfies_supertype",
                "attr": "material",
                "val": "oak",
                "status_ev": "present",
                "query": "wooden chair",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "oak",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "material_supertype_cannot_certify_subtype",
                "attr": "material",
                "val": "wood",
                "status_ev": "present",
                "query": "oak chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "material_siblings_contradict",
                "attr": "material",
                "val": "beech",
                "status_ev": "present",
                "query": "oak chair",
                "exp_claim_status": "CONTRADICTED",
                "exp_obs": "beech",
                "exp_dec_status": "CONTRADICTED",
                "exp_dec_action": "DO_NOT_RECOMMEND",
            },
            {
                "name": "material_sibling_satisfies_negation",
                "attr": "material",
                "val": "beech",
                "status_ev": "present",
                "query": "chair not oak",
                "exp_claim_status": "SUPPORTED",
                "exp_obs": "beech",
                "exp_dec_status": "SUPPORTED",
                "exp_dec_action": "RECOMMEND",
            },
            {
                "name": "material_unrecognized_composition_conservative",
                "attr": "material",
                "val": "oak veneer",
                "status_ev": "present",
                "query": "oak chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "material_unrecognized_composition_negation",
                "attr": "material",
                "val": "oak veneer",
                "status_ev": "present",
                "query": "chair not oak",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            # --- Non-present evidence ---
            {
                "name": "evidence_non_present_unknown",
                "attr": "color",
                "val": "red",
                "status_ev": "unknown",
                "query": "red chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
            {
                "name": "evidence_non_present_missing",
                "attr": "material",
                "val": "oak",
                "status_ev": "missing",
                "query": "oak chair",
                "exp_claim_status": "UNKNOWN",
                "exp_obs": None,
                "exp_dec_status": "UNKNOWN",
                "exp_dec_action": "ABSTAIN",
            },
        ]

        import tempfile
        from edge_rag.store import EvidenceStore, Fact
        from qedge_common.schema import validate

        for c in cases:
            with self.subTest(case=c["name"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    store = EvidenceStore(f"{tmpdir}/store.db")
                    item_id = f"item-{c['name']}"
                    fact = Fact(
                        attribute=c["attr"],
                        value=c["val"],
                        unit=c.get("unit"),
                        status=c["status_ev"],
                        source_location="probe",
                        source_revision="r1",
                    )
                    unit_str = f" {c['unit']}" if c.get("unit") else ""
                    store.upload(item_id, "alice", [f"{c['attr']}: {c['val']}{unit_str}"], facts=[fact])
                    v = Verifier(store=store, auto_load=False)
                    pr = v.parse_requirements(c["query"])
                    dec = v.verify_candidate(item_id, pr, c["query"])

                    # 1. Assert successful contract validation
                    validate("decision", dec)

                    # 2. Assert decision action and status
                    self.assertEqual(dec["status"], c["exp_dec_status"], f"Decision status mismatch in {c['name']}")
                    self.assertEqual(dec["action"], c["exp_dec_action"], f"Decision action mismatch in {c['name']}")

                    # 3. Assert claim status and observation fidelity
                    self.assertTrue(len(dec["claims"]) > 0, f"No claims emitted in {c['name']}")
                    target_claim = dec["claims"][0]
                    self.assertEqual(target_claim["status"], c["exp_claim_status"], f"Claim status mismatch in {c['name']}")
                    self.assertEqual(target_claim["observed_value"], c["exp_obs"], f"Observation fidelity mismatch in {c['name']}")


if __name__ == "__main__":
    unittest.main()
