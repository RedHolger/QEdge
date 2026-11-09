"""Post-repair acceptance probe for Stage 04 (04-assistant-correctness).

Validates:
1. Hidden chair/seat defaults are removed.
2. Bare numbers without units log ambiguity and abstain (no guessed 'cm' or 'width').
3. Generic wood fact does not certify oak requirement (asymmetric taxonomy: UNKNOWN).
4. Oak fact certifies wood requirement (asymmetric taxonomy: SUPPORTED).
5. Negation handling ('not oak' on oak is CONTRADICTED, on aluminum is SUPPORTED).
6. Strict boundary equality (< 60.96cm vs 24in is CONTRADICTED; <= 60.96cm is SUPPORTED).
7. Conflicting source facts in store lead to ABSTAIN decision.
8. Unsupported capability clauses ('built-in quantum charger') lead to ABSTAIN decision.
9. Catalog directory path handled without error in Verifier.load_catalog and CLI.
10. Synthetic benchmark subprocess removed from Generator.
11. 80-case decision program (40 dev / 40 held-out) with zero false positive decisions on adverse cases.
12. All decisions pass contract v2 semantic schema validation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

EDGE_ROOT = Path(__file__).resolve().parents[4] / "edge-rag-demo"
sys.path.insert(0, str(EDGE_ROOT / "src"))

from edge_rag.cli import run_pipeline
from edge_rag.generator import Generator
from edge_rag.store import EvidenceStore, Fact, Principal
from edge_rag.verifier import Verifier
from qedge_common.schema import validate as validate_contract

FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"
FIXTURES_DIR = EDGE_ROOT / "tests" / "fixtures"
CASES_FILE = EDGE_ROOT / "tests" / "fixtures" / "decision_cases_80.json"


def check_1_no_hidden_chair_defaults():
    print("Check 1: No hidden chair/seat defaults in run_pipeline...")
    result = run_pipeline("comfy chair", catalog_path=FIXTURE_CATALOG, k=1)
    parse_result = result["parse_result"]
    assert len(parse_result.requirements) == 0, f"Expected 0 requirements for bare query, got {parse_result.requirements}"
    top = result["verified_results"][0]
    assert top["decision"]["status"] == "ABSTAIN", f"Expected ABSTAIN for unconstrained query, got {top['decision']['status']}"
    print("  PASS: No hidden defaults; bare query safely abstains.")


def check_2_bare_numbers_no_guessed_units():
    print("Check 2: Bare numbers without units do not guess 'cm'...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    res = v.parse_requirements("I want a chair under 60")
    assert len(res.requirements) == 0, "Bare number must not create numeric predicate"
    assert len(res.ambiguities) > 0, "Bare number must be recorded in ambiguities"
    dec = v.verify_candidate("fixture-chair-oak", res, "I want a chair under 60")
    assert dec["status"] == "ABSTAIN", f"Expected ABSTAIN, got {dec['status']}"
    print("  PASS: Bare number recorded as ambiguity and produces ABSTAIN.")


def check_3_asymmetric_taxonomy():
    print("Check 3: Asymmetric taxonomy (wood cannot prove oak; oak proves wood)...")
    store = EvidenceStore(":memory:")
    store.upload(
        "doc-wood",
        Principal.system(),
        ["material: wood (source: material)"],
        is_public=True,
        facts=[Fact(attribute="material", value="wood", status="present", source_location="material", source_revision="rev1")],
    )
    v = Verifier(store=store, auto_load=False)

    # Generic wood cannot certify oak -> UNKNOWN
    res_oak = v.parse_requirements("chair with oak material")
    dec_oak = v.verify_candidate("doc-wood", res_oak, "chair with oak material")
    assert dec_oak["status"] == "UNKNOWN", f"Expected UNKNOWN for generic wood vs oak, got {dec_oak['status']}"

    # Oak certifies wood -> SUPPORTED
    store.upload(
        "doc-oak",
        Principal.system(),
        ["material: oak (source: material)"],
        is_public=True,
        facts=[Fact(attribute="material", value="oak", status="present", source_location="material", source_revision="rev1")],
    )
    res_wood = v.parse_requirements("wooden chair")
    dec_wood = v.verify_candidate("doc-oak", res_wood, "wooden chair")
    assert dec_wood["status"] == "SUPPORTED", f"Expected SUPPORTED for oak vs wood, got {dec_wood['status']}"
    print("  PASS: Asymmetric taxonomy correctly distinguishes subtype from supertype.")


def check_4_negation_handling():
    print("Check 4: Negation handling ('not oak')...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    res = v.parse_requirements("chair not oak")
    dec_oak = v.verify_candidate("fixture-chair-oak", res, "chair not oak")
    assert dec_oak["status"] == "CONTRADICTED", f"Oak chair must contradict 'not oak', got {dec_oak['status']}"
    assert dec_oak["action"] == "DO_NOT_RECOMMEND"

    dec_al = v.verify_candidate("fixture-chair-wide", res, "chair not oak")
    assert dec_al["status"] == "SUPPORTED", f"Aluminum chair must support 'not oak', got {dec_al['status']}"
    assert dec_al["action"] == "RECOMMEND"
    print("  PASS: Negation correctly contradicts matching item and recommends disjoint item.")


def check_5_strict_boundary_equality():
    print("Check 5: Strict boundary equality (24 in == 60.96 cm)...")
    store = EvidenceStore(":memory:")
    store.upload(
        "doc-24in",
        Principal.system(),
        ["width: 24.0 in (source: item_dimensions)"],
        is_public=True,
        facts=[Fact(attribute="width", value=24.0, unit="in", status="present", source_location="item_dimensions", source_revision="rev1")],
    )
    v = Verifier(store=store, auto_load=False)

    # < 60.96cm is False (60.96 is not strictly less than 60.96)
    dec_lt = v.verify_candidate("doc-24in", v.parse_requirements("chair width < 60.96cm"), "q")
    assert dec_lt["status"] == "CONTRADICTED", f"Expected CONTRADICTED for strict < 60.96cm, got {dec_lt['status']}"

    # <= 60.96cm is True
    dec_lte = v.verify_candidate("doc-24in", v.parse_requirements("chair width <= 60.96cm"), "q")
    assert dec_lte["status"] == "SUPPORTED", f"Expected SUPPORTED for <= 60.96cm, got {dec_lte['status']}"
    print("  PASS: Strict (<) vs inclusive (<=) exact boundary comparison verified without rounding.")


def check_6_conflicting_sources_abstain():
    print("Check 6: Conflicting source facts produce ABSTAIN...")
    store = EvidenceStore(":memory:")
    store.upload(
        "doc-conflict",
        Principal.system(),
        ["width: 45.0 cm (source: spec_a)", "width: 75.0 cm (source: spec_b)"],
        is_public=True,
        facts=[
            Fact(attribute="width", value=45.0, unit="cm", status="present", source_location="spec_a", source_revision="rev1"),
            Fact(attribute="width", value=75.0, unit="cm", status="present", source_location="spec_b", source_revision="rev2"),
        ],
    )
    v = Verifier(store=store, auto_load=False)
    dec = v.verify_candidate("doc-conflict", v.parse_requirements("chair width <= 60cm"), "q")
    assert dec["status"] == "ABSTAIN", f"Expected ABSTAIN for conflicting sources, got {dec['status']}"
    assert dec["action"] == "ABSTAIN"
    print("  PASS: Conflicting evidence claims correctly produce ABSTAIN.")


def check_7_unsupported_features_abstain():
    print("Check 7: Unsupported capability clauses produce ABSTAIN...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    res = v.parse_requirements("dining chair with built-in quantum charger")
    assert "built-in quantum charger" in res.unparsed_clauses
    dec = v.verify_candidate("fixture-chair-oak", res, "dining chair with built-in quantum charger")
    assert dec["status"] == "ABSTAIN", f"Expected ABSTAIN for unsupported feature, got {dec['status']}"
    print("  PASS: Unsupported feature retained in unparsed_clauses, preventing positive recommendation.")


def check_8_catalog_directory_path():
    print("Check 8: Catalog directory path supported in Verifier.load_catalog...")
    v = Verifier(auto_load=False)
    count = v.load_catalog(FIXTURES_DIR)
    assert count > 0, f"Expected > 0 items loaded from directory, got {count}"
    print(f"  PASS: Loaded {count} catalog items from directory path.")


def check_9_generator_no_synthetic_qnn():
    print("Check 9: Generator contains no synthetic QNN benchmark execution...")
    gen = Generator()
    metrics = gen.get_benchmark_metrics()
    assert metrics["policy"] == "not measured"
    assert not hasattr(gen, "_benchmark_binary"), "Generator must not search for or execute benchmark binary"
    footer = gen.format_footer()
    assert "QNN/NPU status: unverified" in footer
    print("  PASS: Synthetic QNN benchmark removed from Generator.")


def check_10_80_case_testbed_zero_false_positives():
    print("Check 10: 80-case decision testbed evaluation...")
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) == 80, f"Expected 80 cases, got {len(cases)}"
    v = Verifier(catalog_path=FIXTURE_CATALOG)

    false_positives = []
    dev_correct = 0
    held_correct = 0

    for case in cases:
        q = case["query"]
        item = case["item_id"]
        res = v.parse_requirements(q)
        dec = v.verify_candidate(item, res, q)

        # Enforce contract validation
        validate_contract("decision", dec, version="2")

        st = dec["status"]
        act = dec["action"]

        if case["expected_status"] in ("CONTRADICTED", "UNKNOWN", "ABSTAIN"):
            if st == "SUPPORTED" or act == "RECOMMEND":
                false_positives.append((case["case_id"], q, st, act))

        if st == case["expected_status"] and act == case["expected_action"]:
            if case["split"] == "development":
                dev_correct += 1
            else:
                held_correct += 1

    assert len(false_positives) == 0, f"Zero false positives violated: {false_positives}"
    assert dev_correct == 40, f"Development set accuracy {dev_correct}/40"
    assert held_correct == 40, f"Held-out set accuracy {held_correct}/40"

    # Enforce all 80 cases and all 40 held-out cases have strictly unique (query, item_id) pairs
    held_pairs = [(c["query"], c["item_id"]) for c in cases if c["split"] == "held_out"]
    assert len(set(held_pairs)) == 40, f"Expected 40 unique held-out pairs, got {len(set(held_pairs))}"
    all_pairs = [(c["query"], c["item_id"]) for c in cases]
    assert len(set(all_pairs)) == 80, f"Expected 80 unique pairs, got {len(set(all_pairs))}"
    print(f"  PASS: 80 cases evaluated (40 dev correct, 40 held-out correct, 80 strictly unique pairs, 0 false positives).")


def check_11_mixed_unit_ranges_and_reversed():
    print("Check 11: Mixed-unit range parsing and reversed endpoints...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    # 60cm to 1m on 48cm chair
    res1 = v.parse_requirements("oak chair between 60cm and 1m wide")
    dec1 = v.verify_candidate("fixture-chair-oak", res1, "oak chair between 60cm and 1m wide")
    assert dec1["status"] == "CONTRADICTED", f"Expected CONTRADICTED, got {dec1['status']}"
    assert dec1["action"] == "DO_NOT_RECOMMEND"

    # Reversed: 1m to 60cm wide
    res2 = v.parse_requirements("oak chair between 1m and 60cm wide")
    dec2 = v.verify_candidate("fixture-chair-oak", res2, "oak chair between 1m and 60cm wide")
    assert dec2["status"] == "CONTRADICTED", f"Expected CONTRADICTED, got {dec2['status']}"
    assert dec2["action"] == "DO_NOT_RECOMMEND"
    print("  PASS: Mixed-unit range endpoints normalized before comparison without raw value reordering.")


def check_12_brand_grammar_and_negative():
    print("Check 12: Brand grammar, multiword brands, negation, and complete identity...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    # brand FixtureCo
    res1 = v.parse_requirements("oak chair brand FixtureCo")
    dec1 = v.verify_candidate("fixture-chair-oak", res1, "oak chair brand FixtureCo")
    assert dec1["status"] == "SUPPORTED", f"Expected SUPPORTED, got {dec1['status']}"
    assert dec1["action"] == "RECOMMEND"
    brand_c1 = next(c for c in dec1["claims"] if c["attribute"] == "brand")
    assert brand_c1["observed_value"] == "FixtureCo", f"Expected observed_value 'FixtureCo', got {brand_c1['observed_value']!r}"

    # Complete normalized identity: brand Fixture does NOT match FixtureCo
    res_sub = v.parse_requirements("oak chair brand Fixture")
    dec_sub = v.verify_candidate("fixture-chair-oak", res_sub, "oak chair brand Fixture")
    assert dec_sub["status"] == "CONTRADICTED", f"Expected CONTRADICTED, got {dec_sub['status']}"
    assert dec_sub["action"] == "DO_NOT_RECOMMEND"
    brand_c_sub = next(c for c in dec_sub["claims"] if c["attribute"] == "brand")
    assert brand_c_sub["observed_value"] == "FixtureCo", f"Expected observed_value 'FixtureCo', got {brand_c_sub['observed_value']!r}"

    # Negated brand with substring prefix: not brand Fixture against FixtureCo -> SUPPORTED
    res_neg_sub = v.parse_requirements("oak chair not brand Fixture")
    dec_neg_sub = v.verify_candidate("fixture-chair-oak", res_neg_sub, "oak chair not brand Fixture")
    assert dec_neg_sub["status"] == "SUPPORTED", f"Expected SUPPORTED, got {dec_neg_sub['status']}"
    assert dec_neg_sub["action"] == "RECOMMEND"
    brand_c_neg_sub = next(c for c in dec_neg_sub["claims"] if c["attribute"] == "brand")
    assert brand_c_neg_sub["observed_value"] == "FixtureCo", f"Expected observed_value 'FixtureCo', got {brand_c_neg_sub['observed_value']!r}"

    # not brand FixtureCo
    res2 = v.parse_requirements("oak chair not brand FixtureCo")
    dec2 = v.verify_candidate("fixture-chair-oak", res2, "oak chair not brand FixtureCo")
    assert dec2["status"] == "CONTRADICTED", f"Expected CONTRADICTED, got {dec2['status']}"
    assert dec2["action"] == "DO_NOT_RECOMMEND"

    # non-brand idiom: brand new
    res3 = v.parse_requirements("brand new oak chair")
    brand_reqs = [r for r in res3.requirements if r.attribute == "brand"]
    assert len(brand_reqs) == 0, f"Expected 0 brand requirements for 'brand new', got {brand_reqs}"
    assert len(res3.unparsed_clauses) > 0, "Expected 'brand new' in unparsed clauses"
    print("  PASS: Brand grammar accurately evaluates exact, negated, and non-brand queries without substring equality.")


def check_13_answer_boundary_fidelity():
    print("Check 13: Rendered answer reflects validated decisions and citations...")
    res = run_pipeline("dining chair with oak material under 60cm wide", catalog_path=FIXTURE_CATALOG, k=1)
    ans = res["generated_response"]
    assert "Recommended: Fixture Oak Dining Chair" in ans, f"Unexpected response: {ans}"
    assert "cite:fixture-chair-oak-01" in ans, f"Missing citation in response: {ans}"
    print("  PASS: Answer boundary accurately reflects decision status, action, and citations.")


def check_14_structured_evidence_status():
    print("Check 14: Structured evidence status requirements...")
    store = EvidenceStore(":memory:")
    # 1. Non-present status="unknown"
    store.upload(
        "item_unk",
        Principal.system(),
        ["width: 48 cm"],
        is_public=True,
        item_id="item_unk",
        variant_id="item_unk",
        facts=[Fact(attribute="width", value=48.0, unit="cm", status="unknown", source_location="test", source_revision="r1")],
    )
    v = Verifier(store=store, auto_load=False)
    res_w = v.parse_requirements("chair width < 50cm")
    dec_unk = v.verify_candidate("item_unk", res_w, "chair width < 50cm")
    assert dec_unk["status"] == "UNKNOWN", f"Expected UNKNOWN for status=unknown, got {dec_unk['status']}"
    assert dec_unk["action"] == "ABSTAIN"
    assert dec_unk["claims"][0]["observed_value"] is None

    # 2. Non-present status="missing"
    store.upload(
        "item_miss",
        Principal.system(),
        ["width: 48 cm"],
        is_public=True,
        item_id="item_miss",
        variant_id="item_miss",
        facts=[Fact(attribute="width", value=48.0, unit="cm", status="missing", source_location="test", source_revision="r1")],
    )
    dec_miss = v.verify_candidate("item_miss", res_w, "chair width < 50cm")
    assert dec_miss["status"] == "UNKNOWN", f"Expected UNKNOWN for status=missing, got {dec_miss['status']}"
    assert dec_miss["action"] == "ABSTAIN"

    # 3. Present-fact control
    store.upload(
        "item_pres",
        Principal.system(),
        ["width: 48 cm"],
        is_public=True,
        item_id="item_pres",
        variant_id="item_pres",
        facts=[Fact(attribute="width", value=48.0, unit="cm", status="present", source_location="test", source_revision="r1")],
    )
    dec_pres = v.verify_candidate("item_pres", res_w, "chair width < 50cm")
    assert dec_pres["status"] == "SUPPORTED", f"Expected SUPPORTED for status=present, got {dec_pres['status']}"
    assert dec_pres["action"] == "RECOMMEND"
    assert dec_pres["claims"][0]["observed_value"] == 48.0

    # 4. Observation fidelity in taxonomy: oak preserves observed_value="oak"
    v_cat = Verifier(catalog_path=FIXTURE_CATALOG)
    res_wood = v_cat.parse_requirements("wooden dining chair")
    dec_wood = v_cat.verify_candidate("fixture-chair-oak", res_wood, "wooden dining chair")
    assert dec_wood["status"] == "SUPPORTED"
    mat_claim = next(c for c in dec_wood["claims"] if c["attribute"] == "material")
    assert mat_claim["observed_value"] == "oak", f"Expected observed_value 'oak', got {mat_claim['observed_value']!r}"
    assert mat_claim["required_value"] == "wood"
    print("  PASS: Eligible evidence status enforced; observation fidelity preserved across derivations.")


def check_15_semantic_boundary_reconciliation():
    print("Check 15: Cross-layer semantic boundary reconciliation (brand whitespace, compound color, material composition)...")
    store = EvidenceStore(":memory:")
    # 1. Color: dark red satisfies red chair with observation fidelity
    store.upload(
        "item_color_dark_red",
        Principal.system(),
        ["color: dark red"],
        is_public=True,
        facts=[Fact(attribute="color", value="dark red", status="present", source_location="test", source_revision="r1")],
    )
    v = Verifier(store=store, auto_load=False)
    res_red = v.parse_requirements("red chair")
    dec_red = v.verify_candidate("item_color_dark_red", res_red, "red chair")
    validate_contract("decision", dec_red)
    assert dec_red["status"] == "SUPPORTED"
    assert dec_red["action"] == "RECOMMEND"
    assert dec_red["claims"][0]["observed_value"] == "dark red"

    # 2. Negative color: dark red contradicts chair not red
    res_not_red = v.parse_requirements("chair not red")
    dec_not_red = v.verify_candidate("item_color_dark_red", res_not_red, "chair not red")
    validate_contract("decision", dec_not_red)
    assert dec_not_red["status"] == "CONTRADICTED"
    assert dec_not_red["action"] == "DO_NOT_RECOMMEND"
    assert dec_not_red["claims"][0]["observed_value"] == "dark red"

    # 3. Material: oak veneer conservative evaluation (UNKNOWN / ABSTAIN)
    store.upload(
        "item_mat_veneer",
        Principal.system(),
        ["material: oak veneer"],
        is_public=True,
        facts=[Fact(attribute="material", value="oak veneer", status="present", source_location="test", source_revision="r1")],
    )
    res_oak = v.parse_requirements("oak chair")
    dec_oak = v.verify_candidate("item_mat_veneer", res_oak, "oak chair")
    validate_contract("decision", dec_oak)
    assert dec_oak["status"] == "UNKNOWN"
    assert dec_oak["action"] == "ABSTAIN"
    assert dec_oak["claims"][0]["observed_value"] is None

    # 4. Material: oak veneer cannot prove not oak (UNKNOWN / ABSTAIN)
    res_not_oak = v.parse_requirements("chair not oak")
    dec_not_oak = v.verify_candidate("item_mat_veneer", res_not_oak, "chair not oak")
    validate_contract("decision", dec_not_oak)
    assert dec_not_oak["status"] == "UNKNOWN"
    assert dec_not_oak["action"] == "ABSTAIN"

    # 5. Brand: whitespace normalization with observation fidelity
    store.upload(
        "item_brand_ws",
        Principal.system(),
        ["brand:   FixtureCo  "],
        is_public=True,
        facts=[Fact(attribute="brand", value="  FixtureCo  ", status="present", source_location="test", source_revision="r1")],
    )
    res_brand = v.parse_requirements("chair brand FixtureCo")
    dec_brand = v.verify_candidate("item_brand_ws", res_brand, "chair brand FixtureCo")
    validate_contract("decision", dec_brand)
    assert dec_brand["status"] == "SUPPORTED"
    assert dec_brand["action"] == "RECOMMEND"
    assert dec_brand["claims"][0]["observed_value"] == "  FixtureCo  "

    # 6. Brand: distinct brand satisfies negation
    res_not_brand = v.parse_requirements("chair not brand Acme")
    dec_not_brand = v.verify_candidate("item_brand_ws", res_not_brand, "chair not brand Acme")
    validate_contract("decision", dec_not_brand)
    assert dec_not_brand["status"] == "SUPPORTED"
    assert dec_not_brand["action"] == "RECOMMEND"
    assert dec_not_brand["claims"][0]["observed_value"] == "  FixtureCo  "

    # 7. Non-affirmative color descriptions: negated and alternative observations abstain
    store.upload(
        "item_color_not_red",
        Principal.system(),
        ["color: not red"],
        is_public=True,
        facts=[Fact(attribute="color", value="not red", status="present", source_location="test", source_revision="r1")],
    )
    dec_not_red_obs = v.verify_candidate("item_color_not_red", res_red, "red chair")
    validate_contract("decision", dec_not_red_obs)
    assert dec_not_red_obs["status"] == "UNKNOWN"
    assert dec_not_red_obs["action"] == "ABSTAIN"
    assert dec_not_red_obs["claims"][0]["status"] == "UNKNOWN"

    store.upload(
        "item_color_alt",
        Principal.system(),
        ["color: red or blue"],
        is_public=True,
        facts=[Fact(attribute="color", value="red or blue", status="present", source_location="test", source_revision="r1")],
    )
    dec_alt = v.verify_candidate("item_color_alt", res_red, "red chair")
    validate_contract("decision", dec_alt)
    assert dec_alt["status"] == "UNKNOWN"
    assert dec_alt["action"] == "ABSTAIN"
    assert dec_alt["claims"][0]["status"] == "UNKNOWN"

    # 8. Numeric exact equality without implicit tolerance across decision boundary
    store.upload(
        "item_width_tol",
        Principal.system(),
        ["width: 48.0000005 cm"],
        is_public=True,
        facts=[Fact(attribute="width", value=48.0000005, unit="cm", status="present", source_location="test", source_revision="r1")],
    )
    res_width_eq = v.parse_requirements("chair width = 48cm")
    dec_width_tol = v.verify_candidate("item_width_tol", res_width_eq, "chair width = 48cm")
    validate_contract("decision", dec_width_tol)
    assert dec_width_tol["status"] == "CONTRADICTED"
    assert dec_width_tol["action"] == "DO_NOT_RECOMMEND"
    assert dec_width_tol["claims"][0]["status"] == "CONTRADICTED"
    assert dec_width_tol["claims"][0]["observed_value"] == 48.0000005

    print("  PASS: Semantic boundary reconciled across brand whitespace, compound color, and material composition.")


def main():
    print("=== Running Stage 04 Acceptance Checks ===")
    check_1_no_hidden_chair_defaults()
    check_2_bare_numbers_no_guessed_units()
    check_3_asymmetric_taxonomy()
    check_4_negation_handling()
    check_5_strict_boundary_equality()
    check_6_conflicting_sources_abstain()
    check_7_unsupported_features_abstain()
    check_8_catalog_directory_path()
    check_9_generator_no_synthetic_qnn()
    check_10_80_case_testbed_zero_false_positives()
    check_11_mixed_unit_ranges_and_reversed()
    check_12_brand_grammar_and_negative()
    check_13_answer_boundary_fidelity()
    check_14_structured_evidence_status()
    check_15_semantic_boundary_reconciliation()
    print("\nALL 15 ACCEPTANCE CHECKS PASSED CLEANLY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
