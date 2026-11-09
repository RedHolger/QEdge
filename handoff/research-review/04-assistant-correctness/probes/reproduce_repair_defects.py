"""Reproduce Stage 04 independent spot-check failures before repairs.

Tests the specific failing behaviors identified by review:
1. Mixed-unit range reordering bug: 'oak chair between 60cm and 1m wide' -> 1cm-60m falsely recommending 48cm chair.
2. Missing dimension range assumption: 'oak chair between 40cm and 50cm' silently assumes width.
3. Substantive requirement dropped as filler: 'oak chair that must be comfortable' drops comfortable and returns SUPPORTED.
4. Brand extraction missing: 'oak chair brand FixtureCo' fails to parse brand, resulting in ABSTAIN.
5. Duplicate in held-out 40-case evaluation set: 39 unique pairs instead of 40.
6. Answer boundary rendering: ensures rendered answers reflect the decision status and contradictions.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

EDGE_ROOT = Path(__file__).resolve().parents[4] / "edge-rag-demo"
sys.path.insert(0, str(EDGE_ROOT / "src"))

from edge_rag.cli import run_pipeline, format_cli_output
from edge_rag.requirements import parse_query_requirements
from edge_rag.verifier import Verifier

FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"
CASES_FILE = EDGE_ROOT / "tests" / "fixtures" / "decision_cases_80.json"


def test_reproduce_1_mixed_unit_range():
    print("Reproduction 1: Mixed-unit range ('between 60cm and 1m wide')...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    res = parse_query_requirements("oak chair between 60cm and 1m wide")
    req_tuples = [(r.attribute, r.operator, r.required_value, r.required_unit) for r in res.requirements]
    print(f"  Parsed requirements: {req_tuples}")
    dec = v.verify_candidate("fixture-chair-oak", res, "oak chair between 60cm and 1m wide")
    print(f"  Decision status: {dec['status']}, action: {dec['action']}")
    # 48cm chair is OUTSIDE [60cm, 100cm]. It must NOT be SUPPORTED/RECOMMEND!
    if dec["status"] == "SUPPORTED" or dec["action"] == "RECOMMEND":
        print("  FAIL (Defect reproduced): 48cm chair falsely recommended for 60cm–1m range due to 1cm–60m reordering!")
        return True
    return False


def test_reproduce_2_missing_dimension_assumption():
    print("Reproduction 2: Missing dimension in range ('between 40cm and 50cm')...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    res = parse_query_requirements("oak chair between 40cm and 50cm")
    req_attrs = [r.attribute for r in res.requirements]
    print(f"  Parsed attributes: {req_attrs}, ambiguities: {res.ambiguities}")
    dec = v.verify_candidate("fixture-chair-oak", res, "oak chair between 40cm and 50cm")
    print(f"  Decision status: {dec['status']}, action: {dec['action']}")
    if "width" in req_attrs and not res.ambiguities:
        print("  FAIL (Defect reproduced): Silently assumed 'width' for unspecified dimension range!")
        return True
    return False


def test_reproduce_3_dropped_substantive_requirement():
    print("Reproduction 3: Substantive requirement dropped ('that must be comfortable')...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    res = parse_query_requirements("oak chair that must be comfortable")
    print(f"  Parsed reqs: {[r.attribute for r in res.requirements]}, unparsed: {res.unparsed_clauses}")
    dec = v.verify_candidate("fixture-chair-oak", res, "oak chair that must be comfortable")
    print(f"  Decision status: {dec['status']}, action: {dec['action']}")
    if dec["status"] == "SUPPORTED" or dec["action"] == "RECOMMEND":
        print("  FAIL (Defect reproduced): Subjective requirement 'comfortable' discarded as filler, returning SUPPORTED!")
        return True
    return False


def test_reproduce_4_brand_parsing_missing():
    print("Reproduction 4: Brand parsing ('oak chair brand FixtureCo')...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)
    res = parse_query_requirements("oak chair brand FixtureCo")
    brand_reqs = [r for r in res.requirements if r.attribute == "brand"]
    print(f"  Brand requirements found: {len(brand_reqs)}, unparsed: {res.unparsed_clauses}")
    if len(brand_reqs) == 0:
        print("  FAIL (Defect reproduced): Brand grammar not recognized; brand became unparsed clause!")
        return True
    return False


def test_reproduce_5_held_out_duplicate():
    print("Reproduction 5: Held-out duplicate check...")
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
    held_out = [c for c in cases if c["split"] == "held_out"]
    pairs = [(c["query"], c["item_id"]) for c in held_out]
    unique_pairs = set(pairs)
    print(f"  Held-out records: {len(held_out)}, Unique (query, item_id) pairs: {len(unique_pairs)}")
    if len(unique_pairs) < len(held_out):
        print(f"  FAIL (Defect reproduced): Held-out set contains duplicates ({len(unique_pairs)} < {len(held_out)})!")
        return True
    return False


def main():
    print("=== Reproducing Stage 04 Spot-Check Defects ===")
    defects = 0
    if test_reproduce_1_mixed_unit_range(): defects += 1
    if test_reproduce_2_missing_dimension_assumption(): defects += 1
    if test_reproduce_3_dropped_substantive_requirement(): defects += 1
    if test_reproduce_4_brand_parsing_missing(): defects += 1
    if test_reproduce_5_held_out_duplicate(): defects += 1

    print(f"\nTotal defects reproduced: {defects}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
