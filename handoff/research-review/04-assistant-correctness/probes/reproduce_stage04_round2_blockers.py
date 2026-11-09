"""Reproduce Stage 04 Round 2 review blockers before repair.

1. Brand predicate semantics and observation fidelity:
   - "oak chair brand Fixture" against FixtureCo returns SUPPORTED / RECOMMEND.
   - The source brand is FixtureCo, but claim reports observed_value="Fixture".
   - "oak chair not brand Fixture" returns CONTRADICTED.

2. Structured evidence status:
   - Upload Fact(attribute="width", value=48, unit="cm", status="unknown", ...).
   - Verify "chair width < 50cm" returns SUPPORTED / RECOMMEND instead of UNKNOWN / ABSTAIN.
   - Same for status="missing" and categorical non-present facts.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

EDGE_ROOT = Path(__file__).resolve().parents[4] / "edge-rag-demo"
sys.path.insert(0, str(EDGE_ROOT / "src"))

from edge_rag.store import EvidenceStore, Fact, Principal
from edge_rag.verifier import Verifier

FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"


def test_reproduce_brand_semantics_and_observation_fidelity() -> bool:
    print("Reproduction 1: Brand semantics and observation fidelity...")
    v = Verifier(catalog_path=FIXTURE_CATALOG)

    # Probe 1A: "oak chair brand Fixture" against fixture-chair-oak (brand: FixtureCo)
    res_pos = v.parse_requirements("oak chair brand Fixture")
    dec_pos = v.verify_candidate("fixture-chair-oak", res_pos, "oak chair brand Fixture")
    brand_claims_pos = [c for c in dec_pos["claims"] if c.get("attribute") == "brand"]
    obs_val_pos = brand_claims_pos[0].get("observed_value") if brand_claims_pos else None

    print(f"  [brand Fixture] status={dec_pos['status']}, action={dec_pos['action']}, observed_value={obs_val_pos!r}")

    # Probe 1B: "oak chair not brand Fixture" against fixture-chair-oak (brand: FixtureCo)
    res_neg = v.parse_requirements("oak chair not brand Fixture")
    dec_neg = v.verify_candidate("fixture-chair-oak", res_neg, "oak chair not brand Fixture")
    print(f"  [not brand Fixture] status={dec_neg['status']}, action={dec_neg['action']}")

    defect_found = False
    if dec_pos["status"] == "SUPPORTED" or dec_pos["action"] == "RECOMMEND":
        print("  FAIL (Defect reproduced): 'brand Fixture' accepted substring match on 'FixtureCo'!")
        defect_found = True
    if obs_val_pos == "Fixture":
        print("  FAIL (Defect reproduced): observed_value replaced with required_value 'Fixture'!")
        defect_found = True
    if dec_neg["status"] == "CONTRADICTED" or dec_neg["action"] == "DO_NOT_RECOMMEND":
        print("  FAIL (Defect reproduced): 'not brand Fixture' falsely contradicted 'FixtureCo'!")
        defect_found = True
    return defect_found


def test_reproduce_structured_evidence_status() -> bool:
    print("Reproduction 2: Structured evidence status ignored...")
    store = EvidenceStore(":memory:")
    store.upload(
        "item",
        Principal.system(),
        ["width: 48 cm"],
        is_public=True,
        item_id="item",
        variant_id="item",
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
    v = Verifier(store=store, auto_load=False)
    res = v.parse_requirements("chair width < 50cm")
    dec = v.verify_candidate("item", res, "chair width < 50cm")

    print(f"  [width < 50cm on status=unknown] status={dec['status']}, action={dec['action']}")

    defect_found = False
    if dec["status"] == "SUPPORTED" or dec["action"] == "RECOMMEND":
        print("  FAIL (Defect reproduced): Non-present fact (status=unknown) supported width requirement!")
        defect_found = True

    # Check status="missing"
    store_missing = EvidenceStore(":memory:")
    store_missing.upload(
        "item2",
        Principal.system(),
        ["width: 48 cm"],
        is_public=True,
        item_id="item2",
        variant_id="item2",
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
    v_missing = Verifier(store=store_missing, auto_load=False)
    dec_missing = v_missing.verify_candidate("item2", res, "chair width < 50cm")
    print(f"  [width < 50cm on status=missing] status={dec_missing['status']}, action={dec_missing['action']}")
    if dec_missing["status"] == "SUPPORTED" or dec_missing["action"] == "RECOMMEND":
        print("  FAIL (Defect reproduced): Non-present fact (status=missing) supported width requirement!")
        defect_found = True

    return defect_found


def main() -> int:
    print("=== Checking Stage 04 Round 2 Review Blockers Status ===")
    d1 = test_reproduce_brand_semantics_and_observation_fidelity()
    d2 = test_reproduce_structured_evidence_status()
    total = (1 if d1 else 0) + (1 if d2 else 0)
    print(f"\nActive blockers remaining: {total}/2")
    if total == 0:
        print("ALL ROUND 2 DEFECTS RESOLVED.")
        return 0
    print(f"DEFECTS DETECTED: {total} blockers active.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
