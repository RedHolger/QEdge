#!/usr/bin/env python3
"""Pre- and post-repair reproduction probe for Stage 04 Round 4 Review Blockers.

Blockers under review:
1. observed color="not red", query="red chair"
   Expected: UNKNOWN / ABSTAIN without certifying positive requirement.
2. observed color="red or blue", query="red chair"
   Expected: UNKNOWN / ABSTAIN without certifying positive requirement.
3. attribute="width", value=48.0000005, unit="cm", query="chair width = 48cm"
   Expected: Exact numeric equality without ContractValidationError (CONTRADICTED / DO_NOT_RECOMMEND).
"""
from __future__ import annotations

import sys
import tempfile
from edge_rag.store import EvidenceStore, Fact
from edge_rag.verifier import Verifier
from qedge_common.schema import validate, ContractValidationError


def test_repro_1_not_red() -> tuple[bool, str]:
    """1. observed color='not red', query='red chair'"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(f"{tmpdir}/store.db")
        store.upload(
            "item-not-red",
            "alice",
            ["color: not red"],
            facts=[
                Fact(
                    attribute="color",
                    value="not red",
                    unit=None,
                    status="present",
                    source_location="probe",
                    source_revision="r1",
                )
            ],
        )
        verifier = Verifier(store=store, auto_load=False)
        q = "red chair"
        pr = verifier.parse_requirements(q)
        dec = verifier.verify_candidate("item-not-red", pr, q)
        validate("decision", dec)
        dec_status = dec.get("status")
        dec_action = dec.get("action")
        claim_status = dec["claims"][0].get("status") if dec.get("claims") else None
        if dec_status == "UNKNOWN" and dec_action == "ABSTAIN" and claim_status == "UNKNOWN":
            return True, f"status={dec_status}, action={dec_action}, claim_status={claim_status}"
        return False, f"Incorrect positive certainty: status={dec_status}, action={dec_action}, claim_status={claim_status}"


def test_repro_2_red_or_blue() -> tuple[bool, str]:
    """2. observed color='red or blue', query='red chair'"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(f"{tmpdir}/store.db")
        store.upload(
            "item-red-or-blue",
            "alice",
            ["color: red or blue"],
            facts=[
                Fact(
                    attribute="color",
                    value="red or blue",
                    unit=None,
                    status="present",
                    source_location="probe",
                    source_revision="r1",
                )
            ],
        )
        verifier = Verifier(store=store, auto_load=False)
        q = "red chair"
        pr = verifier.parse_requirements(q)
        dec = verifier.verify_candidate("item-red-or-blue", pr, q)
        validate("decision", dec)
        dec_status = dec.get("status")
        dec_action = dec.get("action")
        claim_status = dec["claims"][0].get("status") if dec.get("claims") else None
        if dec_status == "UNKNOWN" and dec_action == "ABSTAIN" and claim_status == "UNKNOWN":
            return True, f"status={dec_status}, action={dec_action}, claim_status={claim_status}"
        return False, f"Incorrect positive certainty: status={dec_status}, action={dec_action}, claim_status={claim_status}"


def test_repro_3_numeric_equality_tolerance() -> tuple[bool, str]:
    """3. attribute='width', value=48.0000005, unit='cm', query='chair width = 48cm'"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(f"{tmpdir}/store.db")
        val = 48.0000005
        store.upload(
            "item-width",
            "alice",
            [f"width: {val} cm"],
            facts=[
                Fact(
                    attribute="width",
                    value=val,
                    unit="cm",
                    status="present",
                    source_location="probe",
                    source_revision="r1",
                )
            ],
        )
        verifier = Verifier(store=store, auto_load=False)
        q = "chair width = 48cm"
        pr = verifier.parse_requirements(q)
        try:
            dec = verifier.verify_candidate("item-width", pr, q)
            validate("decision", dec)
            dec_status = dec.get("status")
            dec_action = dec.get("action")
            claim_status = dec["claims"][0].get("status") if dec.get("claims") else None
            obs_val = dec["claims"][0].get("observed_value") if dec.get("claims") else None
            # Under exact equality without implicit tolerance, 48.0000005 != 48.0 -> CONTRADICTED / DO_NOT_RECOMMEND
            if dec_status == "CONTRADICTED" and dec_action == "DO_NOT_RECOMMEND" and claim_status == "CONTRADICTED":
                return True, f"status={dec_status}, action={dec_action}, claim_status={claim_status}, obs={obs_val}"
            return False, f"Unexpected decision: status={dec_status}, action={dec_action}, claim_status={claim_status}"
        except ContractValidationError as e:
            return False, f"ContractValidationError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def main() -> int:
    print("=== Checking Stage 04 Round 4 Review Blockers Status ===")
    tests = [
        ("Reproduction 1: Negated color observation ('not red' vs 'red chair')", test_repro_1_not_red),
        ("Reproduction 2: Alternative color observation ('red or blue' vs 'red chair')", test_repro_2_red_or_blue),
        ("Reproduction 3: Numeric equality tolerance mismatch ('48.0000005 cm' vs 'chair width = 48cm')", test_repro_3_numeric_equality_tolerance),
    ]

    active_blockers = 0
    for title, test_fn in tests:
        print(f"{title}...")
        passed, msg = test_fn()
        if passed:
            print(f"  RESOLVED: {msg}")
        else:
            print(f"  ACTIVE BLOCKER: {msg}")
            active_blockers += 1

    print(f"\nActive blockers remaining: {active_blockers}/3")
    if active_blockers == 0:
        print("ALL ROUND 4 DEFECTS RESOLVED.")
        return 0
    else:
        print("DEFECTS REPRODUCED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
