#!/usr/bin/env python3
"""Pre- and post-repair reproduction probe for Stage 04 Round 3 Review Blockers.

Blockers under review:
1. attribute="color", value="dark red"; query="red chair"
   Expected: Validated decision without ContractValidationError.
2. attribute="material", value="oak veneer"; query="oak chair"
   Expected: Validated UNKNOWN / ABSTAIN decision without ContractValidationError.
3. attribute="brand", value=" FixtureCo "; query="chair brand FixtureCo"
   Expected: Validated SUPPORTED / RECOMMEND decision without ContractValidationError.
"""
from __future__ import annotations

import sys
import tempfile
from edge_rag.store import EvidenceStore, Fact
from edge_rag.verifier import Verifier
from qedge_common.schema import validate, ContractValidationError


def test_repro_1_color() -> tuple[bool, str]:
    """1. attribute='color', value='dark red'; query='red chair'"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(f"{tmpdir}/store.db")
        store.upload(
            "item-color",
            "alice",
            ["Color: dark red"],
            facts=[
                Fact(
                    attribute="color",
                    value="dark red",
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
        try:
            dec = verifier.verify_candidate("item-color", pr, q)
            # Contract validation
            validate("decision", dec)
            # Check claim
            claims = dec.get("claims", [])
            obs = claims[0].get("observed_value") if claims else None
            status = claims[0].get("status") if claims else None
            return True, f"status={status}, action={dec.get('action')}, observed_value={obs!r}"
        except ContractValidationError as e:
            return False, f"ContractValidationError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def test_repro_2_material() -> tuple[bool, str]:
    """2. attribute='material', value='oak veneer'; query='oak chair'"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(f"{tmpdir}/store.db")
        store.upload(
            "item-material",
            "alice",
            ["Material: oak veneer"],
            facts=[
                Fact(
                    attribute="material",
                    value="oak veneer",
                    unit=None,
                    status="present",
                    source_location="probe",
                    source_revision="r1",
                )
            ],
        )
        verifier = Verifier(store=store, auto_load=False)
        q = "oak chair"
        pr = verifier.parse_requirements(q)
        try:
            dec = verifier.verify_candidate("item-material", pr, q)
            # Contract validation
            validate("decision", dec)
            claims = dec.get("claims", [])
            obs = claims[0].get("observed_value") if claims else None
            status = claims[0].get("status") if claims else None
            action = dec.get("action")
            dec_status = dec.get("status")
            if dec_status == "UNKNOWN" and action == "ABSTAIN":
                return True, f"status={status}, action={action}, observed_value={obs!r}"
            return False, f"Expected UNKNOWN/ABSTAIN, got status={dec_status}, action={action}"
        except ContractValidationError as e:
            return False, f"ContractValidationError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def test_repro_3_brand() -> tuple[bool, str]:
    """3. attribute='brand', value=' FixtureCo '; query='chair brand FixtureCo'"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(f"{tmpdir}/store.db")
        store.upload(
            "item-brand",
            "alice",
            ["Brand:  FixtureCo "],
            facts=[
                Fact(
                    attribute="brand",
                    value=" FixtureCo ",
                    unit=None,
                    status="present",
                    source_location="probe",
                    source_revision="r1",
                )
            ],
        )
        verifier = Verifier(store=store, auto_load=False)
        q = "chair brand FixtureCo"
        pr = verifier.parse_requirements(q)
        try:
            dec = verifier.verify_candidate("item-brand", pr, q)
            # Contract validation
            validate("decision", dec)
            claims = dec.get("claims", [])
            obs = claims[0].get("observed_value") if claims else None
            status = claims[0].get("status") if claims else None
            action = dec.get("action")
            return True, f"status={status}, action={action}, observed_value={obs!r}"
        except ContractValidationError as e:
            return False, f"ContractValidationError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def main() -> int:
    print("=== Checking Stage 04 Round 3 Review Blockers Status ===")
    tests = [
        ("Reproduction 1: Color token vs string equality ('dark red' vs 'red chair')", test_repro_1_color),
        ("Reproduction 2: Unresolved material composition ('oak veneer' vs 'oak chair')", test_repro_2_material),
        ("Reproduction 3: Brand whitespace normalization (' FixtureCo ' vs 'chair brand FixtureCo')", test_repro_3_brand),
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
        print("ALL ROUND 3 DEFECTS RESOLVED.")
        return 0
    else:
        print("DEFECTS REPRODUCED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
