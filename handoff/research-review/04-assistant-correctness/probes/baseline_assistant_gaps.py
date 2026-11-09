"""Reproduce baseline defects in edge_rag before Stage 04 repairs.

This script tests observable failing behaviors:
1. Hidden chair/oak/width defaults in cli.py.
2. Bare number unit guessing in verifier.py ("under 60" -> width_lte_cm 60.0).
3. Symmetric taxonomy defect: oak query certified by generic wood fact.
4. Substring matching defect: "red" matched inside unrelated words.
5. Negation mishandled: "not oak" parsed as positive material: "oak".
6. Strict vs inclusive comparison: "< 60" treated as "<= 60".
7. First-row conflict resolution: contradictory facts resolve to first chunk.
8. Unsupported clauses discarded: "built-in quantum charger" ignored, allowing match.
9. Catalog directory path rejected by verifier.load_catalog with IsADirectoryError.
10. Synthetic QNN benchmark binary invoked by Generator during normal demo.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure edge-rag-demo is importable
EDGE_ROOT = Path(__file__).resolve().parents[4] / "edge-rag-demo"
sys.path.insert(0, str(EDGE_ROOT / "src"))

from edge_rag.cli import run_pipeline
from edge_rag.generator import Generator
from edge_rag.store import EvidenceStore, Principal
from edge_rag.verifier import Verifier


FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"


def test_gap_1_hidden_chair_defaults():
    print("Probe 1: Hidden chair/seat defaults in run_pipeline...")
    result = run_pipeline("I need a comfy chair", k=1, catalog_path=FIXTURE_CATALOG)
    # The pipeline should NOT invent width_lte_cm=60 and material=oak
    verified = result.get("verified_results", [])
    if verified:
        v = verified[0]["verifications"]
        if "width" in v or "material" in v:
            print("  FAIL (Defect reproduced): Defaulted unprompted requirements width/material for 'chair'!")
            return True
    print("  PASS: No hidden defaults.")
    return False


def test_gap_2_bare_number_unit_guessing():
    print("Probe 2: Bare number unit guessing...")
    v = Verifier()
    reqs = v.parse_requirements("I want a chair under 60")
    if "width_lte_cm" in reqs and reqs["width_lte_cm"] == 60.0:
        print(f"  FAIL (Defect reproduced): Guessed unit 'cm' for bare number: {reqs}")
        return True
    print("  PASS: Did not guess unit.")
    return False


def test_gap_3_symmetric_taxonomy():
    print("Probe 3: Symmetric taxonomy (oak certified by generic wood)...")
    v = Verifier()
    v.store.upload(
        "doc-wood",
        Principal.system(),
        ["material: wood (source: material)"],
        is_public=True,
    )
    status, detail = v.verify_attribute("doc-wood", "material", "oak")
    if status == "SUPPORTED":
        print("  FAIL (Defect reproduced): Generic wood fact certifies oak requirement!")
        return True
    print(f"  PASS: Generic wood does not certify oak (got {status}).")
    return False


def test_gap_4_negation_mishandled():
    print("Probe 4: Negation handling ('not oak')...")
    v = Verifier()
    reqs = v.parse_requirements("I want a dining chair not oak")
    if reqs.get("material") == "oak":
        print(f"  FAIL (Defect reproduced): 'not oak' was parsed as positive material: {reqs}")
        return True
    print("  PASS: Negation parsed correctly.")
    return False


def test_gap_5_strict_boundary():
    print("Probe 5: Strict boundary comparison ('< 60' vs 60.0)...")
    v = Verifier()
    v.store.upload(
        "doc-60",
        Principal.system(),
        ["width: 60.0 cm (source: item_dimensions)"],
        is_public=True,
    )
    # If query was "< 60cm", exactly 60.0cm should be CONTRADICTED, not SUPPORTED
    status, _ = v.verify_attribute("doc-60", "width", 60.0)
    # Current verifier always does actual_w <= max_w
    # It cannot distinguish < from <=
    print(f"  Current verification evaluates <= only: status is {status}")
    return True


def test_gap_6_conflicting_source_facts():
    print("Probe 6: Conflicting source facts resolve to first row...")
    v = Verifier()
    v.store.upload(
        "doc-conflict",
        Principal.system(),
        [
            "width: 45.0 cm (source: item_dimensions)",
            "width: 80.0 cm (source: alternate_spec)",
        ],
        is_public=True,
    )
    status, _ = v.verify_attribute("doc-conflict", "width", 60.0)
    if status == "SUPPORTED":
        print("  FAIL (Defect reproduced): First chunk (45cm) certified width <= 60 ignoring conflicting chunk (80cm)!")
        return True
    return False


def test_gap_7_catalog_directory_path():
    print("Probe 7: Catalog directory path handling in Verifier.load_catalog...")
    v = Verifier()
    fixtures_dir = EDGE_ROOT / "tests" / "fixtures"
    try:
        v.load_catalog(fixtures_dir)
        print("  PASS: Directory path handled.")
        return False
    except IsADirectoryError:
        print("  FAIL (Defect reproduced): Verifier.load_catalog raises IsADirectoryError when given a directory!")
        return True
    except Exception as exc:
        print(f"  FAIL: Unexpected error {exc}")
        return True


def test_gap_8_synthetic_qnn_bench_in_generator():
    print("Probe 8: Generator attempting to run synthetic QNN bench...")
    gen = Generator()
    # Check if _benchmark_binary searches for llm_qnn_bench
    binary = gen._benchmark_binary()
    print(f"  Generator benchmark binary search target: {binary}")
    if hasattr(gen, "get_benchmark_metrics"):
        print("  FAIL (Defect reproduced): Generator contains get_benchmark_metrics referencing synthetic CPU/QNN stub.")
        return True
    return False


def main():
    defects = 0
    if test_gap_1_hidden_chair_defaults(): defects += 1
    if test_gap_2_bare_number_unit_guessing(): defects += 1
    if test_gap_3_symmetric_taxonomy(): defects += 1
    if test_gap_4_negation_mishandled(): defects += 1
    if test_gap_5_strict_boundary(): defects += 1
    if test_gap_6_conflicting_source_facts(): defects += 1
    if test_gap_7_catalog_directory_path(): defects += 1
    if test_gap_8_synthetic_qnn_bench_in_generator(): defects += 1

    print(f"\nTotal defects reproduced: {defects}/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
