"""Acceptance probe: the baseline v1 gaps must be rejected by v2 semantics.

This is an artifact for the Stage 02 handoff, not a replacement for the unit
tests.  It deliberately constructs the same invalid records that the baseline
probe demonstrated, but labels them with schema_version=2 and requires the
current validator to reject them.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

from qedge_common.schema import (
    ContractValidationError,
    example_path,
    load_json,
    validate,
)


def load(name: str, version: str = "2") -> dict:
    return load_json(example_path(name, version))


def expect_rejected(label: str, name: str, payload: dict) -> bool:
    try:
        validate(name, payload)
    except ContractValidationError as exc:
        print(f"PASS rejected {label}: {exc}")
        return True
    print(f"FAIL accepted {label}")
    return False


def expect_valid(label: str, name: str, payload: dict) -> bool:
    try:
        validate(name, payload)
    except Exception as exc:
        print(f"FAIL positive {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"PASS positive {label}")
    return True


def main() -> int:
    ok = True
    decision = load("decision")
    claim = load("evidence_claim")
    benchmark = load("benchmark_result")
    hardware = load("hardware_status")

    empty = copy.deepcopy(decision)
    empty["claims"] = []
    ok &= expect_rejected("empty SUPPORTED decision", "decision", empty)

    contradicted = copy.deepcopy(decision)
    contradicted["claims"][0].update(
        observed_value=75.0,
        status="CONTRADICTED",
        evidence_text="width: 75.0 cm",
    )
    ok &= expect_rejected("contradicted child with positive parent", "decision", contradicted)

    missing = copy.deepcopy(decision)
    missing["claims"] = [missing["claims"][0]]
    ok &= expect_rejected("missing mandatory coverage", "decision", missing)

    mismatched = copy.deepcopy(decision)
    mismatched["claims"][0]["item_id"] = "other-item"
    ok &= expect_rejected("mismatched child item_id", "decision", mismatched)

    bad_time = copy.deepcopy(decision)
    bad_time["created_at"] = "not-a-date"
    ok &= expect_rejected("invalid created_at", "decision", bad_time)

    bad_claim_time = copy.deepcopy(decision)
    bad_claim_time["claims"][0]["retrieved_at"] = "2026-10-27T00:00:00"
    ok &= expect_rejected("invalid nested retrieved_at", "decision", bad_claim_time)

    null_observed = copy.deepcopy(decision)
    null_observed["claims"][0]["observed_value"] = None
    ok &= expect_rejected("null supported observation", "decision", null_observed)

    synthetic_measured = copy.deepcopy(benchmark)
    synthetic_measured["measured"] = True
    ok &= expect_rejected("synthetic record marked measured", "benchmark_result", synthetic_measured)

    measured_unverified = copy.deepcopy(hardware)
    measured_unverified["npu_status"] = "measured"
    ok &= expect_rejected("measured NPU with unverified device", "hardware_status", measured_unverified)

    nonfinite = copy.deepcopy(decision)
    nonfinite["claims"][0]["observed_value"] = math.nan
    ok &= expect_rejected("NaN observed value", "decision", nonfinite)

    ok &= expect_valid("v2 positive decision", "decision", decision)
    ok &= expect_valid("v2 standalone claim", "evidence_claim", claim)
    ok &= expect_valid("v2 fixture-smoke benchmark", "benchmark_result", benchmark)
    ok &= expect_valid("v2 unverified hardware", "hardware_status", hardware)

    v1_decision = load("decision", "1")
    ok &= expect_valid("historical v1 decision", "decision", v1_decision)

    no_candidate = load_json(
        Path(__file__).resolve().parents[4]
        / "qedge-common"
        / "tests"
        / "fixtures"
        / "valid"
        / "v2"
        / "decision_no_candidate.json"
    )
    ok &= expect_valid("explicit no-candidate decision", "decision", no_candidate)

    print("ACCEPTANCE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
