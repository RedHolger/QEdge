"""Reproduce the Stage 02 contract gaps before the versioned migration."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from qedge_common.schema import ContractValidationError, example_path, validate

ROOT = Path(__file__).resolve().parents[4]
FIXTURES = ROOT / "qedge-common" / "tests" / "fixtures" / "valid"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def check(label: str, name: str, payload: dict) -> bool:
    try:
        validate(name, payload)
    except ContractValidationError as exc:
        print(f"REJECTED {label}: {exc}")
        return False
    print(f"ACCEPTED {label}")
    return True


def main() -> int:
    decision = fixture("decision")
    empty = copy.deepcopy(decision)
    empty["claims"] = []
    check("empty SUPPORTED decision", "decision", empty)

    contradicted = copy.deepcopy(decision)
    contradicted["claims"][0]["status"] = "CONTRADICTED"
    check("contradicted child with SUPPORTED parent", "decision", contradicted)

    missing = copy.deepcopy(decision)
    missing["claims"] = []
    check("missing required predicate", "decision", missing)

    mismatched = copy.deepcopy(decision)
    mismatched["claims"][0]["item_id"] = "other-item"
    check("mismatched child item_id", "decision", mismatched)

    bad_time = copy.deepcopy(decision)
    bad_time["created_at"] = "not-a-date"
    check("invalid created_at", "decision", bad_time)

    bad_claim_time = copy.deepcopy(decision)
    bad_claim_time["claims"][0]["retrieved_at"] = "not-a-date"
    check("invalid nested retrieved_at", "decision", bad_claim_time)

    null_observed = copy.deepcopy(decision)
    null_observed["claims"][0]["observed_value"] = None
    check("null supported observation in nested claim", "decision", null_observed)

    claim = fixture("evidence_claim")
    standalone_null = copy.deepcopy(claim)
    standalone_null["observed_value"] = None
    check("null supported observation in standalone claim", "evidence_claim", standalone_null)

    benchmark = fixture("benchmark_result")
    synthetic_measured = copy.deepcopy(benchmark)
    synthetic_measured["measured"] = True
    check("synthetic record marked measured", "benchmark_result", synthetic_measured)

    hardware = fixture("hardware_status")
    measured_unverified = copy.deepcopy(hardware)
    measured_unverified["npu_status"] = "measured"
    check("measured NPU with unverified device", "hardware_status", measured_unverified)

    nonfinite = copy.deepcopy(decision)
    nonfinite["claims"][0]["observed_value"] = math.nan
    check("NaN observed value", "decision", nonfinite)

    print("baseline gaps complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
