"""Fixture-only smoke benchmark for the Stage 01 package.

This is not a retrieval or hardware benchmark. It emits a v2 fixture-smoke
record with ``measured=false`` so later stages cannot mistake it for a headline
number. The producer boundary requires and invokes the qedge-common semantic
validator before emitting records.
"""
from __future__ import annotations

import datetime
import json
import platform
import statistics
import sys
import time
from typing import Any

from .cli import run_pipeline
from .paths import optional_catalog_file


def _validate_record(record: dict[str, Any]) -> None:
    """Validate the benchmark record against the shared contract.

    Validation is mandatory for benchmark records. If qedge-common is not
    installed or fails to import, this raises RuntimeError with the underlying
    cause rather than silently emitting an unverified record.
    """
    try:
        from qedge_common.schema import validate
    except ImportError as exc:
        raise RuntimeError(
            "qedge-common is required to validate benchmark contract records. "
            f"Failed to import validator: {exc}"
        ) from exc
    validate("benchmark_result", record)


def run_smoke(iterations: int = 3) -> dict[str, Any]:
    catalog = optional_catalog_file()
    if catalog is None:
        return {
            "status": "skipped",
            "reason": "No catalog found. Set QEDGE_CATALOG_DIR to run the fixture smoke.",
        }

    totals = []
    last: dict[str, Any] | None = None
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        last = run_pipeline(
            "wooden dining chair under 60cm",
            catalog_path=catalog,
            k=5,
        )
        totals.append((time.perf_counter() - started) * 1000.0)

    assert last is not None
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "schema_version": "2",
        "run_id": f"fixture-smoke-{int(time.time())}",
        "kind": "fixture-smoke",
        "created_at": now,
        "model": None,
        "backend": "cpu-local-bm25",
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "npu_status": "unverified",
        },
        "measured": False,
        "metrics": {
            "iterations": len(totals),
            "median_pipeline_ms": round(statistics.median(totals), 3),
            "min_pipeline_ms": round(min(totals), 3),
            "max_pipeline_ms": round(max(totals), 3),
            "candidates": len(last["candidates"]),
            "documents_checked": len(last["verified_results"]),
        },
        "outputs": {
            "text": last["generated_response"],
            "token_ids": None,
            "token_trace": None,
        },
        "provenance": {
            "command": ".venv/bin/python -m edge_rag.bench",
            "notes": [
                "Fixture catalog only; not an ABO retrieval evaluation.",
                "In-process stage timing; excludes process startup.",
                "No Snapdragon/QNN device was used.",
                "No cache hit rate is reported because this smoke has no cache policy.",
            ],
            "log_path": None,
        },
    }
    _validate_record(record)
    return record


def main() -> int:
    try:
        record = run_smoke()
    except Exception as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2))
    return 0 if record.get("status") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
