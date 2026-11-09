"""Validated contract producers for fixtures and local smoke records.

These helpers are intentionally small.  They demonstrate that a producer
boundary calls the shared validator rather than emitting a schema-shaped dict
and hoping the consumer validates it later.  They do not create measurements,
model runs, or hardware evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import platform
from typing import Any

from .schema import validate


def fixture_smoke_benchmark(
    *,
    run_id: str = "fixture-smoke-contract",
    iterations: int = 1,
    median_pipeline_ms: float = 0.0,
) -> dict[str, Any]:
    """Build and validate a v2 fixture-smoke benchmark record."""
    now = datetime.now(timezone.utc).isoformat()
    record: dict[str, Any] = {
        "schema_version": "2",
        "run_id": run_id,
        "kind": "fixture-smoke",
        "created_at": now,
        "model": None,
        "backend": "cpu",
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "npu_status": "unverified",
        },
        "measured": False,
        "metrics": {
            "iterations": iterations,
            "median_pipeline_ms": median_pipeline_ms,
        },
        "outputs": {
            "text": None,
            "token_ids": None,
            "token_trace": None,
        },
        "cache": {
            "policy": "fixture-no-cache",
            "capacity_tokens": 0,
        },
        "provenance": {
            "command": ".venv/bin/python -m qedge_common.records",
            "notes": [
                "Fixture-only producer; not a performance or model measurement.",
                "No cache hit rate is reported for a non-cache fixture.",
            ],
            "log_path": None,
        },
    }
    validate("benchmark_result", record)
    return record


if __name__ == "__main__":
    import json

    print(json.dumps(fixture_smoke_benchmark(), indent=2))

