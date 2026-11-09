"""Pipeline-level profiler for the QEdge Assist demo.

Stage 01 records wall-clock stage times only. It does not claim that any stage
ran on an NPU, and it does not project Snapdragon latency.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


class PipelineProfiler:
    """Measure RETRIEVE, VERIFY, and GENERATE stage wall time."""

    STAGE_INFO = {
        "RETRIEVE": {
            "npu_candidate": True,
            "target": "retrieval (CPU in Stage 01)",
        },
        "VERIFY": {
            "npu_candidate": False,
            "target": "deterministic evidence checks (CPU)",
        },
        "GENERATE": {
            "npu_candidate": False,
            "target": "template fill (CPU)",
        },
    }

    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.timings_ms[stage] = max(0.1, (t1 - t0) * 1000.0)

    def summary(self) -> dict[str, Any]:
        stages = {}
        for name, meta in self.STAGE_INFO.items():
            stages[name] = {
                "time_ms": round(self.timings_ms.get(name, 0.0), 2),
                "npu_candidate": meta["npu_candidate"],
                "npu_status": "unverified",
                "target": meta["target"],
            }

        total_ms = sum(stage["time_ms"] for stage in stages.values())
        return {
            "stages": stages,
            "total_ms": round(total_ms, 2),
            "measurement_scope": "in-process stage wall time; excludes startup",
            "npu_status": "unverified",
            "npu_offloadable_pct": None,
            "est_npu_total_ms": None,
            "notes": [
                "No Snapdragon/QNN device is present.",
                "Stage metadata is not an offload measurement or a projection.",
            ],
        }
