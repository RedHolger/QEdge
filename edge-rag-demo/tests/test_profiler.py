"""Unit tests for the Stage 01 profiler."""
from __future__ import annotations

import unittest

from edge_rag.profiler import PipelineProfiler


class ProfilerTests(unittest.TestCase):
    def test_profiler_records_measurements(self):
        profiler = PipelineProfiler()
        with profiler.measure("RETRIEVE"):
            pass
        with profiler.measure("VERIFY"):
            pass
        with profiler.measure("GENERATE"):
            pass

        summary = profiler.summary()
        self.assertIn("stages", summary)
        self.assertIn("total_ms", summary)
        self.assertIn("npu_status", summary)
        self.assertIsNone(summary["npu_offloadable_pct"])
        self.assertIsNone(summary["est_npu_total_ms"])

    def test_profiler_does_not_claim_npu_execution(self):
        profiler = PipelineProfiler()
        summary = profiler.summary()
        self.assertEqual(summary["npu_status"], "unverified")
        for stage in summary["stages"].values():
            self.assertEqual(stage["npu_status"], "unverified")

    def test_stage_metadata_is_descriptive_only(self):
        stages = PipelineProfiler.STAGE_INFO
        self.assertIn("CPU in Stage 01", stages["RETRIEVE"]["target"])
        self.assertFalse(stages["GENERATE"]["npu_candidate"])


if __name__ == "__main__":
    unittest.main()
