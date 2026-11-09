"""80-case decision testbed for QEdge Assist (40 development / 40 held-out).

Authored regression cases for internal verification.
Passing these tests validates internal regression invariants (including zero
false positives on adverse cases), but does NOT establish independently
reviewed held-out accuracy. Independent generalization evaluation remains
a pending gate.

Enforces:
- Verification of 40 development and 40 held-out cases (80 strictly unique query/item pairs).
- Zero positive decisions (SUPPORTED/RECOMMEND) on known adverse cases
  (CONTRADICTED, UNKNOWN, ABSTAIN).
- Preservation of authored provenance and notes.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from edge_rag.verifier import Verifier

EDGE_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_CATALOG = EDGE_ROOT / "tests" / "fixtures" / "catalog.jsonl"
CASES_FILE = EDGE_ROOT / "tests" / "fixtures" / "decision_cases_80.json"


class Decision80CasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = Verifier(catalog_path=FIXTURE_CATALOG)
        with open(CASES_FILE, "r", encoding="utf-8") as f:
            cls.cases = json.load(f)

    def test_case_counts_and_splits(self):
        self.assertEqual(len(self.cases), 80)
        dev = [c for c in self.cases if c["split"] == "development"]
        held = [c for c in self.cases if c["split"] == "held_out"]
        self.assertEqual(len(dev), 40)
        self.assertEqual(len(held), 40)

        # Confirm all 80 (query, item_id) pairs are distinct
        pairs = [(c["query"], c["item_id"]) for c in self.cases]
        self.assertEqual(len(set(pairs)), 80, "All 80 cases must have unique (query, item_id) pairs")

        for case in self.cases:
            self.assertIn("case_id", case)
            self.assertIn("query", case)
            self.assertIn("expected_status", case)
            self.assertIn("expected_action", case)
            self.assertIn("provenance", case)
            self.assertEqual(case["provenance"], "authored-regression-suite-v1; internal development benchmark")
            self.assertIn("notes", case)

    def test_zero_false_positive_decisions_on_adverse_cases(self):
        """CRITICAL SAFETY REQUIREMENT: Zero positive decisions on known adverse cases."""
        false_positives = []
        for case in self.cases:
            expected_status = case["expected_status"]
            # Adverse cases are anything not expected to be SUPPORTED
            is_adverse = expected_status in ("CONTRADICTED", "UNKNOWN", "ABSTAIN")
            if not is_adverse:
                continue

            query = case["query"]
            item_id = case["item_id"]
            parse_result = self.verifier.parse_requirements(query)
            decision = self.verifier.verify_candidate(item_id, parse_result, query)

            actual_status = decision["status"]
            actual_action = decision["action"]

            # False positive occurs if verifier issues SUPPORTED or RECOMMEND on an adverse case
            if actual_status == "SUPPORTED" or actual_action == "RECOMMEND":
                false_positives.append({
                    "case_id": case["case_id"],
                    "query": query,
                    "item_id": item_id,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "actual_action": actual_action,
                    "notes": case["notes"],
                })

        self.assertEqual(
            len(false_positives),
            0,
            f"Expected zero false positive decisions on adverse cases, got {len(false_positives)}: {false_positives}",
        )

    def test_development_split_decisions(self):
        dev = [c for c in self.cases if c["split"] == "development"]
        correct = 0
        mismatches = []
        for case in dev:
            query = case["query"]
            item_id = case["item_id"]
            parse_result = self.verifier.parse_requirements(query)
            decision = self.verifier.verify_candidate(item_id, parse_result, query)

            if (
                decision["status"] == case["expected_status"]
                and decision["action"] == case["expected_action"]
            ):
                correct += 1
            else:
                mismatches.append({
                    "case_id": case["case_id"],
                    "query": query,
                    "expected": f"{case['expected_status']}/{case['expected_action']}",
                    "actual": f"{decision['status']}/{decision['action']}",
                    "reason": decision["reason"],
                })

        # All 40 development cases should pass
        self.assertEqual(
            len(mismatches),
            0,
            f"Development set mismatches ({len(mismatches)}/40): {mismatches}",
        )
        self.assertEqual(correct, 40)

    def test_held_out_split_decisions(self):
        held = [c for c in self.cases if c["split"] == "held_out"]
        correct = 0
        mismatches = []
        for case in held:
            query = case["query"]
            item_id = case["item_id"]
            parse_result = self.verifier.parse_requirements(query)
            decision = self.verifier.verify_candidate(item_id, parse_result, query)

            if (
                decision["status"] == case["expected_status"]
                and decision["action"] == case["expected_action"]
            ):
                correct += 1
            else:
                mismatches.append({
                    "case_id": case["case_id"],
                    "query": query,
                    "expected": f"{case['expected_status']}/{case['expected_action']}",
                    "actual": f"{decision['status']}/{decision['action']}",
                    "reason": decision["reason"],
                })

        self.assertEqual(
            len(mismatches),
            0,
            f"Held-out set mismatches ({len(mismatches)}/40): {mismatches}",
        )
        self.assertEqual(correct, 40)


if __name__ == "__main__":
    unittest.main()
