"""Extractive answer generator for QEdge Assist.

Enforces:
- Neutral, extractive answers derived strictly from verified evidence claims.
- Optional generation cannot alter or decide the verdict.
- Removal of synthetic QNN benchmark subprocess execution from normal answer composition.
- Clear reporting of unverified NPU status and unmeasured runtime metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .paths import resolve_llm_inference_root


class Generator:
    """Produces extractive answers from verified evidence without deciding verdicts."""

    def __init__(self, llm_inference_dir: Optional[str | Path] = None):
        self.root = resolve_llm_inference_root(llm_inference_dir)

    def get_benchmark_metrics(self) -> dict[str, str]:
        """Report metrics as unmeasured; synthetic benchmark removed from normal composition."""
        return {
            "policy": "not measured",
            "hit_rate": "not measured",
            "tok_per_mj": "not measured",
            "source": "synthetic benchmark removed from answer composition",
            "npu_status": "unverified",
        }

    def generate(
        self,
        title: str,
        verifications_or_decision: Any = None,
        *,
        decision: Any = None,
    ) -> str:
        """Summarize verification statuses without overriding deterministic decision policy."""
        target = decision if decision is not None else verifications_or_decision

        # Handle v2 decision record
        if isinstance(target, dict) and "action" in target:
            action = target.get("action")
            status = target.get("status")
            reason = target.get("reason", "")
            claims = target.get("claims", [])
            unparsed = target.get("unparsed_clauses", [])
            ambiguities = target.get("ambiguities", [])

            if action == "RECOMMEND":
                citations = []
                for c in claims:
                    if c.get("status") == "SUPPORTED":
                        citations.append(
                            f"{c.get('attribute')}={c.get('observed_value')}{c.get('observed_unit') or ''} ({c.get('evidence_text')})"
                        )
                ev_str = "; ".join(citations) if citations else "evidence verified"
                return f"Recommended: {title}. Verified evidence: [{ev_str}]."

            elif action == "DO_NOT_RECOMMEND":
                contra = []
                for c in claims:
                    if c.get("status") == "CONTRADICTED":
                        contra.append(
                            f"{c.get('attribute')}={c.get('observed_value')}{c.get('observed_unit') or ''} ({c.get('evidence_text')})"
                        )
                contra_str = "; ".join(contra) if contra else "unsupported attributes"
                return f"Do not recommend: {title}. Contradicted requirements: [{contra_str}]."

            else:  # ABSTAIN
                issues = []
                if unparsed:
                    issues.append(f"unresolved clauses: {unparsed}")
                if ambiguities:
                    issues.append(f"ambiguities: {ambiguities}")
                issue_str = f" ({'; '.join(issues)})" if issues else ""
                return f"Abstain for {title}. Reason: {reason}{issue_str}."

        # Legacy dict format: attribute -> (status, detail)
        if isinstance(verifications_or_decision, dict):
            details = []
            for attribute, val in verifications_or_decision.items():
                if isinstance(val, tuple) and len(val) >= 2:
                    st, detail = val[0], val[1]
                    if detail:
                        details.append(f"{attribute} ({detail}): {st}")
                    else:
                        details.append(f"{attribute}: {st}")
                else:
                    details.append(f"{attribute}: {val}")
            summary = ", ".join(details) if details else "no required attributes checked"
            return (
                f"Based on verified evidence: {title}. Verification: [{summary}]. "
                "Stage 01 template only; the Stage 07 decision policy and "
                "recommendation are not implemented."
            )

        return (
            f"Based on verified evidence: {title}. "
            "Stage 01 template only; the Stage 07 decision policy and "
            "recommendation are not implemented."
        )

    def format_footer(
        self,
        docs_verified: int = 3,
        catalog_version: str = "unavailable",
        retrieval_method: str = "local BM25 (Stage 01 baseline)",
    ) -> str:
        """Produce a status box with no unverified NPU or energy claim."""
        inference = (
            "Inference: runtime metrics not measured (synthetic benchmark removed "
            "from answer composition) · QNN/NPU status: unverified"
        )
        divider = "═" * 62
        return (
            f"{divider}\n"
            f" Evidence store: local SQLite · {docs_verified} document(s) checked\n"
            f" Retrieval: {catalog_version} · {retrieval_method}\n"
            f" {inference}\n"
            f"{divider}"
        )
