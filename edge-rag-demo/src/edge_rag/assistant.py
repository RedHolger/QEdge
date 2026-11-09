"""Local Assistant for QEdge Assist (Stage 10).

Integrates:
- Explicit 'find' and 'check' execution modes.
- Deterministic extractive path working offline without model downloads.
- Optional contract-guarded LLM generation with fallback reason tracking.
- Safety invariant: never recommend a candidate with unresolved mandatory predicates.
- Untrusted data boundary and bounded context size.
- Governance: unique request_id, structured errors, stage timings,
  deadlines, and tenant/request isolation.
- Full immutable citation resolution against the evidence store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Optional, Sequence
import uuid

from .generator import Generator
from .paths import (
    CatalogUnavailable,
    resolve_catalog_dir,
    resolve_catalog_file,
    resolve_llm_inference_root,
)
from .requirements import ParseResult, ParsedRequirement
from .retriever import Retriever
from .store import (
    AuthorizationError,
    CitationError,
    EvidenceStore,
    Fact,
    NotFoundError,
    Principal,
    StoreError,
    resolve_principal,
)
from .verifier import Verifier

_CITATION_EXTRACT_RE = re.compile(r"\bcite:[A-Za-z0-9_.~%:-]+")
_IMAGE_QUERY_RE = re.compile(r"(--image|\.(jpg|jpeg|png|webp|bmp|gif)\b|^image:)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Structured Errors
# ---------------------------------------------------------------------------


class AssistantError(Exception):
    """Base exception for assistant errors with structured codes."""

    def __init__(
        self,
        message: str,
        code: str = "ASSISTANT_ERROR",
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class AssistantTimeoutError(AssistantError):
    def __init__(
        self,
        message: str = "Assistant operation timed out",
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message, code="TIMEOUT", details=details)


class AssistantModeError(AssistantError):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="INVALID_MODE", details=details)


class AssistantAuthorizationError(AssistantError):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="UNAUTHORIZED", details=details)


class AssistantInputError(AssistantError):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="INVALID_INPUT", details=details)


class AssistantStoreError(AssistantError):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="STORE_ERROR", details=details)


# ---------------------------------------------------------------------------
# Assistant Result Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AssistantResult:
    """Structured result returned by the Local Assistant."""

    schema_version: str = "2"
    request_id: str = ""
    mode: str = "find"
    query: str = ""
    item_id: Optional[str] = None
    status: str = "OK"  # "OK", "ERROR", "ABSTAIN"
    decision: dict[str, Any] = field(default_factory=dict)
    action: str = "ABSTAIN"  # "RECOMMEND", "DO_NOT_RECOMMEND", "ABSTAIN"
    decision_status: str = "UNKNOWN"  # "SUPPORTED", "CONTRADICTED", "UNKNOWN", "ABSTAIN"
    reason: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    verified_candidates: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    timing_ms: dict[str, float] = field(default_factory=dict)
    generation_fallback_reason: Optional[str] = None
    error: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "mode": self.mode,
            "query": self.query,
            "item_id": self.item_id,
            "status": self.status,
            "decision": self.decision,
            "action": self.action,
            "decision_status": self.decision_status,
            "reason": self.reason,
            "candidates": self.candidates,
            "verified_candidates": self.verified_candidates,
            "citations": self.citations,
            "citation_diagnostics": self.citation_diagnostics,
            "answer": self.answer,
            "timing_ms": self.timing_ms,
            "generation_fallback_reason": self.generation_fallback_reason,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# LocalAssistant
# ---------------------------------------------------------------------------


class LocalAssistant:
    """Unified offline assistant integrating retrieval, verification, and answer composition."""

    def __init__(
        self,
        catalog_path: Optional[str | Path] = None,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
        store: Optional[EvidenceStore] = None,
        generator: Optional[Generator] = None,
        enable_generation: bool = False,
        llm_runner: Optional[Any] = None,
        default_timeout_s: Optional[float] = 30.0,
    ):
        self.catalog_path = catalog_path
        self.enable_generation = enable_generation
        self.llm_runner = llm_runner
        self.default_timeout_s = default_timeout_s

        if verifier is not None:
            self.verifier = verifier
        else:
            selected_cat = catalog_path
            if selected_cat is None and retriever is not None:
                selected_cat = getattr(retriever, "catalog_file", None)
            self.verifier = Verifier(
                catalog_path=selected_cat if selected_cat else None,
                store=store,
            )

        if store is not None:
            self.store = store
            self.verifier.store = store
        else:
            self.store = self.verifier.store

        if retriever is not None:
            self.retriever = retriever
        else:
            try:
                self.retriever = Retriever(catalog_path=catalog_path)
            except CatalogUnavailable:
                self.retriever = None

        if generator is not None:
            self.generator = generator
        else:
            self.generator = Generator()

    @staticmethod
    def _validate_request(query, timeout_s, k=5, top_verify=3):
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise AssistantInputError("query must contain 1–4096 characters")
        if type(k) is not int or not 1 <= k <= 100:
            raise AssistantInputError("k must be an integer between 1 and 100")
        if type(top_verify) is not int or not 0 <= top_verify <= 100:
            raise AssistantInputError("top_verify must be an integer between 0 and 100")
        if timeout_s is not None and (
            isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0
        ):
            raise AssistantInputError("timeout_s must be finite and positive, or None")

    def _check_timeout(self, start_time: float, timeout_s: Optional[float]) -> None:
        if timeout_s is not None and timeout_s > 0:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_s:
                raise AssistantTimeoutError(
                    f"Assistant operation timed out after {timeout_s:.3f}s (elapsed {elapsed:.3f}s)",
                    details={"elapsed_s": elapsed, "timeout_s": timeout_s},
                )

    def _validate_and_filter_citations(
        self,
        decision: dict[str, Any],
        principal: Principal,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Validate citations for all claims in a decision and fail closed on errors.

        Enforces:
        - Citation verification applied to EVERY decision action (RECOMMEND, DO_NOT_RECOMMEND, ABSTAIN).
        - Failed evidence excluded from validated claims, deterministic answers,
          optional model context, and user-facing claim lists.
        - Fails closed to ABSTAIN whenever a decision depends on failed evidence.
        - Keeps diagnostic details separate from validated evidence.

        Returns:
            (sanitized_decision, all_citations, citation_diagnostics)
        """
        claims = decision.get("claims", [])
        validated_claims: list[dict[str, Any]] = []
        failed_claims: list[dict[str, Any]] = []
        valid_citations: list[dict[str, Any]] = []
        citation_diagnostics: list[dict[str, Any]] = []

        for claim in claims:
            ev_text = claim.get("evidence_text", "")
            found_cites = _CITATION_EXTRACT_RE.findall(ev_text)
            claim_cites_valid = True

            for cite_str in found_cites:
                try:
                    resolved = self.store.resolve_citation(cite_str, requester=principal)
                    fact_data = resolved.get("fact") or {}
                    cite_record = {
                        "citation": cite_str,
                        "doc_id": resolved.get("doc_id"),
                        "version": resolved.get("version"),
                        "attribute": fact_data.get("attribute"),
                        "value": fact_data.get("value"),
                        "unit": fact_data.get("unit"),
                        "status": fact_data.get("status"),
                        "valid": True,
                    }
                    valid_citations.append(cite_record)
                except Exception as c_exc:
                    claim_cites_valid = False
                    diag_record = {
                        "citation": cite_str,
                        "valid": False,
                        "error": str(c_exc),
                        "status": "FAILED",
                    }
                    citation_diagnostics.append(diag_record)

            if claim_cites_valid:
                validated_claims.append(claim)
            else:
                failed_claims.append(claim)

        # Build sanitized decision with ONLY validated claims
        sanitized_decision = dict(decision)
        sanitized_decision["claims"] = validated_claims

        orig_action = decision.get("action", "ABSTAIN")
        orig_status = decision.get("status", "UNKNOWN")

        if citation_diagnostics:
            # Fail closed across all actions when evidence verification fails
            # Keep diagnostic details strictly separate from decision reason and answer text
            if orig_action == "RECOMMEND":
                sanitized_decision["status"] = "ABSTAIN"
                sanitized_decision["action"] = "ABSTAIN"
                sanitized_decision["reason"] = (
                    "Evidence verification failed: citation resolution failed for supporting evidence"
                )
            elif orig_action == "DO_NOT_RECOMMEND":
                sanitized_decision["status"] = "ABSTAIN"
                sanitized_decision["action"] = "ABSTAIN"
                sanitized_decision["reason"] = (
                    "Evidence verification failed: citation resolution failed for contradicted evidence"
                )
            else:
                sanitized_decision["status"] = "ABSTAIN"
                sanitized_decision["action"] = "ABSTAIN"
                sanitized_decision["reason"] = (
                    "Evidence verification failed: citation resolution failed"
                )

        return sanitized_decision, valid_citations, citation_diagnostics

    @staticmethod
    def _validate_presentation_selection(
        parsed: Any,
        num_verified_claims: int,
    ) -> tuple[bool, Optional[str], Optional[str], Optional[list[int]]]:
        """Validate model output against the exact presentation-selection contract.

        Enforces:
        - Exact declared keys: {"style", "selected_claims"}. Extra or missing keys rejected.
        - Exact approved styles: {"standard", "compact", "bullet"}. No silent normalization.
        - Selected claims: list of integers within [0, num_verified_claims).
        - Rejects boolean indices (e.g. True/False), non-integers, out-of-range indices.
        - Rejects malformed structures.
        """
        if not isinstance(parsed, dict):
            return False, "contract_violation: model output is not a JSON object", None, None

        declared_keys = {"style", "selected_claims"}
        actual_keys = set(parsed.keys())

        # Reject extra keys
        extra_keys = actual_keys - declared_keys
        if extra_keys:
            return (
                False,
                f"contract_violation: model output contained undeclared extra keys: {sorted(extra_keys)}",
                None,
                None,
            )

        # Reject missing fields
        missing_keys = declared_keys - actual_keys
        if missing_keys:
            return (
                False,
                f"contract_violation: model output missing required declared keys: {sorted(missing_keys)}",
                None,
                None,
            )

        # Validate style: require approved style values exactly; do not silently normalize
        style_val = parsed["style"]
        approved_styles = {"standard", "compact", "bullet"}
        if not isinstance(style_val, str) or style_val not in approved_styles:
            return (
                False,
                f"contract_violation: model selected unapproved style {style_val!r}; expected one of {sorted(approved_styles)}",
                None,
                None,
            )

        # Validate selected_claims: must be a list of ints
        selected_claims = parsed["selected_claims"]
        if not isinstance(selected_claims, list):
            return (
                False,
                f"contract_violation: 'selected_claims' must be a list, got {type(selected_claims).__name__}",
                None,
                None,
            )

        # Reject boolean indices, non-int indices, out-of-range indices
        validated_indices: list[int] = []
        for elem in selected_claims:
            if isinstance(elem, bool) or not isinstance(elem, int):
                return (
                    False,
                    f"contract_violation: claim index must be an integer (boolean or non-int rejected): {elem!r}",
                    None,
                    None,
                )
            if elem < 0 or elem >= num_verified_claims:
                return (
                    False,
                    f"contract_violation: claim index {elem} out of range [0, {num_verified_claims})",
                    None,
                    None,
                )
            validated_indices.append(elem)

        return True, None, style_val, validated_indices

    def run(
        self,
        mode: str,
        query: str,
        item_id: Optional[str] = None,
        k: int = 5,
        top_verify: int = 3,
        requester: Optional[str | Principal] = None,
        timeout_s: Optional[float] = None,
        enable_generation: Optional[bool] = None,
    ) -> AssistantResult:
        """Run the assistant in specified mode ('find' or 'check')."""
        normalized_mode = (mode or "").strip().lower()
        if normalized_mode == "find":
            return self.find(
                query=query,
                k=k,
                top_verify=top_verify,
                requester=requester,
                timeout_s=timeout_s,
                enable_generation=enable_generation,
            )
        elif normalized_mode == "check":
            return self.check(
                item_id=item_id,
                query=query,
                requester=requester,
                timeout_s=timeout_s,
                enable_generation=enable_generation,
            )
        else:
            request_id = f"req-{uuid.uuid4().hex[:12]}"
            err = AssistantModeError(
                f"Unknown mode {mode!r}; expected 'find' or 'check'",
                details={"mode": mode},
            )
            return AssistantResult(
                request_id=request_id,
                mode=mode,
                query=query,
                item_id=item_id,
                status="ERROR",
                action="ABSTAIN",
                decision_status="ABSTAIN",
                reason=err.message,
                error=err.to_dict(),
            )

    def check(
        self,
        item_id: Optional[str],
        query: str,
        requester: Optional[str | Principal] = None,
        timeout_s: Optional[float] = None,
        enable_generation: Optional[bool] = None,
    ) -> AssistantResult:
        """Execute CHECK mode: verify a specific candidate directly without retrieval."""
        start_time = time.perf_counter()
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        gen_enabled = self.enable_generation if enable_generation is None else enable_generation
        principal = resolve_principal(requester)

        timing: dict[str, float] = {
            "parse_ms": 0.0,
            "retrieve_ms": 0.0,
            "verify_ms": 0.0,
            "generate_ms": 0.0,
            "total_ms": 0.0,
        }

        if not item_id:
            err = AssistantInputError("item_id must be provided in check mode")
            timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
            return AssistantResult(
                request_id=request_id,
                mode="check",
                query=query,
                item_id=None,
                status="ERROR",
                action="ABSTAIN",
                decision_status="ABSTAIN",
                reason=err.message,
                timing_ms=timing,
                error=err.to_dict(),
            )

        try:
            self._validate_request(query, effective_timeout)
            self._check_timeout(start_time, effective_timeout)

            # Stage 1: Parse requirements
            t0 = time.perf_counter()
            parse_result = self.verifier.parse_requirements(query)
            timing["parse_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            self._check_timeout(start_time, effective_timeout)

            # Stage 2: Retrieve (0.0 ms in check mode)
            timing["retrieve_ms"] = 0.0

            # Stage 3: Verify Candidate & Resolve Citations
            t0 = time.perf_counter()

            # Enforce tenant isolation on document read
            # If document is private and requester is unauthorized, this raises AuthorizationError
            try:
                self.store.retrieve(item_id, requester=principal)
            except AuthorizationError as exc:
                raise AssistantAuthorizationError(
                    f"Read authorization denied for principal {principal.id!r} on item {item_id!r}",
                    details={"item_id": item_id, "principal": principal.id},
                ) from exc

            raw_decision = self.verifier.verify_candidate(item_id, parse_result, query)
            decision, resolved_citations, citation_diagnostics = self._validate_and_filter_citations(
                raw_decision, principal
            )

            timing["verify_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            self._check_timeout(start_time, effective_timeout)

            # Stage 4: Generate Answer (deterministic or guarded generation)
            t0 = time.perf_counter()
            title = item_id
            # Try to get nicer title from catalog if known
            if hasattr(self.verifier, "catalog_items"):
                cat_item = getattr(self.verifier, "catalog_items", {}).get(item_id)
                if cat_item and "title" in cat_item:
                    title = cat_item["title"]

            answer, fallback_reason = self._generate_answer(
                title=title,
                decision=decision,
                citations=resolved_citations,
                enable_generation=gen_enabled,
                start_time=start_time,
                timeout_s=effective_timeout,
            )
            timing["generate_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            # Deadline verification before success return
            self._check_timeout(start_time, effective_timeout)

            timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)

            return AssistantResult(
                request_id=request_id,
                mode="check",
                query=query,
                item_id=item_id,
                status="OK",
                decision=decision,
                action=decision["action"],
                decision_status=decision["status"],
                reason=decision["reason"],
                candidates=[],
                verified_candidates=[{
                    "item_id": item_id,
                    "title": title,
                    "decision": decision,
                    "citations": resolved_citations,
                    "citation_diagnostics": citation_diagnostics,
                }],
                citations=resolved_citations,
                citation_diagnostics=citation_diagnostics,
                answer=answer,
                timing_ms=timing,
                generation_fallback_reason=fallback_reason,
            )

        except AssistantError as err:
            timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
            return AssistantResult(
                request_id=request_id,
                mode="check",
                query=query,
                item_id=item_id,
                status="ERROR",
                action="ABSTAIN",
                decision_status="ABSTAIN",
                reason=err.message,
                timing_ms=timing,
                error=err.to_dict(),
            )
        except Exception as exc:
            timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
            err = AssistantError(f"Unexpected internal error: {exc}", details={"type": type(exc).__name__})
            return AssistantResult(
                request_id=request_id,
                mode="check",
                query=query,
                item_id=item_id,
                status="ERROR",
                action="ABSTAIN",
                decision_status="ABSTAIN",
                reason=str(exc),
                timing_ms=timing,
                error=err.to_dict(),
            )

    def find(
        self,
        query: str,
        k: int = 5,
        top_verify: int = 3,
        requester: Optional[str | Principal] = None,
        timeout_s: Optional[float] = None,
        enable_generation: Optional[bool] = None,
    ) -> AssistantResult:
        """Execute FIND mode: retrieve candidates, verify evidence, and select recommendation."""
        start_time = time.perf_counter()
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        gen_enabled = self.enable_generation if enable_generation is None else enable_generation
        principal = resolve_principal(requester)

        timing: dict[str, float] = {
            "parse_ms": 0.0,
            "retrieve_ms": 0.0,
            "verify_ms": 0.0,
            "generate_ms": 0.0,
            "total_ms": 0.0,
        }

        try:
            self._validate_request(query, effective_timeout, k, top_verify)
            self._check_timeout(start_time, effective_timeout)

            # Check for image queries: refuse visual dimension inference without grounded visual assets
            if _IMAGE_QUERY_RE.search(query):
                t0 = time.perf_counter()
                now_iso = datetime.now(timezone.utc).isoformat()
                decision = {
                    "schema_version": "2",
                    "decision_id": f"dec-{uuid.uuid4().hex[:12]}",
                    "query": query,
                    "item_id": None,
                    "status": "ABSTAIN",
                    "action": "ABSTAIN",
                    "reason": "Image assets unavailable; visual specification inference refused",
                    "requirements": [],
                    "claims": [],
                    "unparsed_clauses": [query],
                    "ambiguities": [],
                    "created_at": now_iso,
                }
                timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
                return AssistantResult(
                    request_id=request_id,
                    mode="find",
                    query=query,
                    item_id=None,
                    status="ABSTAIN",
                    decision=decision,
                    action="ABSTAIN",
                    decision_status="ABSTAIN",
                    reason=decision["reason"],
                    candidates=[],
                    verified_candidates=[],
                    citations=[],
                    answer="Abstain: Image assets unavailable; visual specification inference refused without verified ground truth.",
                    timing_ms=timing,
                    generation_fallback_reason=None,
                )

            # Stage 1: Parse requirements
            t0 = time.perf_counter()
            parse_result = self.verifier.parse_requirements(query)
            timing["parse_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            self._check_timeout(start_time, effective_timeout)

            # Stage 2: Retrieve candidates and enforce authorization boundary
            t0 = time.perf_counter()
            if self.retriever is None:
                raise AssistantStoreError("No catalog retriever available for find mode")
            raw_candidates = self.retriever.retrieve(query, k=k)

            # Filter every candidate by authorization against evidence store
            # Unauthorized private items must NOT appear in candidates, verified_candidates,
            # citations, metadata, or generated context.
            authorized_candidates: list[dict[str, Any]] = []
            for cand in raw_candidates:
                cid = cand["item_id"]
                try:
                    self.store.retrieve(cid, requester=principal)
                    authorized_candidates.append(cand)
                except AuthorizationError:
                    continue

            candidates = authorized_candidates
            timing["retrieve_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            self._check_timeout(start_time, effective_timeout)

            # Stage 3: Verify top candidates & Resolve Citations
            t0 = time.perf_counter()
            verified_candidates: list[dict[str, Any]] = []
            primary_candidate: Optional[dict[str, Any]] = None

            if not candidates:
                no_cand_decision = self.verifier.verify_candidate(None, parse_result, query)
                timing["verify_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                answer, fallback_reason = self._generate_answer(
                    title="No catalog items matched query",
                    decision=no_cand_decision,
                    citations=[],
                    enable_generation=gen_enabled,
                    start_time=start_time,
                    timeout_s=effective_timeout,
                )
                self._check_timeout(start_time, effective_timeout)
                timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
                return AssistantResult(
                    request_id=request_id,
                    mode="find",
                    query=query,
                    item_id=None,
                    status="OK",
                    decision=no_cand_decision,
                    action=no_cand_decision["action"],
                    decision_status=no_cand_decision["status"],
                    reason=no_cand_decision["reason"],
                    candidates=[],
                    verified_candidates=[],
                    citations=[],
                    answer=answer,
                    timing_ms=timing,
                    generation_fallback_reason=fallback_reason,
                )

            for cand in candidates[:top_verify]:
                cid = cand["item_id"]
                title = cand.get("title", cid)

                raw_dec = self.verifier.verify_candidate(cid, parse_result, query)
                dec, resolved_cites, cand_diag = self._validate_and_filter_citations(raw_dec, principal)

                entry = {
                    "item_id": cid,
                    "title": title,
                    "decision": dec,
                    "citations": resolved_cites,
                    "citation_diagnostics": cand_diag,
                }
                verified_candidates.append(entry)

                # Select first verified candidate that achieves RECOMMEND
                if dec["action"] == "RECOMMEND" and primary_candidate is None:
                    primary_candidate = entry

            # If none reached RECOMMEND, fallback to first verified candidate
            if primary_candidate is None and verified_candidates:
                primary_candidate = verified_candidates[0]

            if primary_candidate is None:
                # All candidates were inaccessible due to authorization
                no_cand_decision = self.verifier.verify_candidate(None, parse_result, query)
                no_cand_decision["reason"] = "No accessible candidates matched query under authorization boundary"
                primary_candidate = {
                    "item_id": None,
                    "title": "No accessible catalog items",
                    "decision": no_cand_decision,
                    "citations": [],
                    "citation_diagnostics": [],
                }

            timing["verify_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            self._check_timeout(start_time, effective_timeout)

            # Stage 4: Generate Answer
            t0 = time.perf_counter()
            chosen_decision = primary_candidate["decision"]
            chosen_title = primary_candidate["title"]
            chosen_citations = primary_candidate["citations"]
            chosen_diagnostics = primary_candidate.get("citation_diagnostics", [])
            chosen_item_id = primary_candidate["item_id"]

            answer, fallback_reason = self._generate_answer(
                title=chosen_title,
                decision=chosen_decision,
                citations=chosen_citations,
                enable_generation=gen_enabled,
                start_time=start_time,
                timeout_s=effective_timeout,
            )
            timing["generate_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            # Deadline verification before success return
            self._check_timeout(start_time, effective_timeout)

            timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)

            return AssistantResult(
                request_id=request_id,
                mode="find",
                query=query,
                item_id=chosen_item_id,
                status="OK",
                decision=chosen_decision,
                action=chosen_decision["action"],
                decision_status=chosen_decision["status"],
                reason=chosen_decision["reason"],
                candidates=candidates,
                verified_candidates=verified_candidates,
                citations=chosen_citations,
                citation_diagnostics=chosen_diagnostics,
                answer=answer,
                timing_ms=timing,
                generation_fallback_reason=fallback_reason,
            )

        except AssistantError as err:
            timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
            return AssistantResult(
                request_id=request_id,
                mode="find",
                query=query,
                item_id=None,
                status="ERROR",
                action="ABSTAIN",
                decision_status="ABSTAIN",
                reason=err.message,
                timing_ms=timing,
                error=err.to_dict(),
            )
        except Exception as exc:
            timing["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 3)
            err = AssistantError(f"Unexpected internal error: {exc}", details={"type": type(exc).__name__})
            return AssistantResult(
                request_id=request_id,
                mode="find",
                query=query,
                item_id=None,
                status="ERROR",
                action="ABSTAIN",
                decision_status="ABSTAIN",
                reason=str(exc),
                timing_ms=timing,
                error=err.to_dict(),
            )

    def _generate_answer(
        self,
        title: str,
        decision: dict[str, Any],
        citations: list[dict[str, Any]],
        enable_generation: bool,
        start_time: float,
        timeout_s: Optional[float],
    ) -> tuple[str, Optional[str]]:
        """Produce answer text: deterministic baseline or contract-guarded generation."""
        # 1. Deterministic baseline
        deterministic_answer = self.generator.generate(title, decision)

        if not enable_generation:
            return deterministic_answer, None

        # 2. Check runner availability
        runner = self.llm_runner
        if runner is None:
            # Try to discover GGUFModelRunner if available
            try:
                from qedge_runtime.quantization.runner import GGUFModelRunner
                models_root = resolve_llm_inference_root() / "models" / "smollm2-135m-instruct-gguf"
                gguf_path = models_root / "smollm2-135m-instruct-q4_k_m.gguf"
                if gguf_path.is_file():
                    runner = GGUFModelRunner(str(gguf_path))
                    self.llm_runner = runner
            except Exception:
                runner = None

        if runner is None:
            return (
                deterministic_answer,
                "generation_runtime_unavailable: no GGUFModelRunner configured or available in environment",
            )

        # 3. Context preparation with strict data/instruction separation and bounding
        # System instructions must remain strictly constant with no untrusted user/catalog data.
        valid_citations = {c["citation"] for c in citations if c.get("valid", True)}
        verified_claims = decision.get("claims", [])

        claims_entries = []
        for idx, c in enumerate(verified_claims):
            attr = c.get("attribute", "")
            val = c.get("observed_value", "")
            unit = c.get("observed_unit") or ""
            st = c.get("status", "")
            claims_entries.append(f"[{idx}] {attr}: {val}{unit} ({st})")

        claims_summary = "\n".join(claims_entries) if claims_entries else "None."
        action = decision.get("action", "ABSTAIN")
        status = decision.get("status", "UNKNOWN")
        reason = decision.get("reason", "")

        # Strict data/instruction separation:
        # Untrusted title and catalog evidence are NEVER in the instruction text.
        # Bounded context: untrusted evidence text bounded to 500 characters.
        system_instructions = (
            "You are a constrained presentation selector for QEdge Assist.\n"
            "Select an approved presentation style ('standard', 'compact', 'bullet') and "
            "the indices of verified claims to highlight.\n"
            "You must output ONLY valid JSON matching this schema:\n"
            '{"style": "standard"|"compact"|"bullet", "selected_claims": [<int>, ...]}\n'
            "Rules:\n"
            "1. Output ONLY valid JSON matching the schema.\n"
            "2. Never output arbitrary free text, invented facts, or unverified claims.\n"
            "3. Any output outside this schema violates the presentation contract."
        )

        untrusted_data_section = (
            "--- UNTRUSTED CATALOG DATA (Bounded) ---\n"
            f"Verdict: {action} ({status})\n"
            f"Reason: {reason[:150]}\n"
            f"Verified Claims:\n{claims_summary[:350]}\n"
            "---------------------------------------"
        )

        prompt = f"{system_instructions}\n\n{untrusted_data_section}\n\nJSON Output:"

        try:
            self._check_timeout(start_time, timeout_s)
            remaining_timeout = max(0.0, timeout_s - (time.perf_counter() - start_time)) if timeout_s else None

            # Propagate deadline/timeout to runner if supported
            try:
                gen_res = runner.generate(prompt, max_tokens=60, timeout_s=remaining_timeout)
            except TypeError:
                gen_res = runner.generate(prompt, max_tokens=60)

            # Check deadline immediately upon return from generation
            self._check_timeout(start_time, timeout_s)

            gen_text = gen_res.get("text", "") if isinstance(gen_res, dict) else str(gen_res)
            gen_text = gen_text.strip()

            if not gen_text:
                return deterministic_answer, "generation_empty_output"

            # Check for hallucinated citations in any returned output
            cited_in_gen = set(_CITATION_EXTRACT_RE.findall(gen_text))
            if not cited_in_gen.issubset(valid_citations):
                hallucinated = cited_in_gen - valid_citations
                return (
                    deterministic_answer,
                    f"contract_violation: model introduced unverified citation(s): {hallucinated}",
                )

            # Check for recommendation violation if non-RECOMMEND
            if action != "RECOMMEND":
                recom_re = re.compile(
                    r"(?<!do not )(?<!not )(?<!cannot )\b(recommend|recommended|suggest|buy|buying|great choice|highly recommend|meets every requirement)\b",
                    re.IGNORECASE,
                )
                if recom_re.search(gen_text):
                    return (
                        deterministic_answer,
                        "contract_violation: model attempted to recommend non-recommended candidate",
                    )

            # Enforce constrained presentation selection contract
            parsed = None
            try:
                clean_json_str = gen_text
                if "```json" in clean_json_str:
                    clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json_str:
                    clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()
                parsed = json.loads(clean_json_str)
            except Exception:
                parsed = None

            if parsed is None:
                # Free text output (invented facts, unconstrained prose, etc.)
                return (
                    deterministic_answer,
                    "contract_violation: model output is not valid JSON matching presentation contract",
                )

            is_valid, violation_reason, style, selected_indices = self._validate_presentation_selection(
                parsed, len(verified_claims)
            )
            if not is_valid:
                return deterministic_answer, violation_reason

            # Render ONLY validated claims with verified evidence
            filtered_claims = [verified_claims[i] for i in selected_indices]
            sub_decision = dict(decision)
            sub_decision["claims"] = filtered_claims

            if style == "bullet":
                rendered_lines = [f"Item: {title}"]
                rendered_lines.append(f"Verdict: {action} ({status})")
                for c in filtered_claims:
                    val = c.get("observed_value")
                    unit = c.get("observed_unit") or ""
                    rendered_lines.append(f"  • {c.get('attribute')}: {val}{unit} ({c.get('status')})")
                rendered_answer = "\n".join(rendered_lines)
            elif style == "compact":
                claim_parts = [
                    f"{c.get('attribute')}={c.get('observed_value')}{c.get('observed_unit') or ''}"
                    for c in filtered_claims
                ]
                rendered_answer = f"{action} {title} [{', '.join(claim_parts)}]"
            else:
                rendered_answer = self.generator.generate(title, sub_decision)

            return rendered_answer, None

        except AssistantTimeoutError:
            raise
        except Exception as exc:
            return deterministic_answer, f"generation_error: {str(exc)}"

    def format_output(self, result: AssistantResult) -> str:
        """Render human-readable CLI output with governance and timing."""
        lines = []
        div = "═" * 64
        sub_div = "─" * 40
        lines.append(div)
        lines.append(f" QEdge Assist — Local Assistant (Stage 10)")
        lines.append(div)
        lines.append(f"Request ID: {result.request_id}")
        lines.append(f"Mode:       {result.mode.upper()}")
        lines.append(f'Query:      "{result.query}"')
        if result.item_id:
            lines.append(f"Item ID:    {result.item_id}")
        lines.append("")

        dec = result.decision
        status = result.decision_status
        action = result.action
        lines.append(f"Verdict:    {status} → {action}")
        if result.reason:
            lines.append(f"Reason:     {result.reason}")
        if dec.get("unparsed_clauses"):
            lines.append(f"Unparsed:   {dec['unparsed_clauses']}")
        if dec.get("ambiguities"):
            lines.append(f"Ambiguity:  {dec['ambiguities']}")
        lines.append("")

        # Stage Timing Table
        lines.append(f"{'Stage':<14}{'Time (ms)':<12}")
        lines.append(sub_div)
        for stage_name in ("parse_ms", "retrieve_ms", "verify_ms", "generate_ms"):
            val = result.timing_ms.get(stage_name, 0.0)
            clean_name = stage_name.replace("_ms", "").upper()
            lines.append(f"{clean_name:<14}{val:<12.1f}")
        total = result.timing_ms.get("total_ms", 0.0)
        lines.append(f"{'TOTAL':<14}{total:<12.1f}")
        lines.append("")

        # Verified Claims (only validated claims with verified evidence)
        claims = dec.get("claims", [])
        lines.append("Verified Claims:")
        if claims:
            for c in claims:
                attr = c.get("attribute")
                st = c.get("status")
                val = c.get("observed_value")
                unit = c.get("observed_unit") or ""
                ev = c.get("evidence_text", "")
                val_str = f" = {val}{unit}" if val is not None else ""
                lines.append(f"  • {attr}{val_str} [{st}] → {ev}")
        else:
            lines.append("  (No verified claims available)")
        lines.append("")

        # Resolved Citations (only successfully verified citations)
        if result.citations:
            lines.append("Resolved Citations:")
            for cite in result.citations:
                c_str = cite.get("citation", "")
                lines.append(f"  • {c_str} [VERIFIED]")
            lines.append("")

        # Diagnostic Details for Failed Citations (kept strictly separate from validated evidence)
        if result.citation_diagnostics:
            lines.append("Citation Diagnostics (Failed Verification):")
            for cite in result.citation_diagnostics:
                c_str = cite.get("citation", "")
                err = cite.get("error", "Unknown citation error")
                lines.append(f"  • {c_str} [FAILED: {err}]")
            lines.append("")

        # Answer
        lines.append("Response:")
        lines.append(result.answer)
        if result.generation_fallback_reason:
            lines.append(f"[Generation Fallback Reason: {result.generation_fallback_reason}]")
        lines.append("")

        # Governance & Hardware status
        lines.append(
            "Hardware status: NPU unverified — CPU baseline execution; no Hexagon NPU offload."
        )
        lines.append(
            "Governance: strictly deterministic predicate resolution; untrusted catalog data bounded."
        )
        lines.append(div)
        return "\n".join(lines)
