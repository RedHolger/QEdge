"""Evidence verification engine backed by the local SQLite evidence store.

Enforces:
- Structured v2 requirement evaluation against SQLite evidence chunks.
- Asymmetric material taxonomy (oak implies wood, wood cannot certify oak).
- Strict and inclusive dimension comparisons without premature rounding.
- Product dimensions distinct from package dimensions.
- Unit normalization (cm, in, mm, m, ft) without rounding.
- Detection and propagation of conflicting source facts leading to ABSTAIN.
- Emission of schema-validated contract v2 decisions and evidence claims.
- Revision-resolving citations for every verified fact.
- Directory and JSONL path support in load_catalog.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Optional
import uuid

from .paths import resolve_catalog_file
from .requirements import ParseResult, ParsedRequirement, parse_query_requirements
from .store import EvidenceStore, Fact, Principal
from .taxonomy import convert_length, evaluate_brand, evaluate_color, evaluate_material

try:
    from qedge_common.schema import validate as validate_contract
except ImportError:
    validate_contract = None

_LENGTH_TO_CM = {
    "cm": 1.0,
    "centimeter": 1.0,
    "centimeters": 1.0,
    "mm": 0.1,
    "millimeter": 0.1,
    "millimeters": 0.1,
    "m": 100.0,
    "meter": 100.0,
    "meters": 100.0,
    "in": 2.54,
    "inch": 2.54,
    "inches": 2.54,
    '"': 2.54,
    "ft": 30.48,
    "foot": 30.48,
    "feet": 30.48,
    "'": 30.48,
}


def convert_length(value: float, from_unit: Optional[str], to_unit: Optional[str]) -> Optional[float]:
    """Convert length value between units without rounding. Return None if incompatible."""
    if from_unit is None or to_unit is None:
        return None
    from_norm = from_unit.strip().lower()
    to_norm = to_unit.strip().lower()
    if from_norm not in _LENGTH_TO_CM or to_norm not in _LENGTH_TO_CM:
        return None
    cm_val = value * _LENGTH_TO_CM[from_norm]
    return cm_val / _LENGTH_TO_CM[to_norm]


class Verifier:
    """Verifies candidate product claims against on-device SQLite evidence chunks."""

    def __init__(
        self,
        catalog_path: Optional[str | Path] = None,
        store: Optional[Any] = None,
        auto_load: bool = True,
    ):
        self.store = store if store is not None else EvidenceStore(":memory:")
        self.item_to_variant: dict[str, str] = {}
        self.verified_count = 0

        if auto_load and catalog_path:
            self.load_catalog(catalog_path)

    def load_catalog(self, catalog_path: str | Path) -> int:
        """Pre-load ABO catalog items into in-memory SQLite store.

        Accepts either a catalog directory (containing catalog.jsonl) or a direct
        catalog.jsonl file path.
        """
        path = resolve_catalog_file(catalog_path)
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                item_id = row.get("item_id")
                variant_id = row.get("variant_id", item_id)
                if not item_id:
                    continue

                self.item_to_variant[item_id] = variant_id

                chunks: list[str] = []
                facts: list[Fact] = []
                attrs = row.get("attributes", {})
                source_rev = row.get("source_revision") or "unknown"

                # Process all present attributes in catalog row
                for attr_name, info in attrs.items():
                    if not isinstance(info, dict):
                        continue
                    status = info.get("status")
                    if status != "present":
                        continue
                    val = info.get("value")
                    if val is None:
                        continue

                    unit = info.get("unit")
                    src_loc = info.get("source_location") or ("package_dimensions" if "package" in attr_name else "item_dimensions" if attr_name in ("width", "height", "length") else attr_name)
                    src_rev = info.get("source_revision") or source_rev

                    # Type normalization for facts
                    if attr_name in ("width", "height", "length", "package_width", "package_height", "package_length"):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            continue
                    elif attr_name in ("material", "color"):
                        val = str(val).strip().lower()
                    else:
                        val = str(val).strip()

                    u_label = f" {unit}" if unit else ""
                    chunks.append(f"{attr_name}: {val}{u_label} (source: {src_loc})")
                    facts.append(
                        Fact(
                            attribute=attr_name,
                            value=val,
                            unit=unit,
                            status="present",
                            source_location=src_loc,
                            source_revision=src_rev,
                        )
                    )

                if not chunks:
                    t_val = str(row.get("title", ""))
                    chunks.append(f"title: {t_val} (source: catalog)")
                    facts.append(
                        Fact(
                            attribute="title",
                            value=t_val,
                            unit=None,
                            status="present",
                            source_location="catalog",
                            source_revision=source_rev,
                        )
                    )

                sys_principal = Principal.system()
                self.store.upload(
                    variant_id,
                    sys_principal,
                    chunks,
                    is_public=True,
                    facts=facts,
                    item_id=item_id,
                    variant_id=variant_id,
                    source_location="catalog",
                    source_revision=source_rev,
                )
                if variant_id != item_id:
                    self.store.upload(
                        item_id,
                        sys_principal,
                        chunks,
                        is_public=True,
                        facts=facts,
                        item_id=item_id,
                        variant_id=variant_id,
                        source_location="catalog",
                        source_revision=source_rev,
                    )
                count += 1

        return count

    def parse_requirements(self, query: str) -> ParseResult:
        """Extract structured attribute requirements, unparsed clauses, and ambiguities."""
        return parse_query_requirements(query)

    def _get_candidate_chunks(self, item_id: str) -> list[dict[str, Any]]:
        """Retrieve all active chunks for candidate item or its variant."""
        resolved_id = self.item_to_variant.get(item_id, item_id)
        chunks = self.store.retrieve(resolved_id, Principal.system())
        if not chunks and resolved_id != item_id:
            chunks = self.store.retrieve(item_id, Principal.system())
        return chunks

    def evaluate_requirement_against_chunks(
        self,
        req: ParsedRequirement,
        candidate_item_id: str,
        chunks: list[dict[str, Any]],
        now_iso: str,
    ) -> list[dict[str, Any]]:
        """Evaluate a single requirement against candidate chunks, emitting v2 claims.

        Returns a list of claim dictionaries.
        """
        attr = req.attribute
        target_op = req.operator
        req_val = req.required_value
        req_unit = req.required_unit

        # Filter relevant chunks
        matching_chunks = []
        for c in chunks:
            c_attr = c.get("attribute")
            if c_attr == attr:
                matching_chunks.append(c)
            elif not c_attr:
                # Text fallback for legacy chunks
                text = c.get("text", "")
                if re.search(rf"\b{re.escape(attr)}:\s*", text, re.IGNORECASE):
                    matching_chunks.append(c)

        # If no chunks found for this attribute
        if not matching_chunks:
            return [
                {
                    "schema_version": "2",
                    "claim_id": f"claim-{req.requirement_id}-none",
                    "item_id": candidate_item_id,
                    "requirement_id": req.requirement_id,
                    "attribute": attr,
                    "operator": target_op,
                    "required_value": req_val,
                    "required_unit": req_unit,
                    "observed_value": None,
                    "observed_unit": None,
                    "status": "UNKNOWN",
                    "source": "evidence_store",
                    "source_revision": "none",
                    "evidence_text": f"No stored evidence found for attribute {attr!r}",
                    "retrieved_at": now_iso,
                }
            ]

        claims = []
        for chunk in matching_chunks:
            chunk_idx = chunk.get("chunk_idx", 0)
            claim_id = f"claim-{req.requirement_id}-{chunk_idx}"
            src_loc = chunk.get("source_location") or "evidence_store"
            src_rev = chunk.get("source_revision") or "unknown"
            citation = chunk.get("citation", "")
            raw_text = chunk.get("text", "")
            ev_text = f"{raw_text} [{citation}]" if citation else raw_text

            chunk_attr = chunk.get("attribute")
            chunk_status = chunk.get("status")

            # Structured fact with non-present status must NOT support or contradict
            if chunk_attr is not None and chunk_status != "present":
                ev_text = f"{ev_text}; evidence status is non-present ({chunk_status!r})"
                claims.append(
                    {
                        "schema_version": "2",
                        "claim_id": claim_id,
                        "item_id": candidate_item_id,
                        "requirement_id": req.requirement_id,
                        "attribute": attr,
                        "operator": target_op,
                        "required_value": req_val,
                        "required_unit": req_unit,
                        "observed_value": None,
                        "observed_unit": None,
                        "status": "UNKNOWN",
                        "source": src_loc,
                        "source_revision": src_rev,
                        "evidence_text": ev_text,
                        "retrieved_at": now_iso,
                    }
                )
                continue

            obs_val = chunk.get("value")
            obs_unit = chunk.get("unit")

            # Fallback parsing from text ONLY for unstructured legacy chunks (no structured attribute)
            if chunk_attr is None and obs_val is None:
                m_dim = re.search(rf"\b{re.escape(attr)}:\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?", raw_text)
                if m_dim:
                    obs_val = float(m_dim.group(1))
                    obs_unit = m_dim.group(2)
                else:
                    m_cat = re.search(rf"\b{re.escape(attr)}:\s*([^(]+)", raw_text)
                    if m_cat:
                        obs_val = m_cat.group(1).strip()
                        obs_unit = None

            status = "UNKNOWN"
            norm_obs_val = obs_val
            norm_obs_unit = obs_unit

            # 1. Dimensions
            if attr in ("width", "height", "length", "package_width", "package_height", "package_length"):
                if obs_val is not None:
                    try:
                        f_obs = float(obs_val)
                        converted = convert_length(f_obs, obs_unit, req_unit)
                        if converted is None:
                            status = "UNKNOWN"
                            norm_obs_val = None
                            norm_obs_unit = None
                        else:
                            norm_obs_val = converted
                            norm_obs_unit = req_unit
                            f_req = float(req_val)
                            if target_op == "<":
                                match = converted < f_req
                            elif target_op == "<=":
                                match = converted <= f_req
                            elif target_op == ">":
                                match = converted > f_req
                            elif target_op == ">=":
                                match = converted >= f_req
                            elif target_op == "==":
                                match = converted == f_req
                            elif target_op == "!=":
                                match = converted != f_req
                            else:
                                match = False
                            status = "SUPPORTED" if match else "CONTRADICTED"
                    except (ValueError, TypeError):
                        status = "UNKNOWN"
                        norm_obs_val = None
                        norm_obs_unit = None

            # 2. Material: preserve actual source observation obs_val
            elif attr == "material":
                if obs_val is not None:
                    status, detail = evaluate_material(target_op, str(req_val), str(obs_val))
                    norm_obs_val = obs_val if status in ("SUPPORTED", "CONTRADICTED") else None
                    norm_obs_unit = None
                    if status in ("SUPPORTED", "CONTRADICTED"):
                        ev_text = f"{ev_text}; taxonomy: {detail}"

            # 3. Color: preserve actual source observation obs_val
            elif attr == "color":
                if obs_val is not None:
                    status, detail = evaluate_color(target_op, str(req_val), str(obs_val))
                    norm_obs_val = obs_val if status in ("SUPPORTED", "CONTRADICTED") else None
                    norm_obs_unit = None

            # 4. Brand: preserve actual source observation obs_val
            elif attr == "brand":
                if obs_val is not None:
                    status, detail = evaluate_brand(target_op, str(req_val), str(obs_val))
                    norm_obs_val = obs_val if status in ("SUPPORTED", "CONTRADICTED") else None
                    norm_obs_unit = None

            # 5. Other categoricals: preserve actual source observation obs_val
            else:
                if obs_val is not None:
                    str_obs = str(obs_val).strip()
                    str_req = str(req_val).strip()
                    if target_op == "==":
                        status = "SUPPORTED" if str_req.casefold() == str_obs.casefold() else "CONTRADICTED"
                    elif target_op == "!=":
                        status = "CONTRADICTED" if str_req.casefold() == str_obs.casefold() else "SUPPORTED"
                    elif target_op == "contains":
                        status = "SUPPORTED" if str_req.casefold() in str_obs.casefold() else "CONTRADICTED"
                    elif target_op == "not_contains":
                        status = "CONTRADICTED" if str_req.casefold() in str_obs.casefold() else "SUPPORTED"
                    else:
                        status = "UNKNOWN"
                    norm_obs_val = obs_val if status in ("SUPPORTED", "CONTRADICTED") else None
                    norm_obs_unit = None

            # Ensure contract compliance: if UNKNOWN, observed_value can be null
            if status == "UNKNOWN":
                norm_obs_val = None
                norm_obs_unit = None

            claims.append(
                {
                    "schema_version": "2",
                    "claim_id": claim_id,
                    "item_id": candidate_item_id,
                    "requirement_id": req.requirement_id,
                    "attribute": attr,
                    "operator": target_op,
                    "required_value": req_val,
                    "required_unit": req_unit,
                    "observed_value": norm_obs_val,
                    "observed_unit": norm_obs_unit,
                    "status": status,
                    "source": src_loc,
                    "source_revision": src_rev,
                    "evidence_text": ev_text,
                    "retrieved_at": now_iso,
                }
            )

        return claims

    def verify_candidate(
        self,
        candidate_item_id: Optional[str],
        parse_result: ParseResult,
        query: str,
    ) -> dict[str, Any]:
        """Verify candidate item against parsed requirements, emitting validated v2 decision."""
        now_iso = datetime.now(timezone.utc).isoformat()
        decision_id = f"dec-{uuid.uuid4().hex[:12]}"
        unparsed = list(parse_result.unparsed_clauses)
        ambiguities = list(parse_result.ambiguities)
        contract_reqs = parse_result.to_contract_requirements()

        # 1. No candidate retrieved or no structured requirements
        if candidate_item_id is None or not contract_reqs:
            if candidate_item_id is None:
                reason = "No candidate item matched the query scope"
            elif unparsed or ambiguities:
                reason = f"Unresolved query clauses ({len(unparsed)}) or ambiguities ({len(ambiguities)}) present"
            else:
                reason = "No structured requirements extracted from query"
            decision = {
                "schema_version": "2",
                "decision_id": decision_id,
                "query": query,
                "item_id": None,
                "status": "ABSTAIN",
                "action": "ABSTAIN",
                "reason": reason,
                "requirements": contract_reqs,
                "claims": [],
                "unparsed_clauses": unparsed,
                "ambiguities": ambiguities,
                "created_at": now_iso,
            }
            if validate_contract:
                validate_contract("decision", decision, version="2")
            return decision

        chunks = self._get_candidate_chunks(candidate_item_id)
        all_claims: list[dict[str, Any]] = []

        for req in parse_result.requirements:
            req_claims = self.evaluate_requirement_against_chunks(
                req, candidate_item_id, chunks, now_iso
            )
            all_claims.extend(req_claims)

        # Apply deterministic decision precedence per CONTRACTS.md:
        # Precedence 1: No requirements -> ABSTAIN
        if not contract_reqs:
            status = "ABSTAIN"
            action = "ABSTAIN"
            reason = "No structured requirements could be extracted from query"

        # Precedence 2: Unresolved clauses or ambiguities -> ABSTAIN
        elif unparsed or ambiguities:
            status = "ABSTAIN"
            action = "ABSTAIN"
            reason = f"Unresolved query clauses ({len(unparsed)}) or ambiguities ({len(ambiguities)}) present"

        else:
            # Check for conflicting evidence for the same requirement
            statuses_by_req: dict[str, set[str]] = {}
            for claim in all_claims:
                statuses_by_req.setdefault(claim["requirement_id"], set()).add(claim["status"])

            has_conflict = any(
                "SUPPORTED" in st and "CONTRADICTED" in st
                for st in statuses_by_req.values()
            )
            if has_conflict:
                status = "ABSTAIN"
                action = "ABSTAIN"
                reason = "Conflicting evidence claims (both SUPPORTED and CONTRADICTED) observed for the same requirement"

            # Precedence 3: Contradiction -> CONTRADICTED / DO_NOT_RECOMMEND
            elif any("CONTRADICTED" in st for st in statuses_by_req.values()):
                status = "CONTRADICTED"
                action = "DO_NOT_RECOMMEND"
                reason = "One or more requirements contradicted by stored evidence"

            # Precedence 4: Supported -> complete mandatory coverage & all claims SUPPORTED
            else:
                mandatory_ids = {r["requirement_id"] for r in contract_reqs if r.get("mandatory", True)}
                covered_mandatory = all(rid in statuses_by_req for rid in mandatory_ids)
                all_supported = bool(all_claims) and all(c["status"] == "SUPPORTED" for c in all_claims)

                if covered_mandatory and all_supported:
                    status = "SUPPORTED"
                    action = "RECOMMEND"
                    reason = "All mandatory requirements verified and supported by stored evidence"
                else:
                    # Precedence 5: Missing or UNKNOWN evidence -> UNKNOWN / ABSTAIN
                    status = "UNKNOWN"
                    action = "ABSTAIN"
                    reason = "Insufficient or missing evidence to satisfy all mandatory requirements"

        decision = {
            "schema_version": "2",
            "decision_id": decision_id,
            "query": query,
            "item_id": candidate_item_id,
            "status": status,
            "action": action,
            "reason": reason,
            "requirements": contract_reqs,
            "claims": all_claims,
            "unparsed_clauses": unparsed,
            "ambiguities": ambiguities,
            "created_at": now_iso,
        }

        if validate_contract:
            validate_contract("decision", decision, version="2")

        self.verified_count += 1
        return decision

    # Backward-compatible helper methods for existing tests
    def verify_attribute(
        self, doc_id: str, attribute: str, requirement: Any
    ) -> tuple[str, Optional[str]]:
        """Legacy helper for backward compatibility with existing tests."""
        chunks = self._get_candidate_chunks(doc_id)
        if not chunks:
            return "UNKNOWN", None

        # Build synthetic requirement
        if attribute in ("width", "width_lte_cm"):
            req = ParsedRequirement(
                requirement_id="req-legacy-width",
                attribute="width",
                operator="<=",
                required_value=float(requirement),
                required_unit="cm",
                mandatory=True,
                source_span=(0, 0),
                raw_text="",
            )
        elif attribute == "material":
            req = ParsedRequirement(
                requirement_id="req-legacy-material",
                attribute="material",
                operator="==",
                required_value=str(requirement).strip().lower(),
                required_unit=None,
                mandatory=True,
                source_span=(0, 0),
                raw_text="",
            )
        elif attribute == "color":
            req = ParsedRequirement(
                requirement_id="req-legacy-color",
                attribute="color",
                operator="==",
                required_value=str(requirement).strip().lower(),
                required_unit=None,
                mandatory=True,
                source_span=(0, 0),
                raw_text="",
            )
        else:
            return "UNKNOWN", None

        now_iso = datetime.now(timezone.utc).isoformat()
        claims = self.evaluate_requirement_against_chunks(req, doc_id, chunks, now_iso)
        if not claims:
            return "UNKNOWN", None

        # If any contradicted, return CONTRADICTED
        for cl in claims:
            if cl["status"] == "CONTRADICTED":
                obs = cl.get("observed_value")
                u = cl.get("observed_unit")
                detail = f"{int(round(obs))}cm" if u == "cm" and isinstance(obs, (int, float)) else str(obs)
                return "CONTRADICTED", detail

        # If any supported
        for cl in claims:
            if cl["status"] == "SUPPORTED":
                obs = cl.get("observed_value")
                u = cl.get("observed_unit")
                detail = f"{int(round(obs))}cm" if u == "cm" and isinstance(obs, (int, float)) else str(obs)
                return "SUPPORTED", detail

        return "UNKNOWN", None

    def verify(
        self, doc_id: str, requirements: Any
    ) -> dict[str, tuple[str, Optional[str]]]:
        """Legacy verify method returning mapping attribute -> (status, detail)."""
        results = {}
        if isinstance(requirements, dict):
            for attr, val in requirements.items():
                if attr == "width_lte_cm":
                    results["width"] = self.verify_attribute(doc_id, "width", val)
                else:
                    results[attr] = self.verify_attribute(doc_id, attr, val)
        elif isinstance(requirements, ParseResult):
            for req in requirements.requirements:
                results[req.attribute] = self.verify_attribute(doc_id, req.attribute, req.required_value)
        return results

    def format_verification(
        self, verifications: dict[str, tuple[str, Optional[str]]]
    ) -> str:
        """Format verification results matching Qualcomm Edge RAG demo specification."""
        parts = []
        for attr, (status, detail) in verifications.items():
            if status == "UNKNOWN" or not detail:
                parts.append(f"{attr}: {status}")
            else:
                parts.append(f"{attr}: {status} ({detail})")
        return " · ".join(parts)
