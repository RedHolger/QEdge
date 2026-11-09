"""Load and validate the shared JSON Schemas and their contract semantics.

The canonical schema files are in ``qedge-common/schemas``.  A built wheel
contains the same files under ``qedge_common/schemas``.  Source-tree checkouts
therefore work without a build step, while installed wheels remain
self-contained.

JSON Schema validation is necessary but not sufficient.  The helpers below
also enforce the predicate, evidence, decision, and measurement invariants that
cannot be expressed reliably with structural keywords alone.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import math
import re
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from .taxonomy import (
    evaluate_material_derivation,
    evaluate_color_derivation,
    evaluate_brand_derivation,
)

SCHEMA_VERSION = "2"

_SCHEMA_VERSIONS: dict[str, tuple[str, ...]] = {
    "product": ("1",),
    "retrieval_request": ("1",),
    "retrieval_result": ("1",),
    "evidence_claim": ("1", "2"),
    "decision": ("1", "2"),
    "benchmark_result": ("1", "2"),
    "hardware_status": ("1", "2"),
}

_SYNTHETIC_KINDS = frozenset({"synthetic-kernel", "fixture-smoke"})
_NUMERIC_OPERATORS = frozenset({"<", "<=", ">", ">="})
_EQUALITY_OPERATORS = frozenset({"==", "!="})
_CONTAINS_OPERATORS = frozenset({"contains", "not_contains"})
_DATE_TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)


class SchemaNotFoundError(FileNotFoundError):
    """Raised when a requested contract name or version is not available."""


class ContractValidationError(ValueError):
    """Raised when a payload does not satisfy its contract."""

    def __init__(self, schema_name: str, errors: list[str], version: str | None = None):
        self.schema_name = schema_name
        self.errors = errors
        self.version = version
        label = f"{schema_name}@v{version}" if version else schema_name
        super().__init__(f"{label}: " + "; ".join(errors))


def _is_rfc3339_datetime(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    if not _DATE_TIME_RE.fullmatch(value):
        return False
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None


_FORMAT_CHECKER = FormatChecker()
_FORMAT_CHECKER.checks("date-time")(_is_rfc3339_datetime)


def schema_dir() -> Path:
    """Return the directory containing canonical schema JSON files."""
    packaged = Path(__file__).resolve().parent / "schemas"
    if packaged.is_dir():
        return packaged
    source = Path(__file__).resolve().parents[2] / "schemas"
    if source.is_dir():
        return source
    raise SchemaNotFoundError("could not locate qedge-common/schemas")


def schema_names() -> tuple[str, ...]:
    """Return the stable schema names in this release."""
    return tuple(_SCHEMA_VERSIONS)


def schema_versions(name: str) -> tuple[str, ...]:
    """Return the schema versions available for *name*."""
    if name not in _SCHEMA_VERSIONS:
        raise SchemaNotFoundError(f"unknown schema: {name!r}")
    return _SCHEMA_VERSIONS[name]


def schema_path(name: str, version: str = "1") -> Path:
    """Return the canonical path for a schema name and version."""
    versions = schema_versions(name)
    if version not in versions:
        raise SchemaNotFoundError(
            f"unknown schema version for {name!r}: {version!r}; available: {versions}"
        )
    if version == "1":
        return schema_dir() / f"{name}.schema.json"
    return schema_dir() / "v2" / f"{name}.schema.json"


def load_schema(name: str, version: str = "1") -> dict[str, Any]:
    with schema_path(name, version).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def payload_schema_version(name: str, payload: Any) -> str:
    """Select the schema version declared by a payload.

    v1 is the default for historical payloads that omit the field; the
    structural validator will still report the missing required field.
    """
    value = payload.get("schema_version", "1") if isinstance(payload, dict) else "1"
    if not isinstance(value, str) or value not in schema_versions(name):
        raise ContractValidationError(
            name,
            [f"schema_version: unsupported version {value!r}"],
            version=None,
        )
    return value


def _render_error(error: ValidationError) -> str:
    location = ".".join(str(part) for part in error.path) or "<root>"
    return f"{location}: {error.message}"


def _nonfinite_errors(value: Any, location: str = "<root>") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{location}: non-finite JSON number")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_nonfinite_errors(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_nonfinite_errors(child, f"{location}[{index}]"))
    return errors


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json(path: str | Path) -> Any:
    """Load JSON while rejecting NaN, Infinity, and overflowed floats."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_json_constant)
    errors = _nonfinite_errors(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def validate(name: str, payload: Any, version: str | None = None) -> None:
    """Validate *payload* structurally and semantically or raise a useful error."""
    if version is None:
        version = payload_schema_version(name, payload)
    elif version not in schema_versions(name):
        raise SchemaNotFoundError(
            f"unknown schema version for {name!r}: {version!r}"
        )

    schema = load_schema(name, version)
    validator = Draft202012Validator(schema, format_checker=_FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    rendered = [_render_error(error) for error in errors]

    finite_errors = _nonfinite_errors(payload)
    rendered.extend(finite_errors)

    if not rendered:
        rendered.extend(_semantic_errors(name, payload, version))

    if rendered:
        raise ContractValidationError(name, rendered, version=version)


def example_path(name: str, version: str = "1") -> Path:
    """Return the valid example for a contract, checking packaged examples first."""
    if version not in schema_versions(name):
        raise SchemaNotFoundError(
            f"unknown schema version for {name!r}: {version!r}"
        )
    packaged = Path(__file__).resolve().parent / "examples"
    if packaged.is_dir():
        candidate = (
            packaged / f"{name}.json"
            if version == "1"
            else packaged / "v2" / f"{name}.json"
        )
        if candidate.is_file():
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    if version == "1":
        return source_root / "tests" / "fixtures" / "valid" / f"{name}.json"
    return source_root / "tests" / "fixtures" / "valid" / "v2" / f"{name}.json"


def validate_example(name: str, version: str = "1") -> None:
    validate(name, load_json(example_path(name, version)), version=version)


# ---------------------------------------------------------------------------
# Semantic helpers
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _unit_errors(unit: Any, location: str) -> list[str]:
    if unit is not None and not _is_nonempty_string(unit):
        return [f"{location}: unit must be a nonempty string or null"]
    return []


def _type_family(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if _is_number(value):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _requirement_errors(requirement: dict[str, Any], location: str) -> list[str]:
    errors: list[str] = []
    operator = requirement.get("operator")
    required_value = requirement.get("required_value")
    unit = requirement.get("required_unit")

    if not isinstance(requirement.get("mandatory"), bool):
        errors.append(f"{location}.mandatory: must be a boolean")
    errors.extend(_unit_errors(unit, f"{location}.required_unit"))

    if operator in _NUMERIC_OPERATORS:
        if not _is_number(required_value):
            errors.append(f"{location}.required_value: numeric operator requires a number")
        if not _is_nonempty_string(unit):
            errors.append(f"{location}.required_unit: numeric comparison requires a unit")
    elif operator in _EQUALITY_OPERATORS:
        if required_value is None:
            errors.append(f"{location}.required_value: must not be null")
        elif isinstance(required_value, bool) or isinstance(required_value, str):
            if unit is not None:
                errors.append(f"{location}.required_unit: boolean/string equality has no unit")
        elif not _is_number(required_value):
            errors.append(f"{location}.required_value: unsupported value type")
    elif operator in _CONTAINS_OPERATORS:
        if not _is_nonempty_string(required_value):
            errors.append(f"{location}.required_value: containment requires a nonempty string")
        if unit is not None:
            errors.append(f"{location}.required_unit: containment has no unit")
    else:
        errors.append(f"{location}.operator: unsupported operator {operator!r}")
    return errors


def _evaluate_predicate(
    operator: Any,
    required_value: Any,
    required_unit: Any,
    observed_value: Any,
    observed_unit: Any,
    attribute: Any = None,
) -> tuple[bool, bool | None]:
    """Return ``(evaluable, result)`` for a claim or requirement pair."""
    if observed_value is None:
        return False, None

    if operator in _NUMERIC_OPERATORS:
        if not (_is_number(required_value) and _is_number(observed_value)):
            return False, None
        if not (
            _is_nonempty_string(required_unit)
            and _is_nonempty_string(observed_unit)
            and required_unit == observed_unit
        ):
            return False, None
        if operator == "<":
            return True, observed_value < required_value
        if operator == "<=":
            return True, observed_value <= required_value
        if operator == ">":
            return True, observed_value > required_value
        return True, observed_value >= required_value

    if operator in _EQUALITY_OPERATORS:
        if _type_family(required_value) != _type_family(observed_value):
            return False, None
        if isinstance(required_value, (bool, str)):
            if required_unit is not None or observed_unit is not None:
                return False, None
        elif _is_number(required_value):
            if (required_unit is None) != (observed_unit is None):
                return False, None
            if required_unit is not None and required_unit != observed_unit:
                return False, None
        else:
            return False, None

        if isinstance(required_value, str) and isinstance(observed_value, str):
            if attribute == "material":
                evaluable, match = evaluate_material_derivation(required_value, observed_value)
                if not evaluable:
                    return False, None
                return True, match if operator == "==" else not match
            if attribute == "color":
                evaluable, match = evaluate_color_derivation(required_value, observed_value)
                if not evaluable:
                    return False, None
                return True, match if operator == "==" else not match
            if attribute == "brand":
                evaluable, match = evaluate_brand_derivation(operator, required_value, observed_value)
                if not evaluable:
                    return False, None
                return True, match
            equal = required_value.strip().casefold() == observed_value.strip().casefold()
        else:
            equal = required_value == observed_value
        return True, equal if operator == "==" else not equal

    if operator in _CONTAINS_OPERATORS:
        if not (
            _is_nonempty_string(required_value)
            and isinstance(observed_value, str)
            and required_unit is None
            and observed_unit is None
        ):
            return False, None
        if attribute == "brand":
            evaluable, match = evaluate_brand_derivation(operator, required_value, observed_value)
            if not evaluable:
                return False, None
            return True, match
        result = required_value.strip().casefold() in observed_value.strip().casefold()
        return True, result if operator == "contains" else not result

    return False, None


def _claim_semantic_errors(
    claim: dict[str, Any],
    location: str,
    *,
    decision_item_id: str | None = None,
    requirement: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    status = claim.get("status")
    operator = claim.get("operator")
    required_value = claim.get("required_value")
    required_unit = claim.get("required_unit")
    observed_value = claim.get("observed_value")
    observed_unit = claim.get("observed_unit")
    source = claim.get("source")

    if decision_item_id is not None and claim.get("item_id") != decision_item_id:
        errors.append(f"{location}.item_id: must match decision item_id")
    errors.extend(_unit_errors(required_unit, f"{location}.required_unit"))
    errors.extend(_unit_errors(observed_unit, f"{location}.observed_unit"))
    if isinstance(source, str) and source.strip().lower() in {"retrieval", "retrieved", "retriever"}:
        errors.append(f"{location}.source: retrieval output is not evidence")

    if requirement is not None:
        if claim.get("requirement_id") != requirement.get("requirement_id"):
            errors.append(f"{location}.requirement_id: does not match requirement")
        for field in ("attribute", "operator", "required_value", "required_unit"):
            if claim.get(field) != requirement.get(field):
                errors.append(f"{location}.{field}: does not match requirement")

    if status in {"SUPPORTED", "CONTRADICTED"}:
        if observed_value is None:
            errors.append(f"{location}.observed_value: supported/contradicted claim needs an observation")
        if not _is_nonempty_string(claim.get("evidence_text")):
            errors.append(f"{location}.evidence_text: supported/contradicted claim needs evidence text")
        evaluable, result = _evaluate_predicate(
            operator,
            required_value,
            required_unit,
            observed_value,
            observed_unit,
            attribute=claim.get("attribute"),
        )
        if not evaluable:
            errors.append(f"{location}: predicate is not evaluable for a supported/contradicted claim")
        else:
            expected = status == "SUPPORTED"
            if result is not expected:
                errors.append(f"{location}.status: does not match predicate evaluation")
    return errors


def _expected_decision(payload: dict[str, Any]) -> tuple[str, str]:
    item_id = payload.get("item_id")
    if item_id is None:
        return "ABSTAIN", "ABSTAIN"

    requirements = payload.get("requirements", [])
    claims = payload.get("claims", [])
    if not requirements:
        return "ABSTAIN", "ABSTAIN"
    if payload.get("unparsed_clauses") or payload.get("ambiguities"):
        return "ABSTAIN", "ABSTAIN"

    statuses: dict[str, set[str]] = {}
    for claim in claims:
        requirement_id = claim.get("requirement_id")
        status = claim.get("status")
        if isinstance(requirement_id, str) and isinstance(status, str):
            statuses.setdefault(requirement_id, set()).add(status)

    if any(
        "SUPPORTED" in values and "CONTRADICTED" in values
        for values in statuses.values()
    ):
        return "ABSTAIN", "ABSTAIN"
    if any("CONTRADICTED" in values for values in statuses.values()):
        return "CONTRADICTED", "DO_NOT_RECOMMEND"

    mandatory = [item for item in requirements if item.get("mandatory") is True]
    covered = all(item.get("requirement_id") in statuses for item in mandatory)
    all_supported = bool(claims) and all(
        claim.get("status") == "SUPPORTED" for claim in claims
    )
    if covered and all_supported:
        return "SUPPORTED", "RECOMMEND"
    return "UNKNOWN", "ABSTAIN"


def _decision_semantic_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    item_id = payload.get("item_id")
    requirements = payload.get("requirements", [])
    claims = payload.get("claims", [])

    requirement_ids: set[str] = set()
    requirements_by_id: dict[str, dict[str, Any]] = {}
    for index, requirement in enumerate(requirements):
        location = f"requirements[{index}]"
        requirement_id = requirement.get("requirement_id")
        if requirement_id in requirement_ids:
            errors.append(f"{location}.requirement_id: duplicate requirement_id")
        elif isinstance(requirement_id, str):
            requirement_ids.add(requirement_id)
            requirements_by_id[requirement_id] = requirement
        errors.extend(_requirement_errors(requirement, location))

    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        location = f"claims[{index}]"
        claim_id = claim.get("claim_id")
        if claim_id in claim_ids:
            errors.append(f"{location}.claim_id: duplicate claim_id")
        elif isinstance(claim_id, str):
            claim_ids.add(claim_id)
        requirement_id = claim.get("requirement_id")
        requirement = requirements_by_id.get(requirement_id)
        if requirement is None:
            errors.append(f"{location}.requirement_id: unknown requirement")
        if item_id is None:
            errors.append(f"{location}: no-candidate response must not contain claims")
        errors.extend(
            _claim_semantic_errors(
                claim,
                location,
                decision_item_id=item_id if isinstance(item_id, str) else None,
                requirement=requirement,
            )
        )

    if item_id is None:
        if any(claim for claim in claims):
            errors.append("item_id: no-candidate response must not contain claims")
    expected_status, expected_action = _expected_decision(payload)
    if payload.get("status") != expected_status:
        errors.append(
            f"status: expected {expected_status} from evidence precedence, "
            f"got {payload.get('status')!r}"
        )
    if payload.get("action") != expected_action:
        errors.append(
            f"action: expected {expected_action} from evidence precedence, "
            f"got {payload.get('action')!r}"
        )

    if item_id is not None and not requirements:
        errors.append("requirements: a candidate decision needs at least one requirement")
    if item_id is None and payload.get("status") != "ABSTAIN":
        errors.append("status: no-candidate response must be ABSTAIN")
    return errors


def _benchmark_semantic_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    kind = payload.get("kind")
    measured = payload.get("measured")
    hardware = payload.get("hardware", {})
    npu_status = hardware.get("npu_status")
    provenance = payload.get("provenance", {})

    if kind in _SYNTHETIC_KINDS and measured is True:
        errors.append(f"measured: {kind} is synthetic and cannot be measured=true")

    if measured is True:
        if not _is_nonempty_string(provenance.get("log_path")):
            errors.append("provenance.log_path: measured run needs a reproducible log/evidence reference")
        if kind == "runtime-generation" and payload.get("model") is None:
            errors.append("model: measured runtime-generation needs a model identity")

    if npu_status == "measured":
        if measured is not True:
            errors.append("hardware.npu_status: measured NPU status requires measured=true")
        evidence = hardware.get("execution_evidence")
        device = hardware.get("device")
        execution_backend = hardware.get("execution_backend")
        backend = payload.get("backend")
        if not _is_nonempty_string(evidence):
            errors.append("hardware.execution_evidence: measured NPU needs execution evidence")
        if not _is_nonempty_string(device):
            errors.append("hardware.device: measured NPU needs a device identity")
        if not _is_nonempty_string(execution_backend):
            errors.append("hardware.execution_backend: measured NPU needs a backend identity")
        elif execution_backend != backend:
            errors.append("hardware.execution_backend: must match top-level backend")
        if _is_nonempty_string(execution_backend) and _is_cpu_backend(execution_backend):
            errors.append("hardware.execution_backend: CPU/Metal/stub backend cannot claim measured NPU")
        elif _is_nonempty_string(execution_backend) and not _is_npu_backend(execution_backend):
            errors.append("hardware.execution_backend: measured NPU needs a QNN/NPU-style backend")
    elif _is_nonempty_string(hardware.get("execution_evidence")):
        errors.append("hardware.execution_evidence: only a measured NPU may carry execution evidence")

    return errors


def _is_cpu_backend(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return (
        "cpu" in normalized
        or "stub" in normalized
        or normalized.startswith("metal")
    )


def _is_npu_backend(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return any(
        marker in normalized
        for marker in ("qnn", "npu", "htp", "hexagon", "snapdragon")
    )


def _hardware_semantic_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    metal = payload.get("metal", {})
    snapdragon = payload.get("snapdragon", {})
    qnn = payload.get("qnn", {})
    npu_status = payload.get("npu_status")
    snap_available = snapdragon.get("available")
    snap_model = snapdragon.get("model")
    sdk_available = qnn.get("sdk_available")
    device_verified = qnn.get("device_verified")
    qnn_device = qnn.get("device")
    evidence = qnn.get("execution_evidence")
    execution_backend = qnn.get("execution_backend")

    if metal.get("used") is True and metal.get("available") is not True:
        errors.append("metal: used=true requires available=true")
    if metal.get("available") is False and metal.get("used") is True:
        errors.append("metal: unavailable backend cannot be used")

    if snap_available is True and not _is_nonempty_string(snap_model):
        errors.append("snapdragon.model: available Snapdragon needs a model identity")
    if snap_available is False and snap_model is not None:
        errors.append("snapdragon.model: absent Snapdragon cannot have a model")

    if device_verified is True:
        if sdk_available is not True:
            errors.append("qnn: device_verified requires sdk_available")
        if snap_available is not True or not _is_nonempty_string(snap_model):
            errors.append("qnn: device_verified requires an available Snapdragon")
    if _is_nonempty_string(qnn_device):
        if snap_available is not True:
            errors.append("qnn.device: QNN device requires an available Snapdragon")
        if _is_nonempty_string(snap_model) and qnn_device != snap_model:
            errors.append("qnn.device: must match snapdragon.model")
    elif qnn_device is not None:
        errors.append("qnn.device: must be a nonempty string or null")

    if npu_status == "measured":
        if snap_available is not True or not _is_nonempty_string(snap_model):
            errors.append("npu_status: measured NPU requires an available Snapdragon model")
        if sdk_available is not True:
            errors.append("npu_status: measured NPU requires qnn.sdk_available")
        if device_verified is not True:
            errors.append("npu_status: measured NPU requires qnn.device_verified")
        if not _is_nonempty_string(qnn_device):
            errors.append("npu_status: measured NPU requires qnn.device")
        if not _is_nonempty_string(evidence):
            errors.append("npu_status: measured NPU requires execution evidence")
        if not _is_nonempty_string(execution_backend):
            errors.append("npu_status: measured NPU requires an execution backend")
        elif _is_cpu_backend(execution_backend):
            errors.append("npu_status: CPU/Metal/stub backend cannot claim measured NPU")
        elif not _is_npu_backend(execution_backend):
            errors.append("npu_status: measured NPU needs a QNN/NPU-style backend")
    elif npu_status == "not-present":
        if snap_available is not False:
            errors.append("npu_status: not-present requires snapdragon.available=false")
        if snap_model is not None:
            errors.append("npu_status: not-present requires snapdragon.model=null")
        if device_verified is True:
            errors.append("npu_status: not-present cannot be device_verified")
        if _is_nonempty_string(qnn_device):
            errors.append("npu_status: not-present cannot have a QNN device")
        if _is_nonempty_string(evidence):
            errors.append("npu_status: not-present cannot have execution evidence")
        if _is_nonempty_string(execution_backend):
            errors.append("npu_status: not-present cannot have an execution backend")
    elif _is_nonempty_string(evidence):
        errors.append("npu_status: unverified status cannot carry execution evidence")

    return errors


def _semantic_errors(name: str, payload: Any, version: str) -> list[str]:
    if version != "2":
        # v1 is the historical reader.  Its structural behavior is preserved;
        # date-time and finite-number checks still run for every version.
        return []
    if not isinstance(payload, dict):
        return []
    if name == "evidence_claim":
        return _claim_semantic_errors(payload, "claim")
    if name == "decision":
        return _decision_semantic_errors(payload)
    if name == "benchmark_result":
        return _benchmark_semantic_errors(payload)
    if name == "hardware_status":
        return _hardware_semantic_errors(payload)
    return []
