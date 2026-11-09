"""Asymmetric taxonomy and attribute domain matching for QEdge Assist.

Enforces:
- Directed 'is-a' relationships (oak -> hardwood -> wood, steel -> metal).
- Asymmetry: oak implies wood, but generic wood cannot prove oak, nor does it
  disprove oak. Unknown composition cannot prove not-oak.
- Disjoint top-level material and color families.
- Compound and single color derivation with token boundary matching and disjoint rejection.
- Conservative handling of unresolved material compositions (e.g. oak veneer alone
  cannot establish solid-oak composition or an unsupported negation).
- Complete normalized brand identity comparison with whitespace stripping.
"""
from __future__ import annotations

import re
from typing import Optional

from qedge_common.taxonomy import (
    _MATERIAL_PARENTS,
    _MATERIAL_FAMILIES,
    _BASE_COLORS,
    _SHADE_MODIFIERS,
    get_all_ancestors,
    get_material_family,
    evaluate_material_derivation,
    evaluate_color_derivation,
    evaluate_brand_derivation,
    parse_color_description,
)

_COMMON_COLORS = _BASE_COLORS


def evaluate_material(
    operator: str, required_value: str, observed_value: Optional[str]
) -> tuple[str, str]:
    """Evaluate material predicate with explicit asymmetric taxonomy.

    Returns:
        (STATUS, reason_detail) where STATUS in {'SUPPORTED', 'CONTRADICTED', 'UNKNOWN'}
    """
    if observed_value is None:
        return "UNKNOWN", "missing observed material"

    req_norm = required_value.strip().casefold()
    obs_norm = observed_value.strip().casefold()

    if not req_norm or not obs_norm:
        return "UNKNOWN", "empty material value"

    if operator in ("contains", "not_contains"):
        has_sub = req_norm in obs_norm
        if operator == "contains":
            return ("SUPPORTED", f"material '{observed_value}' contains '{required_value}'") if has_sub else ("CONTRADICTED", f"material '{observed_value}' does not contain '{required_value}'")
        else:
            return ("CONTRADICTED", f"material '{observed_value}' contains '{required_value}'") if has_sub else ("SUPPORTED", f"material '{observed_value}' does not contain '{required_value}'")

    evaluable, match = evaluate_material_derivation(required_value, observed_value)
    if not evaluable:
        return "UNKNOWN", f"unresolved material taxonomy for '{required_value}' vs '{observed_value}'"

    if operator == "==":
        if match:
            return "SUPPORTED", f"material matches '{observed_value}'"
        return "CONTRADICTED", f"material '{observed_value}' contradicts requirement '{required_value}'"
    elif operator == "!=":
        if match:
            return "CONTRADICTED", f"material '{observed_value}' is '{required_value}'"
        return "SUPPORTED", f"material '{observed_value}' satisfies negation 'not {required_value}'"

    return "UNKNOWN", f"unsupported material operator '{operator}'"


def evaluate_color(
    operator: str, required_value: str, observed_value: Optional[str]
) -> tuple[str, str]:
    """Evaluate color predicate with explicit taxonomy and compound shade derivation.

    Returns:
        (STATUS, reason_detail) where STATUS in {'SUPPORTED', 'CONTRADICTED', 'UNKNOWN'}
    """
    if observed_value is None:
        return "UNKNOWN", "missing observed color"

    req_norm = required_value.strip().casefold()
    obs_norm = observed_value.strip().casefold()

    if not req_norm or not obs_norm:
        return "UNKNOWN", "empty color value"

    if operator in ("contains", "not_contains"):
        has_sub = req_norm in obs_norm
        if operator == "contains":
            return ("SUPPORTED", f"color '{observed_value}' contains '{required_value}'") if has_sub else ("CONTRADICTED", f"color '{observed_value}' does not contain '{required_value}'")
        else:
            return ("CONTRADICTED", f"color '{observed_value}' contains '{required_value}'") if has_sub else ("SUPPORTED", f"color '{observed_value}' does not contain '{required_value}'")

    evaluable, match = evaluate_color_derivation(required_value, observed_value)
    if not evaluable:
        return "UNKNOWN", f"unresolved color taxonomy for '{required_value}' vs '{observed_value}'"

    if operator == "==":
        if match:
            return "SUPPORTED", f"color matches '{observed_value}'"
        return "CONTRADICTED", f"observed color '{observed_value}' contradicts '{required_value}'"
    elif operator == "!=":
        if match:
            return "CONTRADICTED", f"observed color '{observed_value}' is '{required_value}'"
        return "SUPPORTED", f"observed color '{observed_value}' satisfies 'not {required_value}'"

    return "UNKNOWN", f"unsupported color operator '{operator}'"


# Unit conversion factors to cm
_LENGTH_TO_CM: dict[str, float] = {
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


def evaluate_brand(
    operator: str, required_value: str, observed_value: Optional[str]
) -> tuple[str, str]:
    """Evaluate brand matching with complete normalized brand identity equality.

    Separates equality (==, !=) from containment (contains, not_contains).
    Case normalization does not introduce substring equality.
    Leading and trailing whitespace is stripped.
    """
    if observed_value is None:
        return "UNKNOWN", "missing observed brand"
    req_norm = required_value.strip().casefold()
    obs_norm = observed_value.strip().casefold()
    if not req_norm or not obs_norm:
        return "UNKNOWN", "empty brand value"

    evaluable, match = evaluate_brand_derivation(operator, required_value, observed_value)
    if not evaluable:
        return "UNKNOWN", f"unresolved brand evaluation for '{required_value}' vs '{observed_value}'"

    if operator == "==":
        if match:
            return "SUPPORTED", f"brand matches '{observed_value}'"
        return "CONTRADICTED", f"observed brand '{observed_value}' does not match '{required_value}'"
    elif operator == "!=":
        if not match:
            return "CONTRADICTED", f"observed brand '{observed_value}' matches '{required_value}'"
        return "SUPPORTED", f"observed brand '{observed_value}' does not match '{required_value}'"
    elif operator == "contains":
        if match:
            return "SUPPORTED", f"brand '{observed_value}' contains '{required_value}'"
        return "CONTRADICTED", f"observed brand '{observed_value}' does not contain '{required_value}'"
    elif operator == "not_contains":
        if not match:
            return "CONTRADICTED", f"observed brand '{observed_value}' contains '{required_value}'"
        return "SUPPORTED", f"observed brand '{observed_value}' does not contain '{required_value}'"
    return "UNKNOWN", f"unsupported brand operator '{operator}'"


