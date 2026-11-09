"""Typed requirement parser for QEdge Assist.

Parses natural language user queries into structured v2 predicates:
- Preserves source spans and raw matched text.
- Supports width, height, length, package dimensions, color, material, brand.
- Supports strict (<, >) and inclusive (<=, >=) comparisons, ranges, exact (==),
  and negation (!=).
- Preserves explicit units (cm, in, mm, m, ft).
- Never guesses units or dimensions for bare numbers; logs ambiguities.
- Detects mutually conflicting user constraints and logs ambiguities.
- Detects unsupported disjunctions (OR expressions) and logs ambiguities.
- Preserves unparsed and unsupported clauses (e.g. 'comfortable', 'quantum charger').
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Optional

from .taxonomy import convert_length


@dataclass
class ParsedRequirement:
    requirement_id: str
    attribute: str
    operator: str
    required_value: Any
    required_unit: Optional[str]
    mandatory: bool
    source_span: tuple[int, int]
    raw_text: str

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "attribute": self.attribute,
            "operator": self.operator,
            "required_value": self.required_value,
            "required_unit": self.required_unit,
            "mandatory": self.mandatory,
        }


@dataclass
class ParseResult:
    query: str
    requirements: list[ParsedRequirement] = field(default_factory=list)
    unparsed_clauses: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    def to_contract_requirements(self) -> list[dict[str, Any]]:
        return [req.to_contract_dict() for req in self.requirements]


_UNIT_MAP = {
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "in": "in",
    "inch": "in",
    "inches": "in",
    '"': "in",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "'": "ft",
}

_UNIT_PATTERN = r"(?:cm|centimeters?|mm|millimeters?|m|meters?|inches?|in|\"|ft|feet|')"

_MATERIALS = [
    "oak", "beech", "maple", "walnut", "cherry", "teak", "mahogany", "ash", "birch",
    "pine", "cedar", "fir", "hardwood", "softwood", "wooden", "wood",
    "stainless steel", "steel", "aluminum", "aluminium", "metallic", "metal", "iron", "cast iron",
    "brass", "copper", "bronze",
    "velvet", "cotton", "linen", "woolen", "wool", "polyester", "nylon", "fabric",
    "genuine leather", "faux leather", "leather",
    "polypropylene", "polyethylene", "acrylic", "resin", "plastic",
    "tempered glass", "glass",
    "marble", "granite", "slate", "stone",
]

_COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "silver", "gold",
    "grey", "gray", "brown", "orange", "purple", "pink", "natural", "clear",
    "beige", "tan", "navy", "cream", "bronze",
]

# Only purely grammatical words and catalog head categories.
# Subjective requirements like 'comfortable', 'comfy', 'good', 'nice', 'ergonomic'
# are deliberately excluded so they remain substantive unparsed requirements.
_STOP_AND_FILLER_WORDS = frozenset({
    "i", "need", "want", "looking", "for", "find", "me", "a", "an", "the",
    "with", "and", "in", "of", "to", "at",
    "please", "can", "you", "show", "give", "item", "product",
    "chair", "chairs", "seat", "seats", "dining", "desk", "lamp", "lamps", "table",
    "tables", "stool", "stools", "furniture", "that", "is", "are", "be",
    "must", "have", "has",
    "material", "materials", "color", "colors", "colour", "colours",
    "brand", "brands", "size", "sizes", "dimension", "dimensions",
    "made", "from",
})

_CATEGORY_WORDS = frozenset({
    "chair", "chairs", "seat", "seats", "dining", "desk", "lamp", "lamps",
    "table", "tables", "stool", "stools", "furniture", "item", "product",
    "and", "with", "for", "in", "of", "to", "at", "by", "that", "is", "are",
})


def _normalize_unit(unit_str: Optional[str]) -> Optional[str]:
    if not unit_str:
        return None
    cleaned = unit_str.strip().lower()
    return _UNIT_MAP.get(cleaned, cleaned)


def _map_op(op_text: str) -> str:
    cleaned = op_text.lower().strip()
    if cleaned in ("strictly less than", "less than", "under", "narrower than", "shorter than", "<"):
        return "<"
    if cleaned in ("no more than", "at most", "up to", "maximum of", "max of", "max", "<="):
        return "<="
    if cleaned in ("strictly more than", "more than", "over", "wider than", "taller than", ">"):
        return ">"
    if cleaned in ("no less than", "at least", "minimum of", "min of", "min", ">="):
        return ">="
    if cleaned in ("!=", "not equal to", "not", "<>"):
        return "!="
    return "=="


def parse_query_requirements(query: str) -> ParseResult:
    """Parse query into structured requirements, ambiguities, and unparsed clauses."""
    result = ParseResult(query=query)
    covered_spans: list[tuple[int, int]] = []
    req_counter = 0

    # 0. Check for unsupported disjunctions (OR expressions)
    # e.g. "without red or blue", "under 50cm or over 100cm", "oak or metal"
    or_regex = re.compile(
        rf"\b(without\s+[a-z]+\s+or\s+[a-z]+|[a-z0-9_.-]+\s+or\s+[a-z0-9_.-]+)\b",
        re.IGNORECASE,
    )
    for m in or_regex.finditer(query):
        matched = m.group(0).strip()
        tokens = matched.lower().split()
        if "or" in tokens:
            result.ambiguities.append(f"unsupported disjunction (OR expression): {matched!r}")

    # 1. Range expressions e.g. "between 40cm and 60cm wide", "from 60cm to 1m wide"
    # Supports optional prefix dimension word ("width between 60cm and 1m") or postfix ("between 60cm and 1m wide")
    range_regex = re.compile(
        rf"\b(?:(package\s+)?(width|height|length|depth)\s*(?:is|of|:)?\s*)?"
        rf"(?:between|from)\s+(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})?\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})?\s*"
        rf"(package\s+)?(wide|width|high|height|tall|long|length|depth|deep)?\b",
        re.IGNORECASE,
    )
    for m in range_regex.finditer(query):
        v1 = float(m.group(3))
        u1 = _normalize_unit(m.group(4))
        v2 = float(m.group(5))
        u2 = _normalize_unit(m.group(6)) or u1
        u1 = u1 or u2

        # Check prefix vs postfix dimension
        is_pkg_prefix = bool(m.group(1))
        dim_prefix = (m.group(2) or "").lower()
        is_pkg_postfix = bool(m.group(7))
        dim_postfix = (m.group(8) or "").lower()

        is_package = is_pkg_prefix or is_pkg_postfix
        dim_word = dim_prefix or dim_postfix

        if dim_word in ("high", "height", "tall"):
            attr = "package_height" if is_package else "height"
        elif dim_word in ("long", "length", "depth", "deep"):
            attr = "package_length" if is_package else "length"
        elif dim_word in ("wide", "width"):
            attr = "package_width" if is_package else "width"
        else:
            # Dimension is missing: must NOT acquire guessed width!
            result.ambiguities.append(f"unspecified dimension for range constraint: {m.group(0)!r}")
            covered_spans.append((m.start(), m.end()))
            continue

        if not u1 or not u2:
            result.ambiguities.append(f"range constraint missing units: {m.group(0)!r}")
            covered_spans.append((m.start(), m.end()))
            continue

        norm1 = convert_length(v1, u1, "cm")
        norm2 = convert_length(v2, u2, "cm")
        if norm1 is None or norm2 is None:
            result.ambiguities.append(f"range constraint incompatible units: {m.group(0)!r}")
            covered_spans.append((m.start(), m.end()))
            continue

        # Reorder endpoints based on normalized metric units (never reorder raw values independently)
        if norm1 <= norm2:
            min_val, min_unit = v1, u1
            max_val, max_unit = v2, u2
        else:
            # Handle reversed range: e.g. "between 1m and 60cm wide"
            min_val, min_unit = v2, u2
            max_val, max_unit = v1, u1

        # Normalize max_val to min_unit so both bounds share the same requirement unit
        max_val_conv = convert_length(max_val, max_unit, min_unit)

        req_counter += 1
        result.requirements.append(
            ParsedRequirement(
                requirement_id=f"req-{attr}-min-{req_counter}",
                attribute=attr,
                operator=">=",
                required_value=min_val,
                required_unit=min_unit,
                mandatory=True,
                source_span=(m.start(), m.end()),
                raw_text=m.group(0),
            )
        )
        req_counter += 1
        result.requirements.append(
            ParsedRequirement(
                requirement_id=f"req-{attr}-max-{req_counter}",
                attribute=attr,
                operator="<=",
                required_value=max_val_conv,
                required_unit=min_unit,
                mandatory=True,
                source_span=(m.start(), m.end()),
                raw_text=m.group(0),
            )
        )
        covered_spans.append((m.start(), m.end()))

    # 2. Dimensions with prefix attribute: e.g. "package width under 50cm", "width < 60cm", "height 100cm"
    prefix_dim_regex = re.compile(
        rf"\b(package\s+)?(width|height|length|depth)\s*(?:of|is|:)?\s*"
        rf"(strictly\s+less\s+than|less\s+than|under|narrower\s+than|shorter\s+than|<|"
        rf"no\s+more\s+than|at\s+most|up\s+to|maximum\s+of|max\s+of|max|<=|"
        rf"strictly\s+more\s+than|more\s+than|over|wider\s+than|taller\s+than|>|"
        rf"no\s+less\s+than|at\s+least|minimum\s+of|min\s+of|min|>=|"
        rf"!=|not\s+equal\s+to|not\s+equals\s+to|not\s+equal|not|"
        rf"exactly|equal\s+to|=|:)?\s*"
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})?\b",
        re.IGNORECASE,
    )
    for m in prefix_dim_regex.finditer(query):
        if any(s <= m.start() and m.end() <= e for s, e in covered_spans):
            continue
        is_package = bool(m.group(1))
        dim_name = m.group(2).lower()
        op_text = m.group(3) or "=="
        val = float(m.group(4))
        unit = _normalize_unit(m.group(5))

        if dim_name in ("width",):
            attr = "package_width" if is_package else "width"
        elif dim_name in ("height",):
            attr = "package_height" if is_package else "height"
        else:
            attr = "package_length" if is_package else "length"

        operator = _map_op(op_text)
        if not unit:
            result.ambiguities.append(f"unspecified unit for {attr} constraint: {m.group(0)!r}")
        else:
            req_counter += 1
            result.requirements.append(
                ParsedRequirement(
                    requirement_id=f"req-{attr}-{req_counter}",
                    attribute=attr,
                    operator=operator,
                    required_value=val,
                    required_unit=unit,
                    mandatory=True,
                    source_span=(m.start(), m.end()),
                    raw_text=m.group(0),
                )
            )
        covered_spans.append((m.start(), m.end()))

    # 3. Dimensions with postfix attribute or standard comparison:
    # e.g. "under 60cm wide", "24 inches wide", "less than 50cm tall"
    postfix_dim_regex = re.compile(
        rf"\b(strictly\s+less\s+than|less\s+than|under|narrower\s+than|shorter\s+than|<|"
        rf"no\s+more\s+than|at\s+most|up\s+to|maximum\s+of|max\s+of|max|<=|"
        rf"strictly\s+more\s+than|more\s+than|over|wider\s+than|taller\s+than|>|"
        rf"no\s+less\s+than|at\s+least|minimum\s+of|min\s+of|min|>=|"
        rf"!=|not\s+equal\s+to|not\s+equals\s+to|not\s+equal|not|"
        rf"exactly|equal\s+to|=|:)?\s*"
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})?\s*"
        rf"(package\s+)?(wide|width|high|height|tall|long|length|depth|deep)\b",
        re.IGNORECASE,
    )
    for m in postfix_dim_regex.finditer(query):
        if any(s <= m.start() and m.end() <= e for s, e in covered_spans):
            continue
        op_text = m.group(1) or "=="
        val = float(m.group(2))
        unit = _normalize_unit(m.group(3))
        is_package = bool(m.group(4))
        dim_word = m.group(5).lower()

        if dim_word in ("high", "height", "tall"):
            attr = "package_height" if is_package else "height"
        elif dim_word in ("long", "length", "depth", "deep"):
            attr = "package_length" if is_package else "length"
        elif dim_word in ("wide", "width"):
            attr = "package_width" if is_package else "width"
        else:
            result.ambiguities.append(f"unspecified dimension for constraint: {m.group(0)!r}")
            covered_spans.append((m.start(), m.end()))
            continue

        operator = _map_op(op_text)
        if not unit:
            result.ambiguities.append(f"unspecified unit for {attr} constraint: {m.group(0)!r}")
        else:
            req_counter += 1
            result.requirements.append(
                ParsedRequirement(
                    requirement_id=f"req-{attr}-{req_counter}",
                    attribute=attr,
                    operator=operator,
                    required_value=val,
                    required_unit=unit,
                    mandatory=True,
                    source_span=(m.start(), m.end()),
                    raw_text=m.group(0),
                )
            )
        covered_spans.append((m.start(), m.end()))

    # 4. Conjoined comparisons: e.g. "under 50cm and over 60cm wide"
    conjoined_regex = re.compile(
        rf"\b(strictly\s+less\s+than|less\s+than|under|narrower\s+than|shorter\s+than|<|"
        rf"no\s+more\s+than|at\s+most|up\s+to|maximum\s+of|max\s+of|max|<=|"
        rf"strictly\s+more\s+than|more\s+than|over|wider\s+than|taller\s+than|>|"
        rf"no\s+less\s+than|at\s+least|minimum\s+of|min\s+of|min|>=)\s*"
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})?\s*(?:and|&)\s*"
        rf"(strictly\s+less\s+than|less\s+than|under|narrower\s+than|shorter\s+than|<|"
        rf"no\s+more\s+than|at\s+most|up\s+to|maximum\s+of|max\s+of|max|<=|"
        rf"strictly\s+more\s+than|more\s+than|over|wider\s+than|taller\s+than|>|"
        rf"no\s+less\s+than|at\s+least|minimum\s+of|min\s+of|min|>=)\s*"
        rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})?\s*"
        rf"(package\s+)?(wide|width|high|height|tall|long|length|depth|deep)?\b",
        re.IGNORECASE,
    )
    for m in conjoined_regex.finditer(query):
        is_package = bool(m.group(7))
        dim_word = (m.group(8) or "").lower()
        if dim_word in ("high", "height", "tall"):
            attr = "package_height" if is_package else "height"
        elif dim_word in ("long", "length", "depth", "deep"):
            attr = "package_length" if is_package else "length"
        elif dim_word in ("wide", "width"):
            attr = "package_width" if is_package else "width"
        else:
            result.ambiguities.append(f"unspecified dimension for conjoined constraint: {m.group(0)!r}")
            covered_spans.append((m.start(), m.end()))
            continue

        covered_spans = [(s, e) for s, e in covered_spans if not (m.start() <= s and e <= m.end())]
        result.requirements = [
            r for r in result.requirements if not (m.start() <= r.source_span[0] and r.source_span[1] <= m.end())
        ]

        op1 = _map_op(m.group(1))
        val1 = float(m.group(2))
        u1 = _normalize_unit(m.group(3))

        op2 = _map_op(m.group(4))
        val2 = float(m.group(5))
        u2 = _normalize_unit(m.group(6)) or u1
        u1 = u1 or u2

        if not u1 or not u2:
            result.ambiguities.append(f"conjoined {attr} constraint missing unit: {m.group(0)!r}")
        else:
            req_counter += 1
            result.requirements.append(
                ParsedRequirement(
                    requirement_id=f"req-{attr}-{req_counter}",
                    attribute=attr,
                    operator=op1,
                    required_value=val1,
                    required_unit=u1,
                    mandatory=True,
                    source_span=(m.start(), m.end()),
                    raw_text=m.group(0),
                )
            )
            req_counter += 1
            result.requirements.append(
                ParsedRequirement(
                    requirement_id=f"req-{attr}-{req_counter}",
                    attribute=attr,
                    operator=op2,
                    required_value=val2,
                    required_unit=u2,
                    mandatory=True,
                    source_span=(m.start(), m.end()),
                    raw_text=m.group(0),
                )
            )
        covered_spans.append((m.start(), m.end()))

    # 5. Bare numbers and orphan comparisons without unit or dimension
    bare_op_regex = re.compile(
        r"\b(strictly\s+less\s+than|less\s+than|under|narrower\s+than|shorter\s+than|<|"
        r"no\s+more\s+than|at\s+most|up\s+to|maximum\s+of|max\s+of|max|<=|"
        r"strictly\s+more\s+than|more\s+than|over|wider\s+than|taller\s+than|>|"
        r"no\s+less\s+than|at\s+least|minimum\s+of|min\s+of|min|>=|"
        r"exactly|equal\s+to|=|:)?\s*(\d+(?:\.\d+)?)\s*(?!cm|mm|m|in|ft|%)\b",
        re.IGNORECASE,
    )
    for m in bare_op_regex.finditer(query):
        if any(s <= m.start() and m.end() <= e for s, e in covered_spans):
            continue
        clause = m.group(0).strip()
        result.ambiguities.append(f"unspecified dimension or unit for bare number: {clause!r}")
        covered_spans.append((m.start(), m.end()))

    # 6. Check for Material constraints (including negation)
    for mat in _MATERIALS:
        pattern = rf"\b(not\s+|non-|other\s+than\s+|without\s+)?({re.escape(mat)})\b"
        for m in re.finditer(pattern, query, re.IGNORECASE):
            if any(s <= m.start() and m.end() <= e for s, e in covered_spans):
                continue
            neg = bool(m.group(1))
            op = "!=" if neg else "=="
            mat_val = mat.lower()
            if mat_val == "wooden":
                mat_val = "wood"
            elif mat_val == "metallic":
                mat_val = "metal"
            elif mat_val == "woolen":
                mat_val = "wool"
            req_counter += 1
            result.requirements.append(
                ParsedRequirement(
                    requirement_id=f"req-material-{req_counter}",
                    attribute="material",
                    operator=op,
                    required_value=mat_val,
                    required_unit=None,
                    mandatory=True,
                    source_span=(m.start(), m.end()),
                    raw_text=m.group(0),
                )
            )
            covered_spans.append((m.start(), m.end()))

    # 7. Check for Color constraints (including recognized shade modifiers and negation)
    _SHADE_MODS_PAT = r"(?:(?:dark|light|deep|pale|bright|matte|glossy|warm|cool|rich|soft|vivid|muted|medium|pastel|metallic|navy|forest|sky|royal|olive|lime|ruby|emerald|charcoal)\s+)?"
    for col in _COLORS:
        pattern = rf"\b(not\s+|non-|other\s+than\s+|without\s+)?({_SHADE_MODS_PAT})({re.escape(col)})\b"
        for m in re.finditer(pattern, query, re.IGNORECASE):
            if any(s <= m.start() and m.end() <= e for s, e in covered_spans):
                continue
            neg = bool(m.group(1))
            op = "!=" if neg else "=="
            mod = (m.group(2) or "").strip().lower()
            col_val = f"{mod} {col.lower()}".strip()
            req_counter += 1
            result.requirements.append(
                ParsedRequirement(
                    requirement_id=f"req-color-{req_counter}",
                    attribute="color",
                    operator=op,
                    required_value=col_val,
                    required_unit=None,
                    mandatory=True,
                    source_span=(m.start(), m.end()),
                    raw_text=m.group(0),
                )
            )
            covered_spans.append((m.start(), m.end()))

    # 8. Check for Brand constraints (including multiword and negation)
    # Non-brand queries containing 'brand' (e.g. 'brand new', 'name brand', 'any brand') must not create brand requirements.
    brand_regex = re.compile(
        r"\b(not\s+(?:by|brand|from)|other\s+than\s+brand|brand\s+not|(?:made\s+)?by|from|brand(?:\s+is|:)?)\s+([A-Za-z0-9]+(?:[\s-][A-Za-z0-9]+)*)\b",
        re.IGNORECASE,
    )
    non_brand_idioms = {"new", "any", "unknown", "name", "good", "name brand", "off brand"}
    for m in brand_regex.finditer(query):
        if any(s <= m.start() and m.end() <= e for s, e in covered_spans):
            continue
        prefix = m.group(1).lower().strip()
        brand_raw = m.group(2).strip()

        # If prefix is 'brand' and immediately followed by 'new' ('brand new'), skip
        if prefix.startswith("brand") and brand_raw.lower().startswith("new"):
            continue

        tokens = brand_raw.split()
        cleaned_tokens = []
        for tok in tokens:
            tl = tok.lower()
            if (
                tl in _CATEGORY_WORDS
                or tl in _STOP_AND_FILLER_WORDS
                or tl in _MATERIALS
                or tl in _COLORS
                or tl in non_brand_idioms
            ):
                break
            cleaned_tokens.append(tok)
        if not cleaned_tokens:
            continue
        brand_name = " ".join(cleaned_tokens)
        if brand_name.lower() in non_brand_idioms:
            continue

        neg = any(w in prefix for w in ("not", "other than"))
        op = "!=" if neg else "=="

        req_counter += 1
        end_pos = m.start() + len(prefix) + 1 + len(brand_name)
        result.requirements.append(
            ParsedRequirement(
                requirement_id=f"req-brand-{req_counter}",
                attribute="brand",
                operator=op,
                required_value=brand_name,
                required_unit=None,
                mandatory=True,
                source_span=(m.start(), end_pos),
                raw_text=query[m.start():end_pos],
            )
        )
        covered_spans.append((m.start(), end_pos))

    # 9. Check for conflicting user constraints on the same attribute
    reqs_by_attr: dict[str, list[ParsedRequirement]] = {}
    for req in result.requirements:
        reqs_by_attr.setdefault(req.attribute, []).append(req)

    for attr, req_list in reqs_by_attr.items():
        if len(req_list) > 1:
            # Direct contradiction: == X and != X
            equals = [r for r in req_list if r.operator == "=="]
            not_equals = [r for r in req_list if r.operator == "!="]
            for eq in equals:
                for neq in not_equals:
                    if str(eq.required_value).casefold() == str(neq.required_value).casefold():
                        result.ambiguities.append(
                            f"conflicting constraints on '{attr}': requires '{eq.required_value}' and not '{neq.required_value}'"
                        )

            # Mutually exclusive numeric bounds (with unit normalization)
            numeric_reqs = [r for r in req_list if r.operator in ("<", "<=", ">", ">=")]
            max_bounds = [r for r in numeric_reqs if r.operator in ("<", "<=")]
            min_bounds = [r for r in numeric_reqs if r.operator in (">", ">=")]
            for mx in max_bounds:
                for mn in min_bounds:
                    norm_mx = convert_length(float(mx.required_value), mx.required_unit, "cm")
                    norm_mn = convert_length(float(mn.required_value), mn.required_unit, "cm")
                    if norm_mx is not None and norm_mn is not None:
                        if norm_mx < norm_mn:
                            result.ambiguities.append(
                                f"conflicting {attr} bounds: {mx.operator} {mx.required_value}{mx.required_unit} and {mn.operator} {mn.required_value}{mn.required_unit}"
                            )

    # 10. Extract unsupported/unparsed clauses
    covered_spans.sort()
    merged_spans: list[tuple[int, int]] = []
    for start, end in covered_spans:
        if not merged_spans:
            merged_spans.append((start, end))
        else:
            prev_start, prev_end = merged_spans[-1]
            if start <= prev_end + 1:
                merged_spans[-1] = (prev_start, max(prev_end, end))
            else:
                merged_spans.append((start, end))

    uncovered_texts: list[str] = []
    last_idx = 0
    for start, end in merged_spans:
        if start > last_idx:
            chunk = query[last_idx:start].strip()
            if chunk:
                uncovered_texts.append(chunk)
        last_idx = max(last_idx, end)
    if last_idx < len(query):
        chunk = query[last_idx:].strip()
        if chunk:
            uncovered_texts.append(chunk)

    for chunk in uncovered_texts:
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", chunk.lower())
        substantive = [t for t in tokens if t not in _STOP_AND_FILLER_WORDS]
        if substantive:
            clean_clause = re.sub(
                r"^(?:with|and|or|having|featuring|including|that\s+has|that\s+is|that\s+must\s+be|for)\s+",
                "",
                chunk,
                flags=re.IGNORECASE,
            ).strip()
            clean_clause = re.sub(
                r"^(?:dining\s+|desk\s+)?(?:chair|seat|lamp|table|stool)s?\s*(?:with\s+)?",
                "",
                clean_clause,
                flags=re.IGNORECASE,
            ).strip()
            if clean_clause:
                result.unparsed_clauses.append(clean_clause)

    return result
