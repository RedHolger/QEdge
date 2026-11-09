"""Material, color, and brand taxonomy and derivation semantics for QEdge contracts.

Enforces:
- Directed 'is-a' relationships for materials (oak -> hardwood -> wood, steel -> metal).
- Asymmetry: oak implies wood, but generic wood cannot certify oak.
- Disjoint top-level material and color families.
- Conservative handling of unresolved material compositions (e.g. oak veneer alone
  cannot establish solid-oak composition or an unsupported negation).
- Compound and single color derivation with token boundary matching and disjoint rejection.
- Complete normalized brand identity comparison with whitespace stripping.
"""
from __future__ import annotations

import re
from typing import Optional

_MATERIAL_PARENTS: dict[str, list[str]] = {
    "oak": ["hardwood", "wood"],
    "beech": ["hardwood", "wood"],
    "maple": ["hardwood", "wood"],
    "walnut": ["hardwood", "wood"],
    "cherry": ["hardwood", "wood"],
    "teak": ["hardwood", "wood"],
    "mahogany": ["hardwood", "wood"],
    "ash": ["hardwood", "wood"],
    "birch": ["hardwood", "wood"],
    "pine": ["softwood", "wood"],
    "cedar": ["softwood", "wood"],
    "fir": ["softwood", "wood"],
    "hardwood": ["wood"],
    "softwood": ["wood"],
    "steel": ["metal"],
    "stainless steel": ["steel", "metal"],
    "aluminum": ["metal"],
    "aluminium": ["metal"],
    "iron": ["metal"],
    "cast iron": ["iron", "metal"],
    "brass": ["metal"],
    "copper": ["metal"],
    "bronze": ["metal"],
    "velvet": ["fabric"],
    "cotton": ["fabric"],
    "linen": ["fabric"],
    "wool": ["fabric"],
    "polyester": ["fabric"],
    "nylon": ["fabric"],
    "genuine leather": ["leather"],
    "faux leather": ["leather", "synthetic"],
    "polypropylene": ["plastic"],
    "polyethylene": ["plastic"],
    "acrylic": ["plastic"],
    "resin": ["plastic"],
    "tempered glass": ["glass"],
    "marble": ["stone"],
    "granite": ["stone"],
    "slate": ["stone"],
}

_MATERIAL_FAMILIES: dict[str, set[str]] = {
    "wood": {
        "wood", "hardwood", "softwood", "oak", "beech", "maple", "walnut",
        "cherry", "teak", "mahogany", "ash", "birch", "pine", "cedar", "fir",
    },
    "metal": {
        "metal", "steel", "stainless steel", "aluminum", "aluminium",
        "iron", "cast iron", "brass", "copper", "bronze",
    },
    "fabric": {
        "fabric", "velvet", "cotton", "linen", "wool", "polyester", "nylon",
    },
    "leather": {"leather", "genuine leather", "faux leather"},
    "plastic": {
        "plastic", "polypropylene", "polyethylene", "acrylic", "resin",
    },
    "glass": {"glass", "tempered glass"},
    "stone": {"stone", "marble", "granite", "slate"},
}

_BASE_COLORS: frozenset[str] = frozenset({
    "red", "blue", "green", "yellow", "black", "white", "silver", "gold",
    "grey", "gray", "brown", "orange", "purple", "pink", "natural", "clear",
    "beige", "tan", "navy", "cream", "bronze",
})

_COLOR_SYNONYMS: dict[str, str] = {
    "grey": "gray",
}

_SHADE_MODIFIERS: frozenset[str] = frozenset({
    "dark", "light", "deep", "pale", "bright", "matte", "glossy", "warm", "cool",
    "rich", "soft", "vivid", "muted", "medium", "pastel", "metallic",
    "navy", "forest", "sky", "royal", "olive", "lime", "ruby", "emerald", "charcoal",
})


def _canonical_color_token(tok: str) -> str:
    return _COLOR_SYNONYMS.get(tok, tok)


def parse_color_description(text: str) -> Optional[tuple[Optional[str], str]]:
    """Parse text into a recognized affirmative color description (modifier, base_color).

    Returns None for negated ('not red'), alternative ('red or blue'), uncertain
    ('maybe red'), multi-word, or unrecognized descriptions.
    Grammar:
    - <base_color> (e.g. 'red') -> (None, 'red')
    - <modifier> <base_color> (e.g. 'dark red') -> ('dark', 'red')
    - <base_color> <modifier> (e.g. 'red matte') -> ('matte', 'red')
    """
    norm = text.strip().casefold()
    if not norm:
        return None

    raw_tokens = norm.split()
    if not raw_tokens:
        return None

    tokens = [_canonical_color_token(t) for t in raw_tokens]

    # Single base color
    if len(tokens) == 1:
        if tokens[0] in _BASE_COLORS:
            return (None, tokens[0])
        return None

    # Compound shade: modifier + base or base + modifier
    if len(tokens) == 2:
        if tokens[0] in _SHADE_MODIFIERS and tokens[1] in _BASE_COLORS:
            return (tokens[0], tokens[1])
        if tokens[0] in _BASE_COLORS and tokens[1] in _SHADE_MODIFIERS:
            return (tokens[1], tokens[0])
        return None

    return None


def get_all_ancestors(material: str) -> set[str]:
    """Return all transitive ancestor types for a material."""
    norm = material.strip().casefold()
    ancestors = set()
    queue = [norm]
    while queue:
        curr = queue.pop(0)
        parents = _MATERIAL_PARENTS.get(curr, [])
        for p in parents:
            if p not in ancestors:
                ancestors.add(p)
                queue.append(p)
    return ancestors


def get_material_family(material: str) -> Optional[str]:
    """Return the top-level disjoint family name for a material, if known."""
    norm = material.strip().casefold()
    for fam_name, members in _MATERIAL_FAMILIES.items():
        if norm in members:
            return fam_name
    return None


def evaluate_material_derivation(
    required_value: str, observed_value: str
) -> tuple[bool, bool | None]:
    """Evaluate material equality with explicit asymmetric taxonomy derivation.

    Returns:
        (evaluable, match):
        - If evaluable is False, unresolved or non-evaluable material taxonomy (UNKNOWN).
        - If evaluable is True, match indicates whether the derivation holds (True) or contradicts (False).
    """
    req_norm = required_value.strip().casefold()
    obs_norm = observed_value.strip().casefold()

    if not req_norm or not obs_norm:
        return False, None

    # Exact string equality
    if req_norm == obs_norm:
        return True, True

    obs_ancestors = get_all_ancestors(obs_norm)
    req_ancestors = get_all_ancestors(req_norm)
    obs_fam = get_material_family(obs_norm)
    req_fam = get_material_family(req_norm)

    # Subtype satisfies supertype (e.g. oak is-a wood)
    if req_norm in obs_ancestors:
        return True, True

    # Supertype cannot certify subtype (e.g. generic wood cannot prove oak)
    if obs_norm in req_ancestors or obs_norm == req_fam:
        return False, None

    # Disjoint families (e.g. wood vs metal)
    if req_fam is not None and obs_fam is not None and req_fam != obs_fam:
        return True, False

    # Same family but distinct specific types (e.g. oak vs beech)
    if req_fam is not None and req_fam == obs_fam:
        if req_norm != req_fam and obs_norm != obs_fam:
            return True, False

    # Unrecognized composition (e.g. 'oak veneer') alone cannot establish
    # solid-oak composition or an unsupported negation -> conservative UNKNOWN.
    return False, None


def evaluate_color_derivation(
    required_value: str, observed_value: str
) -> tuple[bool, bool | None]:
    """Evaluate color equality with explicit bounded taxonomy and compound shade derivation.

    Enforces:
    - Bounded affirmative color grammar: only recognized base colors and supported shade
      descriptions (e.g. 'dark red') can participate in derivation.
    - Negated ('not red'), alternative ('red or blue'), uncertain ('maybe red'), or
      unrecognized descriptions ('sparkly red') return (False, None) -> UNKNOWN/ABSTAIN.
    - Compound shade observations containing the base color (e.g. 'dark red' observed for
      required 'red') satisfy the base requirement.
    - Generic-to-specific asymmetry: generic base observation ('red') cannot certify specific
      compound requirement ('dark red') -> UNKNOWN.
    - Disjoint color families (e.g. 'red' vs 'blue' or 'dark blue') contradict (evaluable=True, match=False).
    - Distinct specific compound shades in the same family (e.g. 'dark red' vs 'light red')
      contradict (evaluable=True, match=False).
    - Negative predicate alignment: match=False satisfies != and contradicts ==.

    Returns:
        (evaluable, match):
        - If evaluable is False, unresolved or non-evaluable color evidence (UNKNOWN).
        - If evaluable is True, match indicates whether the derivation holds (True) or contradicts (False).
    """
    req_parsed = parse_color_description(required_value)
    obs_parsed = parse_color_description(observed_value)

    # If either requirement or observation is not a recognized affirmative description,
    # evaluate conservatively as UNKNOWN (not evaluable)
    if req_parsed is None or obs_parsed is None:
        return False, None

    req_mod, req_base = req_parsed
    obs_mod, obs_base = obs_parsed

    # Disjoint base color families (e.g. red vs blue, or dark red vs light blue)
    if req_base != obs_base:
        return True, False

    # Same base color family:
    # 1. Exact match (both modifiers match, including both None or both 'dark')
    if req_mod == obs_mod:
        return True, True

    # 2. Required is generic base color, observed is recognized compound shade
    # (e.g. required "red", observed "dark red")
    if req_mod is None and obs_mod is not None:
        return True, True

    # 3. Generic-to-specific asymmetry:
    # Required is specific compound shade, observed is generic base color
    # (e.g. required "dark red", observed "red")
    if req_mod is not None and obs_mod is None:
        return False, None

    # 4. Both are distinct specific compound shades in the same family
    # (e.g. required "dark red", observed "light red")
    if req_mod != obs_mod:
        return True, False

    return False, None


def evaluate_brand_derivation(
    operator: str, required_value: str, observed_value: str
) -> tuple[bool, bool | None]:
    """Evaluate brand matching with complete normalized brand identity.

    Separates equality (==, !=) from containment (contains, not_contains).
    Leading and trailing whitespace is stripped, and case is folded.
    Case normalization does not introduce substring equality for == or !=.
    """
    req_norm = required_value.strip().casefold()
    obs_norm = observed_value.strip().casefold()
    if not req_norm or not obs_norm:
        return False, None

    if operator == "==":
        return True, req_norm == obs_norm
    if operator == "!=":
        return True, req_norm != obs_norm
    if operator == "contains":
        return True, req_norm in obs_norm
    if operator == "not_contains":
        return True, req_norm not in obs_norm
    return False, None

