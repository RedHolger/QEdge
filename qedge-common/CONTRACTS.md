# QEdge contract semantics

This document defines the meaning that the JSON Schemas and the shared
semantic validator enforce. It is the reference for contract v2 and the
compatibility rules for the historical v1 records.

A schema-valid record is not automatically a factual claim. Semantic
validation can prove that a record is internally coherent; it cannot prove
that a catalog source, model run, device, or measurement exists. Those
requirements stay visible in the record through source references, execution
evidence, and provenance.

## Version policy

- `product`, `retrieval_request`, and `retrieval_result` remain at v1 in this
  stage.
- `evidence_claim`, `decision`, `benchmark_result`, and `hardware_status` gain
  an explicit v2 contract. Their v1 files and historical v1 fixtures are
  preserved byte-for-byte.
- New producers must emit the version they intend to satisfy. A v2 record is
  validated with the v2 schema and the semantics below. A v1 record remains
  readable with its v1 schema and historical behavior; it is not silently
  upgraded or treated as a v2 positive decision.
- The validator selects the schema version from `schema_version` when the
  caller does not specify one. `qedge-validate --schema-version 1` is the
  explicit v1-reader path.
- Unknown fields are still rejected. A field that changes meaning requires a
  new schema version.

## Predicate grammar and evaluation

A requirement is one predicate:

```json
{
  "requirement_id": "req-width-max",
  "attribute": "width",
  "operator": "<=",
  "required_value": 60.0,
  "required_unit": "cm",
  "mandatory": true
}
```

`requirement_id` is unique within a decision. `mandatory: true` means the
predicate must be covered by evidence for a positive decision. `mandatory:
false` is optional and is not required for coverage, but a claim that is
present still participates in contradiction and conflict checks.

The supported operators are:

| Operator | Meaning | Value/unit rules |
|---|---|---|
| `<` | strictly less than | numeric values; both units must be present and equal |
| `<=` | less than or equal to | numeric values; both units must be present and equal |
| `>` | strictly greater than | numeric values; both units must be present and equal |
| `>=` | greater than or equal to | numeric values; both units must be present and equal |
| `==` | equality, not negation | same JSON value type; units, when present, must match |
| `!=` | inequality / scalar negation | true exactly when `==` would be false; same type/unit rules |
| `contains` | string containment | string values; `required_value` is a nonempty string; units must be null; case-insensitive substring match |
| `not_contains` | explicit negation of `contains` | string values; units must be null; true exactly when `contains` would be false |

Important consequences:

- `!=` is an explicit scalar inequality. A clause such as `not oak` must be
  represented as `operator: "!=", required_value: "oak"`; it must never be
  encoded as `operator: "==", required_value: "oak"` with a reversed claim.
- `not_contains` is an explicit predicate. General natural-language negation
  that cannot be mapped to one of these predicates is an unparsed clause, not
  a guessed polarity.
- Strict comparisons are not inclusive. `<= 60` is satisfied by 60; `< 60` is
  not.
- Numeric values are finite JSON numbers. `true`/`false` are not numbers.
- Exact numeric equality: Numeric comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`)
  enforce exact finite numeric comparisons without implicit measurement tolerance
  or rounding across both the verifier and contract validator. Two values are
  equal under `==` if and only if they evaluate to the exact same numeric value
  after unit conversion (e.g. `48.0000005` does not equal `48.0`; strict and
  inclusive boundaries are strictly preserved).
- A boolean or string predicate must not carry a unit. An equality across
  different JSON value types is not a match.
- String normalization: String comparisons (`==`, `!=`, `contains`, `not_contains`)
  strip leading/trailing whitespace and fold case (`casefold()`).
- Brand semantics: Brand equality (`==`, `!=`) enforces complete normalized brand
  identities after whitespace stripping and case folding. Normalization does not
  introduce substring matching. Substring containment is permitted exclusively
  via explicit `contains` / `not_contains` operators. Distinct brands contradict on
  `==` and satisfy on `!=`.
- Directed material taxonomy:
  - Directed 'is-a' hierarchy (e.g. oak -> hardwood -> wood, steel -> metal).
  - Subtypes satisfy supertypes under `==` (e.g. oak is wood -> `SUPPORTED`).
  - Asymmetry: generic supertypes cannot certify or disprove specific subtypes
    (e.g. generic wood cannot prove oak -> `UNKNOWN`).
  - Sibling leaf types in the same family contradict under `==` and satisfy under
    `!=` (e.g. oak vs beech -> `CONTRADICTED` on `==`, `SUPPORTED` on `!=`).
  - Disjoint families contradict under `==` and satisfy under `!=` (e.g. wood vs
    metal -> `CONTRADICTED` on `==`, `SUPPORTED` on `!=`).
  - Unrecognized material compositions: conservative evaluation. An unresolved
    composition such as "oak veneer" alone must not establish solid-oak composition
    or an unsupported negation; it evaluates conservatively to `UNKNOWN` (not
    evaluable for a supported/contradicted claim).
- Directed color derivation & bounded affirmative grammar:
  - Single and compound colors: compound shade observation containing the required
    base color token within the bounded grammar (e.g. "dark red" or "light red" for
    requirement "red") satisfies the base requirement (`SUPPORTED` on `==`,
    `CONTRADICTED` on `!=`).
  - Bounded affirmative grammar: derivation applies exclusively to recognized base
    colors and supported shade descriptions (e.g. `<modifier> <base>` where modifier is
    in the recognized shade set such as `dark`, `light`, `deep`, `pale`, `bright`, `navy`,
    `forest`, `matte`, etc.).
  - Non-affirmative descriptions: negated descriptions ("not red"), alternatives /
    disjunctions ("red or blue"), uncertainty ("maybe red"), and unrecognized modifiers
    ("sparkly red") do not establish an affirmative color fact and evaluate
    conservatively to `UNKNOWN` (producing `ABSTAIN` under decision precedence rule 5)
    for both positive and negative requirements.
  - Asymmetry: generic base color observation ("red") cannot certify specific
    compound requirement ("dark red") -> `UNKNOWN`.
  - Disjoint color families (e.g. "red" vs "blue" or "dark blue") contradict on `==`
    and satisfy on `!=`.
  - Distinct specific compound shades in the same family (e.g. "dark red" vs "light red")
    contradict on `==` and satisfy on `!=`.
- Eligible evidence status:
  - Stored evidence facts with status other than `"present"` (e.g. `"missing"`,
    `"unknown"`) must not support or contradict requirements even if values or display
    text exist; they evaluate to `UNKNOWN` and produce `ABSTAIN` under precedence rule 5.
- `UNKNOWN` means the predicate could not be evaluated (for example, missing
  observation, non-present evidence status, incompatible units, unresolved composition,
  or an ambiguous source). A supported or contradicted claim requires a non-null
  observed value and evidence text that can be found at the named source and revision.
- A supported or contradicted claim must agree with the predicate evaluation
  of its observed value. A claim whose observed value contradicts its stated
  status is rejected, not treated as evidence.

## Evidence claims

The standalone `evidence_claim` and a decision's nested `claims` use the same
field rules and the same semantic claim validator. The nested form adds a
required `requirement_id` that links the claim to its requirement.

For `SUPPORTED` and `CONTRADICTED`:

- `observed_value` is non-null;
- `source`, `source_revision`, and `evidence_text` are nonempty;
- the predicate is evaluable and the observed value evaluates to the stated
  result;
- the claim's `item_id` is the candidate's item ID in a decision.

For `UNKNOWN`, the observed value may be null. A claim must not be marked
`SUPPORTED` or `CONTRADICTED` when its predicate is not evaluable.

## Decision response types

A decision is candidate-specific when `item_id` names one retrieved item. A
no-candidate response is explicit:

```json
{
  "schema_version": "2",
  "item_id": null,
  "status": "ABSTAIN",
  "action": "ABSTAIN",
  "claims": []
}
```

No item ID may be invented for a no-result response. A no-candidate response
has no claims; it may list the parsed requirements and the reason it could not
select a candidate.

`unparsed_clauses` lists text that could not be parsed into a predicate.
`ambiguities` lists clauses with more than one plausible interpretation. Every
listed entry is unresolved and mandatory for a positive decision. Either list
being nonempty prevents `SUPPORTED`, regardless of claim coverage.

The response precedence is deterministic:

1. **No candidate:** `item_id: null`, no claims, `ABSTAIN` / `ABSTAIN`.
2. **Unresolved or conflicting evidence:** a mandatory unparsed clause, an
   ambiguity, or both a supported and contradicted claim for the same
   requirement -> `ABSTAIN` / `ABSTAIN`. A conflict is not silently resolved
   in favor of either record.
3. **Contradiction:** any claim `CONTRADICTED` (including an optional claim)
   -> `CONTRADICTED` / `DO_NOT_RECOMMEND`. Contradiction blocks recommendation.
4. **Supported:** at least one requirement, every mandatory requirement
   covered, every claim `SUPPORTED`, no unresolved/ambiguous clauses, consistent
   item IDs, and resolvable evidence -> `SUPPORTED` / `RECOMMEND`.
5. **Unknown:** no contradiction, no conflict, but a mandatory requirement is
   missing or has an `UNKNOWN` claim -> `UNKNOWN` / `ABSTAIN`.
6. A candidate with no requirements and no unsupported claims is not a positive
   response; it is `ABSTAIN` / `ABSTAIN`.

A claim for a requirement that does not exist, a duplicate requirement or claim
ID, a claim with a different item ID, and a supported response with incomplete
coverage are all invalid records. The validator enforces the precedence, not
just the presence of the words.

## Measurement invariants

- `measured: true` is allowed only for a reproducible run with a non-synthetic
  kind and a nonempty provenance log/evidence reference. Synthetic-kernel and
  fixture-smoke records must use `measured: false`.
- A measured model run has a non-null model identity. A synthetic kernel or
  fixture smoke has no measured-model claim.
- `npu_status: "measured"` requires all of: an execution evidence reference, a
  device identity, a QNN-style execution backend, a measured record, and a
  matching backend/device identity. A CPU or Metal execution cannot be labelled
  as measured NPU execution.
- A `hardware_status` record claiming measured NPU execution requires a
  Snapdragon device presence, QNN SDK availability, verified device status,
  execution evidence, and a device identity matching the Snapdragon model.
- Hardware flags must not contradict each other: an unavailable backend cannot
  be used; an absent Snapdragon cannot have a model; a measured NPU cannot be
  unverified; an unverified device cannot have measured execution.
- `cache.hit_rate` is optional. It must not be required or invented for policies
  where hit rate is not a meaningful output.
- `metrics` is for numeric/measured values. Generated text and token traces
  belong in the separate optional `outputs` object in the v2 benchmark
  envelope, never in `metrics`.
- Passing validation does not turn a synthetic record into a measurement and
  does not turn a CPU stub into an NPU result.

## Boundary checks

The Python boundary rejects nonfinite floating-point values (`NaN`,
`Infinity`, and overflow such as `1e309`) and validates RFC 3339
`date-time` strings with an explicit timezone. These checks apply to v1 and v2
JSON; historical valid records remain readable.
