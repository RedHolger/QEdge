# Implementation Plan: 04-assistant-correctness

Date: 30 October 2026
Stage: 04-assistant-correctness
Prerequisites: 02-contracts (ACCEPTED), 03-evidence-store (ACCEPTED)

## 1. Objectives & Boundaries

### Objectives
1. Build a typed requirement parser preserving source spans, explicit units, strict/inclusive operators, ranges, negation, and unparsed/ambiguous clauses.
2. Remove all hidden chair/oak/width defaults and bare number unit guessing.
3. Distinguish product dimensions from package dimensions, and product families from variants.
4. Implement an explicit asymmetric taxonomy (oak is wood, but wood cannot certify oak; unknown composition cannot prove not-oak).
5. Prevent substring matching false positives (e.g., "red" in "shredded" / "reduced").
6. Handle conflicting user constraints explicitly (flagging ambiguities).
7. Handle conflicting source evidence without silently picking the first row (emitting conflicting claims leading to ABSTAIN).
8. Render citations to exact fact revisions in EvidenceStore.
9. Emit contract-validated v2 claims and deterministic decisions matching `CONTRACTS.md` precedence.
10. Remove synthetic QNN benchmark execution from normal answer composition; optional generation must not decide verdict.
11. Support catalog directory paths in CLI and Verifier (`resolve_catalog_file`).
12. Establish the 80-case decision testbed (40 dev / 40 held-out) with provenance; assert zero false positive decisions on adverse cases.

### Scope Constraints
- Bounded strictly to Stage 04 (Assistant Correctness). Do not implement Stage 05 (real model loading) or Stage 06 (retrieval expansion / hybrid retrieval).
- Keep all 7 baseline v1 JSON schemas byte-for-byte untouched.
- Keep all 30 historical manifest artifacts untouched.
- Preserve Stage 01 kernel safety in `llm-inference` (`CTest 3/3`).
- Maintain existing 1 expected skip (`RealCatalogIntegrationTests.test_catalog_loads`).

---

## 2. Baseline Defects to Reproduce

1. **Hidden Query Defaults**: `cli.py` injects `width_lte_cm=60.0, material=oak` if query mentions chair/seat.
2. **Guessed Units**: `verifier.py` assumes `cm` if no unit provided ("under 60" -> 60cm).
3. **Symmetric Material Taxonomy Bug**: `verifier.py` marks `material == "oak"` as SUPPORTED when catalog has generic `material = "wood"`.
4. **Substring Matching Bug**: Substring check for color/material matches within unrelated words.
5. **Negation Unhandled**: Queries like "not oak" are either ignored or treated as positive "oak".
6. **Strict Comparison Conflation**: `<` and `<=` treated identically as `<=`.
7. **Package vs Product Conflation**: Package dimensions satisfy product dimension requirements.
8. **First-Row Arbitrary Resolution**: Conflicting source facts resolve by taking the first row found.
9. **Unsupported Clauses Ignored**: Phrases like "built-in quantum charger" are discarded, allowing positive recommendation.
10. **Synthetic QNN Execution in Demo**: `Generator` attempts to run `llm_qnn_bench` during formatting.
11. **Directory Path Error**: `Verifier.load_catalog` raises `IsADirectoryError` when given a directory path.
12. **Absence of 80-case Program**: No structured 40 dev / 40 held-out testbed.

---

## 3. Step-by-Step Implementation Strategy

1. **Reproduction Probes**:
   - Write `handoff/research-review/04-assistant-correctness/probes/baseline_assistant_gaps.py`.
   - Run probe, capture log in `handoff/research-review/04-assistant-correctness/logs/baseline-assistant-gaps.log`.

2. **Parser & Taxonomy (`requirements.py` & `taxonomy.py`)**:
   - Create `edge-rag-demo/src/edge_rag/taxonomy.py` implementing asymmetric relationships and disjoint categories.
   - Create `edge-rag-demo/src/edge_rag/requirements.py` implementing typed parsing, span tracking, unit normalization, strict/inclusive operators, ranges, negation, and unparsed/ambiguity detection.

3. **Verifier Refactoring (`verifier.py`)**:
   - Use `resolve_catalog_file` in `load_catalog` to handle directory paths.
   - Distinguish product vs package attributes in catalog loading.
   - Verify candidates against requirements producing contract v2 `evidence_claim` and `decision` records.
   - Use revision-resolving citations from `store.py`.
   - Check all evidence chunks; conflicting facts generate conflicting claims resulting in `ABSTAIN`.
   - Validate decisions using `qedge_common.schema.validate("decision", d, version="2")`.

4. **Generator & CLI Refactoring (`generator.py` & `cli.py`)**:
   - Remove QNN subprocess benchmark execution from `Generator`.
   - Make generation extractive based on verified claims; ensure it cannot alter the decision verdict.
   - Remove hidden defaults in `cli.py` and integrate new parser/decision flow.

5. **80-Case Program & Adversarial Test Suites**:
   - Create `edge-rag-demo/tests/fixtures/decision_cases_80.json` (40 dev / 40 held-out) with provenance.
   - Add table-driven adversarial unit tests covering all required cases:
     - red chair, under 60 without units, not oak, oak vs wood, 24 inches vs 60.96cm, strict boundary, multiple dimensions, unknown units, contradictory sources, unsupported features, no constraints, no candidates, revisions changing verdict.
   - Add catalog-directory CLI test.
   - Verify 0 false positives on adverse cases.

6. **Regression & Acceptance Verification**:
   - Run `edge-tests`, `qedge-common-tests`, `historical-hashes`, `schema-hashes`, `llm-ctest`.
   - Write handoff deliverable `handoff/research-review/04-assistant-correctness.md`.
