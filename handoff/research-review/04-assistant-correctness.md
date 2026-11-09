# Research review 04 assistant correctness handoff

Date: 30 October 2026  
Scope: Stage 04 (`04-assistant-correctness`) typed requirement parsing, asymmetric taxonomy, strict/inclusive boundary evaluation, unit normalization without rounding, candidate check separation, deterministic decision precedence, unparsed/ambiguous clause preservation, elimination of hidden defaults and synthetic QNN composition, catalog directory CLI support, brand grammar and negation, mixed-unit range normalization without raw reordering, disjunction detection, answer boundary fidelity, brand predicate semantics, observation fidelity without required-value substitution, eligible structured evidence status, shared authoritative predicate derivation, compound color semantics, conservative unresolved material composition, brand whitespace normalization, cross-layer semantic boundary reconciliation, bounded affirmative color grammar and non-affirmative description abstention, exact numeric equality reconciliation without measurement tolerance across contract boundary, and the 80-case internal decision regression testbed.  
Prerequisites: `00-rebaseline` (ACCEPTED), `01-kernel-safety` (ACCEPTED), `02-contracts` (ACCEPTED), `03-evidence-store` (ACCEPTED).  
All 7 baseline v1 JSON schemas, 30 historical manifest artifacts, and Stage 01 kernel safety implementations in `llm-inference` were strictly preserved.

---

## Verdict

**ACCEPTED (bounded correctness scope following independent review; independent generalization evaluation pending)**

Independent review on 30 October 2026 accepted Stage 04 for its bounded correctness scope:
- Independent reruns confirmed: `edge-rag` 68 passed / 1 expected skip; `qedge-common` 49 passed; acceptance probes 15/15 passed; fresh independent probes 11/11 passed; historical hashes 30/30 verified byte-for-byte; schema hashes 7/7 verified byte-for-byte.
- Negated and uncertain color evidence strictly abstains under bounded affirmative color taxonomy.
- Numeric equality and inequality validate consistently across both layers without measurement tolerance or rounding.
- **Limits:** Independent generalization evaluation remains **pending**. The 80 authored regression cases are strictly internal regression fixtures distinct from `qedge-common` contract schema evaluation. Stage 05 is unblocked for its bounded reference runtime scope.

---

## Commands and Results

All commands were run from the workspace root against isolated environments. Dated logs are preserved in `handoff/research-review/04-assistant-correctness/logs/`.

| Command | Exit | Passed | Failed | Skipped | Status | Evidence |
|---|---:|---:|---:|---:|---|---|
| `edge-tests` | 0 | 68 | 0 | 1 | PASS | [log](04-assistant-correctness/logs/edge-tests.log) |
| `qedge-common-tests` | 0 | 49 | 0 | 0 | PASS | [log](04-assistant-correctness/logs/qedge-common-tests.log) |
| `llm-ctest` | 0 | 3 | 0 | 0 | PASS | [log](04-assistant-correctness/logs/llm-ctest.log) |
| `edge-rag-80-case-eval` | 0 | 4 | 0 | 0 | PASS (80 cases: 40 dev, 40 held-out, 0 FP) | [log](04-assistant-correctness/logs/edge-rag-80-case-eval.log) |
| `edge-bench` | 0 | — | — | — | PASS (emits validated contract v2 smoke) | [log](04-assistant-correctness/logs/edge-bench.log) |
| `historical-hashes` | 0 | 30 | 0 | 0 | PASS (30/30 verified) | [log](04-assistant-correctness/logs/historical-hashes.log) |
| `schema-hashes` | 0 | 7 | 0 | 0 | PASS (all 7 v1 schemas byte-for-byte identical) | [log](04-assistant-correctness/logs/schema-hashes.log) |
| `acceptance-assistant-gaps` | 0 | 15 | 0 | 0 | PASS (all 15 acceptance checks closed) | [probe](04-assistant-correctness/probes/acceptance_assistant_gaps.py) |
| `reproduce-round2-blockers` | 0 | 2 | 0 | 0 | REPRODUCED & RESOLVED (2/2 reproduced pre-repair, 0 post-repair) | [probe](04-assistant-correctness/probes/reproduce_stage04_round2_blockers.py) |
| `reproduce-round3-blockers` | 0 | 3 | 0 | 0 | REPRODUCED & RESOLVED (3/3 reproduced pre-repair, 0 post-repair) | [probe](04-assistant-correctness/probes/reproduce_stage04_round3_blockers.py) |
| `reproduce-round4-blockers` | 0 | 3 | 0 | 0 | REPRODUCED & RESOLVED (3/3 reproduced pre-repair, 0 post-repair) | [probe](04-assistant-correctness/probes/reproduce_stage04_round4_blockers.py) |

*Notes on counts & evaluation commands:*
- `edge-tests`: 69 total tests: 68 passed, 0 failed, 1 explicitly skipped (`RealCatalogIntegrationTests.test_catalog_loads`, requiring `QEDGE_RUN_CATALOG_INTEGRATION=1` with a licensed catalog). This includes 23 tests in `test_evidence_store.py`, 21 tests in `test_assistant_adversarial.py` (including the 30-case cross-layer table-driven test), 4 tests in `test_80_cases.py`, 13 tests in `test_pipeline.py`, 4 in `test_portability.py`, and 3 in `test_profiler.py`.
- `qedge-common-tests`: 49 tests pass across contract semantics and schema definitions.
- `edge-rag-80-case-eval`: Validates 80 decision cases across 80 unique query/item pairs with zero false positives. Internal regression suite distinct from qedge-common contract evaluation.
- `historical-hashes`: All 30 protected historical manifest files match their baseline hashes.
- `schema-hashes`: All 7 baseline v1 schema files match their recorded SHA256 hashes byte-for-byte.
- `llm-ctest`: CTest 3/3 passes in `llm-inference` (`llm_kv_cache`, `kernel_safety`, `qnn_stub_cpu`).

---

## Round 4 Review Findings and Verified Repairs

### 1. Bounded Affirmative Color Grammar and Non-Affirmative Description Abstention
- **Defects:**
  1. Stored observation `attribute="color", value="not red"`; query `"red chair"` produced `SUPPORTED / RECOMMEND` with `observed_value="not red"`.
  2. Stored observation `attribute="color", value="red or blue"`; query `"red chair"` produced `SUPPORTED / RECOMMEND` with `observed_value="red or blue"`.
- **Root Cause:**
  - Token-based word-boundary regex (`\bred\b`) matched the token `"red"` inside negated strings (`"not red"`), disjunctive alternatives (`"red or blue"`), and uncertain descriptions (`"maybe red"`), erroneously certifying positive support for the base color.
- **Repair:**
  - In [`qedge_common.taxonomy`](../../qedge-common/src/qedge_common/taxonomy.py), established a formal affirmative color grammar via `parse_color_description(text: str) -> Optional[tuple[Optional[str], str]]`.
  - Recognized colors strictly require either:
    - A single base color from `_BASE_COLORS` (e.g. `"red"`, `"blue"`, `"green"`, etc.), OR
    - A single modifier from `_SHADE_MODIFIERS` followed by a base color (e.g. `"dark red"`, `"light blue"`, `"ruby red"`, `"navy blue"`).
  - In `evaluate_color_derivation()`:
    - Non-affirmative descriptions, including negations (`"not red"`), alternatives/disjunctions (`"red or blue"`), uncertain phrases (`"maybe red"`), and unrecognized modifiers (`"sparkly red"`), fail parsing and return `(False, None)`.
    - Both positive (`==`) and negative (`!=`) queries evaluate non-affirmative observations conservatively to `evaluable=False, matches=False` -> `UNKNOWN`.
    - In `Verifier`, this emits an `UNKNOWN` claim with `observed_value=None`, producing a validated `UNKNOWN / ABSTAIN` decision under precedence rule 5.
    - Preserved affirmative compound shade derivation: `"dark red"` satisfies `"red"` under `==` (`SUPPORTED`), contradicts `"not red"` under `!=` (`CONTRADICTED`), and disjoint colors (`"blue"`, `"dark blue"`) contradict `==`.
    - In [`edge_rag.requirements.py`](../../edge-rag-demo/src/edge_rag/requirements.py), expanded color extraction regex to match recognized compound shade modifiers (e.g. `"dark red"`), preventing modifier tokens from spilling into `unparsed_clauses`.
- **Evidence:** Reproduction Probe 1 & 2 in `reproduce_stage04_round4_blockers.py`, Check 15 in `acceptance_assistant_gaps.py`, and 30-case table in `test_semantic_cross_layer_decision_boundary`.

### 2. Reconcile Numeric Equality Across Decision Boundary
- **Defect:**
  - Stored fact `attribute="width", value=48.0000005, unit="cm"`; query `"chair width = 48cm"`.
  - Raised `ContractValidationError: claims[0].status: does not match predicate evaluation`.
- **Root Cause:**
  - `Verifier._verify_numeric` used a float tolerance `abs(converted - f_req) < 1e-6` when evaluating `==` and `!=`.
  - The shared contract validator in `qedge_common.schema` enforces exact numeric comparison: `required_value == observed_value`.
  - For `48.0000005 cm`, the verifier evaluated the condition as `SUPPORTED`, but the contract validator correctly evaluated it as non-equal (`matches is False`), rejecting the record at the contract boundary.
- **Repair:**
  - Eliminated the implicit float tolerance (`1e-6`) from `Verifier._verify_numeric` in [`edge-rag-demo/src/edge_rag/verifier.py`](../../edge-rag-demo/src/edge_rag/verifier.py). Both verifier and contract validator enforce exact numeric equality:
    ```python
    if op == "==":
        matches = converted == f_req
    elif op == "!=":
        matches = converted != f_req
    ```
  - In [`qedge_common/CONTRACTS.md`](../../qedge-common/CONTRACTS.md), documented exact numeric equality across both layers without measurement tolerance or rounding.
  - Added support for inequality operators (`!=`, `not equal to`, `not`) in `_map_op`, `prefix_dim_regex`, and `postfix_dim_regex` in [`edge_rag.requirements.py`](../../edge-rag-demo/src/edge_rag/requirements.py).
  - Verified exact numeric comparisons:
    - Exact equality: `48.0 cm == 48 cm` -> `SUPPORTED / RECOMMEND`.
    - Former near-tolerance values: `48.0000005 cm == 48 cm` -> `CONTRADICTED / DO_NOT_RECOMMEND` with observation fidelity (`observed_value=48.0000005`).
    - Inequality: `48.0000005 cm != 48 cm` -> `SUPPORTED / RECOMMEND`.
    - Unit conversion: `24 in == 60.96 cm` evaluates exactly to `SUPPORTED` without rounding.
    - Strict (`<`, `>`) vs inclusive (`<=`, `>=`) boundary semantics preserved.
- **Evidence:** Reproduction Probe 3 in `reproduce_stage04_round4_blockers.py`, Check 15 in `acceptance_assistant_gaps.py`, `test_strict_and_inclusive_comparisons` and `test_equality_inequality_and_containment` in `qedge-common`, and 30-case table in `test_semantic_cross_layer_decision_boundary`.

---

## Round 3 Review Findings and Verified Repairs

### 1. Color Directed Derivation and Compound Shades
- Reconciled semantic boundary by implementing directed color derivation in `qedge_common.taxonomy.evaluate_color_derivation` and wiring it into `qedge_common.schema._evaluate_predicate`.
- Compound shade observations (`"dark red"`) satisfy base requirements (`"red"`) under `==` (`SUPPORTED`), while preserving `"dark red"` as the true `observed_value`.
- Disjoint color families (`"blue"` vs `"red"`) evaluate to `CONTRADICTED` on `==` and `SUPPORTED` on `!=`.

### 2. Conservative Unresolved Material Composition
- Reconciled material evaluation in both layers to treat unresolved compositions (`"oak veneer"`) conservatively.
- Unresolved material taxonomy evaluates to `UNKNOWN` for both `==` and `!=`.
- Verifier emits an explicit `UNKNOWN` claim with `observed_value=None`, producing a validated `UNKNOWN / ABSTAIN` decision under precedence rule 5.

### 3. Brand Whitespace and Case Normalization
- Implemented `qedge_common.taxonomy.evaluate_brand_derivation` and wired it into `qedge_common.schema._evaluate_predicate`.
- String comparisons strip leading and trailing whitespace and fold case.
- Brand equality (`==`, `!=`) enforces complete normalized brand identities without substring equality.
- Source observation fidelity is strictly preserved: claim reports `" FixtureCo "` without rewriting.

---

## Round 2 Review Findings and Verified Repairs

### 1. Brand Predicate Semantics and Observation Fidelity
- Separated equality (`==`, `!=`) from containment (`contains`, `not_contains`) in `evaluate_brand()`.
- Enforced complete normalized brand identity comparison; case normalization does not introduce substring equality.
- Eliminated `norm_obs_val = str(req_val)` rewriting; claims strictly report actual source observations.
- `"oak chair brand Fixture"` evaluates to `CONTRADICTED / DO_NOT_RECOMMEND` with `observed_value="FixtureCo"`.
- `"oak chair not brand Fixture"` evaluates to `SUPPORTED / RECOMMEND` with `observed_value="FixtureCo"`.

### 2. Categorical Observation Fidelity and Contract Derivation Semantics
- Eliminated all rewriting branches in `verifier.py`: `norm_obs_val` is always `obs_val` across brand, material, color, and other categoricals.
- Implemented explicit material taxonomy derivation in `qedge_common.taxonomy.evaluate_material_derivation` and wired it into `qedge_common.schema._evaluate_predicate`.
- Validates that `observed_value="oak"` legitimately derives and satisfies `required_value="wood"` under `==`, while preserving `"oak"` as the true `observed_value`.

### 3. Eligible Structured Evidence Status Enforcement
- Checked eligible status: `chunk_status = chunk.get("status")`.
- If structured fact has `chunk_status != "present"`, verifier immediately emits a claim with `status="UNKNOWN"`, `observed_value=None`, and skips value evaluation and text fallback.
- Preserves deterministic `UNKNOWN / ABSTAIN` decision precedence for non-present numeric and categorical facts.

---

## Round 1 Spot-Check Repairs

1. **Mixed-Unit Range Parsing Normalization:** Preserved `(value, unit)` associations and normalized to `cm` before ordering bounds without raw value reordering.
2. **Implicit Dimension Defaults Removal:** Removed silent fallback to `"width"`. Bare ranges record ambiguities and abstain under contract v2 precedence.
3. **Mandatory Constraint & Disjunction Preservation:** Removed subjective qualifiers (`comfortable`) from filler words, keeping them in `unparsed_clauses` to block positive recommendations. Detected unsupported disjunctions (`OR` expressions).
4. **Fixture Deduplication & Evaluation Claims:** Replaced duplicate `heldout-38` with distinct mixed-unit case. Preserved 80 strictly unique query/item pairs. Corrected provenance to internal authored testbed; independent generalization review is pending.
5. **Answer Boundary Fidelity:** Formatted generator responses reflecting decision `action` (`RECOMMEND`, `DO_NOT_RECOMMEND`, `ABSTAIN`) with exact citations.

---

## 80-Case Internal Regression Testbed Evaluation

The 80-case program ([`decision_cases_80.json`](../../edge-rag-demo/tests/fixtures/decision_cases_80.json)) serves as an internal regression testbed with explicit provenance (`"authored-regression-suite-v1; internal development benchmark"`).

### Results
- **False-Positive Decision Rate on Adverse Cases:** **0 / 60 (0.0%)** (zero false positive decisions on any adverse case across both development and held-out splits).
- **Development Split Accuracy:** **40 / 40 (100.0%)**
- **Held-Out Split Accuracy:** **40 / 40 (100.0%)**
- **Semantic Validation:** 80 / 80 decisions validate against contract v2.
- **Status:** Authored regression checks pass. **Independent generalization review remains pending.**

---

## Files Changed

- [`qedge-common/src/qedge_common/taxonomy.py`](../../qedge-common/src/qedge_common/taxonomy.py): Added `_SHADE_MODIFIERS`, `parse_color_description()`, bounded affirmative color grammar, conservative handling of non-affirmative color descriptions, and authoritative derivation functions.
- [`qedge-common/src/qedge_common/schema.py`](../../qedge-common/src/qedge_common/schema.py): Integrated bounded affirmative color derivation, brand derivation, and whitespace stripping into `_evaluate_predicate()`.
- [`qedge-common/CONTRACTS.md`](../../qedge-common/CONTRACTS.md): Documented exact numeric equality across contract boundary without float tolerance, bounded affirmative color grammar, non-affirmative abstention, string normalization, and eligible evidence status.
- [`qedge-common/tests/test_contract_semantics.py`](../../qedge-common/tests/test_contract_semantics.py): Added tests for non-affirmative color descriptions, exact numeric equality inside/outside former tolerance (`48.0000005`), inequality `!=`, and exact unit conversion equality.
- [`edge-rag-demo/src/edge_rag/verifier.py`](../../edge-rag-demo/src/edge_rag/verifier.py): Removed float tolerance (`1e-6`) in favor of exact equality `==` and inequality `!=`, preserved source observations, emitted `UNKNOWN` for non-affirmative or non-present evidence.
- [`edge-rag-demo/src/edge_rag/taxonomy.py`](../../edge-rag-demo/src/edge_rag/taxonomy.py): Delegated color, material, and brand derivation directly to authoritative `qedge_common.taxonomy` functions.
- [`edge-rag-demo/src/edge_rag/requirements.py`](../../edge-rag-demo/src/edge_rag/requirements.py): Added support for `!=`, `not equal to` in dimension parsing; added recognized compound shade modifiers to color regex to prevent unparsed clauses.
- [`edge-rag-demo/tests/test_assistant_adversarial.py`](../../edge-rag-demo/tests/test_assistant_adversarial.py): Expanded `test_semantic_cross_layer_decision_boundary` to 30 table-driven cross-layer cases covering recognized compound shades, non-affirmative descriptions, generic-to-specific asymmetry, exact numeric equality inside/outside former tolerance, and unit conversion.
- [`handoff/research-review/04-assistant-correctness/results.json`](04-assistant-correctness/results.json): Updated machine-readable results summary with Round 4 repairs and probe metrics.
- [`handoff/research-review/04-assistant-correctness/probes/reproduce_stage04_round4_blockers.py`](04-assistant-correctness/probes/reproduce_stage04_round4_blockers.py): Reproduction probe for Round 4 blockers (3 reproduced pre-repair, 0 active post-repair).
- [`handoff/research-review/04-assistant-correctness/probes/acceptance_assistant_gaps.py`](04-assistant-correctness/probes/acceptance_assistant_gaps.py): Expanded acceptance probe (15 checks with subchecks for affirmative color grammar and numeric tolerance).
- [`handoff/research-review/04-assistant-correctness/logs/`](04-assistant-correctness/logs/): Updated logs for edge-tests, qedge-common-tests, llm-ctest, schema-hashes, historical-hashes, edge-bench, qedge-eval, and round 4 repro/postrepair.

---

## Next Steps

Per the execution contract, work stops here for independent review. Stage 05 (`05-real-model-baseline`) is NOT started and remains blocked until Stage 04 repairs are accepted.
