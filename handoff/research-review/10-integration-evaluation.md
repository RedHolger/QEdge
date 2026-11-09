# Stage 10 Handoff: Integration and Evaluation (Repaired)

**Prompt ID:** `10-integration-evaluation`  
**Date:** 2026-11-05  
**Self-Review Verdict:** `REPAIR COMPLETE` (Bounded self-review; pending independent review)  
**Historical Hashes:** 30/30 Verified (`scripts/verify_historical_hashes.py`)  
**Hardware Status:** NPU unverified / CPU baseline execution (no Hexagon NPU offload or energy measurement claimed)  
**Retrieval Gate Status:** Software repairs ACCEPTED (105/105 tests, 49/49 schema tests); Human retrieval-quality gate remains **BLOCKED** pending annotation of the 1,009 active review queue candidates and completion of the 120-query evaluation  
**Secondary Model Status:** `Qwen/Qwen2.5-0.5B-Instruct` remains **DEFERRED / BLOCKED**

---

## 1. Executive Summary

Stage 10 delivers a single local assistant (`edge_rag.assistant.LocalAssistant`) featuring:
1. **Explicit Find and Check Modes:**
   - `find(query, ...)`: Integrates candidate retrieval with strict authorization filtering, typed requirement parsing, candidate verification, recommendation selection, and citation resolution.
   - `check(item_id, query, ...)`: Directly verifies a target candidate item against parsed requirements, bypassing retrieval (`retrieve_ms = 0.0`), resolving citations, and emitting a deterministic answer.
2. **Complete Offline Extractive Baseline:**
   - Full deterministic functionality operates strictly offline without model downloads or runtime requirements (`enable_generation=False` default).
3. **Structurally Enforced Generation Contract & Strict Presentation Validation:**
   - Preserves the runtime as an optional dependency (`enable_generation=True`).
   - Instruction texts are strictly static system instructions with zero untrusted user or catalog strings.
   - Catalog evidence is strictly isolated in an untrusted data block bounded to $\le 500$ characters.
   - Generation uses a constrained presentation selection interface where models output structured JSON matching exactly `{"style": "standard"|"compact"|"bullet", "selected_claims": [<int>, ...]}`.
   - Strict validation rejects permissive payloads with extra keys (e.g. `{"invented_fact": "fireproof"}`), missing required fields, non-integer or boolean indices (`[true]`), out-of-range indices (`[99]`), or unapproved styles (refusing silent normalization).
   - Any model output outside this contract is rejected and falls back cleanly to the deterministic extractive answer, recording the exact `generation_fallback_reason`.
4. **Strict Fail-Closed Citation Verification & Output-Contract Separation:**
   - Citation verification is applied to every decision action (`RECOMMEND`, `DO_NOT_RECOMMEND`, `ABSTAIN`).
   - When citation resolution raises an error (e.g., `CitationError`, integrity mismatch, access revocation), the decision fails closed to `ABSTAIN`.
   - Contradicted-item answers and decision outputs exclude failed evidence, contradicted facts, and unresolved citations.
   - Strict separation contract in JSON output:
     * `citations` contains ONLY successfully resolved, authorized evidence (`valid: True`).
     * `citation_diagnostics` contains failed-resolution diagnostics (`valid: False`).
     * The identical separation holds inside every `verified_candidates` entry (`cand["citations"]` vs. `cand["citation_diagnostics"]`).
     * In complete citation failure, `citations` is empty (`[]`) and `citation_diagnostics` retains all failure records.
     * In partial citation failure, `citations` contains only the valid citation(s), while `citation_diagnostics` retains the failed citation(s), and failed claims are excluded from `decision["claims"]` and answer text.
     * Diagnostic details are strictly kept separate from validated evidence (reported in `citation_diagnostics` and CLI diagnostics section rather than polluting answer text or decision reasons).
5. **Find-Mode Candidate Authorization Boundary:**
   - Every candidate returned by retrieval is filtered against evidence store read permissions before inclusion. Unauthorized private candidate IDs, titles, metadata, evidence, and citations are completely excluded from results and context.
6. **Deadline Handling & Cooperative Timeout Verification:**
   - Deadlines are cooperatively checked before and after all pipeline stages (parsing, retrieval, verification, and generation) and immediately prior to every success return.
   - Timeouts are propagated to runners supporting `timeout_s`. If a blocking runner exceeds `timeout_s`, deadline violation is cooperatively detected upon return and triggers `AssistantTimeoutError` (`status="ERROR"`).
7. **Mandatory Predicate Safety Invariant:**
   - A candidate is **NEVER** recommended if any mandatory requirement is `UNKNOWN`, if there are `unparsed_clauses`, or if there are `ambiguities` ($\to$ `ABSTAIN`).
   - If any requirement is contradicted, verdict is strictly `DO_NOT_RECOMMEND`.
8. **10 Mandatory Scenario Categories:**
   - All 10 curated integration scenarios pass with actual repository artifacts.
   - Image query refusal (`scen-09`) is explicitly verified as refusal-path coverage, with actual image execution separately skipped.
9. **80-Case Internal Regression Decision Benchmark:**
   - Evaluated 80 previously exercised internal regression cases (`authored-regression-suite-v1`).
   - Untouched held-out claims have been removed; independent generalization evaluation remains pending.
   - **0 / 58 False SUPPORTED decisions on adverse cases** (0.0% observed rate; Wilson 95% CI: $[0.000, 0.0621]$).
   - $100\%$ decision accuracy across all 80 authored regression cases.
10. **Elimination of Stale Claims:**
   - Stale answer text claiming decision policy was unimplemented has been excised from v2 decision generation.
   - "Production-grade" claims have been removed while human retrieval quality, secondary model validation, and hardware offload gates remain unresolved.

---

## 2. Changed and Created Files

| File | Status | Description |
|---|---|---|
| `edge-rag-demo/src/edge_rag/assistant.py` | Repaired | Closed find-mode authorization leak; enforced exact presentation selection contract; enforced strict separation between `citations` (only valid, authorized evidence) and `citation_diagnostics` (failed-resolution diagnostics) at top-level result and within each `verified_candidates` entry; implemented fail-closed citation verification across all actions (RECOMMEND, DO_NOT_RECOMMEND, ABSTAIN); excluded failed claims from answers, claims lists, and model context; added deadline checks before success return and after generation; removed "production-grade" claims. |
| `edge-rag-demo/src/edge_rag/generator.py` | Repaired | Removed stale text claiming Stage 07 decision policy was unimplemented from v2 decision record formatting. |
| `edge-rag-demo/src/edge_rag/integration_eval.py` | Repaired | Labeled 80 cases as internal regression benchmark, removed untouched-held-out claim, documented image refusal as refusal-path coverage, added find-mode candidate leak probe to tenant isolation evaluation. |
| `edge-rag-demo/tests/test_pipeline.py` | Repaired | Updated assertion to verify removal of stale decision policy disclaimer. |
| `edge-rag-demo/tests/test_integration_assistant.py` | Expanded | Added regression unit tests for complete and partial citation failures in find and check modes, verifying strict separation of valid citations and failure diagnostics across result and `verified_candidates`, citation integrity errors, exact presentation contract validation (rejecting extra keys, unapproved styles, boolean indices, and missing keys), find-mode candidate filtering, timeout enforcement after blocking generation, invented facts rejection, and phrasing recommendation rejection on contradicted items (29 passed). |
| `handoff/research-review/10-integration-evaluation/results.json` | Regenerated | Machine-readable payload regenerated with updated benchmark provenance, refusal-path coverage notes, candidate isolation verification, and latency metrics. |
| `handoff/research-review/10-integration-evaluation.md` | Updated | This handoff artifact documenting Stage 10 repairs. |

---

## 3. Verification Commands and Exit Codes

All verification commands executed from the repository root:

```bash
# 1. Historical manifest checksum verification (30/30 verified)
python3 scripts/verify_historical_hashes.py
# Exit Code: 0
# Output: historical manifest: 30 verified, 0 failure(s)

# 2. Integration assistant unit tests (29 passed)
PYTHONPATH=qedge-common/src:edge-rag-demo/src edge-rag-demo/.venv/bin/python -m unittest edge-rag-demo/tests/test_integration_assistant.py
# Exit Code: 0
# Output: Ran 29 tests in 0.359s -- OK

# 3. Full edge-rag-demo test suite (133 passed, 1 skipped; 134 passed with catalog integration)
PYTHONPATH=qedge-common/src:edge-rag-demo/src edge-rag-demo/.venv/bin/python -m unittest discover edge-rag-demo/tests
# Exit Code: 0
# Output: Ran 134 tests in 1.307s -- OK (skipped=1)
# Note: unittest "Ran 134, skipped=1" denotes 133 passed and 1 skipped (catalog integration test).
# Running with QEDGE_RUN_CATALOG_INTEGRATION=1 independently passed all 134 tests (Ran 134 tests in 1.320s -- OK).

# 4. qedge-common shared schema test suite (49 passed)
PYTHONPATH=qedge-common/src edge-rag-demo/.venv/bin/python -m unittest discover qedge-common/tests
# Exit Code: 0
# Output: Ran 49 tests in 0.137s -- OK

# 5. Integration evaluation harness execution
PYTHONPATH=qedge-common/src:edge-rag-demo/src edge-rag-demo/.venv/bin/python -m edge_rag.integration_eval \
  --catalog edge-rag-demo/tests/fixtures/catalog.jsonl \
  --cases edge-rag-demo/tests/fixtures/decision_cases_80.json \
  --output handoff/research-review/10-integration-evaluation/results.json
# Exit Code: 0
# Output: Results written to handoff/research-review/10-integration-evaluation/results.json
```
```

---

## 4. Ten Scenario Categories Results

| Scenario ID | Category | Query / Modality | Candidate | Expected Verdict | Actual Verdict | Status | Notes |
|---|---|---|---|---|---|---|---|
| `scen-01` | `supported` | "dining chair with oak material under 60cm wide" | `fixture-chair-oak` | `RECOMMEND` | `RECOMMEND` | PASS | Exact attribute match ($48\text{ cm} \le 60\text{ cm}$, oak == oak); citations verified. |
| `scen-02` | `contradicted` | "dining chair with oak material under 60cm wide" | `fixture-chair-wide` | `DO_NOT_RECOMMEND` | `DO_NOT_RECOMMEND` | PASS | Direct contradiction ($75\text{ cm} > 60\text{ cm}$, aluminum != oak); recommendation blocked. |
| `scen-03` | `missing_fact` | "chair width <= 60cm" | `fixture-chair-unknown` | `ABSTAIN` | `ABSTAIN` | PASS | Width attribute status is missing in SQLite; decision `UNKNOWN` $\to$ `ABSTAIN`. |
| `scen-04` | `ambiguous_units` | "dining chair under 60" | `fixture-chair-oak` | `ABSTAIN` | `ABSTAIN` | PASS | Bare number flagged as ambiguity; refuses to assume centimeters vs. inches. |
| `scen-05` | `negation` | "dining chair width under 80cm and not aluminum" | `fixture-chair-wide` | `DO_NOT_RECOMMEND` | `DO_NOT_RECOMMEND` | PASS | Negation constraint (`material != aluminum`) violated by observed aluminum chair. |
| `scen-06` | `unsupported_requirement` | "dining chair with built-in quantum charger" | `fixture-chair-oak` | `ABSTAIN` | `ABSTAIN` | PASS | Unsupported capability preserved as unparsed clause; forces `ABSTAIN`. |
| `scen-07` | `source_conflict_update` | "chair width <= 60cm" | `doc-conflict` / `doc-upd` | `ABSTAIN` / update: `RECOMMEND` | `ABSTAIN` / update: `RECOMMEND` | PASS | Conflicting facts cause `ABSTAIN`; updating doc v1 ($75\text{ cm}$) to v2 ($50\text{ cm}$) changes verdict while v1 citation still resolves to $75\text{ cm}$. |
| `scen-08` | `no_result` | "quantum flux capacitor 9999" | None | `ABSTAIN` (`item_id=None`) | `ABSTAIN` (`item_id=None`) | PASS | Zero candidates retrieved; emits explicit `ABSTAIN` decision without item ID. |
| `scen-09` | `optional_image_query` | "--image oak_chair.jpg" | None | `ABSTAIN` | `ABSTAIN` | PASS | Image query refusal-path coverage; visual specification inference refused without verified ground truth; actual image execution separately skipped. |
| `scen-10` | `generation_unavailable_failure` | "dining chair with oak material under 60cm wide" | `fixture-chair-wide` | `DO_NOT_RECOMMEND` | `DO_NOT_RECOMMEND` | PASS | Adversarial model attempting to recommend contradicted item intercepted by Contract Guard; fell back to deterministic response with `generation_fallback_reason: "contract_violation: model attempted to recommend non-recommended candidate"`. |

---

## 5. Internal Regression Decision Benchmark (80 Cases)

### 5.1 Dataset Classification & Scope

> [!NOTE]
> All 80 cases are internal regression cases (`authored-regression-suite-v1; internal development benchmark`). Untouched held-out claims have been removed, and independent generalization evaluation remains pending.

| Metric | Split 1 (40 cases) | Split 2 (40 cases) | Overall (80 cases) |
|---|---|---|---|
| **Accuracy** | 100.0% (40/40) | 100.0% (40/40) | 100.0% (80/80) |
| **Abstention Count / Rate** | 17 / 42.5% | 17 / 42.5% | 34 / 42.5% |
| **Coverage Count / Rate** | 23 / 57.5% | 23 / 57.5% | 46 / 57.5% |
| **False-SUPPORTED Numerator** | 0 | 0 | **0** |
| **False-SUPPORTED Denominator** | 29 adverse cases | 29 adverse cases | **58 adverse cases** |
| **False-SUPPORTED Rate** | **0.0%** | **0.0%** | **0.0%** |
| **Wilson 95% Confidence Interval** | $[0.000, 0.1170]$ | $[0.000, 0.1170]$ | **$[0.000, 0.0621]$** |

### 5.2 Decision Confusion Matrix (Actual vs Predicted)

| Ground Truth \ Predicted | `RECOMMEND` | `DO_NOT_RECOMMEND` | `ABSTAIN` | Total Expected |
|---|:---:|:---:|:---:|:---:|
| **`RECOMMEND`** | **22** | 0 | 0 | 22 |
| **`DO_NOT_RECOMMEND`** | 0 | **24** | 0 | 24 |
| **`ABSTAIN`** | 0 | 0 | **34** | 34 |
| **Total Predicted** | 22 | 24 | 34 | 80 |

---

## 6. End-to-End Latencies and Failure Taxonomy

### 6.1 Stage Latencies (ms across 80 cases in Check Mode)

| Pipeline Stage | Mean (ms) | Min (ms) | p50 (ms) | p95 (ms) | Max (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| **`parse_ms`** | 0.151 | 0.000 | 0.146 | 0.237 | 0.269 |
| **`retrieve_ms`** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **`verify_ms`** | 0.401 | 0.000 | 0.396 | 0.571 | 0.612 |
| **`generate_ms`** | 0.002 | 0.000 | 0.002 | 0.003 | 0.003 |
| **`total_ms`** | **0.561** | 0.007 | **0.546** | **0.802** | **0.854** |

### 6.2 Failure & Abstention Taxonomy Breakdown

Across all non-`RECOMMEND` evaluations in the 80 cases:

| Categorized Reason | Count | Policy & Safety Impact |
|---|:---:|---|
| **`ADVERSE_CONTRADICTION`** | 24 | Explicit negative constraint violated or attribute out of bounds $\to$ `DO_NOT_RECOMMEND`. |
| **`MISSING_SOURCE_EVIDENCE`** | 22 | Mandatory required attribute missing or unknown in catalog $\to$ `ABSTAIN`. |
| **`AMBIGUOUS_UNIT`** | 10 | Unspecified dimension unit (bare number) $\to$ `ABSTAIN`. |
| **`UNRESOLVED_REQUIREMENTS`** | 2 | Unresolved requirement clauses $\to$ `ABSTAIN`. |
| **`CITATION_VERIFICATION_FAILURE`**| 0 | All cited facts successfully resolved against SQLite commits. |
| **`UNPARSED_CLAUSE`** | 0 | Handled under explicit scenario testbed. |

---

## 7. Multi-Tenant Request Isolation & Security Governance

1. **Find-Mode Candidate Authorization Boundary:**
   - Raw candidate retrieval is strictly filtered against document access control.
   - Private candidates owned by Tenant A cannot appear in Tenant B's raw candidate list, verified candidates list, metadata, generated context, or citations.
2. **Check-Mode Direct Read Protection:**
   - Attempting to check a private document owned by another principal raises `AssistantAuthorizationError` (`code: "UNAUTHORIZED"`).
3. **Immutable Citation Auditing:**
   - Citations use format `cite:<encoded_doc_id>:v<ver>:c<idx>:<integrity_hash>`.
   - Cross-tenant citation resolution attempts raise `AuthorizationError`.
   - Document revision updates create immutable versioned commits.

---

## 8. Remaining Gates and Prerequisites

1. **Stage 09 Retrieval Quality Gate: BLOCKED**
   - Software repairs to the retrieval evaluator, candidate run schema, and negative evidence probes are accepted.
   - The human retrieval quality evaluation gate remains **BLOCKED** pending human annotation of the **1,009 active review queue candidates** in `edge-rag-demo/data/annotation/pilot-v1/review_queue.json` and completion of the full 120-query evaluation.
2. **Secondary Model Validation: DEFERRED**
   - Secondary model validation for `Qwen/Qwen2.5-0.5B-Instruct` remains deferred.
3. **Hardware & NPU Claims: UNVERIFIED**
   - No Qualcomm Snapdragon or QNN Hexagon NPU execution is claimed. All measurements are CPU baselines on macOS.
4. **Generalization Evaluation: PENDING**
   - Independent held-out evaluation remains pending; the 80 internal cases are classified strictly as an internal regression benchmark.
5. **Independent Review Requirement:**
   - Stop after Stage 10 self-review for independent review. Do not start subsequent roadmap stages.
