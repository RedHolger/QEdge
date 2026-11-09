# Research review 02 contracts handoff

Date: 29 October 2026  
Scope: Stage 02 (`02-contracts`) shared contracts repair and versioned migration only.  
Prerequisites: `00-rebaseline` (ACCEPTED), `01-kernel-safety` (independently ACCEPTED at `reviews/research-review/01-kernel-safety.md`).  
Application code outside the contract boundary, accepted Stage 01 kernel-safety edits, historical evidence, and claims ledgers were strictly preserved.

---

## Verdict

**Self-assessment: ACCEPTED for the bounded Stage 02 contract-repair scope. Independent review: PENDING.**

This is an author handoff and self-review; it is explicitly **not** an independent review. Stages 03–12 remain blocked pending independent review and acceptance of this stage. Passing schema or semantic validation does not represent factual truth and does not turn synthetic records into model or NPU measurements.

---

## Commands and results

All verification commands were executed from the workspace root against local Python environments. Logs are preserved under `handoff/research-review/02-contracts/logs/`.

| Command | Exit | Passed | Failed | Skipped | Status | Evidence |
|---|---:|---:|---:|---:|---|---|
| `qedge-common-tests` | 0 | 48 | 0 | 0 | PASS | [log](02-contracts/logs/qedge-common-tests.log) |
| `qedge-eval` | 0 | 11 | 0 | 0 | PASS | [log](02-contracts/logs/qedge-eval.log) |
| `edge-tests` | 0 | 20 | 0 | 1 | PASS | [log](02-contracts/logs/edge-tests.log) |
| `edge-bench` | 0 | — | — | — | PASS (emits v2 `measured:false`) | [log](02-contracts/logs/edge-bench.log) |
| `records-smoke` | 0 | — | — | — | PASS (emits validated v2 smoke) | [log](02-contracts/logs/records-smoke.log) |
| `historical-hashes` | 0 | 30 | 0 | 0 | PASS | [log](02-contracts/logs/historical-hashes.log) |
| `schema-hashes` | 0 | 7 | 0 | 0 | PASS (v1 byte-for-byte identical) | [log](02-contracts/logs/schema-hashes.log) |
| `acceptance-gaps` | 0 | 16 | 0 | 0 | PASS (10 gaps rejected, 6 fixtures valid) | [log](02-contracts/logs/acceptance-gaps.log) |
| `baseline-gaps` | 0 | — | — | — | DEFECTS_DEMONSTRATED | [log](02-contracts/logs/baseline-gaps.log) |

*Notes on counts:*
- `qedge-common-tests`: 48 unit tests (covering semantic rules, adversarial negatives, boundary validators, date/finite parsing, v1 reader compatibility, and producer/consumer integration).
- `qedge-eval`: 11 valid example payloads checked across all 7 schemas (v1 examples for all 7, plus v2 examples for the 4 migrated schemas).
- `edge-tests`: 20 passing tests, 0 failures, 1 explicit skip requiring `QEDGE_RUN_CATALOG_INTEGRATION=1` with a licensed catalog.
- `schema-hashes`: All 7 baseline v1 schema files match their recorded SHA256 hashes byte-for-byte.
- `historical-hashes`: All 30 protected historical manifest files match their baseline hashes.

---

## Changed and created files

### Contract documentation and specification
- [`qedge-common/CONTRACTS.md`](../../qedge-common/CONTRACTS.md): Authoritative documentation of predicate grammar (`<`, `<=`, `>`, `>=`, `==`, `!=`, `contains`, `not_contains`), value/unit compatibility, negation mapping, ambiguity/unparsed clauses, candidate vs no-candidate response structures, deterministic decision response precedence, evidence resolution, measurement invariants, and RFC 3339 / finite-number boundaries.
- [`qedge-common/README.md`](../../qedge-common/README.md): Documented the v1/v2 versioning model, `--schema-version` selection, and CONTRACTS reference.
- [`qedge-common/schemas/README.md`](../../qedge-common/schemas/README.md): Documented schema inventory, current versions, and invariants.
- [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md): Updated shared-contract rules to state v1/v2 schema versions, candidate/no-candidate distinctions, mandatory predicate coverage, and non-proof nature of validation.

### Schema versioning and migration (v2 schemas)
- [`qedge-common/schemas/v2/decision.schema.json`](../../qedge-common/schemas/v2/decision.schema.json): Schema v2 for candidate-specific and no-candidate decisions; requires explicit requirements, claims with requirement linkage, unparsed clauses, and ambiguities.
- [`qedge-common/schemas/v2/evidence_claim.schema.json`](../../qedge-common/schemas/v2/evidence_claim.schema.json): Schema v2 for standalone claims; aligns with nested claims, enforces source/revision and conditional observed-value requirements.
- [`qedge-common/schemas/v2/benchmark_result.schema.json`](../../qedge-common/schemas/v2/benchmark_result.schema.json): Schema v2 separating execution metrics from optional outputs/token traces; enforces non-synthetic measured runs, optional cache hit rates, and execution evidence.
- [`qedge-common/schemas/v2/hardware_status.schema.json`](../../qedge-common/schemas/v2/hardware_status.schema.json): Schema v2 enforcing non-contradictory hardware flags and device verification for measured NPU status.
- Preserved byte-for-byte unchanged at v1:
  - `qedge-common/schemas/product.schema.json`
  - `qedge-common/schemas/retrieval_request.schema.json`
  - `qedge-common/schemas/retrieval_result.schema.json`
  - `qedge-common/schemas/decision.schema.json` (v1 historical reader)
  - `qedge-common/schemas/evidence_claim.schema.json` (v1 historical reader)
  - `qedge-common/schemas/benchmark_result.schema.json` (v1 historical reader)
  - `qedge-common/schemas/hardware_status.schema.json` (v1 historical reader)

### Implementation and boundary validators
- [`qedge-common/src/qedge_common/schema.py`](../../qedge-common/src/qedge_common/schema.py):
  - Added version selection (`schema_versions()`, `payload_schema_version()`, explicit `version` argument, default auto-detection).
  - Added `load_json()` rejecting `NaN`, `Infinity`, `-Infinity`, and unparseable floats (`1e309`).
  - Added recursive `_nonfinite_errors()` checking in-memory payloads.
  - Added RFC 3339 date-time format checker with mandatory timezone offsets.
  - Implemented semantic validators: `_claim_semantic_errors()`, `_decision_semantic_errors()`, `_benchmark_semantic_errors()`, and `_hardware_semantic_errors()`.
  - Added predicate evaluation supporting strict and inclusive numerical bounds, equality, scalar inequality `!=`, string `contains`, and `not_contains`.
- [`qedge-common/src/qedge_common/cli.py`](../../qedge-common/src/qedge_common/cli.py): Added `--schema-version` flag and wired `load_json()` and semantic validation into the CLI entry point.
- [`qedge-common/src/qedge_common/eval.py`](../../qedge-common/src/qedge_common/eval.py): Updated evaluation loop to validate both v1 and v2 examples.
- [`qedge-common/pyproject.toml`](../../qedge-common/pyproject.toml): Added wheel force-include for `tests/fixtures/valid` into `qedge_common/examples` so schemas and example fixtures are fully self-contained in packaged installations.
- [`qedge-common/src/qedge_common/records.py`](../../qedge-common/src/qedge_common/records.py): Added validated producer helper `fixture_smoke_benchmark()` emitting valid v2 fixture-smoke benchmark records with `measured: false`, separate outputs, and optional cache policy.
- [`edge-rag-demo/pyproject.toml`](../../edge-rag-demo/pyproject.toml) & [`edge-rag-demo/uv.lock`](../../edge-rag-demo/uv.lock): Declared explicit dependency on `qedge-common` with workspace path source.
- [`edge-rag-demo/src/edge_rag/bench.py`](../../edge-rag-demo/src/edge_rag/bench.py): Updated fixture smoke producer to emit v2 records with `measured: false`, separated outputs, no invented cache hit rate, and enforced contract validation via `_validate_record` (failing loudly with `RuntimeError` if validator is missing or broken, rather than silently bypassing it).
- [`edge-rag-demo/tests/test_pipeline.py`](../../edge-rag-demo/tests/test_pipeline.py): Added tests verifying that `bench.py` emits a valid contract record and fails with an actionable `RuntimeError` if validation cannot run.

### Fixtures and tests
- Checked-in v2 valid fixtures:
  - [`qedge-common/tests/fixtures/valid/v2/decision.json`](../../qedge-common/tests/fixtures/valid/v2/decision.json)
  - [`qedge-common/tests/fixtures/valid/v2/decision_no_candidate.json`](../../qedge-common/tests/fixtures/valid/v2/decision_no_candidate.json)
  - [`qedge-common/tests/fixtures/valid/v2/evidence_claim.json`](../../qedge-common/tests/fixtures/valid/v2/evidence_claim.json)
  - [`qedge-common/tests/fixtures/valid/v2/benchmark_result.json`](../../qedge-common/tests/fixtures/valid/v2/benchmark_result.json)
  - [`qedge-common/tests/fixtures/valid/v2/hardware_status.json`](../../qedge-common/tests/fixtures/valid/v2/hardware_status.json)
- [`qedge-common/tests/test_schemas.py`](../../qedge-common/tests/test_schemas.py): Updated to check Draft 2020-12 validity and examples across all versions.
- [`qedge-common/tests/test_contract_semantics.py`](../../qedge-common/tests/test_contract_semantics.py): 35 new tests testing strict/inclusive comparisons, scalar negation, containment, missing/contradicted/conflicting evidence, no-candidate responses, invalid timestamps, nonfinite numbers, measurement invariants, hardware contradictions, records producer validation, CLI exit codes, and cross-package edge bench boundary validation.

### Stage 02 handoff artifacts
- [`handoff/research-review/02-contracts/PLAN.md`](02-contracts/PLAN.md): Observable failing behavior before editing, test reference, and migration design.
- [`handoff/research-review/02-contracts/baseline-schema-hashes.json`](02-contracts/baseline-schema-hashes.json): Baseline SHA256 hashes of the seven v1 schema files.
- [`handoff/research-review/02-contracts/probes/baseline_gaps.py`](02-contracts/probes/baseline_gaps.py): Pre-fix probe demonstrating the baseline gaps accepted by v1.
- [`handoff/research-review/02-contracts/probes/acceptance_gaps.py`](02-contracts/probes/acceptance_gaps.py): Acceptance probe verifying that v2 rejects all 10 baseline gaps while positive and historical fixtures pass.
- [`handoff/research-review/02-contracts/logs/`](02-contracts/logs/): Full captured logs from test and evaluation executions.

---

## Contract semantics and precedence rules

As documented in [`qedge-common/CONTRACTS.md`](../../qedge-common/CONTRACTS.md):

1. **Predicate grammar:**
   - Supported operators: `<`, `<=`, `>`, `>=`, `==`, `!=`, `contains`, `not_contains`.
   - Strict comparisons `<` and `>` reject boundary values; inclusive `<=` and `>=` accept them.
   - `!=` is explicit scalar inequality; negative clauses (e.g. `not oak`) must not be encoded as `==` with reversed claims.
   - `not_contains` is explicit negated substring containment; units must be null.
   - Types and units must match: numeric comparisons require identical non-empty units; string/boolean predicates must have null units.
2. **Decision response types & precedence:**
   - **No candidate:** `item_id: null`, empty `claims: []`, status `ABSTAIN`, action `ABSTAIN`.
   - **Unresolved / conflicting evidence:** unresolved mandatory clauses (`unparsed_clauses` or `ambiguities`), or conflicting evidence (`SUPPORTED` and `CONTRADICTED` for same requirement) -> `ABSTAIN` / `ABSTAIN`.
   - **Contradiction:** any claim `CONTRADICTED` (including optional) -> `CONTRADICTED` / `DO_NOT_RECOMMEND`. Contradictions block recommendation.
   - **Supported:** at least one requirement, complete mandatory coverage, no unresolved/ambiguous clauses, all corresponding claims `SUPPORTED`, consistent item IDs, and resolvable evidence -> `SUPPORTED` / `RECOMMEND`.
   - **Unknown:** missing mandatory coverage or unknown claims with no contradiction or conflict -> `UNKNOWN` / `ABSTAIN`.
   - Candidate with no requirements -> invalid record (or `ABSTAIN` / `ABSTAIN`).
3. **Measurement invariants:**
   - Synthetic records (`fixture-smoke`, `synthetic-kernel`) cannot masquerade as measured model runs; `measured` must be `false`.
   - `measured: true` requires non-synthetic kind, a non-null model identity, and a reproducible provenance log reference.
   - `npu_status: "measured"` requires Snapdragon presence, QNN SDK availability, verified device identity, QNN/NPU execution backend, and execution evidence reference.
   - Contradictory hardware flags (e.g. absent Snapdragon with non-null model, measured NPU with CPU backend, or measured NPU with unverified device) are rejected.
   - Benchmark outputs/tokens are isolated in the versioned `outputs` object, never placed into `metrics`.
   - `cache.hit_rate` is optional and omitted for non-caching workloads.
4. **Data boundaries:**
   - RFC 3339 date-time strings with explicit timezone offsets are enforced at boundaries.
   - Nonfinite numbers (`NaN`, `Infinity`, `-Infinity`, and unrepresentable magnitudes) are rejected at JSON loading and in-memory validation.
5. **Truth rule:**
   - Schema and semantic validation proves internal coherence; it does **not** constitute proof of real-world catalog facts, model runs, or hardware availability.

---

## Claim changes

- **No claim upgrades:** No new model throughput, energy, latency, offload, retrieval quality, or NPU claims are made.
- **Contract tightening:** Records claiming measurements without evidence or claiming NPU status on CPU stubs now strictly fail validation.
- **Fixture smoke clarification:** Both `qedge_common.records.fixture_smoke_benchmark` and `edge_rag.bench.run_smoke` emit `measured: false` with explicit notes that no Snapdragon/QNN hardware was used.

---

## Open defects and remaining limitations

These limitations are deliberately retained for subsequent roadmap stages:
1. **Evidence store concurrency and authorization (Stage 03 prerequisite):** Stage 02 defines contracts only; SQLite evidence store authorization and concurrency are not modified.
2. **Edge assistant natural language requirement parser (Stage 07):** Assistant query parsing still uses simple regex/fixture rules in `edge_rag`; full predicate parsing and decision policy will be implemented in Stage 07.
3. **Real model assets and NPU toolchain (Stage 05 / 11):** No model weights or QNN SDK binaries are downloaded or assumed. NPU status remains `unverified`.

---

## Next prerequisites

1. Stop for **independent review** of Stage 02 (`02-contracts`).
2. Upon independent acceptance, proceed to **Stage 03** (`03-evidence-store`).
