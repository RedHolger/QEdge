# Review: 03-evidence-store

Date: 30 October 2026  
Reviewed stage: `03-evidence-store`  
Review type: Independent review (evaluating initial handoff, spot-check blocking findings, applied repairs, post-review migration gap, and reproduction evidence)  
Review scope: Stage 03 evidence store ownership, atomic concurrency, typed fact preservation, immutable citations, versioned schema migrations (v1 -> v2 -> v3), and verifier integration only  

---

## Verdict

**ACCEPTED** (post-repair)

The evidence store repairs satisfy all requirements specified in `docs/research-review/prompts/03-evidence-store.md`. In addition to the six initial spot-check defects repaired earlier, a focused migration blocker identified during secondary review (upgrades from the initial Stage 03 v2 layout leaving `integrity_hash` NULL and bypassing canonical verification) was reproduced, resolved via an explicit versioned v3 migration, and verified with dedicated regression tests.

All verification suites pass cleanly:
- `edge-tests`: 43 passed, 0 failed, 1 expected skip (`RealCatalogIntegrationTests.test_catalog_loads`).
- `qedge-common-tests`: 48 passed, 0 failed, 0 skipped.
- `qedge-eval`: 11 valid example payloads checked across 7 schemas (v1 and v2).
- `edge-rag-eval`: Explicit `not-run` placeholder correctly reported (no fabricated scores).
- `historical-hashes`: All 30 protected historical manifest files verified byte-for-byte.
- `schema-hashes`: All 7 baseline v1 schema files verified byte-for-byte.
- `llm-infer-tests`: CTest 3/3 passed (Stage 01 kernel safety preserved).

This acceptance is strictly bounded to Stage 03. It does not accept Stages 04–12, does not implement assistant query parsing or decision recommendation logic (Stage 04), and does not claim real model execution or NPU hardware verification (Stage 11).

---

## Review Findings & Applied Repairs

### Initial Blocking Findings (Repaired)

1. **Access Revocation and Historical Authorization**
   - *Issue:* In `upload()`, `public_flag` was computed as `1 if is_public else (int(row["is_public"]) if row is not None else 0)`, preventing revocation with `is_public=False`. In `resolve_citation()`, the check examined only historical `is_public`, allowing revoked or soft-deleted documents to be resolved via older public citations.
   - *Repair:* `upload()` distinguishes omitted visibility (`None`) from explicit `False`. `resolve_citation()` enforces current document authorization and deletion status; deleted or private documents cannot be accessed via historical citations by unauthorized principals.

2. **Resolution of Migrated Legacy Versions**
   - *Issue:* For legacy v1 databases with chunks at older versions, `_migrate_to_v2()` populated `document_history` solely from the active row in `documents`. Resolving citations to older chunks raised `NotFoundError`.
   - *Repair:* Backfilled all distinct `(doc_id, version)` tuples from `chunks` into `document_history` with conservative defaults (`is_public = 0`, `is_deleted = 0`, `committed_at = 'unknown'`). Added a schema version guard rejecting future versions (`current_ver > SCHEMA_VERSION`).

3. **Typed Fact Preservation & Alignment**
   - *Issue:* Fact values were coerced (booleans to string `"True"`, integers to float `42.0`). Non-finite numbers were accepted silently. Extra facts beyond chunks were silently discarded.
   - *Repair:* Added `value_type` and `value_int` columns to `chunks`. Exact booleans, integers, finite floats, strings, and nulls are preserved. Non-finite numbers (`NaN`, `Infinity`) and unsupported types are rejected. Mismatched facts count vs chunks count raises `ValueError`.

4. **Citation Identity & Canonical Payload Integrity**
   - *Issue:* Document IDs containing colons broke citation parsing. Citation hashes only covered raw text bytes, leaving structured facts and provenance unprotected against database tampering.
   - *Repair:* Added URI percent-encoding for document identifiers (`encode_doc_id()`). Implemented canonical payload hashing (`_compute_integrity_hash()`) covering text, attributes, typed values, units, status, and provenance. Citations embed `integrity_hash[:8]`. Modifying stored facts or provenance fails citation resolution.

5. **Catalog Provenance & Unit Preservation**
   - *Issue:* `load_catalog` hardcoded `source_revision="catalog:fixture"`, defaulted missing units to `"cm"`, and stored converted/rounded dimensions as facts.
   - *Repair:* Extracted `source_revision` from catalog JSON rows. Preserved original values and units without defaulting missing units. Dynamic conversion in `verify_attribute()` avoids premature rounding before comparison.

6. **Connection Ownership & Coherent Transactional Reads**
   - *Issue:* Multi-threaded operations on the same connection lacked synchronization. Read operations did not run inside database transactions, allowing non-coherent point-in-time reads.
   - *Repair:* Added connection serialization via `self._lock = threading.RLock()`. Implemented `_read_transaction()` using `BEGIN` and `COMMIT` within `self._lock` for coherent point-in-time snapshots.

---

### Secondary Migration Blocker & Applied Repair

7. **Prior-v2 Database Migration & Integrity Enforcement**
   - *Issue:* When upgrading a database matching the initial Stage 03 v2 layout (where `schema_migrations` recorded version 2, and `chunks` had `value_text`/`value_num` but lacked `value_type`/`integrity_hash`), opening the database added the new columns with `NULL` values. In `resolve_citation()`, canonical verification was guarded by `if stored_integ_hash is not None:`. Because `stored_integ_hash` was `NULL`, canonical integrity verification was silently bypassed, allowing a tampered width of `999` to resolve successfully.
   - *Repair:*
     - Introduced an explicit, versioned schema migration with `SCHEMA_VERSION = 3`.
     - Implemented transactional `_migrate_v2_to_v3()`:
       - Adds `value_type`, `value_int`, and `integrity_hash` columns if missing.
       - Backfills all distinct historical versions from `chunks` missing from `document_history`.
       - Backfills canonical `integrity_hash` and `value_type` for all retained rows. Recoverable values/types are preserved; string `"True"` remains `"str"` (never inferred as boolean).
       - Verifies post-migration completeness: asserts zero rows with `integrity_hash IS NULL`, failing with `StoreError` otherwise.
       - Records version 3 in `schema_migrations`.
     - In `resolve_citation()`:
       - Rejects missing integrity metadata after migration: `stored_integ_hash is None` raises `CitationError`.
       - Unconditionally executes canonical integrity verification for every chunk.
       - Supports legacy citations carrying `source_hash[:8]` prefix alongside new citations carrying `integrity_hash[:8]`.
     - Opening a current-format database (`version >= 3`) executes no migration or update statements, ensuring corrupted records are never silently rebaselined with replacement hashes.
   - *Verification:* Verified by [`test_v2_to_v3_migration_and_integrity_enforcement`](../../edge-rag-demo/tests/test_evidence_store.py) and [`test_migration_atomic_rollback_on_injected_failure`](../../edge-rag-demo/tests/test_evidence_store.py).

---

## Independently Reproduced Evidence

All verification logs were captured cleanly and dated under `handoff/research-review/03-evidence-store/logs/`.

| Verification Target | Exit | Result | Evidence |
|---|---:|---|---|
| `make -C edge-rag-demo test` | 0 | 43 passed, 0 failed, 1 skipped | [2026-10-30-v2-migration-edge-tests.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-v2-migration-edge-tests.log) |
| `make -C qedge-common test` | 0 | 48 passed, 0 failed, 0 skipped | [2026-10-30-v2-migration-qedge-common-tests.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-v2-migration-qedge-common-tests.log) |
| `make -C qedge-common eval` | 0 | 11 examples across 7 schemas OK | [2026-10-30-v2-migration-qedge-common-eval.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-v2-migration-qedge-common-eval.log) |
| `make -C edge-rag-demo eval` | 0 | Explicit `not-run` placeholder | [2026-10-30-v2-migration-edge-rag-eval.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-v2-migration-edge-rag-eval.log) |
| `make -C edge-rag-demo bench` | 0 | Emits validated v2 contract record | [2026-10-30-edge-bench.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-edge-bench.log) |
| `historical-hashes` | 0 | 30 verified, 0 failures | [2026-10-30-v2-migration-historical-hashes.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-v2-migration-historical-hashes.log) |
| `schema-hashes` | 0 | All 7 v1 schemas byte-for-byte identical | [2026-10-30-v2-migration-schema-hashes.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-v2-migration-schema-hashes.log) |
| `acceptance-store-gaps` | 0 | 13/13 acceptance checks pass | [acceptance_store_gaps.py](../../handoff/research-review/03-evidence-store/probes/acceptance_store_gaps.py) |
| `reproduce-v2-migration-gap` | 0 | Demonstrates fix for NULL hash & tamper bypass | [2026-10-30-v2-migration-repro.log](../../handoff/research-review/03-evidence-store/logs/2026-10-30-v2-migration-repro.log) |
| `make -C llm-inference test` | 0 | CTest 3/3 passed (Release) | CTest output |

---

## Path-Scan Accounting

Running `python3.11 scripts/check_no_absolute_paths.py` reports:
```text
scanned 120 source/test files under /Users/manuvashistha/Developer/qualcomm
FOUND machine path: docs/research-review/evidence/asan-build.log:29
FOUND machine path: docs/research-review/evidence/asan-build.log:35
FOUND machine path: docs/research-review/evidence/research-extracted.txt:2283
3 machine-path match(es)
```
Accounting:
- As previously established, these 3 matches reside entirely within `docs/research-review/evidence/` in historical Stage 00 baseline rebaseline files (`asan-build.log` and `research-extracted.txt`).
- Their SHA256 hashes match [`handoff/research-review/00-rebaseline/preservation.json`](../../handoff/research-review/00-rebaseline/preservation.json).
- No source or test files in `qedge-common/`, `edge-rag-demo/`, or `llm-inference/` contain machine paths.

---

## Non-Blocking Findings & Deferred Work

1. **Natural language query parsing and decision policy:** The edge assistant's requirement extraction and recommendation logic remain template fixtures in `edge_rag`; full assistant correctness is scheduled for **Stage 04 (`04-assistant-correctness`)**.
2. **Real model assets & NPU execution verification:** No model weights or QNN SDK binaries are downloaded or assumed; hardware execution verification belongs to **Stage 11 (`11-qualcomm-deployment`)**. NPU status remains `unverified`.

---

## Conclusion & Next Prerequisites

Stage 03 (`03-evidence-store`) is independently **ACCEPTED** following the complete repair and verification of both the initial spot-check findings and the secondary migration/integrity blocker.

**Next Step:**  
Proceed to **Stage 04: `04-assistant-correctness`**, implementing assistant requirement extraction, taxonomy alignment, decision policy, and citation rendering.
