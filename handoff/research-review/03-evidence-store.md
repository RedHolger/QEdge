# Research review 03 evidence store handoff (post-migration repair)

Date: 30 October 2026  
Scope: Stage 03 (`03-evidence-store`) evidence ownership, atomic concurrency, typed fact preservation, immutable citations, and versioned schema migration (v1 -> v2 -> v3).  
Prerequisites: `00-rebaseline` (ACCEPTED), `01-kernel-safety` (ACCEPTED), `02-contracts` (independently ACCEPTED at `reviews/research-review/02-contracts.md`).  
Application code outside the evidence storage boundary, Stage 01 kernel safety implementations, v1 JSON schemas, and historical evidence artifacts were strictly preserved.

---

## Verdict

**Self-assessment: REPAIRED & ACCEPTED for Stage 03 storage handoff. Independent review: PENDING.**

This is an author handoff and self-review; it is explicitly **not** an independent review. The remaining migration blocker identified during review (prior-v2 databases leaving `integrity_hash` NULL and bypassing canonical verification) was reproduced, repaired via an explicit versioned v3 migration, and verified with dedicated regression tests. Stage 04 (`04-assistant-correctness`) and subsequent roadmap stages remain blocked pending independent review and acceptance of this stage. Integrity hashes prove consistency relative to stored bytes and canonical payloads, not factual truth of external sources. No model, catalog, or NPU claims are made or implied.

---

## Commands and Results

All verification commands were executed from the workspace root against local Python environments. Dated logs are recorded under `handoff/research-review/03-evidence-store/logs/`.

| Command | Exit | Passed | Failed | Skipped | Status | Evidence |
|---|---:|---:|---:|---:|---|---|
| `edge-tests` | 0 | 43 | 0 | 1 | PASS | [log](03-evidence-store/logs/2026-10-30-v2-migration-edge-tests.log) |
| `qedge-common-tests` | 0 | 48 | 0 | 0 | PASS | [log](03-evidence-store/logs/2026-10-30-v2-migration-qedge-common-tests.log) |
| `qedge-eval` | 0 | 11 | 0 | 0 | PASS (11 schema examples across 7 schemas) | [log](03-evidence-store/logs/2026-10-30-v2-migration-qedge-common-eval.log) |
| `edge-rag-eval` | 0 | — | — | — | PASS (explicit `not-run` placeholder) | [log](03-evidence-store/logs/2026-10-30-v2-migration-edge-rag-eval.log) |
| `edge-bench` | 0 | — | — | — | PASS (emits validated v2 smoke) | [log](03-evidence-store/logs/2026-10-30-edge-bench.log) |
| `historical-hashes` | 0 | 30 | 0 | 0 | PASS (30/30 verified) | [log](03-evidence-store/logs/2026-10-30-v2-migration-historical-hashes.log) |
| `schema-hashes` | 0 | 7 | 0 | 0 | PASS (all 7 v1 schemas byte-for-byte identical) | [log](03-evidence-store/logs/2026-10-30-v2-migration-schema-hashes.log) |
| `acceptance-store-gaps` | 0 | 13 | 0 | 0 | PASS (all 13 gap checks closed) | [probe](03-evidence-store/probes/acceptance_store_gaps.py) |
| `reproduce-v2-migration-gap` | 0 | 1 | 0 | 0 | PASS (reproduced before, resolved after) | [log](03-evidence-store/logs/2026-10-30-v2-migration-repro.log) |
| `reproduce-repair-defects` | 0 | 6 | 0 | 0 | PASS (initial spot checks verified closed) | [log](03-evidence-store/logs/2026-10-30-repair-repro.log) |
| `llm-infer-tests` (CTest) | 0 | 3 | 0 | 0 | PASS (Stage 01 kernel safety preserved) | CTest 3/3 |

*Notes on counts & evaluation commands:*
- `edge-tests`: 44 total tests: 43 passed, 0 failed, 1 explicitly skipped (`RealCatalogIntegrationTests.test_catalog_loads`, requiring `QEDGE_RUN_CATALOG_INTEGRATION=1` with a licensed catalog). Includes 23 tests in `test_evidence_store.py` (including regression tests for prior-v2 upgrade, atomic rollback on migration failure, and tamper rejection).
- `qedge-common-tests`: 48 tests pass across contract semantics and schema definitions.
- `qedge-eval`: Validates 11 example payloads across all 7 schemas (v1 and v2).
- `edge-rag-eval`: Explicit `not-run` evaluation placeholder for retrieval and decision quality (which require human relevance judgments scheduled for Stages 06 and 09). It does not emit fabricated zero scores.
- `schema-hashes`: All 7 baseline v1 schema files match their recorded SHA256 hashes byte-for-byte.
- `historical-hashes`: All 30 protected historical manifest files match their baseline hashes.

---

## Repaired Defects and Verification Evidence

### 1. Access Revocation and Historical Authorization
- **Defect:** Uploading a document with `is_public=True`, then updating it with `is_public=False` left `public_flag` as `1` because `is_public` defaulted to `False` and was preserved via `1 if is_public else int(row['is_public'])`. Furthermore, resolving citations to older public versions bypassed revocation when a document was made private or deleted.
- **Repair:**
  - In `upload()`, `is_public` parameter accepts `Optional[bool] = None`. When `None`, existing document visibility is preserved; when explicit `False`, public visibility is revoked (`public_flag = 0`).
  - In `resolve_citation()`, the store queries the active document state. If the active document is deleted, historical resolution is restricted to the owner or administrator. If the active document has been made private, historical public metadata cannot bypass access revocation.
  - Owners retain full access to their deleted documents and retained historical versions.
- **Evidence:** [`test_public_access_revocation_and_historical_authorization`](../../edge-rag-demo/tests/test_evidence_store.py).

### 2. Resolution of Migrated Legacy Versions
- **Defect:** For legacy v1 databases where `documents.version = 2` and chunks existed at versions 1 and 2, older chunks remained in `chunks`, but resolving `cite:old:v1:c0` raised `NotFoundError` because no history row existed for version 1.
- **Repair:**
  - Backfilled all distinct `(doc_id, version)` tuples from `chunks` into `document_history` using `INSERT OR IGNORE`.
  - Legacy historical versions without recorded timestamps or provenance are marked conservatively: `is_public = 0`, `is_deleted = 0`, `committed_at = 'unknown'`, `source_revision = 'unknown'`, and `source_location = 'legacy'`.
  - Added a schema version guard in `_init_or_migrate_db()` that raises `StoreError` when encountering an unsupported future version (`current_ver > SCHEMA_VERSION`).
- **Evidence:** [`test_legacy_migration_retains_all_historical_versions`](../../edge-rag-demo/tests/test_evidence_store.py) and [`test_migration_rollback_and_unsupported_future_version`](../../edge-rag-demo/tests/test_evidence_store.py).

### 3. Typed Fact Preservation & Alignment
- **Defect:** Boolean `True` was stored as string `"True"` and retrieved as string. Integers were coerced to float `42.0`. Non-finite numbers (`NaN`, `Infinity`) were accepted silently. Extra facts beyond chunks were silently dropped.
- **Repair:**
  - Added `value_type` (`bool`, `int`, `float`, `str`, `null`) and `value_int` columns to `chunks`.
  - Implemented `_extract_fact_value()`: booleans stored as `("bool", "true"|"false", None, 1|0)`, exact integers in `value_int`, finite floats in `value_num`.
  - Non-finite numbers (`NaN`, `Infinity`, `-Infinity`) rejected with `ValueError`. Unsupported types with `TypeError`.
  - `upload()` validates that `len(facts) == len(chunks)` if facts are provided; mismatches raise `ValueError`.
  - `_reconstruct_fact_value()` returns exact types across retrieval, citation resolution, and database reopen.
- **Evidence:** [`test_typed_fact_preservation_and_alignment`](../../edge-rag-demo/tests/test_evidence_store.py).

### 4. Citation Identity & Canonical Payload Integrity
- **Defect:** Document IDs with colons (e.g. `"item:variant"`) broke citation parsing. Citation hashes only covered raw text bytes, leaving structured fact values and provenance unprotected against database tampering.
- **Repair:**
  - Added URI percent-encoding: `encode_doc_id()` encodes colons as `%3A` (decoded via `decode_doc_id()`), and `_CITATION_RE` uses non-greedy matching `(?P<doc_id>.+?)`.
  - Implemented `_compute_integrity_hash()` computing SHA-256 over canonical JSON serialization of text, attribute, typed value representation, unit, status, item ID, variant ID, source location, and source revision.
  - Stored `source_hash` (raw text bytes) and `integrity_hash` (canonical payload) as separate columns in `chunks`.
  - Citations embed `integrity_hash[:8]`. Modifying any fact or provenance field fails with `CitationError`.
- **Evidence:** [`test_citation_identity_with_colons_and_delimiters`](../../edge-rag-demo/tests/test_evidence_store.py) and [`test_citation_payload_integrity_verification`](../../edge-rag-demo/tests/test_evidence_store.py).

### 5. Catalog Provenance & Unit Preservation
- **Defect:** `load_catalog` hardcoded `source_revision="catalog:fixture"`, defaulted missing units to `"cm"`, and stored converted/rounded dimensions as facts.
- **Repair:**
  - Updated `load_catalog()` to extract `source_revision` from each catalog row (`row.get("source_revision") or "unknown"`).
  - Preserved original attribute values and units without defaulting missing units to `"cm"`.
  - In `verify_attribute()`, dimension normalization is performed dynamically at query time without premature rounding before comparison (`raw_w * 2.54 <= max_w`).
- **Evidence:** [`test_catalog_provenance_actual_revisions_and_units`](../../edge-rag-demo/tests/test_evidence_store.py).

### 6. Connection Ownership & Coherent Transactional Reads
- **Defect:** Multi-threaded operations on the same connection lacked serialization. Read operations did not execute within database transactions, allowing non-coherent point-in-time reads.
- **Repair:**
  - Added connection serialization via `self._lock = threading.RLock()`.
  - Implemented `_read_transaction()` using `BEGIN` and `COMMIT` within `self._lock` to ensure authorization queries and chunk data queries read from a consistent point-in-time database snapshot.
  - Retained `_write_transaction()` using `BEGIN IMMEDIATE` for atomic cross-connection version checking and writing.
- **Evidence:** [`test_connection_ownership_and_thread_serialization`](../../edge-rag-demo/tests/test_evidence_store.py) and [`test_two_independent_connections_racing_on_same_expected_version`](../../edge-rag-demo/tests/test_evidence_store.py).

### 7. Prior-v2 Database Migration & Integrity Enforcement (Post-Review Fix)
- **Defect:** When upgrading a database matching the initial Stage 03 v2 layout (where `schema_migrations` recorded version 2, and `chunks` had `value_text`/`value_num` but lacked `value_type`/`integrity_hash`), opening the database added the new columns with `NULL` values. In `resolve_citation()`, canonical verification was guarded by `if stored_integ_hash is not None:`. Because `stored_integ_hash` was `NULL`, canonical integrity verification was silently bypassed, allowing a tampered width of `999` to resolve successfully.
- **Repair:**
  - Introduced explicit, versioned schema migration with `SCHEMA_VERSION = 3`.
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
- **Evidence:** [`test_v2_to_v3_migration_and_integrity_enforcement`](../../edge-rag-demo/tests/test_evidence_store.py) and [`test_migration_atomic_rollback_on_injected_failure`](../../edge-rag-demo/tests/test_evidence_store.py).

---

## Changed Files Summary

- [`edge-rag-demo/src/edge_rag/store.py`](../../edge-rag-demo/src/edge_rag/store.py):
  - Bumped `SCHEMA_VERSION = 3`.
  - Implemented versioned migrations `_migrate_v1_to_v2()` and `_migrate_v2_to_v3()`.
  - Enforced non-null `integrity_hash` and unconditional canonical verification in `resolve_citation()`.
  - Preserved legacy strings as strings without boolean inference.
- [`edge-rag-demo/src/edge_rag/verifier.py`](../../edge-rag-demo/src/edge_rag/verifier.py): Catalog provenance and dynamic unit conversion.
- [`edge-rag-demo/tests/test_evidence_store.py`](../../edge-rag-demo/tests/test_evidence_store.py): Added prior-v2 upgrade, atomic rollback on failure, and tamper rejection tests (total 23 tests in suite).
- [`handoff/research-review/03-evidence-store/probes/acceptance_store_gaps.py`](03-evidence-store/probes/acceptance_store_gaps.py): Expanded to 13 acceptance checks.
- [`handoff/research-review/03-evidence-store/probes/reproduce_v2_migration_gap.py`](03-evidence-store/probes/reproduce_v2_migration_gap.py): Reproduction probe for the v2 migration gap.
- [`handoff/research-review/03-evidence-store/logs/`](03-evidence-store/logs/): Fresh dated verification logs.

---

## Remaining Limitations & Next Prerequisites

1. **Natural language assistant parsing & decision policy:** The assistant's natural language requirement parsing and recommendation logic remain template fixtures in `edge_rag`; these belong to **Stage 04 (`04-assistant-correctness`)**.
2. **Real-model integration & NPU toolchain:** No model weights or QNN binaries are downloaded or assumed. Hardware verification remains deferred to **Stage 11 (`11-qualcomm-deployment`)**.

**Next Step:**  
Stop for independent review of Stage 03 repairs. Upon independent acceptance, proceed to **Stage 04: `04-assistant-correctness`**.
