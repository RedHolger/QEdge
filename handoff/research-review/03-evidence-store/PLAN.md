# Stage 03 Plan: Evidence Store Ownership, Atomicity, and Immutable Citations

## Observable Failing Behavior Before Editing

The baseline probe at `handoff/research-review/03-evidence-store/probes/baseline_store_gaps.py` demonstrated that the legacy `store.py` suffered from multiple critical defects:
1. **Cross-owner read permitted:** Bob was able to retrieve Alice's private documents because `retrieve()` discarded the requester identity without any authorization check (`del requester_id`).
2. **Cross-owner overwrite permitted:** Bob was able to upload new chunks to Alice's document ID, silently hijacking her document.
3. **Privilege spoofing:** Any caller could pass the arbitrary string `"system"` as owner or requester and be treated identically to system services.
4. **No deletion / restore / takeover protection:** There was no deletion or restoration API, and anyone could upload to an arbitrary document ID.
5. **Non-atomic concurrency:** Concurrency control relied on an in-memory `threading.Lock()`, which provides zero synchronization across independent SQLite connections or processes.
6. **Mutable citations rewriting historical evidence:** Citations were unversioned (`{doc_id}:{chunk_idx}`). Updating a document from version 1 to version 2 altered the content returned when looking up that document ID, rewriting historical decision evidence.

## Architectural Design

### 1. Trust Boundary and Principal Model
- Supported interface: Local single-user CLI.
- Owner and requester strings are plain identifiers, NOT authentication credentials.
- Passing the string `"system"` treats the caller as an unprivileged user whose username happens to be `"system"`.
- System and administrative privileges are restricted to `Principal` instances constructed via trusted factories (`Principal.system()`), initialized by trusted local runtime code (e.g. the CLI runner or catalog pre-loader).
- Database file security is governed by local filesystem permissions; no claim of protection against physical file tampering is made.

### 2. Structured Evidence Schema
- `documents`: `id`, `owner_id`, `version`, `is_public`, `is_deleted`, `created_at`, `updated_at`.
- `document_history`: `doc_id`, `version`, `owner_id`, `is_public`, `is_deleted`, `committed_at`.
- `chunks`: `doc_id`, `version`, `chunk_idx`, `chunk_id`, `text`, `item_id`, `variant_id`, `attribute`, `value_text`, `value_num`, `unit`, `status`, `source_location`, `source_revision`, `source_hash`, `created_at`.
- Typed facts preserve original numeric values and units without regex-parsing formatted display strings.
- Missing provenance stays explicitly `"unknown"`; revisions are never fabricated.

### 3. Comprehensive Authorization
- `upload`: Enforces expected version, cross-owner write denial, and deleted document takeover denial.
- `retrieve`: Only owner or administrator may read private documents. Public documents (`is_public=True`) are accessible to all principals.
- `delete` / `restore`: Only owner or administrator may soft-delete or restore a document.
- `resolve_citation`: Historical citation access checks authorization against the historical version snapshot.

### 4. Cross-Connection Atomic Transactions
- Updates use `BEGIN IMMEDIATE` transactions in SQLite with `PRAGMA busy_timeout = 10000;`.
- Version comparison (`current == expected_version`), ownership checks, document updates, history logging, and chunk insertions are executed inside a single immediate transaction.
- When two connections race with the same `expected_version`, exactly one transaction succeeds (returning the incremented version) and the other fails with `VersionConflictError`.
- Any failure midway through write triggers `ROLLBACK`, leaving zero partial updates.

### 5. Immutable Citation Resolution
- Format: `cite:{doc_id}:v{version}:c{chunk_idx}:{hash_prefix}`.
- Resolves against historical chunks and `document_history`.
- Updating a source document creates a new version; citations pointing to older versions continue resolving to their original facts and provenance.
- Integrity hashes verify stored byte consistency (not external source truth).

### 6. Safe Versioned Database Migration
- Tracked via `schema_migrations(version, applied_at)`.
- Transactional migration from v1 to v2 adds columns and history tables without data loss.
- Legacy chunks are marked `source_revision: "unknown"` and `source_location: "legacy"`.
- Repeat migration is idempotent; errors trigger atomic rollback.
