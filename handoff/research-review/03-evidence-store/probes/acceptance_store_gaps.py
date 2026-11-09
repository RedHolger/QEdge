"""Acceptance probe for Stage 03 evidence store.

Verifies that the previously demonstrated baseline gaps are fully closed:
1. Cross-owner read is denied with AuthorizationError.
2. Cross-owner overwrite is denied with AuthorizationError.
3. Forged string "system" as requester/owner is denied administrator access.
4. Deleted-object ownership takeover is denied with AuthorizationError.
5. Old citations resolve the original version fact after a source update.
6. Malformed and corrupted citations fail explicitly.
7. Concurrent writes on same expected version yield one success and one conflict.
8. Migration of legacy v1 schema succeeds, preserves data, and enforces ownership.
"""
from __future__ import annotations

import concurrent.futures
import sqlite3
import tempfile
from pathlib import Path
import sys

from edge_rag.store import (
    AuthorizationError,
    CitationError,
    EvidenceStore,
    Fact,
    NotFoundError,
    Principal,
    VersionConflictError,
)


def expect_pass(name: str, fn) -> bool:
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as exc:
        print(f"FAIL: {name} ({type(exc).__name__}: {exc})", file=sys.stderr)
        return False


def test_cross_owner_read_denied():
    store = EvidenceStore(":memory:")
    store.upload("doc-alice", "alice", ["secret alice fact"], is_public=False)
    try:
        store.retrieve("doc-alice", "bob")
        raise AssertionError("Cross-owner read was allowed!")
    except AuthorizationError:
        pass


def test_cross_owner_overwrite_denied():
    store = EvidenceStore(":memory:")
    store.upload("doc-alice", "alice", ["alice fact"])
    try:
        store.upload("doc-alice", "bob", ["hijack"])
        raise AssertionError("Cross-owner write was allowed!")
    except AuthorizationError:
        pass


def test_forged_system_role_denied():
    store = EvidenceStore(":memory:")
    store.upload("doc-private", "alice", ["top secret"], is_public=False)
    # Caller passing "system" string must not bypass private authorization
    try:
        store.retrieve("doc-private", requester="system")
        raise AssertionError("String 'system' bypassed authorization!")
    except AuthorizationError:
        pass
    # Only trusted Principal.system() succeeds
    res = store.retrieve("doc-private", requester=Principal.system())
    assert len(res) == 1


def test_deleted_object_takeover_denied():
    store = EvidenceStore(":memory:")
    store.upload("doc-alice", "alice", ["initial fact"])
    store.delete("doc-alice", "alice")
    try:
        store.upload("doc-alice", "bob", ["bob takeover"])
        raise AssertionError("Deleted document takeover was allowed!")
    except AuthorizationError:
        pass


def test_citation_immutability():
    store = EvidenceStore(":memory:")
    store.upload(
        "item-1",
        "alice",
        ["width: 48cm"],
        facts=[Fact(attribute="width", value=48.0, unit="cm", source_revision="v1-rev")],
        is_public=True,
    )
    cite_v1 = store.retrieve("item-1", "alice")[0]["citation"]

    # Update to v2
    store.upload(
        "item-1",
        "alice",
        ["width: 60cm"],
        facts=[Fact(attribute="width", value=60.0, unit="cm", source_revision="v2-rev")],
        is_public=True,
    )

    # Resolve old citation
    resolved = store.resolve_citation(cite_v1, "alice")
    assert resolved["version"] == 1
    assert resolved["fact"]["value"] == 48.0
    assert resolved["provenance"]["source_revision"] == "v1-rev"


def test_malformed_and_mismatched_citations():
    store = EvidenceStore(":memory:")
    store.upload("doc-1", "alice", ["fact"], is_public=True)
    cite = store.retrieve("doc-1", "alice")[0]["citation"]

    try:
        store.resolve_citation("invalid-citation", "alice")
        raise AssertionError("Malformed citation accepted")
    except CitationError:
        pass

    parts = cite.split(":")
    corrupt = ":".join(parts[:-1] + ["badhash1"])
    try:
        store.resolve_citation(corrupt, "alice")
        raise AssertionError("Mismatched hash accepted")
    except CitationError:
        pass


def test_concurrent_write_atomicity():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "concurrency.db")
        with EvidenceStore(db_path) as s:
            s.upload("concur-doc", "alice", ["seed v0"])

        def worker(wid: int):
            with EvidenceStore(db_path) as s:
                try:
                    s.upload(
                        "concur-doc",
                        "alice",
                        [f"worker {wid}"],
                        expected_version=1,
                    )
                    return True
                except VersionConflictError:
                    return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(worker, 1)
            f2 = ex.submit(worker, 2)
            results = [f1.result(), f2.result()]

        assert results.count(True) == 1, f"Expected exactly 1 success, got {results}"
        assert results.count(False) == 1, f"Expected exactly 1 conflict, got {results}"


def test_migration_and_provenance_preservation():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "legacy_probe.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, owner_id TEXT, version INT, created_at TEXT)")
        conn.execute("CREATE TABLE chunks (doc_id TEXT, chunk_idx INT, text TEXT, version INT, PRIMARY KEY (doc_id, chunk_idx, version))")
        conn.execute("INSERT INTO documents VALUES ('d1', 'alice', 1, '2026-10-29T00:00:00Z')")
        conn.execute("INSERT INTO chunks VALUES ('d1', 0, 'legacy chunk', 1)")
        conn.commit()
        conn.close()

        with EvidenceStore(db_path) as s:
            items = s.retrieve("d1", "alice")
            assert len(items) == 1
            assert items[0]["source_revision"] == "unknown"
            assert items[0]["source_location"] == "legacy"
            try:
                s.retrieve("d1", "bob")
                raise AssertionError("Legacy document was open to cross-owner read")
            except AuthorizationError:
                pass


def test_public_revocation_and_historical_auth():
    store = EvidenceStore(":memory:")
    # Upload public v1
    store.upload("doc-revoc", "alice", ["v1 text"], is_public=True)
    cite_v1 = store.retrieve("doc-revoc", "alice")[0]["citation"]

    # Bob can resolve v1 initially
    res1 = store.resolve_citation(cite_v1, "bob")
    assert res1["text"] == "v1 text"

    # Alice revokes public access in v2
    store.upload("doc-revoc", "alice", ["v2 text"], expected_version=1, is_public=False)

    # Bob cannot retrieve or resolve historical citation
    try:
        store.retrieve("doc-revoc", "bob")
        raise AssertionError("Bob retrieved private document")
    except AuthorizationError:
        pass

    try:
        store.resolve_citation(cite_v1, "bob")
        raise AssertionError("Bob bypassed revocation with historical citation")
    except AuthorizationError:
        pass

    # Alice deletes document: Bob still denied, Alice allowed
    store.delete("doc-revoc", "alice")
    try:
        store.resolve_citation(cite_v1, "bob")
        raise AssertionError("Bob resolved citation of deleted doc")
    except AuthorizationError:
        pass

    res_alice = store.resolve_citation(cite_v1, "alice")
    assert res_alice["text"] == "v1 text"


def test_legacy_multi_version_migration():
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "legacy_v2.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, owner_id TEXT, version INT, created_at TEXT)")
        conn.execute("CREATE TABLE chunks (doc_id TEXT, chunk_idx INT, text TEXT, version INT)")
        conn.execute("INSERT INTO documents VALUES ('doc-legacy', 'alice', 2, '2026-01-01T00:00:00Z')")
        conn.execute("INSERT INTO chunks VALUES ('doc-legacy', 0, 'chunk v1 text', 1)")
        conn.execute("INSERT INTO chunks VALUES ('doc-legacy', 0, 'chunk v2 text', 2)")
        conn.commit()
        conn.close()

        with EvidenceStore(db_path) as s:
            # Resolve historical v1 citation
            v1_cite = "cite:doc-legacy:v1:c0"
            res = s.resolve_citation(v1_cite, "alice")
            assert res["text"] == "chunk v1 text"
            assert res["provenance"]["source_revision"] == "unknown"
            assert res["provenance"]["source_location"] == "legacy"


def test_typed_fact_preservation():
    store = EvidenceStore(":memory:")
    facts = [
        Fact("is_valid", True),
        Fact("total_count", 42),
        Fact("weight", 3.14),
        Fact("tag", "chair"),
        Fact("empty", None),
    ]
    chunks = ["c1", "c2", "c3", "c4", "c5"]
    store.upload("doc-typed", "alice", chunks, facts=facts, is_public=True)
    res = store.retrieve("doc-typed", "alice")

    assert res[0]["value"] is True and isinstance(res[0]["value"], bool)
    assert res[1]["value"] == 42 and isinstance(res[1]["value"], int) and not isinstance(res[1]["value"], bool)
    assert res[2]["value"] == 3.14 and isinstance(res[2]["value"], float)
    assert res[3]["value"] == "chair" and isinstance(res[3]["value"], str)
    assert res[4]["value"] is None


def test_colon_citation_and_payload_integrity():
    store = EvidenceStore(":memory:")
    doc_id = "item:variant:chair_1"
    store.upload(doc_id, "alice", ["sample chunk"], facts=[Fact("attr", "val")], is_public=True)
    cite = store.retrieve(doc_id, "alice")[0]["citation"]

    # Colon ID resolves
    resolved = store.resolve_citation(cite, "alice")
    assert resolved["doc_id"] == doc_id

    # Tampering with attribute triggers integrity error
    store._connection.execute("UPDATE chunks SET attribute = 'tampered' WHERE doc_id = ?", (doc_id,))
    try:
        store.resolve_citation(cite, "alice")
        raise AssertionError("Tampered fact passed integrity check")
    except CitationError:
        pass


def test_prior_v2_migration_and_tamper_rejection():
    import hashlib
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "v2_acceptance.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        conn.execute("INSERT INTO schema_migrations VALUES (2, '2026-10-29T20:00:00Z')")
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, owner_id TEXT, version INT, is_public INT, is_deleted INT, created_at TEXT, updated_at TEXT)")
        conn.execute("CREATE TABLE document_history (doc_id TEXT, version INT, owner_id TEXT, is_public INT, is_deleted INT, committed_at TEXT, PRIMARY KEY (doc_id, version))")
        conn.execute("CREATE TABLE chunks (doc_id TEXT, version INT, chunk_idx INT, chunk_id TEXT, text TEXT, item_id TEXT, variant_id TEXT, attribute TEXT, value_text TEXT, value_num REAL, unit TEXT, status TEXT, source_location TEXT, source_revision TEXT, source_hash TEXT, created_at TEXT, PRIMARY KEY (doc_id, version, chunk_idx))")
        conn.execute("INSERT INTO documents VALUES ('doc-prior-v2', 'alice', 1, 1, 0, '2026-10-29', '2026-10-29')")
        conn.execute("INSERT INTO document_history VALUES ('doc-prior-v2', 1, 'alice', 1, 0, '2026-10-29')")
        txt = "width: 10 cm (source: spec)"
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        conn.execute("INSERT INTO chunks VALUES ('doc-prior-v2', 1, 0, 'doc-prior-v2:v1:c0', ?, 'i1', 'v1', 'width', '10', 10.0, 'cm', 'present', 'spec', 'rev1', ?, '2026-10-29')", (txt, h))
        conn.commit()
        conn.close()

        # Migrate to v3
        with EvidenceStore(db_path) as s:
            cite = f"cite:doc-prior-v2:v1:c0:{h[:8]}"
            res = s.resolve_citation(cite, "alice")
            assert res["fact"]["value"] == 10.0
            # Tampering value must be caught
            s._connection.execute("UPDATE chunks SET value_num = 999.0 WHERE doc_id='doc-prior-v2'")
            try:
                s.resolve_citation(cite, "alice")
                raise AssertionError("Tampered 999.0 resolved successfully!")
            except CitationError:
                pass


def main() -> int:
    tests = [
        ("Cross-owner read denied", test_cross_owner_read_denied),
        ("Cross-owner overwrite denied", test_cross_owner_overwrite_denied),
        ("Forged 'system' string denied", test_forged_system_role_denied),
        ("Deleted-object takeover denied", test_deleted_object_takeover_denied),
        ("Immutable citation resolution", test_citation_immutability),
        ("Malformed and mismatched citations fail", test_malformed_and_mismatched_citations),
        ("Atomic concurrent writes across connections", test_concurrent_write_atomicity),
        ("Legacy migration preserves owners & marks provenance unknown", test_migration_and_provenance_preservation),
        ("Public access revocation & historical authorization enforcement", test_public_revocation_and_historical_auth),
        ("Legacy multi-version retention", test_legacy_multi_version_migration),
        ("Typed fact preservation (bool, int, float, null)", test_typed_fact_preservation),
        ("Colon-delimited citation & payload integrity validation", test_colon_citation_and_payload_integrity),
        ("Prior-v2 database migration integrity and tamper rejection", test_prior_v2_migration_and_tamper_rejection),
    ]

    all_ok = True
    for name, fn in tests:
        if not expect_pass(name, fn):
            all_ok = False

    if all_ok:
        print("ALL ACCEPTANCE CHECKS PASSED")
        return 0
    else:
        print("ACCEPTANCE CHECKS FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
