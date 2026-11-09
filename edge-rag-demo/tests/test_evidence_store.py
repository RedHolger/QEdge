"""Comprehensive acceptance tests for the Stage 03 evidence store.

Tests all required acceptance criteria:
- Cross-owner read/write/delete/restore denial.
- Forged privileged-name denial.
- Authorized and explicit public access.
- Deleted-object ownership takeover denial.
- Two independent connections racing on the same expected version.
- Failure midway through a write leaves no partial update.
- Old citations resolve the original facts after a source update.
- Unauthorized historical citation access is denied.
- Missing, malformed and mismatched citations fail explicitly.
- Persistence across close/reopen and legacy migration correctness.
"""
from __future__ import annotations

import concurrent.futures
import math
import sqlite3
import tempfile
from pathlib import Path
import unittest

from edge_rag.store import (
    AuthorizationError,
    CitationError,
    EvidenceStore,
    Fact,
    NotFoundError,
    Principal,
    StoreError,
    VersionConflictError,
)
from edge_rag.verifier import Verifier


class EvidenceStoreAcceptanceTests(unittest.TestCase):
    def test_cross_owner_read_denial(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-alice", "alice", ["secret alice fact"], is_public=False)

        # Alice can read her document
        results = store.retrieve("doc-alice", "alice")
        self.assertEqual(len(results), 1)
        self.assertIn("secret alice fact", results[0]["text"])

        # Bob cannot read Alice's private document
        with self.assertRaises(AuthorizationError):
            store.retrieve("doc-alice", "bob")

        # Anonymous cannot read Alice's private document
        with self.assertRaises(AuthorizationError):
            store.retrieve("doc-alice", None)

    def test_cross_owner_write_denial(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-alice", "alice", ["original fact"])

        # Bob attempts to overwrite Alice's document
        with self.assertRaises(AuthorizationError):
            store.upload("doc-alice", "bob", ["hijacked by bob"])

        # Content remains Alice's original
        results = store.retrieve("doc-alice", "alice")
        self.assertEqual(results[0]["text"], "original fact")

    def test_cross_owner_delete_and_restore_denial(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-alice", "alice", ["fact"])

        # Bob cannot delete Alice's document
        with self.assertRaises(AuthorizationError):
            store.delete("doc-alice", "bob")

        # Alice deletes her document
        self.assertTrue(store.delete("doc-alice", "alice"))

        # Once deleted, active retrieve returns empty
        self.assertEqual(store.retrieve("doc-alice", "alice"), [])

        # Bob cannot restore Alice's deleted document
        with self.assertRaises(AuthorizationError):
            store.restore("doc-alice", "bob")

        # Alice restores her document
        self.assertTrue(store.restore("doc-alice", "alice"))
        results = store.retrieve("doc-alice", "alice")
        self.assertEqual(len(results), 1)

    def test_forged_privileged_name_denial(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-private", "alice", ["top secret"], is_public=False)

        # Attacker passes string "system" as requester/owner
        with self.assertRaises(AuthorizationError):
            store.retrieve("doc-private", requester="system")

        with self.assertRaises(AuthorizationError):
            store.upload("doc-private", owner="system", text_chunks=["overwritten"])

        with self.assertRaises(AuthorizationError):
            store.delete("doc-private", owner="system")

        # Only a genuine Principal.system() can exercise system access
        trusted_sys = Principal.system()
        results = store.retrieve("doc-private", requester=trusted_sys)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "top secret")

    def test_authorized_and_explicit_public_access(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-catalog", "catalog_owner", ["public catalog specs"], is_public=True)

        # Anyone can read public documents
        res_alice = store.retrieve("doc-catalog", "alice")
        self.assertEqual(len(res_alice), 1)

        res_bob = store.retrieve("doc-catalog", "bob")
        self.assertEqual(len(res_bob), 1)

        res_anon = store.retrieve("doc-catalog", None)
        self.assertEqual(len(res_anon), 1)

        # But non-owner still cannot overwrite public document
        with self.assertRaises(AuthorizationError):
            store.upload("doc-catalog", "bob", ["unauthorized modification"])

    def test_deleted_object_ownership_takeover_denial(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-alice", "alice", ["alice text"])
        store.delete("doc-alice", "alice")

        # Bob attempts to take over the deleted document ID
        with self.assertRaises(AuthorizationError) as caught:
            store.upload("doc-alice", "bob", ["bob taking over"])
        self.assertIn("takeover denied", str(caught.exception))

        # Alice can upload an update to her own deleted document (restoring and incrementing version)
        v2 = store.upload("doc-alice", "alice", ["alice revived text"])
        self.assertEqual(v2, 2)
        results = store.retrieve("doc-alice", "alice")
        self.assertEqual(results[0]["text"], "alice revived text")

    def test_two_independent_connections_racing_on_same_expected_version(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "race.db")
            # Seed document v1
            with EvidenceStore(db_path) as init_store:
                init_store.upload("race-doc", "alice", ["seed v1"])

            # Two independent store instances pointing to the same database file
            def worker(worker_id: int):
                # Each thread opens its own store/connection
                with EvidenceStore(db_path) as s:
                    try:
                        ver = s.upload(
                            "race-doc",
                            "alice",
                            [f"update from worker {worker_id}"],
                            expected_version=1,
                        )
                        return ("SUCCESS", worker_id, ver)
                    except VersionConflictError as exc:
                        return ("CONFLICT", worker_id, str(exc))

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(worker, 1)
                f2 = executor.submit(worker, 2)
                res1 = f1.result()
                res2 = f2.result()

            results = [res1[0], res2[0]]
            self.assertIn("SUCCESS", results)
            self.assertIn("CONFLICT", results)
            self.assertEqual(results.count("SUCCESS"), 1)
            self.assertEqual(results.count("CONFLICT"), 1)

            # Verify final database state has version 2 with exactly one worker's text
            with EvidenceStore(db_path) as final_store:
                items = final_store.retrieve("race-doc", "alice")
                self.assertEqual(len(items), 1)
                self.assertTrue(
                    items[0]["text"] == "update from worker 1"
                    or items[0]["text"] == "update from worker 2"
                )

    def test_failure_midway_through_write_leaves_no_partial_update(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-atomic", "alice", ["initial"])

        # Attempt an upload with an invalid fact object that raises during chunk iteration
        class FaultyFact:
            def to_dict(self):
                raise RuntimeError("simulated error midway through write")

        with self.assertRaises(RuntimeError):
            store.upload(
                "doc-atomic",
                "alice",
                ["chunk 1", "chunk 2"],
                facts=[{"attribute": "ok"}, FaultyFact()],
            )

        # Document must remain at version 1 with only the initial chunk
        results = store.retrieve("doc-atomic", "alice")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "initial")

        # Verify chunks table has no orphan rows
        rows = store._connection.execute(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE doc_id = 'doc-atomic'"
        ).fetchone()
        self.assertEqual(rows["cnt"], 1)

    def test_old_citations_resolve_original_facts_after_source_update(self):
        store = EvidenceStore(":memory:")
        facts_v1 = [
            Fact(
                attribute="width",
                value=48.0,
                unit="cm",
                status="present",
                source_location="spec_v1",
                source_revision="rev-001",
            )
        ]
        store.upload("doc-item", "alice", ["width: 48cm"], facts=facts_v1, is_public=True)

        res_v1 = store.retrieve("doc-item", "alice")
        cite_v1 = res_v1[0]["citation"]  # cite:doc-item:v1:c0:...

        # Source is updated to version 2 with a different value
        facts_v2 = [
            Fact(
                attribute="width",
                value=65.0,
                unit="cm",
                status="present",
                source_location="spec_v2",
                source_revision="rev-002",
            )
        ]
        store.upload("doc-item", "alice", ["width: 65cm"], facts=facts_v2, is_public=True)

        # Active retrieval now shows version 2
        res_v2 = store.retrieve("doc-item", "alice")
        self.assertEqual(res_v2[0]["value"], 65.0)

        # Resolving the old citation returns the original v1 fact and provenance!
        resolved = store.resolve_citation(cite_v1, "alice")
        self.assertEqual(resolved["version"], 1)
        self.assertEqual(resolved["fact"]["value"], 48.0)
        self.assertEqual(resolved["fact"]["unit"], "cm")
        self.assertEqual(resolved["provenance"]["source_revision"], "rev-001")
        self.assertEqual(resolved["text"], "width: 48cm")

    def test_unauthorized_historical_citation_access_is_denied(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-private", "alice", ["private fact"], is_public=False)
        cite = store.retrieve("doc-private", "alice")[0]["citation"]

        # Alice can resolve her citation
        res = store.resolve_citation(cite, "alice")
        self.assertEqual(res["text"], "private fact")

        # Bob cannot resolve Alice's private citation
        with self.assertRaises(AuthorizationError):
            store.resolve_citation(cite, "bob")

        # Anonymous cannot resolve private citation
        with self.assertRaises(AuthorizationError):
            store.resolve_citation(cite, None)

    def test_missing_malformed_and_mismatched_citations_fail(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-1", "alice", ["hello world"], is_public=True)
        valid_cite = store.retrieve("doc-1", "alice")[0]["citation"]

        # 1. Malformed citations
        with self.assertRaises(CitationError):
            store.resolve_citation("not-a-citation", "alice")

        with self.assertRaises(CitationError):
            store.resolve_citation("cite:doc-1", "alice")

        with self.assertRaises(CitationError):
            store.resolve_citation("cite:doc-1:vX:cY", "alice")

        # 2. Missing document / version / chunk
        with self.assertRaises(NotFoundError):
            store.resolve_citation("cite:nonexistent:v1:c0", "alice")

        with self.assertRaises(NotFoundError):
            store.resolve_citation("cite:doc-1:v99:c0", "alice")

        with self.assertRaises(NotFoundError):
            store.resolve_citation("cite:doc-1:v1:c99", "alice")

        # 3. Mismatched integrity hash
        parts = valid_cite.split(":")
        corrupted_cite = ":".join(parts[:-1] + ["00000000"])
        with self.assertRaises(CitationError) as caught:
            store.resolve_citation(corrupted_cite, "alice")
        self.assertIn("Integrity hash mismatch", str(caught.exception))

    def test_persistence_across_close_and_reopen(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "persist.db")
            # Write data and close
            with EvidenceStore(db_path) as s1:
                s1.upload(
                    "item-persist",
                    "alice",
                    ["persistent chunk text"],
                    facts=[Fact(attribute="test_attr", value=123.45)],
                    is_public=True,
                )
                cite = s1.retrieve("item-persist", "alice")[0]["citation"]

            # Reopen new instance from same file
            with EvidenceStore(db_path) as s2:
                items = s2.retrieve("item-persist", "alice")
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]["text"], "persistent chunk text")
                self.assertEqual(items[0]["value"], 123.45)

                # Citation still resolves
                resolved = s2.resolve_citation(cite, "alice")
                self.assertEqual(resolved["text"], "persistent chunk text")
                self.assertEqual(resolved["fact"]["value"], 123.45)

    def test_legacy_migration_correctness_and_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "legacy.db")

            # 1. Create a raw legacy v1 database
            raw_conn = sqlite3.connect(db_path)
            raw_conn.execute(
                """
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            raw_conn.execute(
                """
                CREATE TABLE chunks (
                    doc_id TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    PRIMARY KEY (doc_id, chunk_idx, version)
                )
                """
            )
            raw_conn.execute(
                "INSERT INTO documents VALUES ('doc-legacy', 'alice', 1, '2026-10-29T12:00:00Z')"
            )
            raw_conn.execute(
                "INSERT INTO chunks VALUES ('doc-legacy', 0, 'legacy oak chair', 1)"
            )
            raw_conn.commit()
            raw_conn.close()

            # 2. Open with EvidenceStore: automatic migration to v2
            with EvidenceStore(db_path) as store:
                # Alice can retrieve her migrated legacy document
                items = store.retrieve("doc-legacy", "alice")
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]["text"], "legacy oak chair")
                # Provenance explicitly marked unknown/legacy, never fabricated
                self.assertEqual(items[0]["source_revision"], "unknown")
                self.assertEqual(items[0]["source_location"], "legacy")

                # Bob cannot retrieve Alice's legacy document (owner preserved)
                with self.assertRaises(AuthorizationError):
                    store.retrieve("doc-legacy", "bob")

                # Citations work on migrated legacy chunks
                cite = items[0]["citation"]
                resolved = store.resolve_citation(cite, "alice")
                self.assertEqual(resolved["text"], "legacy oak chair")

                # Schema migration table contains current schema version (version 3)
                row = store._connection.execute(
                    "SELECT MAX(version) AS ver FROM schema_migrations"
                ).fetchone()
                self.assertEqual(row["ver"], 3)

            # 3. Idempotence: opening again does not fail or duplicate
            with EvidenceStore(db_path) as store_reopened:
                items2 = store_reopened.retrieve("doc-legacy", "alice")
                self.assertEqual(len(items2), 1)

    def test_public_access_revocation_and_historical_authorization(self):
        store = EvidenceStore(":memory:")
        # 1. Alice uploads doc1 with is_public=True (v1)
        store.upload("doc-revoc", "alice", ["v1 public text"], is_public=True)
        cite_v1 = store.retrieve("doc-revoc", "alice")[0]["citation"]

        # Bob can read v1 while public
        self.assertEqual(len(store.retrieve("doc-revoc", "bob")), 1)
        self.assertEqual(store.resolve_citation(cite_v1, "bob")["text"], "v1 public text")

        # 2. Alice updates doc1 with is_public=False (v2) - revoking public access
        store.upload("doc-revoc", "alice", ["v2 private text"], expected_version=1, is_public=False)

        # Bob can no longer retrieve active document
        with self.assertRaises(AuthorizationError):
            store.retrieve("doc-revoc", "bob")

        # Bob cannot resolve old v1 citation (historical metadata must not bypass current access revocation!)
        with self.assertRaises(AuthorizationError):
            store.resolve_citation(cite_v1, "bob")

        # Alice CAN resolve old v1 citation
        res_alice = store.resolve_citation(cite_v1, "alice")
        self.assertEqual(res_alice["text"], "v1 public text")

        # 3. Alice deletes doc-revoc
        self.assertTrue(store.delete("doc-revoc", "alice"))

        # After deletion, Bob cannot resolve old public citation
        with self.assertRaises(AuthorizationError):
            store.resolve_citation(cite_v1, "bob")

        # Owner Alice can still resolve historical citation of deleted document
        res_alice_del = store.resolve_citation(cite_v1, "alice")
        self.assertEqual(res_alice_del["text"], "v1 public text")

        # 4. Alice restores and makes it public again
        self.assertTrue(store.restore("doc-revoc", "alice"))
        store.upload("doc-revoc", "alice", ["v3 public text"], expected_version=2, is_public=True)

        # Bob can retrieve active and resolve v1 citation again
        self.assertEqual(len(store.retrieve("doc-revoc", "bob")), 1)
        self.assertEqual(store.resolve_citation(cite_v1, "bob")["text"], "v1 public text")

    def test_legacy_migration_retains_all_historical_versions(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "legacy_multi.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE chunks (
                    doc_id TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    PRIMARY KEY (doc_id, chunk_idx, version)
                )
                """
            )
            conn.execute("INSERT INTO documents VALUES ('doc-legacy-multi', 'alice', 2, '2026-01-01T00:00:00Z')")
            conn.execute("INSERT INTO chunks VALUES ('doc-legacy-multi', 0, 'chunk v1 text', 1)")
            conn.execute("INSERT INTO chunks VALUES ('doc-legacy-multi', 0, 'chunk v2 text', 2)")
            conn.commit()
            conn.close()

            # Migrate to v2
            with EvidenceStore(db_path) as store:
                # Active version 2 retrieved
                items = store.retrieve("doc-legacy-multi", "alice")
                self.assertEqual(len(items), 1)
                self.assertEqual(items[0]["text"], "chunk v2 text")

                # Resolving citation for retained historical version 1 succeeds for owner!
                v1_cite = "cite:doc-legacy-multi:v1:c0"
                resolved_v1 = store.resolve_citation(v1_cite, "alice")
                self.assertEqual(resolved_v1["text"], "chunk v1 text")
                self.assertEqual(resolved_v1["version"], 1)
                # Unrecorded legacy metadata is explicitly unknown / legacy
                self.assertEqual(resolved_v1["provenance"]["source_revision"], "unknown")
                self.assertEqual(resolved_v1["provenance"]["source_location"], "legacy")
                self.assertEqual(resolved_v1["provenance"]["committed_at"], "unknown")

                # Conservative authorization: Bob is denied access to unrecorded legacy history
                with self.assertRaises(AuthorizationError):
                    store.resolve_citation(v1_cite, "bob")

    def test_migration_rollback_and_unsupported_future_version(self):
        with tempfile.TemporaryDirectory() as td:
            # 1. Unsupported future version
            future_db = str(Path(td) / "future.db")
            conn = sqlite3.connect(future_db)
            conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            conn.execute("INSERT INTO schema_migrations VALUES (999, '2030-01-01T00:00:00Z')")
            conn.commit()
            conn.close()

            with self.assertRaises(StoreError) as caught:
                EvidenceStore(future_db)
            self.assertIn("Unsupported future database schema version", str(caught.exception))

    def test_typed_fact_preservation_and_alignment(self):
        store = EvidenceStore(":memory:")
        fact_bool = Fact(attribute="is_active", value=True)
        fact_int = Fact(attribute="item_count", value=42)
        fact_float = Fact(attribute="weight_kg", value=3.75)
        fact_str = Fact(attribute="material", value="solid oak")
        fact_null = Fact(attribute="notes", value=None)

        facts = [fact_bool, fact_int, fact_float, fact_str, fact_null]
        chunks = ["c1", "c2", "c3", "c4", "c5"]

        store.upload("doc-types", "alice", chunks, facts=facts, is_public=True)
        retrieved = store.retrieve("doc-types", "alice")

        # Verify exact types
        self.assertIs(retrieved[0]["value"], True)
        self.assertIsInstance(retrieved[0]["value"], bool)

        self.assertEqual(retrieved[1]["value"], 42)
        self.assertIsInstance(retrieved[1]["value"], int)
        self.assertNotIsInstance(retrieved[1]["value"], bool)

        self.assertEqual(retrieved[2]["value"], 3.75)
        self.assertIsInstance(retrieved[2]["value"], float)

        self.assertEqual(retrieved[3]["value"], "solid oak")
        self.assertIsInstance(retrieved[3]["value"], str)

        self.assertIsNone(retrieved[4]["value"])

        # Citation resolution round-trip preserves types
        for idx in range(len(facts)):
            cite = retrieved[idx]["citation"]
            resolved = store.resolve_citation(cite, "alice")
            self.assertEqual(resolved["fact"]["value"], retrieved[idx]["value"])
            self.assertEqual(type(resolved["fact"]["value"]), type(retrieved[idx]["value"]))

        # Non-finite floats rejected
        with self.assertRaises(ValueError):
            store.upload("doc-nan", "alice", ["text"], facts=[Fact("bad", float("nan"))])
        with self.assertRaises(ValueError):
            store.upload("doc-inf", "alice", ["text"], facts=[Fact("bad", float("inf"))])

        # Unsupported type rejected
        with self.assertRaises(TypeError):
            store.upload("doc-list", "alice", ["text"], facts=[Fact("bad", [1, 2, 3])])

        # Facts count vs chunks count alignment validation
        with self.assertRaises(ValueError) as caught:
            store.upload("doc-mismatch", "alice", ["chunk1"], facts=[Fact("f1", 1), Fact("f2", 2)])
        self.assertIn("Facts count", str(caught.exception))

    def test_citation_identity_with_colons_and_delimiters(self):
        store = EvidenceStore(":memory:")
        # Document ID containing ':'
        doc_id = "item:variant:chair_01"
        store.upload(doc_id, "alice", ["chunk with colon id"], is_public=True)

        retrieved = store.retrieve(doc_id, "alice")
        cite = retrieved[0]["citation"]

        # Encoded citation resolves successfully
        resolved = store.resolve_citation(cite, "alice")
        self.assertEqual(resolved["doc_id"], doc_id)
        self.assertEqual(resolved["text"], "chunk with colon id")

        # Malformed citations raise CitationError
        with self.assertRaises(CitationError):
            store.resolve_citation("cite::v1:c0", "alice")  # empty doc_id
        with self.assertRaises(CitationError):
            store.resolve_citation("cite:doc:v-1:c0", "alice")  # negative version
        with self.assertRaises(CitationError):
            store.resolve_citation("cite:doc:v1:c-5", "alice")  # negative chunk
        with self.assertRaises(CitationError):
            store.resolve_citation("cite:doc:v1:c0:NOT_HEX!", "alice")  # invalid hash

    def test_citation_payload_integrity_verification(self):
        store = EvidenceStore(":memory:")
        fact = Fact(attribute="width", value=48.0, unit="cm", source_location="spec_sheet", source_revision="rev_10")
        store.upload("doc-integrity", "alice", ["original text"], facts=[fact], is_public=True)

        cite = store.retrieve("doc-integrity", "alice")[0]["citation"]

        # Untouched citation resolves cleanly
        resolved = store.resolve_citation(cite, "alice")
        self.assertEqual(resolved["text"], "original text")
        self.assertEqual(resolved["fact"]["value"], 48.0)

        # 1. Tamper with structured fact attribute in database while leaving text intact
        store._connection.execute("UPDATE chunks SET attribute = 'tampered_width' WHERE doc_id = 'doc-integrity'")
        with self.assertRaises(CitationError) as caught:
            store.resolve_citation(cite, "alice")
        self.assertIn("stored integrity hash does not match canonical fact and provenance payload", str(caught.exception))

        # Revert attribute, tamper with value_num
        store._connection.execute("UPDATE chunks SET attribute = 'width', value_num = 99.0 WHERE doc_id = 'doc-integrity'")
        with self.assertRaises(CitationError) as caught:
            store.resolve_citation(cite, "alice")
        self.assertIn("stored integrity hash does not match canonical fact and provenance payload", str(caught.exception))

        # Revert value_num, tamper with provenance source_revision
        store._connection.execute("UPDATE chunks SET value_num = 48.0, source_revision = 'forged_rev' WHERE doc_id = 'doc-integrity'")
        with self.assertRaises(CitationError) as caught:
            store.resolve_citation(cite, "alice")
        self.assertIn("stored integrity hash does not match canonical fact and provenance payload", str(caught.exception))

        # Revert provenance, tamper with text bytes
        store._connection.execute("UPDATE chunks SET source_revision = 'rev_10', text = 'tampered text' WHERE doc_id = 'doc-integrity'")
        with self.assertRaises(CitationError) as caught:
            store.resolve_citation(cite, "alice")
        self.assertTrue(
            "stored integrity hash does not match" in str(caught.exception)
            or "text bytes do not match source hash" in str(caught.exception)
        )

    def test_catalog_provenance_actual_revisions_and_units(self):
        verifier = Verifier(auto_load=False)
        catalog_fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "catalog.jsonl"
        verifier.load_catalog(catalog_fixture)

        # Retrieve oak chair chunks
        chunks = verifier.store.retrieve("fixture-chair-oak-01", Principal.system())
        self.assertTrue(len(chunks) > 0)

        width_chunk = next(c for c in chunks if c.get("attribute") == "width")
        # source_revision must be the actual value from catalog.jsonl ("fixture-v1"), NOT "catalog:fixture"
        self.assertEqual(width_chunk["source_revision"], "fixture-v1")
        self.assertEqual(width_chunk["value"], 48.0)
        self.assertEqual(width_chunk["unit"], "cm")

        # Test verification uses exact conversion without premature rounding
        status, detail = verifier.verify_attribute("fixture-chair-oak-01", "width", 50.0)
        self.assertEqual(status, "SUPPORTED")
        self.assertEqual(detail, "48cm")

        status, detail = verifier.verify_attribute("fixture-chair-oak-01", "width", 40.0)
        self.assertEqual(status, "CONTRADICTED")
        self.assertEqual(detail, "48cm")

    def test_connection_ownership_and_thread_serialization(self):
        store = EvidenceStore(":memory:")
        store.upload("doc-thread", "alice", ["initial"])

        # Multiple threads invoking operations on the same connection are serialized via self._lock
        def worker(idx: int):
            return store.retrieve("doc-thread", "alice")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [f.result() for f in futures]

        for res in results:
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["text"], "initial")

    def test_v2_to_v3_migration_and_integrity_enforcement(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "v2_upgrade.db")

            # 1. Create a database matching initial stage-03 v2 layout
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            conn.execute("INSERT INTO schema_migrations VALUES (2, '2026-10-29T20:00:00Z')")
            conn.execute(
                """
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    is_public INTEGER NOT NULL DEFAULT 1,
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE document_history (
                    doc_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    owner_id TEXT NOT NULL,
                    is_public INTEGER NOT NULL DEFAULT 1,
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    committed_at TEXT NOT NULL,
                    PRIMARY KEY (doc_id, version)
                )
                """
            )
            # v2 chunks table lacks value_type, value_int, integrity_hash
            conn.execute(
                """
                CREATE TABLE chunks (
                    doc_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    item_id TEXT,
                    variant_id TEXT,
                    attribute TEXT,
                    value_text TEXT,
                    value_num REAL,
                    unit TEXT,
                    status TEXT NOT NULL DEFAULT 'present',
                    source_location TEXT,
                    source_revision TEXT NOT NULL DEFAULT 'unknown',
                    source_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (doc_id, version, chunk_idx)
                )
                """
            )
            # Active document is version 2
            conn.execute("INSERT INTO documents VALUES ('chair-01', 'alice', 2, 1, 0, '2026-10-29T20:00:00Z', '2026-10-29T20:00:00Z')")
            # Only version 2 in document_history (v2 defect: version 1 omitted)
            conn.execute("INSERT INTO document_history VALUES ('chair-01', 2, 'alice', 1, 0, '2026-10-29T20:00:00Z')")

            t1 = "width: 10 cm (source: spec)"
            h1 = hashlib.sha256(t1.encode("utf-8")).hexdigest()
            # Version 1 chunk
            conn.execute(
                """
                INSERT INTO chunks VALUES (
                    'chair-01', 1, 0, 'chair-01:v1:c0', ?, 'item-1', 'item-1',
                    'width', '10', 10.0, 'cm', 'present', 'spec_v1', 'rev-1', ?, '2026-10-29T20:00:00Z'
                )
                """,
                (t1, h1),
            )

            t2 = "width: 15 cm (source: spec)"
            h2 = hashlib.sha256(t2.encode("utf-8")).hexdigest()
            # Version 2 chunk
            conn.execute(
                """
                INSERT INTO chunks VALUES (
                    'chair-01', 2, 0, 'chair-01:v2:c0', ?, 'item-1', 'item-1',
                    'width', '15', 15.0, 'cm', 'present', 'spec_v2', 'rev-2', ?, '2026-10-29T20:00:00Z'
                )
                """,
                (t2, h2),
            )

            # Also add a chunk with legacy string "True" to verify it is NOT inferred as boolean
            t3 = "enabled: True"
            h3 = hashlib.sha256(t3.encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT INTO chunks VALUES (
                    'chair-01', 2, 1, 'chair-01:v2:c1', ?, 'item-1', 'item-1',
                    'enabled', 'True', NULL, NULL, 'present', 'spec_v2', 'rev-2', ?, '2026-10-29T20:00:00Z'
                )
                """,
                (t3, h3),
            )
            conn.commit()
            conn.close()

            # 2. Open using EvidenceStore -> triggers v2 to v3 migration
            with EvidenceStore(db_path) as s1:
                # Check migration version recorded
                ver_row = s1._connection.execute("SELECT MAX(version) AS ver FROM schema_migrations").fetchone()
                self.assertEqual(ver_row["ver"], 3)

                # Check historical-version completeness: version 1 now in document_history
                hist_v1 = s1._connection.execute("SELECT * FROM document_history WHERE doc_id='chair-01' AND version=1").fetchone()
                self.assertIsNotNone(hist_v1)
                self.assertEqual(hist_v1["committed_at"], "unknown")

                # Verify all chunks have non-null integrity_hash and value_type
                missing = s1._connection.execute("SELECT COUNT(*) AS cnt FROM chunks WHERE integrity_hash IS NULL").fetchone()["cnt"]
                self.assertEqual(missing, 0)

                # Check that legacy string "True" remained str, never inferred as boolean
                chunk_bool_str = s1._connection.execute("SELECT value_type, value_text FROM chunks WHERE chunk_idx=1").fetchone()
                self.assertEqual(chunk_bool_str["value_type"], "str")
                self.assertEqual(chunk_bool_str["value_text"], "True")

                # Resolving citation for v1 works
                cite_v1_legacy = f"cite:chair-01:v1:c0:{h1[:8]}"
                res_v1 = s1.resolve_citation(cite_v1_legacy, "alice")
                self.assertEqual(res_v1["fact"]["value"], 10.0)

            # 3. Close and reopen: ensure no silent rebaselining or rehashing occurs
            with EvidenceStore(db_path) as s2:
                # Tamper with stored numeric value (10 -> 999) without touching hash
                s2._connection.execute("UPDATE chunks SET value_num = 999.0 WHERE doc_id='chair-01' AND version=1")

                # Re-opening did not recalculate replacement hashes; tampering is caught!
                with self.assertRaises(CitationError) as caught:
                    s2.resolve_citation(cite_v1_legacy, "alice")
                self.assertIn("stored integrity hash does not match", str(caught.exception))

                # Revert value, tamper with unit
                s2._connection.execute("UPDATE chunks SET value_num = 10.0, unit = 'inches' WHERE doc_id='chair-01' AND version=1")
                with self.assertRaises(CitationError) as caught:
                    s2.resolve_citation(cite_v1_legacy, "alice")
                self.assertIn("stored integrity hash does not match", str(caught.exception))

                # Revert unit, tamper with provenance
                s2._connection.execute("UPDATE chunks SET unit = 'cm', source_revision = 'forged-rev' WHERE doc_id='chair-01' AND version=1")
                with self.assertRaises(CitationError) as caught:
                    s2.resolve_citation(cite_v1_legacy, "alice")
                self.assertIn("stored integrity hash does not match", str(caught.exception))

                # Revert provenance, now resolves cleanly again
                s2._connection.execute("UPDATE chunks SET source_revision = 'rev-1' WHERE doc_id='chair-01' AND version=1")
                res_fixed = s2.resolve_citation(cite_v1_legacy, "alice")
                self.assertEqual(res_fixed["fact"]["value"], 10.0)

    def test_migration_atomic_rollback_on_injected_failure(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "rollback.db")
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            conn.execute("INSERT INTO schema_migrations VALUES (2, '2026-10-29T20:00:00Z')")
            conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, owner_id TEXT, version INT, is_public INT, is_deleted INT, created_at TEXT, updated_at TEXT)")
            # Create a chunks table that has a constraint or schema that will fail when v3 columns are added or updated
            conn.execute("CREATE TABLE chunks (doc_id TEXT NOT NULL, version INT, chunk_idx INT, text TEXT, PRIMARY KEY (doc_id, version, chunk_idx))")
            conn.execute("INSERT INTO documents VALUES ('d1', 'alice', 1, 0, 0, '2026-01-01', '2026-01-01')")
            conn.execute("INSERT INTO chunks VALUES ('d1', 1, 0, 'sample text')")
            conn.commit()
            conn.close()

            # Temporarily simulate a failure during _migrate_v2_to_v3
            orig_migrate = EvidenceStore._migrate_v2_to_v3
            def faulty_migrate(self):
                # Add columns then fail midway
                self._connection.execute("ALTER TABLE chunks ADD COLUMN value_type TEXT")
                raise RuntimeError("simulated disk or migration failure")

            EvidenceStore._migrate_v2_to_v3 = faulty_migrate
            try:
                with self.assertRaises(RuntimeError):
                    EvidenceStore(db_path)

                # Verify transaction rolled back: schema_migrations is still at version 2
                conn = sqlite3.connect(db_path)
                row = conn.execute("SELECT MAX(version) AS ver FROM schema_migrations").fetchone()
                self.assertEqual(row[0], 2)
                # value_type column should not exist due to rollback
                cols = [c[1] for c in conn.execute("PRAGMA table_info(chunks)").fetchall()]
                self.assertNotIn("value_type", cols)
                conn.close()
            finally:
                EvidenceStore._migrate_v2_to_v3 = orig_migrate


if __name__ == "__main__":
    unittest.main()
