"""Probe to reproduce the v2 migration integrity gap."""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "edge-rag-demo" / "src"))

from edge_rag.store import EvidenceStore, CitationError


def reproduce_v2_migration_integrity_gap():
    print("--- Probing V2 Migration Integrity Gap ---")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Step 1: Create initial Stage 03 v2 database
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
        # chunks lacks value_type, value_int, integrity_hash
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
        conn.execute(
            "INSERT INTO documents VALUES ('doc1', 'alice', 1, 1, 0, '2026-10-29T20:00:00Z', '2026-10-29T20:00:00Z')"
        )
        conn.execute(
            "INSERT INTO document_history VALUES ('doc1', 1, 'alice', 1, 0, '2026-10-29T20:00:00Z')"
        )
        import hashlib
        text = "width: 10 cm (source: spec)"
        s_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            """
            INSERT INTO chunks (
                doc_id, version, chunk_idx, chunk_id, text, item_id, variant_id,
                attribute, value_text, value_num, unit, status, source_location,
                source_revision, source_hash, created_at
            )
            VALUES ('doc1', 1, 0, 'doc1:v1:c0', ?, 'item1', 'item1', 'width', '10', 10.0, 'cm', 'present', 'spec', 'rev1', ?, '2026-10-29T20:00:00Z')
            """,
            (text, s_hash),
        )
        conn.commit()
        conn.close()

        # Step 3: Open using EvidenceStore
        store = EvidenceStore(db_path)

        # Step 4: Check if integrity_hash is NULL
        row = store._connection.execute("SELECT integrity_hash, value_type FROM chunks WHERE doc_id='doc1'").fetchone()
        print(f"integrity_hash in chunks after open: {row['integrity_hash']!r}")
        print(f"value_type in chunks after open: {row['value_type']!r}")

        # Step 5: Change value_num to 999
        store._connection.execute("UPDATE chunks SET value_num = 999.0 WHERE doc_id='doc1'")

        # Step 6: resolve_citation
        citation = f"cite:doc1:v1:c0:{s_hash[:8]}"
        try:
            res = store.resolve_citation(citation, "alice")
            print(f"DEFECT REPRODUCED: Citation resolved with tampered value: {res['fact']['value']} (expected CitationError)!")
        except CitationError as e:
            print(f"Citation correctly failed: {e}")

        store.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    reproduce_v2_migration_integrity_gap()
