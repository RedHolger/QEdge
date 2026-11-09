"""Diagnostic script to reproduce the 6 spot-check defects in Stage 03 baseline."""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Add edge-rag-demo/src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "edge-rag-demo" / "src"))

from edge_rag.store import (
    EvidenceStore,
    Fact,
    Principal,
    AuthorizationError,
    VersionConflictError,
    CitationError,
    NotFoundError,
)
from edge_rag.verifier import Verifier


def test_defect_1_access_revocation():
    print("--- Probing Defect 1: Access Revocation and Historical Authorization ---")
    store = EvidenceStore(":memory:")
    # Alice uploads public doc
    store.upload("doc1", "alice", ["chunk1"], is_public=True)
    
    # Alice updates doc1 with is_public=False
    store.upload("doc1", "alice", ["chunk2"], expected_version=1, is_public=False)
    
    # Check if Bob can retrieve active chunks
    try:
        res = store.retrieve("doc1", "bob")
        if res:
            print("DEFECT REPRODUCED: Bob can retrieve doc1 even after Alice updated with is_public=False!")
        else:
            print("Access correctly denied for Bob on retrieve")
    except AuthorizationError:
        print("Access correctly denied with AuthorizationError on retrieve")

    # Alice deletes doc1
    store.delete("doc1", "alice")
    # Bob tries to resolve old citation cite:doc1:v1:c0:...
    # Fetch chunk hash for v1
    v1_row = store._connection.execute("SELECT source_hash FROM chunks WHERE doc_id='doc1' AND version=1").fetchone()
    citation_v1 = f"cite:doc1:v1:c0:{v1_row['source_hash'][:8]}"
    try:
        res = store.resolve_citation(citation_v1, "bob")
        print("DEFECT REPRODUCED: Bob can resolve old public citation of deleted document!")
    except (AuthorizationError, NotFoundError) as e:
        print(f"Citation correctly denied for Bob: {type(e).__name__}")


def test_defect_2_legacy_historical_versions():
    print("--- Probing Defect 2: Legacy Historical Versions ---")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Create legacy v1 db with doc version=2 and chunks for v1 and v2
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, owner_id TEXT, version INTEGER, created_at TEXT)")
        conn.execute("CREATE TABLE chunks (doc_id TEXT, chunk_idx INTEGER, text TEXT, version INTEGER)")
        conn.execute("INSERT INTO documents VALUES ('doc_legacy', 'alice', 2, '2026-01-01T00:00:00Z')")
        conn.execute("INSERT INTO chunks VALUES ('doc_legacy', 0, 'chunk v1 text', 1)")
        conn.execute("INSERT INTO chunks VALUES ('doc_legacy', 0, 'chunk v2 text', 2)")
        conn.commit()
        conn.close()

        # Open with EvidenceStore to trigger migration
        store = EvidenceStore(db_path)
        # Try resolving citation for v1
        # In current code:
        try:
            store.resolve_citation("cite:doc_legacy:v1:c0", "alice")
            print("Legacy v1 citation resolved successfully")
        except NotFoundError as e:
            print(f"DEFECT REPRODUCED: Legacy v1 citation failed with NotFoundError: {e}")
        store.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_defect_3_typed_facts():
    print("--- Probing Defect 3: Typed Fact Preservation ---")
    store = EvidenceStore(":memory:")
    fact_bool = Fact(attribute="enabled", value=True)
    fact_int = Fact(attribute="count", value=42)
    store.upload("doc_types", "alice", ["chunk A", "chunk B"], facts=[fact_bool, fact_int])
    
    chunks = store.retrieve("doc_types", "alice")
    val_bool = chunks[0]["value"]
    val_int = chunks[1]["value"]
    print(f"Retrieved bool fact: {val_bool!r} (type={type(val_bool).__name__})")
    if val_bool == "True" and isinstance(val_bool, str):
        print("DEFECT REPRODUCED: Boolean fact was retrieved as string 'True'!")
    print(f"Retrieved int fact: {val_int!r} (type={type(val_int).__name__})")
    if isinstance(val_int, float):
        print("DEFECT REPRODUCED: Integer fact was retrieved as float 42.0 instead of int!")


def test_defect_4_citation_with_colon():
    print("--- Probing Defect 4: Citation Identity (Colon in Document ID) & Integrity ---")
    store = EvidenceStore(":memory:")
    store.upload("item:variant", "alice", ["sample text"])
    chunks = store.retrieve("item:variant", "alice")
    cit = chunks[0]["citation"]
    print(f"Generated citation: {cit}")
    try:
        resolved = store.resolve_citation(cit, "alice")
        print(f"Resolved doc_id: {resolved['doc_id']}")
    except CitationError as e:
        print(f"DEFECT REPRODUCED: Colon in doc_id caused CitationError: {e}")

    # Integrity probe: alter attribute in db while leaving text intact
    store.upload("doc_integrity", "alice", ["unaltered text"], facts=[Fact("attr", "val1")])
    c_list = store.retrieve("doc_integrity", "alice")
    cit_integ = c_list[0]["citation"]
    # Tamper with attribute directly in db
    store._connection.execute("UPDATE chunks SET value_text = 'tampered' WHERE doc_id='doc_integrity'")
    try:
        res = store.resolve_citation(cit_integ, "alice")
        print("DEFECT REPRODUCED: Citation resolved despite tampered structured fact (integrity hash only checked text)!")
    except CitationError:
        print("Citation failed integrity check when structured fact was tampered")


def test_defect_5_catalog_provenance():
    print("--- Probing Defect 5: Catalog Provenance ---")
    verifier = Verifier(auto_load=False)
    catalog_fixture = Path(__file__).resolve().parents[4] / "edge-rag-demo" / "tests" / "fixtures" / "catalog.jsonl"
    verifier.load_catalog(catalog_fixture)
    chunks = verifier.store.retrieve("fixture-chair-oak-01", "system")
    first_chunk = chunks[0]
    print(f"Loaded chunk source_revision: {first_chunk.get('source_revision')!r}")
    if first_chunk.get("source_revision") == "catalog:fixture":
        print("DEFECT REPRODUCED: source_revision was hardcoded to 'catalog:fixture' instead of catalog row value!")


if __name__ == "__main__":
    test_defect_1_access_revocation()
    test_defect_2_legacy_historical_versions()
    test_defect_3_typed_facts()
    test_defect_4_citation_with_colon()
    test_defect_5_catalog_provenance()
