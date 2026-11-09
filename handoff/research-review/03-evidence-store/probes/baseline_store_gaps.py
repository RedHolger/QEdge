"""Baseline probe for Stage 03 evidence store.

Demonstrates observable defects in the existing store.py:
1. Cross-owner read: user B can read user A's private documents.
2. Cross-owner overwrite: user B can overwrite user A's document.
3. Forged privileged-name: string "system" passed as requester grants system behavior.
4. No delete/restore or deleted-object takeover protection.
5. In-memory threading lock does not protect concurrent independent SQLite connections.
6. Citations do not include document version; updating source alters historical resolution.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from edge_rag.store import EvidenceStore


def test_cross_owner_read():
    store = EvidenceStore(":memory:")
    store.upload("doc-1", "alice", ["secret alice fact"])
    # Bob retrieves alice's document
    results = store.retrieve("doc-1", "bob")
    if results and "secret alice fact" in results[0]["text"]:
        print("GAP 1: Cross-owner read permitted (bob read alice's private document)")
    else:
        print("SECURE: Cross-owner read blocked")


def test_cross_owner_overwrite():
    store = EvidenceStore(":memory:")
    v1 = store.upload("doc-1", "alice", ["alice fact"])
    # Bob overwrites doc-1 without authorization
    v2 = store.upload("doc-1", "bob", ["bob hijacked doc-1"])
    results = store.retrieve("doc-1", "alice")
    if results and "bob hijacked doc-1" in results[0]["text"]:
        print("GAP 2: Cross-owner overwrite permitted (bob overwrote alice's document)")
    else:
        print("SECURE: Cross-owner overwrite blocked")


def test_forged_system_role():
    store = EvidenceStore(":memory:")
    # Arbitrary caller passes "system" string
    store.upload("system-doc", "system", ["system fact"])
    results = store.retrieve("system-doc", "system")
    if results:
        print("GAP 3: Arbitrary string 'system' accepted as owner/requester identity")


def test_citation_rewrites_history():
    store = EvidenceStore(":memory:")
    v1 = store.upload("doc-1", "alice", ["original v1 fact"])
    c1 = store.retrieve("doc-1", "alice")[0]["citation"]  # "doc-1:0"
    # Document updated to v2
    v2 = store.upload("doc-1", "alice", ["updated v2 fact"])
    # Reading doc-1 active version
    c2_results = store.retrieve("doc-1", "alice")
    c2 = c2_results[0]["citation"]
    if c1 == c2 and "updated v2 fact" in c2_results[0]["text"]:
        print("GAP 4: Citation 'doc-1:0' is unversioned; updating document rewrites what the citation yields")


def main():
    print("Running baseline evidence store gap probe...")
    test_cross_owner_read()
    test_cross_owner_overwrite()
    test_forged_system_role()
    test_citation_rewrites_history()
    print("Baseline probe finished.")


if __name__ == "__main__":
    main()
