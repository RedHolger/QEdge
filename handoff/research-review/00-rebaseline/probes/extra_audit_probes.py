"""Additional read-only audit probes for 00-rebaseline.

This script uses only the synthetic fixture catalog and in-memory SQLite.  It
does not modify application source or persistent data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "edge-rag-demo/src"), str(ROOT / "qedge-common/src")]

from edge_rag.store import EvidenceStore
from edge_rag.verifier import Verifier

fixture = ROOT / "edge-rag-demo/tests/fixtures/catalog.jsonl"
verifier = Verifier(catalog_path=fixture)

queries = [
    "not oak chair",
    "chair under 60",
    "chair under 24 inches",
    "red chair",
]
results = {
    "parsed_requirements": {
        query: verifier.parse_requirements(query) for query in queries
    },
    "verification_against_oak_fixture": {},
    "retrieved_chunk_shape": None,
    "generic_wood_material_verification": None,
}

for query in ("not oak chair", "chair under 60"):
    results["verification_against_oak_fixture"][query] = {
        "status": verifier.verify("fixture-chair-oak", verifier.parse_requirements(query)),
        "formatted": verifier.format_verification(
            verifier.verify("fixture-chair-oak", verifier.parse_requirements(query))
        ),
    }

results["retrieved_chunk_shape"] = verifier.store.retrieve(
    "fixture-chair-oak-01", "system"
)

wood_store = EvidenceStore()
wood_store.upload("doc-generic-wood", "alice", ["material: wood (source: material)"])
wood_verifier = Verifier(store=wood_store, auto_load=False)
results["generic_wood_material_verification"] = wood_verifier.verify_attribute(
    "doc-generic-wood", "material", "oak"
)

print(json.dumps(results, indent=2, sort_keys=True))
