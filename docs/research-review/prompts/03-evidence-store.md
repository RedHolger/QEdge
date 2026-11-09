# Enforce evidence ownership and immutable revisions

Prompt ID: `03-evidence-store`  
Existing plan: Bring original Stage 07 storage work forward  
Prerequisites: 02-contracts

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Inspect edge-rag-demo/src/edge_rag/store.py and verifier loading. Choose and document the supported boundary: local single-user CLI initially, with an interface capable of explicit authenticated principals later. Never treat an arbitrary caller-supplied string `system` as an administrator credential or claim authentication where none exists.

Represent facts as typed values with units, status, item/variant, source location, source revision/hash, document version, and chunk ID. Enforce owner authorization on read, update, delete, and restore. Prevent ownership takeover of deleted records. Handle public facts through an explicit read policy. Add a schema migration for existing databases and preserve historical versions needed by citations.

Make expected-version comparison and write one transaction across independent SQLite connections/processes, using appropriate transaction modes/conditional writes and explicit conflict responses. Define connection/thread ownership. Keep citations immutable and able to resolve the exact fact version used; updating the current source must not rewrite historical decision evidence. Integrity hashes do not themselves establish authenticity.

Acceptance: cross-owner read/write denial; forged privileged-name denial at the chosen interface; authorized/public access; deleted-object takeover denial; two separate connections racing with the same expected version yield one success and one explicit conflict; atomic rollback; deletion/reload behavior; and old citations resolve after a new revision. Use temporary databases. Do not build an unrelated remote authentication service.
