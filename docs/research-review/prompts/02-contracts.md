# Make contracts enforce decision and measurement meaning

Prompt ID: `02-contracts`  
Existing plan: Shared-contract prerequisite for original Stages 02–09  
Prerequisites: 00-rebaseline

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Inspect qedge-common schemas, schema.py, fixtures, tests, and every consumer. Design a versioned migration while retaining validation of historical v1 artifacts. First document predicates and response types: strict/inclusive comparisons, equality, inequality/negation, contains, ambiguity, unparsed clauses, candidate-specific decisions, and a no-candidate response. Do not encode `not oak` as operator == oak or invent an item ID for no result.

Implement shared semantic validation in addition to structural JSON Schema. A positive decision requires nonempty requirements, coverage of every required predicate, no unresolved mandatory clauses, all corresponding claims SUPPORTED, consistent item identities, and resolvable evidence. Contradiction blocks recommendation; missing/conflicting evidence abstains under a documented precedence policy. Align nested and standalone claim validation, including observed-value requirements. Enforce date-time formats and reject nonfinite numbers on JSON boundaries.

Define measurement invariants: synthetic records cannot masquerade as measured model runs; measured NPU status requires an execution evidence reference and matching backend/device identity; hardware flags cannot contradict each other. Separate outputs/token traces from benchmark records or introduce an explicit versioned envelope. Do not require invented cache hit rates for algorithms where that metric is meaningless.

Acceptance: negative tests reject empty SUPPORTED, contradicted children with a positive parent, missing required constraints, mismatched item IDs, invalid timestamps, null supported observations, and measured/unverified hardware contradictions. Positive fixtures and historical v1 readers remain deliberate and documented. New producer/consumer boundaries actually call validators. Schema success alone must not be presented as factual truth.
