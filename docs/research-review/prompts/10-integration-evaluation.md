# Integrate a conservative offline assistant

Prompt ID: `10-integration-evaluation`  
Existing plan: Original Stages 08/09  
Prerequisites: 04-assistant-correctness and 09-catalog-retrieval; 05 for optional generation

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Create a single local demo with explicit find/check modes and a fully working extractive path without model downloads. Preserve the runtime as an optional dependency. Pass only authorized, versioned evidence to generation, bound context size, separate data from instructions, and never allow catalog text to override system policy. A model may phrase an existing decision; it cannot change status, create new facts, or produce unresolved citations.

Implement request IDs, structured errors, cancellation/timeouts where supported, and stage-level timing. Verify citations before rendering. If generated output violates the evidence/decision contract, use the deterministic answer and record the reason. Do not claim arbitrary free text can be proven factual by a JSON schema or a second LLM judge.

Exercise the ten scenario categories in the existing integration plan with actual artifacts: supported, contradicted, missing fact, ambiguous units, negation, unsupported requirement, source conflict/update, no result, optional image query, and generation unavailable/failure. Explicitly skip image execution if rights/assets are absent; do not invent it.

Acceptance: offline after explicit bootstrap; isolation between requests/users under the declared boundary; no recommendation when a mandatory predicate is unresolved; all citations resolve; source update changes results; held-out decision confusion matrix, false-SUPPORTED denominator and confidence interval, abstention/coverage curve, retrieval metrics, end-to-end timing and failure taxonomy. Keep 40 held-out decision cases untouched during tuning; changing them invalidates the held-out claim. A complete local demo may ship with NPU unverified, but human-evaluation and hardware gates remain separately labelled.
