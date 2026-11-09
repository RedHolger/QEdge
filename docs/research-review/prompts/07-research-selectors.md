# Evaluate faithful selectors and labelled hypotheses

Prompt ID: `07-research-selectors`  
Existing plan: Original Stage 03 research  
Prerequisites: 06-cache-contract

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Read version-pinned KeyDiff and QUOKA primary papers and any official reference implementation. Write an algorithm card specifying exact equations, where selection occurs, head aggregation, chunk size, candidate set, budget meaning, tie-breaking, sink/recency protections, and deviations. Keep paper-faithful variants separate from centroid-neighbor, attention-decay, or value-norm hypotheses.

Implement KeyDiff against an independent small reference for the chosen paper variant. Implement QUOKA as chunked-prefill representative-query/key selection, not merely decayed attention-score eviction. Do not silently discard the permanent cache when the paper only reduces pairs per attention evaluation. Use correctly transformed queries and account for selection/gather cost. A document-inspired attention/value heuristic can be named attention_value_decay and evaluated separately; it is not automatically CAOTE or H2O either.

Use matched model, precision, prompt sets, seeds, hardware, and well-defined budgets. Do not compare a KV-pair-per-evaluation percentage directly to a resident-memory percentage. Evaluate quality and prefill/decode costs independently. Include a selector ablation where Q changes with K/V held fixed, and a synthetic geometry check where mean-anchor and newest-key selectors differ.

Acceptance: paper selector agreement on small deterministic cases; full-budget parity; nontrivial pressure at reduced budgets; quality/latency/memory curves with raw trials, confidence intervals, and failures. Do not require an optimized policy to win or policies to differ on every workload. A negative result is acceptable. Report evidence supporting or rejecting each hypothesis without importing the research document's performance numbers.
