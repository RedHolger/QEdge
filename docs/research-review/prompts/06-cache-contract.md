# Implement a position-correct bounded cache

Prompt ID: `06-cache-contract`  
Existing plan: Original Stage 03 infrastructure  
Prerequisites: 05-real-model-baseline

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Specify a cache contract for the pinned model attention path before implementing policies. Track absolute token positions separately from resident indices and total tokens seen. Distinguish permanent eviction from an attention-time selected view. Define the resident-byte budget, temporary-memory allowance, per-layer/head layout, and batch policy. Start with batch size one and reject larger batches if independent selection is not implemented.

Implement full, FIFO, and a correctly defined recency baseline. State what access means in dense attention: touching every resident token makes classical LRU degenerate. Do not rename attention-frequency scoring LRU. Keep keys, values, positions, masks, and score metadata aligned through every gather. Apply rotary transforms consistently to Q and K before comparing them. Avoid private-version assumptions about DynamicCache; use a tested adapter and documented compatibility boundary.

Test multi-token and chunked prefill before enabling arbitrary interior eviction. Causal permission must follow original positions, including non-contiguous keys. Keep the current token where the design requires it. Decide whether a prefill chunk may temporarily exceed the budget and report that peak explicitly. Do not derive peak memory solely from final resident tensors.

Acceptance: unchanged/no-eviction path matches baseline tokens and logits; hand-calculated attention examples verify interior eviction and future-token masking; capacity 1, tiny budgets, duplicate positions, zero norms, padding, GQA and repeated requests are handled; memory accounting includes K/V, metadata, and transient gathers. Resident positions stay ordered/unique and eviction counters reconcile. No speedup or quality ranking is required for this correctness gate.
