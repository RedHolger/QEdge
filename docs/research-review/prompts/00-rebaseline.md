# Confirm the live workspace and freeze evidence

Prompt ID: `00-rebaseline`  
Existing plan: Existing Stage 00/01 audit refresh  
Prerequisites: None

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Inspect the current files, applicable AGENTS.md instructions, nested Git roots, dirty changes, and active handoffs. Compare docs/research-review/ANALYSIS.md with live source; OpenCode may have advanced since the review. Preserve concurrent changes. The workspace root is not one Git repository, so do not run a root reset or assume one commit covers everything.

Reproduce the Python tests and the isolated sanitizer build described in evidence/COMMANDS.md. Run evidence/reproduce_gaps.py and record the results. Inspect the source behind each finding rather than treating probe output as the whole specification. Record compiler, Python, platform, dependency versions, source hashes, and exit codes. Classify each G01–G12 finding as still present, fixed with evidence, or not reproduced.

Create a current claims ledger distinguishing implemented, tested, measured, calculated, synthetic, proposed, and blocked. Mark every performance table in the research without matching raw artifacts as unverified. Inventory cached models/catalogs and verify availability without downloading anything. Identify the exact scoped files for the next repair.

Acceptance: a reproducible baseline manifest, explicit failed and skipped tests, unchanged historical hash evidence, and a gap-to-prompt map. Do not fix application code in this stage. Finish with the next runnable repair, not an assertion that all old stages are accepted.
