# Implement explicit requirements and conservative decisions

Prompt ID: `04-assistant-correctness`  
Existing plan: Original Stage 07 core repair before retrieval expansion  
Prerequisites: 02-contracts and 03-evidence-store

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Read cli.py, verifier.py, generator.py, current fixtures, and the migrated contracts. Remove all hidden chair/oak/width defaults. Build a typed requirement parser that preserves source spans and unsupported/ambiguous clauses. Support width/height/length, color/material/brand, explicit units, strict and inclusive comparisons, ranges, and negation. A bare number or unspecified dimension must not acquire a guessed unit or attribute. Handle conflicting user constraints explicitly.

Keep original values and units; normalize numerically without rounding before comparison. Distinguish product from package dimensions and product families from variants. Implement an explicit asymmetric taxonomy: oak can imply wood, but unspecified wood cannot prove oak; similarly, unknown composition cannot prove not-oak. Do not match substrings such as red inside unrelated words. Conflicting source facts must not resolve by taking the first row.

Emit validated claims and a deterministic decision using complete requirement coverage. Preserve unsupported requests such as built-in quantum charger even if retrieval finds a chair. Separate find (candidate ranking) from check (evidence verdict). Render citations to exact fact revisions. Optional generation must not decide the verdict. Remove execution of a synthetic QNN benchmark from normal answer composition.

Acceptance: table-driven adversarial tests for red chair, under 60 without units, not oak, oak versus generic wood, 24 inches versus 60.96cm, strict boundary equality, multiple dimensions, unknown units, contradictory sources, unsupported features, no constraints, no candidates, and revisions changing a verdict. Include a catalog-directory path CLI test: the current verifier accepts a file while parts of the public interface promise a directory. Start the existing 80-case program (40 development/40 held-out) with provenance and independent review; do not claim human labels before they exist. Zero positive decisions on the known adverse cases is required, but not a universal zero-error claim.
