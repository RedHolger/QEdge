# Release only what the evidence supports

Prompt ID: `12-release-review`  
Existing plan: Original Stage 10  
Prerequisites: Completed local acceptance gates; hardware claims require 11

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Audit the completed prompts and existing stage handoffs. Resolve stale claims and verify that README instructions, CLI labels, benchmark JSON, reports and the demo agree. Produce a capability matrix with local CPU, optional accelerator, and device-verified paths; every unavailable capability has a clear reason and activation command.

Separate explicit dependency/model/data acquisition from offline test/demo/eval commands. Test clean package installation and a fresh checkout/environment without inherited absolute paths or undeclared sibling imports. Require release and sanitizer checks appropriate to supported platforms. Exercise malformed requests, resource limits, missing assets, outdated schemas, migrations, stale indexes, and corrupted/missing evidence artifacts.

Create a five-minute reproducible demo, a technical walkthrough, and claim-to-artifact links. Performance statements must specify model revision, hardware, precision, context/generation length, trials and date. State finite evaluation denominators; do not claim universal hallucination elimination or NPU metrics from Mac runs. Preserve Stage 00/01 history and label superseded results.

Acceptance: one command each for installed demo, benchmark and evaluation; clean-environment logs; successful regression/sanitizer results; license/provenance inventory; known limitations; rollback/migration notes; reviewer verdicts for all required gates. Describe this as a validated portfolio release unless actual production operating requirements and device evidence have been met. Prepare release material; do not publish, push, or tag on the user's behalf without the applicable authorization.
