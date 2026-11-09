# Stage 00 — Audit and Baseline

## Goal

Establish the truth about the current repositories, hardware, dependencies, test state, and headline claims. Do not change source code.

## Read first

- `00-START-HERE.md`
- `01-STRATEGY.md`
- `04-ACCEPTANCE-AND-DELIVERABLES.md` (stage 00)

## Tasks

1. Inventory both repositories:
   - file tree excluding `build/`, `.venv`, `__pycache__`, and model caches;
   - LOC by language;
   - build systems and test commands;
   - declared and actual dependencies;
   - external repos, absolute paths, and symlinks.

2. Record the environment:
   - OS and version;
   - CPU/GPU/NPU model;
   - RAM and free disk;
   - Python, `uv`, compiler, CMake, Xcode tools;
   - model caches and their sizes;
   - Snapdragon/QNN/AI Hub availability.

3. Run the current test and demo commands exactly as documented:
   - `llm-inference`: build, `run_tests`, `run_qnn_tests`, `llm_infer`, `llm_qnn_bench`;
   - `edge-rag-demo`: `uv run python -m unittest discover -s tests -v`, demo pipeline;
   - record pass/fail/skip and the reason for each skip.

4. Build a **claims ledger** for every current headline claim:
   - 12,460 tok/s;
   - 21/21 tests;
   - 3 backends;
   - 4 policies;
   - 79% hit rate / 40% memory reduction;
   - 100% NPU-offloadable;
   - < 3 ms;
   - tamper-proof;
   - hallucination-free;
   - single command / no external dependencies.
   For each: source file/README line, supporting artifact (or none), hardware, status (`measured`, `modelled`, `unverified`, `false`).

5. Identify:
   - synthetic data paths;
   - hardcoded outputs;
   - hardcoded metric labels;
   - absolute paths;
   - missing files referenced by docs;
   - tests that cannot fail;
   - claims with no artifact.

6. Save logs and hashes:
   - `handoff/00/logs/`;
   - hash historical benchmark files before touching anything;
   - list `git status` / untracked files (if a repo exists).

## Acceptance

- no source changes;
- exact commands and outputs recorded;
- hardware matrix complete;
- claims ledger complete;
- top 10 risks ranked;
- no stage 01 work started.

## Required handoff

`llm-inference/handoff/00.md` and `edge-rag-demo/handoff/00.md`:

1. environment;
2. commands and results;
3. pass/fail/skip counts;
4. claims ledger;
5. blockers;
6. recommended stage 01 scope;
7. explicit "stopped here."

## Guardrails

- Do not download models or catalog data.
- Do not "fix" anything.
- Do not claim a test passed if it was skipped.
- Do not delete build artifacts you need for baseline hashes.

## Stop

Stop after both handoffs and the claims ledger. Ask for review.
