# Stage 08 — End-to-End Integration

## Goal

Deliver a working assistant with two modes — `find` and `check` — that uses real retrieval, real evidence, and optionally real generation. It must work when the model is unavailable.

## Read first

- `02-TECHNICAL-SPEC.md` section 3
- `04-ACCEPTANCE-AND-DELIVERABLES.md` (stage 08)
- `handoff/06.md`, `handoff/07.md`, `handoff/03.md`

## Tasks

1. **CLI**
   - `qedge find <query> --k 5`
   - `qedge find --image <path> --k 5`
   - `qedge check --item <id> --require "width<=60cm" --require "material=oak"`
   - `qedge compare <id1> <id2> --require ...`
   - `qedge ask "<natural-language question>"` (optional)
   - `qedge status`

2. **Two modes**
   - `find`: retrieval only; no attribute claims from images.
   - `check`: evidence decisions; no positive answer unless supported.

3. **Optional generation**
   - Use QEdge Runtime if the model is available.
   - Pass only verified claims to the model.
   - Validate generated claims; fall back to extractive.
   - The demo must work with `--no-llm`.

4. **Measured stage timings**
   - retrieval;
   - reranking;
   - evidence lookup;
   - verification;
   - generation;
   - total.
   - Remove hardcoded NPU candidate flags and `<3 ms` strings.
   - Report which backend ran each stage.

5. **Ten scenarios**
   - text exact attribute;
   - image query;
   - unit conversion;
   - contradiction;
   - missing attribute;
   - stale source;
   - zero-overlap;
   - negation;
   - two-product comparison;
   - abstention.

6. **Graceful failure**
   - no model: extractive;
   - no index: clear error;
   - no image license: placeholder;
   - no DB: local SQLite;
   - no NPU: `hardware_status` says unverified.

7. **Demo**
   - `make demo` starts the CLI or a minimal local web UI.
   - No network after setup.
   - No hardcoded answers.

## Acceptance

- one command starts the demo;
- all 10 scenarios run;
- no "matches" after a contradiction;
- image search does not claim dimensions;
- timings are measured;
- fallback works;
- no hardcoded stage labels.

## Required handoff

`edge-rag-demo/handoff/08.md`:

- commands;
- scenario table with expected/actual;
- timings;
- fallbacks;
- open issues;
- "stopped here."

## Guardrails

- Do not add a cloud service.
- Do not require the LLM.
- Do not add a large UI framework.
- Do not hide a failed scenario.

## Stop

Stop after the handoff. Ask for review.
