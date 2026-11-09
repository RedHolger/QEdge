# Stage 10 — Release and Interview Package

## Goal

Turn the working system into a portfolio deliverable that a reviewer can run, understand, and interrogate in five minutes.

## Read first

- `05-DEMO-AND-INTERVIEW.md`
- `04-ACCEPTANCE-AND-DELIVERABLES.md` (stage 10)
- `handoff/09.md`

## Tasks

1. **One-command setup**
   - `make setup`
   - `make demo`
   - `make bench`
   - `make eval`
   - Document prerequisites and expected runtime.

2. **README**
   - What it is.
   - Architecture diagram.
   - Quick start.
   - What is measured vs modelled vs unverified.
   - Hardware status.
   - Results table with links to artifacts.
   - Limitations.
   - License and data provenance.

3. **Reports**
   - Runtime: model, cache, quantization, latency, memory.
   - Assistant: retrieval, evidence, abstention.
   - End-to-end: 10 scenarios.
   - Failure analysis.

4. **Demo**
   - 5-minute script.
   - 15-minute deep dive.
   - Record a video or produce a terminal recording.
   - Prepare screenshots/JSON artifacts.

5. **Interview**
   - 30-second pitch.
   - 10 likely questions and answers.
   - Resume bullets backed by artifacts.
   - A "what I would do next" slide.

6. **Repository hygiene**
   - Remove `build/`, `.venv`, `__pycache__`, model weights, and large caches from the deliverable.
   - Add `.gitignore`.
   - Check licenses and attribution.
   - Tag a release.

7. **Fresh-machine test**
   - Clone/copy to a clean directory.
   - Run setup and demo.
   - Record failures.
   - Fix only packaging issues.

## Acceptance

- fresh checkout works;
- `make demo` runs without network;
- `make bench` reproduces the headline table;
- docs agree with artifacts;
- no absolute paths;
- no fabricated numbers;
- hardware status accurate;
- demo video/script exists.

## Required handoff

`handoff/10.md` in each repo plus root `handoff/10-root.md`:

- fresh-machine log;
- commands;
- demo links;
- claims ledger;
- remaining limitations;
- "stopped here."

## Guardrails

- Do not add features during release.
- Do not rewrite historical results.
- Do not hide a broken path behind a screenshot.
- Do not claim production readiness.

## Stop

Stop after the handoff. This is the portfolio release.
