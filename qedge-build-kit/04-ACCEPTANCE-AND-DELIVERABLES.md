# QEdge Acceptance Criteria and Deliverables

## 1. How acceptance works

A stage is accepted only when:

1. The required artifacts exist and are reproducible.
2. The behavior in the prompt is demonstrated, not just unit-tested.
3. The handoff states exactly what passed, failed, and was skipped.
4. No claim in the handoff or README exceeds the evidence.
5. The diff is limited to the stage scope.

Green tests are necessary but not sufficient.

## 2. Stage acceptance table

### 00 — Audit

Required artifacts:

- `llm-inference/handoff/00.md`
- `edge-rag-demo/handoff/00.md`
- `handoff/00/logs/` with test, build, and environment logs
- `handoff/00/claims-ledger.md`

Acceptance:

- no source changes;
- exact commands and outputs recorded;
- every current headline claim mapped to evidence (or marked unverified);
- hardware matrix recorded;
- absolute paths and external dependencies listed.

### 01 — Foundations

Required artifacts:

- root `Makefile` with `setup`, `test`, `demo`, `bench`, `eval`;
- `pyproject.toml` + lockfile for Python;
- CMake presets for C++;
- `qedge-common/schemas/` with the shared contracts;
- `docs/ARCHITECTURE.md`, `docs/CLAIMS.md`;
- `handoff/01.md` in each repo.

Acceptance:

- a fresh checkout runs `make test` without manual path edits;
- no `/Users/` absolute paths in source or tests;
- historical result hashes unchanged;
- missing integrations skip explicitly, not silently;
- wheel/editable install verified outside the source tree.

### 02 — Real model

Required artifacts:

- `runtime/model_runner.py`;
- `runtime/tokenizer` path or HF tokenizer usage;
- `runtime/cli.py`;
- `benchmarks/02/` generation records;
- `handoff/02.md`.

Acceptance:

- a real model generates coherent text from a real prompt;
- tokenizer round-trip is exact for a fixed sample;
- greedy output matches the reference path for a fixed seed/prompt;
- TTFT and decode tok/s are recorded with hardware metadata;
- no random tensors are used as model input;
- no claim of NPU execution.

### 03 — KV cache

Required artifacts:

- `runtime/cache/` with full/FIFO/LRU/KeyDiff/QUOKA;
- tests for shape, GQA, eviction, position IDs, and 100% parity;
- `benchmarks/03/` quality/latency JSON;
- `handoff/03.md`.

Acceptance:

- at 100% budget, output matches full cache;
- at 50%/25%, policies produce different measured curves;
- QUOKA uses a real query signal;
- no out-of-bounds/NaN/sanitizer failures;
- cache stats reconcile;
- a quality/latency table exists with seeds and CIs.

### 04 — Quantization + benchmark

Required artifacts:

- `benchmarks/04/` FP16/Q8_0/Q4_K_M results;
- `runtime/bench/` harness with warmup and repeats;
- `reports/04-benchmark.md`;
- `handoff/04.md`.

Acceptance:

- at least three precision variants measured on the same prompts;
- model size and peak RSS measured;
- p50/p95 and CIs reported;
- energy is labelled measured or modelled;
- no stale or hardcoded numbers;
- one command reproduces the table.

### 05 — QNN/AI Hub

Required artifacts:

- `qnn/export_onnx.py`;
- `qnn/run_ort_qnn.py`;
- `qnn/README.md`;
- `qnn/hardware_status.json`;
- `qnn/device_runbook.md`;
- `handoff/05.md`.

Acceptance:

- export runs locally;
- ONNX Runtime CPU path runs;
- QNN EP integration compiles/imports or has a clear blocker;
- device runbook is complete;
- if device logs exist, they are included;
- if not, every NPU performance claim is marked `unverified`.

### 06 — Catalog + retrieval

Required artifacts:

- `catalog/LICENSE-DECISION.md`;
- `catalog/catalog.jsonl` + `manifest.json`;
- `catalog/provenance_report.md`;
- `catalog/sample_gallery.html`;
- retrieval index build command;
- `eval/queries.dev.jsonl`, `eval/qrels.dev.jsonl`;
- `eval/annotation/` tool;
- `reports/06-retrieval.md`;
- `handoff/06.md`.

Acceptance:

- 500 items (or a documented smaller licensed subset);
- target 500 rows but report unique families separately;
- no external absolute paths;
- retrieval runs on text and, if licensed, image queries;
- 30-query human pilot exists;
- metrics have CIs;
- license decision is explicit.

### 07 — Evidence + abstention

Required artifacts:

- `src/edge_rag/evidence/` store and verifier;
- `eval/decisions.dev.jsonl`, `eval/decisions.test.jsonl`;
- `reports/07-evidence.md`;
- `handoff/07.md`.

Acceptance:

- 80 decision cases (40 dev/40 test);
- `SUPPORTED`/`CONTRADICTED`/`UNKNOWN` behavior matches expectations;
- no positive recommendation when any required attribute is not supported;
- citations resolve to a stored source;
- unit conversion and missing-unit behavior are tested;
- source update changes the decision;
- no "matches your requirements" hardcoded text.

### 08 — Integration

Required artifacts:

- `edge-rag-demo/demo/` CLI or web app;
- `reports/08-scenarios.md` with 10 scenarios;
- `handoff/08.md`.

Acceptance:

- one command starts the demo;
- 10 scenarios cover text, image, contradiction, unknown, abstention, and fallback;
- the demo works without a model download (extractive fallback);
- with the model, generation uses only verified evidence;
- stage timings are measured, not hardcoded;
- no network required after data/models are cached.

### 09 — Evaluation

Required artifacts:

- `reports/09-final-evaluation.md`;
- `reports/09-tables/*.json|csv`;
- `reports/09-plots/`;
- updated `docs/CLAIMS.md`;
- `handoff/09.md`.

Acceptance:

- retrieval, evidence, abstention, runtime, and end-to-end metrics are reported;
- 120-query set and 40-case held-out set used as specified;
- ablations exist;
- confidence intervals and effect sizes exist;
- failure taxonomy exists;
- every headline claim links to an artifact;
- no claim relies on a Mac number for an NPU statement.

### 10 — Release + interview

Required artifacts:

- root README with one-command setup;
- architecture diagram;
- benchmark report;
- demo video or script;
- 5-minute demo script;
- 15-minute deep-dive notes;
- resume bullets;
- `handoff/10.md`.

Acceptance:

- fresh machine test passes;
- `make demo` and `make bench` work;
- all docs agree on numbers;
- hardware status is accurate;
- no absolute paths, secrets, or model weights committed;
- tagged release.

### 11 — Hardware stretch (optional)

Required artifacts:

- device logs and traces;
- `reports/11-device.md`;
- updated claims ledger.

Acceptance:

- real Snapdragon or AI Hub device logs;
- exact model, quantization, runtime, driver, and OS;
- performance claims only from those logs;
- if unavailable, stage is not started.

## 3. Required artifacts in every handoff

Each `handoff/NN.md` must contain:

1. stage name and date;
2. commands run;
3. test counts: passed/failed/skipped, with reasons;
4. artifacts produced, with paths;
5. claims added or changed;
6. known limitations;
7. security implications (if any);
8. next-stage prerequisites;
9. explicit "stopped here" statement.

## 4. Review checklist

A reviewer must:

- reproduce the stage from a clean checkout;
- run the acceptance command, not just the unit tests;
- inspect the diff for scope creep;
- grep for hardcoded numbers, `/Users/`, `TODO`, `stub`, `fake`, `mock`, `100%`;
- verify every number in the handoff;
- verify hardware labels;
- check that skips are explicit;
- check that historical results are unchanged;
- reject claims that are not backed by artifacts.

## 5. Definition of done

### Runtime

- real model, real tokenizer, real generation;
- cache correctness at full budget;
- policy curves under pressure;
- quantization table;
- reproducible benchmark;
- QNN path documented and honestly gated.

### Assistant

- real catalog with provenance;
- text and (if licensed) image retrieval;
- human-reviewed retrieval set;
- evidence decisions with citations;
- abstention;
- end-to-end demo;
- measured latency/memory.

### Portfolio

- one-command setup;
- one-command demo;
- one-command benchmark;
- one-command evaluation;
- a 5-minute story;
- no unsupported claim.

## 6. Stop rules

Stop and ask for review if:

- a required dataset or model license is unresolved;
- the only way to pass is to hardcode an output;
- the benchmark is not reproducible;
- a test passes but the behavior is wrong;
- an NPU claim is about to be made without device logs;
- the stage is more than 2x its estimate;
- the diff touches unrelated projects.

## 7. Claims ledger template

```markdown
| Claim | Value | Evidence artifact | Hardware | Status |
|---|---|---:|---|---|
| Decode throughput | ... | benchmarks/04/....json | Apple M2 CPU | measured |
| NPU throughput | — | none | Snapdragon X Elite | unverified |
| Retrieval Recall@5 | ... | reports/09/... | — | measured |
| False SUPPORTED rate | ... | reports/09/... | — | measured |
```

No row may say `measured` without a path to a log or JSON artifact.
