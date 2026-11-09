# QEdge claims ledger

Last verified: 2026-10-27 (Stage 01)

Status values:

- `measured` — reproducible output exists in the linked log;
- `modelled` — derived from a model or estimate, not a device measurement;
- `projected` — explicitly a future/other-hardware estimate;
- `unverified` — no acceptable evidence yet;
- `false` — contradicted by the audit or source;
- `structural` — true by construction/tests, not a performance claim;
- `legacy` — historical Stage 00 material retained for traceability only.

The Stage 00 ledger is `handoff/00/claims-ledger.md`. Do not edit it.

## Current claims

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| A real model is loaded and generates tokens. | false | `llm-inference/src/main.cpp`; Stage 02 not started | No tokenizer, checkpoint, or generation path exists. |
| C++ `tok/s` output is model throughput. | false | `llm-inference/src/benchmark/benchmark.cpp`; `legacy/README.stage00.md` | Random synthetic tensors and cache bookkeeping. |
| QNN path executes on Hexagon/NPU. | false | `llm-inference/src/backend/qnn_stub_backend.cpp` | CPU stub; the code now avoids an NPU speedup projection. |
| Snapdragon/QNN performance is measured. | unverified | `handoff/00/logs/llm_env.log`; no device run | No adjacent hardware, SDK, or AI Hub run. |
| Edge retrieval is FAISS + BM25 + RRF. | false | `edge-rag-demo/src/edge_rag/retriever.py` | Stage 01 is a local BM25 baseline. |
| Edge pipeline is 100% NPU-offloadable and under 3 ms on Snapdragon. | false | `edge-rag-demo/legacy/README.stage00.md`; profiler tests | Removed; no offload or projection is reported. |
| Edge decision policy is implemented. | false | `edge-rag-demo/src/edge_rag/generator.py` | Stage 01 prints a neutral template; Stage 07 owns decisions. |
| Edge evidence store is tamper-proof. | unverified | `edge-rag-demo/src/edge_rag/store.py` | Local SQLite mechanics only; authorization/concurrency risks remain open. |
| ABO catalog license is approved for shipping. | unverified | `handoff/00/` audit | Stage 00 evidence favors CC-BY-4.0, but no formal shipping decision is recorded. |
| Retrieval-quality score exists. | unverified | `edge-rag-demo/src/edge_rag/eval.py` | No human relevance labels; Stage 06/09 gate. |
| Shared schemas are valid Draft 2020-12 and examples validate. | measured | `handoff/01/logs/contracts-eval.log` | 7/7 schemas; `qedge-common` tests. |
| Python contract tests pass. | measured | `handoff/01/logs/qedge-common-tests.log` | 13 tests. |
| Edge unit tests pass with one explicit integration skip. | measured | `handoff/01/logs/edge-tests.log` | Fixture tests; skip reason is printed. |
| C++ CTest suite passes. | measured | `handoff/01/logs/llm-ctest.log` | 2/2 CTest tests on the synthetic kernel lab. |
| A fresh checkout runs `make setup && make test`. | measured | `handoff/01/logs/fresh-checkout.log` | Fresh copy without local `.venv` or build outputs. |
| No machine-specific home-directory path exists in source or tests. | measured | `handoff/01/logs/path-scan.log` | Scanner excludes historical logs, legacy copies, and build environments. |
| Edge package installs as a wheel outside the source tree. | measured | `handoff/01/logs/wheel-install.log` | Console entry point runs from a temporary venv. |
| Shared contract package installs as a wheel outside the source tree. | measured | `handoff/01/logs/wheel-install.log` | Installed package loads its packaged schemas and validates a copied product JSON. |
| C++ kernel-lab binaries and headers install outside the source tree. | measured | `handoff/01/logs/llm-install.log` | `cmake --install`; installed binaries run from a temporary prefix. |
| Historical Stage 00 result hashes are unchanged. | measured | `handoff/01/logs/historical-hashes.log` | `handoff/00/artifact-manifest.sha256`. |
| Stage 01 edge demo runs against the synthetic fixture. | measured | `handoff/01/logs/edge-demo.log` | Fixture mechanics only; not ABO or model quality. |
| Root `make demo`, `make bench`, and `make eval` gates complete. | measured | `handoff/01/logs/root-demo.log`; `root-bench.log`; `root-eval.log` | Synthetic/lab outputs only; `make bench` emits `measured:false`. |
| Tests avoid model/catalog downloads. | structural | `Makefile` targets; fixture catalog | No model or catalog file is added; setup only resolves locked Python packages. |
| Both projects use the same product field names. | structural | `qedge-common/tests/test_schemas.py`; `edge-rag-demo/tests/test_pipeline.py` | Shared schema is the source of truth. |

## Retired Stage 00 numbers

The following may appear only in `legacy/` or `handoff/00/` with a clear
historical label:

- 12,460 tok/s and related synthetic C++ throughput rows;
- 100% NPU-offloadable and `< 3 ms` Snapdragon projection;
- 79% / 6,368 tok/mJ edge footer values;
- “hallucination-resistant” and “tamper-proof” claims.

## Rule for new measurements

A new `measured` row must link a log containing the command, timestamp,
hardware/platform, configuration, and raw output. Synthetic runs use
`measured: false` in `benchmark_result.schema.json`.
