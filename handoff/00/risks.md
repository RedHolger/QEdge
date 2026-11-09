# Stage 00 — Top 10 Risks (Ranked)

Date: 2026-10-27
Scope: `/Users/manuvashistha/Developer/qualcomm/llm-inference` and `/Users/manuvashistha/Developer/qualcomm/edge-rag-demo`

| Rank | Risk | Severity | Evidence | Next action |
|---:|---|---|---|---|
| 1 | `llm-inference` has no real model runtime; its headline throughput is synthetic. | Critical | `models/` empty; no weight/tokenizer loader; `Benchmark::generate_random_tensor` reseeds `rng(42)` on every call (`src/benchmark/benchmark.cpp:12`). | Stage 01/02: build a real SmolLM2-135M reference path; move synthetic numbers to `legacy/`. |
| 2 | All Qualcomm NPU claims are unverified; QNN is a CPU stub. | Critical | `QnnStubBackend` delegates to CPU; no QNN/QAIRT/AI Hub/device found; `llm_env.log`. | Keep NPU status `unverified` until a device run with logs; stage 05 gate. |
| 3 | `edge-rag-demo` always answers “matches your requirements,” even for contradicted or unknown evidence. | Critical | `src/edge_rag/generator.py:72`; reproduced in `edge_adversarial.log`; test at `tests/test_pipeline.py:44` enshrines it. | Stage 07: verdict-aware response, abstention, and negative tests. |
| 4 | `edge-rag-demo` is not reproducible outside this Mac. | High | `.venv` symlink to `discoveryeval/.venv`; five absolute source/test references; undeclared real dependencies. | Stage 01: remove symlink/absolute paths, declare dependencies, add fixture-based tests. |
| 5 | The LLM attention benchmark has mismatched batch shapes and can read out of bounds. | High | `q` batch 32 vs `k`/`v` batch 1 (`src/benchmark/benchmark.cpp:57-59,73-74`); CPU indexing at `src/backend/cpu_backend.cpp:141-148`; Release asserts disabled. | Stage 01: add ASan/UBSan debug preset and a shape-correctness test. |
| 6 | The edge pipeline invents width/oak requirements and silently defaults missing units to cm. | High | `demo_pipeline.py:52-54`; `src/edge_rag/verifier.py:130-139`; red-chair demo and adversarial log. | Stage 07: explicit requirement model; reject missing/ambiguous units; never inject defaults. |
| 7 | Hardware telemetry and headline claims are hardcoded or stale. | High | `PipelineProfiler.STAGE_INFO`; 25x/5x projection; 79%/6,368 fallback; README claims 6,368 tok/mJ and <3 ms. | Stage 01: remove projections from user-facing output; add a claims gate. |
| 8 | The LLM `tok/mJ` metric mixes full-run throughput with op-only energy time. | High | `src/backend/qnn_stub_backend.cpp:188-212`; `src/benchmark/power_report.cpp:50-55`; QUOKA is worse than FIFO in every run. | Stage 05: one consistent measurement window and a real power source; no canned recommendation. |
| 9 | Packaging and licence hygiene are incomplete. | Medium-High | Missing `llm-inference/LICENSE`; macOS-only CMake; no CTest; no project `LICENSE` in `edge-rag-demo`; catalog licence decision pending. | Stage 01: add licence files/decisions and portable build boundaries. |
| 10 | Evaluation data and human labels are not ready. | Medium-High | No 30-query pilot labels, 120 reviewed queries, or 80 evidence cases; no family-level ID in the 500-row catalog. | Stage 03/04: create labels and family-aware splits before quality claims. |

## Gate

No claim above may be promoted to a headline until its evidence column points to a reproducible artifact with a hardware label.
