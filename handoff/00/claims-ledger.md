# Stage 00 — Combined Claims Ledger

Date: 2026-10-27
Scope: `/Users/manuvashistha/Developer/qualcomm/llm-inference` and `/Users/manuvashistha/Developer/qualcomm/edge-rag-demo`
Status values: `measured`, `modelled`, `projected`, `unverified`, `false`, `structural`.

## llm-inference

| Claim | Source | Recorded value | Current evidence | Hardware | Status | Notes |
|---|---|---|---:|---|---|---|
| Peak throughput | `llm-inference/README.md:40,144-160` | 12,460 tok/s | Observed 12,285.0–19,701.2 tok/s across three logged synthetic runs | Apple M2 CPU | false | No model is loaded; benchmark times cache bookkeeping and a synthetic attention call. |
| Test suite | `llm-inference/README.md:52,115` | 21/21 | 21/21 combined runner; 12/12 QNN runner | Apple M2 | measured | Green tests, but no model/tokenizer/quality coverage. |
| Backends | `llm-inference/README.md:48-49` | 3 (CPU/Metal/QNN) | CPU executes; Metal attention is incomplete; QNN is a CPU stub | Apple M2 | false | Metal `attention()` returns `matmul(q,k)`; QNN has no QNN SDK/device. |
| KV-cache policies | `llm-inference/README.md:44,84-86` | 4 | FIFO/LRU/KeyDiff/QUOKA implemented in C++ | Apple M2 | structural | Policies operate on synthetic tensors; QUOKA has no query signal. |
| Headline cache run | `llm-inference/README.md:144-160` | 100% hit, 0 evictions | 512 capacity / 8 layers = 64 per layer; 64 tokens | Apple M2 | structural | Capacity equals token count; zero evictions is guaranteed, not a result. |
| QUOKA energy advantage | `llm-inference/README.md:385` | Up to 40% higher tok/mJ | QNN logs: FIFO 1,100–1,281 vs QUOKA 703–740 tok/mJ across three runs | Apple M2 (CPU stub) | false | Also, the report prints FIFO as the recommendation and then a canned QUOKA claim. |
| Model inference | `llm-inference/README.md:75` | Full decoder-only pipeline | No model loader/tokenizer; `models/` empty | — | false | `scripts/download_model.py` downloads only; no C++/Python runner reads weights. |
| Metal throughput | `llm-inference/README.md:149-156` | 3,346 tok/s | Observed 2,789.8–3,914.1 tok/s across three logged synthetic runs | Apple M2 Metal | false | Not model throughput; attention path is not a full attention implementation. |
| QNN throughput | `llm-inference/handoff/A-qnn-stub.md:70-85` | ~5,000 tok/s | Observed 6,517–7,442 tok/s across three logged CPU-stub runs | Apple M2 CPU stub | false | QNN stub delegates all operations to CPU. |
| Peak cache memory | `llm-inference/README.md:144-156` | 2 MB | Observed 2 MB in CPU/Metal synthetic runs; 0.95 MB in QNN stub | Apple M2 | structural | Cache bookkeeping only; not model/KV memory for a real model. |
| Portable C++ | `llm-inference/README.md:10` | Cross-platform | macOS frameworks (Accelerate, Metal, Foundation) | macOS | false | No Linux/Android build path. |

## edge-rag-demo

| Claim | Source | Recorded value | Current evidence | Hardware | Status | Notes |
|---|---|---|---:|---|---|---|
| Test suite | `edge-rag-demo/handoff/D-integration.md:18` | 10/10 | 10/10 passed in 7.802 s | Apple M2 | measured | Tests rely on an external `.venv` and external repos; several assertions enshrine wrong behavior. |
| End-to-end latency | `edge-rag-demo/README.md:25-42` | ~55 ms / 28 ms | Observed 57.9–113.3 ms | Apple M2 CPU | measured | In-process pipeline time; varies by run; README numbers are stale. |
| NPU-offloadable | `edge-rag-demo/demo_pipeline.py:132` | 100% | Derived from hardcoded stage metadata | Apple M2 | false | `PipelineProfiler.STAGE_INFO` sets the flags; no NPU is present. |
| Snapdragon latency | `edge-rag-demo/README.md:34`; `edge-rag-demo/demo_pipeline.py:133` | < 3 ms | Hardcoded string; projection uses 25x/5x assumptions | — | projected | No Snapdragon device, QNN SDK, or AI Hub run. |
| QUOKA hit rate | `edge-rag-demo/README.md:29`; `edge-rag-demo/src/edge_rag/generator.py:50-55` | 79% | QNN benchmark observed 52.4% | Apple M2 CPU stub | false | Fallback string in `generator.py`; not the live result. |
| QUOKA energy | `edge-rag-demo/README.md:29`; `edge-rag-demo/src/edge_rag/generator.py:50-55` | 6,368 tok/mJ | Observed 723–861 tok/mJ in demo footers; QNN stub runs 703–740 tok/mJ | Apple M2 CPU stub | false | README and footer overstate by roughly 7–9x. |
| Memory reduction | `edge-rag-demo/README.md:29` | 40% | No measured comparison | — | unverified | QNN stub reports identical 0.95 MB peak for every policy. |
| Hallucination-free | `edge-rag-demo/README.md:23,29` | Stated | Generator always says "matches your requirements" | Apple M2 | false | Reproduced with CONTRADICTED and UNKNOWN inputs. |
| Tamper-proof evidence | `edge-rag-demo/README.md:5` | Stated | Local SQLite; no threat model or tamper test | Apple M2 | unverified | "Tamper-proof" is not established. |
| Catalog | `edge-rag-demo/README.md:5` | 500 items | 500 rows, 500 unique item IDs, 500 unique variant IDs | — | measured | No family ID; duplicate-family analysis is still required. |
| Catalog license | `discoveryeval/data/catalog/abo-v2-pilot-500/sources/registry.yaml:2,14` | CC-BY-4.0 | All 500 rows `license_ref=CC-BY-4.0`; upstream license file and registry note present | — | measured | Registry says ABO moved from CC BY-NC to CC BY in 2023; final shipping decision still required. |
| Self-contained | `edge-rag-demo/README.md:5` | No external dependencies | `.venv` symlink and absolute paths to two external repos | — | false | `uv run` resolves Python from `discoveryeval/.venv`. |

## Risk register

The ranked top-10 risks are in `/Users/manuvashistha/Developer/qualcomm/handoff/00/risks.md`.

## Key conclusion

The green suites do not support the headline claims. `llm-inference` has no real model path; `edge-rag-demo` has a real catalog lookup but a template answer and hardcoded hardware telemetry. Stage 01 must establish packaging and truthful baselines before any performance or NPU work.
