# Current gap classification

All G01–G12 remain present. Static classifications are not claimed as separately executed numerical experiments. All 46 audited source hashes still match.

| Gap | Basis | Finding | Source | Next prompt |
| --- | --- | --- | --- | --- |
| G01 | dynamic | [CPU attention head indexing and per-head output are unsafe; ASan fails CTest and both entry points.](logs/finalization/asan-ctest.log) | `llm-inference/src/backend/cpu_backend.cpp:127` | 01-kernel-safety |
| G02 | source inspection | [Metal attention is raw matmul(q,k), ignoring V, softmax and mask.](../../../docs/research-review/ANALYSIS.md) | `llm-inference/src/backend/metal_backend.mm:111` | 01-kernel-safety |
| G03 | source inspection | [RoPE has fixed base, repeated position and incorrect split-half frequency exponent.](../../../docs/research-review/ANALYSIS.md) | `llm-inference/src/backend/cpu_backend.cpp:35` | 01-kernel-safety |
| G04 | source inspection | [RNG is reseeded per tensor, workload is synthetic, retrieval failure ignored and attention uses a single retrieved token.](../../../docs/research-review/ANALYSIS.md) | `llm-inference/src/benchmark/benchmark.cpp:10` | 01-kernel-safety (unsafe calls); 08-quantization-measurement (benchmark redesign) |
| G05 | source inspection | [LRU uses counts, KeyDiff newest-key similarity, QUOKA has no query input.](../../../docs/research-review/ANALYSIS.md) | `llm-inference/src/kv_cache/kv_cache.cpp` | 06-cache-contract and 07-research-selectors |
| G06 | inventory | [Runtime is a README placeholder; project models directory is empty.](logs/asset-inventory.log) | `llm-inference/runtime/README.md` | 05-real-model-baseline |
| G07 | dynamic | [Red-chair defaults, missing-unit cm assumption, and negation parsed as positive oak.](logs/finalization/gap-probe.log) | `edge-rag-demo/src/edge_rag/cli.py` | 04-assistant-correctness |
| G08 | dynamic | [Bob reads and overwrites Alice-owned document.](logs/finalization/gap-probe.log) | `edge-rag-demo/src/edge_rag/store.py` | 03-evidence-store |
| G09 | dynamic | [Empty SUPPORTED and unverified measured hardware accepted by validators.](logs/finalization/gap-probe.log) | `qedge-common/schemas/decision.schema.json; hardware_status.schema.json` | 02-contracts |
| G10 | dynamic | [Generic wood falsely supports oak; retrieved chunks are flattened text without revision/hash.](logs/finalization/extra-probe.log) | `edge-rag-demo/src/edge_rag/verifier.py` | 03-evidence-store and 04-assistant-correctness |
| G11 | source/inventory/evaluation | [BM25 fixture pipeline only; no integrated neural/image retrieval or human quality labels.](logs/finalization/edge-eval.log) | `edge-rag-demo/src/edge_rag/retriever.py` | 09-catalog-retrieval |
| G12 | source inspection | [CPU-stub energy model; backend-operation and outer workload time scopes differ. No measured device energy.](../../../docs/research-review/ANALYSIS.md) | `llm-inference/src/backend/qnn_stub_backend.cpp` | 08-quantization-measurement and 11-qualcomm-deployment |
