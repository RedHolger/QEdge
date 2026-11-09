# Review: 06-cache-contract

Date: 2 November 2026  
Reviewed stage: `06-cache-contract`  
Review type: Independent review (evaluating KV cache contract, position semantics, failure atomicity, tensor contracts, allocation-aware analytical peak memory telemetry, and memory allowance repairs)  
Review scope: Stage 06 position-correct, bounded KV cache contract under `llm-inference/runtime`, primary checkpoint `HuggingFaceTB/SmolLM2-135M-Instruct` pinned to commit `12fd25f77366fa6b3b4b768ec3050bf629380bac`, formal contract specification (`CACHE_CONTRACT.md`), `BoundedKVCache` supporting `full`, `fifo`, and `recency_window` policies, `QEdgeTransformersCacheAdapter` subclassing `transformers.cache_utils.Cache`, non-contiguous 4D causal attention masking, RoPE phase invariance, GQA support, failure atomicity of pre-mutation validation, strict rejection of out-of-order and replayed positions, layout/dtype/device contract enforcement, factual resident payload derivation directly from live tensors, authoritative analytical coexisting peak memory accounting, separation of process-wide allocator snapshots from cache bounds, universal preflight budget validation across `update()`, `evict_to_capacity()`, and `evict_positions()`, and expanded regression testbed (72 tests).

---

## Verdict

**ACCEPTED (bounded cache-correctness scope)**

The Stage 06 bounded KV cache contract repairs satisfy all requirements specified in `docs/research-review/prompts/06-cache-contract.md` and subsequent review findings. Independent verification confirms:

- **Runtime Tests:** **72 / 72 passed**, including bounded cache behavior, full-cache 100% greedy token parity, logits agreement, chunked prefill equivalence, and CPU/MPS parity.
- **Budget Rejection Failure Atomicity:** Rejected updates preserve successful state and peak telemetry; separate `last_rejected_attempt` diagnostics are recorded cleanly without mutating committed cache state or counters.
- **Multi-Layer Pruning & Explicit Eviction Atomicity:** Budget-rejected multi-layer pruning (`evict_to_capacity`) and explicit eviction (`evict_positions`) preflight candidate peaks across all layers and preserve resident positions, counters, and peak history with zero partial mutation.
- **Allocator Snapshot Isolation:** Global accelerator allocation snapshots (e.g., MPS `torch.mps.current_allocated_memory()` or CUDA process allocations) remain separate from analytical cache peaks (`allocator_snapshot`), while `measured_peak_bytes` remains `null` in the absence of an isolated cache-specific measurement.
- **Authoritative Analytical Bounds:** Memory figures remain analytical bounds within the documented scope (coexisting K/V tensors, position tensors, index tensors) rather than measured total-process peaks, and documented exclusions (caller-owned attention masks, Python overhead, allocator page pooling) are preserved.
- **Suite Integrity:**
  - `llm-runtime-tests`: **72 / 72 passed**.
  - `cache-contract-probes`: **10 / 10 passed** (`verify_cache_contract.py`).
  - `llm-ctest`: **3 / 3 passed** (`llm_kv_cache`, `kernel_safety`, `qnn_stub_cpu`).
  - `edge-tests`: **68 passed, 0 failed, 1 expected skip** (`RealCatalogIntegrationTests.test_catalog_loads`).
  - `qedge-common-tests`: **49 passed, 0 failed, 0 skipped**.
  - `historical-hashes`: **30 / 30 verified**.
  - `schema-hashes`: **7 / 7 verified byte-for-byte**.

---

## Limits and Scope Boundaries

1. **Bounded Cache-Correctness Scope Only:**
   This acceptance establishes positional and memory correctness for bounded KV caching. It establishes no speedup, throughput advantages, or downstream quality ranking claims for bounded policies over the baseline.
2. **Second Model Deferral:**
   Per instructions, the secondary model (`Qwen/Qwen2.5-0.5B-Instruct`) remains deferred.
3. **Stage 07 Scope:**
   Proceed to Stage 07 (`07-research-selectors`) only. Defer Stage 08 and Qualcomm QNN/Hexagon execution.
