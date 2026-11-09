# Handoff: 06-cache-contract

Date: 31 October 2026  
Prompt ID: `06-cache-contract`  
Prerequisites: `05-real-model-baseline` (Accepted following independent review)  
Stage Verdict: **SELF_ASSESSMENT_READY_FOR_REVIEW** (Independent review required; do not imply this handoff constitutes an independent review)

---

## 1. Executive Summary

Stage 06 implements a position-correct, bounded Key-Value (KV) cache contract for the pinned `HuggingFaceTB/SmolLM2-135M-Instruct` reference runtime. Prior to policy implementation, the mathematical and architectural contract was formalized in [`llm-inference/runtime/docs/CACHE_CONTRACT.md`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/docs/CACHE_CONTRACT.md), establishing:

1. **Explicit Position Invariants:** Absolute sequence positions ($P_{\text{abs}} \in \mathbb{N}_0$) are decoupled from resident memory indices ($i_{\text{resident}} \in [0, S_{\text{resident}}-1]$) and total tokens seen ($T_{\text{seen}}$).
2. **Strict Batch Size 1 Policy:** Multi-batch eviction ($B > 1$) is rejected upfront with `ValueError` to prevent unsound cross-sequence eviction coupling.
3. **Classical LRU Degeneracy Resolution:** In dense self-attention, every query step evaluates $Q K^T$ across all resident keys, touching every token identically. Classical LRU tracking access timestamps or touch frequency is degenerate; the well-defined recency baseline is token arrival order (`recency_window`), pinned with initial attention sinks.
4. **RoPE Phase Invariance & GQA:** Key tensors preserve their true absolute rotation $R(P_{\text{abs}})$. Rotary dot products $Q_{\text{rot}}(P_q) K_{\text{rot}}(P_k)^T$ depend solely on relative phase $P_q - P_k$, maintaining exact mathematical invariance when intermediate keys are evicted. Grouped-Query Attention ($H_q = 9, H_{\text{kv}} = 3$) stores 3 KV heads and expands cleanly via head repetition.
5. **Tested Transformers Cache Adapter:** Subclasses `transformers.cache_utils.Cache`, establishing a documented compatibility boundary (`transformers >= 4.36`, verified on pinned `5.22.1`) without ad-hoc internal patching.
6. **Non-Contiguous Causal Attention Masking:** 4D attention masks ($[1, 1, S_q, S_k]$) enforce causality for non-contiguous resident positions, strictly suppressing future tokens ($P_k > P_q$) with $-\infty$ (yielding exact $0.0$ attention weight).
7. **Prefill Overflow & Factual Memory Accounting:** Multi-token prefill preserves full causal self-attention across the prompt, factually reporting the transient peak memory ($S_{\text{prompt}}$ tokens) before post-prefill eviction prunes resident occupancy to capacity $C$. Hand-calculated memory telemetry accounts for KV payload, capacity payload, metadata bytes, and peak temporary gathers.

---

## 2. Changed Files and Artifacts

| File | Change Nature | Purpose |
|---|---|---|
| [`reviews/research-review/05-real-model-baseline.md`](file:///Users/manuvashistha/Developer/qualcomm/reviews/research-review/05-real-model-baseline.md) | Created | Records formal independent review acceptance of Stage 05. |
| [`llm-inference/runtime/docs/CACHE_CONTRACT.md`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/docs/CACHE_CONTRACT.md) | Created | Mathematical specification of KV cache contract, failure atomicity, position invariants, and analytical peak memory formulas. |
| [`llm-inference/runtime/src/qedge_runtime/cache/contract.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/cache/contract.py) | Created | `EvictionPolicyKind`, `CacheMemoryStats` (with analytical peak and allowance fields), `LayerCacheState` (with `max_position_seen` cursor), and `compute_causal_attention_mask()`. |
| [`llm-inference/runtime/src/qedge_runtime/cache/bounded_cache.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/cache/bounded_cache.py) | Created | `BoundedKVCache` implementing two-phase failure-atomic validation, coexisting peak tracking, temporary memory allowance, and factual resident payload derivation. |
| [`llm-inference/runtime/src/qedge_runtime/cache/adapter.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/cache/adapter.py) | Created | `QEdgeTransformersCacheAdapter` subclassing `transformers.cache_utils.Cache` with full `cache_position` routing and `active_positions_context`. |
| [`llm-inference/runtime/src/qedge_runtime/cache/__init__.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/cache/__init__.py) | Created | Module exports for the cache package. |
| [`llm-inference/runtime/src/qedge_runtime/runner.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/runner.py) | Modified | Integrated `BoundedKVCache` and `adapter` into prefill and autoregressive decode with explicit active position routing; added cache telemetry. |
| [`llm-inference/runtime/src/qedge_runtime/benchmark.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/benchmark.py) | Modified | Added `--cache-policy`, `--cache-capacity`, `--sink-tokens` CLI flags; emits validated contract v2 records. |
| [`llm-inference/runtime/tests/test_cache_contract.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/tests/test_cache_contract.py) | Created | 28 focused test cases covering hand-calculated attention, failure atomicity, layout/dtype contracts, position routing, coexisting peak buffers, and chunked prefill. |
| [`handoff/research-review/06-cache-contract/probes/verify_cache_contract.py`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/06-cache-contract/probes/verify_cache_contract.py) | Created | Standalone 10-point verification probe. |
| [`handoff/research-review/06-cache-contract/results.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/06-cache-contract/results.json) | Created | Structured verification metrics, invariants, and scope boundaries. |

---

## 3. Formal Cache Contract Specification

### 3.1 Mathematical Definitions
- **Absolute Position ($P_{\text{abs}} \in \mathbb{N}_0$):** Canonical sequential index of a token in the input prompt or generated stream. Invariant across all tensor moves, gathers, and evictions.
- **Resident Index ($i_{\text{resident}} \in [0, S_{\text{resident}}-1]$):** Physical offset of a retained token inside key/value tensors along dimension 2.
- **Total Tokens Seen ($T_{\text{seen}} \in \mathbb{N}_0$):** Monotonic counter of all tokens admitted to the cache.
- **Reconciliation Invariant:** At every step across every layer:
  $$T_{\text{seen}} - S_{\text{resident}} = N_{\text{evictions}}$$

### 3.2 Permanent Eviction vs. Attention Selection
- **Permanent Eviction (`BoundedKVCache`):** Permanently removes token keys, values, and position metadata from tensor memory via `torch.index_select`, updating `N_{\text{evictions}}`.
- **Attention-Time Selection:** Temporary gather applied per query step (e.g., top-$k$ attention selection in Stage 07) that does not deallocate resident tensor capacity.

### 3.3 Classical LRU Degeneracy in Dense Attention
In dense causal attention, every query token evaluates cross-attention against **all** resident keys. Because every resident key is accessed on every decode step, access timestamps and frequency counters are degenerate (identical for all resident tokens). Renaming attention weights to "access frequency" or "LRU" conflates importance with recency. Consequently:
- `fifo`: First-in, first-out eviction of the oldest tokens outside the attention sink set.
- `recency_window`: Strictly defined sliding arrival recency window retaining the $W$ most recently arrived tokens plus $S_{\text{sink}}$ pinned initial tokens.

### 3.4 Rotary Position Embeddings (RoPE) Invariance
In SmolLM2, rotary embeddings rotate query and key vectors:
$$Q_{\text{rot}}(P_q) = R(P_q) Q, \quad K_{\text{rot}}(P_k) = R(P_k) K$$
The attention logits evaluate:
$$Q_{\text{rot}}(P_q) K_{\text{rot}}(P_k)^T = Q R(P_q) R(P_k)^T K^T = Q R(P_q - P_k) K^T$$
Because $K_{\text{rot}}(P_k)$ is rotated using its canonical absolute position $P_k$ before admission to the cache, its rotary phase is fixed. Evicting intermediate tokens ($k+1 \dots q-1$) does not alter $P_k$ or $R(P_k)$. The relative rotary phase between query $P_q$ and resident key $P_k$ remains exactly $P_q - P_k$.

### 3.5 Memory Accounting Formulas
For batch size $B=1$, dtype bytes $b$, $L$ layers, $H_{\text{kv}}$ heads, head dimension $D$, capacity $C$, and resident tokens $S$:
$$\text{Bytes per token per layer} = 2 \times H_{\text{kv}} \times D \times b$$
$$\text{Bytes per token model } (B_{\text{model}}) = \text{Bytes per token per layer} \times L$$
$$\text{Resident payload bytes} = \sum_{l=0}^{L-1} \left( \text{nelement}(K_l) \times b + \text{nelement}(V_l) \times b \right) = S \times B_{\text{model}}$$
$$\text{Capacity payload bytes} = C \times B_{\text{model}}$$
$$\text{Metadata bytes} = L \times S \times 8 + 64$$

#### Analytical Coexisting Peak Memory
When updating layer $l$ from $S_{\text{old}}$ tokens with $S_{\text{incoming}}$ tokens under capacity $C$ where $S_{\text{old}} + S_{\text{incoming}} > C$:
$$\text{Peak}_{\text{coexisting}} = \text{Payload}_{\text{other\_layers}} + \text{Payload}_{\text{old\_kv}} + \text{Payload}_{\text{incoming\_kv}} + \text{Payload}_{\text{concat\_kv}} + \text{Payload}_{\text{gathered\_kv}} + \text{Metadata}_{\text{bytes}}$$
This strictly bounds the transient physical memory required during tensor concatenation and index gathering before the old and unpruned buffers are released.

---

## 4. Review Defect Repairs and Verification

The independent review reported four blocking contract defects. All four were repaired, validated with targeted unit tests, and integrated into the verification probe:

### 4.1 Defect 1: Preserve Absolute Positions Through Adapter
- **Before:** `QEdgeTransformersCacheAdapter.update` dropped caller-provided `cache_kwargs={"cache_position": tensor([10])}`, causing `BoundedKVCache` to assume a zero start position `[0]`.
- **Repair:** `QEdgeTransformersCacheAdapter.update` extracts cache positions from `kwargs["cache_position"]`, `kwargs["query_positions"]`, `kwargs["position_ids"]`, `cache_kwargs` dict, positional argument dicts, or the new explicit caller channel `adapter.active_positions_context(positions)`. Tested across nonzero starts, sequence gaps, and chunked prefill.

### 4.2 Defect 2: Pre-Mutation Validation Atomicity
- **Before:** Calling `cache.update(query_positions=[5])` after resident position `[10]` raised a `ValueError` *after* appending `5` into the positions tensor (leaving corrupt state `[10, 5]`) and incrementing `tokens_seen=2`.
- **Repair:** Implemented strict two-phase execution. Phase 1 validates rank, batch size 1, KV heads, head dimension, dtype, device, position tensor integer dtype, non-negativity, strictly increasing ordering, and `incoming[0] > max_position_seen`. If any check fails, an exception is raised immediately with **zero** mutations to positions, keys, values, tokens seen, or eviction counters. Replaying previously evicted positions is explicitly rejected.

### 4.3 Defect 3: Layout, Dtype, and Device Contract Enforcement
- **Before:** A cache configured for $H_{\text{kv}}=1, D=1$, `float32` accepted tensors of shape `[1, 2, 1, 3]` in `float64` and reported 8 bytes for 96 bytes of physical payload.
- **Repair:** Added strict pre-mutation validation against cache configuration ($H_{\text{kv}}$, $D$, dtype, and device). Derived `actual_resident_payload_bytes` directly by summing `nelement() * element_size()` across all resident tensors in memory.

### 4.4 Defect 4: Accurate Peak Memory Telemetry & Allowance
- **Before:** A capacity-2 update with dimension 128 reported 3,152 peak bytes, ignoring coexisting buffers totaling $\ge 7,168$ bytes. Furthermore, on MPS/CUDA, process-wide allocation snapshots were mistakenly labelled as cache `measured_peak_bytes`, causing reported peaks to fluctuate with unrelated memory allocations. Additionally, budget rejection was not failure-atomic, mutating `analytical_peak_bytes` and setting `temporary_memory_overflow=True` on rejected attempts. Finally, multi-layer pruning and explicit eviction lacked consistent preflight allowance enforcement.
- **Repair:**
  1. **Authoritative Analytical Accounting:** Bounded cache peak telemetry (`analytical_peak_bytes`, `peak_temporary_bytes`) is derived authoritatively from coexisting buffers across all layers. Process-wide allocator snapshots are captured separately (`allocator_snapshot` with `backend`, `scope="global_process"`, `sampling_point="post_update"`), while `measured_peak_bytes` remains `null` unless an isolated cache measurement is performed.
  2. **Failure-Atomic Budget Rejection:** Preflight evaluation validates candidate bounds before committing any telemetry. Rejected candidate operations raise `MemoryError` with zero mutation to resident tensors, positions, counters, or historical peak telemetry. Rejected attempt diagnostics are recorded separately in `last_rejected_attempt`.
  3. **Universal Preflight Allowance Checks:** Preflight allowance calculation is enforced across `update()`, `evict_to_capacity()`, and `evict_positions()`, protecting multi-layer operations from partial mutations.

---

## 5. Hand-Calculated Mathematical Verification

A hand-calculated numeric attention example was executed and verified against `BoundedKVCache` in [`test_hand_calculated_attention_interior_eviction`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/tests/test_cache_contract.py#L40-L115):

```
Setup:
  L=1, H_kv=1, D=2, Scale = 1 / sqrt(2) ≈ 0.70710678
  Initial keys: K0=[1, 0] (p=0), K1=[0, 1] (p=1), K2=[1, 1] (p=2), K3=[0, 0] (p=3)
  Initial values: V0=[10, 20], V1=[30, 40], V2=[50, 60], V3=[70, 80]
  Action: Evict interior token 1 (pos 1). Resident positions: [0, 2, 3]
  Action: Add token 4: K4=[2, 0], V4=[90, 100] (pos 4). Resident positions: [0, 2, 3, 4]
  Query at pos 4: Q = [1, 0]

Dot products Q . K_i^T:
  Q . K0 = 1.0  => score s0 = 0.70710678  => exp(s0) ≈ 2.028115
  Q . K2 = 1.0  => score s2 = 0.70710678  => exp(s2) ≈ 2.028115
  Q . K3 = 0.0  => score s3 = 0.0         => exp(s3) = 1.000000
  Q . K4 = 2.0  => score s4 = 1.41421356  => exp(s4) ≈ 4.113250
  Sum(exp) ≈ 9.169480

Softmax weights:
  w0 ≈ 0.221181, w2 ≈ 0.221181, w3 ≈ 0.109057, w4 ≈ 0.448581

Expected Output Vector Out = sum(w_i * V_i):
  Out[0] = 0.221181*10 + 0.221181*50 + 0.109057*70 + 0.448581*90 = 61.2774
  Out[1] = 0.221181*20 + 0.221181*60 + 0.109057*80 + 0.448581*100 = 71.2774

Actual runtime tensor result matches expected within 7.63e-06.
```

Future-token masking was verified in [`test_hand_calculated_future_token_causal_masking`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/tests/test_cache_contract.py#L117-L157): for query $P_q = 2$ attending to resident keys at $[0, 2, 5]$, key 5 ($5 > 2$) is masked to $-\infty$, resulting in an attention weight of exactly $0.0$.

---

## 6. Verification Commands and Exit Codes

| Command | Working Directory | Exit Code | Results |
|---|---|:---:|---|
| `llm-inference/runtime/.venv/bin/python handoff/research-review/06-cache-contract/probes/verify_cache_contract.py` | Repo root | 0 | **10 / 10 passed** |
| `make -C llm-inference/runtime test` | Repo root | 0 | **72 / 72 passed** (0 failed, 0 skipped) |
| `make -C llm-inference test` | Repo root | 0 | **3 / 3 passed** (`llm_kv_cache`, `kernel_safety`, `qnn_stub_cpu`) |
| `make -C edge-rag-demo test` | Repo root | 0 | **68 passed, 0 failed, 1 expected skip** |
| `make -C qedge-common test` | Repo root | 0 | **49 passed, 0 failed, 0 skipped** |
| `python3 scripts/verify_historical_hashes.py` | Repo root | 0 | **30 / 30 verified**, 0 failures |
| Schema hash verification (7 v1 schemas) | Repo root | 0 | **7 / 7 verified byte-for-byte** |
| `python -m qedge_runtime.benchmark --cache-policy full` | Repo root | 0 | Emits validated contract v2 record |
| `python -m qedge_runtime.benchmark --cache-policy fifo --cache-capacity 48 --sink-tokens 4` | Repo root | 0 | Emits validated contract v2 record with eviction telemetry |

---

## 7. Verification Probe Highlights (`verify_cache_contract.py`)

1. **Check 1: Model checkpoint assets and architecture:** SmolLM2-135M verified (30 layers, 9 Q heads, 3 KV heads, head_dim 64).
2. **Check 2: Hand-calculated attention with interior eviction:** Max difference $7.63 \times 10^{-6}$.
3. **Check 3: Non-contiguous causal masking:** Future keys masked to $-\infty$, weight strictly $0.0$.
4. **Check 4: Capacity 1 and tiny budget reconciliation:** 10 tokens seen $-$ 1 resident token $=$ 9 evictions.
5. **Check 5: Review defects verification & negative invariant enforcement:** Adapter position routing preserves absolute positions; failure atomicity prevents state, counter, or peak history mutation on invalid/out-of-order/replayed/budget-rejected updates; layout, dtype, and device mismatches rejected; coexisting peak memory ($\ge 7,168$ bytes), budget rejection atomicity, multi-layer preflight allowance atomicity, accelerator allocation isolation, duplicates, and batch size $> 1$ strictly verified.
6. **Check 6: Exact memory accounting:** Factual resident payload and analytical coexisting peak memory verified bit-for-bit.
7. **Check 7: Full-cache baseline parity:** 100% greedy token agreement and zero logit divergence against Stage 05 reference.
8. **Check 8: Bounded FIFO generation & reconciliation:** $54$ tokens seen, $10$ resident, $44$ evictions ($54 - 10 = 44$).
9. **Check 9: Multi-token chunked prefill equivalence:** Chunked prefill logits match single-pass prefill within $3.27 \times 10^{-5}$.
10. **Check 10: Contract v2 benchmark emission:** Validated against `https://qedge.local/schemas/v2/benchmark_result.schema.json`.

---

## 8. Limitations, Claim Boundaries, and Next Prerequisites

1. **Strictly an Attention Correctness Gate:**
   Stage 06 establishes positional and memory correctness for bounded KV caching. It does **not** assert speedup, throughput advantages, or downstream quality ranking for bounded policies over the baseline.
2. **Model Scope:**
   Bounded strictly to `HuggingFaceTB/SmolLM2-135M-Instruct`. The secondary model (`Qwen/Qwen2.5-0.5B-Instruct`) remains deferred.
3. **Hardware Boundary:**
   Verified on CPU and Apple Silicon MPS reference execution. No Qualcomm QNN or Snapdragon NPU offload is claimed.
4. **Next Prerequisites:**
   Proceed to **Stage 07: `07-attention-selectors`** only after independent review acceptance of this Stage 06 handoff.
