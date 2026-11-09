# QEdge KV-Cache Position & Memory Contract

**Stage:** `06-cache-contract`  
**Status:** SPECIFICATION & IMPLEMENTATION CONTRACT  
**Model Target:** `HuggingFaceTB/SmolLM2-135M-Instruct` (30 layers, 9 Q heads, 3 KV heads, head dimension 64)  
**Applicable Transformers Versions:** Verified specifically on pinned `transformers == 5.22.1` via the tested `Cache` adapter interface.

---

## 1. Identity, Positions, and Addressing

A fundamental defect in ad-hoc KV cache compression is conflating local tensor indices with absolute sequence positions.

### 1.1 Core Coordinate Definitions
1. **Total Tokens Seen ($T_{\text{seen}}$):**
   A strictly monotonically increasing integer counter representing the cumulative number of sequence tokens processed by the cache since sequence inception ($T_{\text{seen}} \ge 0$).
2. **Absolute Token Position ($P_{\text{abs}} \in \mathbb{N}_0$):**
   The invariant sequence index of a token within the original sequence ($0, 1, 2, \dots$). Absolute positions are immutable: once assigned to a Key/Value pair, $P_{\text{abs}}$ never changes regardless of subsequent gathers, compactions, or evictions.
3. **Resident Index ($i_{\text{res}} \in [0, S_{\text{resident}}-1]$):**
   The physical position of a Key/Value vector inside the currently allocated cache tensor along sequence dimension 2. When an interior token is evicted, subsequent tokens shift their resident index ($i \to i-1$), but preserve their absolute position $P_{\text{abs}}$.
4. **Absolute Position Cursor ($P_{\text{cursor}}$):**
   The highest absolute position observed so far ($P_{\text{cursor}} = \max(P_{\text{abs}})$). It is strictly distinct from the resident count $S_{\text{resident}}$ and cumulative token count $T_{\text{seen}}$. An incoming sequence must begin at $P_{\text{incoming}} > P_{\text{cursor}}$. Replaying previously evicted positions ($P \le P_{\text{cursor}}$) without a clean `reset()` is strictly prohibited.

### 1.2 Mathematical Invariants
At all times across all layers:
$$\forall i \in [0, S_{\text{resident}} - 2]: \quad P_{\text{abs}}[i] < P_{\text{abs}}[i+1]$$
$$\forall i \in [0, S_{\text{resident}} - 1]: \quad 0 \le P_{\text{abs}}[i] \le P_{\text{cursor}}$$
$$S_{\text{resident}} \le C \quad (\text{where } C \text{ is declared capacity})$$
$$T_{\text{seen}} - S_{\text{resident}} = N_{\text{evictions}}$$

### 1.3 Validation Precedence and Failure Atomicity
All input validations must occur **prior** to any state mutation:
- Tensor rank, batch size, heads, head dimension, dtype, and device must be validated against the cache configuration and resident tensors before allocating or updating.
- Position sequences must be non-negative integers (`torch.long`), strictly increasing within the incoming batch, and strictly greater than $P_{\text{cursor}}$.
- If any validation fails, the cache raises an appropriate `ValueError` or `TypeError` and leaves resident tensors, positions, counters, and statistics **completely unmodified**.

### 1.4 Adapter Position Channel
The adapter (`QEdgeTransformersCacheAdapter`) routes explicit position metadata:
1. Inspects `cache_kwargs["cache_position"]`, `kwargs["cache_position"]`, `kwargs["position_ids"]`, or `kwargs["query_positions"]`.
2. Supports an explicit caller channel (`adapter.set_active_positions(positions)` / `adapter.active_positions_context(positions)`) for caller architectures where Transformers does not forward positions to `self_attn`.
3. If no explicit positions are supplied, the cache advances from $P_{\text{cursor}} + 1$. The adapter never silently substitutes resident indices for absolute positions.

---

## 2. Permanent Eviction vs. Attention Selection

- **Permanent Eviction:** Irrevocable physical removal of a token's Key and Value vectors from memory buffers to satisfy a hardware or budget bound ($S_{\text{resident}} \le C$). Evicted vectors are released; $N_{\text{evictions}}$ increments permanently.
- **Attention Selection (Temporary View):** Dynamic query-conditioned masking or sub-viewing over resident keys during self-attention computation, leaving buffer occupancy unchanged. (Belongs strictly to Stage 07 selectors).

---

## 3. Tensor Layout, GQA Alignment, and Batch Policy

### 3.1 Tensor Layout Contract
Key and Value tensors must strictly conform to:
$$K, V \in \mathbb{R}^{B \times H_{\text{kv}} \times S_{\text{resident}} \times D}$$
- **Rank:** Exactly 4 dimensions.
- **Batch Size ($B = 1$):** Batch size is strictly one. Multi-batch tensors ($B > 1$) are rejected with `ValueError` to prevent cross-sequence eviction interference.
- **KV Heads ($H_{\text{kv}}$):** Must strictly match the configured cache KV heads ($H_{\text{kv}} = 3$ for SmolLM2-135M).
- **Head Dimension ($D$):** Must strictly match configured head dimension ($D = 64$ for SmolLM2-135M).
- **Dtype & Device:** Incoming tensors must match configured cache dtype (`float32`, `bfloat16`, `float16`) and resident device. Mismatched dtypes or devices raise `TypeError` / `ValueError`.
- **GQA Alignment:** SmolLM2 maps $H_q = 9$ query heads to $H_{\text{kv}} = 3$ KV heads (group ratio 3). Cache stores 3 KV heads; expansion to 9 heads occurs per attention step.

### 3.2 Non-Contiguous Causal Masking
When tokens are permanently evicted, resident keys have non-contiguous absolute positions $\vec{P}_{\text{keys}}$.
The causal attention mask between query tokens at positions $\vec{P}_{\text{queries}}$ and resident keys $\vec{P}_{\text{keys}}$ is given by:
$$M_{\text{causal}}[r, c] = \begin{cases} 0.0 & \text{if } P_{\text{queries}}[r] \ge P_{\text{keys}}[c] \\ -\infty & \text{if } P_{\text{queries}}[r] < P_{\text{keys}}[c] \end{cases}$$
Future tokens ($P_{\text{keys}}[c] > P_{\text{queries}}[r]$) are suppressed with $-\infty$, guaranteeing an exact $0.0$ attention weight after Softmax.

### 3.3 Rotary Position Embeddings (RoPE) Invariance
RoPE encodes absolute positions into $Q$ and $K$ prior to attention:
$$Q_m = R(m) Q, \quad K_n = R(n) K$$
$$Q_m^T K_n = Q^T R(m - n) K$$
Key vectors stored in the cache have rotary position $n = P_{\text{abs}}$ embedded at creation time. Because $R(n)$ depends strictly on the absolute sequence position $n$ and never on the resident index $i$, evicting interior tokens leaves the relative rotary distance $m - n$ intact. Compacting resident indices must **never** re-encode or modify rotary coordinates.

---

## 4. Dense Attention and Classical LRU Degeneracy

In classical CPU cache architectures, access is sparse: only requested cache lines are touched.

In dense transformer self-attention:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{D}}\right) V$$
where $Q \in \mathbb{R}^{B \times H \times 1 \times D}$ and $K \in \mathbb{R}^{B \times H \times S \times D}$.
Every single resident key participates in the inner product $Q K^T$ on **every** generation step. If "access" is defined as participation in attention:
1. Every resident token is accessed simultaneously at every step.
2. Classical LRU access timestamps increment identically for all resident tokens.
3. Classical LRU degenerates to FIFO order or tie-breaking arbitrariness.

### Baseline Policy Definitions
To provide mathematically sound, non-degenerate baselines without relabeling attention weights as LRU:
1. **`full`:** Unbounded retention ($C = \infty$, $N_{\text{evictions}} = 0$).
2. **`fifo`:** First-in, first-out permanent eviction. The oldest non-sink tokens by absolute position are evicted first when capacity $C$ is reached.
3. **`recency_window`:** Explicit sliding-window arrival recency. Protects $S_{\text{sink}}$ initial attention sink tokens and retains the $W = C - S_{\text{sink}}$ most recently arrived sequence tokens.

---

## 5. Memory Accounting & Buffer Budgets

### 5.1 Factual Resident Payload Accounting
Resident memory is derived directly from live tensor allocations across all $L$ layers:
$$M_{\text{resident}} = \sum_{l=0}^{L-1} \left( \text{numel}(K_l) \times \text{element\_size}(K_l) + \text{numel}(V_l) \times \text{element\_size}(V_l) \right)$$
For a uniform cache of $S$ tokens:
$$M_{\text{resident}} = 2 \times L \times H_{\text{kv}} \times S \times D \times \text{element\_size}$$
- For SmolLM2 (`float32`, $L=30, H_{\text{kv}}=3, D=64$):
  $$B_{\text{model}} = 2 \times 30 \times 3 \times 64 \times 4 = 46,080 \text{ bytes/token}$$
  At $S=64$, $M_{\text{resident}} = 2,949,120 \text{ bytes}$.

### 5.2 Allocation-Aware Peak Temporary Memory Bound
During an allocating operation on layer $l$, the following buffers coexist momentarily before Python garbage collection:
1. Prior resident buffers: $K_{\text{old}}, V_{\text{old}}$ ($S_{\text{old}}$ tokens).
2. Incoming buffers (in `update`): $K_{\text{new}}, V_{\text{new}}$ ($S_{\text{new}}$ tokens).
3. Concatenation buffers (in `update`): $K_{\text{cat}}, V_{\text{cat}}$ ($S_{\text{old}} + S_{\text{new}}$ tokens).
4. Gathered buffers (during eviction/pruning): $K_{\text{resident}}, V_{\text{resident}}$ ($S_{\text{retained}}$ tokens).
5. Index and position tensors: $(S_{\text{old}} + S_{\text{new}} + 2 \times S_{\text{retained}}) \times 8 \text{ bytes}$ plus control counters ($64 \text{ bytes}$).

For layer $l$ updating from capacity $C=2$ with $S_{\text{new}}=1$ ($D=128, H_{\text{kv}}=1, L=1$, FP32):
- Old resident: $2 \times 1,024 = 2,048$ bytes.
- Incoming: $1 \times 1,024 = 1,024$ bytes.
- Concatenation: $3 \times 1,024 = 3,072$ bytes.
- Gathered: $2 \times 1,024 = 2,048$ bytes.
- Coexisting K/V payload alone: $2,048 + 1,024 + 3,072 + 2,048 = 8,192$ bytes (or $7,168$ bytes excluding caller's incoming handle).

Across the entire model during layer $l$'s update, the remaining $L-1$ layers hold their resident tensors:
$$M_{\text{peak\_analytical}} = \sum_{j \ne l} M_{\text{resident}}(j) + M_{\text{coexisting}}(l) + M_{\text{metadata\_peak}}$$

#### Included Buffers vs Documented Excluded Scopes
- **Included in Analytical Scope:** Coexisting K and V tensors (old resident, incoming, concatenated, gathered), position tensors across all layers, and index tensors allocated for `torch.index_select`.
- **Documented Exclusions:** Dynamic attention masks ($[1, 1, S_q, S_k]$) created in the attention caller per query step (not retained by cache storage), Python interpreter object overhead (PyObject headers, frame stacks), and underlying hardware allocator rounding/pooling (e.g., MPS/CUDA page rounding).

### 5.3 Authoritative Analytical Accounting vs Allocator Snapshots
1. **Authoritative Cache Bounds:** `analytical_peak_bytes` and `peak_temporary_bytes` represent the mathematically verified, allocation-aware temporary peak memory of the cache.
2. **Global Allocator Snapshots (`allocator_snapshot`):** Accelerators like Apple MPS (`torch.mps.current_allocated_memory()`) and CUDA (`torch.cuda.memory_allocated()`) expose process-wide global memory snapshots. These reflect all active allocations across the entire process (including unrelated model weights, embeddings, or external tensors) and instantaneous sampling points, rather than cache-isolated historical peaks. Such snapshots are captured and reported separately with metadata (`backend`, `scope="global_process"`, `sampling_point="post_update"`). They must **never** overwrite or replace authoritative cache bounds.
3. **Scoped Measured Peaks (`measured_peak_bytes`):** Defaults to `null` (`None`). Only populated when an explicit, isolated, cache-scoped memory measurement is performed.

### 5.4 Preflight Validation & Failure-Atomic Budget Rejection
All allocating operations—`update()`, `evict_to_capacity()`, and `evict_positions()`—enforce strict failure atomicity with preflight allowance validation:
1. **Preflight Candidate Calculation:** The operation computes the complete candidate coexisting peak memory before modifying any tensors, positions, or counters. Multi-layer pruning (`evict_to_capacity`) and multi-layer eviction (`evict_positions`) preflight all layers sequentially to find the global candidate peak.
2. **Allowance Enforcement:** If `temporary_memory_allowance_bytes` is defined and `candidate_peak > allowance`, a `MemoryError` is raised immediately.
3. **Zero Mutation Guarantee:** Rejected operations leave resident tensors, position arrays, tokens seen counters, eviction counters, and historical peak telemetry (`analytical_peak_bytes`, `peak_temporary_bytes`) 100% unchanged. Committed `temporary_memory_overflow` remains `False`.
4. **Diagnostic Separation:** Rejected-attempt diagnostics are captured separately in `last_rejected_attempt` (recording operation, candidate peak bytes, and allowance) without corrupting committed telemetry.

### 5.5 Prefill Overflow Policy
During initial prompt prefill ($S_{\text{prompt}} > C$), self-attention across the full prompt is computed to preserve causal correctness. Transient peak memory reaches $S_{\text{prompt}}$ tokens before post-prefill eviction prunes resident occupancy to capacity $C$. The factual prefill peak is explicitly reported in telemetry.
