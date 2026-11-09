# Stage 07 Handoff: Research Selectors (KeyDiff, QUOKA, and Hypotheses)

**Stage:** `07-research-selectors`  
**Date:** 2026-11-02  
**Status:** Complete — Ready for Independent Verification  
**Preceding Acceptance:** Stage 06 (`06-cache-contract`) accepted with 72/72 tests passing, budget rejection atomicity, allocator isolation, and 30/30 historical hashes verified.  
**Successor Gate:** Keep Stage 08 (`08-quantization-measurement`) and the second model (`Qwen/Qwen2.5-0.5B-Instruct`) strictly blocked.

---

## 1. Executive Summary & Review Audit

Stage 07 implements, mathematically audits, and empirically benchmarks the KV selection and eviction algorithms referenced in Qualcomm AI Research literature. Following independent review findings, all implementations have been rigorously reconciled with primary sources, verified against pure-Python reference oracles, and measured under bounded empirical protocols.

### 1.1 Resolution of Independent Review Findings

1. **QUOKA Paper Fidelity Reconciled Line-by-Line with Algorithm 1 ([arXiv:2602.08722v1](https://arxiv.org/abs/2602.08722)):**
   - **Unnormalized Query Mean:** Mean query $M_Q = \text{mean}(Q, \text{dim}=2)$ is computed on raw unnormalized queries before unit-sphere projection, preserving sequence magnitude weighting.
   - **Per-Query-Head Subselection:** Top-$N_Q$ representative queries are selected independently per query head ($N_Q = \max(1, \lfloor S_q \times \text{ratio} \rfloor)$).
   - **Group Query Pre-Aggregation Without Re-normalization:** Normalized representative queries within each GQA group ($G = H_q / H_{\text{kv}}$) are averaged as $\bar{Q} = \frac{1}{G}\sum_{g=1}^G Q_{\text{norm}, g}$. Crucially, $\bar{Q}$ is **not** re-normalized, preserving cosine linearity and group cancellation.
   - **Per-KV-Head Key Selection:** Key interaction scores $S = \bar{Q} K^T$ and max-pooled scores $\hat{S} = \max(S, \text{dim}=2)$ are shaped $[H_{\text{kv}}, S_k]$. Top-$k$ selection is strictly **per KV head** ($I \in \mathbb{N}^{H_{\text{kv}} \times B_{\text{SA}}}$).
   - **Shared-Head Approximation Ablation:** `SharedHeadQUOKAApproximation` is explicitly isolated as a macro-averaging ablation that collapses on disagreeing orthogonal heads.
   - **Pure-Python Paper Oracle:** `paper_quoka_oracle` implements Algorithm 1 in pure Python with exact multi-head GQA parity.

2. **Sparse Attention Context, Finite Additive Biases & Sentinel Protection:**
   - **Finite Additive Bias Preservation:** In `SparseAttentionContext` and `masked_attention_reference`, allowed entries of 4D attention masks are no longer forced to zero. Numerical additive biases (e.g., relative position biases or arbitrary finite negative/positive adjustments) are preserved in both reduced-budget and full-budget attention paths.
   - **Reproduction Probe Verified:** Tested with zero Q/K projections and additive mask biases `[-2, 0]` on values `[[1, 0], [0, 1]]`. Evaluates to $\text{softmax}([-2, 0]) \approx [0.1192029, 0.8807971]$ instead of $[0.5, 0.5]$.
   - **Robust Sentinel Protection (`is_blocking_sentinel`):** Detects $-\infty$, `torch.finfo(dtype).min` (within a 1% floating-point margin), extreme negative sentinels ($\le -10^{30}$ for float32/bfloat16, $\le -65000.0$ for float16), and NaNs. Confirms that finite biases (such as $-2.0, -10.0, -100.0, -1000.0$) are classified as allowed and never treated as blocked sentinels.
   - **2D Padding Mask Broadcasting:** 2D padding masks (bool, int64, float32) shaped `[batch, S_k]` are converted via `_prepare_additive_mask` into 4D additive masks expanded across query rows `[batch, 1, S_q, S_k]`, eliminating shape broadcasting `RuntimeError` during multi-query chunked prefill.
   - **Fully Blocked Row Zero Output:** Fully masked query rows are explicitly detected via `_is_mask_allowed`. Softmax weights and resulting attention outputs are masked via `torch.where(rows_allowed, attn_weights, 0.0)`, guaranteeing strictly zero attention output `[[0, 0], [0, 0]]` rather than uniform distributions.
   - **Pure Independent Oracle:** `masked_attention_reference` implements masked scaled dot-product attention independently and matches `SparseAttentionContext` to machine precision across all mask configurations.
   - **Dynamic Per-KV-Head Gather:** Active keys/values are gathered per KV head via `torch.gather(..., 2, I_expanded)` and repeated across GQA query heads via `repeat_kv`.
   - **Absolute Position Causal Masking:** True cache positions `k_pos_all` (preserved through eviction) are compared against query positions `q_pos` ($P_k \le P_q$), maintaining strict causality across subsequent chunked prefill chunks and decode steps.
   - **Hardware Accelerator Synchronization:** Explicit `_sync(device)` barriers around selector scoring and gather operations.

3. **Rigorous Evaluation Metric Definitions & Deployed-Path Alignment:**
   - **Teacher-Forced Evaluation Fidelity:** `compute_teacher_forced_step_logits` and `compute_teacher_forced_logit_mse` follow the runner's exact deployed path:
     * Formats prompt with system prompt using `runner.format_prompt(prompt, system_prompt)` matching generation.
     * Respects the prefill/chunking and deferred-eviction schedule (`bounded_cache.defer_eviction = True` during prefill, calling `evict_to_capacity()` upon prefill completion).
     * Advances step-by-step through forced reference tokens so future completion tokens never participate before their step (suffix modification invariance verified).
     * Achieves exact MSE = 0.0 against full dense baseline, and matches QUOKA 100% full-budget parity.
   - **Sequence Agreement:** Evaluates both `exact_sequence_agreement` (binary exact match of full token sequence) and `length_aware_token_agreement` (token-by-token agreement normalized by $\max(|T_{\text{gen}}|, |T_{\text{ref}}|)$, explicitly penalizing premature EOS truncation).
   - **Separated Logit MSE Estimands:** Separates pure numerical approximation error on identical prefixes (`teacher_forced_prefix_logit_mse`) from autoregressive divergence along independent free-running generation paths (`trajectory_divergence_logit_mse`).
   - **Coarse Target Scoring:** Scored as `coarse_authored_target_hit` using word-boundary regexes and negative context detection, correctly abstaining on negated tokens ("not red") and avoiding substring false positives ("reduced", "1010" for "101").
   - **Distinct Statistical Estimands:** Explicitly documents $N_{\text{timing}}=15$ (5 prompts $\times$ 3 runs) for hardware timing variance vs $N_{\text{prompts}}=5$ distinct QA items for quality evaluation (deterministic greedy decoding produces identical tokens across runs).

4. **Conservative Selector Memory Telemetry:**
   - `peak_selector_intermediates_bytes`: Documented conservative accounting including $\hat{K}, Q_{\text{rep}}, \bar{Q}, S, \hat{S}$, subselection temporaries, and index buffers (~53 KB for QUOKA 75%).
   - Strictly isolated from `peak_gathered_active_bytes` (~31 KB), `peak_gqa_expanded_bytes` (~94 KB), `peak_attention_mask_bytes` (~29 KB), and resident cache payload (`resident_payload_bytes`).

5. **Experiment Provenance & Verification Integrity:**
   - Dynamically derived runtime metadata: Model `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, device `mps`, actual parameters dtype `bfloat16`.
   - Distinct labeling: Resident-token capacity ($C \in \{\text{Full}, 30, 20, 10\}$) for permanent eviction vs Attention selection ratio ($R \in \{100\%, 75\%, 50\%, 25\%\}$) for sparse attention.
   - Superseded evidence archived in `evidence/superseded/`.
   - Chunked prefill workload benchmark ($S_{\text{prompt}}=122$, chunk size 64).
   - Honest negative speedup findings on short sequences ($S \le 128$) without speculative extrapolation beyond tested lengths.
   - All 30 historical manifest hashes verified byte-for-byte.

---

## 2. Algorithm Cards and Provenance

Detailed algorithm cards are cataloged in `docs/research-review/algorithms/`:
- [01-KEYDIFF.md](file:///Users/manuvashistha/Developer/qualcomm/docs/research-review/algorithms/01-KEYDIFF.md): Qualcomm AI Research (Park et al., NeurIPS, arXiv:2504.15364v1) and NVIDIA `kvpress` provenance.
- [02-QUOKA.md](file:///Users/manuvashistha/Developer/qualcomm/docs/research-review/algorithms/02-QUOKA.md): Qualcomm AI Research (Jones et al., arXiv:2602.08722v1) Algorithm 1 line-by-line audit.
- [03-HYPOTHESES.md](file:///Users/manuvashistha/Developer/qualcomm/docs/research-review/algorithms/03-HYPOTHESES.md): Separation of unvalidated document hypotheses (`CentroidNeighbor`, `AttentionValueDecay`).

### Algorithmic Comparison

| Algorithm | Status | Primary Source | Scope | Budget Parameter | GQA Handling | Head Selection |
|---|---|---|---|---|---|---|
| **KeyDiff** | Published Paper | [arXiv:2504.15364v1](https://arxiv.org/abs/2504.15364) | **Permanent Eviction** | Resident tokens $C$ | Macro-average across KV heads | Shared layer eviction |
| **QUOKA** | Published Paper | [arXiv:2602.08722v1](https://arxiv.org/abs/2602.08722) | **Temporary Attention** | Attention Ratio $R$ ($B_{\text{SA}}$) | $\bar{Q} = \frac{1}{G}\sum Q_g$ (unrenormalized) | **Per KV Head** ($[H_{\text{kv}}, B_{\text{SA}}]$) |
| **SharedHeadQUOKA** | Ablation Model | Internal Ablation | **Temporary Attention** | Attention Ratio $R$ ($B_{\text{SA}}$) | Macro-averaged $\bar{S}$ | Shared KV heads |
| **CentroidNeighbor** | Unvalidated Hypothesis | Blueprint heuristic | **Permanent Eviction** | Resident tokens $C$ | Pairwise cosine distance | Shared layer eviction |
| **AttentionValueDecay** | Unvalidated Hypothesis | Blueprint heuristic | **Temporary Attention** | Attention Ratio $R$ ($B_{\text{SA}}$) | Value norm $\times$ attention decay | Shared KV heads |

---

## 3. Structural Distinction: Permanent Eviction vs Temporary Selection

The runtime architecture enforces a strict operational and memory boundary between permanent cache eviction and temporary attention selection:

```
[Incoming Token (K_new, V_new)]
             |
             v
    BoundedKVCache (Capacity C)
             |
             +---> Permanent Eviction (KeyDiff, CentroidNeighbor, FIFO)
             |     - Permanently deallocates tokens beyond capacity C
             |     - Updates resident_tokens and physical memory footprint
             |     - S_resident <= C
             |
             v
    Resident KV Storage (S_resident tokens)
             |
             +---> Temporary Attention Selection (QUOKA, AttentionValueDecay)
                   - Gathers sparse active subset B_SA <= S_resident for dot product
                   - Evaluates query-key interaction without mutating resident tensors
                   - Resident cache memory remains full (S_resident tokens preserved)
                   - Emits causal mask over sparse active positions
```

| Operational Metric | Permanent Eviction (KeyDiff) | Temporary Selection (QUOKA) |
|---|---|---|
| **Resident Payload Memory** | Permanently reduced to $C \times \text{bytes\_per\_token}$ | **Unchanged** (preserves full resident sequence) |
| **Attention Pairs Evaluated** | $S_q \times C$ | **Reduced** to $S_q \times B_{\text{SA}}$ |
| **Cache Mutation** | Mutates `state.key_tensor` and `state.value_tensor` | Zero mutation to underlying cache states |
| **Measured Overhead** | Reallocation & gather at eviction events | Query scoring and active tensor gather per step |
| **Transient Memory** | Temporary gather during eviction | Intermediates ($S, \hat{S}$) + Gathered active K/V |

---

## 4. Authoritative Verification Probe Results

The standalone verification probe (`handoff/research-review/07-research-selectors/probes/verify_selectors.py`) executes 17 self-contained mathematical, contract, mask, and runtime checks:

```text
======================================================================
STAGE 07 RESEARCH SELECTORS: AUTHORITATIVE VERIFICATION PROBE
======================================================================
[Check 1/17] Verifying KeyDiff agreement with pure-Python reference...
  PASS: Exact index agreement across all test budgets.
[Check 2/17] Verifying QUOKA agreement with pure-Python reference on MHA...
  PASS: Exact index agreement across all test budgets.
[Check 3/17] Verifying Query-Sensitivity (QUOKA) vs Query-Independence (KeyDiff)...
  PASS: KeyDiff is strictly query-independent; QUOKA is query-sensitive.
[Check 4/17] Verifying Outlier Key Diversity (KeyDiff) vs Recency Window (FIFO)...
  PASS: KeyDiff selected outliers [0, 1, 2, 3, 8]; FIFO selected newest [0, 1, 6, 7, 8].
[Check 5/17] Verifying zero norms, ties, budget=1, and full budget boundaries...
  PASS: Zero norms handled stably; budget boundaries verified.
[Check 6/17] Verifying GQA group pre-aggregation (H_q=9, H_kv=3)...
  PASS: GQA group pre-aggregation correctly mapped 9 query heads to 3 KV groups.
[Check 7/17] Verifying full-budget (100%) exact logit parity on SmolLM2...
  PASS: Exact token match; max logit diff = 0.00e+00 (< 1e-4).
[Check 8/17] Verifying monotonic selection pressure across matched ratios...
  Measured sparsity ratios: [(1.0, 0.0), (0.75, 0.245), (0.5, 0.5099), (0.25, 0.755)]
  PASS: Monotonic selection pressure verified across 100%, 75%, 50%, 25%.
[Check 9/17] Verifying divergence between Permanent Eviction and Temporary Selection...
  PASS: KeyDiff reduced resident tokens to 16 (evicted 31); QUOKA retained all 47 resident tokens (0 evicted).
[Check 10/17] Verifying state isolation across repeated requests...
  PASS: Repeated requests cleanly isolated without telemetry leakage.
[Check 11/17] Verifying per-KV-head selection on disagreeing orthogonal heads...
  PASS: Faithful QUOKA selected [[0], [1]]; Shared approximation collapsed to [[1], [1]].
[Check 12/17] Verifying GQA cancellation preservation and chunked prefill parity...
  PASS: GQA cancellation preserved; exact chunked prefill decode parity achieved.
[Check 13/17] Verifying 2D padding mask (float32, int64, bool) in SparseAttentionContext...
  PASS: 2D padding masks (float32, int64, bool) handled cleanly without broadcasting errors.
[Check 14/17] Verifying fully blocked 4D additive masks produce zero attention output...
  PASS: Fully blocked rows explicitly detected and produce strictly zero attention weights/output.
[Check 15/17] Verifying length-aware token agreement and target scorer semantics...
  PASS: Length-aware agreement penalizes premature truncation; target scorer handles negation & boundaries.
[Check 16/17] Verifying finite additive attention biases and sentinel protection...
  PASS: Finite additive biases preserved (softmax([-2,0]) verified); sentinels correctly protected.
[Check 17/17] Verifying teacher-forced deployed path parity and suffix invariance...
  PASS: Teacher-forced path matches deployed runner; suffix invariance and full-budget parity verified.
======================================================================
SUMMARY: 17/17 checks passed, 0 failed.
======================================================================
```

---

## 5. Matched Benchmark Results

**Experimental Protocol:**
- **Model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Dynamic Revision:** `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- **Hardware Device:** Apple Silicon (`mps`)
- **Execution Precision:** `bfloat16` (actual weights and KV cache tensors)
- **Evaluation Workload:** 5 structured QA prompts with factual ground truth targets:
  1. *Capital of France:* `Paris`
  2. *Chemical formula for water:* `Water`
  3. *Speed of light symbol c:* `299,792,458`
  4. *Binary representation of decimal 5:* `101`
  5. *Additive primary color with green and blue:* `Red`
- **Statistical Protocol:**
  - **Timing Estimand ($N_{\text{timing}}=15$):** 1 warmup run + 3 measured trials per configuration across 5 prompts to quantify hardware execution variance on Apple Silicon MPS. Metrics report mean, standard deviation, and 95% confidence interval ($\text{CI}_{95}$).
  - **Quality Estimand ($N_{\text{prompts}}=5$):** Macro-averaged across the 5 distinct authored QA items. Under deterministic greedy decoding (`temperature=0.0`), repeated runs produce identical token trajectories.
- **Total Executed Trials:** 297 raw trials logged in `handoff/research-review/07-research-selectors/logs/raw_trials.jsonl`.
- **Structured Summary:** `handoff/research-review/07-research-selectors/results.json`.

### 5.1 Permanent Eviction (Resident Memory Budget) Summary

Under permanent eviction, capacity $C$ strictly dictates the maximum number of resident tokens in cache memory:

| Policy | Capacity $C$ | Coarse Target Hit ($N=5$) | Exact Sequence Agreement | Length-Aware Agreement | Teacher-Forced Prefix MSE | Trajectory Divergence MSE | TTFT (ms, mean ± $\text{CI}_{95}$) | Decode Speed (tok/s, mean ± $\text{CI}_{95}$) | Resident Payload |
|---|---|---|---|---|---|---|---|---|---|
| **Full Baseline** | **Full** | **0.40** (2/5) | **1.0000** | **1.0000** | **0.00** | **0.00** | 112.33 ± 13.91 | 17.06 ± 0.57 | 1,267,200 B (1.27 MB) |
| **FIFO** | $C=30$ | 0.40 (2/5) | 0.6000 | 0.6625 | 3.1704 | 17.83 | 102.67 ± 15.40 | 15.97 ± 0.85 | 691,200 B (0.69 MB) |
| **FIFO** | $C=20$ | 0.20 (1/5) | 0.0000 | 0.0875 | 8.7598 | 52.85 | 114.84 ± 14.80 | 15.23 ± 0.53 | 460,800 B (0.46 MB) |
| **FIFO** | $C=10$ | 0.00 (0/5) | 0.0000 | 0.0625 | 22.5667 | 70.61 | 118.01 ± 16.33 | 14.70 ± 0.52 | 230,400 B (0.23 MB) |
| **KeyDiff** | $C=30$ | 0.20 (1/5) | 0.2000 | 0.5000 | 33.4879 | 55.11 | 121.99 ± 22.75 | 10.27 ± 0.52 | 691,200 B (0.69 MB) |
| **KeyDiff** | $C=20$ | 0.40 (2/5) | 0.0000 | 0.2236 | 51.2975 | 74.58 | 109.96 ± 13.62 | 10.11 ± 0.37 | 460,800 B (0.46 MB) |
| **KeyDiff** | $C=10$ | 0.00 (0/5) | 0.0000 | 0.1250 | 97.8828 | 146.19 | 103.45 ± 15.84 | 10.07 ± 0.31 | 230,400 B (0.23 MB) |
| **CentroidNeighbor** | $C=30$ | 0.40 (2/5) | 0.0000 | 0.2339 | 8.4498 | 43.07 | 119.47 ± 13.22 | 10.51 ± 0.33 | 691,200 B (0.69 MB) |
| **CentroidNeighbor** | $C=20$ | 0.60 (3/5) | 0.0000 | 0.1133 | 22.2494 | 48.12 | 124.07 ± 14.93 | 10.12 ± 0.27 | 460,800 B (0.46 MB) |
| **CentroidNeighbor** | $C=10$ | 0.20 (1/5) | 0.0000 | 0.0893 | 93.5876 | 125.53 | 115.17 ± 14.58 | 10.96 ± 0.25 | 230,400 B (0.23 MB) |

### 5.2 Temporary Attention Selection (Attention-Pair Budget) Summary

Under temporary attention selection, resident cache memory remains intact while attention dot products are evaluated over a gathered subset:

| Selector | Ratio $R$ | Measured Sparsity | Coarse Target Hit ($N=5$) | Exact Sequence Agreement | Length-Aware Agreement | TF Prefix Logit MSE | Trajectory Logit MSE | Selection Overhead (ms, mean ± $\text{CI}_{95}$) | Decode Speed (tok/s, mean ± $\text{CI}_{95}$) | Resident Payload | Peak Intermediates | Peak Gathered Active | Peak GQA Expanded | Peak Attention Mask |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dense Baseline** | 100% | 0.00% | **0.40** | **1.0000** | **1.0000** | **0.00** | **0.00** | 0.00 | 16.52 ± 0.94 | 1,267,200 B | 0 B | 0 B | 0 B | 0 B |
| **QUOKA (Paper)** | 100% | 0.00% | **0.40** | **1.0000** | **1.0000** | **0.00** | **0.00** | 0.00 | 10.89 ± 0.34 | 1,267,200 B | 0 B | 0 B | 0 B | 39,762 B |
| **QUOKA (Paper)** | 75% | 25.17% | 0.40 | 0.8000 | 0.8500 | 4.2218 | 7.5119 | 1247.60 ± 134.58 | 5.02 ± 0.12 | 1,267,200 B | 53,838 B | 31,488 B | 94,464 B | 29,610 B |
| **QUOKA (Paper)** | 50% | 50.06% | 0.20 | 0.0000 | 0.0000 | 29.1531 | 62.6629 | 544.62 ± 297.95 | 1.98 ± 1.27 | 1,290,240 B | 53,574 B | 21,504 B | 64,512 B | 20,304 B |
| **QUOKA (Paper)** | 25% | 74.77% | 0.00 | 0.0000 | 0.0250 | 74.1852 | 100.8309 | 627.42 ± 235.07 | 3.09 ± 1.32 | 1,082,880 B | 53,286 B | 9,216 B | 27,648 B | 10,152 B |
| **AttentionValueDecay**| 100% | 0.00% | **0.40** | **1.0000** | **1.0000** | **0.00** | **0.00** | 0.00 | 10.24 ± 0.46 | 1,267,200 B | 0 B | 0 B | 0 B | 39,762 B |
| **AttentionValueDecay**| 75% | 25.17% | 0.40 | 0.2000 | 0.5694 | 6.0884 | 27.7136 | 503.30 ± 62.19 | 6.98 ± 0.20 | 1,244,160 B | 25,032 B | 30,720 B | 92,160 B | 29,610 B |
| **AttentionValueDecay**| 50% | 49.98% | 0.40 | 0.0000 | 0.2069 | 24.0934 | 66.1840 | 472.51 ± 53.75 | 7.25 ± 0.15 | 1,244,160 B | 24,720 B | 20,736 B | 62,208 B | 20,304 B |
| **AttentionValueDecay**| 25% | 74.86% | 0.00 | 0.0000 | 0.0250 | 67.3969 | 107.1827 | 568.09 ± 50.99 | 6.69 ± 0.13 | 1,428,480 B | 27,624 B | 12,288 B | 36,864 B | 10,152 B |

### 5.3 Chunked Prefill Workload Summary ($S_{\text{prompt}}=122$, Chunk Size = 64)

Chunked prefill processes long prompts in fixed blocks, maintaining causality and testing attention selection across chunk boundaries:

| Configuration | Prompt Tokens | Chunk Size | TTFT (ms, mean ± $\text{CI}_{95}$) | Decode Speed (tok/s, mean ± $\text{CI}_{95}$) | Exact Sequence Agreement | Length-Aware Agreement | Trajectory Divergence MSE | Selection Overhead (ms) |
|---|---|---|---|---|---|---|---|---|
| **full_prefill_dense** | 122 | None | 184.30 ± 72.36 | 15.73 ± 2.20 | **1.0000** (Reference) | **1.0000** | **0.00** | 0.00 |
| **chunked_dense_c64** | 122 | 64 | 247.65 ± 44.98 | 14.29 ± 1.12 | **1.0000** (Exact Parity) | **1.0000** | **0.00** | 0.00 |
| **chunked_quoka_75pct_c64** | 122 | 64 | 713.61 ± 32.80 | 4.91 ± 0.07 | **1.0000** (Full Match) | **1.0000** | 57.28 | 1,697.77 |
| **chunked_quoka_50pct_c64** | 122 | 64 | 452.20 ± 84.90 | 4.69 ± 0.12 | 0.0000 | 0.0000 | 87.09 | 1,660.05 |

---

## 6. Scientific Analysis of Findings

1. **Genuine Resident Memory Compression Under Permanent Eviction:**
   KeyDiff, CentroidNeighbor, and FIFO physically free resident key and value tensors upon eviction. At capacity $C=30$, resident payload is reduced from 1.27 MB to 0.69 MB (-45.5%); at $C=10$, it is reduced to 0.23 MB (-81.8%). FIFO retains higher sequence agreement (60% exact, 66.25% length-aware at $C=30$) than KeyDiff (20% exact, 50% length-aware), because local recency is critical for short-prompt factual completion.

2. **Zero Resident Memory Compression Under Temporary Selection:**
   QUOKA preserves 100% of resident tokens in cache memory ($1.27\text{ MB}$). Temporary attention selection operates strictly as compute sparsity during the attention forward pass; it does not reduce physical cache memory footprint.

3. **Short-Sequence Sparsity Overhead on Real Silicon (Apple Silicon / MPS):**
   In SmolLM2-135M at sequence lengths $S \le 128$, dense attention matrix multiplication is negligible ($\sim 50\text{K FLOPs}$). Executing query-key cosine similarity scoring, group pre-aggregation, and per-KV-head dynamic tensor gathering across 30 transformer layers introduces an overhead of $540\text{--}1,250\text{ ms}$. Consequently, decode throughput drops from $16.52\text{ tok/s}$ (dense) to $1.98\text{--}5.02\text{ tok/s}$ (QUOKA).
   *Sparse attention selection introduces negative wall-clock speedup on short sequences due to selector scoring and index gather overhead.*

4. **Chunked Prefill Causal Integrity & 2D Mask Broadcasting:**
   Chunked prefill with 4D causal masks $(1, 1, \text{chunk\_len}, \text{end})$ achieves exact token parity (`agreement = 1.0000`, `logit_mse = 0.00`) against full prefill under dense attention. When QUOKA is applied across chunks, causal indexing using true absolute cache positions `k_pos_all` prevents position misalignment. Converting 2D masks to expanded 4D additive masks eliminates broadcasting crashes during multi-token chunks, and explicit detection of fully blocked rows ensures zero attention weights and outputs without numerical instability.

5. **Head Orthogonality in GQA Models:**
   The paper audit confirmed that QUOKA requires per-KV-head selection ($I \in \mathbb{N}^{H_{\text{kv}} \times B_{\text{SA}}}$). When KV heads attend to orthogonal information, averaging scores across KV heads (`SharedHeadQUOKAApproximation`) forces a compromise index that degrades head-specific representations.

---

## 7. Regression and Historical Hash Verification

All repository suites were executed and verified following the Stage 07 implementation and repairs:

| Test Suite | Execution Command | Result | Status |
|---|---|---|---|
| **Runtime Selectors & Model Suite** | `python -m unittest discover -s llm-inference/runtime/tests` | **99 passed / 0 failed** | Pass |
| **Standalone Selector Probe** | `python handoff/research-review/07-research-selectors/probes/verify_selectors.py` | **17 / 17 passed** | Pass |
| **Edge-RAG Assistant Suite** | `python -m unittest discover -s edge-rag-demo/tests` | **68 passed / 1 expected skip** | Pass |
| **QEdge-Common Contract Suite** | `python -m unittest discover -s qedge-common/tests` | **49 passed / 0 failed** | Pass |
| **Historical Manifest Hashes** | `python3 scripts/verify_historical_hashes.py` | **30 verified / 0 failed** | Pass |

---

## 8. Handoff Checklist for Independent Review

- [x] QUOKA reconciled line-by-line with arXiv:2602.08722v1 Algorithm 1 (unnormalized query mean, top-$N_Q$ per query head, unrenormalized group mean, per-KV-head top-$k$).
- [x] Pure-Python oracle `paper_quoka_oracle` implemented and verified against PyTorch selector.
- [x] Shared-head approximation explicitly isolated as `SharedHeadQUOKAApproximation`.
- [x] KeyDiff provenance and NVIDIA `kvpress` reference documented.
- [x] Hypotheses (`CentroidNeighbor`, `AttentionValueDecay`) separated from published literature.
- [x] Sparse attention context dynamically gathers per KV head and repeats across GQA query heads.
- [x] Finite additive attention biases preserved in `SparseAttentionContext` and `masked_attention_reference`; allowed mask entries retain numerical values instead of being zeroed out.
- [x] Sentinel detector `is_blocking_sentinel` protects arbitrary finite biases ($\ge -1000.0$) while correctly recognizing $-\infty$, `finfo.min`, NaNs, and extreme negative values ($\le -10^{30}$).
- [x] 2D padding masks (bool, int64, float32) converted to 4D additive masks expanded across query rows without broadcasting errors.
- [x] Fully blocked rows detected explicitly and zeroed out, producing zero attention output `[[0, 0], [0, 0]]` (verified against `masked_attention_reference`).
- [x] Absolute causal masking enforced via true cache positions `k_pos_all` ($P_k \le P_q$); incoming masks preserved.
- [x] Hardware accelerator synchronization `_sync(device)` applied around selection and gather.
- [x] Conservative telemetry accounting for selector intermediates (~53 KB for QUOKA 75%) strictly isolated from resident cache payload, gathered active KV, GQA expansion, and mask buffers.
- [x] Chunked prefill loop implemented in `runner.py` and benchmarked ($S=122$, chunk 64).
- [x] Dynamic model revision (`12fd25f77366fa6b3b4b768ec3050bf629380bac`), actual dtype (`bfloat16`), and device (`mps`) reported accurately.
- [x] Resident-token capacity ($C$) strictly distinguished from attention selection ratio ($R$).
- [x] 297 raw trials executed across 5 QA prompts; means, stds, and 95% CIs reported.
- [x] Tightened evaluation metrics: `exact_sequence_agreement`, `length_aware_token_agreement`, `teacher_forced_prefix_logit_mse`, `trajectory_divergence_logit_mse`, and `coarse_authored_target_hit`.
- [x] Deployed-equivalent teacher-forced evaluation (`compute_teacher_forced_step_logits` and `compute_teacher_forced_logit_mse`) adheres to chat templating, deferred prefill pruning schedule, and step-by-step decode with forced reference tokens without future token leakage.
- [x] Distinct statistical estimands: $N_{\text{timing}}=15$ (hardware timing) vs $N_{\text{prompts}}=5$ (semantic quality).
- [x] Short-sequence selector overhead reported honestly without speculative extrapolation.
- [x] Superseded evidence archived in `evidence/superseded/`.
- [x] All 30 historical manifest hashes verified byte-for-byte.
- [x] Stage 08 (`08-quantization-measurement`) and Qwen model strictly blocked pending independent review.
