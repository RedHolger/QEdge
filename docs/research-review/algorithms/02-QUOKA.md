# Algorithm Card: QUOKA (Faithful Paper Variant)

## 1. Specification & Provenance

- **Algorithm Name:** QUOKA (Query-Oriented KV Selection for Efficient LLM Prefill)
- **Primary Source Authors:** Dalton Jones, Junyoung Park, Matthew J. Morse, Raghavv Goel, Mingu Lee, Christopher M. Lott (Qualcomm AI Research).
- **Publication:** Pinned paper arXiv:2602.08722v1.
- **Paper URL:** [https://arxiv.org/abs/2602.08722](https://arxiv.org/abs/2602.08722)
- **Target Phase:** Chunked prefill & autoregressive generation.
- **Classification:** Training-free, query-dependent **temporary attention-time selection** (sparse attention).

---

## 2. Line-by-Line Audit of Algorithm 1

From arXiv:2602.08722v1, Algorithm 1 (*QUOKA Key-Value Selection*):

```text
Input: Queries Q in R^{b x n_Q x S_q x d}, Keys K in R^{b x n_KV x S_k x d},
       Number of representative queries N_Q, Budget B_SA
Output: Selected key indices I in N^{b x n_KV x B_SA}

1: if S_q > N_Q then
2:     # Step 1: Subselect representative queries per query head
3:     M_Q <- mean(Q, dim=2)                          # Shape: [b, n_Q, 1, d]
4:     S_Q <- CosSim(Q, M_Q)                           # Shape: [b, n_Q, S_q]
5:     Q_rep <- gather(topk(-S_Q, N_Q), Q)             # Shape: [b, n_Q, N_Q, d]
6: else
7:     Q_rep <- Q
8: Q <- Q_rep / norm(Q_rep)                            # Unit sphere L2 normalization
9: K <- K / norm(K)                                    # Unit sphere L2 normalization
10: Q_bar <- mean(Q.reshape(b, n_KV, G, N_Q, d), dim=2) # Pre-aggregation: average across GQA group; DO NOT renormalize!
11: S <- Q_bar @ K^T                                    # Shape: [b, n_KV, N_Q, S_k]
12: S_hat <- max(S, dim=2)                             # Shape: [b, n_KV, S_k] (max-pool across queries)
13: I <- topk(S_hat, B_SA)                             # Shape: [b, n_KV, B_SA] (per KV head!)
```

### Critical Mathematical Invariants:
1. **Query Mean Ordering (Line 3):**
   $M_Q = \text{mean}(Q, \text{dim}=2)$ is computed on the **raw unnormalized queries**. This preserves the sequence's natural magnitude weighting across positions before calculating cosine similarities.
2. **Per-Query-Head Subselection (Lines 4–5):**
   Representative queries are selected per query head ($n_Q$ independent top-$N_Q$ selections), retaining the most diverse queries (lowest cosine similarity to $M_Q$).
3. **Group Query Pre-Aggregation Without Re-normalization (Line 10):**
   Normalized representative queries $Q_{\text{norm}}$ are averaged within each GQA group of size $G = n_Q / n_{\text{KV}}$:
   $$\bar{Q} = \frac{1}{G} \sum_{g=1}^G Q_{\text{norm}, g}$$
   **Crucial:** $\bar{Q}$ is **not** re-normalized. Leaving $\bar{Q}$ unnormalized preserves cosine linearity ($\frac{1}{G}\sum \hat{q}_g \cdot \hat{k} = \bar{Q} \cdot \hat{k}$) and naturally attenuates directions where group query heads cancel each other out.
4. **Per-KV-Head Key Selection (Lines 11–13):**
   The interaction matrix $S = \bar{Q} K^T$ and max-pooled score vector $\hat{S} = \max_{q \in Q_{\text{rep}}} S(q, k)$ are shaped $[n_{\text{KV}}, S_k]$. Top-$k$ selection is strictly **per KV head**:
   $$I \in \mathbb{N}^{n_{\text{KV}} \times B_{\text{SA}}}$$
   Each KV head independently selects its own set of $B_{\text{SA}}$ keys. A two-head system where Head 0 requires token 0 and Head 1 requires token 1 selects index 0 for Head 0 and index 1 for Head 1.

---

## 3. Algorithmic Dimensions & Parameters

| Dimension | Specification | Notes |
|---|---|---|
| **Selection Location** | Attention computation (`SparseAttentionContext`) | Forward-hook dynamic gather during prefill and decode. |
| **Selection Scope** | **Temporary Attention Selection** | Does **not** evict resident keys from memory; reduces active attention pairs. |
| **Budget Units** | Active Attention Pairs $B_{\text{SA}}$ (or Sparsity Ratio) | Directly reduces attention FLOPs and latency. |
| **GQA Pre-Aggregation** | Group query pre-aggregation without re-norm | $\bar{Q} = \frac{1}{G}\sum Q_g$ maintains linear compute scaling and group cancellation. |
| **Head Aggregation** | **Per KV Head** | Top-$B_{\text{SA}}$ selected independently per KV head (shape: $[H_{\text{kv}}, B_{\text{SA}}]$). |
| **Candidate Pool** | Resident keys $k_j$ up to current step | Sinks ($S_{\text{sink}}$) and recent ($S_{\text{recent}}$) tokens protected if configured. |
| **Tie-Breaking** | Stable ascending token position order | Preserves deterministic selection. |
| **Causal Consistency** | Causal mask re-indexed for sparse active keys | Absolute key positions checked against absolute query positions ($P_k \le P_q$). |
| **Mask Preservation** | Preserves padding/additive mask | Fully masked rows sanitized to 0.0 attention weights (no NaNs). |

---

## 4. Shared-Head Approximation (`SharedHeadQUOKAApproximation`)

In addition to the faithful Algorithm 1 per-head implementation, an explicit approximation is provided:
- **`SharedHeadQUOKAApproximation`:** Averages the max-pooled scores across all KV heads into a single shared score vector:
  $$\hat{S}_{\text{shared}} = \frac{1}{H_{\text{kv}}} \sum_{h=1}^{H_{\text{kv}}} \hat{S}_h$$
  and selects a single shared index set of size $B_{\text{SA}}$.
- **Behavioral Distinction:** When different KV heads attend to orthogonal information (e.g. Head 0 attends to token 0, Head 1 attends to token 1), the shared approximation collapses to a compromise token, dropping high-relevance tokens on one head. The faithful per-head implementation maintains full fidelity to the paper.
