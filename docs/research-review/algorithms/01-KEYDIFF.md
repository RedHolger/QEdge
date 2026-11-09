# Algorithm Card: KeyDiff (Faithful Paper Variant)

## 1. Specification & Provenance

- **Algorithm Name:** KeyDiff (Key Similarity-Based KV Cache Eviction)
- **Primary Source Authors:** Junyoung Park, Dalton Jones, Matthew J. Morse, Raghavv Goel, Mingu Lee, Christopher M. Lott (Qualcomm AI Research).
- **Publication:** Pinned paper arXiv:2504.15364v1 (NeurIPS).
- **Official Implementation Provenance:** `NVIDIA/kvpress` (`KeyDiffPress`).
- **Paper URL:** [https://arxiv.org/abs/2504.15364](https://arxiv.org/abs/2504.15364)
- **Target Phase:** Autoregressive decode & post-prefill eviction.
- **Classification:** Training-free, query-independent **permanent KV cache eviction**.

---

## 2. Mathematical Formulation

### 2.1 Key Normalization and Central Mean Anchor
Given candidate resident key vectors $K = [k_1, k_2, \dots, k_n] \in \mathbb{R}^{n \times D}$ for a key-value head:
1. Normalize each key vector to unit sphere:
   $$\hat{k}_i = \frac{k_i}{\|k_i\|_2 + \epsilon}$$
   where $\epsilon = 10^{-12}$ prevents division by zero.
2. Compute the central mean anchor vector representing the geometric cluster center of the candidate keys:
   $$\mu(\hat{K}) = \frac{1}{n} \sum_{i=1}^n \hat{k}_i$$
3. Re-normalize the mean anchor:
   $$\hat{\mu}(\hat{K}) = \frac{\mu(\hat{K})}{\|\mu(\hat{K})\|_2 + \epsilon}$$

### 2.2 Eviction Scoring Function
KeyDiff operates on the geometric principle that keys with the lowest similarity to the central mean anchor are the most geometrically distinctive / isolated tokens, capturing rare, high-information context that receives strong global attention.

The retention score $s_i$ is defined as the negative cosine similarity between key $\hat{k}_i$ and the mean anchor $\hat{\mu}(\hat{K})$:
$$s_i = -\text{CosSim}(\hat{\mu}(\hat{K}), k_i) = -\frac{\hat{\mu}(\hat{K}) \cdot \hat{k}_i}{\|\hat{\mu}(\hat{K})\|_2 + \epsilon} = -\hat{\mu}(\hat{K}) \cdot \hat{k}_i$$

Keys with the highest score $s_i$ (i.e. lowest cosine similarity to the mean anchor) are prioritized for retention. Keys near the centroid (redundant cluster center) receive low scores and are evicted.

---

## 3. Algorithmic Dimensions & Parameters

| Dimension | Specification | Notes |
|---|---|---|
| **Selection Location** | `BoundedKVCache._select_retained_indices()` | Evaluated during eviction when $S_{\text{total}} > C$. |
| **Selection Scope** | **Permanent Eviction** | Permanently frees K/V memory for evicted tokens from physical cache. |
| **Budget Units** | Resident tokens $C$ (Cache capacity) | Directly reduces resident memory payload in DRAM/SRAM. |
| **Candidate Pool** | $\mathcal{I}_{\text{cand}} = [S_{\text{sink}}, n - S_{\text{recent}} - 1]$ | Candidate pool strictly excludes protected sinks and recent window. |
| **Protected Tokens** | $S_{\text{sink}}$ (initial tokens) + $S_{\text{recent}}$ (tail tokens) | Sinks preserve initial attention anchors; recent window preserves local syntax. |
| **Head Scoring** | Evaluated per KV head | Scored per KV head; aggregated across KV heads for unified layer eviction. |
| **Eviction Schedule** | Post-prefill and per decode step | Evaluated after prefill when $S_{\text{prompt}} > C$, and during decode whenever an added token exceeds $C$. |
| **Tie-Breaking** | Stable ascending token position order | When scores tie, earlier token positions are preserved. |
| **RoPE Interaction** | Evaluated on RoPE-rotated keys | Keys retain their canonical absolute rotation $R(P_{\text{abs}})$. |

---

## 4. Reference Implementations and Baselines

1. **Faithful KeyDiff (`KeyDiffSelector` / `reference_keydiff`):**
   - Implements the exact mean-anchor negative-cosine formulation from Park et al. (arXiv:2504.15364).
   - Validated against pure-Python reference implementation with deterministic tie-breaking.
2. **Official Open-Source Provenance:**
   - NVIDIA `kvpress` library provides the canonical reference implementation `KeyDiffPress`.
3. **Centroid-Neighbor Ablation (`CentroidNeighborSelector`):**
   - Evaluates a heuristic hypothesis that pairs tokens with nearest cluster neighbors. Distinct from published KeyDiff.
