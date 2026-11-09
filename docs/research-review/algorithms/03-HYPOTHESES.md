# Algorithm Card: Document-Derived Hypotheses & Heuristics

This document formalizes the unvalidated heuristics found in the executive blueprint reference material, strictly separating them from peer-reviewed, published algorithms.

---

## 1. Hypothesis A: `centroid_neighbor` (Key Clustering Eviction)

### 1.1 Provenance
- **Origin:** Executive blueprint reference document (paragraphs 2359–2370 and 2923–2930).
- **Distinction:** In some sections of the unvalidated document, this heuristic was erroneously conflated with KeyDiff. The published KeyDiff paper (Bae et al., 2025) uses central mean-anchor dissimilarity ($-\text{CosSim}(\mu, k_i)$). `centroid_neighbor` is an unvalidated clustering heuristic.

### 1.2 Mathematical Formulation
1. **Centroid Computation:**
   $$\mu = \frac{1}{n} \sum_{i=1}^n \frac{k_i}{\|k_i\|_2 + \epsilon}$$
2. **Nearest-Neighbor Pair Distance:**
   For candidate key $k_i$, find its nearest neighbor $k_{\text{nn}(i)} = \arg\max_{j \ne i} \text{CosSim}(k_i, k_j)$.
3. **Eviction Criterion:**
   Among the closest neighbor pair $(k_i, k_j)$ having maximal mutual cosine similarity, evict the key that is closer to the global centroid $\mu$, retaining the more peripheral key:
   $$\text{evict} = \arg\max_{x \in \{i, j\}} \text{CosSim}(k_x, \mu)$$
   If $k_i$ is within a tight cluster near the center, it is deemed redundant.

### 1.3 Hypothesized Advantage vs Falsification Criterion
- **Hypothesis:** Evicting redundant neighbor pairs directly targets local cluster density more effectively than global mean dissimilarity.
- **Falsification:** If pairwise similarity evaluation incurs $O(n^2)$ latency overhead without measurable perplexity improvement over the $O(n)$ mean-anchor KeyDiff, the hypothesis is rejected.

---

## 2. Hypothesis B: `attention_value_decay` (Attention-Value Accumulator)

### 2.1 Provenance
- **Origin:** Executive blueprint reference document (paragraphs 3123–3133).
- **Distinction:** Erroneously labelled "H2O-QUOKA" or "CAOTE" in the unvalidated document. Published QUOKA is a chunked-prefill sparse attention selection algorithm utilizing representative queries. `attention_value_decay` is an autoregressive attention/value heuristic.

### 2.2 Mathematical Formulation
1. **Exponential Attention Decay:**
   At decode step $t$, with current attention distribution $\alpha^{(t)}$ over resident tokens:
   $$\bar{A}_i^{(t)} = \gamma \bar{A}_i^{(t-1)} + \alpha_i^{(t)}$$
   where $\gamma \in (0, 1]$ is a recency decay factor (e.g., $\gamma = 0.95$).
2. **Value Vector Norm Weighting:**
   Incorporates the $L_2$ norm of the value vector $v_i$, hypothesizing that tokens with larger value representations contribute more activation energy:
   $$s_i = \bar{A}_i^{(t)} \cdot \|v_i\|_2$$
3. **Eviction / Selection:**
   Tokens with lowest score $s_i$ outside the sink set are candidates for eviction or exclusion.

### 2.3 Hypothesized Advantage vs Falsification Criterion
- **Hypothesis:** Scaling accumulated attention by value norms prevents evicting tokens that have moderate attention frequency but high value magnitude impact.
- **Falsification:** If tracking attention histories across decode steps introduces tracking overhead while achieving parity or worse than recency windowing, the hypothesis is rejected.
