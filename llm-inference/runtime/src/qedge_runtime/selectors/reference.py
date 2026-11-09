"""Independent pure-Python reference implementations of KeyDiff and QUOKA.

Used as ground-truth mathematical baselines for deterministic unit test verification.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch


def _vec_norm(v: List[float], eps: float = 1e-12) -> float:
    return math.sqrt(sum(x * x for x in v)) + eps


def _dot(v1: List[float], v2: List[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))


def _cos_sim(v1: List[float], v2: List[float], eps: float = 1e-12) -> float:
    return _dot(v1, v2) / (_vec_norm(v1, eps) * _vec_norm(v2, eps))


def reference_keydiff(
    keys: List[List[float]],
    budget: int,
    sink_tokens: int = 4,
    recent_tokens: int = 1,
) -> List[int]:
    """Pure-Python independent reference for KeyDiff mean-anchor key diversity.
    
    Primary source: Park et al., arXiv:2504.15364 (Qualcomm AI Research, NeurIPS).
    """
    total_count = len(keys)
    if total_count <= budget:
        return list(range(total_count))

    if budget <= 1:
        return [total_count - 1]

    sink_count = min(sink_tokens, budget - 1)
    sink_indices = list(range(sink_count))

    recent_budget = min(recent_tokens, budget - sink_count)
    recent_start = max(sink_count, total_count - recent_budget)
    recent_indices = list(range(recent_start, total_count))

    candidate_indices = [
        i for i in range(total_count)
        if i not in sink_indices and i not in recent_indices
    ]
    candidate_budget = max(0, budget - len(sink_indices) - len(recent_indices))

    if candidate_budget <= 0 or not candidate_indices:
        return sorted(sink_indices + recent_indices)[:budget]

    # Normalize candidate keys
    dim = len(keys[0])
    normed_keys: List[List[float]] = []
    for idx in candidate_indices:
        k = keys[idx]
        norm = _vec_norm(k)
        normed_keys.append([x / norm for x in k])

    # Compute mean anchor
    n_cand = len(normed_keys)
    mean_anchor = [sum(normed_keys[i][d] for i in range(n_cand)) / n_cand for d in range(dim)]
    anchor_norm = _vec_norm(mean_anchor)
    normed_anchor = [x / anchor_norm for x in mean_anchor]

    # Score: negative cosine similarity to mean anchor
    scores: List[Tuple[float, int, int]] = []
    for i, idx in enumerate(candidate_indices):
        sim = _dot(normed_keys[i], normed_anchor)
        score = -sim
        # Use -idx for stable tie-breaking (prefer earlier tokens)
        scores.append((score, -idx, idx))

    scores.sort(reverse=True)
    selected_candidates = [item[2] for item in scores[:candidate_budget]]

    return sorted(sink_indices + selected_candidates + recent_indices)


def paper_quoka_oracle(
    queries: List[List[List[float]]],   # [H_q, S_q, D]
    keys: List[List[List[float]]],      # [H_kv, S_k, D]
    budget: int,                        # B_SA
    max_queries: Optional[int] = None,  # N_Q
    sink_tokens: int = 0,
    recent_tokens: int = 0,
    eps: float = 1e-12,
    representative_ratio: Optional[float] = 0.25,
) -> List[List[int]]:                   # [H_kv, B_SA]
    """Authoritative, independently transcribed paper oracle for QUOKA Algorithm 1.
    
    Primary source: Jones et al., arXiv:2602.08722v1 (Qualcomm AI Research).
    
    Line-by-line fidelity to Algorithm 1:
    1. Query Subselection:
       - M_Q = mean(Q, dim=2) BEFORE query normalization
       - S_Q = CosSim(Q, M_Q) evaluated per query head
       - Q_rep = topk(-S_Q, N_Q) gathered per query head
    2. Efficient Cosine Similarity via Pre-aggregation:
       - Q and K normalized to unit sphere
       - Average normalized queries within each GQA group: Q_bar = mean(Q_norm, dim=GQA)
       - DO NOT re-normalize Q_bar (preserves cosine linearity)
    3. Interaction & Max-Pooling:
       - S = Q_bar @ K^T per KV head
       - S_hat = max(S, dim=queries) per KV head
    4. Token Subselection:
       - Top-k selection is strictly PER KV HEAD: I = topk(S_hat, B_SA)
    """
    # Handle both 2D [S, D] and 3D [H, S, D] inputs
    is_2d_query = isinstance(queries[0][0], (int, float))
    if is_2d_query:
        queries = [queries]  # Wrap to [1, S_q, D]

    is_2d_key = isinstance(keys[0][0], (int, float))
    if is_2d_key:
        keys = [keys]  # Wrap to [1, S_k, D]

    h_q = len(queries)
    s_q = len(queries[0])
    d = len(queries[0][0])

    h_kv = len(keys)
    s_k = len(keys[0])

    assert h_q % h_kv == 0, f"H_q ({h_q}) must be divisible by H_kv ({h_kv})"
    group_size = h_q // h_kv

    if max_queries is not None:
        n_q = max_queries
    elif representative_ratio is not None:
        n_q = max(1, int(s_q * representative_ratio))
    else:
        n_q = s_q
    n_q = min(s_q, max(1, n_q))

    # -------------------------------------------------------------
    # 1. Query Subselection (Algorithm 1, Lines 1-6)
    # -------------------------------------------------------------
    q_rep: List[List[List[float]]] = []
    if s_q > n_q:
        for h in range(h_q):
            # M_Q <- mean(Q, dim=2) computed BEFORE query normalization!
            m_q = [sum(queries[h][i][dim_idx] for i in range(s_q)) / s_q for dim_idx in range(d)]
            m_q_norm = _vec_norm(m_q, eps)
            m_q_hat = [x / m_q_norm for x in m_q]

            # S_Q <- CosSim(Q, M_Q) per query head
            s_q_scores: List[Tuple[float, int, List[float]]] = []
            for i in range(s_q):
                qi = queries[h][i]
                qi_norm = _vec_norm(qi, eps)
                qi_hat = [x / qi_norm for x in qi]
                cos_sim = _dot(qi_hat, m_q_hat)
                # Negated cosine similarity: -S_Q (lowest similarity prioritized)
                neg_sim = -cos_sim
                s_q_scores.append((neg_sim, -i, qi))

            # Stable sort descending: highest neg_sim (lowest similarity to mean)
            s_q_scores.sort(reverse=True)
            top_q = [item[2] for item in s_q_scores[:n_q]]
            q_rep.append(top_q)
    else:
        q_rep = queries

    # -------------------------------------------------------------
    # 2. Efficient Cosine Similarity via Pre-aggregation (Lines 7-10)
    # -------------------------------------------------------------
    # Line 8: Q <- Q / norm(Q)
    q_norm: List[List[List[float]]] = []
    for h in range(h_q):
        h_norm: List[List[float]] = []
        for j in range(n_q):
            v = q_rep[h][j]
            norm = _vec_norm(v, eps)
            h_norm.append([x / norm for x in v])
        q_norm.append(h_norm)

    # Line 9: K <- K / norm(K)
    k_norm: List[List[List[float]]] = []
    for h_k in range(h_kv):
        hk_norm: List[List[float]] = []
        for t in range(s_k):
            v = keys[h_k][t]
            norm = _vec_norm(v, eps)
            hk_norm.append([x / norm for x in v])
        k_norm.append(hk_norm)

    # Line 10: Q_bar <- mean(Q.reshape(b, n_kv, G, n_q, d), dim=2)
    # Average normalized queries within GQA group. DO NOT renormalize!
    q_bar: List[List[List[float]]] = []
    for h_k in range(h_kv):
        group_start = h_k * group_size
        group_end = group_start + group_size
        k_group_rep: List[List[float]] = []
        for j in range(n_q):
            avg_vec = [
                sum(q_norm[gh][j][dim_idx] for gh in range(group_start, group_end)) / group_size
                for dim_idx in range(d)
            ]
            k_group_rep.append(avg_vec)
        q_bar.append(k_group_rep)

    # -------------------------------------------------------------
    # 3. Scoring & Max-Pooling (Lines 11-12)
    # -------------------------------------------------------------
    # Line 11: S <- Q_bar @ K^T  shape: (h_kv, n_q, s_k)
    # Line 12: S_hat <- max(S, dim=2) shape: (h_kv, s_k)
    per_head_selected: List[List[int]] = []

    for h_k in range(h_kv):
        s_hat: List[float] = []
        for t in range(s_k):
            k_vec = k_norm[h_k][t]
            # Max cosine interaction across representative queries
            max_inter = max(_dot(q_bar[h_k][j], k_vec) for j in range(n_q))
            s_hat.append(max_inter)

        # ---------------------------------------------------------
        # 4. Token Subselection per KV head (Line 13)
        # ---------------------------------------------------------
        if s_k <= budget:
            selected = list(range(s_k))
        elif sink_tokens == 0 and recent_tokens == 0:
            scored_cands = [(s_hat[idx], -idx, idx) for idx in range(s_k)]
            scored_cands.sort(reverse=True)
            selected = sorted([item[2] for item in scored_cands[:budget]])
        elif budget <= 1:
            if recent_tokens > 0:
                selected = [s_k - 1]
            else:
                selected = [0]
        else:
            sink_count = min(sink_tokens, budget - 1)
            sink_indices = list(range(sink_count))

            recent_budget = min(recent_tokens, budget - sink_count)
            recent_start = max(sink_count, s_k - recent_budget)
            recent_indices = list(range(recent_start, s_k))

            candidate_indices = [
                i for i in range(s_k)
                if i not in sink_indices and i not in recent_indices
            ]
            cand_budget = max(0, budget - len(sink_indices) - len(recent_indices))

            if cand_budget <= 0 or not candidate_indices:
                selected = sorted(sink_indices + recent_indices)[:budget]
            else:
                scored_cands = [(s_hat[idx], -idx, idx) for idx in candidate_indices]
                scored_cands.sort(reverse=True)
                chosen_cands = [item[2] for item in scored_cands[:cand_budget]]
                selected = sorted(sink_indices + chosen_cands + recent_indices)

        per_head_selected.append(selected)

    if is_2d_query and is_2d_key:
        return per_head_selected[0]

    return per_head_selected


# Backward compatibility alias
reference_quoka = paper_quoka_oracle


def is_blocking_sentinel(mask_tensor: torch.Tensor) -> torch.Tensor:
    """Detect documented blocking sentinels in an additive attention mask.

    Supported blocking sentinels:
    1. Negative infinity: `torch.isneginf(x)` or `x == float("-inf")`.
    2. Finite dtype minimums: Values within 1% of `torch.finfo(dtype).min`
       (e.g. `torch.finfo(torch.float32).min` ~ -3.4028e38, `torch.finfo(torch.bfloat16).min` ~ -3.3895e38,
       `torch.finfo(torch.float16).min` ~ -65504.0).
    3. Extreme sentinel threshold: Any value <= -1e30 for 32/64-bit floats or <= -65000.0 for 16-bit floats.
    4. NaN values: `torch.isnan(x)`.

    Arbitrary finite attention biases (e.g. -2.0, -10.0, -100.0, +5.0) are NOT sentinels
    and are preserved as numerical additive biases during attention calculation.
    """
    if not mask_tensor.is_floating_point():
        return mask_tensor == 0
    neginf = torch.isneginf(mask_tensor)
    isnan = torch.isnan(mask_tensor)
    fmin = torch.finfo(mask_tensor.dtype).min
    is_fmin = mask_tensor <= (0.99 * fmin) if fmin < 0 else torch.zeros_like(neginf)
    if mask_tensor.dtype == torch.float16:
        is_extreme = mask_tensor <= -65000.0
    else:
        is_extreme = mask_tensor <= -1e30
    return neginf | isnan | is_fmin | is_extreme


def _is_mask_allowed(mask_tensor: torch.Tensor) -> torch.Tensor:
    """Determine boolean allowed status from an additive mask tensor."""
    return ~is_blocking_sentinel(mask_tensor)


def masked_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    q_pos: Optional[torch.Tensor] = None,
    k_pos: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure, independent reference for masked scaled dot-product attention.
    
    Serves as an authoritative mathematical reference to verify SparseAttentionContext
    across:
    - 2D padding masks (bool, integer, float: 1=allowed, 0=blocked)
    - 3D and 4D additive masks (allowed ~0.0, blocked <= -1000.0 or finite sentinels like finfo.min)
    - Absolute causal masking: k_pos <= q_pos
    - GQA query-head replication
    - Fully blocked row detection: strictly zero attention weights and zero attention output
    """
    import torch
    from transformers.models.llama.modeling_llama import repeat_kv

    batch_size, h_q, s_q, d = query.shape
    b_k, h_kv, s_k, d_k = key.shape
    device = query.device
    dtype = query.dtype

    if scaling is None:
        scaling = 1.0 / math.sqrt(d)

    group_size = h_q // h_kv
    k_rep = repeat_kv(key, group_size)
    v_rep = repeat_kv(value, group_size)

    if q_pos is None:
        q_pos = torch.arange(s_q, device=device, dtype=torch.long)
    if k_pos is None:
        k_pos = torch.arange(s_k, device=device, dtype=torch.long)

    # 1. Convert attention_mask to 4D additive form [batch, num_heads, s_q, s_k]
    add_mask: Optional[torch.Tensor] = None
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                allowed = attention_mask
            elif not torch.is_floating_point(attention_mask):
                allowed = attention_mask != 0
            else:
                allowed = attention_mask > 0.5
            add_mask = torch.where(
                allowed,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), float("-inf"), dtype=dtype, device=device),
            ).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, s_q, s_k)
        elif attention_mask.dim() == 3:
            add_mask = attention_mask.unsqueeze(1).to(device=device, dtype=dtype)
        elif attention_mask.dim() == 4:
            add_mask = attention_mask.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported mask dimension: {attention_mask.dim()}")

    # 2. Build combined causal and additive mask per query head
    head_masks = []
    head_rows_allowed = []

    for h in range(h_q):
        head_k_pos = k_pos[h // group_size] if k_pos.dim() == 2 else k_pos
        causal_allowed = (head_k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1, 1, s_q, s_k]

        if add_mask is not None:
            mask_slice = (
                add_mask[:, h : h + 1, :, :]
                if add_mask.shape[1] == h_q
                else add_mask[:, 0:1, :, :]
            )
            mask_allowed = _is_mask_allowed(mask_slice)
            overall_allowed = causal_allowed & mask_allowed
            eff_mask = torch.where(
                overall_allowed,
                mask_slice,
                torch.full((), float("-inf"), dtype=dtype, device=device),
            )
        else:
            overall_allowed = causal_allowed
            eff_mask = torch.where(
                overall_allowed,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), float("-inf"), dtype=dtype, device=device),
            )

        row_has_allowed = overall_allowed.any(dim=-1, keepdim=True)
        head_masks.append(eff_mask)
        head_rows_allowed.append(row_has_allowed)

    eff_mask_4d = torch.cat(head_masks, dim=1)  # [batch, h_q, s_q, s_k]
    rows_allowed_4d = torch.cat(head_rows_allowed, dim=1)  # [batch, h_q, s_q, 1]

    # 3. Scaled dot product
    scores = torch.matmul(query, k_rep.transpose(-2, -1)) * scaling
    scores = scores + eff_mask_4d
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    weights = torch.nan_to_num(weights, nan=0.0).to(dtype)

    # Fully blocked rows set strictly to 0.0
    weights = torch.where(rows_allowed_4d, weights, torch.zeros_like(weights))
    output = torch.matmul(weights, v_rep)

    return output, weights

