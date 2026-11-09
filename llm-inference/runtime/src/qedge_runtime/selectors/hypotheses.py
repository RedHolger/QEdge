"""Unvalidated document-derived hypotheses and heuristics.

These implementations correspond to heuristics referenced in the executive blueprint:
1. CentroidNeighborSelector: A key-clustering heuristic pairing keys with nearest neighbors.
2. AttentionValueDecaySelector: An attention accumulation and value-norm heuristic (formerly mislabelled 'H2O-QUOKA').

Kept strictly separated from published algorithms (KeyDiff and QUOKA).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .base import BaseSelector, SelectionKind, SelectorResult


class CentroidNeighborSelector(BaseSelector):
    """Document-derived hypothesis: Centroid-plus-neighbor key clustering eviction.
    
    Pairs candidate keys with their nearest neighbors and penalizes tokens
    that are both mutually redundant and clustered near the global centroid.
    """

    def __init__(
        self,
        sink_tokens: int = 4,
        recent_tokens: int = 1,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__(
            name="centroid_neighbor",
            kind=SelectionKind.PERMANENT_EVICTION,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            epsilon=epsilon,
        )

    def select_indices(
        self,
        keys: torch.Tensor,
        budget: int,
        queries: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SelectorResult:
        t0 = time.perf_counter()

        b, h_kv, s, d = keys.shape
        if b != 1:
            raise ValueError(f"Batch size must be 1, got {b}")

        if s <= budget:
            t1 = time.perf_counter()
            return SelectorResult(
                retained_indices=list(range(s)),
                selection_time_ms=(t1 - t0) * 1000.0,
            )

        sink_indices, candidate_indices, recent_indices, candidate_budget = (
            self.get_protected_and_candidate_indices(total_count=s, budget=budget)
        )

        if candidate_budget <= 0 or not candidate_indices:
            t1 = time.perf_counter()
            selected = sorted(sink_indices + recent_indices)[:budget]
            return SelectorResult(
                retained_indices=selected,
                selection_time_ms=(t1 - t0) * 1000.0,
            )

        cand_idx_tensor = torch.tensor(candidate_indices, device=keys.device, dtype=torch.long)
        cand_keys = torch.index_select(keys, 2, cand_idx_tensor)  # [1, H_kv, N_cand, D]

        # 1. Normalize candidate keys
        cand_keys_norm = cand_keys / (torch.norm(cand_keys, p=2, dim=-1, keepdim=True) + self.epsilon)

        # 2. Centroid per KV head: [1, H_kv, 1, D]
        centroid = torch.mean(cand_keys_norm, dim=2, keepdim=True)
        centroid_norm = centroid / (torch.norm(centroid, p=2, dim=-1, keepdim=True) + self.epsilon)

        # Cosine similarity to centroid: [1, H_kv, N_cand]
        sim_to_centroid = torch.sum(cand_keys_norm * centroid_norm, dim=-1)

        # 3. Pairwise cosine similarity between candidate keys: [1, H_kv, N_cand, N_cand]
        pairwise_sim = torch.matmul(cand_keys_norm, cand_keys_norm.transpose(-2, -1))
        # Mask self-similarity (diagonal)
        eye = torch.eye(len(candidate_indices), device=keys.device).unsqueeze(0).unsqueeze(0)
        pairwise_sim_no_diag = pairwise_sim - 2.0 * eye

        # Max nearest-neighbor similarity: [1, H_kv, N_cand]
        max_neighbor_sim, _ = torch.max(pairwise_sim_no_diag, dim=-1)

        # Retention score: penalize tokens that have high neighbor similarity AND high centroid similarity
        # score = -(sim_to_centroid + max_neighbor_sim)
        raw_scores = -(sim_to_centroid + max_neighbor_sim)
        scores = torch.mean(raw_scores, dim=1).squeeze(0)  # Shape [N_cand]

        _, sorted_positions = torch.sort(scores, descending=True, stable=True)
        selected_cand_positions = sorted_positions[:candidate_budget].tolist()
        selected_cand_indices = [candidate_indices[p] for p in selected_cand_positions]

        final_indices = sorted(sink_indices + selected_cand_indices + recent_indices)
        t1 = time.perf_counter()

        return SelectorResult(
            retained_indices=final_indices,
            selection_time_ms=(t1 - t0) * 1000.0,
            scores=scores,
            metadata={"heuristic": "centroid_neighbor", "num_candidates": len(candidate_indices)},
        )


class AttentionValueDecaySelector(BaseSelector):
    """Document-derived hypothesis: Attention-value decay heuristic.
    
    Weights query-key dot products or attention weights by value vector norms,
    with an exponential decay factor.
    """

    def __init__(
        self,
        sink_tokens: int = 4,
        recent_tokens: int = 1,
        decay_factor: float = 0.95,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__(
            name="attention_value_decay",
            kind=SelectionKind.TEMPORARY_ATTENTION_SELECTION,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            epsilon=epsilon,
        )
        self.decay_factor = decay_factor
        self.accumulated_attention: Optional[torch.Tensor] = None

    def reset_accumulator(self) -> None:
        self.accumulated_attention = None

    def select_indices(
        self,
        keys: torch.Tensor,
        budget: int,
        queries: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SelectorResult:
        t0 = time.perf_counter()

        b, h_kv, s, d = keys.shape
        if s <= budget:
            t1 = time.perf_counter()
            return SelectorResult(
                retained_indices=list(range(s)),
                selection_time_ms=(t1 - t0) * 1000.0,
            )

        sink_indices, candidate_indices, recent_indices, candidate_budget = (
            self.get_protected_and_candidate_indices(total_count=s, budget=budget)
        )

        if candidate_budget <= 0 or not candidate_indices:
            t1 = time.perf_counter()
            selected = sorted(sink_indices + recent_indices)[:budget]
            return SelectorResult(
                retained_indices=selected,
                selection_time_ms=(t1 - t0) * 1000.0,
            )

        cand_idx_tensor = torch.tensor(candidate_indices, device=keys.device, dtype=torch.long)
        cand_keys = torch.index_select(keys, 2, cand_idx_tensor)

        # 1. Query-key similarity
        if queries is not None:
            # Macro-average queries across heads if shapes differ
            q_mean = torch.mean(queries, dim=1, keepdim=True)  # [1, 1, S_q, D]
            if q_mean.shape[2] > 1:
                q_vec = q_mean[:, :, -1:, :]  # Use last query token
            else:
                q_vec = q_mean
            q_norm = q_vec / (torch.norm(q_vec, p=2, dim=-1, keepdim=True) + self.epsilon)
            k_norm = cand_keys / (torch.norm(cand_keys, p=2, dim=-1, keepdim=True) + self.epsilon)
            attn_proxy = torch.matmul(q_norm, k_norm.transpose(-2, -1)).squeeze(-2)  # [1, H_kv, N_cand]
            attn_scores = torch.mean(attn_proxy, dim=1).squeeze(0)  # [N_cand]
        else:
            attn_scores = torch.ones(len(candidate_indices), device=keys.device)

        # 2. Value norm weighting
        if values is not None:
            cand_values = torch.index_select(values, 2, cand_idx_tensor)
            val_norms = torch.norm(cand_values, p=2, dim=-1)  # [1, H_kv, N_cand]
            val_weights = torch.mean(val_norms, dim=1).squeeze(0)  # [N_cand]
        else:
            val_weights = torch.ones(len(candidate_indices), device=keys.device)

        # Combined score: attention proxy * value norm
        scores = attn_scores * val_weights

        _, sorted_positions = torch.sort(scores, descending=True, stable=True)
        selected_cand_positions = sorted_positions[:candidate_budget].tolist()
        selected_cand_indices = [candidate_indices[p] for p in selected_cand_positions]

        final_indices = sorted(sink_indices + selected_cand_indices + recent_indices)
        t1 = time.perf_counter()

        return SelectorResult(
            retained_indices=final_indices,
            selection_time_ms=(t1 - t0) * 1000.0,
            scores=scores,
            metadata={"heuristic": "attention_value_decay", "num_candidates": len(candidate_indices)},
        )
