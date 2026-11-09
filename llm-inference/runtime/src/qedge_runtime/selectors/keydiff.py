"""KeyDiff: Key Similarity-Based KV Cache Eviction.

Implements the faithful algorithm from Qualcomm AI Research (arXiv:2504.15364):
Authors: Junyoung Park, Dalton Jones, Matthew J. Morse, Raghavv Goel, Mingu Lee, Christopher M. Lott.
Published at: NeurIPS.
Official implementation provenance: NVIDIA kvpress (kvpress.presses.KeyDiffPress) and Qualcomm AI Research.

Mathematical specification:
- Evaluates geometric diversity of resident keys relative to the central mean anchor per KV head.
- Anchor vector: anchor_h = mean(normalize(k_h), dim=tokens)
- Key similarity: cos_sim_h = CosSim(k_h, anchor_h)
- Importance score: score_h = -cos_sim_h (lower similarity to mean anchor -> highest retention priority)
- Candidate pool: resident tokens excluding attention sinks and sliding recency window.
- Sequence-level eviction: macro-averages score across KV heads to preserve uniform-sequence cache layout.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .base import BaseSelector, SelectionKind, SelectorResult


class KeyDiffSelector(BaseSelector):
    """Faithful KeyDiff mean-anchor key diversity selector."""

    def __init__(
        self,
        sink_tokens: int = 4,
        recent_tokens: int = 1,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__(
            name="keydiff",
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
        """Select tokens to retain according to KeyDiff mean-anchor diversity.
        
        Args:
            keys: Key tensor of shape [1, H_kv, S, D].
            budget: Number of tokens to retain.
            queries: Unused (KeyDiff is query-independent).
            values: Unused.
            
        Returns:
            SelectorResult with sorted retained indices and execution time.
        """
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

        # Candidate keys: shape [1, H_kv, N_cand, D]
        cand_idx_tensor = torch.tensor(candidate_indices, device=keys.device, dtype=torch.long)
        cand_keys = torch.index_select(keys, 2, cand_idx_tensor)

        # 1. Normalize candidate keys: k_norm = k / (||k|| + eps)
        norms = torch.norm(cand_keys, p=2, dim=-1, keepdim=True)
        cand_keys_norm = cand_keys / (norms + self.epsilon)

        # 2. Central mean anchor per KV head: shape [1, H_kv, 1, D]
        mean_anchor = torch.mean(cand_keys_norm, dim=2, keepdim=True)
        mean_anchor_norm = mean_anchor / (torch.norm(mean_anchor, p=2, dim=-1, keepdim=True) + self.epsilon)

        # 3. Cosine similarity to mean anchor: shape [1, H_kv, N_cand]
        # cos_sim = sum_d (cand_keys_norm * mean_anchor_norm)
        cos_sim = torch.sum(cand_keys_norm * mean_anchor_norm, dim=-1)

        # 4. KeyDiff score: negative cosine similarity (lower similarity -> higher priority)
        # Average across KV heads for multi-head consistency: shape [N_cand]
        scores = -torch.mean(cos_sim, dim=1).squeeze(0)

        # 5. Select top candidate_budget indices by score
        # Using stable sort to preserve deterministic token ordering on ties
        _, sorted_positions = torch.sort(scores, descending=True, stable=True)
        selected_cand_positions = sorted_positions[:candidate_budget].tolist()
        selected_cand_indices = [candidate_indices[p] for p in selected_cand_positions]

        # Combine protected and selected tokens
        final_indices = sorted(sink_indices + selected_cand_indices + recent_indices)
        t1 = time.perf_counter()

        return SelectorResult(
            retained_indices=final_indices,
            selection_time_ms=(t1 - t0) * 1000.0,
            scores=scores,
            metadata={
                "mean_anchor_norm": torch.norm(mean_anchor, p=2).item(),
                "head_scores": -cos_sim[0],
                "num_candidates": len(candidate_indices),
                "num_selected_candidates": len(selected_cand_indices),
            },
        )
