"""QUOKA: Query-Oriented KV Selection For Efficient LLM Prefill.

Implements the faithful algorithm from Qualcomm AI Research (arXiv:2602.08722v1):
Authors: Dalton Jones, Junyoung Park, Matthew Morse, Mingu Lee, Chris Lott, M. Harper Langston.

Line-by-line fidelity to Algorithm 1:
1. Query Sub-selection:
   - Sequence mean query M_Q is computed BEFORE query normalization.
   - S_Q = CosSim(Q, M_Q) evaluated per query head.
   - Top-N_Q queries with lowest cosine similarity are selected PER QUERY HEAD.
2. Efficient Cosine Similarity via Pre-aggregation:
   - Queries and keys are normalized to unit L2 norm.
   - Normalized queries are averaged within GQA groups: Q_bar = mean(Q_norm, dim=GQA).
   - Q_bar is NOT re-normalized (preserves cosine linearity and head-agreement magnitude).
3. Interaction Scoring via Max-Pooling:
   - S = Q_bar @ K^T evaluated per KV head.
   - S_hat = max(S, dim=queries) max-pooled per KV head.
4. Token Subselection:
   - Top-k selection is strictly evaluated PER KV HEAD.
   - Each KV head retains its own subset of keys aligned with its query group.

Also provides `SharedHeadQUOKAApproximation` as an explicit named ablation where
scores are macro-averaged across KV heads into a single compromise index set.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseSelector, SelectionKind, SelectorResult


class QUOKASelector(BaseSelector):
    """Faithful QUOKA query-oriented sparse attention selector (Algorithm 1)."""

    def __init__(
        self,
        sink_tokens: int = 4,
        recent_tokens: int = 1,
        representative_ratio: float = 0.25,
        max_queries: Optional[int] = None,
        min_representative_queries: int = 1,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__(
            name="quoka",
            kind=SelectionKind.TEMPORARY_ATTENTION_SELECTION,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            epsilon=epsilon,
        )
        self.representative_ratio = representative_ratio
        self.max_queries = max_queries
        self.min_representative_queries = max(1, min_representative_queries)

    def select_indices(
        self,
        keys: torch.Tensor,
        budget: int,
        queries: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SelectorResult:
        """Select active key indices for attention evaluation following Algorithm 1.
        
        Args:
            keys: Key tensor of shape [1, H_kv, S_k, D].
            budget: Number of active attention key pairs to retain per KV head (B_SA).
            queries: Query tensor of shape [1, H_q, S_q, D]. Required for QUOKA.
            values: Unused.
            
        Returns:
            SelectorResult with per_head_indices [H_kv, B_SA] and head_retained_indices.
        """
        t0 = time.perf_counter()

        if queries is None:
            raise ValueError("QUOKA requires a query tensor [1, H_q, S_q, D] for query-oriented selection")

        b_k, h_kv, s_k, d = keys.shape
        b_q, h_q, s_q, d_q = queries.shape

        if b_k != 1 or b_q != 1:
            raise ValueError(f"Batch size must be 1, got key={b_k}, query={b_q}")
        if d != d_q:
            raise ValueError(f"Head dimension mismatch: key={d}, query={d_q}")
        if h_q % h_kv != 0:
            raise ValueError(f"Number of query heads {h_q} must be divisible by KV heads {h_kv}")

        group_size = h_q // h_kv

        # Full budget: retain all tokens across all heads
        if s_k <= budget:
            t1 = time.perf_counter()
            all_idx = list(range(s_k))
            per_head = torch.arange(s_k, device=keys.device, dtype=torch.long).unsqueeze(0).expand(h_kv, -1)
            return SelectorResult(
                retained_indices=all_idx,
                selection_time_ms=(t1 - t0) * 1000.0,
                head_retained_indices=[all_idx for _ in range(h_kv)],
                per_head_indices=per_head,
            )

        sink_indices, candidate_indices, recent_indices, candidate_budget = (
            self.get_protected_and_candidate_indices(total_count=s_k, budget=budget)
        )

        if candidate_budget <= 0 or not candidate_indices:
            t1 = time.perf_counter()
            selected = sorted(sink_indices + recent_indices)[:budget]
            per_head = torch.tensor(selected, device=keys.device, dtype=torch.long).unsqueeze(0).expand(h_kv, -1)
            return SelectorResult(
                retained_indices=selected,
                selection_time_ms=(t1 - t0) * 1000.0,
                head_retained_indices=[selected for _ in range(h_kv)],
                per_head_indices=per_head,
            )

        # -----------------------------------------------------------------
        # STEP 1: Representative Query Subselection (Algorithm 1, Lines 1-6)
        # -----------------------------------------------------------------
        if self.max_queries is not None:
            n_q = self.max_queries
        else:
            n_q = max(self.min_representative_queries, int(s_q * self.representative_ratio))
        n_q = min(s_q, max(1, n_q))

        if s_q > n_q:
            # Line 3: M_Q <- mean(Q, dim=2) computed BEFORE query normalization!
            m_q = torch.mean(queries, dim=2, keepdim=True)  # [1, H_q, 1, D]

            # Line 4: S_Q <- CosSim(Q, M_Q) evaluated per query head
            s_q_sim = F.cosine_similarity(queries, m_q, dim=-1, eps=self.epsilon)  # [1, H_q, S_q]

            # Line 5: Q <- gather(topk(-S_Q, N_Q), Q)
            # Queries with LOWEST cosine similarity to mean query are prioritized
            _, top_q_idx = torch.topk(-s_q_sim, k=n_q, dim=-1, sorted=True)  # [1, H_q, N_Q]
            q_rep = torch.gather(queries, 2, top_q_idx.unsqueeze(-1).expand(-1, -1, -1, d))  # [1, H_q, N_Q, D]
        else:
            q_rep = queries  # [1, H_q, S_q, D]

        num_rep = q_rep.shape[2]

        # -----------------------------------------------------------------
        # STEP 2: Efficient Cosine Similarity via Pre-aggregation (Lines 7-10)
        # -----------------------------------------------------------------
        # Line 8: Q <- Q / norm(Q, dim=-1)
        q_norm = F.normalize(q_rep, p=2, dim=-1, eps=self.epsilon)  # [1, H_q, N_Q, D]

        # Line 9: K <- K / norm(K, dim=-1)
        k_norm = F.normalize(keys, p=2, dim=-1, eps=self.epsilon)    # [1, H_kv, S_k, D]

        # Line 10: Q_bar <- mean(Q.reshape(b, n_kv, group_size, N_Q, d), dim=2)
        q_norm_grouped = q_norm.view(1, h_kv, group_size, num_rep, d)
        q_bar = torch.mean(q_norm_grouped, dim=2)  # Shape: [1, H_kv, N_Q, D]
        # CRITICAL: Do NOT re-normalize Q_bar (preserves cosine linearity and group alignment)

        # -----------------------------------------------------------------
        # STEP 3: Scoring & Max-Pooling (Lines 11-12)
        # -----------------------------------------------------------------
        # Line 11: S <- Q_bar @ K^T  shape: [1, H_kv, N_Q, S_k]
        s_matrix = torch.matmul(q_bar, k_norm.transpose(-2, -1))

        # Line 12: S_hat <- max(S, dim=2) shape: [1, H_kv, S_k]
        s_hat, _ = torch.max(s_matrix, dim=2)

        # -----------------------------------------------------------------
        # STEP 4: Token Subselection Per KV Head (Line 13)
        # -----------------------------------------------------------------
        head_retained_indices: List[List[int]] = []
        per_head_tensor_list: List[torch.Tensor] = []

        cand_tensor = torch.tensor(candidate_indices, device=keys.device, dtype=torch.long)

        for h in range(h_kv):
            # Extract scores for candidate tokens in this KV head
            head_cand_scores = s_hat[0, h, cand_tensor]  # [N_cand]
            _, sorted_positions = torch.sort(head_cand_scores, descending=True, stable=True)
            chosen_cand_pos = sorted_positions[:candidate_budget]
            chosen_cands = cand_tensor[chosen_cand_pos].tolist()

            head_indices = sorted(sink_indices + chosen_cands + recent_indices)
            head_retained_indices.append(head_indices)
            per_head_tensor_list.append(torch.tensor(head_indices, device=keys.device, dtype=torch.long))

        per_head_indices = torch.stack(per_head_tensor_list, dim=0)  # Shape: [H_kv, B_SA]

        t1 = time.perf_counter()

        return SelectorResult(
            retained_indices=head_retained_indices[0],
            selection_time_ms=(t1 - t0) * 1000.0,
            scores=s_hat,
            head_retained_indices=head_retained_indices,
            per_head_indices=per_head_indices,
            metadata={
                "algorithm": "QUOKA_Algorithm_1_per_head",
                "num_representative_queries": num_rep,
                "group_size": group_size,
                "num_candidates": len(candidate_indices),
                "num_selected_candidates": candidate_budget,
            },
        )


class SharedHeadQUOKAApproximation(BaseSelector):
    """Document-derived ablation: Shared-Head QUOKA Approximation.
    
    Averages key interaction scores across all KV heads into a single compromise
    retained index set. Documented deviation from the primary paper (arXiv:2602.08722v1).
    """

    def __init__(
        self,
        sink_tokens: int = 4,
        recent_tokens: int = 1,
        representative_ratio: float = 0.25,
        min_representative_queries: int = 1,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__(
            name="shared_head_quoka_approximation",
            kind=SelectionKind.TEMPORARY_ATTENTION_SELECTION,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            epsilon=epsilon,
        )
        self.representative_ratio = representative_ratio
        self.min_representative_queries = max(1, min_representative_queries)

    def select_indices(
        self,
        keys: torch.Tensor,
        budget: int,
        queries: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SelectorResult:
        t0 = time.perf_counter()
        if queries is None:
            raise ValueError("QUOKA requires queries")

        b_k, h_kv, s_k, d = keys.shape
        b_q, h_q, s_q, d_q = queries.shape

        if s_k <= budget:
            t1 = time.perf_counter()
            all_idx = list(range(s_k))
            per_head = torch.arange(s_k, device=keys.device, dtype=torch.long).unsqueeze(0).expand(h_kv, -1)
            return SelectorResult(
                retained_indices=all_idx,
                selection_time_ms=(t1 - t0) * 1000.0,
                head_retained_indices=[all_idx for _ in range(h_kv)],
                per_head_indices=per_head,
            )

        sink_indices, candidate_indices, recent_indices, candidate_budget = (
            self.get_protected_and_candidate_indices(total_count=s_k, budget=budget)
        )

        n_q = max(self.min_representative_queries, int(s_q * self.representative_ratio))
        n_q = min(s_q, max(1, n_q))

        if s_q > n_q:
            m_q = torch.mean(queries, dim=2, keepdim=True)
            s_q_sim = F.cosine_similarity(queries, m_q, dim=-1, eps=self.epsilon)
            _, top_q_idx = torch.topk(-s_q_sim, k=n_q, dim=-1, sorted=True)
            q_rep = torch.gather(queries, 2, top_q_idx.unsqueeze(-1).expand(-1, -1, -1, d))
        else:
            q_rep = queries

        group_size = h_q // h_kv
        q_norm = F.normalize(q_rep, p=2, dim=-1, eps=self.epsilon)
        k_norm = F.normalize(keys, p=2, dim=-1, eps=self.epsilon)
        q_bar = torch.mean(q_norm.view(1, h_kv, group_size, q_rep.shape[2], d), dim=2)

        s_matrix = torch.matmul(q_bar, k_norm.transpose(-2, -1))
        s_hat, _ = torch.max(s_matrix, dim=2)

        # Macro-average across KV heads into a single shared score
        scores = torch.mean(s_hat, dim=1).squeeze(0)  # [S_k]
        cand_tensor = torch.tensor(candidate_indices, device=keys.device, dtype=torch.long)
        cand_scores = scores[cand_tensor]

        _, sorted_positions = torch.sort(cand_scores, descending=True, stable=True)
        chosen_cands = cand_tensor[sorted_positions[:candidate_budget]].tolist()
        final_indices = sorted(sink_indices + chosen_cands + recent_indices)

        per_head = torch.tensor(final_indices, device=keys.device, dtype=torch.long).unsqueeze(0).expand(h_kv, -1)
        t1 = time.perf_counter()

        return SelectorResult(
            retained_indices=final_indices,
            selection_time_ms=(t1 - t0) * 1000.0,
            scores=scores,
            head_retained_indices=[final_indices for _ in range(h_kv)],
            per_head_indices=per_head,
            metadata={"algorithm": "SharedHeadQUOKAApproximation_ablation"},
        )
