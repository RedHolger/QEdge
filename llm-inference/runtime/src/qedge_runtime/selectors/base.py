"""Base abstractions for KV cache selectors and eviction policies.

Distinguishes:
1. Permanent Eviction: Drops keys, values, and position metadata permanently from cache storage.
2. Temporary Attention Selection: Gathers a sparse subset of active keys/values per attention
   forward step without mutating underlying cache residency.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
import enum
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


class SelectionKind(str, enum.Enum):
    """Scope of token selection."""
    PERMANENT_EVICTION = "permanent_eviction"            # Frees memory permanently
    TEMPORARY_ATTENTION_SELECTION = "temporary_selection" # Sparse attention for dot product only


@dataclass
class SelectorResult:
    """Outcome of a token selection evaluation with precise latency telemetry."""
    retained_indices: List[int]
    selection_time_ms: float
    gather_time_ms: float = 0.0
    scores: Optional[torch.Tensor] = None
    head_retained_indices: Optional[List[List[int]]] = None
    per_head_indices: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_overhead_ms(self) -> float:
        """Total overhead of selection logic and tensor gather."""
        return self.selection_time_ms + self.gather_time_ms


class BaseSelector(abc.ABC):
    """Abstract base class for KV selectors."""

    def __init__(
        self,
        name: str,
        kind: SelectionKind,
        sink_tokens: int = 4,
        recent_tokens: int = 1,
        epsilon: float = 1e-12,
    ) -> None:
        self.name = name
        self.kind = kind
        self.sink_tokens = max(0, sink_tokens)
        self.recent_tokens = max(0, recent_tokens)
        self.epsilon = epsilon

    def get_protected_and_candidate_indices(
        self,
        total_count: int,
        budget: int,
    ) -> Tuple[List[int], List[int], List[int], int]:
        """Partition token indices into sinks, candidates, and recent window.
        
        Args:
            total_count: Total number of resident tokens.
            budget: Number of tokens to retain.
            
        Returns:
            Tuple of (sink_indices, candidate_indices, recent_indices, candidate_budget).
        """
        if total_count <= budget:
            return list(range(total_count)), [], [], 0

        # Pure paper mode with no protected sinks or recent window
        if self.sink_tokens == 0 and self.recent_tokens == 0:
            return [], list(range(total_count)), [], budget

        if budget <= 1:
            if self.recent_tokens > 0:
                # Budget 1 with protected recent window: strictly retain newest token
                return [], [], [total_count - 1], 0
            else:
                return [0], [], [], 0

        # Protect sinks (initial tokens)
        sink_count = min(self.sink_tokens, budget - 1)
        sink_indices = list(range(sink_count))

        # Protect recent window (tail tokens)
        recent_budget = min(self.recent_tokens, budget - sink_count)
        recent_start = max(sink_count, total_count - recent_budget)
        recent_indices = list(range(recent_start, total_count))

        # Candidate indices are the interior tokens subject to selection
        candidate_indices = [
            i for i in range(total_count)
            if i not in sink_indices and i not in recent_indices
        ]

        candidate_budget = max(0, budget - len(sink_indices) - len(recent_indices))
        return sink_indices, candidate_indices, recent_indices, candidate_budget

    @abc.abstractmethod
    def select_indices(
        self,
        keys: torch.Tensor,
        budget: int,
        queries: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> SelectorResult:
        """Evaluate selection scores and return selected indices with timings.
        
        Args:
            keys: Key tensor of shape [B, H_kv, S, D].
            budget: Number of tokens to retain.
            queries: Optional query tensor of shape [B, H_q, S_q, D] for query-aware selectors.
            values: Optional value tensor of shape [B, H_kv, S, D] for value-aware selectors.
            
        Returns:
            SelectorResult with sorted retained indices and execution time.
        """
        pass
