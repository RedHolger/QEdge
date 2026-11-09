"""Selectors and sparse attention policies for QEdge LLM Runtime."""

from .base import BaseSelector, SelectionKind, SelectorResult
from .keydiff import KeyDiffSelector
from .quoka import QUOKASelector, SharedHeadQUOKAApproximation
from .hypotheses import CentroidNeighborSelector, AttentionValueDecaySelector
from .reference import reference_keydiff, reference_quoka, paper_quoka_oracle, masked_attention_reference
from .context import SparseAttentionContext, SparseAttentionStats, calculate_selector_intermediates_bytes, is_blocking_sentinel

__all__ = [
    "BaseSelector",
    "SelectionKind",
    "SelectorResult",
    "KeyDiffSelector",
    "QUOKASelector",
    "SharedHeadQUOKAApproximation",
    "CentroidNeighborSelector",
    "AttentionValueDecaySelector",
    "reference_keydiff",
    "reference_quoka",
    "paper_quoka_oracle",
    "masked_attention_reference",
    "SparseAttentionContext",
    "SparseAttentionStats",
    "calculate_selector_intermediates_bytes",
    "is_blocking_sentinel",
]
