"""Formal KV Cache Contract Specification for QEdge Real-Model Runtime.

Defines the mathematical invariants, position semantics, memory accounting,
and eviction interfaces for bounded KV cache implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


class EvictionPolicyKind(str, enum.Enum):
    """Supported eviction policies in Stage 06/07."""
    FULL = "full"                      # Unbounded cache baseline (no eviction)
    FIFO = "fifo"                      # First-in, first-out permanent eviction
    RECENCY_WINDOW = "recency_window"  # Strictly defined sliding recency window
    KEYDIFF = "keydiff"                # KeyDiff mean-anchor key diversity permanent eviction
    CENTROID_NEIGHBOR = "centroid_neighbor"  # Document-derived heuristic permanent eviction


@dataclass
class CacheMemoryStats:
    """Accurate allocation-aware memory accounting for resident and transient allocations."""
    dtype_bytes: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    resident_tokens: int
    capacity_tokens: int
    peak_tokens: int = 0
    total_tokens_seen: int = 0
    total_evictions: int = 0
    actual_resident_payload_bytes: int = 0
    analytical_peak_bytes: int = 0
    measured_peak_bytes: Optional[int] = None
    temporary_memory_allowance_bytes: Optional[int] = None
    temporary_memory_overflow: bool = False
    allocator_snapshot_bytes: Optional[int] = None
    allocator_snapshot_backend: Optional[str] = None
    allocator_snapshot_scope: Optional[str] = None
    allocator_snapshot_sampling_point: Optional[str] = None
    last_rejected_attempt: Optional[Dict[str, Any]] = None

    @property
    def bytes_per_token_per_layer(self) -> int:
        """K and V bytes for one token in one layer: 2 * num_kv_heads * head_dim * dtype_bytes."""
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def bytes_per_token_model(self) -> int:
        """K and V bytes for one token across all model layers."""
        return self.bytes_per_token_model_per_layer * self.num_layers if hasattr(self, "bytes_per_token_model_per_layer") else self.bytes_per_token_per_layer * self.num_layers

    @property
    def resident_payload_bytes(self) -> int:
        """Actual steady-state K+V payload bytes resident across all layers."""
        if self.actual_resident_payload_bytes > 0:
            return self.actual_resident_payload_bytes
        return self.resident_tokens * self.bytes_per_token_model

    @property
    def capacity_payload_bytes(self) -> int:
        """Maximum K+V payload bytes at declared capacity."""
        return self.capacity_tokens * self.bytes_per_token_model

    @property
    def metadata_bytes(self) -> int:
        """Memory consumed by position tracking and eviction metadata."""
        # 8 bytes per resident position (int64) per layer + fixed control counters
        position_tensor_bytes = self.num_layers * self.resident_tokens * 8
        counters_bytes = 64
        return position_tensor_bytes + counters_bytes

    @property
    def is_measured_peak(self) -> bool:
        """True only if an authoritative cache-scoped physical measurement is recorded."""
        return self.measured_peak_bytes is not None

    @property
    def peak_temporary_bytes(self) -> int:
        """Authoritative cache peak temporary allocation (analytical coexisting buffer bound).
        
        Analytical cache accounting is authoritative unless a properly scoped
        cache measurement exists. Unrelated global allocator snapshots do not
        replace the analytical cache bound.
        """
        if self.analytical_peak_bytes > 0:
            return self.analytical_peak_bytes
        if self.measured_peak_bytes is not None:
            return self.measured_peak_bytes
        if self.resident_tokens == 0:
            return 0
        # Fallback conservative baseline: resident payload + metadata
        return self.resident_payload_bytes + self.metadata_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resident_tokens": self.resident_tokens,
            "capacity_tokens": self.capacity_tokens,
            "peak_tokens": self.peak_tokens,
            "total_tokens_seen": self.total_tokens_seen,
            "total_evictions": self.total_evictions,
            "resident_payload_bytes": self.resident_payload_bytes,
            "actual_resident_payload_bytes": self.actual_resident_payload_bytes,
            "capacity_payload_bytes": self.capacity_payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "peak_temporary_bytes": self.peak_temporary_bytes,
            "analytical_peak_bytes": self.analytical_peak_bytes,
            "measured_peak_bytes": self.measured_peak_bytes,
            "is_measured_peak": self.measured_peak_bytes is not None,
            "temporary_memory_allowance_bytes": self.temporary_memory_allowance_bytes,
            "temporary_memory_overflow": self.temporary_memory_overflow,
            "allocator_snapshot": (
                {
                    "bytes": self.allocator_snapshot_bytes,
                    "backend": self.allocator_snapshot_backend,
                    "scope": self.allocator_snapshot_scope,
                    "sampling_point": self.allocator_snapshot_sampling_point,
                }
                if self.allocator_snapshot_bytes is not None
                else None
            ),
            "last_rejected_attempt": self.last_rejected_attempt,
            "bytes_per_token_per_layer": self.bytes_per_token_per_layer,
            "bytes_per_token_model": self.bytes_per_token_model,
        }


@dataclass
class LayerCacheState:
    """Position-correct storage for a single transformer layer's Key/Value tensors."""
    layer_idx: int
    key_tensor: Optional[torch.Tensor] = None       # Shape: [1, H_kv, S_resident, D]
    value_tensor: Optional[torch.Tensor] = None     # Shape: [1, H_kv, S_resident, D]
    absolute_positions: torch.Tensor = field(       # Shape: [S_resident], dtype=torch.long
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    tokens_seen: int = 0
    evictions_count: int = 0
    max_position_seen: int = -1                     # Invariant: cursor tracking highest position observed

    def actual_payload_bytes(self) -> int:
        """Return actual live K and V payload bytes for this layer."""
        k_bytes = self.key_tensor.numel() * self.key_tensor.element_size() if self.key_tensor is not None else 0
        v_bytes = self.value_tensor.numel() * self.value_tensor.element_size() if self.value_tensor is not None else 0
        return k_bytes + v_bytes

    def verify_alignment(self) -> Tuple[bool, Optional[str]]:
        """Verify structural and position alignment invariants."""
        if self.key_tensor is None and self.value_tensor is None:
            if len(self.absolute_positions) != 0:
                return False, f"Layer {self.layer_idx}: empty KV but non-empty positions ({len(self.absolute_positions)})"
            return True, None

        if self.key_tensor is None or self.value_tensor is None:
            return False, f"Layer {self.layer_idx}: mismatched None between key and value tensors"

        k_shape = self.key_tensor.shape
        v_shape = self.value_tensor.shape

        if k_shape != v_shape:
            return False, f"Layer {self.layer_idx}: key shape {k_shape} != value shape {v_shape}"

        if k_shape[0] != 1:
            return False, f"Layer {self.layer_idx}: batch size must be 1, got {k_shape[0]}"

        s_resident = k_shape[2]
        p_len = len(self.absolute_positions)
        if s_resident != p_len:
            return False, f"Layer {self.layer_idx}: resident tokens {s_resident} != positions length {p_len}"

        # Positions must be strictly ordered and unique
        if p_len > 1:
            diffs = self.absolute_positions[1:] - self.absolute_positions[:-1]
            if not torch.all(diffs > 0):
                return False, f"Layer {self.layer_idx}: positions are not strictly monotonically increasing: {self.absolute_positions.tolist()}"

        # Check cursor alignment
        if p_len > 0:
            last_pos = int(self.absolute_positions[-1].item())
            if last_pos > self.max_position_seen:
                return False, f"Layer {self.layer_idx}: resident position {last_pos} exceeds max_position_seen {self.max_position_seen}"

        return True, None


def compute_causal_attention_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute exact causal mask for non-contiguous resident key positions.
    
    query_positions: [S_q] absolute positions of query tokens.
    key_positions: [S_k] absolute positions of resident key tokens.
    Returns: [1, 1, S_q, S_k] float mask where allowed attention is 0.0 and blocked is -inf.
    
    Causal Rule: A query at position P_q may attend to key at position P_k iff P_k <= P_q.
    Future keys (P_k > P_q) receive -inf, guaranteeing 0.0 softmax attention weight.
    """
    if device is None:
        device = query_positions.device

    q_pos = query_positions.to(device=device, dtype=torch.long).unsqueeze(1)  # [S_q, 1]
    k_pos = key_positions.to(device=device, dtype=torch.long).unsqueeze(0)    # [1, S_k]

    # Causal condition: k_pos <= q_pos
    allowed = (k_pos <= q_pos)  # [S_q, S_k] boolean

    mask = torch.zeros((1, 1, len(query_positions), len(key_positions)), dtype=dtype, device=device)
    mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask
