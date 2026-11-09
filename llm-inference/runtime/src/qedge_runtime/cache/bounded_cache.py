"""Position-correct bounded KV cache implementation.

Supports:
- 'full': Unbounded contiguous baseline
- 'fifo': First-in, first-out permanent eviction
- 'recency_window': Precisely defined sliding-window recency baseline

Ensures absolute token positions, keys, values, masks, and metadata remain
strictly aligned through every gather and permanent eviction.
All input validation precedes state mutation (failure atomicity).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .contract import (
    EvictionPolicyKind,
    LayerCacheState,
    CacheMemoryStats,
    compute_causal_attention_mask,
)


class BoundedKVCache:
    """Position-correct bounded Key-Value cache for transformer inference.
    
    Invariants:
    1. Absolute token positions are tracked explicitly per resident token.
    2. Batch size is strictly 1 (batched eviction rejected upfront).
    3. Keys, values, and absolute positions remain aligned through every eviction gather.
    4. Resident positions are ordered and unique.
    5. Input validation precedes mutation (failure atomicity).
    6. Classical LRU degeneracy in dense attention is acknowledged:
       'recency_window' provides the mathematically well-defined arrival recency baseline.
    7. Peak temporary allocation accounts for coexisting old, concatenated, gathered,
       and metadata buffers.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        capacity_tokens: int,
        policy: str = "full",
        sink_tokens: int = 4,
        dtype: torch.dtype = torch.float32,
        temporary_memory_allowance_bytes: Optional[int] = None,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if capacity_tokens <= 0 and policy != EvictionPolicyKind.FULL.value:
            raise ValueError(f"capacity_tokens must be >= 1 for bounded policy, got {capacity_tokens}")

        try:
            self.policy_kind = EvictionPolicyKind(policy.lower())
        except ValueError:
            raise ValueError(
                f"Unsupported cache policy: {policy!r}. Supported: {[p.value for p in EvictionPolicyKind]}"
            )

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity_tokens = capacity_tokens if self.policy_kind != EvictionPolicyKind.FULL else 1_000_000
        self.sink_tokens = max(0, sink_tokens)
        self.dtype = dtype
        self.dtype_bytes = torch.tensor([], dtype=dtype).element_size()
        self.defer_eviction: bool = False
        self.temporary_memory_allowance_bytes: Optional[int] = temporary_memory_allowance_bytes

        # Selectors for algorithm-driven permanent eviction
        if self.policy_kind == EvictionPolicyKind.KEYDIFF:
            from ..selectors.keydiff import KeyDiffSelector
            self._selector = KeyDiffSelector(sink_tokens=self.sink_tokens, recent_tokens=1)
        elif self.policy_kind == EvictionPolicyKind.CENTROID_NEIGHBOR:
            from ..selectors.hypotheses import CentroidNeighborSelector
            self._selector = CentroidNeighborSelector(sink_tokens=self.sink_tokens, recent_tokens=1)
        else:
            self._selector = None

        # Allocate per-layer cache states
        self.states: List[LayerCacheState] = [
            LayerCacheState(layer_idx=i) for i in range(num_layers)
        ]

        # Global sequence and memory tracking
        self.total_tokens_seen: int = 0
        self.peak_tokens: int = 0
        self.total_evictions: int = 0
        self.analytical_peak_bytes: int = 0
        self.measured_peak_bytes: Optional[int] = None
        self.temporary_memory_overflow: bool = False
        self.allocator_snapshot_bytes: Optional[int] = None
        self.allocator_snapshot_backend: Optional[str] = None
        self.allocator_snapshot_scope: Optional[str] = None
        self.allocator_snapshot_sampling_point: Optional[str] = None
        self.last_rejected_attempt: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.states = [LayerCacheState(layer_idx=i) for i in range(self.num_layers)]
        self.total_tokens_seen = 0
        self.peak_tokens = 0
        self.total_evictions = 0
        self.defer_eviction = False
        self.analytical_peak_bytes = 0
        self.measured_peak_bytes = None
        self.temporary_memory_overflow = False
        self.allocator_snapshot_bytes = None
        self.allocator_snapshot_backend = None
        self.allocator_snapshot_scope = None
        self.allocator_snapshot_sampling_point = None
        self.last_rejected_attempt = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of currently resident tokens in the specified layer."""
        if 0 <= layer_idx < self.num_layers:
            state = self.states[layer_idx]
            if state.key_tensor is not None:
                return state.key_tensor.shape[2]
        return 0

    def get_absolute_positions(self, layer_idx: int = 0) -> torch.Tensor:
        """Return the absolute positions of resident tokens for the specified layer."""
        if 0 <= layer_idx < self.num_layers:
            return self.states[layer_idx].absolute_positions
        return torch.empty(0, dtype=torch.long)

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a given layer with new Key/Value tensors.
        
        All validation occurs strictly before mutating any state (failure atomicity).
        
        Args:
            layer_idx: Layer index [0 .. num_layers - 1].
            key_states: Tensor of shape [B, H_kv, S_new, D].
            value_states: Tensor of shape [B, H_kv, S_new, D].
            query_positions: Optional 1D/2D integer tensor of absolute positions [S_new].
        
        Returns:
            Tuple of resident (key_states, value_states) after eviction.
        """
        # =====================================================================
        # PHASE 1: PRE-MUTATION VALIDATION (Zero state changes on failure)
        # =====================================================================
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        state = self.states[layer_idx]

        # 1. Validate tensor rank
        if key_states.ndim != 4:
            raise ValueError(
                f"key_states must be a 4D tensor [B, H_kv, S, D], got {key_states.ndim}D shape {key_states.shape}"
            )
        if value_states.ndim != 4:
            raise ValueError(
                f"value_states must be a 4D tensor [B, H_kv, S, D], got {value_states.ndim}D shape {value_states.shape}"
            )

        # 2. Validate batch size (strictly 1)
        b_k, h_k, s_new, d_k = key_states.shape
        b_v, h_v, s_v, d_v = value_states.shape

        if b_k != 1 or b_v != 1:
            raise ValueError(
                f"Batch size must be 1 (got key={b_k}, val={b_v}). "
                "Batched KV eviction is unsupported: independent sequences require separate position tracking."
            )

        # 3. Validate shape agreement
        if (b_k, h_k, s_new, d_k) != (b_v, h_v, s_v, d_v):
            raise ValueError(f"Mismatched key shape {key_states.shape} and value shape {value_states.shape}")

        # 4. Validate configured KV heads and head dimension
        if h_k != self.num_kv_heads:
            raise ValueError(
                f"Configured num_kv_heads is {self.num_kv_heads}, but received key_states with {h_k} heads"
            )
        if d_k != self.head_dim:
            raise ValueError(
                f"Configured head_dim is {self.head_dim}, but received key_states with head_dim {d_k}"
            )

        # 5. Validate dtype contract
        if key_states.dtype != self.dtype:
            raise TypeError(
                f"Configured cache dtype is {self.dtype}, but received key_states dtype={key_states.dtype}"
            )
        if value_states.dtype != self.dtype:
            raise TypeError(
                f"Configured cache dtype is {self.dtype}, but received value_states dtype={value_states.dtype}"
            )

        # 6. Validate device contract
        if key_states.device != value_states.device:
            raise ValueError(
                f"Mismatched device between key ({key_states.device}) and value ({value_states.device})"
            )
        device = key_states.device
        if state.key_tensor is not None and device != state.key_tensor.device:
            raise ValueError(
                f"Incoming tensor device {device} does not match resident cache device {state.key_tensor.device}"
            )

        # 7. Validate position sequence
        if query_positions is not None:
            # Reject non-integral dtypes without silent truncation
            if torch.is_floating_point(query_positions) or query_positions.dtype == torch.bool:
                raise TypeError(
                    f"query_positions must be an integer tensor (int64/int32), got {query_positions.dtype}"
                )
            if query_positions.numel() != s_new:
                raise ValueError(
                    f"query_positions count {query_positions.numel()} != incoming tokens {s_new}"
                )
            incoming_pos = query_positions.view(-1).to(device=device, dtype=torch.long)

            # Reject negative positions
            if torch.any(incoming_pos < 0):
                raise ValueError(f"Positions must be non-negative: {incoming_pos.tolist()}")

            # Invariant: incoming positions must be strictly monotonically increasing
            if s_new > 1 and not torch.all(incoming_pos[1:] > incoming_pos[:-1]):
                raise ValueError(f"Incoming positions must be strictly increasing: {incoming_pos.tolist()}")

            # Invariant: reject out-of-order arrival and replay of previously observed/evicted positions
            first_incoming = int(incoming_pos[0].item())
            if state.max_position_seen >= 0 and first_incoming <= state.max_position_seen:
                raise ValueError(
                    f"Incoming position {first_incoming} must be strictly greater than previous position "
                    f"{state.max_position_seen} (replaying evicted or out-of-order positions without reset is forbidden)"
                )
            if len(state.absolute_positions) > 0:
                last_resident = int(state.absolute_positions[-1].item())
                if first_incoming <= last_resident:
                    raise ValueError(
                        f"Incoming position {first_incoming} must be strictly greater than resident position {last_resident}"
                    )
        else:
            # Sequential arrival advancing past cursor
            cursor = max(state.max_position_seen + 1, 0)
            incoming_pos = torch.arange(cursor, cursor + s_new, device=device, dtype=torch.long)

        # =====================================================================
        # PHASE 2: ALLOCATION-AWARE PEAK MEMORY ACCOUNTING
        # =====================================================================
        s_old = state.key_tensor.shape[2] if state.key_tensor is not None else 0
        s_total = s_old + s_new

        # Determine eviction outcome
        will_evict = (
            self.policy_kind != EvictionPolicyKind.FULL
            and not self.defer_eviction
            and s_total > self.capacity_tokens
        )
        s_post_evict = self.capacity_tokens if will_evict else s_total

        bytes_per_tok_layer = 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes
        old_kv_bytes = state.actual_payload_bytes()
        incoming_kv_bytes = s_new * bytes_per_tok_layer
        concat_kv_bytes = s_total * bytes_per_tok_layer
        gathered_kv_bytes = s_post_evict * bytes_per_tok_layer if will_evict else 0

        # Coexisting buffers during this layer update:
        # old_resident + incoming + concatenated + gathered (if evicted)
        layer_coexisting_kv = old_kv_bytes + incoming_kv_bytes + concat_kv_bytes + gathered_kv_bytes

        # Other layers currently holding their steady-state resident tensors
        other_layers_kv = sum(
            s.actual_payload_bytes() for i, s in enumerate(self.states) if i != layer_idx
        )

        # Metadata tensors coexisting at this step
        total_resident_positions = sum(len(s.absolute_positions) for s in self.states)
        step_metadata_bytes = (
            (total_resident_positions + s_new + s_total + (2 * s_post_evict if will_evict else 0)) * 8
            + 64
        )

        candidate_peak = other_layers_kv + layer_coexisting_kv + step_metadata_bytes

        # Verify temporary memory allowance before committing or mutating any state
        if self.temporary_memory_allowance_bytes is not None:
            if candidate_peak > self.temporary_memory_allowance_bytes:
                self.last_rejected_attempt = {
                    "operation": "update",
                    "layer_idx": layer_idx,
                    "candidate_peak_bytes": candidate_peak,
                    "allowance_bytes": self.temporary_memory_allowance_bytes,
                }
                raise MemoryError(
                    f"Temporary memory allowance exceeded: step peak {candidate_peak:,} bytes "
                    f"> allowance {self.temporary_memory_allowance_bytes:,} bytes"
                )

        # =====================================================================
        # PHASE 3: STATE MUTATION & EVICTION
        # =====================================================================
        if state.key_tensor is None:
            k_cat = key_states
            v_cat = value_states
            pos_cat = incoming_pos
        else:
            k_cat = torch.cat([state.key_tensor, key_states], dim=2)
            v_cat = torch.cat([state.value_tensor, value_states], dim=2)
            pos_cat = torch.cat([state.absolute_positions.to(device=device), incoming_pos])

        self.peak_tokens = max(self.peak_tokens, s_total)

        if will_evict:
            keep_indices = self._select_retained_indices(pos_cat, s_total, keys=k_cat, values=v_cat)
            idx_tensor = torch.tensor(keep_indices, device=device, dtype=torch.long)

            # Gather keys, values, and absolute positions in lockstep
            k_resident = torch.index_select(k_cat, 2, idx_tensor)
            v_resident = torch.index_select(v_cat, 2, idx_tensor)
            pos_resident = torch.index_select(pos_cat, 0, idx_tensor)

            num_evicted = s_total - len(keep_indices)
            state.evictions_count += num_evicted
            if layer_idx == 0:
                self.total_evictions += num_evicted
        else:
            k_resident = k_cat
            v_resident = v_cat
            pos_resident = pos_cat

        # Commit state updates
        state.key_tensor = k_resident
        state.value_tensor = v_resident
        state.absolute_positions = pos_resident.detach().cpu()
        state.tokens_seen += s_new
        state.max_position_seen = int(incoming_pos[-1].item())

        if layer_idx == 0:
            self.total_tokens_seen += s_new

        # Commit successful analytical peak telemetry (monotonic)
        self.analytical_peak_bytes = max(self.analytical_peak_bytes, candidate_peak)

        # Record global allocator observation separately if available, clearly scoped as global_process
        if device.type == "cuda" and torch.cuda.is_available():
            self.allocator_snapshot_bytes = torch.cuda.memory_allocated(device)
            self.allocator_snapshot_backend = "cuda"
            self.allocator_snapshot_scope = "global_process"
            self.allocator_snapshot_sampling_point = "post_update"
        elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                self.allocator_snapshot_bytes = torch.mps.current_allocated_memory()
                self.allocator_snapshot_backend = "mps"
                self.allocator_snapshot_scope = "global_process"
                self.allocator_snapshot_sampling_point = "post_update"
            except Exception:
                pass

        ok, err = state.verify_alignment()
        if not ok:
            raise RuntimeError(f"Cache state alignment invariant violated: {err}")

        return k_resident, v_resident

    def evict_to_capacity(self) -> int:
        """Prune resident tokens across all layers down to capacity_tokens.
        
        Preflights the entire multi-layer prune operation to compute candidate peak
        memory and validate temporary allowance BEFORE mutating any layer.
        
        Returns total number of evicted tokens in this prune.
        """
        if self.policy_kind == EvictionPolicyKind.FULL:
            return 0

        prune_plans: Dict[int, List[int]] = {}
        for l in range(self.num_layers):
            st = self.states[l]
            if st.key_tensor is None:
                continue
            s_cur = st.key_tensor.shape[2]
            if s_cur > self.capacity_tokens:
                keep_idx = self._select_retained_indices(st.absolute_positions, s_cur, keys=st.key_tensor, values=st.value_tensor)
                prune_plans[l] = keep_idx

        if not prune_plans:
            return 0

        bytes_per_tok_layer = 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

        # Preflight memory calculation across sequential layer prunes
        candidate_peak = 0
        for l in prune_plans:
            # Earlier layers (< l) have already been pruned
            kv_earlier = sum(
                (len(prune_plans[j]) if j in prune_plans else self.states[j].key_tensor.shape[2]) * bytes_per_tok_layer
                for j in range(l)
                if self.states[j].key_tensor is not None
            )
            # Layer l has old resident + gathered resident coexisting
            old_l = self.states[l].actual_payload_bytes()
            gathered_l = len(prune_plans[l]) * bytes_per_tok_layer
            kv_l = old_l + gathered_l

            # Later layers (> l) are not yet pruned
            kv_later = sum(
                self.states[j].actual_payload_bytes()
                for j in range(l + 1, self.num_layers)
            )

            pos_earlier = sum(
                (len(prune_plans[j]) if j in prune_plans else len(self.states[j].absolute_positions)) * 8
                for j in range(l)
            )
            old_pos_l = len(self.states[l].absolute_positions) * 8
            gathered_pos_l = len(prune_plans[l]) * 8
            idx_tensor_l = len(prune_plans[l]) * 8
            pos_l = old_pos_l + gathered_pos_l + idx_tensor_l

            pos_later = sum(
                len(self.states[j].absolute_positions) * 8
                for j in range(l + 1, self.num_layers)
            )

            step_metadata = pos_earlier + pos_l + pos_later + 64
            step_peak = kv_earlier + kv_l + kv_later + step_metadata
            candidate_peak = max(candidate_peak, step_peak)

        # Allowance check BEFORE ANY layer is mutated
        if self.temporary_memory_allowance_bytes is not None and candidate_peak > self.temporary_memory_allowance_bytes:
            self.last_rejected_attempt = {
                "operation": "evict_to_capacity",
                "candidate_peak_bytes": candidate_peak,
                "allowance_bytes": self.temporary_memory_allowance_bytes,
            }
            raise MemoryError(
                f"Temporary memory allowance exceeded in evict_to_capacity: "
                f"candidate peak {candidate_peak:,} bytes > allowance {self.temporary_memory_allowance_bytes:,} bytes"
            )

        # Execution phase (atomic across all layers)
        evicted_count = 0
        for l, keep_indices in prune_plans.items():
            state = self.states[l]
            idx_tensor = torch.tensor(keep_indices, device=state.key_tensor.device, dtype=torch.long)
            s_before = state.key_tensor.shape[2]
            state.key_tensor = torch.index_select(state.key_tensor, 2, idx_tensor)
            state.value_tensor = torch.index_select(state.value_tensor, 2, idx_tensor)
            state.absolute_positions = torch.index_select(
                state.absolute_positions, 0, torch.tensor(keep_indices, dtype=torch.long)
            )

            num_evicted = s_before - len(keep_indices)
            state.evictions_count += num_evicted
            if l == 0:
                evicted_count = num_evicted
                self.total_evictions += num_evicted

            ok, err = state.verify_alignment()
            if not ok:
                raise RuntimeError(f"Cache state alignment invariant violated after prune: {err}")

        # Commit peak telemetry only on success
        self.analytical_peak_bytes = max(self.analytical_peak_bytes, candidate_peak)
        return evicted_count

    def evict_positions(self, positions_to_evict: List[int]) -> int:
        """Explicitly evict specific absolute positions across all layers (for interior eviction testing).
        
        Preflights the entire multi-layer eviction operation to compute candidate peak
        memory and validate temporary allowance BEFORE mutating any layer.
        
        Returns number of evicted positions.
        """
        if not positions_to_evict:
            return 0

        target_set = set(positions_to_evict)
        evict_plans: Dict[int, List[int]] = {}

        for l in range(self.num_layers):
            st = self.states[l]
            if st.key_tensor is None or len(st.absolute_positions) == 0:
                continue
            cur_pos = st.absolute_positions.tolist()
            keep_idx = [i for i, pos in enumerate(cur_pos) if pos not in target_set]
            if len(keep_idx) < len(cur_pos):
                evict_plans[l] = keep_idx

        if not evict_plans:
            return 0

        bytes_per_tok_layer = 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

        # Preflight memory calculation across sequential layer evictions
        candidate_peak = 0
        for l in evict_plans:
            kv_earlier = sum(
                (len(evict_plans[j]) if j in evict_plans else self.states[j].key_tensor.shape[2]) * bytes_per_tok_layer
                for j in range(l)
                if self.states[j].key_tensor is not None
            )
            old_l = self.states[l].actual_payload_bytes()
            gathered_l = len(evict_plans[l]) * bytes_per_tok_layer
            kv_l = old_l + gathered_l

            kv_later = sum(
                self.states[j].actual_payload_bytes()
                for j in range(l + 1, self.num_layers)
            )

            pos_earlier = sum(
                (len(evict_plans[j]) if j in evict_plans else len(self.states[j].absolute_positions)) * 8
                for j in range(l)
            )
            old_pos_l = len(self.states[l].absolute_positions) * 8
            gathered_pos_l = len(evict_plans[l]) * 8
            idx_tensor_l = len(evict_plans[l]) * 8
            pos_l = old_pos_l + gathered_pos_l + idx_tensor_l

            pos_later = sum(
                len(self.states[j].absolute_positions) * 8
                for j in range(l + 1, self.num_layers)
            )

            step_metadata = pos_earlier + pos_l + pos_later + 64
            step_peak = kv_earlier + kv_l + kv_later + step_metadata
            candidate_peak = max(candidate_peak, step_peak)

        # Allowance check BEFORE ANY layer is mutated
        if self.temporary_memory_allowance_bytes is not None and candidate_peak > self.temporary_memory_allowance_bytes:
            self.last_rejected_attempt = {
                "operation": "evict_positions",
                "candidate_peak_bytes": candidate_peak,
                "allowance_bytes": self.temporary_memory_allowance_bytes,
            }
            raise MemoryError(
                f"Temporary memory allowance exceeded in evict_positions: "
                f"candidate peak {candidate_peak:,} bytes > allowance {self.temporary_memory_allowance_bytes:,} bytes"
            )

        # Execution phase (atomic across all layers)
        evicted_count = 0
        for l, keep_indices in evict_plans.items():
            state = self.states[l]
            s_before = len(state.absolute_positions)
            idx_tensor = torch.tensor(keep_indices, device=state.key_tensor.device, dtype=torch.long)
            state.key_tensor = torch.index_select(state.key_tensor, 2, idx_tensor)
            state.value_tensor = torch.index_select(state.value_tensor, 2, idx_tensor)
            state.absolute_positions = torch.index_select(
                state.absolute_positions, 0, torch.tensor(keep_indices, dtype=torch.long)
            )
            num_evicted = s_before - len(keep_indices)
            state.evictions_count += num_evicted
            if l == 0:
                evicted_count = num_evicted
                self.total_evictions += num_evicted

            ok, err = state.verify_alignment()
            if not ok:
                raise RuntimeError(f"Cache state alignment invariant violated after evict_positions: {err}")

        # Commit peak telemetry only on success
        self.analytical_peak_bytes = max(self.analytical_peak_bytes, candidate_peak)
        return evicted_count

    def predict_retained_positions(self, incoming_positions: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """Predict absolute positions resident after an update with incoming_positions."""
        incoming = incoming_positions.view(-1).cpu()
        state = self.states[layer_idx]
        if len(state.absolute_positions) == 0:
            pos_cat = incoming
        else:
            pos_cat = torch.cat([state.absolute_positions, incoming])

        total_count = len(pos_cat)
        if (
            self.policy_kind != EvictionPolicyKind.FULL
            and not self.defer_eviction
            and total_count > self.capacity_tokens
        ):
            keep_indices = self._select_retained_indices(pos_cat, total_count)
            return pos_cat[keep_indices]
        return pos_cat

    def build_causal_mask(
        self,
        query_positions: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Build causal attention mask for query_positions attending to resident keys after update."""
        q_pos = query_positions.view(-1)
        if device is None:
            device = q_pos.device
        pred_key_pos = self.predict_retained_positions(q_pos, layer_idx=layer_idx).to(device=device)
        return compute_causal_attention_mask(q_pos, pred_key_pos, dtype=dtype, device=device)

    def _select_retained_indices(
        self,
        positions: torch.Tensor,
        total_count: int,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Select strictly sorted indices to retain according to policy.
        
        Preserves alignment between absolute positions and tensor slices.
        Always reserves at least 1 slot for the current/most recent token.
        """
        capacity = self.capacity_tokens
        if total_count <= capacity:
            return list(range(total_count))

        if capacity <= 1:
            # Capacity 1: retain the single most recent token
            return [total_count - 1]

        # For capacity > 1, reserve at least 1 slot for recent/current tokens
        sink_count = min(self.sink_tokens, capacity - 1)
        recent_budget = capacity - sink_count

        if self.policy_kind in (EvictionPolicyKind.KEYDIFF, EvictionPolicyKind.CENTROID_NEIGHBOR):
            if keys is not None and self._selector is not None:
                res = self._selector.select_indices(keys=keys, budget=capacity, values=values)
                return res.retained_indices

        if self.policy_kind in (EvictionPolicyKind.FIFO, EvictionPolicyKind.RECENCY_WINDOW) or keys is None:
            # 1. Protect attention sink tokens (initial sink_count tokens by index)
            sink_indices = list(range(sink_count))

            # 2. Select recent tokens from the tail of the sequence (including the current token)
            tail_start = max(sink_count, total_count - recent_budget)
            recent_indices = list(range(tail_start, total_count))

            # Combine and eliminate potential overlap, preserving strict ascending index order
            retained = sorted(set(sink_indices + recent_indices))
            return retained[:capacity]

        # Default fallback: last capacity tokens
        return list(range(total_count - capacity, total_count))

    def get_memory_stats(self) -> CacheMemoryStats:
        """Compute accurate memory telemetry derived directly from live resident tensors."""
        resident_tokens = self.get_seq_length(0)
        actual_payload_bytes = sum(s.actual_payload_bytes() for s in self.states)

        return CacheMemoryStats(
            dtype_bytes=self.dtype_bytes,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            resident_tokens=resident_tokens,
            capacity_tokens=self.capacity_tokens,
            peak_tokens=self.peak_tokens,
            total_tokens_seen=self.total_tokens_seen,
            total_evictions=self.total_evictions,
            actual_resident_payload_bytes=actual_payload_bytes,
            analytical_peak_bytes=self.analytical_peak_bytes,
            measured_peak_bytes=self.measured_peak_bytes,
            temporary_memory_allowance_bytes=self.temporary_memory_allowance_bytes,
            temporary_memory_overflow=self.temporary_memory_overflow,
            allocator_snapshot_bytes=self.allocator_snapshot_bytes,
            allocator_snapshot_backend=self.allocator_snapshot_backend,
            allocator_snapshot_scope=self.allocator_snapshot_scope,
            allocator_snapshot_sampling_point=self.allocator_snapshot_sampling_point,
            last_rejected_attempt=self.last_rejected_attempt,
        )
