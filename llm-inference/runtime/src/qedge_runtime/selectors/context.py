"""Execution context and forward hook manager for temporary sparse attention selection.

Safely wraps transformer attention modules during generation passes to:
1. Evaluate query-dependent or heuristic attention selectors (e.g. QUOKA).
2. Gather sparse active Key and Value subsets dynamically per attention step per KV head.
3. Build causal masks aligned with absolute query and key positions, preserving supplied attention masks.
4. Convert 2D padding masks (bool, int, float) into standard additive masks expanded across query rows.
5. Define supported 4D additive mask representations, including finite minimum sentinels.
6. Detect fully blocked rows explicitly and produce strictly zero attention weights and zero attention output.
7. Measure exact selection and gather overheads with proper accelerator synchronization.
8. Track transient selector intermediates and active gather allocations separately from resident KV cache.
9. Reliably restore original forward methods upon exit.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from .base import BaseSelector, SelectionKind
from .quoka import QUOKASelector
from .hypotheses import AttentionValueDecaySelector


def _sync(device: torch.device) -> None:
    """Synchronize accelerator device for authoritative timing measurements."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _prepare_additive_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    s_q: int,
    s_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert 2D padding mask or 3D/4D mask into standard 4D additive mask [batch, heads, s_q, s_k].
    
    Supported representations:
    1. 2D padding mask [batch, s_k] (bool, integer, or float):
       - 1 / True: allowed (converted to additive 0.0)
       - 0 / False: blocked (converted to additive -inf)
       - Explicitly expanded across query dimension: [batch, 1, s_q, s_k]
    2. 3D mask [batch, s_q, s_k]:
       - Expanded to [batch, 1, s_q, s_k]
    3. 4D additive mask [batch, num_heads, s_q, s_k]:
       - Allowed values are near 0.0
       - Blocked values are <= -1000.0, -inf, or finite minimum sentinels (e.g. finfo.min).
    """
    if attention_mask is None:
        return None

    if attention_mask.dim() == 2:
        # [batch, s_k] padding mask: 1=allowed, 0=blocked
        if attention_mask.dtype == torch.bool:
            allowed = attention_mask
        elif not torch.is_floating_point(attention_mask):
            allowed = attention_mask != 0
        else:
            # Floating point padding mask: 1.0 is allowed, 0.0 is blocked
            allowed = attention_mask > 0.5

        add_mask = torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), float("-inf"), dtype=dtype, device=device),
        )
        # Expand across query rows: [batch, 1, s_q, s_k]
        return add_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, s_q, s_k)

    elif attention_mask.dim() == 3:
        # [batch, s_q, s_k] -> [batch, 1, s_q, s_k]
        return attention_mask.unsqueeze(1).to(device=device, dtype=dtype)

    elif attention_mask.dim() == 4:
        # [batch, num_heads, s_q, s_k]
        return attention_mask.to(device=device, dtype=dtype)

    else:
        raise ValueError(f"Unsupported attention mask dimensions: {attention_mask.dim()}")


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
    """Determine boolean allowed status from an additive mask tensor.

    A position is allowed if it is NOT a documented blocking sentinel.
    Finite additive biases (e.g. -2.0, 0.0, +1.5) are allowed and preserved.
    """
    return ~is_blocking_sentinel(mask_tensor)


def calculate_selector_intermediates_bytes(
    h_q: int,
    h_kv: int,
    s_q: int,
    s_k: int,
    n_rep_q: int,
    b_sa: int,
    head_dim: int,
    elem_bytes: int,
) -> int:
    """Documented conservative accounting of coexisting selector buffers during scoring.
    
    Coexisting allocations in QUOKA scoring:
    - Normalized candidate keys: H_kv * S_k * D * elem_bytes
    - Representative queries & normalized rep queries: 2 * H_q * N_rep_q * D * elem_bytes
    - Group-averaged queries Q_bar: H_kv * N_rep_q * D * elem_bytes
    - Interaction matrix S = Q_bar @ K^T: H_kv * N_rep_q * S_k * elem_bytes
    - Max-pooled scores S_hat: H_kv * S_k * elem_bytes
    - Representative query subselection buffers (if S_q > N_rep_q): H_q * (S_q + D) * elem_bytes
    - Selected index tensors: H_kv * B_sa * 8 bytes (int64)
    """
    subselect_bytes = (h_q * (s_q + head_dim) * elem_bytes) if s_q > n_rep_q else 0
    k_norm_bytes = h_kv * s_k * head_dim * elem_bytes
    q_rep_bytes = 2 * h_q * n_rep_q * head_dim * elem_bytes
    q_bar_bytes = h_kv * n_rep_q * head_dim * elem_bytes
    s_matrix_bytes = h_kv * n_rep_q * s_k * elem_bytes
    s_hat_bytes = h_kv * s_k * elem_bytes
    indices_bytes = h_kv * b_sa * 8

    return (
        k_norm_bytes
        + q_rep_bytes
        + q_bar_bytes
        + s_matrix_bytes
        + s_hat_bytes
        + subselect_bytes
        + indices_bytes
    )


class SparseAttentionStats:
    """Telemetry collector for temporary attention selection."""

    def __init__(self) -> None:
        self.total_selector_time_ms: float = 0.0
        self.total_gather_time_ms: float = 0.0
        self.attention_pairs_evaluated: int = 0
        self.dense_attention_pairs: int = 0
        self.steps_evaluated: int = 0
        self.last_step_retained_indices: Optional[List[int]] = None
        self.last_step_head_retained_indices: Optional[List[List[int]]] = None
        self.per_layer_calls: int = 0
        self.peak_selector_intermediates_bytes: int = 0
        self.peak_gathered_active_bytes: int = 0
        self.peak_gqa_expanded_bytes: int = 0
        self.peak_attention_mask_bytes: int = 0

    @property
    def sparsity_ratio(self) -> float:
        """Fraction of attention pairs pruned relative to dense baseline."""
        if self.dense_attention_pairs == 0:
            return 0.0
        return 1.0 - (self.attention_pairs_evaluated / self.dense_attention_pairs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_selector_time_ms": round(self.total_selector_time_ms, 3),
            "total_gather_time_ms": round(self.total_gather_time_ms, 3),
            "total_selection_overhead_ms": round(self.total_selector_time_ms + self.total_gather_time_ms, 3),
            "attention_pairs_evaluated": self.attention_pairs_evaluated,
            "dense_attention_pairs": self.dense_attention_pairs,
            "sparsity_ratio": round(self.sparsity_ratio, 4),
            "steps_evaluated": self.steps_evaluated,
            "per_layer_calls": self.per_layer_calls,
            "peak_selector_intermediates_bytes": self.peak_selector_intermediates_bytes,
            "peak_gathered_active_bytes": self.peak_gathered_active_bytes,
            "peak_gqa_expanded_bytes": self.peak_gqa_expanded_bytes,
            "peak_attention_mask_bytes": self.peak_attention_mask_bytes,
        }


class SparseAttentionContext:
    """Context manager for temporary sparse attention forward passes."""

    def __init__(
        self,
        model: nn.Module,
        selector_name: str,
        budget: Optional[int] = None,
        ratio: Optional[float] = None,
        sink_tokens: int = 4,
        recent_tokens: int = 1,
    ) -> None:
        self.model = model
        self.selector_name = selector_name.lower()
        self.budget = budget
        self.ratio = ratio
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens

        if self.selector_name == "quoka":
            self.selector: BaseSelector = QUOKASelector(
                sink_tokens=sink_tokens,
                recent_tokens=recent_tokens,
            )
        elif self.selector_name == "attention_value_decay":
            self.selector = AttentionValueDecaySelector(
                sink_tokens=sink_tokens,
                recent_tokens=recent_tokens,
            )
        else:
            raise ValueError(
                f"Unsupported attention selector: {selector_name!r}. "
                f"Supported: ['quoka', 'attention_value_decay']"
            )

        self.stats = SparseAttentionStats()
        self._original_forwards: List[Tuple[nn.Module, Any]] = []

    def _compute_step_budget(self, s_resident: int) -> int:
        """Determine token budget for the current attention step."""
        if self.budget is not None:
            return max(1, min(s_resident, self.budget))
        if self.ratio is not None:
            return max(1, min(s_resident, int(round(s_resident * self.ratio))))
        return s_resident

    def __enter__(self) -> "SparseAttentionContext":
        self._original_forwards.clear()
        layers = getattr(getattr(self.model, "model", self.model), "layers", None)
        if layers is None:
            raise ValueError("Model does not expose decoder layers for attention hooking")

        for layer in layers:
            attn_mod = layer.self_attn
            orig_forward = attn_mod.forward
            self._original_forwards.append((attn_mod, orig_forward))

            def _make_sparse_forward(mod: nn.Module, old_f: Any):
                def _forward(
                    hidden_states: torch.Tensor,
                    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    past_key_values: Optional[Any] = None,
                    **kwargs: Any,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, mod.head_dim)
                    device = hidden_states.device
                    dtype = hidden_states.dtype
                    batch_size = input_shape[0] if len(input_shape) > 0 else 1

                    # 1. Project Q, K, V
                    query_states = mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_states = mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_states = mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    # 2. Apply Rotary Position Embeddings
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    # 3. Update Resident Cache
                    if past_key_values is not None:
                        key_states, value_states = past_key_values.update(
                            key_states, value_states, mod.layer_idx
                        )

                    s_q = query_states.shape[2]
                    s_k = key_states.shape[2]
                    h_q = query_states.shape[1]
                    h_kv = key_states.shape[1]
                    group_size = h_q // h_kv
                    elem_bytes = key_states.element_size()

                    # Extract absolute query positions
                    q_pos: Optional[torch.Tensor] = None
                    if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                        q_pos = kwargs["position_ids"].view(-1).to(device)
                    elif "cache_position" in kwargs and kwargs["cache_position"] is not None:
                        q_pos = kwargs["cache_position"].view(-1).to(device)
                    elif past_key_values is not None and getattr(past_key_values, "_active_positions", None) is not None:
                        q_pos = past_key_values._active_positions.view(-1).to(device)
                    else:
                        q_pos = torch.arange(s_q, device=device, dtype=torch.long)

                    # Extract absolute resident key positions from cache contract
                    k_pos_all: Optional[torch.Tensor] = None
                    if (
                        past_key_values is not None
                        and hasattr(past_key_values, "bounded_cache")
                        and mod.layer_idx < len(past_key_values.bounded_cache.states)
                    ):
                        st_pos = past_key_values.bounded_cache.states[mod.layer_idx].absolute_positions
                        if len(st_pos) == s_k:
                            k_pos_all = st_pos.to(device)

                    if k_pos_all is None:
                        # Fallback to contiguous positions ending at max query position
                        last_p = int(q_pos[-1].item())
                        first_p = max(0, last_p - s_k + 1)
                        k_pos_all = torch.arange(first_p, first_p + s_k, device=device, dtype=torch.long)

                    # Convert incoming attention mask into 4D additive form [batch, num_heads, s_q, s_k]
                    base_additive_mask = _prepare_additive_mask(
                        attention_mask,
                        batch_size=batch_size,
                        s_q=s_q,
                        s_k=s_k,
                        dtype=dtype,
                        device=device,
                    )

                    step_budget = self._compute_step_budget(s_k)

                    # Evaluate sparse selection if under budget pressure
                    if step_budget < s_k:
                        _sync(device)
                        t0 = time.perf_counter()
                        res = self.selector.select_indices(
                            keys=key_states,
                            budget=step_budget,
                            queries=query_states,
                            values=value_states,
                        )
                        _sync(device)
                        t1 = time.perf_counter()
                        self.stats.total_selector_time_ms += (t1 - t0) * 1000.0

                        # Extract per-head indices (shape: [H_kv, B_SA])
                        if res.per_head_indices is not None:
                            per_head_I = res.per_head_indices.to(device)  # [H_kv, B_SA]
                        else:
                            idx_list = res.retained_indices
                            per_head_I = torch.tensor(idx_list, device=device, dtype=torch.long).unsqueeze(0).expand(h_kv, -1)

                        b_sa = per_head_I.shape[1]
                        n_rep_q = res.metadata.get("num_representative_queries", 1)

                        # Documented conservative accounting of coexisting selector buffers
                        inter_bytes = calculate_selector_intermediates_bytes(
                            h_q=h_q,
                            h_kv=h_kv,
                            s_q=s_q,
                            s_k=s_k,
                            n_rep_q=n_rep_q,
                            b_sa=b_sa,
                            head_dim=mod.head_dim,
                            elem_bytes=elem_bytes,
                        )
                        self.stats.peak_selector_intermediates_bytes = max(
                            self.stats.peak_selector_intermediates_bytes, inter_bytes
                        )

                        _sync(device)
                        tg0 = time.perf_counter()

                        I_expanded = per_head_I.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, mod.head_dim)

                        # Gather active keys and values per KV head: shape [batch, H_kv, B_SA, D]
                        k_active = torch.gather(key_states, 2, I_expanded)
                        v_active = torch.gather(value_states, 2, I_expanded)

                        _sync(device)
                        tg1 = time.perf_counter()
                        self.stats.total_gather_time_ms += (tg1 - tg0) * 1000.0

                        gather_bytes = 2 * batch_size * h_kv * b_sa * mod.head_dim * elem_bytes
                        self.stats.peak_gathered_active_bytes = max(
                            self.stats.peak_gathered_active_bytes, gather_bytes
                        )

                        # Expand active KV across GQA query heads: shape [batch, H_q, B_SA, D]
                        k_active_rep = repeat_kv(k_active, group_size)
                        v_active_rep = repeat_kv(v_active, group_size)

                        gqa_bytes = 2 * batch_size * h_q * b_sa * mod.head_dim * elem_bytes
                        self.stats.peak_gqa_expanded_bytes = max(
                            self.stats.peak_gqa_expanded_bytes, gqa_bytes
                        )

                        # Build head-specific causal & additive attention mask: shape [batch, H_q, S_q, B_SA]
                        head_masks: List[torch.Tensor] = []
                        head_rows_allowed: List[torch.Tensor] = []
                        for h in range(h_q):
                            h_k = h // group_size
                            head_k_indices = per_head_I[h_k]  # [B_SA]
                            head_k_pos = k_pos_all[head_k_indices]  # [B_SA]

                            # Absolute causal condition: key position <= query position
                            causal_allowed = (head_k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, B_SA]

                            if base_additive_mask is not None:
                                mask_slice = (
                                    base_additive_mask[:, h : h + 1, :, :]
                                    if base_additive_mask.shape[1] == h_q
                                    else base_additive_mask[:, 0:1, :, :]
                                )
                                gathered_m = torch.index_select(mask_slice, -1, head_k_indices)
                                mask_allowed = _is_mask_allowed(gathered_m)
                                overall_allowed = causal_allowed & mask_allowed
                                eff_head_mask = torch.where(
                                    overall_allowed,
                                    gathered_m,
                                    torch.full((), float("-inf"), dtype=dtype, device=device),
                                )
                            else:
                                overall_allowed = causal_allowed
                                eff_head_mask = torch.where(
                                    overall_allowed,
                                    torch.zeros((), dtype=dtype, device=device),
                                    torch.full((), float("-inf"), dtype=dtype, device=device),
                                )

                            # Detect whether each query row has at least one allowed active key
                            row_has_allowed = overall_allowed.any(dim=-1, keepdim=True)  # [batch, 1, S_q, 1]

                            head_masks.append(eff_head_mask)
                            head_rows_allowed.append(row_has_allowed)

                        active_mask = torch.cat(head_masks, dim=1)  # [batch, H_q, S_q, B_SA]
                        rows_allowed = torch.cat(head_rows_allowed, dim=1)  # [batch, H_q, S_q, 1]

                        mask_bytes = batch_size * h_q * s_q * b_sa * elem_bytes
                        self.stats.peak_attention_mask_bytes = max(
                            self.stats.peak_attention_mask_bytes, mask_bytes
                        )

                        self.stats.attention_pairs_evaluated += s_q * b_sa
                        self.stats.dense_attention_pairs += s_q * s_k
                        self.stats.last_step_retained_indices = res.retained_indices
                        self.stats.last_step_head_retained_indices = res.head_retained_indices
                    else:
                        k_active_rep = repeat_kv(key_states, group_size)
                        v_active_rep = repeat_kv(value_states, group_size)

                        head_masks = []
                        head_rows_allowed = []
                        for h in range(h_q):
                            causal_allowed = (k_pos_all.unsqueeze(0) <= q_pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, S_k]

                            if base_additive_mask is not None:
                                mask_slice = (
                                    base_additive_mask[:, h : h + 1, :, :]
                                    if base_additive_mask.shape[1] == h_q
                                    else base_additive_mask[:, 0:1, :, :]
                                )
                                mask_allowed = _is_mask_allowed(mask_slice)
                                overall_allowed = causal_allowed & mask_allowed
                                eff_head_mask = torch.where(
                                    overall_allowed,
                                    mask_slice,
                                    torch.full((), float("-inf"), dtype=dtype, device=device),
                                )
                            else:
                                overall_allowed = causal_allowed
                                eff_head_mask = torch.where(
                                    overall_allowed,
                                    torch.zeros((), dtype=dtype, device=device),
                                    torch.full((), float("-inf"), dtype=dtype, device=device),
                                )

                            row_has_allowed = overall_allowed.any(dim=-1, keepdim=True)  # [batch, 1, S_q, 1]
                            head_masks.append(eff_head_mask)
                            head_rows_allowed.append(row_has_allowed)

                        active_mask = torch.cat(head_masks, dim=1)  # [batch, H_q, S_q, S_k]
                        rows_allowed = torch.cat(head_rows_allowed, dim=1)  # [batch, H_q, S_q, 1]

                        mask_bytes = batch_size * h_q * s_q * s_k * elem_bytes
                        self.stats.peak_attention_mask_bytes = max(
                            self.stats.peak_attention_mask_bytes, mask_bytes
                        )

                        self.stats.attention_pairs_evaluated += s_q * s_k
                        self.stats.dense_attention_pairs += s_q * s_k

                    self.stats.per_layer_calls += 1
                    if getattr(mod, "layer_idx", 0) == 0:
                        self.stats.steps_evaluated += 1

                    # Attention dot product: [batch, H_q, S_q, S_active]
                    attn_weights = torch.matmul(query_states, k_active_rep.transpose(2, 3)) * mod.scaling
                    attn_weights = attn_weights + active_mask

                    # Softmax over active keys
                    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
                    attn_weights = torch.nan_to_num(attn_weights, nan=0.0).to(dtype)

                    # CRITICAL: Detect fully blocked rows explicitly and produce zero attention weights/output
                    # nan_to_num after softmax is insufficient when a finite sentinel yields a uniform distribution
                    attn_weights = torch.where(rows_allowed, attn_weights, torch.zeros_like(attn_weights))

                    dropout_p = getattr(mod, "attention_dropout", 0.0)
                    if mod.training and dropout_p > 0.0:
                        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)

                    attn_output = torch.matmul(attn_weights, v_active_rep)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(*input_shape, -1)
                    attn_output = mod.o_proj(attn_output)

                    return attn_output, attn_weights

                return _forward

            attn_mod.forward = _make_sparse_forward(attn_mod, orig_forward)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for mod, orig_f in self._original_forwards:
            mod.forward = orig_f
        self._original_forwards.clear()
