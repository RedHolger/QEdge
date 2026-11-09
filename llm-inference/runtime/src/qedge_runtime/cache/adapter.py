"""Tested adapter for the pinned Transformers cache interface.

Subclasses transformers.cache_utils.Cache, providing a clean compatibility
boundary without scattered assumptions about private DynamicCache internals.
Routes position metadata from cache_kwargs, kwargs, and explicit caller channels.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import transformers
from transformers.cache_utils import Cache

try:
    from transformers.cache_utils import DynamicLayer
except ImportError:
    DynamicLayer = None

from .bounded_cache import BoundedKVCache
from .contract import EvictionPolicyKind


PINNED_TESTED_TRANSFORMERS_VERSION: str = "5.22.1"
MIN_TRANSFORMERS_VERSION: str = "4.36.0"


class QEdgeTransformersCacheAdapter(Cache):
    """Adapter wrapping BoundedKVCache into transformers.cache_utils.Cache.
    
    Compatible with the tested transformers Cache contract (verified specifically
    on pinned transformers 5.22.1).
    Fails clearly if incompatible APIs are detected.
    """

    def __init__(self, bounded_cache: BoundedKVCache) -> None:
        # Documented version and API compatibility boundary
        self._check_version_compatibility()

        if DynamicLayer is not None:
            super().__init__(layer_class_to_replicate=DynamicLayer)
        else:
            # Fallback for earlier Cache interfaces if DynamicLayer is not present
            super().__init__()

        self.bounded_cache = bounded_cache
        self._active_positions: Optional[torch.Tensor] = None

    @staticmethod
    def _check_version_compatibility() -> None:
        """Verify transformers version meets tested compatibility requirements."""
        version_str = getattr(transformers, "__version__", "0.0.0")
        parts = version_str.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            major, minor = 0, 0

        # Require >= 4.36
        if major < 4 or (major == 4 and minor < 36):
            raise RuntimeError(
                f"Unsupported transformers version '{version_str}'. "
                f"QEdge cache adapter requires tested transformers >= {MIN_TRANSFORMERS_VERSION}."
            )

        required_methods = ["update", "get_seq_length"]
        missing = [m for m in required_methods if not hasattr(Cache, m)]
        if missing:
            raise RuntimeError(
                f"Incompatible transformers.cache_utils.Cache API: missing required methods {missing}."
            )

    def set_active_positions(self, positions: Optional[torch.Tensor]) -> None:
        """Explicitly set current token positions for subsequent layer updates."""
        self._active_positions = positions

    @contextlib.contextmanager
    def active_positions_context(self, positions: torch.Tensor) -> Iterator[QEdgeTransformersCacheAdapter]:
        """Context manager setting active positions during a model forward pass."""
        prev = self._active_positions
        self._active_positions = positions
        try:
            yield self
        finally:
            self._active_positions = prev

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Intercept layer KV updates, extract position metadata, and route to BoundedKVCache."""
        # Extract explicit position metadata from supported channels
        query_positions: Optional[torch.Tensor] = None

        if "cache_position" in kwargs and kwargs["cache_position"] is not None:
            query_positions = kwargs["cache_position"]
        elif "query_positions" in kwargs and kwargs["query_positions"] is not None:
            query_positions = kwargs["query_positions"]
        elif "position_ids" in kwargs and kwargs["position_ids"] is not None:
            query_positions = kwargs["position_ids"]
        elif "cache_kwargs" in kwargs and isinstance(kwargs["cache_kwargs"], dict):
            ck = kwargs["cache_kwargs"]
            query_positions = ck.get("cache_position") or ck.get("position_ids") or ck.get("query_positions")
        elif args and isinstance(args[0], dict):
            # Positional cache_kwargs (e.g. update(k, v, layer_idx, cache_kwargs))
            ck = args[0]
            query_positions = ck.get("cache_position") or ck.get("position_ids") or ck.get("query_positions")

        # Fallback to explicit caller channel
        if query_positions is None:
            query_positions = self._active_positions

        return self.bounded_cache.update(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
            query_positions=query_positions,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return resident sequence length for the specified layer."""
        return self.bounded_cache.get_seq_length(layer_idx)

    def get_max_length(self, layer_idx: Optional[int] = None) -> int:
        """Return maximum capacity of the cache."""
        return self.bounded_cache.capacity_tokens

    def get_usable_length(self, new_seq_len: int, layer_idx: int = 0) -> int:
        """Return usable past sequence length for attention masking."""
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """Return (kv_length, kv_offset) for Transformers mask creation."""
        resident_len = self.bounded_cache.get_seq_length(layer_idx)
        if (
            self.bounded_cache.policy_kind != EvictionPolicyKind.FULL
            and not self.bounded_cache.defer_eviction
        ):
            kv_len = min(self.bounded_cache.capacity_tokens, resident_len + query_length)
        else:
            kv_len = resident_len + query_length
        return kv_len, 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Return query offset for Transformers mask creation."""
        return self.bounded_cache.get_seq_length(layer_idx)

    def reset(self) -> None:
        """Reset underlying bounded cache."""
        self.bounded_cache.reset()
        self._active_positions = None

    @property
    def key_cache(self) -> List[Optional[torch.Tensor]]:
        """Compatibility property returning resident key tensors per layer."""
        return [state.key_tensor for state in self.bounded_cache.states]

    @property
    def value_cache(self) -> List[Optional[torch.Tensor]]:
        """Compatibility property returning resident value tensors per layer."""
        return [state.value_tensor for state in self.bounded_cache.states]
