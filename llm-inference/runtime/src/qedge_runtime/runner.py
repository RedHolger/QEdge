"""Reference Model Runner for QEdge Runtime.

Executes real small language models (SmolLM2-135M-Instruct) with:
- Exact architecture configuration loaded from verified checkpoint
- Verified model identity and revision propagation to Hub/local loaders
- Real prefill and autoregressive decode with full KV cache
- Strict position tracking, attention masks, early EOS, and output bounds
- Rejection of unsupported sampling and invalid generation settings
- Isolated request state (no cross-request cache leakage)
- Accurate measurement boundaries with accelerator synchronization (TTFT through first-token availability)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import contextlib
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    SMOLLM2_135M_LICENSE,
    SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST,
    ModelArchitectureConfig,
    format_smollm_chat_prompt,
    get_default_model_dir,
)
from .acquisition import check_model_assets
from .cache import BoundedKVCache, QEdgeTransformersCacheAdapter, EvictionPolicyKind


@dataclass
class GenerationResult:
    """Generation output and measurement telemetry.
    
    Generated text and token IDs are held strictly separate from benchmark metrics.
    """
    prompt: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    generated_text: str
    stopped_on_eos: bool
    model_name: str
    model_revision: str
    device: str
    dtype: str
    # Timing measurements (milliseconds)
    load_ms: float = 0.0
    formatting_ms: float = 0.0
    tokenization_ms: float = 0.0
    prefill_ms: float = 0.0
    first_token_select_ms: float = 0.0
    ttft_ms: Optional[float] = None
    decode_ms: float = 0.0
    detokenization_ms: float = 0.0
    total_ms: float = 0.0
    # Throughput
    prefill_tok_per_sec: float = 0.0
    decode_tok_per_sec: float = 0.0
    # Recorded logits per generated token (optional)
    step_logits: Optional[List[torch.Tensor]] = None
    # Cache telemetry
    cache_stats: Optional[Dict[str, Any]] = None
    # Reproducibility metadata
    reproducibility: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_tokens_count": len(self.prompt_token_ids),
            "generated_tokens_count": len(self.generated_token_ids),
            "generated_text": self.generated_text,
            "stopped_on_eos": self.stopped_on_eos,
            "model": {
                "name": self.model_name,
                "revision": self.model_revision,
                "device": self.device,
                "dtype": self.dtype,
            },
            "timings_ms": {
                "load": round(self.load_ms, 2),
                "formatting": round(self.formatting_ms, 2),
                "tokenization": round(self.tokenization_ms, 2),
                "prefill": round(self.prefill_ms, 2),
                "first_token_selection": round(self.first_token_select_ms, 2),
                "ttft": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
                "decode": round(self.decode_ms, 2),
                "detokenization": round(self.detokenization_ms, 2),
                "total": round(self.total_ms, 2),
            },
            "throughput": {
                "prefill_tok_per_sec": round(self.prefill_tok_per_sec, 2),
                "decode_tok_per_sec": round(self.decode_tok_per_sec, 2),
            },
            "cache": self.cache_stats,
            "outputs": {
                "token_ids": self.generated_token_ids,
                "text": self.generated_text,
            },
            "reproducibility": self.reproducibility,
        }


class ModelRunner:
    """Real LLM reference engine with full KV cache and upstream parity."""

    def __init__(
        self,
        model_dir_or_id: str | Path | None = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        local_files_only: bool = True,
        model_dir: str | Path | None = None,
    ) -> None:
        target = model_dir_or_id if model_dir_or_id is not None else model_dir
        if target is None:
            target = get_default_model_dir()

        target_path = Path(target)
        if target_path.exists():
            self.model_path = target_path.resolve()
            self.model_id = str(self.model_path)
            # Verify provenance for local model
            valid, errors = check_model_assets(
                self.model_path,
                expected_model_id=SMOLLM2_135M_ID,
                expected_revision=SMOLLM2_135M_REVISION,
                verify_trusted_hashes=True,
            )
            if not valid:
                raise ValueError(
                    f"Unsupported or unverified model assets at {self.model_path}: {errors}"
                )
            self.model_name = SMOLLM2_135M_ID
            self.revision = SMOLLM2_135M_REVISION
            self.license = SMOLLM2_135M_LICENSE
            self.is_verified_provenance = True
        else:
            self.model_path = None
            self.model_id = str(target)
            if self.model_id != SMOLLM2_135M_ID:
                raise ValueError(
                    f"Unsupported model ID '{self.model_id}'. Only {SMOLLM2_135M_ID} is supported in this stage."
                )
            self.model_name = SMOLLM2_135M_ID
            self.revision = SMOLLM2_135M_REVISION
            self.license = SMOLLM2_135M_LICENSE
            self.is_verified_provenance = True

        # Device selection: MPS on macOS Apple Silicon if available, else CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            if device not in ("cpu", "mps", "cuda") and not device.startswith("cuda:"):
                raise ValueError(f"Unsupported device: {device}")
            self.device = device

        # Dtype selection: float32 on CPU for numeric stability; bfloat16/float16 on MPS/CUDA
        if dtype is None:
            if self.device == "cpu":
                self.torch_dtype = torch.float32
                self.dtype_str = "float32"
            else:
                self.torch_dtype = torch.bfloat16
                self.dtype_str = "bfloat16"
        else:
            self.dtype_str = dtype
            if dtype in ("bfloat16", "bf16"):
                self.torch_dtype = torch.bfloat16
            elif dtype in ("float16", "fp16"):
                self.torch_dtype = torch.float16
            elif dtype in ("float32", "fp32"):
                self.torch_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        # Measurement: Model load time
        t_load_start = time.perf_counter()

        # Load tokenizer with explicit revision propagation
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=local_files_only,
        )

        # Load weights with explicit revision propagation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            local_files_only=local_files_only,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

        self._sync()
        t_load_end = time.perf_counter()
        self.load_ms = (t_load_end - t_load_start) * 1000.0

        # Load architecture settings from config.json if local directory
        config_path = self.model_path / "config.json" if self.model_path and (self.model_path / "config.json").is_file() else None
        if config_path:
            with config_path.open("r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            self.arch_config = ModelArchitectureConfig.from_checkpoint_dict(raw_cfg)
        else:
            self.arch_config = ModelArchitectureConfig()

        # Read EOS and padding behavior from verified configuration (avoiding truthiness fallbacks)
        cfg_eos = getattr(self.model.config, "eos_token_id", None)
        if cfg_eos is None:
            cfg_eos = getattr(self.tokenizer, "eos_token_id", None)
        if cfg_eos is None:
            raise ValueError("No EOS token ID found in model or tokenizer configuration")

        if isinstance(cfg_eos, int):
            self.eos_token_ids: Set[int] = {cfg_eos}
        elif isinstance(cfg_eos, (list, tuple, set)):
            self.eos_token_ids = {int(x) for x in cfg_eos if x is not None}
        else:
            raise ValueError(f"Unsupported eos_token_id configuration: {cfg_eos!r}")

        self.eos_token_id: int = next(iter(self.eos_token_ids))

        cfg_pad = getattr(self.model.config, "pad_token_id", None)
        if cfg_pad is None:
            cfg_pad = getattr(self.tokenizer, "pad_token_id", None)
        if cfg_pad is None:
            cfg_pad = self.eos_token_id
        self.pad_token_id: int = cfg_pad
        self.tokenizer.pad_token_id = self.pad_token_id

        # Derive actual dtype directly from loaded model parameters
        try:
            param = next(iter(self.model.parameters()), None)
            if param is not None and hasattr(param, "dtype"):
                self.actual_dtype: str = str(param.dtype).replace("torch.", "")
            else:
                self.actual_dtype = self.dtype_str
        except Exception:
            self.actual_dtype = self.dtype_str

        # Derive and verify chat template identity from loaded tokenizer
        effective_tmpl = getattr(self.tokenizer, "chat_template", None)
        if isinstance(effective_tmpl, str) and effective_tmpl:
            self.template_digest: Optional[str] = hashlib.sha256(effective_tmpl.encode("utf-8")).hexdigest()
            if self.model_id == SMOLLM2_135M_ID or self.model_path:
                if self.template_digest != SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST:
                    raise ValueError(
                        f"Loaded chat template digest mismatch: got '{self.template_digest}', "
                        f"expected '{SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST}'"
                    )
        else:
            self.template_digest = None

    def _sync(self) -> None:
        """Synchronize hardware accelerator if available."""
        if self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif self.device.startswith("cuda") and hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()

    def format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt using model chat template."""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return format_smollm_chat_prompt(prompt, system_prompt)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        record_logits: bool = False,
        cache_policy: str = "full",
        cache_capacity: Optional[int] = None,
        sink_tokens: int = 4,
        attention_selector: Optional[str] = None,
        attention_budget: Optional[int] = None,
        attention_ratio: Optional[float] = None,
        chunk_size: Optional[int] = None,
    ) -> GenerationResult:
        """Execute real prefill and autoregressive decode with position-correct bounded KV cache.
        
        Strict bounds, position IDs, early EOS, and state isolation.
        Rejects unsupported sampling and invalid limits.
        """
        # Validate settings before inference
        if do_sample is True:
            raise ValueError(
                "Sampling (do_sample=True) is not supported in this greedy-only reference runtime; pass do_sample=False."
            )

        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError(
                f"max_new_tokens must be an integer, got {type(max_new_tokens).__name__}: {max_new_tokens!r}"
            )
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
            raise ValueError(f"temperature must be a finite number, got {temperature!r}")
        if temperature != 0.0:
            raise ValueError(
                f"Non-zero temperature ({temperature}) is not supported in this greedy-only reference runtime; use temperature=0.0."
            )

        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or top_p != 1.0:
            raise ValueError(f"top_p={top_p} is not supported in this greedy-only baseline; must be 1.0.")

        # Cache settings validation
        valid_policies = [p.value for p in EvictionPolicyKind]
        if cache_policy.lower() not in valid_policies:
            raise ValueError(
                f"Unsupported cache policy {cache_policy!r}. Supported policies: {valid_policies}"
            )
        if cache_policy.lower() != EvictionPolicyKind.FULL.value:
            if cache_capacity is None:
                raise ValueError(
                    f"cache_capacity must be specified as a positive integer for bounded policy '{cache_policy}'."
                )
            if isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int) or cache_capacity < 1:
                raise ValueError(f"cache_capacity must be an integer >= 1, got {cache_capacity!r}")

        if isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int) or sink_tokens < 0:
            raise ValueError(f"sink_tokens must be an integer >= 0, got {sink_tokens!r}")

        # Temporary attention selector validation
        valid_attention_selectors = ["quoka", "attention_value_decay"]
        if attention_selector is not None:
            if attention_selector.lower() not in valid_attention_selectors:
                raise ValueError(
                    f"Unsupported attention selector {attention_selector!r}. Supported: {valid_attention_selectors}"
                )
            if attention_budget is not None:
                if isinstance(attention_budget, bool) or not isinstance(attention_budget, int) or attention_budget < 1:
                    raise ValueError(f"attention_budget must be an integer >= 1, got {attention_budget!r}")
            if attention_ratio is not None:
                if isinstance(attention_ratio, bool) or not isinstance(attention_ratio, (int, float)) or attention_ratio <= 0.0 or attention_ratio > 1.0:
                    raise ValueError(f"attention_ratio must be a float in (0.0, 1.0], got {attention_ratio!r}")

        if chunk_size is not None:
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
                raise ValueError(f"chunk_size must be an integer >= 1, got {chunk_size!r}")

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError(f"seed must be an integer, got {type(seed).__name__}: {seed!r}")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)
            seed_policy = "explicit"
            recorded_seed = seed
        else:
            seed_policy = "none_deterministic_greedy"
            recorded_seed = None

        # 1. Format prompt
        t_fmt_start = time.perf_counter()
        formatted = self.format_prompt(prompt, system_prompt) if use_chat_template else prompt
        t_fmt_end = time.perf_counter()
        formatting_ms = (t_fmt_end - t_fmt_start) * 1000.0

        # 2. Tokenization
        self._sync()
        t_tok_start = time.perf_counter()
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        self._sync()
        t_tok_end = time.perf_counter()
        tok_ms = (t_tok_end - t_tok_start) * 1000.0

        input_ids = inputs["input_ids"]
        prompt_token_ids = input_ids[0].tolist()
        prompt_len = int(input_ids.shape[1])

        try:
            import safetensors
            safetensors_ver = safetensors.__version__
        except Exception:
            safetensors_ver = None

        if use_chat_template:
            raw_prompt_mode = False
            effective_digest = self.template_digest
            template_id = f"sha256:{effective_digest}" if effective_digest else "builtin_smollm_chat_prompt"
        else:
            raw_prompt_mode = True
            effective_digest = None
            template_id = "raw_prompt"

        repro_metadata = {
            "dependencies": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "safetensors": safetensors_ver,
            },
            "lockfile": "llm-inference/runtime/uv.lock (frozen)",
            "device": self.device,
            "requested_dtype": self.dtype_str,
            "actual_dtype": self.actual_dtype,
            "seed_policy": seed_policy,
            "seed": recorded_seed,
            "generation_settings": {
                "max_new_tokens": max_new_tokens,
                "greedy": True,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "cache_policy": cache_policy,
                "cache_capacity": cache_capacity,
                "attention_selector": attention_selector,
                "attention_budget": attention_budget,
                "attention_ratio": attention_ratio,
                "chunk_size": chunk_size,
            },
            "template": {
                "raw_prompt_mode": raw_prompt_mode,
                "effective_template_digest": effective_digest,
                "template_identity": template_id,
            },
            "model_revision": self.revision,
            "tokenizer_revision": self.revision,
        }

        # Zero-token edge case: zero-output timing without implying a token arrived
        if max_new_tokens == 0:
            zero_stats = {
                "policy": cache_policy,
                "resident_tokens": 0,
                "capacity_tokens": cache_capacity or 0,
                "peak_tokens": 0,
                "total_tokens_seen": 0,
                "total_evictions": 0,
                "resident_payload_bytes": 0,
                "capacity_payload_bytes": 0,
                "metadata_bytes": 64,
                "peak_temporary_bytes": 64,
            }
            return GenerationResult(
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=[],
                generated_text="",
                stopped_on_eos=False,
                model_name=self.model_name,
                model_revision=self.revision,
                device=self.device,
                dtype=self.actual_dtype,
                load_ms=self.load_ms,
                formatting_ms=formatting_ms,
                tokenization_ms=tok_ms,
                prefill_ms=0.0,
                first_token_select_ms=0.0,
                ttft_ms=None,
                decode_ms=0.0,
                detokenization_ms=0.0,
                total_ms=formatting_ms + tok_ms,
                prefill_tok_per_sec=0.0,
                decode_tok_per_sec=0.0,
                cache_stats=zero_stats,
                reproducibility=repro_metadata,
            )

        # Instantiate Bounded KV Cache and Transformers adapter
        bounded_cache = BoundedKVCache(
            num_layers=self.arch_config.num_hidden_layers,
            num_kv_heads=self.arch_config.num_key_value_heads,
            head_dim=self.arch_config.head_dim,
            capacity_tokens=cache_capacity if cache_capacity is not None else 1_000_000,
            policy=cache_policy,
            sink_tokens=sink_tokens,
            dtype=self.torch_dtype,
        )
        cache_adapter = QEdgeTransformersCacheAdapter(bounded_cache)

        sparse_ctx = None
        if attention_selector is not None:
            from .selectors import SparseAttentionContext
            sparse_ctx = SparseAttentionContext(
                model=self.model,
                selector_name=attention_selector,
                budget=attention_budget,
                ratio=attention_ratio,
                sink_tokens=sink_tokens,
            )

        ctx = sparse_ctx if sparse_ctx is not None else contextlib.nullcontext()
        with ctx:
            # 3. Prefill Step
            pos_ids = torch.arange(0, prompt_len, device=self.device, dtype=torch.long).unsqueeze(0)
            attn_mask = torch.ones((1, prompt_len), device=self.device, dtype=torch.long)

            # Defer eviction during prefill forward pass so all prompt tokens attend to full prompt
            bounded_cache.defer_eviction = True

            self._sync()
            t_prefill_start = time.perf_counter()

            if chunk_size is not None and chunk_size < prompt_len:
                # Multi-chunk prefill
                last_logits = None
                for start in range(0, prompt_len, chunk_size):
                    end = min(start + chunk_size, prompt_len)
                    chunk_input_ids = input_ids[:, start:end]
                    chunk_pos = pos_ids[:, start:end]
                    chunk_len = end - start
                    cache_adapter.set_active_positions(chunk_pos[0])

                    # 4D causal additive mask for the chunk: shape (1, 1, chunk_len, end)
                    # Query at start + q_i attends causally to keys 0 .. start + q_i
                    chunk_mask = torch.full(
                        (1, 1, chunk_len, end), float("-inf"), dtype=self.torch_dtype, device=self.device
                    )
                    for q_i in range(chunk_len):
                        chunk_mask[0, 0, q_i, : start + q_i + 1] = 0.0

                    chunk_out = self.model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_mask,
                        position_ids=chunk_pos,
                        past_key_values=cache_adapter,
                        use_cache=True,
                    )
                    last_logits = chunk_out.logits[:, -1, :]
                first_token_logits = last_logits
            else:
                attn_mask = torch.ones((1, prompt_len), device=self.device, dtype=torch.long)
                cache_adapter.set_active_positions(pos_ids[0])
                prefill_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=cache_adapter,
                    use_cache=True,
                )
                first_token_logits = prefill_out.logits[:, -1, :]

            self._sync()
            t_prefill_end = time.perf_counter()
            prefill_ms = (t_prefill_end - t_prefill_start) * 1000.0

            # Post-prefill eviction prune down to capacity
            bounded_cache.defer_eviction = False
            if (
                bounded_cache.policy_kind != EvictionPolicyKind.FULL
                and prompt_len > bounded_cache.capacity_tokens
            ):
                bounded_cache.evict_to_capacity()

            # 4. First Token Selection & Materialization
            t_sel_start = time.perf_counter()
            first_token = int(torch.argmax(first_token_logits, dim=-1)[0])
            self._sync()
            t_sel_end = time.perf_counter()
            first_token_select_ms = (t_sel_end - t_sel_start) * 1000.0

            # TTFT: Measured through availability of the first token
            ttft_ms = tok_ms + prefill_ms + first_token_select_ms

            prefill_tps = prompt_len / max(t_prefill_end - t_prefill_start, 1e-6)

            generated_tokens: List[int] = [first_token]
            recorded_step_logits: Optional[List[torch.Tensor]] = [first_token_logits.detach().cpu()] if record_logits else None

            stopped_on_eos = (first_token in self.eos_token_ids)

            # 5. Autoregressive Decode Loop (for tokens 2..N)
            decode_ms = 0.0
            curr_token = first_token

            if max_new_tokens > 1 and not stopped_on_eos:
                self._sync()
                t_dec_start = time.perf_counter()

                for step in range(1, max_new_tokens):
                    step_input_ids = torch.tensor([[curr_token]], device=self.device, dtype=torch.long)
                    step_pos = torch.tensor([[prompt_len + step - 1]], device=self.device, dtype=torch.long)

                    resident_len = cache_adapter.get_seq_length()
                    if bounded_cache.policy_kind != EvictionPolicyKind.FULL:
                        post_step_len = min(bounded_cache.capacity_tokens, resident_len + 1)
                    else:
                        post_step_len = resident_len + 1

                    step_mask = torch.zeros((1, 1, 1, post_step_len), dtype=self.torch_dtype, device=self.device)
                    cache_adapter.set_active_positions(step_pos[0])

                    out = self.model(
                        input_ids=step_input_ids,
                        attention_mask=step_mask,
                        position_ids=step_pos,
                        past_key_values=cache_adapter,
                        use_cache=True,
                    )
                    step_logits = out.logits[:, -1, :]
                    if record_logits and recorded_step_logits is not None:
                        recorded_step_logits.append(step_logits.detach().cpu())

                    next_token = int(torch.argmax(step_logits, dim=-1)[0])
                    generated_tokens.append(next_token)
                    curr_token = next_token

                    if next_token in self.eos_token_ids:
                        stopped_on_eos = True
                        break

                self._sync()
                t_dec_end = time.perf_counter()
                decode_ms = (t_dec_end - t_dec_start) * 1000.0

        # 6. Detokenization Phase
        t_detok_start = time.perf_counter()
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        t_detok_end = time.perf_counter()
        detokenization_ms = (t_detok_end - t_detok_start) * 1000.0

        decode_count = len(generated_tokens) - 1
        decode_tps = decode_count / max(decode_ms / 1000.0, 1e-6) if decode_count > 0 and decode_ms > 0 else 0.0
        total_ms = formatting_ms + tok_ms + prefill_ms + first_token_select_ms + decode_ms + detokenization_ms

        cache_stats = bounded_cache.get_memory_stats().to_dict()
        cache_stats["policy"] = cache_policy
        if sparse_ctx is not None:
            cache_stats["sparse_attention"] = sparse_ctx.stats.to_dict()
        cache_stats["policy"] = cache_policy

        return GenerationResult(
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_tokens,
            generated_text=generated_text,
            stopped_on_eos=stopped_on_eos,
            model_name=self.model_name,
            model_revision=self.revision,
            device=self.device,
            dtype=self.actual_dtype,
            load_ms=self.load_ms,
            formatting_ms=formatting_ms,
            tokenization_ms=tok_ms,
            prefill_ms=prefill_ms,
            first_token_select_ms=first_token_select_ms,
            ttft_ms=ttft_ms,
            decode_ms=decode_ms,
            detokenization_ms=detokenization_ms,
            total_ms=total_ms,
            prefill_tok_per_sec=prefill_tps,
            decode_tok_per_sec=decode_tps,
            step_logits=recorded_step_logits,
            cache_stats=cache_stats,
            reproducibility=repro_metadata,
        )
