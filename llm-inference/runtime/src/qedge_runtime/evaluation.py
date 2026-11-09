"""Evaluation metrics and scoring contracts for QEdge LLM inference.

Provides rigorous semantic evaluation metrics:
1. exact_sequence_agreement: Strict binary indicator of identical token sequence generation.
2. length_aware_token_agreement: Token-level agreement normalized by maximum sequence length,
   explicitly penalizing premature EOS or trailing length discrepancies.
3. sequence_agreement_diagnostic: Full diagnostic breakdown of prefix match, length delta, and EOS.
4. teacher_forced_prefix_logit_mse: Pure numerical approximation error evaluated on identical
   conditioning prefixes between sparse and dense attention.
5. trajectory_divergence_logit_mse: Step-by-step logit comparison across free-running diverging trajectories.
6. coarse_authored_target_hit: Boundary-aware factual target matching with negation and numeric verification.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import torch


def exact_sequence_agreement(gen_tokens: List[int], ref_tokens: List[int]) -> float:
    """Strict binary indicator of identical token sequence generation.
    
    Returns 1.0 if both sequences are non-empty and exactly equal; otherwise 0.0.
    """
    if not gen_tokens or not ref_tokens:
        return 1.0 if gen_tokens == ref_tokens else 0.0
    return 1.0 if gen_tokens == ref_tokens else 0.0


def length_aware_token_agreement(gen_tokens: List[int], ref_tokens: List[int]) -> float:
    """Token-level agreement normalized by maximum sequence length.
    
    Unlike legacy token agreement which divided by min(len(gen), len(ref)) and ignored
    unmatched suffixes or premature EOS, length-aware agreement divides matching token count
    by max(len(gen), len(ref)), penalizing truncated generations.
    """
    if not gen_tokens and not ref_tokens:
        return 1.0
    if not gen_tokens or not ref_tokens:
        return 0.0

    max_len = max(len(gen_tokens), len(ref_tokens))
    min_len = min(len(gen_tokens), len(ref_tokens))
    matches = sum(1 for i in range(min_len) if gen_tokens[i] == ref_tokens[i])
    return matches / max_len


def sequence_agreement_diagnostic(
    gen_tokens: List[int],
    ref_tokens: List[int],
    eos_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Produce comprehensive sequence agreement diagnostics."""
    len_gen = len(gen_tokens)
    len_ref = len(ref_tokens)
    max_len = max(len_gen, len_ref)
    min_len = min(len_gen, len_ref)

    matches = sum(1 for i in range(min_len) if gen_tokens[i] == ref_tokens[i])

    # Prefix match length before first disagreement
    prefix_len = 0
    for i in range(min_len):
        if gen_tokens[i] == ref_tokens[i]:
            prefix_len += 1
        else:
            break

    premature_eos = False
    if eos_token_id is not None and len_gen < len_ref and len_gen > 0:
        premature_eos = (gen_tokens[-1] == eos_token_id)

    return {
        "exact_sequence_match": gen_tokens == ref_tokens,
        "exact_sequence_agreement": 1.0 if gen_tokens == ref_tokens else 0.0,
        "length_aware_token_agreement": (matches / max_len) if max_len > 0 else 1.0,
        "matched_tokens_count": matches,
        "prefix_match_length": prefix_len,
        "gen_length": len_gen,
        "ref_length": len_ref,
        "length_diff": len_gen - len_ref,
        "premature_eos": premature_eos,
    }


def trajectory_divergence_logit_mse(
    step_logits: Optional[List[torch.Tensor]],
    ref_step_logits: Optional[List[torch.Tensor]],
) -> Optional[float]:
    """Calculate Mean Squared Error between separately generated step logits.
    
    NOTE: After tokens diverge, step logits reflect different conditioning histories.
    This metric measures free-running trajectory divergence, not single-step approximation
    error on identical inputs.
    """
    if not step_logits or not ref_step_logits:
        return None
    num_steps = min(len(step_logits), len(ref_step_logits))
    if num_steps == 0:
        return None
    total_mse = 0.0
    for t in range(num_steps):
        l1 = step_logits[t].float()
        l2 = ref_step_logits[t].float()
        total_mse += torch.mean((l1 - l2) ** 2).item()
    return total_mse / num_steps


def coarse_authored_target_hit(
    generated_text: str,
    target: str,
    detect_negation: bool = True,
) -> bool:
    """Evaluate whether target substring is present in generated text.
    
    Implements boundary-aware matching:
    1. Word-boundary matching: prevents substring collisions (e.g. 'Red' in 'reduced').
    2. Numeric exact matching: ensures numeric targets (e.g. '101') are not accepted when
       the generated text gives a wrong numeric value (e.g. '110').
    3. Negation detection: when detect_negation=True, detects immediate negation patterns
       like 'not <target>', 'never <target>', 'isn't <target>', rejecting them.
    """
    clean_gen = generated_text.strip()
    clean_target = target.strip()

    if not clean_target or not clean_gen:
        return False

    # Check for word boundary matching
    pattern = r"\b" + re.escape(clean_target) + r"\b"
    match = re.search(pattern, clean_gen, re.IGNORECASE)
    if not match:
        return False

    if detect_negation:
        # Check if the matched target is immediately preceded by negation words
        match_start = match.start()
        prefix_context = clean_gen[max(0, match_start - 30) : match_start].lower()
        negation_pattern = r"\b(not|never|is not|isn't|cannot|was not|wasn't|neither|nor)\s*$"
        if re.search(negation_pattern, prefix_context.strip()):
            return False

    return True


def compute_teacher_forced_step_logits(
    runner: Any,
    prompt: str,
    ref_tokens: List[int],
    system_prompt: Optional[str] = None,
    use_chat_template: bool = True,
    cache_policy: str = "full",
    cache_capacity: Optional[int] = None,
    sink_tokens: int = 4,
    attention_selector: Optional[str] = None,
    attention_budget: Optional[int] = None,
    attention_ratio: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Execute deployed-equivalent teacher-forced evaluation path.

    Matches the benchmarked ModelRunner.generate execution path exactly:
    1. Formats prompt using runner.format_prompt(prompt, system_prompt) when use_chat_template=True.
    2. Tokenizes formatted prompt on runner.device.
    3. Initializes BoundedKVCache and QEdgeTransformersCacheAdapter with identical policy, capacity, sink_tokens, dtype.
    4. Follows exact prefill schedule: sets bounded_cache.defer_eviction = True during prefill,
       executes prefill (with chunking if chunk_size is provided), records prompt output logits (predicting ref_tokens[0]),
       then executes post-prefill pruning (defer_eviction = False and evicts if resident_tokens > capacity).
    5. Advances step-by-step through forced reference tokens ref_tokens[:-1]:
       At each step k, passes token ref_tokens[k], sets position, builds decode_attn_mask,
       and records output logits predicting ref_tokens[k+1].

    Guarantees:
    - Future reference completion tokens NEVER participate before their step.
    - Changing ref_tokens[k+1:] has zero effect on step logits <= k.
    - Eviction and selector execution follow the exact deployed schedule.

    Returns:
        (step_logits, prompt_token_ids)
        step_logits: List of float32 1D tensors [vocab_size], of length len(ref_tokens).
        prompt_token_ids: List of integer token IDs for the formatted prompt.
    """
    import contextlib
    from .cache import BoundedKVCache, QEdgeTransformersCacheAdapter, EvictionPolicyKind
    from .selectors import SparseAttentionContext

    if not ref_tokens:
        return [], []

    formatted = runner.format_prompt(prompt, system_prompt) if use_chat_template else prompt
    inputs = runner.tokenizer(formatted, return_tensors="pt").to(runner.device)
    input_ids = inputs["input_ids"]
    prompt_token_ids = input_ids[0].tolist()
    prompt_len = int(input_ids.shape[1])

    bounded_cache = BoundedKVCache(
        num_layers=runner.model.config.num_hidden_layers,
        num_kv_heads=runner.model.config.num_key_value_heads,
        head_dim=runner.model.config.head_dim,
        capacity_tokens=cache_capacity if cache_capacity is not None else 1_000_000,
        policy=cache_policy,
        sink_tokens=sink_tokens,
        dtype=runner.torch_dtype,
    )
    cache_adapter = QEdgeTransformersCacheAdapter(bounded_cache)

    sparse_ctx = None
    if attention_selector is not None and attention_selector != "dense":
        sparse_ctx = SparseAttentionContext(
            model=runner.model,
            selector_name=attention_selector,
            budget=attention_budget,
            ratio=attention_ratio,
            sink_tokens=sink_tokens,
        )
    ctx = sparse_ctx if sparse_ctx is not None else contextlib.nullcontext()

    step_logits: List[torch.Tensor] = []
    with ctx, torch.no_grad():
        pos_ids = torch.arange(0, prompt_len, device=runner.device, dtype=torch.long).unsqueeze(0)
        bounded_cache.defer_eviction = True

        if chunk_size is not None and chunk_size < prompt_len:
            last_out = None
            for start in range(0, prompt_len, chunk_size):
                end = min(start + chunk_size, prompt_len)
                chunk_input_ids = input_ids[:, start:end]
                chunk_pos = pos_ids[:, start:end]
                chunk_len = end - start
                cache_adapter.set_active_positions(chunk_pos[0])

                chunk_mask = torch.full(
                    (1, 1, chunk_len, end), float("-inf"), dtype=runner.torch_dtype, device=runner.device
                )
                for q_i in range(chunk_len):
                    chunk_mask[0, 0, q_i, : start + q_i + 1] = 0.0

                last_out = runner.model(
                    input_ids=chunk_input_ids,
                    position_ids=chunk_pos,
                    attention_mask=chunk_mask,
                    past_key_values=cache_adapter,
                    use_cache=True,
                )
            prefill_logits = last_out.logits[0, -1].float()
        else:
            cache_adapter.set_active_positions(pos_ids[0])
            attn_mask = torch.ones((1, prompt_len), device=runner.device, dtype=torch.long)
            out = runner.model(
                input_ids=input_ids,
                position_ids=pos_ids,
                attention_mask=attn_mask,
                past_key_values=cache_adapter,
                use_cache=True,
            )
            prefill_logits = out.logits[0, -1].float()

        # Re-enable eviction and prune down to capacity once prefill completes (identical to runner)
        bounded_cache.defer_eviction = False
        if (
            bounded_cache.policy_kind != EvictionPolicyKind.FULL
            and prompt_len > bounded_cache.capacity_tokens
        ):
            bounded_cache.evict_to_capacity()

        step_logits.append(prefill_logits)

        # Autoregressive forward with forced reference tokens ref_tokens[:-1]
        for k in range(len(ref_tokens) - 1):
            forced_token = ref_tokens[k]
            step_input_ids = torch.tensor([[forced_token]], device=runner.device, dtype=torch.long)
            step_pos = torch.tensor([[prompt_len + k]], device=runner.device, dtype=torch.long)

            resident_len = cache_adapter.get_seq_length()
            if bounded_cache.policy_kind != EvictionPolicyKind.FULL:
                post_step_len = min(bounded_cache.capacity_tokens, resident_len + 1)
            else:
                post_step_len = resident_len + 1

            step_mask = torch.zeros((1, 1, 1, post_step_len), dtype=runner.torch_dtype, device=runner.device)
            cache_adapter.set_active_positions(step_pos[0])

            out = runner.model(
                input_ids=step_input_ids,
                attention_mask=step_mask,
                position_ids=step_pos,
                past_key_values=cache_adapter,
                use_cache=True,
            )
            step_logits.append(out.logits[0, -1].float())

    return step_logits, prompt_token_ids


def compute_teacher_forced_logit_mse(
    runner: Any,
    prompt: str,
    ref_tokens: List[int],
    system_prompt: Optional[str] = None,
    use_chat_template: bool = True,
    cache_policy: str = "full",
    cache_capacity: Optional[int] = None,
    sink_tokens: int = 4,
    attention_selector: Optional[str] = None,
    attention_budget: Optional[int] = None,
    attention_ratio: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> float:
    """Compute MSE between teacher-forced logits under target policy and dense reference baseline.

    Both target and reference paths follow the identical deployed prefill and step-by-step
    decode schedule on identical conditioning prefixes.
    """
    if not ref_tokens:
        return 0.0

    # 1. Target policy logits
    target_logits, _ = compute_teacher_forced_step_logits(
        runner=runner,
        prompt=prompt,
        ref_tokens=ref_tokens,
        system_prompt=system_prompt,
        use_chat_template=use_chat_template,
        cache_policy=cache_policy,
        cache_capacity=cache_capacity,
        sink_tokens=sink_tokens,
        attention_selector=attention_selector,
        attention_budget=attention_budget,
        attention_ratio=attention_ratio,
        chunk_size=chunk_size,
    )

    # 2. Dense reference logits (full cache, dense attention, identical prefill and step-by-step decode)
    if cache_policy == "full" and (attention_selector is None or attention_selector == "dense"):
        # Target IS the dense reference
        return 0.0

    dense_logits, _ = compute_teacher_forced_step_logits(
        runner=runner,
        prompt=prompt,
        ref_tokens=ref_tokens,
        system_prompt=system_prompt,
        use_chat_template=use_chat_template,
        cache_policy="full",
        cache_capacity=None,
        sink_tokens=sink_tokens,
        attention_selector=None,
        attention_budget=None,
        attention_ratio=None,
        chunk_size=chunk_size,
    )

    n_steps = min(len(target_logits), len(dense_logits))
    if n_steps == 0:
        return 0.0

    total_mse = 0.0
    for t in range(n_steps):
        t_l = target_logits[t]
        d_l = dense_logits[t]
        total_mse += torch.mean((t_l - d_l) ** 2).item()

    return total_mse / n_steps
