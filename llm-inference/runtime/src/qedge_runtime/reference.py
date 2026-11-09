"""Unmodified upstream reference generator and parity verification for QEdge Runtime.

Compares custom loop directly against transformers model.generate() with
identical input IDs, device, dtype, and greedy decoding parameters.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def generate_upstream_reference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    formatted_prompt: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[List[int], str, List[torch.Tensor]]:
    """Execute unmodified upstream model.generate() on the exact same inputs."""
    target_device = device or next(model.parameters()).device
    text = formatted_prompt if formatted_prompt is not None else prompt
    inputs = tokenizer(text, return_tensors="pt").to(target_device)
    input_ids = inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    if max_new_tokens == 0:
        return [], "", []

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_ids = outputs.sequences[0][prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Convert output scores (tuples of logits per step) to list of CPU tensors
    scores = [s.detach().cpu() for s in outputs.scores] if hasattr(outputs, "scores") and outputs.scores else []

    return generated_ids, generated_text, scores


def verify_greedy_token_parity(
    runner_tokens: List[int],
    reference_tokens: List[int],
) -> Tuple[bool, str]:
    """Verify exact token-by-token greedy equality."""
    if runner_tokens == reference_tokens:
        return True, f"Parity exact: {len(runner_tokens)} tokens match reference."

    min_len = min(len(runner_tokens), len(reference_tokens))
    first_diff = -1
    for i in range(min_len):
        if runner_tokens[i] != reference_tokens[i]:
            first_diff = i
            break

    if first_diff >= 0:
        msg = (
            f"Token mismatch at index {first_diff}: "
            f"runner token {runner_tokens[first_diff]} != reference token {reference_tokens[first_diff]} "
            f"(runner length {len(runner_tokens)}, ref length {len(reference_tokens)})"
        )
    else:
        msg = f"Length mismatch: runner emitted {len(runner_tokens)} tokens, reference emitted {len(reference_tokens)}."

    return False, msg


def verify_logits_agreement(
    runner_logits: List[torch.Tensor],
    reference_logits: List[torch.Tensor],
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Tuple[bool, float, str]:
    """Verify numerical logits agreement across decoding steps within documented tolerances."""
    if not runner_logits or not reference_logits:
        return False, 0.0, "Logits list is empty"

    if len(runner_logits) != len(reference_logits):
        return False, 0.0, f"Logits step count mismatch: {len(runner_logits)} vs {len(reference_logits)}"

    max_diff = 0.0
    for step, (r_l, ref_l) in enumerate(zip(runner_logits, reference_logits)):
        # Normalize shapes: ensure [1, vocab_size]
        r_flat = r_l.view(-1).float()
        ref_flat = ref_l.view(-1).float()
        diff = torch.max(torch.abs(r_flat - ref_flat)).item()
        if diff > max_diff:
            max_diff = diff
        if not torch.allclose(r_flat, ref_flat, atol=atol, rtol=rtol):
            return (
                False,
                diff,
                f"Logits diverge at step {step}: max abs diff {diff:.6f} exceeds atol {atol}",
            )

    return True, max_diff, f"Logits agree across {len(runner_logits)} steps (max abs diff: {max_diff:.8f})"
