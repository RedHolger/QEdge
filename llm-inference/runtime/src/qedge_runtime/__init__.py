"""QEdge Real-Model Reference Runtime."""
from __future__ import annotations

from .evaluation import (
    exact_sequence_agreement,
    length_aware_token_agreement,
    sequence_agreement_diagnostic,
    trajectory_divergence_logit_mse,
    coarse_authored_target_hit,
    compute_teacher_forced_step_logits,
    compute_teacher_forced_logit_mse,
)
from .quantization import (
    GGUFModelRunner,
    build_quantized_artifacts,
    inspect_gguf_model,
    run_full_quantization_comparison,
    generate_hardware_status,
)

__version__ = "0.1.0"

__all__ = [
    "exact_sequence_agreement",
    "length_aware_token_agreement",
    "sequence_agreement_diagnostic",
    "trajectory_divergence_logit_mse",
    "coarse_authored_target_hit",
    "compute_teacher_forced_step_logits",
    "compute_teacher_forced_logit_mse",
    "GGUFModelRunner",
    "build_quantized_artifacts",
    "inspect_gguf_model",
    "run_full_quantization_comparison",
    "generate_hardware_status",
    "__version__",
]
