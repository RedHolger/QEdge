"""Benchmark measurement boundaries and contract-validated emission for QEdge Runtime.

Measures loading, formatting, tokenization, prefill, first-token selection,
TTFT (through first token availability), decode, and detokenization separately with
accelerator synchronization. Emits schema v2 benchmark_result records.
Generated tokens and text are held strictly in outputs, not in metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional

from qedge_common.schema import validate as validate_contract

from .config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    get_default_model_dir,
)
from .acquisition import check_model_assets
from .runner import ModelRunner, GenerationResult


def build_benchmark_record(
    result: GenerationResult,
    run_id: str = "runtime-baseline-smollm2",
    log_path: Optional[str] = "handoff/research-review/05-real-model-baseline/logs/benchmark.log",
) -> Dict[str, Any]:
    """Construct and validate a schema v2 benchmark_result record."""
    now = datetime.now(timezone.utc).isoformat()
    metrics: Dict[str, Any] = {
        "load_ms": result.load_ms,
        "formatting_ms": result.formatting_ms,
        "tokenization_ms": result.tokenization_ms,
        "prefill_ms": result.prefill_ms,
        "first_token_selection_ms": result.first_token_select_ms,
        "decode_ms": result.decode_ms,
        "detokenization_ms": result.detokenization_ms,
        "total_ms": result.total_ms,
        "prefill_tok_per_sec": result.prefill_tok_per_sec,
        "decode_tok_per_sec": result.decode_tok_per_sec,
        "prompt_tokens": len(result.prompt_token_ids),
        "generated_tokens": len(result.generated_token_ids),
        "stopped_on_eos": result.stopped_on_eos,
    }
    if result.ttft_ms is not None:
        metrics["ttft_ms"] = result.ttft_ms

    record: Dict[str, Any] = {
        "schema_version": "2",
        "run_id": run_id,
        "kind": "runtime-generation",
        "created_at": now,
        "model": {
            "name": result.model_name,
            "revision": result.model_revision,
            "quantization": result.dtype,
        },
        "backend": result.device,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "gpu": "Apple Silicon GPU (MPS)" if result.device == "mps" else None,
            "device": platform.node(),
            "npu_status": "not-present",
            "execution_evidence": None,
            "execution_backend": None,
        },
        "measured": True,
        "metrics": metrics,
        "outputs": {
            "text": result.generated_text,
            "token_ids": result.generated_token_ids,
            "token_trace": None,
        },
        "cache": {
            "policy": (result.cache_stats or {}).get("policy", "full"),
            "capacity_tokens": (result.cache_stats or {}).get("capacity_tokens", len(result.prompt_token_ids) + len(result.generated_token_ids)),
            "evictions": (result.cache_stats or {}).get("total_evictions", 0),
            "peak_bytes": (result.cache_stats or {}).get("peak_temporary_bytes", 0),
        },
        "provenance": {
            "command": "python -m qedge_runtime.benchmark",
            "notes": [
                "Real model baseline run without cache compression or QNN.",
                "Frozen dependencies from committed llm-inference/runtime/uv.lock.",
                "TTFT defined through availability of first token (tokenization + prefill + selection).",
                "Greedy-only generation (do_sample=False, temperature=0.0, seed=42).",
                "Outputs kept strictly separate from numeric scalar metrics.",
            ],
            "log_path": log_path or "handoff/research-review/06-cache-contract/logs/benchmark.log",
        },
    }

    validate_contract("benchmark_result", record)
    return record


def build_smoke_record() -> Dict[str, Any]:
    """Emit an explicit synthetic smoke record when requested."""
    now = datetime.now(timezone.utc).isoformat()
    record: Dict[str, Any] = {
        "schema_version": "2",
        "run_id": "runtime-smoke-baseline",
        "kind": "fixture-smoke",
        "created_at": now,
        "model": None,
        "backend": "cpu",
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "npu_status": "unverified",
        },
        "measured": False,
        "metrics": {
            "load_ms": 0.0,
            "ttft_ms": 0.0,
            "decode_ms": 0.0,
        },
        "outputs": {
            "text": None,
            "token_ids": None,
            "token_trace": None,
        },
        "cache": {
            "policy": "fixture-no-cache",
            "capacity_tokens": 0,
        },
        "provenance": {
            "command": "python -m qedge_runtime.benchmark --smoke",
            "notes": ["Explicit smoke benchmark record without real weights."],
            "log_path": None,
        },
    }
    validate_contract("benchmark_result", record)
    return record


def run_benchmark(
    model_dir: Path,
    prompt: str = "Explain edge computing in one concise paragraph.",
    max_new_tokens: int = 32,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    cache_policy: str = "full",
    cache_capacity: Optional[int] = None,
    sink_tokens: int = 4,
) -> Dict[str, Any]:
    """Run real generation and return validated contract benchmark record."""
    sys.stderr.write(f"Loading model from {model_dir} (device: {device or 'auto'}, dtype: {dtype or 'auto'})...\n")
    runner = ModelRunner(model_dir=model_dir, device=device, dtype=dtype, local_files_only=True)

    sys.stderr.write(
        f"Generating (max_new_tokens={max_new_tokens}, policy={cache_policy}, capacity={cache_capacity})...\n"
    )
    result = runner.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        cache_policy=cache_policy,
        cache_capacity=cache_capacity,
        sink_tokens=sink_tokens,
    )

    ttft_str = f"{result.ttft_ms:.1f}ms" if result.ttft_ms is not None else "N/A"
    sys.stderr.write(
        f"Completed: {len(result.generated_token_ids)} tokens in {result.total_ms:.1f}ms "
        f"(prefill: {result.prefill_ms:.1f}ms, selection: {result.first_token_select_ms:.1f}ms, "
        f"decode: {result.decode_ms:.1f}ms, TTFT: {ttft_str})\n"
    )

    record = build_benchmark_record(result)
    return record


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark runner for QEdge Runtime")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local model directory (default: unified canonical path)",
    )
    parser.add_argument("--prompt", default="Explain edge computing in one concise paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    parser.add_argument(
        "--cache-policy",
        choices=["full", "fifo", "recency_window"],
        default="full",
        help="KV cache eviction policy (default: full)",
    )
    parser.add_argument(
        "--cache-capacity",
        type=int,
        default=None,
        help="Capacity limit in tokens for bounded cache policy",
    )
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=4,
        help="Number of initial attention sink tokens to protect (default: 4)",
    )
    parser.add_argument("--smoke", action="store_true", help="Explicitly emit synthetic smoke record")
    args = parser.parse_args(argv)

    if args.smoke:
        record = build_smoke_record()
        print(json.dumps(record, indent=2))
        return 0

    target_dir = (args.model_dir or get_default_model_dir()).resolve()

    valid, errors = check_model_assets(target_dir, expected_model_id=SMOLLM2_135M_ID, expected_revision=SMOLLM2_135M_REVISION)
    if not valid:
        sys.stderr.write(f"Missing dependency: Model assets not available or invalid in {target_dir}:\n")
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        sys.stderr.write("Run 'make acquire' to download pinned model assets, or pass --smoke for explicit fixture smoke.\n")
        return 1

    try:
        record = run_benchmark(
            model_dir=target_dir,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype=args.dtype,
            cache_policy=args.cache_policy,
            cache_capacity=args.cache_capacity,
            sink_tokens=args.sink_tokens,
        )
        print(json.dumps(record, indent=2))
        return 0
    except Exception as e:
        sys.stderr.write(f"Benchmark failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
