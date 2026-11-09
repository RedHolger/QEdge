"""Command-line interface for QEdge Real-Model Reference Runtime.

Prints valid JSON to stdout and logs/diagnostics to stderr.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from .config import (
    SMOLLM2_135M_ID,
    SMOLLM2_135M_REVISION,
    get_default_model_dir,
)
from .acquisition import check_model_assets
from .runner import ModelRunner


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QEdge Real-Model Reference Runtime CLI")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local model directory (default: unified canonical path)",
    )
    parser.add_argument("--prompt", default="What are three physical constraints of edge AI devices?", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum new tokens to generate")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None, help="Execution device")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None, help="Tensor dtype")
    parser.add_argument("--no-chat-template", action="store_true", help="Do not format with chat template")
    parser.add_argument("--raw-text", action="store_true", help="Output only generated text to stdout")
    args = parser.parse_args(argv)

    target_dir = (args.model_dir or get_default_model_dir()).resolve()

    valid, errors = check_model_assets(target_dir, expected_model_id=SMOLLM2_135M_ID, expected_revision=SMOLLM2_135M_REVISION)
    if not valid:
        sys.stderr.write(f"Error: Model assets not found or invalid in {target_dir}:\n")
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        sys.stderr.write("Use 'python -m qedge_runtime.acquisition' to download the pinned model.\n")
        return 1

    try:
        sys.stderr.write(f"Initializing QEdge Runtime (device: {args.device or 'auto'}, dtype: {args.dtype or 'auto'})...\n")
        runner = ModelRunner(
            model_dir=target_dir,
            device=args.device,
            dtype=args.dtype,
            local_files_only=True,
        )

        sys.stderr.write(f"Generating for prompt: {args.prompt[:60]}...\n")
        result = runner.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=not args.no_chat_template,
        )

        ttft_str = f"{result.ttft_ms:.1f}ms" if result.ttft_ms is not None else "N/A"
        sys.stderr.write(
            f"Done: {len(result.generated_token_ids)} tokens generated "
            f"(prefill {result.prefill_ms:.1f}ms, selection {result.first_token_select_ms:.1f}ms, "
            f"decode {result.decode_ms:.1f}ms, TTFT {ttft_str})\n"
        )

        if args.raw_text:
            print(result.generated_text)
        else:
            print(json.dumps(result.to_dict(), indent=2))
        return 0

    except Exception as e:
        sys.stderr.write(f"Error during runtime execution: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
