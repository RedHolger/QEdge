"""Probe to verify offline execution without network access."""
import json
from pathlib import Path
import socket
import sys

from qedge_runtime.runner import ModelRunner
from qedge_runtime.acquisition import check_model_assets

MODEL_DIR = Path("llm-inference/models/smollm2-135m-instruct").resolve()


def run_offline_test() -> int:
    print("=== Testing Offline Execution (Simulating Network Block) ===")
    valid, missing = check_model_assets(MODEL_DIR)
    if not valid:
        print(f"FAIL: Assets missing: {missing}", file=sys.stderr)
        return 1

    # Intercept socket to prevent network egress
    orig_socket = socket.socket

    def guard_socket(*args, **kwargs):
        raise OSError("Network egress blocked in offline test")

    socket.socket = guard_socket
    try:
        runner = ModelRunner(model_dir=MODEL_DIR, device="cpu", dtype="float32", local_files_only=True)
        res = runner.generate("What is 5 + 5?", max_new_tokens=8)
        assert len(res.generated_token_ids) > 0, "No tokens generated"
        print(f"PASS: Offline generation succeeded without network access: {res.generated_text!r}")
        return 0
    finally:
        socket.socket = orig_socket


if __name__ == "__main__":
    sys.exit(run_offline_test())
