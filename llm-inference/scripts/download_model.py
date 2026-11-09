#!/usr/bin/env python3
"""
Helper script to download and convert a small LLM for testing.
Target: SmolLM-135M or TinyLlama-1.1B in a simple format.

Usage:
  python scripts/download_model.py --model tinyllama-1.1b
  python scripts/download_model.py --model smolLM-135m
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "tinyllama-1.1b": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "files": ["model.safetensors", "tokenizer.json", "config.json"],
    },
    "smolLM-135m": {
        "repo": "HuggingFaceTB/SmolLM2-135M",
        "files": ["model.safetensors", "tokenizer.json", "config.json"],
    },
}

def main():
    parser = argparse.ArgumentParser(description="Download models for inference engine")
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()

if __name__ == "__main__":
    args = main()
    model_dir = Path(args.output_dir) / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    info = MODELS[args.model]
    print(f"Downloading {args.model} from {info['repo']}...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(info["repo"], local_dir=str(model_dir))
        print(f"Downloaded to {model_dir}")
    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        print(f"Or manually download {info['repo']} to {model_dir}")
        sys.exit(1)
