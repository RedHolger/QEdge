#!/usr/bin/env python3
"""Fail if machine-specific user paths leak into source or tests."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

MARKER = "/" + "Users" + "/"

DEFAULT_TARGETS = (
    "qedge-common/src",
    "qedge-common/tests",
    "qedge-common/schemas",
    "qedge-common/README.md",
    "edge-rag-demo/src",
    "edge-rag-demo/tests",
    "edge-rag-demo/demo_pipeline.py",
    "edge-rag-demo/README.md",
    "edge-rag-demo/Makefile",
    "edge-rag-demo/pyproject.toml",
    "llm-inference/src",
    "llm-inference/tests",
    "llm-inference/include",
    "llm-inference/README.md",
    "llm-inference/cpp/README.md",
    "llm-inference/runtime/README.md",
    "llm-inference/CMakeLists.txt",
    "llm-inference/CMakePresets.json",
    "llm-inference/Makefile",
    "README.md",
    "docs",
    "scripts/check_no_absolute_paths.py",
    "scripts/verify_historical_hashes.py",
    "Makefile",
)

SKIP_PARTS = {
    ".git",
    ".venv",
    "build",
    "dist",
    "__pycache__",
    "legacy",
    "handoff",
    "qedge-build-kit",
    ".pytest_cache",
    ".mypy_cache",
}


def iter_files(root: Path, target: str):
    path = root / target
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        # Immutable audit evidence can retain original machine paths.
        if child.relative_to(root).parts[:3] == ("docs", "research-review", "evidence"):
            continue
        if any(part in SKIP_PARTS for part in child.relative_to(root).parts):
            continue
        yield child


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()

    matches: list[tuple[str, int]] = []
    scanned = 0
    for target in DEFAULT_TARGETS:
        for path in iter_files(root, target):
            scanned += 1
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if MARKER in line:
                    matches.append((str(path.relative_to(root)), line_number))

    print(f"scanned {scanned} source/test files under {root}")
    if matches:
        for path, line_number in matches:
            print(f"FOUND machine path: {path}:{line_number}", file=sys.stderr)
        print(f"{len(matches)} machine-path match(es)", file=sys.stderr)
        return 1
    print("machine-specific user paths: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
