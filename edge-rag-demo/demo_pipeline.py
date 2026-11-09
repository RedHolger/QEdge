#!/usr/bin/env python3
"""Source-tree wrapper for the QEdge Assist CLI.

Installed packages should use the ``edge-rag-demo`` console script instead.
"""
from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
