"""Repository-relative path resolution for QEdge Assist.

No source file hardcodes a developer's home directory.  Data and optional
integrations are selected in this order:

1. an explicit argument;
2. a documented ``QEDGE_*`` environment variable;
3. a path relative to this repository.

Missing data is an explicit error, never a silent fallback to another
checkout.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class CatalogUnavailable(FileNotFoundError):
    """Raised when no catalog directory or catalog.jsonl file is available."""


def repo_root() -> Path:
    """Return the project root for editable checkouts, else the current directory.

    A wheel lives in site-packages, which is not a data directory. In that case
    catalog resolution starts from the caller's working directory and can be
    overridden with ``QEDGE_CATALOG_DIR``.
    """
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
        return candidate
    return Path.cwd().resolve()


def _resolve(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_catalog_dir(catalog_dir: Optional[str | Path] = None) -> Path:
    """Resolve a catalog directory (or a direct JSONL path)."""
    if catalog_dir is not None:
        return _resolve(catalog_dir)
    env_value = os.environ.get("QEDGE_CATALOG_DIR")
    if env_value:
        return _resolve(env_value)
    return (repo_root() / "catalog").resolve()


def resolve_catalog_file(catalog_dir: Optional[str | Path] = None) -> Path:
    """Return the catalog JSONL path, raising an explicit, actionable error."""
    candidate = resolve_catalog_dir(catalog_dir)
    if candidate.is_file():
        path = candidate
    else:
        path = candidate / "catalog.jsonl"
    if not path.is_file():
        raise CatalogUnavailable(
            "No catalog found. Set QEDGE_CATALOG_DIR to a directory containing "
            "catalog.jsonl, or pass catalog_dir. For an offline fixture demo, "
            "run qedge-portfolio demo."
        )
    return path


def resolve_llm_inference_root(llm_inference_dir: Optional[str | Path] = None) -> Path:
    """Resolve the sibling llm-inference checkout without a machine path."""
    if llm_inference_dir is not None:
        return _resolve(llm_inference_dir)
    env_value = os.environ.get("QEDGE_LLM_INFERENCE_DIR")
    if env_value:
        return _resolve(env_value)
    return (repo_root().parent / "llm-inference").resolve()


def resolve_index_dir(index_dir: Optional[str | Path] = None) -> Path:
    """Resolve the neural index directory without a machine path."""
    if index_dir is not None:
        return _resolve(index_dir)
    env_value = os.environ.get("QEDGE_INDEX_DIR")
    if env_value:
        return _resolve(env_value)
    return (repo_root() / "indexes" / "abo-v2-neural-v1").resolve()


def resolve_models_dir(models_dir: Optional[str | Path] = None) -> Path:
    """Resolve the local models directory without a machine path."""
    if models_dir is not None:
        return _resolve(models_dir)
    env_value = os.environ.get("QEDGE_MODELS_DIR")
    if env_value:
        return _resolve(env_value)
    return (repo_root() / "models").resolve()


def resolve_annotation_dir(annotation_dir: Optional[str | Path] = None) -> Path:
    """Resolve the annotation directory without a machine path."""
    if annotation_dir is not None:
        return _resolve(annotation_dir)
    env_value = os.environ.get("QEDGE_ANNOTATION_DIR")
    if env_value:
        return _resolve(env_value)
    return (repo_root() / "data" / "annotation").resolve()


def optional_catalog_file(catalog_dir: Optional[str | Path] = None) -> Optional[Path]:
    """Return a catalog file if it exists, otherwise ``None`` for callers that can skip."""
    try:
        return resolve_catalog_file(catalog_dir)
    except CatalogUnavailable:
        return None
