"""Deterministic catalog retrieval supporting BM25 baseline and neural/hybrid retrieval.

Maintains BM25 as the default baseline. Neural text (MiniLM) and image (OpenCLIP)
modalities are gated behind an explicit feature flag (enable_neural or QEDGE_ENABLE_NEURAL=1).
Persisted indexes are verified against catalog, model, and preprocessing revisions;
any mismatch strictly invalidates the index.
Deterministic weighted Reciprocal Rank Fusion (RRF) combines modalities with
1-based ranks, stable tie-breaking, and optional family deduplication.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence
import uuid

from .paths import (
    CatalogUnavailable,
    resolve_annotation_dir,
    resolve_catalog_dir,
    resolve_catalog_file,
    resolve_index_dir,
    resolve_models_dir,
)

PREPROCESS = "v1:minilm-256tokens;clip-ViT-B-32-default-eval;fp32-normalized;cpu"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has in is it its of on or that the "
    "this to was were will with".split()
)


REQUIRED_INDEX_FILES = ("ids.json", "text.npy", "image.npy")


def sha256_file(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_index_key(
    catalog_manifest_path: Path,
    model_manifest_path: Path,
    preprocess: str = PREPROCESS,
) -> str:
    """Compute the immutable index key from catalog, model, and preprocessing revisions."""
    cat_digest = sha256_file(catalog_manifest_path)
    mod_digest = sha256_file(model_manifest_path)
    payload = (cat_digest + mod_digest + preprocess).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tokenize(text: str) -> list[str]:
    return [
        term
        for term in _TOKEN_RE.findall((text or "").lower())
        if term not in _STOPWORDS
    ]


@dataclass(frozen=True, slots=True)
class ScoredItem:
    item_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id is required")
        if math.isnan(self.score) or math.isinf(self.score):
            raise ValueError("score must be finite")


class _BM25Index:
    def __init__(self, item_ids: list[str], texts: list[str]):
        self.item_ids = item_ids
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.doc_lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.document_frequency: dict[str, int] = {}
        for counts in self.term_counts:
            for term in counts:
                self.document_frequency[term] = self.document_frequency.get(term, 0) + 1

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self.item_ids or self.average_length <= 0:
            return []
        k1, b = 1.2, 0.75
        n_docs = len(self.item_ids)
        scored: list[tuple[str, float]] = []
        for item_id, counts, length in zip(
            self.item_ids, self.term_counts, self.doc_lengths
        ):
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self.document_frequency.get(term, 0)
                if document_frequency == 0:
                    continue
                inverse_document_frequency = math.log(
                    1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * length / self.average_length
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (k1 + 1.0)
                    / denominator
                )
            if score > 0:
                scored.append((item_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]


def load_persisted_index(
    index_dir: Path,
    expected_key: str,
    catalog_ids: Optional[Sequence[str]] = None,
) -> tuple[list[str], Any, Any, dict[str, Any]]:
    """Load and validate persisted neural embeddings.

    Strictly validates key and checksums. Requires digests for all expected index
    files (ids.json, text.npy, image.npy) in manifest['files']. Validates ID uniqueness,
    membership, coverage, and row alignment against catalog.
    Raises ValueError on any corruption, mismatch, missing digest, or misaligned row.
    """
    import numpy as np

    manifest_file = index_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Index manifest not found: {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("key") != expected_key:
        raise ValueError("stale catalog/model/preprocessing index")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("index manifest missing 'files' dictionary")

    # Strictly require digests for all required index files
    for req_file in REQUIRED_INDEX_FILES:
        if req_file not in files or not files[req_file]:
            raise ValueError(f"missing required file digest in index manifest: {req_file}")

    for name, exp_hash in files.items():
        file_path = index_dir / name
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing index asset: {file_path}")
        if sha256_file(file_path) != exp_hash:
            raise ValueError(f"index file checksum mismatch: {name}")

    ids_file = index_dir / "ids.json"
    ids = json.loads(ids_file.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids:
        raise ValueError("index ids.json must contain a non-empty list of item IDs")

    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item IDs in index ids.json")

    if catalog_ids is not None:
        cat_ids_list = [str(x) for x in catalog_ids]
        if set(ids) != set(cat_ids_list):
            raise ValueError("index item set does not match catalog items")
        if list(ids) != cat_ids_list:
            raise ValueError("index item alignment mismatch with catalog")

    text_matrix = np.load(index_dir / "text.npy", allow_pickle=False)
    image_matrix = np.load(index_dir / "image.npy", allow_pickle=False)

    for matrix, dim, name in [
        (text_matrix, 384, "text"),
        (image_matrix, 512, "image"),
    ]:
        if matrix.shape != (len(ids), dim) or not np.isfinite(matrix).all():
            raise ValueError(f"index shape or numeric corruption: {name}")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(f"unnormalized index: {name}")

    return ids, text_matrix, image_matrix, manifest


class NeuralEncoders:
    """Lazy loader for pinned MiniLM text and OpenCLIP text/vision encoders."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._minilm = None
        self._clip = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._minilm is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import open_clip
        from sentence_transformers import SentenceTransformer
        import torch

        torch.set_num_threads(4)
        self._torch = torch

        minilm_path = self.models_dir / "minilm"
        self._minilm = SentenceTransformer(
            str(minilm_path), device="cpu", local_files_only=True
        )
        self._minilm.max_seq_length = 256

        clip_weights = self.models_dir / "clip" / "open_clip_model.safetensors"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=str(clip_weights),
            device="cpu",
        )
        model.eval()
        self._clip = model
        self._clip_preprocess = preprocess
        self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def encode_text(self, text: str) -> Any:
        import numpy as np

        self._ensure_loaded()
        emb = self._minilm.encode([text], convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.maximum(norm, 1e-8))[0]

    def encode_clip_text(self, text: str) -> Any:
        import numpy as np

        self._ensure_loaded()
        tokens = self._clip_tokenizer([text])
        with self._torch.inference_mode():
            emb = self._clip.encode_text(tokens).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.maximum(norm, 1e-8))[0]

    def encode_image(self, image_data: Any) -> Any:
        import numpy as np
        from PIL import Image

        self._ensure_loaded()
        if isinstance(image_data, (bytes, bytearray)):
            im = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, (str, Path)):
            im = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, Image.Image):
            im = image_data.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_data)}")

        processed = self._clip_preprocess(im).unsqueeze(0)
        with self._torch.inference_mode():
            emb = self._clip.encode_image(processed).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.maximum(norm, 1e-8))[0]


def rrf(
    rankings: dict[str, Sequence[tuple[str, float]]],
    weights: Optional[dict[str, float]] = None,
    k: int = 60,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Deterministic weighted reciprocal-rank fusion with one-based ranks.

    Score(d) = sum_m [ w_m * (1 / (k + rank_m(d))) ]
    Tie-breaking is strictly (-score, item_id).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if limit is not None and limit <= 0:
        return []

    scores: dict[str, float] = {}
    component_scores: dict[str, dict[str, float]] = {}
    rank_origins: dict[str, dict[str, int]] = {}

    for mod_name, ranking in rankings.items():
        w = weights.get(mod_name, 1.0) if weights else 1.0
        seen: set[str] = set()
        for rank, (item_id, raw_score) in enumerate(ranking, start=1):
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            increment = w * (1.0 / (k + rank))
            scores[item_id] = scores.get(item_id, 0.0) + increment

            if item_id not in component_scores:
                component_scores[item_id] = {}
                rank_origins[item_id] = {}
            component_scores[item_id][mod_name] = float(raw_score)
            rank_origins[item_id][mod_name] = rank

    ordered = [
        {
            "item_id": item_id,
            "score": score,
            "component_scores": component_scores[item_id],
            "rank_origins": rank_origins[item_id],
        }
        for item_id, score in scores.items()
    ]
    ordered.sort(key=lambda x: (-x["score"], x["item_id"]))
    return ordered[:limit] if limit is not None else ordered


class Retriever:
    """Retrieve product candidates supporting BM25 and neural/hybrid modalities.

    BM25 is the default baseline (local-bm25-v0). Dense MiniLM and OpenCLIP are
    gated behind `enable_neural=True` or `QEDGE_ENABLE_NEURAL=1`.
    """

    method = "local-bm25"
    model_version = "local-bm25-v0"

    def __init__(
        self,
        catalog_dir: Optional[str | Path] = None,
        models_dir: Optional[str | Path] = None,
        index_dir: Optional[str | Path] = None,
        engine: Optional[Any] = None,
        catalog_path: Optional[str | Path] = None,
        enable_neural: Optional[bool] = None,
        rrf_k: int = 60,
        weights: Optional[dict[str, float]] = None,
        dedup_families: bool = False,
    ):
        self.engine = engine
        self.catalog_file: Optional[Path] = None
        self.catalog_dir: Optional[Path] = None
        self.rows: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.catalog_version = "unavailable"
        self._index: Optional[_BM25Index] = None
        self.rrf_k = rrf_k
        self.weights = weights or {"bm25": 1.0, "text": 1.0, "text_to_image": 1.0}
        self.dedup_families = dedup_families
        self.family_map: dict[str, str] = {}
        self.last_result: Optional[dict[str, Any]] = None

        if enable_neural is None:
            enable_neural = os.environ.get("QEDGE_ENABLE_NEURAL", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        self.enable_neural = enable_neural

        self.neural_encoders: Optional[NeuralEncoders] = None
        self.text_matrix: Optional[Any] = None
        self.image_matrix: Optional[Any] = None
        self.index_manifest: Optional[dict[str, Any]] = None
        self.index_version: str = "unavailable"
        self.item_positions: dict[str, int] = {}
        self.asset_to_id: dict[str, str] = {}

        if engine is not None:
            self.catalog_version = str(getattr(engine, "catalog_version", "injected"))
            return

        selected = catalog_path if catalog_path is not None else catalog_dir
        self.catalog_file = resolve_catalog_file(selected)
        self.catalog_dir = self.catalog_file.parent
        self.rows = self._load_rows(self.catalog_file)
        self.by_id = {str(row["item_id"]): row for row in self.rows}
        versions = {str(row.get("catalog_version", "unknown")) for row in self.rows}
        self.catalog_version = versions.pop() if len(versions) == 1 else "mixed"
        self._index = _BM25Index(
            [str(row["item_id"]) for row in self.rows],
            [f"{row.get('title', '')}\n{row.get('text', '')}" for row in self.rows],
        )

        for row in self.rows:
            item_id = str(row["item_id"])
            image_asset = row.get("image_asset_id")
            if image_asset:
                self.asset_to_id[str(image_asset)] = item_id

        # Load duplicate families if available
        dup_file = self.catalog_dir / "reports" / "duplicate_families.json"
        if dup_file.is_file():
            try:
                families = json.loads(dup_file.read_text(encoding="utf-8"))
                for fam in families:
                    fam_id = fam.get("family_id") or fam.get("canonical_id")
                    for member in fam.get("member_ids", []):
                        self.family_map[member] = fam_id
            except Exception:
                pass

        if self.enable_neural:
            resolved_models = resolve_models_dir(models_dir)
            resolved_index = resolve_index_dir(index_dir)
            cat_manifest = self.catalog_dir / "manifest.json"
            mod_manifest = resolved_models / "manifest.json"

            if not cat_manifest.is_file() or not mod_manifest.is_file():
                raise ValueError("Neural retrieval requested but manifest missing")

            expected_key = compute_index_key(cat_manifest, mod_manifest)
            ids, text_matrix, image_matrix, manifest = load_persisted_index(
                resolved_index,
                expected_key,
                catalog_ids=[str(row["item_id"]) for row in self.rows],
            )
            self.text_matrix = text_matrix
            self.image_matrix = image_matrix
            self.index_manifest = manifest
            self.index_version = expected_key
            self.item_positions = {item_id: idx for idx, item_id in enumerate(ids)}
            self.neural_encoders = NeuralEncoders(resolved_models)
            self.model_version = "MiniLM+OpenCLIP-frozen"
            self.method = "hybrid"

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("item_id"):
                    raise ValueError(f"invalid catalog row at {path}:{line_number}")
                rows.append(row)
        if not rows:
            raise CatalogUnavailable(f"catalog is empty: {path}")
        return rows

    def retrieve(
        self,
        query: str,
        k: int = 5,
        method: str = "bm25",
        image_asset_id: Optional[str] = None,
        uploaded_image: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve ranked catalog rows conforming to Stage 01/04 baseline expectations."""
        if k <= 0:
            return []
        if self.engine is not None:
            injected = getattr(self.engine, "retrieve", None)
            if injected is None:
                raise TypeError("injected retrieval engine must provide retrieve(query, k)")
            return list(injected(query, k))

        request = {
            "schema_version": "1",
            "query_id": f"q-{uuid.uuid4().hex[:8]}",
            "text": query or None,
            "image_asset_id": image_asset_id,
            "k": k,
            "catalog_version": self.catalog_version,
            "method": method,
        }
        res = self.search_request(request, uploaded_image=uploaded_image)
        self.last_result = res

        results: list[dict[str, Any]] = []
        for candidate in res.get("candidates", []):
            item_id = candidate["item_id"]
            row = self.by_id.get(item_id, {})
            results.append(
                {
                    "item_id": item_id,
                    "variant_id": row.get("variant_id", item_id),
                    "title": row.get("title", "Untitled Product"),
                    "category": row.get("category", ""),
                    "rank": candidate["rank"],
                    "score": round(float(candidate["score"]), 4),
                    "attributes": row.get("attributes", {}),
                    "component_scores": candidate.get("component_scores", {}),
                }
            )
        return results

    def search_request(
        self,
        request: dict[str, Any],
        uploaded_image: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Execute a structured search request conforming to retrieval_result.schema.json."""
        start_time = time.perf_counter()
        req_method = request.get("method", "bm25")
        k = request.get("k", 10)
        query_text = request.get("text")
        image_asset_id = request.get("image_asset_id")
        request_id = request.get("query_id", uuid.uuid4().hex)

        warnings: list[str] = []
        encode_start = time.perf_counter()

        # Handle neural feature flag check
        if req_method != "bm25" and not self.enable_neural:
            warnings.append(
                f"Neural method '{req_method}' requested but neural retrieval is disabled. "
                "Returning BM25 baseline only."
            )
            req_method = "bm25"

        q_text_emb = None
        q_clip_text_emb = None
        q_image_emb = None

        if self.enable_neural and self.neural_encoders is not None:
            if query_text and req_method in ("text", "hybrid"):
                q_text_emb = self.neural_encoders.encode_text(query_text)
                q_clip_text_emb = self.neural_encoders.encode_clip_text(query_text)

            if uploaded_image is not None and req_method in ("image", "hybrid"):
                q_image_emb = self.neural_encoders.encode_image(uploaded_image)
            elif image_asset_id and req_method in ("image", "hybrid"):
                if image_asset_id in self.asset_to_id:
                    matched_id = self.asset_to_id[image_asset_id]
                    if matched_id in self.item_positions:
                        pos = self.item_positions[matched_id]
                        q_image_emb = self.image_matrix[pos]
                else:
                    warnings.append(f"Unknown image_asset_id: {image_asset_id}")

        encode_ms = (time.perf_counter() - encode_start) * 1000.0

        retrieve_start = time.perf_counter()
        rankings: dict[str, list[tuple[str, float]]] = {}
        component_maps: dict[str, dict[str, float]] = {}

        # 1. BM25
        if query_text and self._index is not None:
            bm25_matches = self._index.search(query_text, k=100)
            rankings["bm25"] = bm25_matches
            component_maps["bm25"] = dict(bm25_matches)

        # 2. Text (MiniLM)
        if q_text_emb is not None and self.text_matrix is not None:
            scores = self.text_matrix @ q_text_emb
            text_ranked = [
                (item_id, float(scores[idx]))
                for idx, item_id in enumerate(self.item_positions.keys())
            ]
            text_ranked.sort(key=lambda pair: (-pair[1], pair[0]))
            rankings["text"] = text_ranked[:100]
            component_maps["text"] = dict(text_ranked[:100])

        # 3. Text to Image (OpenCLIP)
        if q_clip_text_emb is not None and self.image_matrix is not None:
            scores = self.image_matrix @ q_clip_text_emb
            clip_ranked = [
                (item_id, float(scores[idx]))
                for idx, item_id in enumerate(self.item_positions.keys())
            ]
            clip_ranked.sort(key=lambda pair: (-pair[1], pair[0]))
            rankings["text_to_image"] = clip_ranked[:100]
            component_maps["text_to_image"] = dict(clip_ranked[:100])

        # 4. Image to Image (OpenCLIP)
        if q_image_emb is not None and self.image_matrix is not None:
            scores = self.image_matrix @ q_image_emb
            img_ranked = [
                (item_id, float(scores[idx]))
                for idx, item_id in enumerate(self.item_positions.keys())
            ]
            img_ranked.sort(key=lambda pair: (-pair[1], pair[0]))
            rankings["image"] = img_ranked[:100]
            component_maps["image"] = dict(img_ranked[:100])

        retrieve_ms = (time.perf_counter() - retrieve_start) * 1000.0

        rerank_start = time.perf_counter()
        candidates: list[dict[str, Any]] = []

        if req_method in ("bm25", "text", "image"):
            target_ranking = rankings.get(req_method, [])
            seen_families: set[str] = set()
            rank_counter = 1
            for item_id, sc in target_ranking:
                if self.dedup_families and self.family_map:
                    fam_id = self.family_map.get(item_id, item_id)
                    if fam_id in seen_families:
                        continue
                    seen_families.add(fam_id)

                c_scores = {req_method: float(sc)}
                for other_mod, m_map in component_maps.items():
                    if other_mod != req_method and item_id in m_map:
                        c_scores[other_mod] = float(m_map[item_id])
                candidates.append(
                    {
                        "item_id": item_id,
                        "rank": rank_counter,
                        "score": float(sc),
                        "component_scores": c_scores,
                    }
                )
                rank_counter += 1
                if rank_counter > k:
                    break
        else:  # hybrid
            fused = rrf(rankings, weights=self.weights, k=self.rrf_k, limit=100)
            seen_families: set[str] = set()
            rank_counter = 1
            for item in fused:
                item_id = item["item_id"]
                if self.dedup_families and self.family_map:
                    fam_id = self.family_map.get(item_id, item_id)
                    if fam_id in seen_families:
                        continue
                    seen_families.add(fam_id)

                candidates.append(
                    {
                        "item_id": item_id,
                        "rank": rank_counter,
                        "score": float(item["score"]),
                        "component_scores": item["component_scores"],
                    }
                )
                rank_counter += 1
                if rank_counter > k:
                    break

        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        total_ms = (time.perf_counter() - start_time) * 1000.0

        if not warnings:
            warnings.append("Exploratory ranking; similarity is not proof of product facts.")

        response = {
            "schema_version": "1",
            "request_id": request_id,
            "catalog_version": self.catalog_version,
            "model_version": self.model_version,
            "index_version": self.index_version,
            "ranker_version": None,
            "candidates": candidates,
            "timing": {
                "encode_ms": round(encode_ms, 2),
                "retrieve_ms": round(retrieve_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "warnings": warnings,
        }

        # Validate against schema strictly; propagate failures as errors
        try:
            import qedge_common.schema as qs
        except ImportError as exc:
            raise RuntimeError(
                f"qedge-common is required to validate retrieval results: {exc}"
            ) from exc

        qs.validate("retrieval_result", response)

        return response

    def describe(self) -> str:
        if self.engine is not None:
            return "injected retrieval engine"
        if self.enable_neural:
            return f"hybrid retriever (MiniLM + OpenCLIP + BM25, k={self.rrf_k})"
        return "local BM25 (Stage 01 baseline; not neural)"
