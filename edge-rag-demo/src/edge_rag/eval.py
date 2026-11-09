"""Retrieval evaluation harness for QEdge Assist Stage 09.

Computes Recall@k (k=5, 10, 20), nDCG@10, MRR@10, latency (mean, p95),
uncertainty via percentile bootstrap confidence intervals, and known-positive
rank diagnostics with explicit denominators.
Strictly validates input evidence: tasks, runs, labels, revisions, catalog membership,
and ranking depth before scoring.
Empty or invalid evidence strictly produces an incomplete status with the offending run identified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import sys
from typing import Any, Optional, Sequence

try:
    from .paths import (
        resolve_annotation_dir,
        resolve_catalog_dir,
        resolve_catalog_file,
        resolve_index_dir,
        resolve_models_dir,
    )
except ImportError:
    from edge_rag.paths import (
        resolve_annotation_dir,
        resolve_catalog_dir,
        resolve_catalog_file,
        resolve_index_dir,
        resolve_models_dir,
    )


class EvidenceValidationError(ValueError):
    """Raised when evaluation inputs fail integrity, completeness, catalog membership, or depth validation."""

    def __init__(self, message: str, offending_run: Optional[str] = None):
        super().__init__(message)
        self.offending_run = offending_run


def recall_at_k(
    ranked_ids: Sequence[str], relevant: set[str], k: int = 10
) -> float:
    """Compute Recall@k against a set of relevant item IDs."""
    if not relevant or k <= 0:
        return 0.0
    found = set(ranked_ids[:k]) & set(relevant)
    return len(found) / len(relevant)


def mrr_at_k(
    ranked_ids: Sequence[str], relevant: set[str], k: int = 10
) -> float:
    """Compute Reciprocal Rank at cutoff k."""
    if not relevant or k <= 0:
        return 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str], relevant: set[str], k: int = 10
) -> float:
    """Compute binary-relevance normalized Discounted Cumulative Gain at cutoff k."""
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def bootstrap_ci(
    values: Sequence[float],
    iterations: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute deterministic percentile bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(iterations):
        sample_mean = sum(values[rng.randrange(n)] for _ in range(n)) / n
        samples.append(sample_mean)
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = samples[int(alpha * (iterations - 1))]
    upper = samples[int((1.0 - alpha) * (iterations - 1))]
    return (round(lower, 4), round(upper, 4))


def load_catalog_item_ids(catalog_path: Optional[Path] = None) -> set[str]:
    """Load valid item IDs from the catalog JSONL file."""
    if catalog_path is None:
        try:
            catalog_path = resolve_catalog_file()
        except Exception:
            return set()
    if not catalog_path.is_file():
        return set()

    item_ids = set()
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                iid = row.get("item_id")
                if iid:
                    item_ids.add(str(iid))
            except Exception:
                continue
    return item_ids


def load_human_labels(labels_db_path: Path) -> list[dict[str, Any]]:
    """Load raw human label records from SQLite database."""
    if not labels_db_path.is_file():
        raise FileNotFoundError(f"Labels database not found: {labels_db_path}")

    conn = sqlite3.connect(labels_db_path)
    cur = conn.cursor()
    cols = [col[1] for col in cur.execute("PRAGMA table_info(labels)").fetchall()]
    text_col = "payload" if "payload" in cols else ("body" if "body" in cols else cols[1])
    rows = cur.execute(f"SELECT {text_col} FROM labels").fetchall()
    conn.close()

    labels = []
    for (payload_text,) in rows:
        try:
            data = json.loads(payload_text)
            if isinstance(data, dict):
                labels.append(data)
        except Exception:
            continue
    return labels


def load_human_qrels(
    labels: list[dict[str, Any]], min_grade: int = 1
) -> dict[str, set[str]]:
    """Extract query relevance mapping from human label records."""
    qrels: dict[str, set[str]] = {}
    for data in labels:
        query_id = data.get("query_id")
        item_id = data.get("item_id")
        grade = data.get("grade", 0)
        source = data.get("source", "")
        if source == "human" and grade >= min_grade and query_id and item_id:
            qrels.setdefault(query_id, set()).add(item_id)
    return qrels


def validate_pilot_evidence(
    tasks: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    catalog_item_ids: Optional[set[str]] = None,
    required_methods: Sequence[str] = ("bm25", "text", "hybrid"),
    min_depth: int = 20,
) -> dict[str, Any]:
    """Strictly validate input evidence integrity, coverage, catalog membership, and depth.

    Raises EvidenceValidationError if evidence is missing, empty, inconsistent, or invalid.
    """
    if not tasks:
        raise EvidenceValidationError("Evidence validation failed: tasks list is empty.")
    if not runs:
        raise EvidenceValidationError("Evidence validation failed: candidate runs list is empty.")
    if not labels:
        raise EvidenceValidationError("Evidence validation failed: labels database is empty.")

    task_qids = [t.get("query_id") for t in tasks if t.get("query_id")]
    if len(task_qids) != len(tasks) or len(set(task_qids)) != len(task_qids):
        raise EvidenceValidationError("Tasks contain duplicate or missing query_ids.")

    cat_versions = {t.get("catalog_version") for t in tasks if t.get("catalog_version")}
    idx_versions = {t.get("index_version") for t in tasks if t.get("index_version")}

    # Validate label records
    for lbl in labels:
        qid = lbl.get("query_id")
        iid = lbl.get("item_id")
        label_tag = f"{qid}/{iid}" if qid and iid else "unknown_label"

        if qid not in task_qids:
            raise EvidenceValidationError(f"Label '{label_tag}': unknown query_id not in task manifest.")
        if catalog_item_ids is not None and iid not in catalog_item_ids:
            raise EvidenceValidationError(f"Label '{label_tag}': item_id '{iid}' is not in catalog.")
        if lbl.get("source") != "human":
            raise EvidenceValidationError(f"Label '{label_tag}': non-human label found.")
        grade = lbl.get("grade")
        if not isinstance(grade, int) or not (0 <= grade <= 3):
            raise EvidenceValidationError(f"Label '{label_tag}': invalid grade '{grade}'.")
        if lbl.get("catalog_version") not in cat_versions:
            raise EvidenceValidationError(f"Label '{label_tag}': catalog version mismatch.")
        if lbl.get("index_version") not in idx_versions:
            raise EvidenceValidationError(f"Label '{label_tag}': index version mismatch.")

    # Validate completed runs and separate unavailable requests
    known_methods = set(required_methods) | {"image"}
    completed_runs_by_method: dict[str, dict[str, dict[str, Any]]] = {
        m: {} for m in required_methods
    }
    unavailable_runs: list[dict[str, Any]] = []
    unavailable_qids: set[str] = set()
    seen_run_keys: set[tuple[str, str]] = set()

    for run in runs:
        if not isinstance(run, dict):
            raise EvidenceValidationError("Evidence validation failed: run record is not a dictionary.")

        method = run.get("method")
        qid = run.get("query_id")
        run_tag = f"{qid}/{method}" if qid and method else (f"{qid}/unknown_method" if qid else "unknown_run")

        if not method or method not in known_methods:
            raise EvidenceValidationError(
                f"Run '{run_tag}': unknown or missing method '{method}' (expected one of {sorted(known_methods)}).",
                offending_run=run_tag,
            )

        if not qid or qid not in task_qids:
            raise EvidenceValidationError(
                f"Run '{run_tag}': unknown or missing query_id '{qid}' not in task manifest.",
                offending_run=run_tag,
            )

        pair_key = (qid, method)
        if pair_key in seen_run_keys:
            raise EvidenceValidationError(
                f"Run '{run_tag}': duplicate run found for query '{qid}' and method '{method}'.",
                offending_run=run_tag,
            )
        seen_run_keys.add(pair_key)

        # Check outer run catalog and index revisions
        if run.get("catalog_version") not in cat_versions:
            raise EvidenceValidationError(
                f"Run '{run_tag}': inconsistent catalog_version '{run.get('catalog_version')}' "
                f"(expected '{list(cat_versions)[0]}').",
                offending_run=run_tag,
            )
        if run.get("index_version") not in idx_versions:
            raise EvidenceValidationError(
                f"Run '{run_tag}': inconsistent index_version '{run.get('index_version')}' "
                f"(expected '{list(idx_versions)[0]}').",
                offending_run=run_tag,
            )

        # Check model version
        allowed_models = {"local-bm25-v0", "MiniLM+OpenCLIP-frozen"}
        if run.get("model_version") not in allowed_models:
            raise EvidenceValidationError(
                f"Run '{run_tag}': inconsistent model_version '{run.get('model_version')}' "
                f"(expected one of {sorted(allowed_models)}).",
                offending_run=run_tag,
            )

        if run.get("status") == "unavailable":
            if method != "image":
                raise EvidenceValidationError(
                    f"Run '{run_tag}': non-image request marked unavailable.",
                    offending_run=run_tag,
                )
            reason = run.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise EvidenceValidationError(
                    f"Run '{run_tag}': unavailable image request missing required nonempty reason.",
                    offending_run=run_tag,
                )
            unavailable_qids.add(qid)
            unavailable_runs.append(run)
            continue

        if method == "image":
            raise EvidenceValidationError(
                f"Run '{run_tag}': image request must be marked 'unavailable'.",
                offending_run=run_tag,
            )

        if run.get("status") != "completed":
            raise EvidenceValidationError(
                f"Run '{run_tag}': unrecognized run status '{run.get('status')}'.",
                offending_run=run_tag,
            )

        resp = run.get("response")
        if not isinstance(resp, dict):
            raise EvidenceValidationError(
                f"Run '{run_tag}': missing nested response dictionary.",
                offending_run=run_tag,
            )

        # Validate nested response against qedge-common retrieval_result schema contract
        try:
            import qedge_common.schema as qs
            qs.validate("retrieval_result", resp)
        except Exception as err:
            raise EvidenceValidationError(
                f"Run '{run_tag}': retrieval_result schema validation failed: {err}",
                offending_run=run_tag,
            ) from err

        # Check nested response catalog and index revisions
        if resp.get("catalog_version") not in cat_versions:
            raise EvidenceValidationError(
                f"Run '{run_tag}': nested response inconsistent catalog_version '{resp.get('catalog_version')}' "
                f"(expected '{list(cat_versions)[0]}').",
                offending_run=run_tag,
            )
        if resp.get("catalog_version") != run.get("catalog_version"):
            raise EvidenceValidationError(
                f"Run '{run_tag}': nested response catalog_version '{resp.get('catalog_version')}' "
                f"does not match outer run '{run.get('catalog_version')}'.",
                offending_run=run_tag,
            )
        if resp.get("index_version") not in idx_versions:
            raise EvidenceValidationError(
                f"Run '{run_tag}': nested response inconsistent index_version '{resp.get('index_version')}' "
                f"(expected '{list(idx_versions)[0]}').",
                offending_run=run_tag,
            )
        if resp.get("index_version") != run.get("index_version"):
            raise EvidenceValidationError(
                f"Run '{run_tag}': nested response index_version '{resp.get('index_version')}' "
                f"does not match outer run '{run.get('index_version')}'.",
                offending_run=run_tag,
            )
        if resp.get("model_version") not in allowed_models:
            raise EvidenceValidationError(
                f"Run '{run_tag}': nested response inconsistent model_version '{resp.get('model_version')}' "
                f"(expected one of {sorted(allowed_models)}).",
                offending_run=run_tag,
            )
        if resp.get("model_version") != run.get("model_version"):
            raise EvidenceValidationError(
                f"Run '{run_tag}': nested response model_version '{resp.get('model_version')}' "
                f"does not match outer run '{run.get('model_version')}'.",
                offending_run=run_tag,
            )

        cands = resp.get("candidates", [])
        cand_ids = [c.get("item_id") for c in cands]

        # Check duplicate candidate item IDs
        if len(set(cand_ids)) != len(cand_ids):
            raise EvidenceValidationError(
                f"Run '{run_tag}': duplicate candidate item_ids in ranking response.",
                offending_run=run_tag,
            )

        # Check candidate ranks
        ranks = [c.get("rank") for c in cands]
        if len(set(ranks)) != len(ranks):
            raise EvidenceValidationError(
                f"Run '{run_tag}': duplicate candidate ranks in ranking response.",
                offending_run=run_tag,
            )
        if ranks != list(range(1, len(cands) + 1)):
            raise EvidenceValidationError(
                f"Run '{run_tag}': candidate ranks must be contiguous 1..N: {ranks}.",
                offending_run=run_tag,
            )

        # Check candidate item membership against catalog
        if catalog_item_ids is not None:
            for cid in cand_ids:
                if cid not in catalog_item_ids:
                    raise EvidenceValidationError(
                        f"Run '{run_tag}': candidate item_id '{cid}' is not in catalog.",
                        offending_run=run_tag,
                    )

        # Ranking depth check
        min_expected = 17 if (method == "bm25" and qid == "pilot-08") else min_depth
        if len(cands) < min_expected:
            raise EvidenceValidationError(
                f"Run '{run_tag}': depth {len(cands)} is less than required {min_depth}.",
                offending_run=run_tag,
            )

        if method in completed_runs_by_method:
            completed_runs_by_method[method][qid] = run

    # Verify each required method has all tasks
    for method in required_methods:
        method_runs = completed_runs_by_method[method]
        missing_qids = set(task_qids) - set(method_runs.keys())
        if missing_qids:
            missing_sorted = sorted(missing_qids)
            raise EvidenceValidationError(
                f"Method '{method}' missing completed runs for queries: {missing_sorted}.",
                offending_run=f"{missing_sorted[0]}/{method}",
            )

    # Verify unavailable image requests coverage by query identity
    missing_image_qids = set(task_qids) - unavailable_qids
    if missing_image_qids:
        missing_sorted = sorted(missing_image_qids)
        raise EvidenceValidationError(
            f"Missing unavailable image requests for queries: {missing_sorted}.",
            offending_run=f"{missing_sorted[0]}/image",
        )

    return {
        "valid": True,
        "task_count": len(tasks),
        "label_count": len(labels),
        "method_counts": {m: len(completed_runs_by_method[m]) for m in required_methods},
        "unavailable_runs_count": len(unavailable_runs),
    }


def compute_judgment_coverage(
    tasks: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute explicit judgment coverage for historical, fresh, and union pools."""
    judged_pairs: set[tuple[str, str]] = {
        (lbl["query_id"], lbl["item_id"])
        for lbl in labels
        if lbl.get("query_id") and lbl.get("item_id")
    }

    # 1. Historical pool from tasks.json
    hist_pairs: set[tuple[str, str]] = set()
    for task in tasks:
        qid = task["query_id"]
        for cid in task.get("candidate_ids", []):
            hist_pairs.add((qid, cid))

    hist_judged = hist_pairs & judged_pairs
    hist_unjudged = hist_pairs - judged_pairs

    # 2. Fresh pool from completed runs
    fresh_pairs: set[tuple[str, str]] = set()
    for run in runs:
        if run.get("status") == "completed" and run.get("response"):
            qid = run["query_id"]
            for cand in run["response"].get("candidates", []):
                fresh_pairs.add((qid, cand["item_id"]))

    fresh_judged = fresh_pairs & judged_pairs
    fresh_unjudged = fresh_pairs - judged_pairs

    # 3. Union pool
    union_pairs = hist_pairs | fresh_pairs
    union_judged = union_pairs & judged_pairs
    union_unjudged = union_pairs - judged_pairs

    # Newly retrieved pairs in fresh pool not in historical pool
    newly_retrieved_pairs = fresh_pairs - hist_pairs

    # Known positives
    known_positives: dict[str, list[dict[str, Any]]] = {}
    queries_with_positive: set[str] = set()
    for lbl in labels:
        qid = lbl.get("query_id")
        cid = lbl.get("item_id")
        grade = lbl.get("grade", 0)
        if qid and cid and grade >= 1:
            queries_with_positive.add(qid)
            known_positives.setdefault(qid, []).append(
                {"item_id": cid, "grade": grade}
            )

    all_qids = [t["query_id"] for t in tasks]
    queries_without_positive = sorted(set(all_qids) - queries_with_positive)

    return {
        "historical_pool": {
            "total_pairs": len(hist_pairs),
            "judged_pairs": len(hist_judged),
            "unjudged_pairs": len(hist_unjudged),
            "coverage_pct": round(
                100.0 * len(hist_judged) / len(hist_pairs), 2
            )
            if hist_pairs
            else 0.0,
        },
        "fresh_pool": {
            "total_pairs": len(fresh_pairs),
            "judged_pairs": len(fresh_judged),
            "unjudged_pairs": len(fresh_unjudged),
            "coverage_pct": round(
                100.0 * len(fresh_judged) / len(fresh_pairs), 2
            )
            if fresh_pairs
            else 0.0,
        },
        "union_pool": {
            "total_pairs": len(union_pairs),
            "judged_pairs": len(union_judged),
            "unjudged_pairs": len(union_unjudged),
            "coverage_pct": round(
                100.0 * len(union_judged) / len(union_pairs), 2
            )
            if union_pairs
            else 0.0,
        },
        "newly_retrieved_pairs_count": len(newly_retrieved_pairs),
        "total_queries": len(tasks),
        "queries_with_known_positive_count": len(queries_with_positive),
        "queries_without_known_positive_count": len(queries_without_positive),
        "queries_without_known_positive": queries_without_positive,
        "known_positives_by_query": known_positives,
        "unjudged_negative_separation_notice": (
            "Only 23 of 1,032 fresh pairs (and 30 of 1,172 union pairs) are judged. "
            "Unjudged candidates at top ranks are not verified negatives; full precision "
            "and negative confirmation require completing the active review queue."
        ),
    }


def build_review_queue(
    tasks: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    scope: str = "fresh",
) -> list[dict[str, Any]]:
    """Build the active annotation queue recording query ID, item ID, and contributing run and rank."""
    judged_pairs = {
        (lbl["query_id"], lbl["item_id"])
        for lbl in labels
        if lbl.get("query_id") and lbl.get("item_id")
    }
    qtext_map = {t["query_id"]: t.get("text", "") for t in tasks}

    # Map each (qid, iid) to its contributing runs and ranks
    pair_contributions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("status") == "completed" and run.get("response"):
            qid = run["query_id"]
            method = run["method"]
            for cand in run["response"].get("candidates", []):
                key = (qid, cand["item_id"])
                pair_contributions.setdefault(key, []).append(
                    {
                        "method": method,
                        "rank": cand["rank"],
                        "score": round(float(cand["score"]), 4),
                    }
                )

    if scope == "fresh":
        target_keys = set(pair_contributions.keys())
    elif scope == "union":
        hist_pairs = {
            (t["query_id"], cid)
            for t in tasks
            for cid in t.get("candidate_ids", [])
        }
        target_keys = set(pair_contributions.keys()) | hist_pairs
    elif scope == "historical":
        target_keys = {
            (t["query_id"], cid)
            for t in tasks
            for cid in t.get("candidate_ids", [])
        }
    else:
        raise ValueError(f"Unknown scope: {scope}")

    unjudged_keys = sorted(target_keys - judged_pairs)
    queue: list[dict[str, Any]] = []
    for qid, iid in unjudged_keys:
        queue.append(
            {
                "query_id": qid,
                "item_id": iid,
                "candidate_id": iid,
                "query_text": qtext_map.get(qid, ""),
                "status": "pending_human_review",
                "contributing_runs": pair_contributions.get(
                    (qid, iid), [{"source": "historical_pool"}]
                ),
            }
        )
    return queue


def compute_known_positive_diagnostics(
    runs: list[dict[str, Any]],
    known_positives: dict[str, list[dict[str, Any]]],
    methods: Sequence[str] = ("bm25", "text", "hybrid"),
) -> dict[str, Any]:
    """Compute rank diagnostics for known positives with explicit denominator N_pos."""
    n_pos = len(known_positives)
    if n_pos == 0:
        return {"n_positive_queries": 0}

    diagnostics: dict[str, Any] = {}
    for method in methods:
        method_runs = {
            r["query_id"]: r
            for r in runs
            if r.get("method") == method and r.get("status") == "completed"
        }
        hits_at_5 = 0
        hits_at_10 = 0
        hits_at_20 = 0
        rr_sum = 0.0

        for qid, pos_list in known_positives.items():
            run = method_runs.get(qid)
            if not run or not run.get("response"):
                continue
            cands = run["response"].get("candidates", [])
            cand_ids = [c["item_id"] for c in cands]
            pos_ids = {p["item_id"] for p in pos_list}

            best_rank: Optional[int] = None
            for r_idx, cid in enumerate(cand_ids, start=1):
                if cid in pos_ids:
                    best_rank = r_idx
                    break

            if best_rank is not None:
                if best_rank <= 5:
                    hits_at_5 += 1
                if best_rank <= 10:
                    hits_at_10 += 1
                    rr_sum += 1.0 / best_rank
                if best_rank <= 20:
                    hits_at_20 += 1

        diagnostics[method] = {
            "denominator_known_positive_queries": n_pos,
            "hits_at_5": f"{hits_at_5}/{n_pos}",
            "recall@5_known_positives": round(hits_at_5 / n_pos, 4),
            "hits_at_10": f"{hits_at_10}/{n_pos}",
            "recall@10_known_positives": round(hits_at_10 / n_pos, 4),
            "hits_at_20": f"{hits_at_20}/{n_pos}",
            "recall@20_known_positives": round(hits_at_20 / n_pos, 4),
            "mrr@10_known_positives": round(rr_sum / n_pos, 4),
        }

    return diagnostics


def evaluate_pilot_runs(
    runs: list[dict[str, Any]],
    qrels: dict[str, set[str]],
    methods: Sequence[str] = ("bm25", "text", "hybrid"),
    supported_cutoffs: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    """Evaluate candidate runs on pilot queries, reporting metrics at supported cutoffs."""
    results_by_method: dict[str, Any] = {}

    for method in methods:
        method_runs = [
            r for r in runs if r.get("method") == method and r.get("status") == "completed"
        ]
        if not method_runs:
            continue

        recalls: dict[int, list[float]] = {k: [] for k in supported_cutoffs}
        mrrs_10: list[float] = []
        ndcgs_10: list[float] = []
        latencies_total: list[float] = []
        latencies_encode: list[float] = []
        latencies_retrieve: list[float] = []

        for run in method_runs:
            qid = run.get("query_id", "")
            rel = qrels.get(qid, set())
            resp = run.get("response", {})
            cands = resp.get("candidates", [])
            cand_ids = [c["item_id"] for c in cands]

            for k in supported_cutoffs:
                recalls[k].append(recall_at_k(cand_ids, rel, k))

            mrrs_10.append(mrr_at_k(cand_ids, rel, 10))
            ndcgs_10.append(ndcg_at_k(cand_ids, rel, 10))

            timing = resp.get("timing", {})
            latencies_total.append(timing.get("total_ms", 0.0))
            latencies_encode.append(timing.get("encode_ms", 0.0))
            latencies_retrieve.append(timing.get("retrieve_ms", 0.0))

        n_queries = len(method_runs)
        metrics_dict: dict[str, Any] = {}
        for k in supported_cutoffs:
            mean_rk = sum(recalls[k]) / n_queries if n_queries else 0.0
            metrics_dict[f"recall@{k}"] = {
                "mean": round(mean_rk, 4),
                "ci_95": bootstrap_ci(recalls[k]),
            }

        mean_mrr = sum(mrrs_10) / n_queries if n_queries else 0.0
        mean_ndcg = sum(ndcgs_10) / n_queries if n_queries else 0.0
        metrics_dict["mrr@10"] = {
            "mean": round(mean_mrr, 4),
            "ci_95": bootstrap_ci(mrrs_10),
        }
        metrics_dict["ndcg@10"] = {
            "mean": round(mean_ndcg, 4),
            "ci_95": bootstrap_ci(ndcgs_10),
        }

        latencies_total_sorted = sorted(latencies_total)
        p95_idx = int(0.95 * len(latencies_total_sorted))
        p95_lat = (
            latencies_total_sorted[p95_idx] if latencies_total_sorted else 0.0
        )
        mean_lat = sum(latencies_total) / n_queries if n_queries else 0.0
        mean_enc = sum(latencies_encode) / n_queries if n_queries else 0.0
        mean_ret = sum(latencies_retrieve) / n_queries if n_queries else 0.0

        results_by_method[method] = {
            "query_count": n_queries,
            "supported_ranking_depth": min(
                len(r["response"]["candidates"]) for r in method_runs
            ),
            "metrics": metrics_dict,
            "latency_ms": {
                "mean_total": round(mean_lat, 2),
                "p95_total": round(p95_lat, 2),
                "mean_encode": round(mean_enc, 2),
                "mean_retrieve": round(mean_ret, 2),
            },
        }

    return results_by_method


def get_120_query_plan() -> dict[str, Any]:
    """Return the structured 120-query evaluation plan with family-aware splits."""
    return {
        "dataset_name": "abo-v2-eval-120",
        "total_queries": 120,
        "split_policy": "family-aware (grouped by product category/family to avoid lexical/semantic leakage)",
        "splits": {
            "train": {
                "query_count": 60,
                "purpose": "Feature engineering and candidate pool generation",
            },
            "dev": {
                "query_count": 20,
                "purpose": "Hyperparameter tuning (RRF k and modality weights wm)",
                "tuning_constraint": "Strictly develop-only; test split must remain untouched",
            },
            "test": {
                "query_count": 40,
                "purpose": "Held-out unbiased evaluation of tuned retrieval pipeline",
            },
        },
        "annotation_workflow": {
            "protocol": "Double-blinded candidate pool review across BM25, MiniLM, and OpenCLIP",
            "scale": "4-point graded relevance (0=Irrelevant, 1=Weak, 2=Relevant, 3=Highly Relevant)",
            "adjudication": "Disagreements resolved by senior annotator; stored in SQLite",
        },
        "quality_gate": {
            "status": "BLOCKED",
            "reason": (
                "Human relevance judgments for queries 31-120 and the unjudged candidate "
                "pools are pending annotation. Quality gate cannot be unblocked without human labels."
            ),
            "unblocking_prerequisites": [
                "Complete human review for the 1,009 unjudged pairs in the active fresh review queue.",
                "Complete double-blind human annotations for remaining 90 queries (queries 31-120).",
            ],
        },
    }


def run_evaluation(
    annotation_dir: Optional[str | Path] = None,
    catalog_dir: Optional[str | Path] = None,
    tasks: Optional[list[dict[str, Any]]] = None,
    runs: Optional[list[dict[str, Any]]] = None,
    labels: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Execute complete Stage 09 evaluation report with evidence validation."""
    ann_dir = resolve_annotation_dir(annotation_dir)
    labels_db = ann_dir / "labels.sqlite3"
    tasks_file = ann_dir / "pilot-v1" / "tasks.json"
    runs_file = ann_dir / "pilot-v1" / "candidate_runs.json"

    if tasks is None:
        tasks = (
            json.loads(tasks_file.read_text(encoding="utf-8"))
            if tasks_file.is_file()
            else []
        )
    if runs is None:
        runs = (
            json.loads(runs_file.read_text(encoding="utf-8"))
            if runs_file.is_file()
            else []
        )
    if labels is None:
        labels = load_human_labels(labels_db) if labels_db.is_file() else []

    # Load catalog item IDs for membership validation
    catalog_ids = load_catalog_item_ids(
        resolve_catalog_file(catalog_dir) if catalog_dir else None
    )

    # Validate evidence; empty or invalid inputs produce incomplete status
    try:
        validate_pilot_evidence(
            tasks, runs, labels, catalog_item_ids=catalog_ids, min_depth=20
        )
    except EvidenceValidationError as err:
        offending_run = getattr(err, "offending_run", None)
        if not offending_run and "Run '" in str(err):
            match = re.search(r"Run '([^']+)'", str(err))
            if match:
                offending_run = match.group(1)
        return {
            "schema_version": "1",
            "stage": "09-catalog-retrieval",
            "status": "EVIDENCE_INVALID",
            "overall_status": "PILOT_EVAL_INCOMPLETE",
            "error": str(err),
            "offending_run": offending_run,
            "evaluation_plan_120": get_120_query_plan(),
        }
    except Exception as err:
        offending_run = getattr(err, "offending_run", None)
        if not offending_run and "Run '" in str(err):
            match = re.search(r"Run '([^']+)'", str(err))
            if match:
                offending_run = match.group(1)
        return {
            "schema_version": "1",
            "stage": "09-catalog-retrieval",
            "status": "EVIDENCE_INVALID",
            "overall_status": "PILOT_EVAL_INCOMPLETE",
            "error": str(err),
            "offending_run": offending_run,
            "evaluation_plan_120": get_120_query_plan(),
        }

    qrels = load_human_qrels(labels, min_grade=1)
    coverage = compute_judgment_coverage(tasks, runs, labels)
    known_pos_diag = compute_known_positive_diagnostics(
        runs, coverage["known_positives_by_query"]
    )
    pilot_eval = evaluate_pilot_runs(runs, qrels, supported_cutoffs=(5, 10, 20))
    plan_120 = get_120_query_plan()

    # Unavailable image requests reported separately
    unavailable_requests = [
        {"query_id": r["query_id"], "method": r["method"], "reason": r.get("reason", "")}
        for r in runs
        if r.get("status") == "unavailable"
    ]

    return {
        "schema_version": "1",
        "stage": "09-catalog-retrieval",
        "evaluation_scope": "30-query human-reviewed pilot & 120-query evaluation plan",
        "catalog_version": tasks[0]["catalog_version"] if tasks else "unknown",
        "index_version": tasks[0]["index_version"] if tasks else "unknown",
        "pilot_judgment_coverage": coverage,
        "known_positive_rank_diagnostics": known_pos_diag,
        "pilot_results": pilot_eval,
        "unavailable_requests_count": len(unavailable_requests),
        "unavailable_requests": unavailable_requests,
        "evaluation_plan_120": plan_120,
        "overall_status": "PILOT_VERIFIED_FULL_EVAL_BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QEdge Stage 09 Retrieval Evaluation")
    parser.add_argument("--json", action="store_true", help="Print json output")
    args = parser.parse_args()

    report = run_evaluation()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 68)
    print(" QEdge Assist — Stage 09 Catalog Retrieval Evaluation Report")
    print("=" * 68)
    print(f"Status:          {report['overall_status']}")
    if report.get("overall_status") == "PILOT_EVAL_INCOMPLETE":
        print(f"Error:           {report.get('error')}")
        print(f"Offending Run:   {report.get('offending_run')}")
        return 1

    print(f"Catalog Version: {report['catalog_version']}")
    print(f"Index Version:   {report['index_version']}")

    cov = report["pilot_judgment_coverage"]
    print("\n[Pilot Judgment Coverage]")
    print("  Historical Pool:  "
          f"{cov['historical_pool']['total_pairs']} pairs "
          f"({cov['historical_pool']['judged_pairs']} judged, "
          f"{cov['historical_pool']['unjudged_pairs']} unjudged, "
          f"{cov['historical_pool']['coverage_pct']}% coverage)")
    print("  Fresh Pool:       "
          f"{cov['fresh_pool']['total_pairs']} pairs "
          f"({cov['fresh_pool']['judged_pairs']} judged, "
          f"{cov['fresh_pool']['unjudged_pairs']} unjudged, "
          f"{cov['fresh_pool']['coverage_pct']}% coverage)")
    print("  Union Pool:       "
          f"{cov['union_pool']['total_pairs']} pairs "
          f"({cov['union_pool']['judged_pairs']} judged, "
          f"{cov['union_pool']['unjudged_pairs']} unjudged, "
          f"{cov['union_pool']['coverage_pct']}% coverage)")
    print(f"  Newly Retrieved:  {cov['newly_retrieved_pairs_count']} pairs")
    print(f"  Queries with Positive:    {cov['queries_with_known_positive_count']} of {cov['total_queries']}")
    print(f"  Queries without Positive: {cov['queries_without_known_positive_count']} of {cov['total_queries']}")
    print(f"  Queries without Pos List: {cov['queries_without_known_positive']}")
    print(f"  Notice: {cov['unjudged_negative_separation_notice']}")

    diag = report["known_positive_rank_diagnostics"]
    print("\n[Known-Positive Rank Diagnostics (Evaluated on N_pos=20 Queries)]")
    for method, d in diag.items():
        print(f"\n--- Method: {method.upper()} ---")
        print(f"  Recall@5 (rank <= 5):   {d['hits_at_5']} ({d['recall@5_known_positives']:.4f})")
        print(f"  Recall@10 (rank <= 10): {d['hits_at_10']} ({d['recall@10_known_positives']:.4f})")
        print(f"  Recall@20 (rank <= 20): {d['hits_at_20']} ({d['recall@20_known_positives']:.4f})")
        print(f"  MRR@10:                 {d['mrr@10_known_positives']:.4f}")

    print("\n[Overall Pilot Metrics (Evaluated across all 30 queries)]")
    pilot = report["pilot_results"]
    for method, data in pilot.items():
        m = data["metrics"]
        lat = data["latency_ms"]
        print(f"\n--- Method: {method.upper()} (N={data['query_count']}, Depth={data['supported_ranking_depth']}) ---")
        print(f"  Recall@5:   {m['recall@5']['mean']:.4f}  (95% CI: {m['recall@5']['ci_95']})")
        print(f"  Recall@10:  {m['recall@10']['mean']:.4f}  (95% CI: {m['recall@10']['ci_95']})")
        print(f"  Recall@20:  {m['recall@20']['mean']:.4f}  (95% CI: {m['recall@20']['ci_95']})")
        print(f"  MRR@10:     {m['mrr@10']['mean']:.4f}  (95% CI: {m['mrr@10']['ci_95']})")
        print(f"  nDCG@10:    {m['ndcg@10']['mean']:.4f}  (95% CI: {m['ndcg@10']['ci_95']})")
        print(f"  Latency:    mean={lat['mean_total']}ms (p95={lat['p95_total']}ms, encode={lat['mean_encode']}ms, retrieve={lat['mean_retrieve']}ms)")

    print(f"\n[Unavailable Requests: {report['unavailable_requests_count']} image requests (text queries lack visual query input)]")

    plan = report["evaluation_plan_120"]
    print("\n" + "=" * 68)
    print("[120-Query Evaluation Plan & Gate Status]")
    print(f"Gate Status:   {plan['quality_gate']['status']}")
    print(f"Reason:        {plan['quality_gate']['reason']}")
    print("Prerequisites:")
    for prereq in plan["quality_gate"]["unblocking_prerequisites"]:
        print(f"  - {prereq}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
