# Stage 09 Handoff: Provenance-Governed Catalog Retrieval & Evaluation

**Stage:** `09-catalog-retrieval`  
**Date:** 2026-11-03  
**Status:** REPAIRED & ACCEPTED for bounded 30-query pilot scope; 120-query evaluation gate remains BLOCKED pending human annotations for queries 31–120.  
**Preceding Acceptance:** Stage 08 (`08-quantization-measurement`) accepted for local SmolLM2 GGUF comparison, 24/24 quantization tests passed, 240 retained benchmark trials under `run_manifest_9d09726bb29b7e85`, 5 artifact/tool hashes verified, 30/30 historical manifest hashes verified.  
**Successor Gate:** Stage 10 (`10-integration-evaluation`) remains strictly BLOCKED until human annotations for queries 31–120 are completed and Stage 09 is independently reviewed. Second LLM `Qwen/Qwen2.5-0.5B-Instruct` remains strictly BLOCKED.  

---

## 1. Executive Summary & Review Audit

Following independent review findings on pilot judgment coverage, evaluation gate behavior, and neural index integrity, Stage 09 was completed with four bounded repairs:

### 1.1 Resolution of Review Findings

1. **Closed Neural Index Integrity Bypass & Propagated Schema Errors:**
   - **Mandatory File Digests:** `load_persisted_index` strictly verifies that `manifest["files"]` contains digests for all required assets: `ids.json`, `text.npy`, and `image.npy`. An index with `files: {}` is rejected immediately with `ValueError("missing required file digest in index manifest: ids.json")` before attempting to read any data.
   - **Catalog Alignment & Membership Verification:** `ids.json` is checked for unique item IDs and compared directly against loaded catalog rows (`catalog_ids`). Any invented item ID raises `ValueError("index item set does not match catalog items")`, and any permutation or row-swap raises `ValueError("index item alignment mismatch with catalog")`.
   - **Direct Schema Failure Propagation:** In `Retriever.search_request`, `qedge_common.schema.validate("retrieval_result", response)` now directly propagates contract validation errors (`ContractValidationError`) rather than catching them as warnings. Invalid candidate structures or fields cannot enter an evaluation run.

2. **Strict Evidence Validation & Provenance Verification:**
   - **Schema Contract Enforcement:** `validate_pilot_evidence` now validates every completed nested response against the canonical `qedge-common` `retrieval_result` schema contract before computing metrics. Contract validation failures are captured cleanly and converted to `status: "EVIDENCE_INVALID"` and `overall_status: "PILOT_EVAL_INCOMPLETE"` with the offending run tag and specific schema violation.
   - **Comprehensive Run Provenance Checks:** Validates every completed run and nested response against expected query ID, method (`bm25`, `text`, `hybrid`), catalog version (`abo-v2-pilot-500`), model versions (`local-bm25-v0`, `MiniLM+OpenCLIP-frozen`), index key (`a0cc05ffa89f8f148da8718649a919e8d54b4bf601ec2f8c14188fad5fcc6cca`), and candidate IDs.
   - **Explicit Method & Uniqueness Enforcement:** Explicitly rejects unknown methods outside `{"bm25", "text", "hybrid", "image"}` and enforces unique `(query_id, method)` pairs across all runs (completed and unavailable).
   - **Unavailable Image Request Validation:** Enforces exact 1:1 coverage by query identity (not just record count) ensuring all 30 task queries have an unavailable image request. Strictly requires a nonempty `reason` string for every unavailable run.
   - **Structural Integrity Enforcement:** Rejects duplicate candidate item IDs, duplicate ranks, non-contiguous ranks (enforces contiguous 1..N), unknown query IDs, and catalog item membership violations.
   - **Strict Label Verification:** Validates that label query IDs exist in the task manifest, item IDs exist in the catalog, `source == "human"`, grades are bounded in [0, 3], and catalog/index versions match task/run revisions.
   - **Ranking Depth Cutoffs:** Enforces minimum ranking depth ($k \ge 20$, with 17 allowed only for `pilot-08` BM25 where the 500-item catalog contains only 17 lexical matches). Runs with truncated depth fail validation with `EvidenceValidationError`.
   - **Independent Regression Tests (All Through `run_evaluation`):**
     - Nonnumeric candidate score (`"not-a-number"`): Caught by `retrieval_result` schema validation; returns `EVIDENCE_INVALID` / `PILOT_EVAL_INCOMPLETE` with offending run (`pilot-01/bm25`).
     - Missing schema-required response field (e.g. missing `timing`): Caught by schema validation; returns `EVIDENCE_INVALID` / `PILOT_EVAL_INCOMPLETE` with offending run (`pilot-01/bm25`).
     - Duplicate image request and missing query: Caught by duplicate pair validation; returns `EVIDENCE_INVALID` / `PILOT_EVAL_INCOMPLETE` with offending run (`pilot-01/image`).
     - Missing or blank unavailable reason: Caught by reason check; returns `EVIDENCE_INVALID` / `PILOT_EVAL_INCOMPLETE` with offending run (`pilot-01/image`).
     - Unknown method (`"unknown-method"`): Explicitly rejected; returns `EVIDENCE_INVALID` / `PILOT_EVAL_INCOMPLETE` with offending run (`pilot-01/unknown-method`).
     - Tampered index version (`wrong-revision`): Returns `EVIDENCE_INVALID` / `PILOT_EVAL_INCOMPLETE` with offending run (`pilot-01/bm25`).
     - Tampered candidate item ID (`invented-item`): Returns `EVIDENCE_INVALID` / `PILOT_EVAL_INCOMPLETE` with offending run (`pilot-01/bm25`).

3. **Explicit Judgment Coverage & Active Review Queue Generation:**
   - **Reconciled Pools Accounting:** Dynamically computes and reports coverage across historical, fresh, and union pools without hardcoded numbers:
     - **Historical Pool (`tasks.json`):** 685 candidate pairs (30 judged, 655 unjudged, 4.38% coverage).
     - **Fresh Pool (`candidate_runs.json`):** 1,032 unique pairs (23 judged, 1,009 unjudged, 2.23% coverage).
     - **Union Pool:** 1,172 unique pairs (30 judged, 1,142 unjudged, 2.56% coverage).
     - **Newly Retrieved Pairs:** 487 pairs in fresh rankings that were not in the historical pool.
   - **Active Review Queue:** Generated [`data/annotation/pilot-v1/review_queue.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/review_queue.json) containing exactly **1,009 unjudged pairs** from the fresh evaluated rankings with `query_id`, `item_id`, `candidate_id`, `query_text`, `status: "pending_human_review"`, and detailed `contributing_runs` (method, rank, score).
   - **Queue Preservations:** Preserved the 655-item historical review queue in [`data/annotation/pilot-v1/review_queue_historical.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/review_queue_historical.json) and exported the 1,142-item union queue in [`data/annotation/pilot-v1/review_queue_union.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/review_queue_union.json).
   - **Separation of Unjudged Candidates from Judged Negatives:** The evaluation explicitly notes that the 1,009 unjudged candidate pairs are not confirmed negatives.
   - **Known-Positive Rank Diagnostics ($N_{\text{pos}}=20$):** Known-positive rank diagnostics are scoped strictly to the 20 queries with verified positive labels ($\text{grade} \ge 1$), reported with their explicit denominator ($N_{\text{pos}}=20$).
   - **Reconciled Query Review Status:** Updated `status` in `tasks.json` to `"query_reviewed"` to reconcile with `labels.sqlite3`.

4. **Fresh Candidate Runs Bound to Implementation Under Review:**
   - Preserved historical 90 depth-10 runs in [`data/annotation/pilot-v1/candidate_runs_historical.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/candidate_runs_historical.json).
   - Generated 120 fresh candidate runs at depth $k=20$ directly through `Retriever(enable_neural=True)` across available methods (`bm25`, `text`, `hybrid`), recording method, actual modality (`lexical`, `dense_text`, `multimodal_rrf`), catalog/model/index revisions, depth, and timing. Explicitly recorded `image` search as unavailable for text-only queries.

---

## 2. Catalog & Index Provenance Inventory

| Dimension | Specification | Verification Result |
|---|---|---|
| **Catalog Version** | `abo-v2-pilot-500` | Pinned in `catalog.jsonl` and `manifest.json` |
| **Total Items / Variants** | 500 items / 500 unique variants | Exactly 500 unique `item_id`, 500 unique `variant_id` |
| **Catalog Manifest Files** | 548 files | 548/548 SHA256 checksums verified against disk |
| **License Attribution** | CC BY 4.0 | `sources/LICENSE-CC-BY-4.0.txt` (`419896aea50c...`) |
| **Source Git Commit** | `c5e7723f4a4afa3da4f35c497ab30ac0f8ee5b3e473ec0ffa897511e127e9324` | Pinned in `acquisition.json` and `sources/registry.yaml` |
| **Index Key** | `a0cc05ffa89f8f148da8718649a919e8d54b4bf601ec2f8c14188fad5fcc6cca` | Cryptographically derived and verified |
| **Index Preprocessing** | `v1:minilm-256tokens;clip-ViT-B-32-default-eval;fp32-normalized;cpu` | Checked in index manifest and loader |
| **Text Embedding Matrix** | `text.npy`: [500, 384] float32 | SHA256 `64a53488553f...`, unit normalized |
| **Image Embedding Matrix** | `image.npy`: [500, 512] float32 | SHA256 `486c05820c1c...`, unit normalized |
| **Index ID Inventory** | `ids.json`: 500 items | SHA256 `322ecc5e84d5...`, 1:1 aligned with catalog |

---

## 3. Pilot Judgment Coverage & Reconciled Review Queues

Across the 30 pilot queries, judgment coverage is dynamically reconciled across three scopes:

| Pool Scope | Total Candidate Pairs | Judged Pairs | Unjudged Pairs | Coverage (%) |
|---|---|---|---|---|
| **Historical Pool** (`tasks.json`) | 685 | 30 | 655 | 4.38% |
| **Fresh Pool** (`candidate_runs.json`, depth-20) | 1,032 | 23 | 1,009 | 2.23% |
| **Union Pool** (Historical $\cup$ Fresh) | 1,172 | 30 | 1,142 | 2.56% |

- **Newly Retrieved Pairs (Fresh $\setminus$ Historical):** **487 pairs** appeared in the evaluated top-20 candidate runs that were absent from the historical pool.
- **Queries with Known Positives:** **20 of 30 queries** (66.7%) have a confirmed positive label ($\text{grade} \ge 1$).
- **Queries without Known Positives (only $\text{grade} = 0$ judged):** **10 of 30 queries** (33.3%):  
  `['pilot-02', 'pilot-08', 'pilot-12', 'pilot-13', 'pilot-14', 'pilot-18', 'pilot-20', 'pilot-25', 'pilot-27', 'pilot-30']`
- **Active Review Queue ([`data/annotation/pilot-v1/review_queue.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/review_queue.json)):**  
  Contains exactly **1,009 entries** corresponding to the unjudged candidate pairs in the evaluated fresh pool. Each entry records `query_id`, `item_id`, `candidate_id`, `query_text`, `status: "pending_human_review"`, and detailed contributing runs with rank and score.
- **Historical & Union Queues:**  
  Historical 655-entry queue preserved at [`data/annotation/pilot-v1/review_queue_historical.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/review_queue_historical.json). Union 1,142-entry queue exported at [`data/annotation/pilot-v1/review_queue_union.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/review_queue_union.json).
- **Separation Notice:**  
  Because only 23 of 1,032 fresh pairs (and 30 of 1,172 union pairs) are judged, unjudged candidates at top ranks are **not confirmed negatives**. Full precision, false-positive analysis, and negative confirmation cannot be asserted until the 1,009 pairs in `review_queue.json` are evaluated by human reviewers.

---

## 4. Evaluation Results: Known-Positive Diagnostics & Full Pilot Metrics

### 4.1 Known-Positive Rank Diagnostics (Denominator $N_{\text{pos}} = 20$)

Evaluated strictly on the 20 queries containing a verified human positive ($\text{grade} \ge 1$):

| Diagnostic Metric | BM25 Baseline | Dense Text (MiniLM) | Hybrid (BM25 + Neural RRF) |
|---|---|---|---|
| **Hits at Rank $\le 5$** | $2 / 20$ ($0.1000$) | $\mathbf{6 / 20}$ ($\mathbf{0.3000}$) | $5 / 20$ ($0.2500$) |
| **Hits at Rank $\le 10$** | $6 / 20$ ($0.3000$) | $\mathbf{10 / 20}$ ($\mathbf{0.5000}$) | $9 / 20$ ($0.4500$) |
| **Hits at Rank $\le 20$** | $7 / 20$ ($0.3500$) | $10 / 20$ ($0.5000$) | $\mathbf{11 / 20}$ ($\mathbf{0.5500}$) |
| **Known-Positive MRR@10** | $0.0648$ | $\mathbf{0.2214}$ | $0.1905$ |

*Observation:* When evaluated on known positives, Dense Text retrieves 50% of targets in the top 10 with MRR@10 of 0.2214, while Hybrid RRF extends top-20 coverage to 55% (11/20).

### 4.2 Overall Pilot Metrics (Evaluated across all 30 queries)

Computed from fresh candidate runs at depth $k=20$ ([`edge-rag-demo/data/annotation/pilot-v1/candidate_runs.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/candidate_runs.json)):

| Metric | BM25 Baseline | Dense Text (MiniLM) | Hybrid (BM25 + Neural RRF) |
|---|---|---|---|
| **Recall@5** | $0.0667$ $[0.0000, 0.1667]$ | $\mathbf{0.2000}$ $[0.0667, 0.3333]$ | $0.1667$ $[0.0333, 0.3000]$ |
| **Recall@10** | $0.2000$ $[0.0667, 0.3667]$ | $\mathbf{0.3333}$ $[0.1667, 0.5000]$ | $0.3000$ $[0.1667, 0.4667]$ |
| **Recall@20** | $0.2333$ $[0.1000, 0.4000]$ | $0.3333$ $[0.1667, 0.5000]$ | $\mathbf{0.3667}$ $[0.2000, 0.5333]$ |
| **MRR@10** | $0.0432$ $[0.0120, 0.0875]$ | $\mathbf{0.1476}$ $[0.0526, 0.2648]$ | $0.1270$ $[0.0429, 0.2319]$ |
| **nDCG@10** | $0.0789$ $[0.0249, 0.1441]$ | $\mathbf{0.1896}$ $[0.0890, 0.3123]$ | $0.1671$ $[0.0689, 0.2810]$ |
| **Mean Encode Time** | $0.00$ ms | $291.09$ ms | $23.13$ ms |
| **Mean Retrieve Time** | $0.28$ ms | $0.71$ ms | $0.65$ ms |
| **Mean Total Time** | $0.30$ ms | $291.81$ ms | $23.99$ ms |
| **p95 Total Time** | $0.41$ ms | $26.66$ ms | $25.10$ ms |

*Note: 95% confidence intervals computed via 1,000 deterministic bootstrap iterations (seed=42).*

---

## 5. 120-Query Evaluation Plan & Gate Status

- **Dataset Name:** `abo-v2-eval-120` (120 queries across distinct product categories).
- **Split Policy:** Family-aware allocation (60 train, 20 dev, 40 test queries).
- **Tuning Constraint:** Hyperparameter tuning ($k=60$, modality weights $w_m$) strictly restricted to the development set (`dev`); test set remains sealed until evaluation.
- **Quality Gate:** **BLOCKED**
  - *Reason:* Human relevance judgments for queries 31–120 and the unjudged candidate pools are pending annotation. Quality gate cannot be unblocked without human labels.
  - *Unblocking Prerequisites:*
    1. Complete human review for the 1,009 unjudged candidate pairs in the active fresh review queue ([`review_queue.json`](file:///Users/manuvashistha/Developer/qualcomm/edge-rag-demo/data/annotation/pilot-v1/review_queue.json)).
    2. Complete double-blind human annotations for the remaining 90 queries (queries 31–120).

---

## 6. Verification Commands & Regression Log

| Test Suite | Command | Exit Code | Result |
|---|---|---|---|
| **Historical Manifest Hashes** | `python3 scripts/verify_historical_hashes.py` | `0` | **30 verified, 0 failures** |
| **Edge-RAG Baseline Tests** | `PYTHONPATH=qedge-common/src:edge-rag-demo/src edge-rag-demo/.venv/bin/python -m unittest discover edge-rag-demo/tests` | `0` | **104 passed, 1 skipped** |
| **Edge-RAG Integrated Tests** | `QEDGE_RUN_CATALOG_INTEGRATION=1 PYTHONPATH=qedge-common/src:edge-rag-demo/src edge-rag-demo/.venv/bin/python -m unittest discover edge-rag-demo/tests` | `0` | **105 passed, 0 skipped** |
| **Retrieval & Probe Tests** | `PYTHONPATH=qedge-common/src:edge-rag-demo/src edge-rag-demo/.venv/bin/python -m unittest edge-rag-demo/tests/test_retrieval.py` | `0` | **36 passed, 0 failed** |
| **Shared Schema Tests** | `PYTHONPATH=qedge-common/src edge-rag-demo/.venv/bin/python -m unittest discover qedge-common/tests` | `0` | **49 passed, 0 failed** |
| **Quantization Tests** | `cd llm-inference/runtime && .venv/bin/python -m unittest tests/test_quantization.py` | `0` | **24 passed, 0 failed** |
| **All Runtime Tests** | `cd llm-inference/runtime && .venv/bin/python -m unittest discover tests` | `0` | **123 passed, 0 failed** |

---

## 7. Self-Review Verdict & Successor Gate

- **Stage 09 Self-Review Verdict:**
  - **ACCEPTED for the bounded 30-query pilot scope:** Verified licensed ABO 500-item catalog, closed neural index integrity bypass, enforced membership/coverage/alignment checks, direct schema error propagation, comprehensive evidence and provenance validation (rejecting unknown items, duplicate candidates/ranks, non-contiguous ranks, inconsistent revisions, invalid labels, and truncated depth), independent negative probes passing, explicit judgment coverage across historical (685), fresh (1,032), and union (1,172) pools, 1,009-item active fresh review queue, separated 30 unavailable image requests, and known-positive rank diagnostics ($N_{\text{pos}}=20$).
  - **BLOCKED for the 120-query evaluation quality gate:** Queries 31–120 and the 1,009 pilot review pairs await human relevance annotations.
- **Successor Stage Gate:**
  - Stage 10 (`10-integration-evaluation`) and subsequent stages remain **strictly BLOCKED**.
  - Secondary model (`Qwen/Qwen2.5-0.5B-Instruct`) remains **strictly BLOCKED**.
  - All work stopped for independent review.
