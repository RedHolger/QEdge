# Stage 09 — Final Evaluation and Failure Analysis

## Goal

Produce the evidence package: retrieval, evidence, abstention, runtime, and end-to-end results with confidence intervals, ablations, and failure analysis. No new features.

## Read first

- `03-DATA-AND-EVALUATION.md`
- `04-ACCEPTANCE-AND-DELIVERABLES.md` (stage 09)
- all previous handoffs

## Tasks

1. **Freeze splits**
   - 30-query dev, 90-query held-out test (or the documented smaller set).
   - 40-case dev, 40-case held-out decision test.
   - No tuning on test.
   - Record split hashes.

2. **Retrieval**
   - BM25, dense, hybrid, hybrid+rerank.
   - Recall@5, NDCG@10, MRR@10.
   - Paired bootstrap CIs.
   - Per-slice results.
   - Zero-overlap failures.

3. **Evidence**
   - Decision accuracy.
   - False-`SUPPORTED` rate.
   - Citation validity.
   - Abstention precision/recall.
   - Unit conversion accuracy.
   - Stale-source behavior.

4. **Runtime**
   - FP16 vs Q8_0 vs Q4_K_M.
   - Full vs FIFO/LRU/KeyDiff/QUOKA at 100/75/50/25%.
   - TTFT, prefill/decode tok/s, peak RSS.
   - Quality proxy.
   - Hardware labels.

5. **End-to-end**
   - 10 scenarios.
   - Task success.
   - Unsupported-claim rate.
   - p50/p95 latency.
   - Memory.
   - Backend per stage.

6. **Ablations**
   - remove image retrieval;
   - remove reranker;
   - remove KV policy;
   - remove generation;
   - show what each contributes.

7. **Failure taxonomy**
   - retrieval miss;
   - wrong attribute extraction;
   - missing unit;
   - conflicting source;
   - verifier error;
   - generation unsupported claim;
   - latency regression.

8. **Claims ledger**
   - Update `docs/CLAIMS.md`.
   - Every headline claim links to a JSON/CSV artifact and a command.
   - Mark unverified NPU claims as unverified.

9. **Report**
   - `reports/09-final-evaluation.md`.
   - Tables and plots.
   - Limitations.
   - What you would do with more data/hardware.

## Acceptance

- all required metrics reported;
- CIs and effect sizes;
- no test-set tuning;
- failures included;
- every claim backed;
- no stale README number;
- report reproducible from `make eval`.

## Required handoff

`handoff/09.md` in each repo:

- tables;
- artifact paths;
- claims;
- limitations;
- "stopped here."

## Guardrails

- Do not add features to improve a number.
- Do not remove a failing slice.
- Do not report only the best run.
- Do not compare across hardware without labels.

## Stop

Stop after the handoff. Ask for review.
