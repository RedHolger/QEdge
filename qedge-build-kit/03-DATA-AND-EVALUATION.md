# QEdge Data, Training, and Evaluation Protocol

## 1. Data sources and the license gate

### Catalog

Preferred: Amazon Berkeley Objects (ABO) 500-item pilot, with provenance from the upstream catalog.

Hard gate before any image is shipped:

- Confirm the exact license for the catalog metadata, images, and derived embeddings.
- If the license is ambiguous or non-commercial, either:
  - ship metadata only and use placeholders, or
  - use a clearly licensed substitute catalog.
- Record the decision in `edge-rag-demo/catalog/LICENSE-DECISION.md`.

No stage may say "500 real product images" unless the license decision is recorded.

### Models

- SmolLM2-135M-Instruct: Apache-2.0.
- Qwen2.5-0.5B-Instruct: Apache-2.0.
- GGUF quantized files: record the exact repository, revision, hash, and license.
- Do not commit model weights to the repository. Record download commands and hashes.

### Human judgments

Human labels are the ground truth for retrieval and evidence quality. OpenCode may build the tool and prepare candidate sets; it may not invent labels.

## 2. Catalog preparation

For each item:

- stable `item_id` and `variant_id`;
- title, category, text description;
- source URL, revision, hash;
- image path and image hash (if licensed);
- attributes with `value`, `unit`, `status`;
- license reference.

Normalization rules:

- store original value and unit;
- convert only with an explicit formula and record both;
- no default unit if the source omitted one;
- no inferred dimensions from the title unless the title is the cited source;
- deduplicate families so near-identical variants do not inflate retrieval metrics.

Deliverables:

- `catalog/catalog.jsonl`
- `catalog/manifest.json` (counts, hashes, license, source revision)
- `catalog/provenance_report.md`
- `catalog/sample_gallery.html` (local images or placeholders)
- `catalog/duplicate_families.json`

## 3. Retrieval evaluation

### Query sets

Pilot: 30 queries.

Portfolio: 120 queries.

Splits:

- 30-query dev set for iteration;
- 90-query held-out test set, frozen before tuning;
- no query in both splits;
- no product from a test query used to write a dev query.

Query types:

- exact attribute ("oak dining chair under 60cm");
- visual/taste ("chair like this image");
- broad category ("accent chair");
- zero-overlap ("chair with a built-in phone charger");
- ambiguous ("small chair");
- negation ("chair, not leather");
- unit conversion ("under 24 inches").

Labels:

- For each query, a human reviews a candidate pool of at least 50 items.
- Mark each candidate `relevant`, `partially_relevant`, or `not_relevant`.
- Record the reviewer, timestamp, and rationale for ambiguous cases.
- At least 20% of queries double-annotated; report agreement.

Do not generate qrels by selecting the product used to write the query. That measures the construction, not retrieval.

### Metrics

- Recall@5, Recall@10
- NDCG@10
- MRR@10
- Zero-overlap failure rate
- Per-slice results: text-only, image-only, attribute-heavy, negation, unit conversion

Statistical protocol:

- paired bootstrap over queries (10,000 resamples) for method comparisons;
- report 95% confidence intervals;
- do not claim significance from overlapping marginal intervals;
- report effect sizes, not just p-values.

### Baselines and ablations

- BM25
- Dense text embeddings (MiniLM or similar)
- CLIP text→image
- Hybrid RRF
- Hybrid + cross-encoder reranker
- Remove one component at a time
- Oracle upper bound (human labels)

## 4. Evidence and abstention evaluation

### Decision cases

80 cases total:

- 40 dev, 40 held-out test;
- 20 `SUPPORTED`
- 20 `CONTRADICTED`
- 20 `UNKNOWN` / insufficient evidence
- 20 edge cases (unit mismatch, conflicting attributes, stale source, missing image)

Each case:

```json
{
  "case_id": "...",
  "query": "...",
  "item_id": "...",
  "requirements": {"width_lte_cm": 60, "material": "oak"},
  "expected_decision": "SUPPORTED",
  "expected_evidence": ["catalog.item_dimensions.width"],
  "notes": "..."
}
```

Metrics:

- decision accuracy;
- false-`SUPPORTED` rate (the most important safety metric);
- citation validity (does the cited source contain the claim?);
- abstention precision/recall on `UNKNOWN`;
- unit-conversion accuracy;
- stale-source behavior.

Rules:

- A `CONTRADICTED` or `UNKNOWN` item must never be presented as matching.
- The answer must show the observed value and the requirement.
- If the source is missing, the answer must abstain.

## 5. Runtime evaluation

### Quality

- Greedy output parity with the reference implementation for a fixed prompt set.
- Perplexity on a held-out text sample, if the model/tokenizer supports it.
- Task accuracy on a small set of instruction prompts (e.g., 20 prompts with deterministic checks).
- Cache-policy quality at budgets 100%, 75%, 50%, 25%.

### Performance

Measure on named hardware:

- TTFT (ms);
- prefill tok/s;
- decode tok/s;
- p50/p95 over at least 20 runs after 3 warmups;
- peak RSS;
- model file size;
- cache size;
- energy proxy (see below).

Energy:

- On Mac, `powermetrics` may require elevated privileges and is noisy; use it only if available and record the command.
- Otherwise report a **modelled** energy estimate with the exact formula and label it `modelled`.
- Never call a modelled number a measurement.
- On Snapdragon, use QNN/AI Hub profiling or device power telemetry; record the trace.

### Ablations

- FP16 vs Q8_0 vs Q4_K_M;
- full cache vs FIFO/LRU/KeyDiff/QUOKA;
- 100/75/50/25% budgets;
- CPU vs Metal (Mac) if available;
- CPU vs QNN (only with device logs).

### Statistical protocol

- same prompt set for all methods;
- fixed seeds;
- report medians and confidence intervals;
- separate cold-start from warm runs;
- record thermal state if available;
- do not compare runs from different hardware in the same table without a label.

## 6. End-to-end evaluation

For 10 demo scenarios, record:

- retrieval rank of the correct item;
- decision correctness;
- citation correctness;
- whether the answer abstained when it should;
- TTFT and total latency;
- peak memory;
- which stage ran on which backend.

Task success = correct item retrieved + correct decision + valid citation + no unsupported claim.

## 7. Annotation workflow

Build a small local tool:

- input: query + candidate pool;
- output: relevance label, rationale, reviewer, timestamp;
- keyboard-first UI;
- no external service;
- export to JSONL;
- validation: every query has at least one label; every case has an expected decision.

Pilot protocol:

1. 30 queries.
2. Two reviewers on 20% of queries.
3. Compute agreement.
4. Fix ambiguous guidelines.
5. Freeze the dev/test split.
6. Only then run the 120-query set.

## 8. What invalidates an experiment

- Synthetic qrels presented as human labels.
- Tuning on the test split.
- Comparing methods with different candidate pools.
- Comparing MRR at different cutoffs.
- Reporting only successful queries.
- Reporting a single run without warmup.
- Calling a modelled number measured.
- Calling a Mac CPU result an NPU result.
- Changing historical results without preserving the original.

## 9. Reproducibility checklist

- pinned Python version and lockfile;
- pinned model revisions and hashes;
- pinned catalog revision and hash;
- fixed seeds;
- one command per experiment;
- JSON artifacts for every table;
- hardware metadata in every artifact;
- no absolute paths;
- no network requirement for tests;
- explicit skips for missing hardware.

## 10. Minimum evidence for the portfolio claim

You may say:

> "On a 120-query human-reviewed set, hybrid retrieval improved Recall@5 over BM25 by X points (95% CI ...), and the evidence verifier had a false-SUPPORTED rate of Y% on 40 held-out cases."

You may not say:

> "State-of-the-art retrieval" or "hallucination-free" or "NPU-accelerated" without the corresponding artifact.
