# Stage 06 — Real Catalog and Multimodal Retrieval

## Goal

Make `edge-rag-demo` self-contained and real: a licensed 500-product catalog, provenance, text and image retrieval, a human-label annotation workflow, and a reproducible retrieval evaluation.

## Read first

- `02-TECHNICAL-SPEC.md` section 3
- `03-DATA-AND-EVALUATION.md` sections 1–3
- `handoff/01.md`
- `handoff/05.md`

## License gate — do this first

1. Identify the exact license for:
   - catalog metadata;
   - images;
   - derived embeddings.
2. If the license is unclear or non-commercial:
   - do not ship images;
   - use metadata-only or a clearly licensed substitute;
   - document the decision.
3. Write `catalog/LICENSE-DECISION.md`.
4. If the decision is unresolved, stop and ask. Do not download 500 images first.

## Tasks

1. **Catalog build**
   - Target 500 items; report unique families separately.
   - Required fields: item_id, variant_id, title, category, text, source_url, source_revision, license, attributes.
   - Normalize attributes with explicit units.
   - No invented units.
   - Deduplicate families.
   - Save `catalog/catalog.jsonl`, `manifest.json`, `provenance_report.md`, `duplicate_families.json`.

2. **Gallery**
   - `catalog/sample_gallery.html`.
   - Local images if licensed; placeholders otherwise.
   - Show title, attributes, and source.

3. **Embeddings and index**
   - Text embeddings: MiniLM or equivalent.
   - Image embeddings: CLIP or equivalent.
   - Build FAISS or a deterministic local index.
   - Build BM25.
   - Implement RRF.
   - Optional cross-encoder reranker.
   - Record model names, revisions, and hashes.

4. **Search**
   - `qedge find "text query" --k 5`.
   - `qedge find --image path.jpg --k 5`.
   - Return item, score, component scores, and image path.
   - Do not infer attributes from images.

5. **Annotation tool**
   - Local, keyboard-first.
   - Query + candidate pool.
   - Labels: relevant / partially relevant / not relevant.
   - Rationale, reviewer, timestamp.
   - Export JSONL.
   - Validation.

6. **Pilot evaluation**
   - 30 queries.
   - Candidate pool of at least 50 per query.
   - Double-annotate 20%.
   - Report agreement.
   - Compute Recall@5, NDCG@10, MRR@10 with paired bootstrap CIs.
   - Compare BM25, dense, hybrid, and rerank.

7. **Report**
   - `reports/06-retrieval.md`.
   - Per-slice results: text, image, attributes, zero-overlap, negation, unit conversion.
   - Failure examples.
   - No claim of "state of the art."

## Acceptance

- license decision recorded;
- 500 items or documented licensed subset;
- no external absolute paths;
- text and (if licensed) image queries work;
- 30-query human pilot exists;
- metrics with CIs;
- annotation tool works;
- index build is reproducible.

## Required handoff

`edge-rag-demo/handoff/06.md`:

- license decision;
- catalog counts and hashes;
- index/embedding models;
- retrieval commands;
- pilot metrics;
- annotation status;
- remaining labels;
- "stopped here."

## Guardrails

- Do not use synthetic qrels.
- Do not turn image similarity into dimension claims.
- Do not download 5,000 items.
- Do not claim multimodal retrieval unless the image path actually ran.

## Stop

Stop after the handoff. Ask for review.
