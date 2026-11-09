# Build provenance-governed retrieval evaluation

Prompt ID: `09-catalog-retrieval`  
Existing plan: Original Stage 06  
Prerequisites: 02-contracts and 04-assistant-correctness

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Implement the existing catalog/data plan with a documented licensed subset and explicit provenance. Target the existing 500-item plan or explain a smaller authorized subset. Record item, variant, family, source revision, original facts/units, missing/conflicting values, image rights, hashes, and ingestion transforms. Do not distribute assets without a supported license decision.

Keep BM25 as the baseline. Add a pinned real text embedding model and optionally a licensed image/text encoder behind explicit feature flags. Persist indexes keyed by catalog/model revisions and invalidate stale indexes. If neural assets are missing, report BM25-only; do not relabel character-trigram hashing as MiniLM, CLIP, or semantic inference. Implement deterministic weighted reciprocal-rank fusion with documented rank origin, weights and deduplication.

Separate retrieval relevance from factual verification. High image similarity cannot certify material or dimensions. An out-of-scope capability may still retrieve lexical neighbors; preserve its unverified requirement for the decision engine. Return modality/model provenance and measured stage timing.

Acceptance: 30-query human-reviewed pilot followed by the existing 120-query evaluation plan; family-aware split to avoid leakage; qrels/annotation provenance; Recall@k, nDCG/MRR where appropriate, latency, uncertainty, and BM25/text/image/fusion ablations. Thresholds use development data only. Supply an annotation workflow if human judgments are unavailable and mark the quality gate blocked. Fixture success alone is not catalog retrieval quality.
