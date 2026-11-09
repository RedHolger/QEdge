# Stage 03 — Real KV Cache and Policies

## Goal

Implement a real KV cache for the model runner and measure FIFO, LRU, KeyDiff, and query-aware QUOKA at matched memory budgets. This is the core technical contribution.

## Read first

- `02-TECHNICAL-SPEC.md` section 2.3
- `03-DATA-AND-EVALUATION.md` section 5
- `handoff/02.md`

## Tasks

1. **Cache integration**
   - Implement a cache object that the model's attention actually uses.
   - If using `transformers`, subclass/adapt its cache API.
   - Support GQA shapes and per-layer caches.
   - Track position IDs correctly after eviction.
   - Never evict the current token.
   - Expose stats: resident tokens, capacity, hits, misses, evictions, peak bytes.

2. **Policies**
   - `full`: no eviction.
   - `fifo`: oldest resident token.
   - `lru`: least recently attended token.
   - `keydiff`: document the exact definition. At minimum compare key similarity and evict redundant tokens; do not only compare to the newest key unless you name it `newest-similarity`.
   - `quoka`: use real query/attention relevance. If attention scores are unavailable, log that and use a documented proxy; do not call it QUOKA unless the query signal is real.

3. **Parity tests**
   - At 100% budget, output must match full cache for the same prompt and seed.
   - Verify logits/perplexity parity within a stated tolerance.
   - Test eviction boundaries, empty cache, one-token cache, and max-length context.
   - Test GQA shapes.
   - Test position IDs after eviction.

4. **Pressure evaluation**
   - Budgets: 100%, 75%, 50%, 25%.
   - Tasks: at least one long-context retrieval task (passkey or needle), one QA task, and one summarization/instruction task.
   - Same prompts and seeds across policies.
   - Report quality, TTFT, decode tok/s, peak RSS, and evictions.
   - At full budget all policies must match the baseline; the interesting results are at reduced budgets.

5. **Policy claim discipline**
   - If QUOKA does not beat FIFO, report it.
   - Do not tune on the held-out prompts.
   - Do not compare policies at different cache budgets.
   - Do not use identical/random keys as the only evidence.

6. **Sanitizers**
   - Run the C++ path with ASan/UBSan if it is used.
   - Run Python with `PYTHONMALLOC=debug` or equivalent.
   - Fix out-of-bounds and NaN issues before reporting.

7. **Report**
   - `reports/03-cache.md`.
   - Table: budget × policy × quality × TTFT × tok/s × peak RSS × evictions.
   - CIs over prompts.
   - Failure examples.
   - A clear statement of which policy wins on which slice, or that none does.

## Acceptance

- full-budget parity passes;
- reduced-budget curves differ measurably;
- QUOKA uses a real query signal or is renamed;
- no OOB/NaN/sanitizer failure;
- cache stats reconcile;
- report has seeds, hardware, and CIs;
- old synthetic policy ranking is no longer the headline.

## Required handoff

`llm-inference/handoff/03.md`:

- implementation summary;
- policy definitions;
- parity evidence;
- pressure table;
- failure cases;
- test counts;
- "stopped here."

## Guardrails

- Do not add quantization yet.
- Do not change the model.
- Do not claim an NPU result.
- Do not hide a negative result.

## Stop

Stop after the handoff. Ask for review.
