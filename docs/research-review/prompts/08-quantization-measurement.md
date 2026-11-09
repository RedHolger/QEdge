# Build honest quantization and benchmark artifacts

Prompt ID: `08-quantization-measurement`  
Existing plan: Original Stage 04  
Prerequisites: 05-real-model-baseline; 06/07 for cache combinations

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Build a reproducible benchmark harness with fixed prompt/evaluation manifests, fresh-run memory accounting, declared warmup, repeated trials, synchronization, p50/p95, uncertainty, and full provenance. Define TTFT scope and decode denominator; an immediate EOS gives zero decode steps, not a forced positive count. Track actual weight, activation, and KV precision separately.

Measure a supported full-precision reference and real quantized artifacts. For the build kit's Q8_0/Q4_K_M study, use a pinned GGUF conversion/runtime path and a same-engine full-precision comparison where possible; isolate backend changes from quantization effects. Treat AIMET-to-QNN as a separate route with verified model/operator compatibility. If unavailable locally, provide a pinned runnable environment and mark execution blocked.

For AIMET use supported APIs, disjoint calibration/evaluation data, stored encodings, actual exported artifacts and post-export validation. Do not call a handwritten float dequantization helper production INT4 or claim CLE without graph-valid transformations. Weight quantization does not establish quantized KV storage; implement/test KV quantization separately before reporting its memory savings.

Acceptance: at least three supported precision variants for the chosen completed route, measured artifact sizes, quality degradation and latency/RSS with comparable settings, raw outputs and one table-generation command. Unsupported configurations remain explicit blockers. Do not call calculated energy measured; collect power using a documented instrument and time-integrated interval, or omit energy. Count selection overhead and temporary allocations. Record out-of-memory and failures instead of dropping them.
