# Repair attention and rotary correctness

Prompt ID: `01-kernel-safety`  
Existing plan: Repair gate before original Stage 02/03  
Prerequisites: 00-rebaseline

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Read llm-inference/src/backend/cpu_backend.cpp, metal_backend.mm, backend interfaces, Tensor code, benchmark.cpp, and both C++ test executables. Fix G01–G03 and unsafe benchmark calls with the smallest compatible changes.

Document the existing 3D tensor layout as [heads, sequence, dimension]; do not call the head axis a batch axis. Validate dtype, rank, nonzero dimensions, Q/K head divisibility, matching K/V shapes, mask broadcast rules, and output shape in Release as well as Debug. Implement h_kv = h_q / (Hq/Hkv), per-head QK scaling, stable masked softmax, and per-head multiplication by V. Define behavior for fully masked rows. Never index a failed retrieval's placeholder as a valid multi-head tensor.

For RoPE, use checkpoint-specific base, split-half/interleaved convention, frequency exponent, and each token's original position; support non-contiguous positions explicitly where needed. Do not silently retrofit a different layout. For Metal attention, either implement and verify it or return a clear unsupported result / explicitly reported CPU fallback; never use raw QK matmul as attention.

Acceptance: independent scalar-reference numerical tests for MHA, MQA, GQA ratios 3:1, 4:1, 7:1; distinct per-head V values; q_len and kv_len greater than one; causal/no-mask/broadcast-mask/fully-masked cases; invalid shapes; and multi-position RoPE with bases 10000, 100000, 1000000. All tests must work with NDEBUG. Run ASan/UBSan CTest and both demo/benchmark entry points with no sanitizer errors. If disabling an unsupported Metal route, its failure must be intentional and documented. Preserve historical synthetic logs and do not claim real model throughput.
