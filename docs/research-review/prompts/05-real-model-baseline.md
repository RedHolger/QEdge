# Build one real-model reference runtime

Prompt ID: `05-real-model-baseline`  
Existing plan: Original Stage 02  
Prerequisites: 01-kernel-safety and 02-contracts

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

Create a separately packaged runtime under llm-inference/runtime following the existing Stage 02 layout. Use the selected supported Hugging Face model implementation first. Start with HuggingFaceTB/SmolLM2-135M-Instruct; add Qwen/Qwen2.5-0.5B-Instruct only after the first baseline passes. Load architecture, tokenizer, rotary settings, and EOS rules from pinned checkpoint revisions instead of TinyLlama defaults.

Pin a compatible Python/PyTorch/Transformers stack and record licenses, artifact hashes, device, actual dtype, tokenizer revision, prompt template, and seed. Add an explicit acquisition command with estimated size and an offline local-path mode. Never download models during unit tests or on an ordinary import. If assets are unavailable, complete dependency-free work and record a blocked integration result rather than returning canned text.

Implement real prefill and autoregressive decode with full cache and well-defined output limits, masks, positions, EOS and padding handling. Reject unsupported settings. Store generated token IDs separately from validated benchmark metadata. Use upstream unmodified generation as the reference with identical input IDs and generation configuration.

Acceptance: deterministic greedy parity and numerical logits checks on a small prompt suite with the same backend/dtype; correct zero/one-token and early-EOS behavior; repeated independent requests do not leak state; cached assets run offline; stdout JSON is valid and logs use stderr. Timing has defined boundaries and accelerator synchronization when used. Report model load, tokenization, prefill, TTFT, and decode separately. Do not add cache compression or QNN to rescue an incorrect baseline.
