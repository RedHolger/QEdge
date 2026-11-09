# Stage 04 — Quantization and Benchmark Harness

## Goal

Measure the accuracy/performance/memory tradeoff of FP16, GGUF Q8_0, and GGUF Q4_K_M on the same model and prompts, and replace ad-hoc timing with a reproducible benchmark harness.

## Read first

- `02-TECHNICAL-SPEC.md` section 2.4
- `03-DATA-AND-EVALUATION.md` section 5
- `handoff/03.md`

## Tasks

1. **Quantized baselines**
   - Use llama.cpp GGUF builds of SmolLM2-135M-Instruct:
     - Q8_0
     - Q4_K_M
   - Add a llama.cpp runner/backend if stage 02 did not already create one.
   - Record repository, revision, hash, and license.
   - Do not quantize by hand unless you have a reference check.
   - Validate that the GGUF tokenizer and chat template match the HF model.

2. **Benchmark harness**
   - `runtime/bench/`.
   - Warmups: 3.
   - Measured runs: at least 10.
   - Report median, p50, p95, min/max.
   - Fixed prompt set.
   - Fixed max new tokens.
   - Fixed seed for sampling; greedy for parity.
   - Same hardware within a table.

3. **Quality**
   - Perplexity on a fixed held-out text sample if the backend supports it.
   - A small deterministic task set (e.g., 20 instruction prompts with checkable answers).
   - At least one long-context task.
   - Report quality alongside performance; do not trade quality for speed silently.

4. **Memory**
   - Model file size.
   - Peak RSS during decode.
   - KV-cache bytes at each budget.
   - Report the measurement method.

5. **Energy**
   - If `powermetrics` or a device power API is available, record the command and trace.
   - Otherwise compute a modelled energy estimate and label it `modelled`.
   - Do not call a modelled number measured.

6. **Roofline sanity**
   - Estimate memory traffic per token for the model size and cache.
   - Compare measured tok/s to a bandwidth-bound estimate.
   - Explain any large gap.
   - Do not claim an NPU roofline without device counters.

7. **Report**
   - `reports/04-benchmark.md`.
   - Tables: precision × model size × peak RSS × TTFT × decode tok/s × quality.
   - Include the exact commands and artifact paths.

## Acceptance

- at least three precision variants measured;
- same prompts and hardware within each comparison;
- CIs and warmup reported;
- quality is measured, not assumed;
- energy label is correct;
- one command reproduces the table;
- old synthetic numbers are absent from the headline.

## Required handoff

`llm-inference/handoff/04.md`:

- quantization files and hashes;
- benchmark commands;
- tables;
- quality results;
- hardware;
- limitations;
- "stopped here."

## Guardrails

- Do not start QNN yet.
- Do not claim INT4 NPU performance.
- Do not compare a Mac CPU number to a published Snapdragon number.
- Do not omit a quality regression.

## Stop

Stop after the handoff. Ask for review.
