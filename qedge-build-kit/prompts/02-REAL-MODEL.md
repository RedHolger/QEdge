# Stage 02 — Real Model Inference

## Goal

Make `llm-inference` run a real small language model end to end: tokenizer, prefill, decode, sampling, streaming, and measurement. Replace the synthetic benchmark as the headline path.

## Read first

- `02-TECHNICAL-SPEC.md` sections 2.1–2.2
- `03-DATA-AND-EVALUATION.md` section 5
- `handoff/01.md`

## Model choice and download bounds

Primary model:

- `HuggingFaceTB/SmolLM2-135M-Instruct`
- Apache-2.0.

Secondary (optional in this stage):

- `Qwen/Qwen2.5-0.5B-Instruct`
- Apache-2.0.

Rules:

- Download only the primary model in this stage.
- Record repository, revision, file list, SHA-256, and license.
- Do not commit weights.
- Do not download a 7B model.
- Keep the download under 1 GiB where possible.

## Tasks

1. **Model runner**
   - Add `runtime/model_runner.py`.
   - Load the tokenizer and model with `transformers`.
   - Use CPU or MPS; record which.
   - Apply the chat template for instruct prompts.
   - Support greedy and sampled generation.
   - Support streaming output.
   - Return a generation record (see `02-TECHNICAL-SPEC.md`).

2. **CLI**
   - `qedge run --model ... --prompt ... --max-tokens ... --temperature ...`.
   - Print the generated text and a machine-readable summary.
   - Add `--json` for the record.

3. **Reference parity**
   - Compare your runner against `model.generate()` on the same prompt, seed, and max tokens.
   - For greedy decoding, token IDs must match exactly.
   - For sampling, use the same seed and document any unavoidable differences.
   - Save the parity test and its output.

4. **Real metrics**
   - Measure TTFT.
   - Measure prefill tokens/s.
   - Measure decode tokens/s.
   - Measure peak RSS.
   - Record model size and quantization.
   - Run at least 3 warmups and 10 measured runs; report median and p95.

5. **Remove the synthetic path from the headline**
   - Keep the old benchmark only as `legacy/`.
   - The main `make bench` must use the real model.
   - The README must no longer headline synthetic tok/s.

6. **Tests**
   - tokenizer round-trip;
   - chat template renders;
   - generation record has required fields;
   - greedy parity;
   - max-token/stop behavior;
   - streaming yields the same tokens as non-streaming.

## Acceptance

- a real prompt produces coherent text;
- greedy output matches the reference token-for-token;
- TTFT/decode metrics are recorded with hardware metadata;
- no random tensors are used as model inputs;
- no NPU claim;
- one command reproduces the generation and metrics.

## Required handoff

`llm-inference/handoff/02.md`:

- model/revision/hash;
- commands;
- generation example;
- parity result;
- metrics table;
- test counts;
- known limitations;
- "stopped here."

## Guardrails

- Do not start KV-cache policy work yet.
- Do not tune quality with a larger model.
- Do not present a single run as a benchmark.
- Do not commit model weights or caches.

## Stop

Stop after the handoff. Ask for review.
