# QEdge Demo and Interview Guide

## 1. The 30-second pitch

> "I built QEdge, a local-first product research assistant. It runs a real small language model on-device, uses a custom KV-cache policy to fit a fixed memory budget, and retrieves products by text or image. On top of that, it checks whether a product actually meets hard requirements — width, material, dimensions — and returns supported, contradicted, or unknown with citations. The point is not another chatbot; it is to measure the memory/quality tradeoff of on-device inference and then use that budget to make grounded decisions."

## 2. The 5-minute demo

### Minute 0:00 — Framing

"I'm going to show two things: a real on-device LLM runtime under a memory budget, and a product assistant that refuses to recommend something it cannot prove."

Show the hardware status:

```bash
qedge status
```

It should say exactly what is measured: Mac CPU/Metal, Snapdragon unverified.

### Minute 0:30 — Real generation

```bash
qedge run --model smollm2-135m --prompt "Explain why KV cache memory matters on a phone." --max-tokens 64
```

Show:

- real tokens;
- TTFT;
- decode tok/s;
- peak RSS;
- cache budget;
- model/quantization.

Say: "This is a real 135M model, not a benchmark of random tensors."

### Minute 1:15 — KV-cache tradeoff

Run the cache benchmark:

```bash
qedge bench cache --budget 1.0,0.5,0.25
```

Show a table:

| Budget | Policy | Quality | TTFT | tok/s | Peak RSS |
|---|---|---|---:|---:|---:|
| 100% | full | baseline | ... | ... | ... |
| 50% | FIFO | ... | ... | ... | ... |
| 50% | QUOKA | ... | ... | ... | ... |
| 25% | FIFO | ... | ... | ... | ... |
| 25% | QUOKA | ... | ... | ... | ... |

Say: "At full budget all policies should match the baseline. The interesting result is at 50% and 25%: which policy preserves task accuracy per byte?"

Do not present a number if the run has not happened. Use the actual report.

### Minute 2:30 — Find

```bash
qedge find "wooden dining chair under 60cm" --k 5
```

Show text results.

Then the image path:

```bash
qedge find --image demo/oak_chair.jpg --k 5
```

Say: "The visual encoder finds similar objects; it does not infer dimensions. That is the next stage's job."

### Minute 3:15 — Check

```bash
qedge check --item <item_id> --require "width<=60cm" --require "material=oak"
```

Show:

```
Decision: CONTRADICTED
Width: 48 cm (SUPPORTED, <= 60 cm)
Material: leather (CONTRADICTED, required oak)
Source: catalog.item_dimensions, revision ...
```

Then a case with missing data:

```
Decision: ABSTAIN
Width: UNKNOWN — source has no dimension value
```

Say: "This is the part that matters: an attractive image cannot prove a dimension. The system refuses to guess."

### Minute 4:00 — Grounding

Show the answer with citations:

> "This chair is 48 cm wide (supported) but the material is leather (contradicted). It does not meet your oak requirement."

Then show a bad candidate that says "matches" in the old version — and explain the fix.

### Minute 4:30 — Hardware status

Show `qnn/hardware_status.json`:

```json
{"snapdragon": {"available": false}, "qnn": {"device_verified": false}}
```

Say: "I did not claim NPU numbers. I built the ONNX/QNN path and the runbook. On a Snapdragon X Elite, the next step is to compile and measure; until then the status is unverified."

### Minute 5:00 — Close

"The result is a real on-device stack with a measurable memory/quality tradeoff and a product assistant that abstains instead of hallucinating. The next step is device measurement and a larger catalog."

## 3. The 15-minute deep dive

1. **Why KV cache matters** — decode is memory-bandwidth bound; cache grows with context; phones have fixed SRAM/DRAM.
2. **Policy design** — FIFO/LRU/KeyDiff/QUOKA; why query-aware eviction can preserve task-relevant tokens.
3. **Measurement** — matched budgets, same prompts, seeds, CIs, cache parity at 100%.
4. **Quantization** — FP16 vs Q8_0 vs Q4_K_M; quality/memory tradeoff.
5. **QNN path** — ONNX export, QNN EP, HTP config, ahead-of-time compile, device runbook.
6. **Retrieval** — BM25, dense, CLIP, RRF, reranker; human labels; zero-overlap failures.
7. **Evidence** — attribute claims, units, citations, abstention, false-SUPPORTED rate.
8. **Failure analysis** — what broke, what you would do next.

## 4. Likely interview questions

### "Why not just use llama.cpp?"

"llama.cpp is my real-model baseline and GGUF runtime. QEdge is the evaluation and policy layer around it: I need to change the KV-cache budget and measure quality, which is not a first-class knob in a standard chat demo. I also keep a small C++ kernel lab to understand the operations that QNN would fuse."

### "Why not just use a bigger model?"

"Because the target is on-device. A 135M model lets me iterate on memory, cache, and quantization on a laptop. The same harness scales to 0.5B/1B; the engineering question is the memory/quality curve, not the parameter count."

### "Is QUOKA actually better?"

"At 100% budget, all policies match the baseline. At 50% and 25%, I report the measured quality/latency/memory curve with confidence intervals. If QUOKA does not beat FIFO on my tasks, I say so. The contribution is the measurement, not a predetermined winner."

### "Did you run this on Hexagon?"

"Only if the device log exists. My hardware status file says `unverified` otherwise. I built the ONNX/QNN path and the runbook; I did not invent NPU numbers."

### "What is the hardest part?"

"Making the cache policy affect the real model without breaking position IDs, GQA shapes, or attention semantics — and then measuring quality in a way that is not just a synthetic benchmark."

### "How would you productionize it?"

"Pin the model and QNN SDK, run on a matrix of Snapdragon devices, collect thermal/power traces, add regression tests for cache parity, and gate releases on quality and latency SLOs."

### "What did you learn?"

"Memory is the product constraint. A 1% quality change can be worth a 2x memory reduction, but only if the evaluation is honest. And an evidence system must be allowed to say 'I don't know.'"

## 5. Resume bullets (only after the numbers exist)

- Built a local-first on-device LLM runtime for SmolLM2-135M with real tokenization, prefill/decode, and streaming; measured TTFT, decode throughput, and peak RSS.
- Implemented and evaluated four KV-cache eviction policies at 100/75/50/25% budgets; reported quality/latency/memory tradeoffs with confidence intervals.
- Added FP16/GGUF Q8_0/Q4_K_M quantization comparisons and an ONNX/QNN deployment path for Snapdragon X Elite.
- Built a provenance-backed 500-product catalog and multimodal retrieval stack (BM25 + CLIP/MiniLM + RRF + reranker) evaluated on human-reviewed queries.
- Built an evidence verifier with citations and abstention; measured false-SUPPORTED rate on 40 held-out decision cases.
- Packaged a one-command reproducible demo with explicit hardware status and no fabricated NPU claims.

Do not use a bullet until the corresponding artifact exists.

## 6. Claims to avoid

- "12,460 tok/s" from the old synthetic benchmark.
- "100% NPU-offloadable."
- "Hallucination-free."
- "Tamper-proof."
- "State-of-the-art retrieval."
- "40% memory reduction" without a measured table.
- "Runs on Snapdragon" without device logs.
- "Production-ready."

## 7. Demo data

Prepare 10 scenarios:

1. text query with exact attributes;
2. image query;
3. query with a unit conversion;
4. query with a contradiction;
5. query with a missing attribute;
6. query with a stale source;
7. query with a zero-overlap term;
8. query with a negation;
9. query with two products to compare;
10. a cache-budget demo.

Each scenario has an expected decision and a screenshot/JSON artifact.

## 8. What to show if the model is unavailable

The assistant must still work extractively:

- retrieve;
- verify;
- cite;
- abstain.

Say: "Generation is optional; truth is not. The extractive path is the fallback and the baseline for grounding."

## 9. What to show if Snapdragon is unavailable

Show:

- the ONNX export;
- the QNN EP adapter;
- the AI Hub submission script;
- the device runbook;
- `hardware_status.json` with `unverified`.

Say: "I know exactly what I would run on device and what I have not measured. I would rather show that than a fake NPU number."

## 10. Final interview line

> "The project is designed so that every claim has a command behind it, every dataset has provenance, and every hardware result has a device label. That is how I would build on-device AI at Qualcomm."
