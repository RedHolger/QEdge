# Qualcomm Alignment Guide

## 1. Why this portfolio maps to Qualcomm

Qualcomm's AI work is fundamentally about making models run well on constrained devices:

- Snapdragon X Elite / X2 Elite laptops and Snapdragon 8 Elite phones;
- Hexagon NPU with FP16/INT8/INT16 support;
- AI Hub for model compilation, profiling, and deployment;
- QNN/QAIRT for ahead-of-time compiled graphs and HTP execution;
- Genie/GenieX for on-device generative AI;
- the system problems: quantization, KV-cache memory, bandwidth, thermal limits, power, and operator placement.

QEdge is deliberately built around those problems.

## 2. Feature-to-technology map

| QEdge feature | Qualcomm technology | What it demonstrates |
|---|---|---|
| Real SmolLM2/Qwen inference | Snapdragon CPU/NPU, AI Hub | Model bring-up and evaluation |
| KV-cache policies | Hexagon SRAM/DRAM constraints | Memory-aware runtime design |
| QUOKA query-aware eviction | Attention/LLM runtime | Algorithm + systems tradeoff |
| FP16/INT8/INT4 comparison | QNN quantization, AIMET concepts | Accuracy/performance tradeoff |
| ONNX export | AI Hub / QNN toolchain | Deployment workflow |
| QNN EP integration | QAIRT / QNN | Runtime integration |
| Device runbook | Snapdragon X Elite/8 Elite | Hardware bring-up discipline |
| Per-stage latency | QNN profiler / AI Hub | Performance analysis |
| Energy proxy | Thermal/power constraints | Honest measurement |
| Multimodal retrieval | Hexagon vision/transformer acceleration | On-device multimodal |
| Evidence/abstention | On-device assistants | Trustworthy product behavior |

## 3. Role-specific framing

### ML runtime / inference engineer

Emphasize:

- KV-cache design and memory accounting;
- quantization and its effect on quality;
- tokenizer/prefill/decode/sampling;
- latency breakdown and roofline reasoning;
- QNN/ONNX integration.

Artifacts to show:

- `runtime/`, `benchmarks/03`, `benchmarks/04`, `qnn/`.

### Model optimization / quantization engineer

Emphasize:

- FP16 vs Q8_0 vs Q4_K_M;
- accuracy/memory/throughput tradeoff;
- per-layer sensitivity (optional);
- what would change for INT4 on HTP;
- calibration and evaluation protocol.

Artifacts:

- `reports/04-benchmark.md`, `reports/09-final-evaluation.md`.

### AI Hub / deployment engineer

Emphasize:

- ONNX export with past key/values;
- QNN EP configuration;
- ahead-of-time compile vs on-device compile;
- device matrix and reproducibility;
- hardware status and failure handling.

Artifacts:

- `qnn/`, `qnn/device_runbook.md`, device logs.

### Applied ML / search / recommendations

Emphasize:

- multimodal retrieval;
- human-labeled evaluation;
- RRF and reranking;
- zero-overlap failures;
- evidence-backed decisions.

Artifacts:

- `edge-rag-demo/catalog/`, `eval/`, `reports/06-retrieval.md`.

### Product / applied AI

Emphasize:

- the user problem: find + verify;
- abstention as a product requirement;
- latency/memory budget;
- why on-device matters (privacy, offline, cost);
- what would be needed for production.

Artifacts:

- `demo/`, `05-DEMO-AND-INTERVIEW.md`.

## 4. What Qualcomm interviewers will probe

- "Show me the model generating a token." → stage 02.
- "How do you know the cache is correct?" → stage 03 parity tests.
- "What happens at 25% cache?" → stage 03 quality curve.
- "Why does quantization help?" → stage 04 memory/throughput table.
- "Did you run it on Hexagon?" → stage 05 hardware status.
- "How do you know retrieval is good?" → stage 06 human labels.
- "How do you prevent hallucination?" → stage 07 abstention and false-SUPPORTED rate.
- "What would you do next?" → stage 11, with device measurement and larger catalog.

## 5. Hardware options

### Option A — Mac only (default)

- CPU and Metal.
- QNN path built but `unverified`.
- Honest and still useful: the runtime, cache, quantization, retrieval, and evidence work are real.

### Option B — Snapdragon X Elite laptop

- Windows on Arm or Linux.
- Install QNN SDK / ONNX Runtime QNN EP.
- Run the exported model on HTP.
- Record device logs, driver version, OS, and thermal state.
- This is the strongest option for a Qualcomm application.

### Option C — Snapdragon 8 Elite phone

- Android; more deployment work.
- Use AI Hub or a prebuilt runtime.
- Good for mobile constraints, but higher setup cost.

### Option D — AI Hub Workbench

- Cloud device access for compilation/profiling.
- Check account availability and terms; do not assume free access.
- Record job IDs and device logs.
- Do not present AI Hub numbers as local measurements.

## 6. Reading list

- Qualcomm AI Hub: <https://aihub.qualcomm.com/>
- AI Hub Workbench docs: <https://workbench.aihub.qualcomm.com/docs/hub/>
- AI Hub models: <https://github.com/qualcomm/ai-hub-models>
- Hexagon NPU: <https://www.qualcomm.com/processors/hexagon>
- Hexagon agentic AI: <https://www.qualcomm.com/news/onq/2026/09/hexagon-npu-agentic-ai-architecture>
- On-device RAG on Snapdragon X Elite: <https://arxiv.org/html/2606.11257v1>
- ONNX Runtime QNN EP: <https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>
- GenieX / on-device Gen AI: search Qualcomm developer blog for GenieX.

## 7. What not to say

- "I built a full LLM inference engine" unless stage 02/03 prove it.
- "It runs on the NPU" unless stage 05 has device logs.
- "12k tok/s" from the old benchmark.
- "Hallucination-free" or "tamper-proof."
- "Production-ready."
- "State-of-the-art."

## 8. The Qualcomm story in one paragraph

> "I wanted to understand the real constraint in on-device AI: memory. So I built a small LLM runtime, implemented KV-cache eviction policies, and measured the quality/memory tradeoff at matched budgets. Then I built a product assistant on top of it that retrieves by text or image, verifies requirements against source evidence, and abstains when the evidence is missing. I kept the NPU path honest: I built the ONNX/QNN integration and a device runbook, and I only claim device performance when I have device logs."
