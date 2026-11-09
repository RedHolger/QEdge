# QEdge Build Kit — Start Here

## What this is

A staged build plan for turning the two projects in `/Users/manuvashistha/Developer/qualcomm` into a credible, demonstrable Qualcomm portfolio:

- **QEdge Runtime** — from `llm-inference`
- **QEdge Assist** — from `edge-rag-demo`

The kit is written for OpenCode. Each numbered prompt is one work session. Each session ends with a written handoff and a hard stop. Do not skip stages, and do not proceed until the previous handoff has been reviewed.

## Why this exists

The current projects run, but they are demo-shaped:

- the "LLM engine" does not run a real LLM;
- the KV-cache benchmark is synthetic and cannot distinguish policies;
- the "NPU" and "tok/mJ" numbers are modelled or scraped, not measured;
- the RAG demo depends on absolute paths outside the project;
- its generator says "matches your requirements" even when evidence contradicts the request;
- its "100% NPU-offloadable" and "< 3 ms" claims are hardcoded labels.

Those are fatal in a Qualcomm interview because the interviewer can ask one question — "show me the model generating a token" — and the demo stops being evidence.

The target is not a bigger demo. It is a smaller, honest system with real model inference, real memory pressure, real retrieval, real evidence, and a hardware path that says exactly what has and has not been verified.

## The two deliverables

### QEdge Runtime (from `llm-inference`)

A reproducible on-device LLM runtime and KV-cache research harness:

- real SmolLM2-135M-Instruct (and optionally Qwen2.5-0.5B-Instruct) inference;
- real tokenizer, prefill, decode, sampling, and streaming;
- a custom KV cache with FIFO, LRU, KeyDiff, and query-aware QUOKA;
- matched-budget evaluation: 100%, 75%, 50%, 25% cache;
- FP16 and GGUF Q8_0/Q4_K_M comparisons;
- TTFT, prefill/decode throughput, peak memory, and an honest energy proxy;
- an ONNX → QNN/QAIRT deployment path for Snapdragon X Elite / 8 Elite;
- a `hardware_status.json` that distinguishes measured, modelled, and unverified.

What you can show: a real prompt produces real tokens; shrinking the cache changes quality in a measured way; quantization changes memory and speed; the QNN adapter is either measured on a device or explicitly marked unverified.

### QEdge Assist (from `edge-rag-demo`)

A self-contained on-device product research assistant:

- a real, licensed home-furnishings catalog with provenance;
- text and image search over CLIP/MiniLM embeddings plus BM25 and RRF;
- a reranker and human-reviewed retrieval evaluation;
- evidence-backed requirement checks with citations;
- `SUPPORTED`, `CONTRADICTED`, and `UNKNOWN` outcomes;
- explicit abstention when the evidence is insufficient;
- optional local LLM answer generation through QEdge Runtime;
- latency and memory accounting per stage, with an NPU-offload map.

What you can show: "find chairs that look like this" followed by "is this one under 60 cm and oak?" and a correct answer that cites the source — or refuses to answer.

## Non-negotiable rules

1. **No fabricated numbers.** Every number in a README, report, or demo must have a command, artifact, and hardware label behind it.
2. **Hardware gating.** A Mac CPU/Metal result is not an NPU result. If no Snapdragon device or AI Hub run exists, write `unverified` and stop.
3. **Real data only.** No random tensors as model inputs, no synthetic qrels presented as ground truth, no self-generated "evidence" presented as independent facts.
4. **Preserve history.** Do not edit old benchmark results. Put corrected results beside them and explain the difference.
5. **Human labels are human.** The annotation tool can prepare candidates; it cannot invent judgments. A 30-query pilot comes before 120 queries.
6. **One stage per session.** Each prompt ends with `handoff/NN.md`. Stop after it.
7. **A passing test is not an accepted stage.** Acceptance requires the behavior in the prompt, not just a green suite.
8. **No paid services.** Local-first. Optional free AI Hub access is allowed only if it is genuinely available and the terms are clear.

## Stage map

| Stage | Prompt | Primary output | Gate |
|---|---|---|---|
| 00 | `prompts/00-AUDIT.md` | Baseline audit + claims ledger | No source changes; exact logs |
| 01 | `prompts/01-FOUNDATIONS.md` | Portable repos, shared contracts, clean commands | Fresh checkout works; no absolute paths |
| 02 | `prompts/02-REAL-MODEL.md` | Real SmolLM2/Qwen inference | Coherent generation; parity check |
| 03 | `prompts/03-KV-CACHE.md` | Real cache policies under pressure | Policies differ; no OOB; quality table |
| 04 | `prompts/04-QUANTIZATION-BENCH.md` | Quantization + benchmark harness | Reproducible numbers; hardware labels |
| 05 | `prompts/05-QNN-DEPLOY.md` | ONNX/QNN/AI Hub deployment path | Builds; device runbook; honest status |
| 06 | `prompts/06-CATALOG-RETRIEVAL.md` | Real catalog + multimodal retrieval | License verified; 500 items; labels |
| 07 | `prompts/07-EVIDENCE-ABSTENTION.md` | Evidence verifier + abstention | 80 decision cases; no false matches |
| 08 | `prompts/08-INTEGRATION.md` | End-to-end assistant | One command; 10 scenarios; fallbacks |
| 09 | `prompts/09-EVALUATION.md` | Evaluation report | CIs; ablations; claims ledger |
| 10 | `prompts/10-RELEASE-INTERVIEW.md` | Release + interview package | Fresh machine; demo video |
| 11 | `prompts/11-HARDWARE-STRETCH.md` | Optional device/5k/advanced work | Measured device logs or stop |

Human labeling is the long pole. You may start the license decision, catalog preparation, and annotation-tool work in parallel once stage 02 is accepted, but the formal stage 06 handoff still follows stage 05. Keep the handoff chain sequential; do not let parallel work bypass a gate.

## How to use a stage

Paste this pattern into OpenCode, changing the stage number:

```text
Read the QEdge build kit at:
/Users/manuvashistha/Developer/qualcomm/qedge-build-kit

Read:
- 00-START-HERE.md
- 01-STRATEGY.md
- 02-TECHNICAL-SPEC.md
- 03-DATA-AND-EVALUATION.md
- prompts/NN-<stage>.md

Execute prompts/NN-<stage>.md only.

Preserve existing changes and historical results. Do not start the
next stage. Write handoff/NN.md with exact commands, test counts,
artifact paths, and remaining issues. Stop after the handoff.
```

After the session, run the review prompt in `prompts/REVIEW-TEMPLATE.md`. Do not accept the stage because tests pass; accept it because the artifacts prove the required behavior.

## Human inputs you must provide

These cannot be generated by OpenCode:

- **ABO license decision.** Confirm whether the catalog and images are CC BY 4.0, CC BY-NC, or another license. If unclear, do not ship images; use metadata-only or a clearly licensed substitute.
- **Model download approval.** SmolLM2-135M and Qwen2.5-0.5B are small, but record model cards, licenses, checksums, and disk usage.
- **Human relevance judgments.** 30-query pilot first; 120 reviewed queries for the portfolio tier.
- **Decision cases.** 80 evidence cases: 40 dev, 40 test, with `SUPPORTED`, `CONTRADICTED`, and `UNKNOWN` expected outcomes.
- **Curated demo scenarios.** 10 queries, including one image query, one contradiction, one unknown, and one abstention.
- **Hardware decision.** Snapdragon X Elite laptop, Snapdragon 8 Elite phone, AI Hub Workbench, or "Mac only, QNN unverified."

## Known blockers and how the kit handles them

- **No Snapdragon hardware:** stage 05 still produces a real ONNX/QNN integration, compile scripts, and a runbook. It records `unverified` and does not claim NPU performance.
- **ABO license ambiguity:** stage 06 has a hard license gate. A metadata-only catalog is acceptable for retrieval; image claims require cleared images.
- **Human labeling time:** stage 06 builds the annotation tool and runs the 30-query pilot. Stage 09 cannot claim retrieval quality without reviewed labels.
- **Disk/RAM:** target 500 products and a 135M model. Do not start the 5,000-product or 1B+ model tier until stage 11.
- **OpenCode context:** each stage is self-contained and references the specs; do not paste all stages into one session.

## Definition of a portfolio-ready result

At the end, a reviewer can:

1. Clone the repo and run one command.
2. See a real model generate text on a real prompt.
3. Change the KV-cache budget and see a measured quality/latency tradeoff.
4. Run a text or image search over a licensed catalog.
5. Ask a requirements question and get `SUPPORTED`, `CONTRADICTED`, or `UNKNOWN` with citations.
6. See exactly which stages ran on CPU, Metal, or Snapdragon, and which are unverified.
7. Read a report where every claim links to an artifact.

That is the bar. Not "it compiles." Not "the test suite is green." Not "the README says 12,000 tok/s."
