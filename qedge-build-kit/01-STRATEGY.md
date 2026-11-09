# QEdge Strategy — What We Are Building and Why Qualcomm Should Care

## 1. What to copy from the Pinterest/DiscoveryEval/EvidenceRAG process

The transcript describes a strong agentic build protocol:

- audit before code;
- freeze historical results;
- stage-by-stage prompts with acceptance criteria;
- a written handoff after every stage;
- independent review loops that caught real defects;
- explicit separation of passed, failed, and skipped tests;
- human judgment where human judgment is required;
- "no fabricated improvement" as a hard rule.

That protocol is worth reusing. It produced real fixes: authorization checks, connection pooling, version races, catalog integrity, and missing artifacts.

But the transcript also exposes four failure modes to avoid:

1. **Green tests were treated as acceptance.** Several stages passed their own tests and were later found to have defects. Acceptance must be behavior-based and artifact-based, not test-count-based.
2. **The review loop became the project.** Repeated "accepted / needs repair" cycles consumed attention. Our plan uses fewer stages and a single review checklist per stage.
3. **The product thesis drifted toward generic RAG.** Retrieval quality is important, but a Qualcomm role will not be won by another product-search demo. The differentiator must be on-device inference efficiency.
4. **Claims outran evidence.** The fix is a claims ledger and hardware gating, not more prose.

## 2. What we are deliberately not copying from Pinterest

Pinterest's product problem is visual discovery and shopping at internet scale. Qualcomm's problem is making AI run well on a device with fixed memory, power, and thermal limits.

We keep the **workload** — product search and evidence-backed comparison — because it is concrete and demoable. We do not keep the **thesis**. A Qualcomm interviewer will not care that the UI resembles Pinterest; they will care that the model runs, the cache fits, the quantization tradeoff is measured, and the deployment path is real.

That is why the runtime stages come first and the assistant is built on top of a measured budget, not the other way around.

## 3. What Qualcomm is actually working on

Qualcomm's public AI story is not generic cloud RAG. It is:

- **Snapdragon X Elite / X2 Elite** laptops and **Snapdragon 8 Elite** mobile platforms;
- **Hexagon NPU** with INT8/INT16/FP16 support and a growing agentic-AI architecture;
- **Qualcomm AI Hub** for model compilation, profiling, and device deployment;
- **QNN / QAIRT (Qualcomm AI Engine Direct)** for ahead-of-time compiled graphs and HTP execution;
- **Genie / GenieX** for on-device generative AI orchestration;
- the hard problems that come with all of that: quantization, KV-cache memory, bandwidth, thermal limits, power, operator fusion, and model partitioning.

Public references worth reading before stage 02:

- Qualcomm AI Hub: <https://aihub.qualcomm.com/>
- AI Hub Workbench docs: <https://workbench.aihub.qualcomm.com/docs/hub/>
- Hexagon NPU: <https://www.qualcomm.com/processors/hexagon>
- Hexagon agentic-AI architecture: <https://www.qualcomm.com/news/onq/2026/09/hexagon-npu-agentic-ai-architecture>
- AI Hub models repository: <https://github.com/qualcomm/ai-hub-models>
- On-device RAG on Snapdragon X Elite: <https://arxiv.org/html/2606.11257v1>

The interview story is therefore not "I built a RAG app." It is:

> "I built a small on-device LLM runtime and used it to study the memory/quality tradeoff of KV-cache eviction policies, then wrapped it in a product-research assistant that can prove when a product does or does not meet a requirement."

That is a Qualcomm-shaped story: memory-bound inference, quantization, KV cache, NPU deployment, and an application that justifies the hardware.

## 4. Product thesis

**QEdge** is a local-first product research stack:

1. Find products that match a visual or textual taste.
2. Check whether a specific product satisfies hard requirements using source evidence.
3. Refuse to answer when the evidence is insufficient.
4. Do all of it inside a memory and latency budget that could plausibly run on a Snapdragon device.

The two projects are two layers of the same product:

- **QEdge Runtime** supplies real generation and the memory/latency model.
- **QEdge Assist** supplies retrieval, evidence, and a reason for the runtime to exist.

The demo should feel like a shopping assistant, but the engineering evidence is about on-device inference and grounded decisions.

## 5. Current-state gap analysis

| Area | Current state | Target state |
|---|---|---|
| Model inference | No model loaded; synthetic tensors | Real SmolLM2/Qwen generation with tokenizer and sampling |
| Throughput claim | 12k tok/s from cache bookkeeping | Measured TTFT and decode tok/s on a named model and hardware |
| KV cache | Policies on random, identical tensors; no evictions in headline run | Real cache policies under matched budgets; quality measured |
| QUOKA | Mean-key proxy, no query | Query-aware eviction using real attention scores |
| Quantization | None | FP16 vs GGUF Q8_0/Q4_K_M; memory and quality table |
| NPU | CPU stub; modelled 45 TOPS | ONNX/QNN path; device runbook; measured or `unverified` |
| Power | Modelled pJ counters; unstable tok/mJ | Honest energy proxy; no NPU energy claim without device telemetry |
| Retrieval | Wrapper around external repos; absolute paths | Self-contained catalog, embeddings, FAISS, BM25, RRF, reranker |
| Evidence | Regex over self-generated chunks | Attribute-level claims, citations, source updates, abstention |
| Answer | Template always says "matches" | Correct decision; refuse when contradicted/unknown |
| Metrics | Hardcoded NPU labels and "<3 ms" | Measured stage timings and a claims ledger |
| Tests | Green but tautological | Behavior tests, negative tests, hardware-gated skips |
| Packaging | Absolute paths, symlinked venv, build artifacts | One-command, clean checkout, pinned dependencies |

## 6. Scope tiers

### Tier 1 — Portfolio (must finish)

- Real SmolLM2-135M-Instruct inference.
- Real KV cache with 4 policies at 100%/50%/25% budgets.
- One quantization comparison (FP16 vs Q8_0 vs Q4_K_M).
- Reproducible benchmark JSON + report.
- Self-contained 500-product catalog with provenance.
- Text retrieval with BM25 + embeddings + RRF.
- Evidence checks with `SUPPORTED`/`CONTRADICTED`/`UNKNOWN`.
- 10 scripted demo scenarios.
- Honest hardware status.

### Tier 2 — Strong (should finish)

- Image query with CLIP.
- Cross-encoder reranker.
- 120 human-reviewed queries.
- 80 evidence decision cases.
- ONNX export and QNN EP integration.
- One measured Snapdragon run (device or AI Hub).
- 5,000-product catalog if disk/RAM allow.

### Tier 3 — Stretch (only after Tier 1 and 2)

- Custom C++ transformer forward pass with real weights.
- INT4 group-wise quantization.
- Paged attention / block allocator.
- Speculative decoding.
- Multi-user authorization and revocation.
- On-device thermal/power measurement.

## 7. Non-goals

- Training a foundation model.
- Beating a 7B model with a 135M model.
- A production multi-tenant SaaS.
- A mobile app.
- Paid inference APIs.
- Claiming Hexagon performance without Hexagon logs.
- Replacing llama.cpp or QNN with a from-scratch implementation unless the custom path is the point of the experiment.

## 8. Success metrics

### Runtime

- Greedy output matches the reference implementation for the same prompt and seed.
- Full-cache logits/perplexity match the reference within a stated tolerance.
- At 100% budget, custom cache output matches full cache.
- At 50% and 25%, policies produce different quality/latency curves.
- No out-of-bounds or sanitizer failures.
- Peak memory and model size are measured, not estimated.

### Assistant

- Retrieval: Recall@5, NDCG@10, MRR@10 on human-reviewed queries.
- Evidence: decision accuracy, citation validity, false-`SUPPORTED` rate.
- Abstention: precision/recall on insufficient-evidence cases.
- End-to-end: task success, unsupported-claim rate, p50/p95 latency, peak memory.
- No answer says "matches" when a required attribute is `CONTRADICTED` or `UNKNOWN`.

### Portfolio

- A fresh clone runs `make demo` and `make bench`.
- Every README number links to an artifact.
- The claims ledger has no unverified performance claim.
- The demo can be given in 5 minutes without internet.

## 9. Time budget

Assuming part-time work with OpenCode:

| Stage | Estimate |
|---|---|
| 00 Audit | 2–4 hours |
| 01 Foundations | 1–2 days |
| 02 Real model | 2–4 days |
| 03 KV cache | 3–5 days |
| 04 Quantization + bench | 2–3 days |
| 05 QNN/AI Hub | 3–5 days |
| 06 Catalog + retrieval | 3–5 days + labeling |
| 07 Evidence + abstention | 3–4 days |
| 08 Integration | 3–4 days |
| 09 Evaluation | 2–3 days |
| 10 Release | 2 days |
| 11 Stretch | optional |

If a stage exceeds twice its estimate, cut scope in the prompt rather than lowering the acceptance bar. The most common cause of overrun will be model export/QNN and human labeling, not the demo UI.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Model download/export is harder than expected | Start with SmolLM2-135M; use `transformers` and GGUF before custom C++. |
| No Snapdragon hardware | Deliver QNN integration + runbook; mark performance unverified. |
| ABO license is unclear | Metadata-only catalog or licensed substitute; no image claims until cleared. |
| Human labels are slow | 30-query pilot first; start early; do not fake qrels. |
| Benchmark variance on a laptop | Warmup, repeated runs, medians, CIs, hardware metadata. |
| Scope creep into a full app | Tier 1 is a CLI; UI is optional. |
| Claims drift | Claims ledger + review protocol. |
| OpenCode overfits to tests | Acceptance requires artifacts and negative cases. |

## 11. The one-sentence pitch

> "QEdge is a local-first product research assistant built on a real on-device LLM runtime: it finds products by text or image, checks hard requirements against source evidence, abstains when evidence is missing, and quantifies the memory/quality tradeoff of KV-cache eviction on a Snapdragon-class budget."
