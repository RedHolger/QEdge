# Frozen Stage 00 edge README excerpt

> **Not current.** This is the Stage 00 description of the demo. It was
> replaced during Stage 01. The original file was not copied before the
> rewrite; this excerpt preserves the claims that were visible in the Stage 00
> audit. Raw Stage 00 evidence is in `../handoff/00/`.

# Qualcomm Edge RAG Demo — On-Device, No Cloud, No GPU

## What This Is

An end-to-end on-device Retrieval-Augmented Generation (RAG) pipeline designed
for Qualcomm Snapdragon mobile and edge compute platforms. It integrates
hardware-aware KV-cache management (QUOKA policy), multimodal neural vector
retrieval (FAISS + BM25 + Reciprocal Rank Fusion on 500 catalog items), and
tamper-proof evidence verification (zero-container embedded SQLite store). The
entire pipeline executes fully locally on device without external network
dependencies, cloud APIs, or discrete server GPUs.

## What It Proved (Stage 00 claim — not accepted as current evidence)

This demo proves that low-latency, hallucination-resistant edge AI is achievable
within Snapdragon thermal and memory envelopes by mapping dense tensor
operations to the Hexagon NPU while executing deterministic logic on lightweight
CPU threads.

Stage 00 recorded claims that were later marked false or unverified:

- `NPU-offloadable: 100% of pipeline wall time`;
- `On Snapdragon 8 Elite (45 TOPS): estimated total < 3 ms`;
- retrieval always used FAISS + BM25 + RRF on ABO;
- the answer was hallucination-resistant;
- the evidence store was tamper-proof.

The Stage 00 handoff documents why these were not supported. See
`../handoff/00.md` and `../../docs/CLAIMS.md`.
