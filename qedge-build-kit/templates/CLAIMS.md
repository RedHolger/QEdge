# Claims Ledger

Every headline number must appear here. No `measured` row without an artifact path.

| Claim | Value | Evidence artifact | Hardware | Status | Notes |
|---|---|---:|---|---|---|
| Real model generation | — | | | | |
| Greedy parity | — | | | | |
| TTFT | — | | | | |
| Decode tok/s | — | | | | |
| Peak RSS | — | | | | |
| Cache budget quality curve | — | | | | |
| Quantization tradeoff | — | | | | |
| Retrieval Recall@5 | — | | | | |
| Retrieval NDCG@10 | — | | | | |
| False SUPPORTED rate | — | | | | |
| Abstention precision | — | | | | |
| End-to-end latency | — | | | | |
| Snapdragon NPU performance | — | | Snapdragon | unverified | |

Allowed statuses:

- `measured` — produced by a command on named hardware; artifact exists.
- `modelled` — derived from a formula; label the formula.
- `projected` — extrapolated; label the assumption.
- `unverified` — no evidence.
- `false` — contradicted by evidence.

Rules:

- A Mac number cannot be `measured` for Snapdragon.
- A single run is not a benchmark.
- A synthetic benchmark cannot support a model-quality claim.
- A README number must link to the artifact.
