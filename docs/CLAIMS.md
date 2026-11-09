# Current claims and evidence — 9 November 2026

This ledger supersedes the original Stage 01 workspace ledger, archived under
`handoff/research-review/12-release-review/superseded-docs/`. Stage 00/01 raw
history remains unchanged.

| Claim | Scope/status | Evidence |
|---|---|---|
| Installed offline assistant works without source siblings | Validated wheel-based CPU fixture path | [relocated installation](../handoff/research-review/12-release-review/logs/relocated-install.log) |
| Deterministic decision and citation handling | Internal regression coverage, not universal factual correctness | [installed evaluation](../handoff/research-review/12-release-review/evaluation.json) |
| Fixture timing | Measured, finite trials; no model or NPU inference | [raw timing](../handoff/research-review/12-release-review/benchmark.json) |
| Real model CPU generation exists | Pinned SmolLM2, separate runtime | [Stage 05](../handoff/research-review/05-real-model-baseline.md) |
| GGUF quantization measurements exist | Model/tool/context-specific CPU results | [Stage 08](../handoff/research-review/08-quantization-measurement.md) |
| ONNX CPU parity | Full-prefix no-cache graph, tested prompts/lengths only | [Stage 11](../handoff/research-review/11-qualcomm-deployment.md) |
| Kernel sanitizer checks pass | Current macOS build; historical baseline failures preserved | [Debug CTest](../handoff/research-review/12-release-review/logs/ctest-debug.log) |
| Catalog retrieval quality is established | BLOCKED; sparse judgments are diagnostics | [Stage 09](../handoff/research-review/09-catalog-retrieval.md) |
| Untouched held-out decision accuracy exists | PENDING; existing 80 cases are internal regressions | [Stage 10](../handoff/research-review/10-integration-evaluation.md) |
| QNN executes on HTP | UNVERIFIED/BLOCKED; C++ backend is a CPU stub | [Stage 11](../handoff/research-review/11-qualcomm-deployment.md) |
| Metal attention is GPU-native | False; explicit CPU fallback | Stage 01 kernel review |
| Snapdragon energy, latency, or VTCM residency measured | Not measured | No device run artifacts |
| Production-ready multi-user service | Not claimed | Local API boundary and remaining gates in [release guide](RELEASE.md) |

Performance claims must carry artifact/model revision, hardware, precision, input
and output lengths, trial counts and date. Synthetic kernel counters and analytical
memory estimates cannot be reported as model/NPU throughput or energy efficiency.
