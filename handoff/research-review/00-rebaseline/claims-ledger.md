# Research review baseline claims ledger

29 October 2026. This additive ledger qualifies current evidence; `docs/CLAIMS.md` and historical ledgers remain unchanged. Test outcomes are measured local correctness checks, not NPU or model performance measurements.

| Claim | Classification | Evidence and limitation |
| --- | --- | --- |
| C++ Release CTest passes | tested | finalization/release-ctest.log: 2/2; does not establish memory safety. |
| C++ sanitizer gate passes | false; repair required | finalization/asan-ctest.log: 0/2, both heap-buffer-overflow. Both entry points abort too. |
| Assistant unit tests pass | tested | finalization/edge-tests.log: 18 passed, 1 explicit licensed-catalog skip. |
| Shared schema tests pass | tested | finalization/schema-tests.log: 13/13. Structural examples validate 7/7. |
| Tests prove all decision/authorization invariants | false | finalization/gap-probe.log and extra-probe.log reproduce logical and ownership defects. |
| Real model generation is implemented | false / proposed | runtime contains only README; model-dependent gate blocked pending implementation/assets. |
| Local BM25 and SQLite fixture plumbing exist | implemented and tested | Source and fixture tests; no retrieval-quality claim. |
| Human-labelled retrieval/decision quality exists | blocked | finalization/edge-eval.log explicitly returns not-run. |
| Synthetic C++ tok/s is model throughput | false | Benchmark uses generated tensors and bookkeeping. |
| Stub energy is measured device energy | false / synthetic-modelled | QNN CPU stub and power report sources. |
| Snapdragon/NPU throughput, energy or offload measured | unverified / blocked | No authorized device experiment; local CPU stub is not NPU execution. |
| Memory formula proves VTCM residency | false | Byte arithmetic is calculated; placement requires backend/device evidence. |
| Research replacement code and result tables are validated | proposed / unverified | No matching execution artifacts; do not import as code or measured results. |
| External models/catalogs are integrated or license-approved | false | Inventory only; no integration/approval established by file presence or hash equality. |
| Historical evidence is preserved | verified | 30/30 historical hashes; preservation.json records protected files and unchanged Git status. |
| The baseline independently approves itself | false | Completion self-assessment only; reviewer acceptance remains pending. |

Evidence logs are under `logs/`, with the final checks under `logs/finalization/`. Existing Stage 01 statements that CTest passes must be interpreted as Release-only evidence; sanitizer failures override any implication of memory safety. No performance number has been promoted to a new headline claim.
