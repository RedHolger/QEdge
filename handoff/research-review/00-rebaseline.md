# Research review 00 rebaseline handoff

Date: 29 October 2026. Scope: execute 00-rebaseline only. Application code, existing OpenCode edits, nested Git state, `docs/CLAIMS.md`, and historical evidence were preserved.

## Verdict

**Baseline evidence: COMPLETE; self-assessment ACCEPTED for the bounded evidence-collection task. Application: REPAIR REQUIRED. Independent/reviewer acceptance: PENDING.** This is not permission to start stage 01. Release CTest success does not satisfy the sanitizer gate. No implementation repair was made.

## Reproduced results

The saved work-state logs were inspected and the principal commands rerun during finalization, with explicit process exits and fresh logs. Debug configuration/build and Release configuration/build succeeded. Both sanitizer CTest executables and both sanitizer entry points reproduce heap-buffer-overflow. The demo's failure must not be assumed to have exactly the same indexing cause as the benchmark without inspecting its stack/mask path during repair.

| Command | Exit | Passed | Failed | Skipped | Status | Evidence |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| edge-tests | 0 | 18 | 0 | 1 | PASS | [log](00-rebaseline/logs/finalization/edge-tests.log) |
| schema-tests | 0 | 13 | 0 | 0 | PASS | [log](00-rebaseline/logs/finalization/schema-tests.log) |
| gap-probe | 0 | — | — | — | DEFECTS_REPRODUCED | [log](00-rebaseline/logs/finalization/gap-probe.log) |
| extra-probe | 0 | — | — | — | DEFECTS_REPRODUCED | [log](00-rebaseline/logs/finalization/extra-probe.log) |
| historical-hashes | 0 | 30 | 0 | 0 | PASS | [log](00-rebaseline/logs/finalization/historical-hashes.log) |
| contracts-eval | 0 | 7 | 0 | 0 | PASS | [log](00-rebaseline/logs/finalization/contracts-eval.log) |
| edge-eval | 0 | 0 | 0 | — | NOT_RUN | [log](00-rebaseline/logs/finalization/edge-eval.log) |
| asan-configure | 0 | — | — | — | PASS | [log](00-rebaseline/logs/finalization/asan-configure.log) |
| asan-build | 0 | — | — | — | PASS | [log](00-rebaseline/logs/finalization/asan-build.log) |
| asan-ctest | 8 | 0 | 2 | 0 | FAIL | [log](00-rebaseline/logs/finalization/asan-ctest.log) |
| asan-qnn-bench | 134 | — | — | — | FAIL | [log](00-rebaseline/logs/finalization/asan-qnn-bench.log) |
| asan-llm-infer | 134 | — | — | — | FAIL | [log](00-rebaseline/logs/finalization/asan-llm-infer.log) |
| release-configure | 0 | — | — | — | PASS | [log](00-rebaseline/logs/finalization/release-configure.log) |
| release-build | 0 | — | — | — | PASS | [log](00-rebaseline/logs/finalization/release-build.log) |
| release-ctest | 0 | 2 | 0 | 0 | PASS | [log](00-rebaseline/logs/finalization/release-ctest.log) |

Counts refer to the units named by each command: Python tests, CTest executables, schema examples, or historical files. Do not sum them into a single test count. Probe exit zero means the diagnostic ran and demonstrated defects, not that the application passed. Edge evaluation exit zero carries `status=not-run`, not a quality score. The one skipped Python test requires `QEDGE_RUN_CATALOG_INTEGRATION=1` and a licensed `QEDGE_CATALOG_DIR`; owner is the project maintainer. No catalog was acquired or integration enabled.

## Reproducibility and preservation

- [Machine-readable baseline manifest](00-rebaseline/baseline-manifest.json) indexes hashed evidence and source provenance.
- [Commands and exit codes](00-rebaseline/COMMANDS.md) and [pass/fail/skip records](00-rebaseline/results.json) reproduce the local checks.
- [Current source hashes](00-rebaseline/current-source-hashes.json): 46/46 equal the original audit. Finalization rechecked those hashes.
- [Preservation verification](00-rebaseline/preservation.json) records before/after hashes of non-handoff project files, excluding generated environments/build/cache/Git internals, plus unchanged nested Git status.
- Historical manifest: 30 verified, 0 failures. Full protected historical hashes appear in `logs/historical-artifact-hashes.log` and preservation.json.
- [Environment](00-rebaseline/logs/environment.log): macOS 26.3.1 arm64, Apple clang 21.0.0, CMake 4.4.2, Python 3.11.15; package versions included.
- [Git inventory](00-rebaseline/logs/git-state.log): root, edge and shared package are not Git repositories. Nested llm-inference is dirty on main at `edf35b722c1eeb690602e840ff5052c4f2cb170c`; existing Stage 01/QNN edits retained. No resets, commits, staging or cleanup were performed.
- No applicable workspace AGENTS.md was found. Prior root Stage 00 evidence, Stage 01 logs, and component handoffs remain unchanged. Completing this baseline does not accept those older stages.

## Findings and claim changes

[The gap-to-prompt map](00-rebaseline/gap-to-prompt-map.md) classifies all G01–G12 as still present, distinguishing dynamic failures, static findings and missing integrations. [The additive claims ledger](00-rebaseline/claims-ledger.md) records implemented/tested versus proposed/synthetic/calculated/blocked capabilities.

The dynamic probes reconfirm hidden requirements, unit guessing, negation inversion, cross-owner access/overwrite, schema semantic gaps, generic wood falsely supporting oak, and flattened evidence without source revisions. Release green does not establish safe attention. No new throughput, energy, real-model or NPU claim is supported.

## Assets and blockers

Project `llm-inference/models` is empty; the only project catalog is the four-row synthetic fixture. External MiniLM/CLIP files, ABO catalogs and an Ollama blob were inventoried in the existing logs. Hash agreement establishes identity against the recorded external manifest, not model integration, runtime compatibility, or license approval. They were not used in these checks or copied into the project. No downloads are authorized.

Model-dependent stages require implementation plus approved/cached assets. Retrieval quality requires reviewed labels. Hardware measurement requires a compatible actual device/toolchain and run evidence. These do not block collecting the baseline, but remain explicit prerequisites for later claims.

## Exact next repair scope

After reviewer acceptance of this handoff, execute **01-kernel-safety only**. Start with:

- `llm-inference/src/backend/cpu_backend.cpp`: GQA mapping, per-head V multiplication, runtime shape checks, mask broadcasting/fully masked behavior, rotary frequency/base/positions.
- `llm-inference/src/backend/metal_backend.mm`: replace invalid attention behavior with verified implementation or explicit unsupported/labelled CPU fallback.
- `llm-inference/src/benchmark/benchmark.cpp`: handle failed retrieval and unsafe inputs; no broad benchmark redesign.
- `llm-inference/tests/test_kv_cache.cpp` and `tests/test_qnn_stub.cpp`: independent numerical references, mask and GQA regression cases, invalid shapes and rotary cases; inspect both entry points' sanitizer traces.
- `llm-inference/CMakeLists.txt` only if adding a test target is necessary.

Any required public-interface change (for example exposing rotary base or explicit positions) must be identified and justified as a minimal scope amendment before editing files outside that list. Do not silently broaden the repair into cache-policy research, contracts, ownership, assistant parsing, model integration or downloads.

Acceptance for the next repair is numerical correctness plus clean Debug ASan/UBSan tests and entry points, with Release validation too. Passing Release alone is insufficient. Use the full stage-01 prompt's test cases; do not suppress sanitizer failures or change expected outputs to preserve a bug.

## Files changed and stopping point

This completion added the handoff, commands/results, preservation record, claims ledger, gap map, manifest and finalization logs under `handoff/research-review/`. Existing baseline logs/probes were read and preserved. No application code or existing claims ledger changed.

**Stopped after 00-rebaseline. Stage 01 has not started; awaiting review.**
