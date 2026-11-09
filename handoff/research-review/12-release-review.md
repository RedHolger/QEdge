# Stage 12 — bounded local CPU portfolio release

Date: 9 November 2026. Verdict: **ACCEPTED for the validated local CPU portfolio**.
Implementation and verification were completed by this agent at the user's request.
A separate independent release review is not claimed. No publishing, push or tag
was performed. Hardware and human-evaluation gates remain open.

## What changed and why

- Finished Stage 11 exact-axis ONNX provenance checks and boundary regressions.
- Added the installed `qedge-portfolio demo|benchmark|evaluate` entry point.
  Authored fixtures are included in the wheel. No source checkout, neural model,
  licensed catalog, or runtime dependency is needed for these commands.
- Added explicit bootstrap and relocated/offline installation verification scripts.
- Added query/candidate/deadline limits and malformed-request regressions. CLI input
  now reaches validation rather than silently clamping invalid candidate limits.
- Declared optional ONNX dependencies for the separate runtime and updated its lock.
- Replaced stale Stage 01 README/architecture/claims text. Original copies are in
  `12-release-review/superseded-docs`; original Stage 00/01 artifacts remain intact.
- Added release procedure, capability matrix, provenance/license inventory,
  rollback/migration notes, finite evaluation denominators and remaining gates.

Primary files: `llm-inference/runtime/src/qedge_runtime/qualcomm/provenance.py`,
`llm-inference/runtime/tests/test_provenance_dimensions.py`, runtime pyproject/lock,
`edge-rag-demo/src/edge_rag/portfolio.py`, assistant/CLI/path modules,
`edge-rag-demo/tests/test_release_requests.py`, edge pyproject, root Makefile,
`scripts/bootstrap_portfolio.sh`, `scripts/verify_portfolio_release.py`,
`scripts/release_inventory.py`, READMEs, `docs/RELEASE.md`, `docs/CLAIMS.md`,
`docs/ARCHITECTURE.md`, and Stage 11/12 handoffs.

## Verification

All listed successful checks exited 0. Exact commands and log paths are in
[results.json](12-release-review/results.json).

| Check | Result | Evidence |
|---|---|---|
| Full runtime, including real export/quantization/parity | 137 passed | [log](12-release-review/logs/runtime-tests.log) |
| Assistant, catalog integration enabled | 135 passed | [log](12-release-review/logs/edge-tests.log) |
| Shared contracts | 49 passed | [log](12-release-review/logs/common-tests.log) |
| New dimension boundary checks | 2 passed | [log](12-release-review/logs/dimensions.log) |
| macOS Release CTest | 3/3 passed | [log](12-release-review/logs/ctest-release.log) |
| macOS Debug ASan/UBSan CTest | 3/3 passed | [log](12-release-review/logs/ctest-debug.log) |
| Sanitized llm_infer and llm_qnn_bench | Both exit 0, no sanitizer markers | [infer](12-release-review/logs/sanitized-infer.log), [stub](12-release-review/logs/sanitized-qnn-stub.log) |
| Fresh wheel installation | Passed | [log](12-release-review/logs/clean-install.log) |
| Relocated build, sources removed, isolated Python, socket connections denied | Demo/benchmark/evaluate passed | [log](12-release-review/logs/relocated-install.log) |
| Historical hashes / v1 schema hashes | 30/30 and 7/7 | [historical](12-release-review/logs/historical-hashes.log), [schemas](12-release-review/logs/schema-hashes.log) |
| Installed internal evaluation | 80/80 cases; 10 scenario paths | [payload](12-release-review/evaluation.json) |

The internal evaluation observed 0 false-SUPPORTED decisions among 58 adverse cases,
with reported Wilson 95% interval [0, 0.062118]. These cases were previously exercised;
this is not independent generalization evidence. Image refusal is covered, image
execution is skipped. Fixture timing is in [benchmark.json](12-release-review/benchmark.json),
including platform, date, query, catalog hash, trial count and individual samples.
It contains no model or device throughput claim.

Existing regression suites cover migrations, schema errors, stale indexes, missing
and corrupted model/evidence artifacts, authorization, malformed requests and
resource limits. The full dirty C++ checkout still has pre-existing trailing
whitespace in tests/test_kv_cache.cpp:195 (`git diff --check` exit 2). This unrelated
line was preserved. Runtime tests emitted a /dev/null ResourceWarning with exit 0.

## Package and data scope

Clean relocated installation was verified for the assistant and shared-contract
wheels. The optional runtime wheel builds; heavyweight runtime dependencies were
exercised in its existing environment, not newly installed in a completely isolated
dependency environment. This limitation is part of the optional runtime boundary.
Dependency versions/license metadata are recorded in
[portfolio inventory](12-release-review/portfolio-dependencies.json) and
[runtime inventory](12-release-review/runtime-dependencies.json).
Third-party models and ABO data are not bundled in the assistant wheel.

## Stage/gate disposition

Stages 00–10 retain prior bounded acceptance recorded in the review conversation
and their handoffs. Where written independent reviews exist, they remain under
`reviews/research-review/`. This release does not manufacture missing reviewer
signatures. Stage 11's remaining local repair is now implemented and tested.
Stage 12 local release preparation is complete, with implementing-agent verification.

Still open: Qualcomm target preparation and physical HTP execution/performance,
1,009 fresh human relevance judgments plus the 120-query evaluation,
independent generalization evaluation, and deferred Qwen validation. None is
converted to PASS by completing this release. See [release guide](../../docs/RELEASE.md)
for activation/dependency steps and rollback notes.
