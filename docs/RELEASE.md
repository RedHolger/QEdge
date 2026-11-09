# Local CPU portfolio release — 9 November 2026

## Five-minute reproducible demo

Acquisition/build is explicit and separate from execution. With Python 3.11 and uv,
run `sh scripts/bootstrap_portfolio.sh` once from the workspace. This creates
`.release/venv` and wheels; it does not download model weights or ABO assets.

Then run these installed commands, with no network or model required:

```sh
.release/venv/bin/qedge-portfolio demo
.release/venv/bin/qedge-portfolio benchmark --trials 10
.release/venv/bin/qedge-portfolio evaluate
```

The demo finds a fixture oak chair that satisfies width constraints and displays
its evidence. Evaluation exercises supported, contradicted, missing, ambiguous,
negated, unsupported, changed/conflicting source, no-result, image-refusal and
failed-generation paths. It also checks isolation and the 80 authored regression
cases. A perfect regression score does not establish generalization or catalog
retrieval quality. The benchmark is CPU fixture pipeline timing, including the
first request on a reused assistant, not neural generation throughput.

`python3 scripts/verify_portfolio_release.py` rebuilds from relocated package
sources, installs wheels into a disposable environment, removes source trees,
and runs all three commands with Python isolation and socket connections denied.
Run bootstrap first to populate explicit offline build/dependency caches.

## Capability and activation matrix

| Capability | Release status | Activation / evidence |
|---|---|---|
| Offline deterministic assistant | Validated with packaged fixtures | `qedge-portfolio demo` |
| Internal decision evaluation | 80 cases, previously exercised | `qedge-portfolio evaluate` |
| CPU fixture timing | Measured locally with finite trials | `qedge-portfolio benchmark --trials 10` |
| User catalog | Explicit data required | `edge-rag-demo --catalog PATH --mode find QUERY` |
| Neural catalog retrieval | Optional assets/index required; human quality blocked | Set `QEDGE_ENABLE_NEURAL=1`, `QEDGE_MODELS_DIR`, `QEDGE_INDEX_DIR`, `QEDGE_CATALOG_DIR`; acquire/verify assets explicitly first |
| Real SmolLM2 CPU generation | Separate optional runtime and checkpoint | `make -C llm-inference/runtime acquire`, then its documented CPU CLI |
| ONNX CPU export/parity | Validated no-KV full-prefix baseline | `uv sync --project llm-inference/runtime --extra onnx`; use Stage 11 export/parity functions |
| GGUF variants | Optional real-model CPU experiment | Stage 08 handoff and runtime quantization module |
| C++ kernel lab | macOS Release and ASan/UBSan checks | `cmake --preset release`, `cmake --preset debug`, matching build/ctest presets in llm-inference |
| Metal attention | CPU fallback | Explicit backend label; not GPU attention performance |
| Qualcomm HTP | BLOCKED, UNVERIFIED | No activation shortcut: obtain compatible hardware/SDK, prepare specialized graph, follow provisional Stage 11 runbook and validate placement logs |
| Image execution | Not part of installed fixture release | Refusal path only; assets and an evaluated image path required |
| Independent generalization | PENDING | Independently curate and freeze new cases before tuning |
| Human retrieval quality | BLOCKED | Complete 1,009 fresh pair judgments and the 120-query plan |
| Qwen second model | DEFERRED | No activation included or new download authorized by this release |

## Evidence and provenance inventory

- `qedge-common` wheels include canonical v1/v2 schemas. Versioned contracts,
  migrations, stale indexes and corrupted/missing evidence are exercised by the
  shared and edge test suites. Original v1 schema hashes are preserved.
- Installed fixture catalog and decision cases are repository-authored regression
  material. Only these fixtures are bundled with the portfolio wheel.
- SmolLM2 revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` and local asset digests
  are governed by acquisition checks. The runtime records Apache-2.0 provenance.
  Weights, GGUF files and ONNX files are not bundled in the assistant wheel.
- The local ABO subset has an acquisition manifest and
  `edge-rag-demo/catalog/reports/license_attribution.md`. This release does not
  redistribute the ABO catalog, images, or third-party embedding weights.
- Optional neural model manifests live under `edge-rag-demo/models/`; their license
  and attribution records must accompany any separate redistribution.
- Dependency versions for the installed portfolio and optional runtime are recorded
  in the release handoff. Upstream dependency licenses remain applicable.
- No new blanket license for repository code is asserted. No public package, tag,
  push or external publication is performed by this release preparation.

## Rollback and migrations

Retain old wheels and evidence manifests before replacing an environment. Reinstall
an archived wheel into a separate environment rather than editing site-packages.
Remove only the disposable `.release/venv` if rolling back the portfolio install;
leave catalogs, model assets, historical results and user databases untouched.

The evidence store uses transactional schema-v3 migration for supported older
stores. Back up the SQLite database (using SQLite's backup API for a live store)
before migration. Downgrading application code does not reverse schema changes:
restore the matching database backup instead. Do not infer legacy string booleans.
Indexes with mismatched catalog/model/preprocessing digests must be rebuilt, not
relabelled. Never overwrite historical model/build manifests to make validation pass.

## Limits and interpretation

This is a validated local CPU portfolio, not a production service. Principal-based
access checks assume a trusted local caller that supplies identity. Network
authentication, deployment isolation, load/SLO testing and physical-device evidence
are outside this release. Deadlines are cooperative. Request limits are 4,096 query
characters and at most 100 retrieved/verified candidates. Untrusted free-form model
prose is never used as the assistant answer.

The ONNX manifest validator proves declared artifact/interface consistency and
checks configured source assets. It is not a signature establishing who exported
an arbitrary graph. Unknown historical exporter metadata remains unknown.
Target graph preparation and hardware validation are still incomplete. Provisional
SDK commands are not tested-device instructions. Analytical cache byte calculations
are not measured transfer latency or VTCM residency.

Stage 00/01 history is preserved. Superseded active READMEs are archived under the
Stage 12 handoff. Earlier failures are historical findings, not current pass results.
Stage 11/12 implementation verification is by this implementing agent; a separate
independent release reviewer has not been claimed.
