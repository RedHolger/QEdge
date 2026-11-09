# QEdge Common Contracts

These are the versioned boundary contracts shared by QEdge Runtime
(`llm-inference`) and QEdge Assist (`edge-rag-demo`).

| File | Current version | Purpose |
|---|---|---|
| `product.schema.json` | v1 | A licensed catalog record with explicit missing/unknown attribute states. |
| `retrieval_request.schema.json` | v1 | A query request; text, image, or saved-item seeds. |
| `retrieval_result.schema.json` | v1 | Ranked candidates plus timing and warnings. |
| `evidence_claim.schema.json` | v1 / v2 | One attribute checked against a stored source record. |
| `decision.schema.json` | v1 / v2 | A candidate-specific verdict over required evidence, including no-candidate responses. |
| `benchmark_result.schema.json` | v1 / v2 | A measurement or explicitly synthetic record; generated outputs are separate in v2. |
| `hardware_status.schema.json` | v1 / v2 | The single source of truth for CPU/GPU/NPU availability. |

The v1 files are preserved as historical readers for existing artifacts. The
v2 files live under `v2/` and add the semantic fields and invariants documented
in [`../CONTRACTS.md`](../CONTRACTS.md). New producers must select the version
they intend to satisfy. The validator selects it from `schema_version` unless
the caller supplies one explicitly.

Rules:

- Unknown fields are rejected at boundaries; add a new schema version instead
  of silently changing a contract.
- A claim requires `source` and `source_revision`; retrieval is not evidence.
  Supported and contradicted claims also require a non-null observation and
  evidence text.
- A benchmark result is `measured: false` for synthetic microbenchmarks,
  fixture smokes, or projections. Never set it to true without a reproducible
  run and hardware log.
- `hardware_status.npu_status` must remain `unverified` until a Snapdragon/QNN
  device run exists. A measured NPU status in v2 also requires execution
  evidence and a matching backend/device identity.
- `cache.hit_rate` is optional. Do not invent it for a policy where it is not
  meaningful.
- Passing schema or semantic validation is not factual proof. It is a
  boundary check over the supplied evidence and provenance.

Validation:

```bash
cd qedge-common
uv sync --python 3.11
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m qedge_common.eval
.venv/bin/qedge-validate product tests/fixtures/valid/product.json
```
