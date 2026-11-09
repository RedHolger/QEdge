# QEdge Assist — portable retrieval and evidence foundation

`edge-rag-demo` is the QEdge Assist prototype: a local retrieval, evidence
verification, and answer-composition path for on-device product assistance.

**Stage 01 status:** portable packaging and tests. This is not an NPU result,
not a real hybrid neural retrieval benchmark, and not a decision-policy release.

## Current path

```text
query
  └─> local BM25 baseline over a catalog JSONL file
        └─> SQLite evidence chunks
              └─> SUPPORTED / CONTRADICTED / UNKNOWN attribute checks
                    └─> neutral Stage 01 template (no recommendation yet)
```

The package uses only the Python 3.11 standard library at runtime. The old
external `discoveryeval` and `evidence-rag` imports and the symlinked `.venv`
were removed in Stage 01.

## Quick start

From the workspace root:

```bash
make setup
make test
make demo
```

Or from this directory:

```bash
uv sync --python 3.11
.venv/bin/python -m unittest discover -s tests -v
QEDGE_CATALOG_DIR=tests/fixtures .venv/bin/edge-rag-demo "wooden dining chair under 60cm"
```

`make demo` uses the synthetic four-row fixture in `tests/fixtures/`. The fixture
is a packaging and mechanics test, not ABO data. A real licensed catalog arrives
in Stage 06 and can be selected explicitly:

```bash
QEDGE_CATALOG_DIR=/path/to/catalog-directory .venv/bin/edge-rag-demo "oak chair under 80 cm"
```

No catalog is downloaded by the commands above.

## Install outside the source tree

```bash
uv build
uv venv /tmp/qedge-assist-venv --python 3.11
uv pip install --python /tmp/qedge-assist-venv/bin/python dist/edge_rag_demo-0.2.0-py3-none-any.whl
/tmp/qedge-assist-venv/bin/edge-rag-demo --help
```

The wheel contains only `edge_rag`; the fixture catalog is test data and is not
shipped as real product data.

## What is verified in Stage 01

- package imports and CLI entry point work from a wheel installed outside the
  source tree;
- no machine-specific home-directory path appears in source or tests;
- tests run without network, models, external checkouts, or Docker;
- local BM25 retrieval mechanics work on the fixture;
- SQLite evidence chunks and SUPPORTED/CONTRADICTED/UNKNOWN checks work;
- profiler records stage wall time and reports `npu_status: unverified`.

## What is not verified or implemented

- no Snapdragon, QNN SDK, AI Hub, or NPU execution;
- no NPU-offload percentage and no `< 3 ms` projection;
- no neural CLIP/MiniLM retrieval, FAISS index, or RRF fusion;
- no human relevance labels or retrieval-quality score;
- no decision policy: the Stage 01 template never recommends a product;
- requirement parsing and default injection still contain the Stage 00 defects
  (for example, bare chair queries may still inject width/oak defaults);
- the evidence store is not an authorization or tamper-resistance claim;
- the catalogue licence decision is still pending shipping approval.

## Layout

```text
edge-rag-demo/
├── src/edge_rag/       # installable package (retriever, verifier, CLI, profiler)
├── tests/              # unit tests + synthetic fixture catalog
├── legacy/             # frozen Stage 00 README and synthetic claims
├── handoff/            # stage audits and handoffs
├── pyproject.toml
└── uv.lock
```

See the workspace [`docs/CLAIMS.md`](../docs/CLAIMS.md) and
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the evidence status and
the planned end-to-end path. The old synthetic claims are preserved in
[`legacy/`](legacy/). The Stage 01 acceptance criteria are in the workspace
[`qedge-build-kit/`](../qedge-build-kit/).
