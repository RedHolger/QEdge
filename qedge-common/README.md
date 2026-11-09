# qedge-common

Shared JSON Schemas and validation utilities for the QEdge portfolio.

This is a foundation package, not a model runtime or retrieval engine. It owns
the field names that both projects use for products, retrieval, evidence,
decisions, benchmark records, and hardware status.

```bash
uv sync --python 3.11
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m qedge_common.eval
```

The canonical schemas are in [`schemas/`](schemas/). The semantic rules are in
[`CONTRACTS.md`](CONTRACTS.md). If a contract changes, bump its version and
update the checked-in examples and tests in the same change.

The migrated contracts (`decision`, `evidence_claim`, `benchmark_result`, and
`hardware_status`) have explicit v1 and v2 readers. Use `--schema-version` to
force the historical reader; otherwise the validator selects the version from
the payload. `qedge-common` applies RFC 3339 and finite-number checks at every
version and applies the v2 semantic invariants to v2 records.
