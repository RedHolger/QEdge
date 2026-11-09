# Stage 02 plan: contract semantics and versioned migration

## Observable failure before editing

The baseline probe at
`handoff/research-review/02-contracts/probes/baseline_gaps.py` was run against
the unchanged v1 schemas. Its output is in
`handoff/research-review/02-contracts/logs/baseline-gaps.log`. The current
validator accepts all of these invalid records:

- a `decision` with `status: SUPPORTED`, `action: RECOMMEND`, and no claims;
- a `decision` with a contradicted child claim and a positive parent;
- a `decision` missing a required predicate (in the probe, no claims at all);
- a `decision` whose child `item_id` differs from the parent;
- invalid `created_at` and nested `retrieved_at` values;
- a null observed value in a nested supported claim (the standalone v1 schema
  rejects the same value, so the nesting is inconsistent);
- `measured: true` on a fixture-smoke record with no model/run evidence;
- `npu_status: measured` with `device_verified: false`;
- `NaN` as an observed value.

The acceptance reference is a new v2 suite that must reject every one of these
cases with a semantic error while the existing v1 fixture and v1 reader remain
valid. Positive v2 fixtures must pass. The probe is diagnostic only; it is not
an acceptance test until its expectations are inverted and moved to the test
suite.

## Contract design

1. Preserve the seven v1 schema files byte-for-byte. They remain the historical
   reader and are recorded in `baseline-schema-hashes.json`.
2. Add v2 schemas for `decision`, `evidence_claim`, `benchmark_result`, and
   `hardware_status` under `qedge-common/schemas/v2/`. `product`,
   `retrieval_request`, and `retrieval_result` remain v1.
3. Document the predicate and response semantics in
   `qedge-common/CONTRACTS.md` before implementing them. The document covers
   strict/inclusive comparisons, equality, inequality/negation, containment,
   ambiguity/unparsed clauses, candidate-specific decisions, no-candidate
   responses, evidence resolution, and response precedence.
4. Extend `qedge_common.schema` with explicit version selection, strict JSON
   loading, finite-number checks, RFC 3339 date-time checks, and semantic
   validators for claims, decisions, benchmark results, and hardware status.
5. Update the CLI and eval consumer to use the validators and expose both
   versions. Add a small qedge-common producer helper that constructs and
   validates a v2 fixture-smoke benchmark record, so a producer boundary cannot
   bypass the semantic validator.
6. Update the edge-rag fixture-smoke producer to emit the v2 envelope with
   `measured: false`, separate outputs, and no invented cache hit rate. Because
   edge-rag intentionally does not depend on the sibling qedge-common package,
   its record will be validated at the qedge-common boundary in tests rather
   than by a cross-checkout import. No download or external dependency is
   added.
7. Add adversarial negative tests for every acceptance case, positive v1/v2
   fixtures, version compatibility tests, schema-hash preservation tests, and
   explicit checks that structural schema success alone is not accepted as a
   factual measurement.

## Tests and commands

Initial regression baseline:

```sh
qedge-common/.venv/bin/python -m unittest discover -s qedge-common/tests -v
qedge-common/.venv/bin/python -m qedge_common.eval
edge-rag-demo/.venv/bin/python -m unittest discover -s edge-rag-demo/tests -v
```

Final checks will add:

```sh
qedge-common/.venv/bin/python handoff/research-review/02-contracts/probes/baseline_gaps.py
qedge-common/.venv/bin/python -m qedge_common.eval
qedge-common/.venv/bin/python -m unittest discover -s tests -v   # from qedge-common
edge-rag-demo/.venv/bin/python -m unittest discover -s tests -v  # from edge-rag-demo
```

The final probe is expected to reject the baseline failures under v2. The
handoff will record exact exits, pass/fail/skip counts, changed files, schema
hashes, and remaining limitations. No model, catalog, GPU, or NPU evidence is
created by this stage.
