# QEdge architecture

## Status

Stage 01 makes the two projects reproducible and gives them shared contracts.
It does not add a model, a catalog, a retrieval-quality result, or an NPU run.

```text
                         Stage 01 current path
                         =====================

  edge-rag-demo (QEdge Assist)
    fixture/optional catalog JSONL
      -> local BM25 baseline
      -> SQLite evidence chunks
      -> SUPPORTED / CONTRADICTED / UNKNOWN claims
      -> neutral Stage 01 template (no recommendation)
      -> in-process stage timing (CPU only)

  llm-inference (QEdge Runtime)
    synthetic tensors
      -> CPU / experimental Metal kernels
      -> KV-cache policies
      -> synthetic microbenchmark and QNN CPU-stub report
      -> no model, tokenizer, or NPU execution
```

## Target architecture

The later stages replace lab scaffolding with a real runtime and evidence path:

```text
  QEdge Runtime (llm-inference)
    Python runtime/
      -> tokenizer
      -> real small-model weights (approved in Stage 02)
      -> CPU reference generation
      -> KV-cache policy comparisons
      -> optional, separately gated Snapdragon/QNN deployment (Stage 05)

  QEdge Assist (edge-rag-demo)
    licensed catalog + provenance
      -> text/image retrieval baseline
      -> evidence claims with source revision
      -> decision policy (SUPPORTED / CONTRADICTED / UNKNOWN / ABSTAIN)
      -> extractive answer with citations
      -> optional QEdge Runtime generation

  qedge-common
      -> shared JSON Schemas for every boundary above
```

The runtime and assistant remain separate installable units. They share
contracts, not hidden imports.

## Boundaries and truth rules

### Kernel lab vs product runtime

`llm-inference` root C++ code is the **kernel lab**. It is retained for kernel
experiments and tests. The product runtime path is `runtime/` and starts in
Stage 02. A synthetic microbenchmark must never be labelled as model
throughput.

### Hardware status

`npu_status` is `unverified` until a Snapdragon/QNN device run is recorded with
logs. The QNN stub is a CPU implementation. It cannot produce an NPU claim,
offload percentage, energy result, or Snapdragon projection.

### Retrieval and evidence

The Stage 01 edge baseline is local BM25 over catalog JSONL. It is not
FAISS+BM25+RRF, not a neural retriever, and not a retrieval-quality result. The
SQLite store is local and simple; it is not a security or tamper-resistance
claim.

### Missing data

Missing catalog integrations fail explicitly or skip with a printed reason.
There is no fallback to another checkout and no silent empty-success path.

## Shared contracts

`qedge-common/schemas/` contains:

- `product.schema.json`
- `retrieval_request.schema.json`
- `retrieval_result.schema.json`
- `evidence_claim.schema.json`
- `decision.schema.json`
- `benchmark_result.schema.json`
- `hardware_status.schema.json`

Rules:

- boundaries reject unknown fields;
- a claim requires a source and source revision;
- a benchmark record has `measured: false` unless a reproducible hardware run
  exists;
- missing/unknown/conflicting product attributes are preserved, not guessed;
- `product`, `retrieval_request`, and `retrieval_result` remain at
  `schema_version` `"1"`; `decision`, `evidence_claim`, `benchmark_result`, and
  `hardware_status` have v2 contracts while retaining their historical v1
  readers;
- v2 decisions distinguish candidate-specific and no-candidate responses,
  enforce complete mandatory predicate coverage, and block recommendation on
  contradiction or conflicting evidence;
- passing a schema or semantic check is not factual proof of a catalog source,
  model run, or hardware measurement.

## Commands and gates

| Command | Stage 01 meaning |
|---|---|
| `make setup` | local Python envs, locked dependencies, CMake configure |
| `make test` | path scan, schema tests, edge tests, C++ CTest |
| `make demo` | synthetic C++ kernel-lab output + fixture-only edge demo |
| `make bench` | synthetic C++ report + fixture smoke record (`measured:false`) |
| `make eval` | historical hash check + contract validation + explicit stage gates |

`make test` does not download a model or catalog and does not use the network.
`make setup` may fetch the locked Python packages.

## Open defects kept for later stages

These are deliberately not fixed in Stage 01:

- edge requirement parsing and default injection (Stage 07);
- edge unconditional/decision wording (Stage 07);
- evidence-store authorization and concurrency boundaries (later security
  stage);
- known attention shape/out-of-bounds risk in the C++ lab (Stage 03);
- real catalog license shipping decision (before catalog distribution);
- NPU deployment and measurement (Stage 05).

Stage 01 only labels, isolates, and tests around them.
