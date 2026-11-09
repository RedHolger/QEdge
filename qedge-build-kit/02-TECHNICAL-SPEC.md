# QEdge Technical Specification

## 1. Repository layout

Keep the existing project directories; add a shared workspace around them.

```text
/qualcomm/
├── qedge-build-kit/              # this plan
├── qedge-common/                 # shared schemas, eval utilities, catalog client
│   ├── schemas/
│   ├── eval/
│   └── pyproject.toml
├── llm-inference/                # QEdge Runtime
│   ├── runtime/
│   │   ├── model_runner.py       # real model, tokenizer, generation
│   │   ├── cache/                # KV-cache policies
│   │   └── bench/                # benchmark harness
│   ├── cpp/                      # existing C++ kernel lab
│   ├── qnn/                      # ONNX/QNN/AI Hub scripts
│   ├── reports/
│   ├── handoff/
│   └── Makefile
├── edge-rag-demo/                # QEdge Assist
│   ├── src/edge_rag/
│   ├── catalog/                  # licensed data, manifests, provenance
│   ├── eval/                     # qrels, decision cases, annotation
│   ├── demo/                     # CLI/web
│   ├── reports/
│   ├── handoff/
│   └── Makefile
└── Makefile                      # top-level commands
```

Do not delete the existing C++ code. Reclassify it honestly:

- `runtime/` is the primary, real-model path.
- `cpp/` is the kernel lab and optimization path.
- `qnn/` is the deployment path.
- `legacy/` holds old synthetic results that are preserved but no longer headline evidence.

## 2. Runtime architecture

```text
CLI / Python API
      │
      ▼
ModelRunner ──────► Tokenizer (HF tokenizers)
      │
      ├──► PyTorch reference backend (CPU / MPS)
      ├──► llama.cpp GGUF backend (quantization baseline)
      └──► ONNX Runtime backend (CPU; QNN EP when available)
      │
      ▼
QEdgeCache (KV cache)
      ├── full / contiguous
      ├── FIFO
      ├── LRU
      ├── KeyDiff
      └── QUOKA (query-aware)
      │
      ▼
Sampler (greedy / temperature / top-p)
      │
      ▼
Profiler (TTFT, prefill, decode, memory, cache stats)
```

### 2.1 Model choices

Primary:

- `HuggingFaceTB/SmolLM2-135M-Instruct`
- Apache-2.0; small enough for CPU/Mac testing; 135M parameters.
- Use the instruct model for chat and the base model only if perplexity experiments need it.

Secondary:

- `Qwen/Qwen2.5-0.5B-Instruct`
- Apache-2.0; better quality, still small.
- Use as a robustness check, not the default benchmark.

Quantized:

- GGUF Q8_0 and Q4_K_M versions of the primary model through llama.cpp.
- Record the exact repository, revision, file hash, and license.

Do not download a 7B model in Tier 1. It will make iteration slow and tell you nothing new about the code.

### 2.2 Real inference requirements

The runner must support:

- tokenizer encode/decode;
- chat template application;
- prefill of the prompt;
- token-by-token decode;
- greedy and sampled generation;
- streaming output;
- seed control;
- max new tokens and stop conditions;
- a machine-readable generation record.

A generation record contains:

```json
{
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "revision": "...",
  "quantization": "fp16",
  "prompt": "...",
  "prompt_tokens": 0,
  "output_tokens": 0,
  "ttft_ms": 0.0,
  "decode_tokens_per_sec": 0.0,
  "prefill_tokens_per_sec": 0.0,
  "peak_rss_mb": 0.0,
  "cache_budget": 1.0,
  "policy": "full",
  "seed": 0,
  "hardware": "Apple M2 / CPU",
  "measured": true
}
```

### 2.3 KV cache design

Implement a cache object compatible with the model's attention layer. In the PyTorch path, subclass or adapt the `transformers` cache API. Do not fake it with a separate tensor store that never reaches attention.

Required behavior:

- per-layer key/value tensors;
- grouped-query attention (GQA): query heads > KV heads;
- position tracking;
- append, retrieve, evict;
- capacity measured in tokens per layer or total tokens;
- budget = current capacity / full context capacity;
- stats: hits, misses, evictions, resident tokens, peak bytes.

Policies:

| Policy | Real behavior |
|---|---|
| FIFO | Evict oldest resident token |
| LRU | Evict least recently attended token |
| KeyDiff | Evict one token from the most similar key pair, or the token closest to the newest key; document the exact definition |
| QUOKA | Evict the token with the lowest query-key relevance for the current query/attention distribution; use real query scores, not a mean-key proxy |

QUOKA must use the current query vector or accumulated attention scores. A mean-of-keys proxy is not QUOKA and must not be labelled as such.

Correctness tests:

- at 100% budget, output matches the full-cache reference for the same prompt and seed;
- at reduced budget, no out-of-bounds and no NaNs;
- cache stats sum correctly;
- eviction never removes the current token;
- position IDs remain correct after eviction;
- GQA shapes are respected.

### 2.4 Quantization

Two paths:

1. **GGUF via llama.cpp** — Q8_0 and Q4_K_M. This is the fast, credible baseline.
2. **ONNX quantization** — optional; required only if the QNN path needs it.

Measure:

- model file size;
- peak RSS;
- TTFT;
- decode tok/s;
- quality on a fixed prompt set (perplexity if available, task accuracy otherwise).

Do not claim an "INT4 NPU kernel" unless it was actually executed on an NPU.

### 2.5 QNN / AI Hub path

The goal is a real deployment path, not a fake NPU number.

Required artifacts:

- `qnn/export_onnx.py` — export the model with past key/values.
- `qnn/run_ort_qnn.py` — ONNX Runtime with the QNN execution provider.
- `qnn/README.md` — exact commands for Snapdragon X Elite / 8 Elite.
- `qnn/compile_aihub.py` or a documented AI Hub Workbench flow.
- `qnn/hardware_status.json` — available/unavailable and evidence paths.
- `qnn/device_runbook.md` — prerequisites, driver, Python, ORT version, env vars.

If no device is available:

```json
{
  "device": "Snapdragon X Elite",
  "status": "unverified",
  "reason": "No device or AI Hub run available in this environment",
  "local_backend": "CPU/Metal only"
}
```

No README table may imply an NPU measurement when this status is `unverified`.

## 3. Assistant architecture

```text
CLI / FastAPI
   │
   ├── Query parser (text, image, requirements)
   ├── Retriever
   │     ├── BM25
   │     ├── CLIP text→image
   │     ├── MiniLM text→text
   │     ├── RRF fusion
   │     └── optional cross-encoder reranker
   ├── Evidence store (SQLite; optional Postgres)
   ├── Verifier
   │     ├── width/height/length/material/color/brand
   │     ├── unit normalization
   │     └── source citations
   ├── Decision policy
   │     ├── SUPPORTED
   │     ├── CONTRADICTED
   │     ├── UNKNOWN
   │     └── ABSTAIN
   └── Answer composer
         ├── extractive (default)
         └── optional QEdge Runtime generation
```

### 3.1 Catalog

A catalog item:

```json
{
  "item_id": "...",
  "variant_id": "...",
  "title": "...",
  "category": "...",
  "text": "...",
  "image_path": "...",
  "image_sha256": "...",
  "source_url": "...",
  "source_revision": "...",
  "license": "...",
  "attributes": {
    "width": {"value": 48.0, "unit": "cm", "status": "present"},
    "material": {"value": "oak", "status": "present"},
    "color": {"value": "beige", "status": "present"}
  }
}
```

Rules:

- never invent a unit; if the source has no unit, store `missing` and abstain;
- convert inches to cm explicitly and record the original;
- preserve `missing`/`unknown`/`conflicting` states;
- deduplicate variants by family;
- keep provenance and license per item.

### 3.2 Evidence and citations

An evidence claim:

```json
{
  "claim_id": "...",
  "item_id": "...",
  "attribute": "width",
  "operator": "<=",
  "required_value": 60.0,
  "required_unit": "cm",
  "observed_value": 48.0,
  "observed_unit": "cm",
  "status": "SUPPORTED",
  "source": "catalog.item_dimensions",
  "source_revision": "...",
  "evidence_text": "width: 48.0 cm",
  "retrieved_at": "..."
}
```

The verifier must not treat retrieval as evidence. Evidence must come from a stored source record with a citation.

### 3.3 Decision policy

- If all required attributes are `SUPPORTED`: `SUPPORTED`.
- If any required attribute is `CONTRADICTED`: `CONTRADICTED`.
- If a required attribute is missing/unknown: `UNKNOWN` and `ABSTAIN` from a positive recommendation.
- Never say "matches your requirements" unless the decision is `SUPPORTED`.
- Show the reason and the citation for every decision.

### 3.4 Optional generation

Generation is optional. If used:

- pass only verified evidence to the prompt;
- require citations in the output;
- run a claim-support check after generation;
- fall back to extractive answer if the check fails.

The demo must work without a model download. The model improves fluency, not truth.

## 4. Shared contracts

Put these in `qedge-common/schemas/`:

- `product.schema.json`
- `retrieval_request.schema.json`
- `retrieval_result.schema.json`
- `evidence_claim.schema.json`
- `decision.schema.json`
- `benchmark_result.schema.json`
- `hardware_status.schema.json`

Generate Python and C++ bindings or at least validate with `jsonschema`/`pydantic`. Both projects must use the same catalog and benchmark records.

## 5. Security and multi-user behavior

Tier 1 can be single-user. If the assistant exposes document writes:

- derive identity from the session, not the request body;
- scope document IDs by tenant;
- check existing membership before overwrite;
- use optimistic concurrency with explicit `expected_version`;
- reauthorize on every historical citation;
- test revocation and deleted-ID reuse.

Do not claim "secure" because tests pass. Claim only the specific properties that have tests.

## 6. Observability

Every run should emit:

- a JSON benchmark record;
- a stage timing breakdown;
- cache stats;
- model/hardware metadata;
- a claims ledger entry for any headline number.

Logs must never contain secrets, tokens, or full user documents.

## 7. Hardware status

`hardware_status.json` is the single source of truth:

```json
{
  "cpu": "Apple M2",
  "metal": {"available": true, "used": true},
  "snapdragon": {"available": false, "model": null},
  "qnn": {"sdk_available": false, "device_verified": false},
  "last_updated": "..."
}
```

Every report table must state which row is measured on which hardware. "Projected" and "modelled" are allowed words; "measured" is not, unless a device log exists.

## 8. Legacy mapping

| Current file/claim | New status |
|---|---|
| `src/benchmark/benchmark.cpp` synthetic cache benchmark | `legacy/` or clearly labelled microbenchmark |
| `build/llm_qnn_bench` power report | legacy; not a headline number |
| `qnn_stub_backend` | useful interface prototype; not NPU evidence |
| `MetalBackend::attention` | must be fixed or removed from claims |
| `edge_rag.generator` template | replaced by decision-aware answer composer |
| `PipelineProfiler` hardcoded NPU flags | replaced by measured stage metadata |
| README "12,460 tok/s" / "79% / 40%" / "< 3 ms" | removed or linked to a new artifact |

## 9. Definition of a real end-to-end path

A reviewer must be able to run:

```bash
make setup
make demo
make bench
make eval
```

and get:

1. a generated answer from a real model;
2. a retrieval result over a real catalog;
3. a decision with citations;
4. a benchmark record with hardware labels;
5. an evaluation report with confidence intervals.

If any of those is a template or a hardcoded string, the stage is not done.
