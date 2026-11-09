# QEdge research review and engineering direction

Reviewed 29 October 2026 against the live workspace and the supplied Executive Blueprint Word document. The source filename and checksum are in [source-manifest.json](evidence/source-manifest.json); paragraph identifiers below refer to [research-extracted.txt](evidence/research-extracted.txt), not Word page numbers.

## Principal engineering decision

Keep the three-component architecture and the existing build kit. Repair correctness before expanding the architecture. Build a small, reproducible local product with evidence-backed decisions and one real model; then evaluate cache policies, quantization, and an explicitly gated Qualcomm deployment path. Calling this production-grade today would exceed the evidence.

The research is useful as a hypothesis and defect inventory. It is not an executable specification or a validated patch. It contains two overlapping implementation sets, incompatible algorithm descriptions, incomplete deployment code, and numerical result tables without the raw experiments needed to establish those results. Do not paste its replacement files into the repository wholesale.

This review adds documentation, prompts, and audit evidence only. Existing OpenCode changes and application source are preserved. The earlier `documentation.md` remains its original snapshot. No model was downloaded, no research replacement was installed, and no Snapdragon run was performed.

## What was independently reproduced

| Check | Result | Evidence |
| --- | --- | --- |
| Existing assistant tests | 19 discovered: 18 passed, 1 licensed-catalog integration skipped | [edge-tests.log](evidence/edge-tests.log) |
| Existing shared schema tests | 13 passed | [schema-tests.log](evidence/schema-tests.log) |
| Separate C++ Debug build with ASan and UBSan | Build succeeded | [asan-build.log](evidence/asan-build.log) |
| CTest on that build | Both CTest executables aborted with heap-buffer-overflow in CPU attention | [asan-ctest.log](evidence/asan-ctest.log) |
| QNN CPU-stub benchmark on that build | Aborted with heap-buffer-overflow; exit status 134 | [asan-qnn-bench.log](evidence/asan-qnn-bench.log) |
| Extra parser, ownership, and schema probes | All six defect categories below demonstrated | [reproduced-gaps.json](evidence/reproduced-gaps.json) |
| Historical evidence hashes | 30 verified; 0 failures | [historical-hashes.log](evidence/historical-hashes.log) |

The C++ sanitizer build is isolated under `/tmp/qedge-research-asan`. These failures are new reproductions, not quotations from the research. The Python test successes demonstrate coverage gaps, not correct end-to-end behavior. The audit probes are diagnostic examples; they do not assert the defective behavior should remain valid.

## Current project gaps and priorities

| ID | Priority | Finding and source | Required outcome |
| --- | --- | --- | --- |
| G01 | P0 | `llm-inference/src/backend/cpu_backend.cpp:127`: Q heads index K as if head counts matched; the final GEMM does not perform independent per-head attention. Sanitizers reproduce the overflow. | Correct GQA mapping, shape validation, masking, and independent reference tests. |
| G02 | P0 | `src/backend/metal_backend.mm:111`: attention returns `matmul(q,k)` and ignores V, softmax, and the mask. | Correct implementation or explicit unsupported/labelled CPU fallback. |
| G03 | P1 | `src/backend/cpu_backend.cpp:35`: RoPE repeats one position, hardcodes the base, and uses `j/head_dim` rather than `2*j/head_dim` for split-half pairing. | Match the selected checkpoint's rotary convention and original positions. |
| G04 | P1 | `src/benchmark/benchmark.cpp:10`: RNG is re-created with seed 42 on each call; the workload retrieves one synthetic past token and ignores retrieval failure. Headline capacity does not force eviction. | Truthful synthetic fixtures; real runtime measurements separated from kernel timings. |
| G05 | P1 | `src/kv_cache/kv_cache.cpp`: LRU is access-count based; KeyDiff compares to newest key; QUOKA has no query input. | Define actual semantics; separate paper reproductions from named local heuristics. |
| G06 | P1 | `llm-inference/runtime/` has only a README. | One pinned model and tokenizer, baseline generation, parity, and artifacts before compression. |
| G07 | P0 for recommendations | `edge-rag-demo/src/edge_rag/cli.py`: red-chair query adds oak and width requirements. Parser treats `chair under 60` as cm and `not oak chair` as positive oak. | Explicit requirement representation, ambiguity handling, and deterministic decisions. |
| G08 | P0 before sharing | `edge-rag-demo/src/edge_rag/store.py`: another requester can read and another owner can overwrite an existing document. | Enforce a declared local-single-user or authenticated multi-user boundary; transactional versions and immutable citations. |
| G09 | P0 for evidence claims | Current schemas accept SUPPORTED with empty claims and `npu_status=measured` with `device_verified=false`. | Structural validation plus cross-field and cross-record semantic invariants. |
| G10 | P1 | Verifier flattens source facts into strings; generic wood may certify oak; source units/provenance need preservation. | Typed facts, asymmetric taxonomy, exact units, revision-resolving citations. |
| G11 | P1 | BM25 fixture pipeline is implemented; neural/image retrieval and independent relevance labels are not demonstrated. | Provenance-governed catalog and held-out retrieval evaluation before hybrid claims. |
| G12 | P1 | QNN stub models energy; profiling scope differs from the overall timed workload. | Remove measured-energy implications; time scopes and units must reconcile. |

Priority here reflects the current application: local single-user storage is not automatically a remotely exploitable service. It becomes a release blocker when ownership isolation is claimed or a shared interface is exposed. Missing research features are scope gaps, not automatically security defects.

## Where the research itself needs correction

### Algorithm naming and reproduction

The first synthesis describes QUOKA as chunked-prefill query subselection; the later section and code label a decayed attention-score/value-norm eviction heuristic “H2O-QUOKA.” Those are different experiments. The published [QUOKA abstract](https://arxiv.org/abs/2602.08722) describes representative-query selection followed by key selection during chunked prefill. Its CPU/GPU performance results are not Snapdragon measurements. Call the proposed hybrid `attention_value_decay` until independently specified and evaluated; do not claim a QUOKA reproduction merely because a query affects eviction.

Likewise, the document alternates a normalized-key mean-anchor method and a centroid-plus-neighbor rule (paragraphs 2359–2370 and 2923–2930). The latter needs its own name and ablation. Pin a version of the [KeyDiff paper](https://arxiv.org/html/2504.15364v1), identify the precise variant and head aggregation, and test its selector against a small reference. Geometry-based relevance is an empirical/model-dependent relationship, not proof every retained key matters on every task.

### Proposed runtime integration is not yet sound

In the later runner, the `q_proj` hook supplies pre-RoPE queries (paragraphs 3123–3133), while cached keys in the intended model attention path have rotary positions applied. Their raw dot product is not the model's actual attention score. The cache code selects indices from batch element zero and applies them to the entire batch (2915–2949); either implement per-sequence selection or explicitly restrict batch size to one.

Arbitrary eviction also needs retained absolute positions and masks that follow the gathered keys. Passing increasing `position_ids` alone is insufficient. Dropping K/V inside `update()` during multi-token prefill can make the existing causal mask inconsistent with the shortened or non-contiguous keys. Test this directly instead of assuming a DynamicCache subclass works across Transformers versions. [Hugging Face's cache documentation](https://huggingface.co/docs/transformers/v4.50.0/en/cache_explanation) describes the required mask-length and cache-position coordination; the actual implementation must target a pinned supported version.

The later benchmark payload includes `output_text` and `token_ids` at the top level (3226–3227), but today's benchmark schema forbids them. It also hardcodes hit_rate=1.0, can label FP32 CPU execution as float16, forces a positive decode count when there may be zero decode steps, and lacks explicit accelerator timing synchronization. These are static findings in proposed code; it was not installed or executed here. Store tokens separately, validate records, report actual dtype, and define timing boundaries.

### Quantization and hardware claims

The proposed `apply_aimet_style_cle_and_adaround` helper (3280–3324) is a local weight-rounding experiment. It does not by itself implement model-wide CLE, activation calibration, ONNX export, packed INT4 storage, or HTP execution. The following module ends after configuration and hardware-status helpers; it does not complete the advertised export-and-run pipeline. [AIMET's AdaRound tutorial](https://quic.github.io/aimet-pages/AimetDocs/tutorials/notebooks/torch/quantization/adaround.html) separates rounding, quantization simulation, frozen parameter encodings, calibration, and evaluation. Treat each as a verifiable artifact-producing step.

An environment variable naming a device log does not establish a device run (3353–3371). An execution provider in a session's provider list does not establish full graph offload. Record actual graph placement, runtime fallback, successful inference, and device provenance. CPU wrappers may be legitimate but must be reported. Use version-matched [ONNX Runtime QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) to distinguish strict partition fallback controls from runtime provider fallback and wrapper-node requirements.

Do not equate GGUF Q4_K_M with AIMET W4A8, or a four-bit weight format with an INT8 KV cache. These describe different storage, activation, cache, and execution properties.

### Memory arithmetic versus memory placement

For batch one, dense K+V payload bytes are `2 × layers × kv_heads × resident_tokens × head_dim × bytes_per_element` before metadata and temporary buffers. Checkpoint configs confirm SmolLM2-135M uses 30 layers/3 KV heads and Qwen2.5-0.5B uses 24 layers/2 KV heads, both head dimension 64. See the publisher's [SmolLM2 config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/config.json) and [Qwen config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json); pin revisions when implementing.

At 2,048 resident tokens, FP16 payloads are 45 MiB and 24 MiB. At 512 tokens and actual INT8 KV storage, the payloads would be 5.625 MiB and 3 MiB. These are calculated payloads, not measured resident allocations. INT8 scale/zero-point metadata, alignment, scratch buffers, activation tiles, double buffering, and the actual backend allocator also matter. A calculated payload below a hypothetical 8 MiB budget does not prove VTCM residency, absence of DRAM traffic, or a speedup. The supplied Python cache stores floating-point tensors and does not implement the INT8 KV storage needed for those estimates.

Do not adopt universal 8 MiB VTCM, DSP address limits, HTP architecture IDs, shard counts, or bandwidth numbers from this synthesis as cross-device constants. Verify them for the actual SoC, firmware, SDK, and allocation mechanism. Hub affiliations and the many peripheral paper claims were not exhaustively verified in this review and are not used as implementation requirements.

### Proposed assistant implementation and evidence guarantees

The first “dense” retriever hashes character trigrams into 256 dimensions (1449–1475); it does not load MiniLM or CLIP. That can be a lexical baseline, but cannot support the claimed learned multimodal retrieval feature.

The later store grants special access to the caller-supplied string `system` (3474 and 3532), defaults reads to that identity, and relaxes overwrite checks on deleted records. That is not an authenticated tenant boundary. A Python lock on one instance also does not settle multiple connections/processes competing to update the same record.

The later verifier hardcodes `source_revision=fixture-v1`, maps dimensional claims to `<=` regardless of the parsed operator, and may recover numeric observations from rounded display text (3857–3879). Keep source values and predicates as data throughout. Existing v1 schemas cannot honestly represent strict `<`, `>`, negation, or a no-candidate decision with no item ID without a design change. Add an explicit migration, not fabricated placeholders or changed operator meanings.

The “zero-overlap” quantum-charger example still shares the word chair with a chair catalog. Retrieval can return chairs; the unsupported required capability must remain UNKNOWN and block a positive decision. Neither a score threshold nor JSON Schema guarantees that all user constraints were parsed or that source facts are true.

The claimed 68.4%, 84.2%, 91.5%, and 93.8% retention results and “measured local” labels in paragraphs 3914–3974 lack supporting run records in this workspace. Preserve them only as unverified claims from the source. Zero false SUPPORTED examples in a finite test set is an observed rate with uncertainty, never a universal zero-error guarantee.

## Recommended architecture and bounded research scope

Keep `llm-inference` as a visibly labelled kernel lab plus a separately packaged real-model runtime. Use a pinned upstream model implementation for the first reference; do not build two independent full inference engines at once. Introduce compression through a small tested adapter after baseline parity.

Keep `edge-rag-demo` independent of generation: `request → parsed requirements → retrieval → authorized versioned facts → semantic checks → decision → extractive answer`. Optional model generation may phrase verified facts but may not create predicates, change the verdict, or invent citations. Return unparsed/ambiguous requirements explicitly.

Make `qedge-common` the source of versioned data contracts and semantic validators. Record model/data revisions, execution identities, and evidence provenance. Preserve v1 artifacts and migrate new formats deliberately.

Prioritize three falsifiable studies: whether a cache selector improves quality at equal resident-byte budgets; whether quantization improves latency/memory at an acceptable measured quality loss; and whether hybrid retrieval improves held-out relevance without increasing false positive decisions. Defer speculative decoding, episodic memory, sub-4-bit training, custom HMX kernels, and OS QoS work until these foundations work. Each adds a separate failure surface and requires evidence beyond this research document.

## How to use the implementation prompts

Start with [START-HERE.md](START-HERE.md). The prompts insert repair gates before the existing stages and preserve their useful acceptance criteria. They do not authorize overwriting concurrent OpenCode edits. Each prompt requires a reproducible handoff and stops after its bounded scope so the next stage does not conceal a failed prerequisite.
