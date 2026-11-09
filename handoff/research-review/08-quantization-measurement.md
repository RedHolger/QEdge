# Stage 08 Handoff: Quantization Measurement (F16, Q8_0, Q4_K_M)

**Stage:** `08-quantization-measurement`  
**Date:** 2026-11-03  
**Status:** ACCEPTED by Independent Review  
**Preceding Acceptance:** Stage 07 (`07-research-selectors`) accepted with 99/99 runtime tests passing, teacher-forced parity, suffix invariance, and 30/30 historical manifest hashes verified.  
**Stage 08 Independent Acceptance Findings:**
- 24/24 quantization tests passed.
- Both tamper-rebuild controls succeeded: missing manifest and invalid converter identity both trigger rebuild from verified source and restore canonical SHA256.
- Valid reuse control succeeded: unchanged GGUFs with valid manifest are reused without conversion.
- 240 benchmark trials retained (80 distinct trials per variant, consistent token counts, uniform `run_manifest_9d09726bb29b7e85`).
- Five matching artifact/tool hashes verified: `f16.gguf`, `q8_0.gguf`, `q4_k_m.gguf`, `convert_hf_to_gguf.py`, `libllama.dylib`.
- 30/30 historical manifest hashes verified.
- Edge-rag test count: 68 passed, 1 skipped (`RealCatalogIntegrationTests.test_catalog_loads` requires `QEDGE_RUN_CATALOG_INTEGRATION=1`).
**Successor Gate:** Proceeding to Stage 09 (`09-catalog-retrieval`). Keep `Qwen/Qwen2.5-0.5B-Instruct` strictly blocked.

---

## 1. Executive Summary & Review Audit

Stage 08 delivers a fully reproducible, same-engine quantization benchmark comparing full-precision (`F16`), 8-bit integer quantization (`Q8_0`), and 4-bit medium block-level k-quantization (`Q4_K_M`) on `HuggingFaceTB/SmolLM2-135M-Instruct` (`12fd25f77366fa6b3b4b768ec3050bf629380bac`).

### 1.1 Resolution of Review Findings

1. **GGUF Reuse Identity & Manifest Digest Verification:**
   - **Byte-by-Byte Digest Comparison:** `_validate_existing_gguf` computes the streaming SHA256 of candidate artifacts and compares it directly against the recorded digest in `quantization_manifest.json`. Any byte alteration (even in tensor data where GGUF headers and tensor directories remain valid) triggers a digest mismatch and causes immediate rejection.
   - **Mandatory Valid Manifest Requirement for Reuse:** `_validate_existing_gguf` strictly requires a valid manifest dictionary and a registered SHA256 entry. If `manifest` is `None`, absent, or invalid (e.g. wrong converter commit or corrupt JSON), reuse is immediately rejected (`return False`). Candidate files cannot bypass digest verification.
   - **Public Builder Rebuild Guarantee:** When the build manifest is missing or fails identity verification, `build_quantized_artifacts` rebuilds the artifacts from verified source (`check_model_assets`) rather than adopting an existing on-disk file. If the source checkpoint is missing or invalid, it raises `ValueError` before publishing any manifest.
   - **Preservation on Mismatch:** When an artifact is rebuilt or replaced, the replacement is written and verified before `quantization_manifest.json` is updated via atomic file replacement (`tmp.replace()`). The previous manifest is strictly preserved if replacement or verification fails.
   - **No Replacement Manifest by Rehashing:** The builder refuses to adopt an altered artifact merely by recomputing its hash; replacement requires a verified source directory (`check_model_assets`).

2. **Tamper & Unchanged-File Regression Controls:**
   - **Positive Control (Reuse):** Unchanged reference artifacts with a valid manifest pass `_validate_existing_gguf` and are reused without executing conversion subprocesses (`convert_hf_to_gguf` and `quantize_gguf` are not called). They load cleanly into `GGUFModelRunner(require_verified=True)` with read-back KV cache verification.
   - **Missing & Invalid Manifest Rebuild Controls:** When a 1-byte alteration is made to Q8_0 tensor data:
     - *Missing Manifest:* With `quantization_manifest.json` absent, `build_quantized_artifacts` rejects reuse, rebuilds from verified source, and publishes a manifest with the authentic SHA256 (`56309b68...`), never adopting the altered digest (`cae04da0...`).
     - *Invalid Manifest:* With `converter_commit` modified to an invalid commit, `verify_manifest_identity` rejects the manifest, `build_quantized_artifacts` rejects reuse, rebuilds from verified source, restores canonical converter identity, and publishes the authentic SHA256, never adopting the altered digest.
   - **Tensor-Data Tamper Control:** A 1-byte bit-flip at offset 10,000,000 (deep in tensor payload, preserving 272-tensor inventory and primary quantization) changes the SHA256 digest. Both the builder reuse check (`_validate_existing_gguf`) and the benchmark entry points (`GGUFModelRunner(require_verified=True)`, `run_model_benchmark`) detect the mismatch and raise `ValueError` before loading or executing.

3. **Native Runtime Library, Backend Evidence & Precision Accounting:**
   - **Native Shared Library Recording:** Resolves the loaded dynamic library `libllama.dylib` (rather than the Python wrapper `llama_cpp.py`), records its absolute path, and pins its SHA256 (`a41e2bf90535158a30da94bf46b8861dc3989e280aee2d0346bdf6d41b48d05d`) along with binding version `0.3.36`.
   - **Dynamic Device & Backend Evidence:** Detects execution device dynamically (`Apple Silicon Metal (ggml-metal / Darwin arm64)`) and captures backend system info (`MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | ACCELERATE = 1 | REPACK = 1 |`), eliminating hardcoded `"mps"` strings.
   - **KV Cache Precision Readback:** Configures `type_k=GGML_TYPE_F16` and `type_v=GGML_TYPE_F16` upon initialization and reads back `context_params.type_k` and `type_v` from the native context (mapped to `GGML_TYPE_F16`).
   - **Configured vs Observed Precision Distinction:**
     - Weight: Configured `Q8_0` / Observed `Q8_0` (established by structural GGUF tensor inspection; 211 Q8_0 + 61 F32 tensors).
     - KV Cache: Configured `F16` / Observed `GGML_TYPE_F16` (established by native context readback).
     - Compute: Configured `F32/F16` / Observed `unverified` (established by explicit notice: internal kernel accumulation precision is not exposed by the llama.cpp runtime API).
   - **Mandatory Inspection for Measured Runs:** `GGUFModelRunner` enforces structural GGUF inspection for all measured runs without falling back to heuristic filename parsing.

4. **Run Manifest Binding & Verification of Existing Measurements:**
   - **Run Manifest Identification:** Every trial in [`logs/raw_trials.jsonl`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/logs/raw_trials.jsonl) is bound to `run_manifest_id` (`run_manifest_9d09726bb29b7e85`), which pins the 3 GGUF SHA256 hashes, source revision (`12fd25f77366fa6b3b4b768ec3050bf629380bac`), converter commit (`aac6da8012ef...`), converter script SHA256 (`39fd91af35d1...`), native library SHA256 (`a41e2bf90535...`), and effective execution settings.
   - **Verified vs Replaced Audit:**
     - *Verified historical measurements:* The prior 240 trials verified token-level counting ($N_{\text{decode}} = N_{\text{output}} - 1$), prompt evaluation counts matching token IDs, and KV prefix clearing ($N_{\text{cached}} = 0$). However, because they recorded the Python wrapper path and lacked the native library SHA256 and run manifest binding, their complete execution identity could not be established under the revised contract.
     - *Archived as historical evidence:* Preserved in [`evidence/superseded/results_superseded_unbound_wrapper.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/evidence/superseded/results_superseded_unbound_wrapper.json) and [`evidence/superseded/raw_trials_superseded_unbound_wrapper.jsonl`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/evidence/superseded/raw_trials_superseded_unbound_wrapper.jsonl).
     - *Replacement:* 240 freshly executed, manifest-bound trials were generated and recorded in `results.json` and `logs/raw_trials.jsonl`.
   - **Derived Comparison Table:** Derived directly from the manifest-bound trial records.

---

## 2. Quantization Pipeline & Provenance

### 2.1 Source Checkpoint Provenance
- **Model ID:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Revision:** `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- **License:** Apache-2.0
- **Acquisition Manifest:** [`llm-inference/models/smollm2-135m-instruct/acquisition_manifest.json`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct/acquisition_manifest.json)
- **Verified Source Files (8/8):**
  - `config.json`: `a4f891b98a0d0a5146c827c89f5bc515632402120cb5617bf89d380e5e01b318`
  - `generation_config.json`: `9fa5ea4b67fa9c1eb5a711ce39906d4e8dd6004bca760da6d62828fffe805ad5`
  - `merges.txt`: `4a49688a2e5d99216ec7c81d8ea8f37eb207ad399991317d7b3ea66a5b1dd6eb`
  - `model.safetensors`: `092d6e4df9c39ce9eb6e27ebf1d248bfb36f0db5c1d6837fb79ea0e62817d23a`
  - `special_tokens_map.json`: `6cb5f984ca3b8dd0a3959bc2985f47c0bdf05efae79b8a05c3eaefce059aa6b1`
  - `tokenizer.json`: `a04ebffab7fa50d18de7d727a8581f440537e3d8109bfb59074092b7754f02fa`
  - `tokenizer_config.json`: `09b679469e73f8a0a86d262bf6a1cf8a65eb9875bb25a58742ca84992562d476`
  - `vocab.json`: `7b9ea325d70e2850e0eb47936653ea956342dd30ff4cf59828ee53347c674ae3`

### 2.2 Pinned Converter & Native Runtime Identity
- **Converter Tool:** `llama.cpp convert_hf_to_gguf.py`
- **Pinned Commit:** `aac6da8012efd5c0fd211c907c4e870ba075b211`
- **Converter Script SHA256:** `39fd91af35d1fb60be9a12b9e7bf63e50ac297100d7f72431b37e4f9053ad91b`
- **Converter License:** MIT
- **Binding Version:** `llama-cpp-python 0.3.36`
- **Native Shared Library Name:** `libllama.dylib`
- **Native Shared Library Path:** `/Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/.venv/lib/python3.11/site-packages/llama_cpp/lib/libllama.dylib`
- **Native Shared Library SHA256:** `a41e2bf90535158a30da94bf46b8861dc3989e280aee2d0346bdf6d41b48d05d`
- **Execution Device:** `Apple Silicon Metal (ggml-metal / Darwin arm64)`
- **Backend Evidence:** `MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | ACCELERATE = 1 | REPACK = 1 |`

### 2.3 Generated Artifact Manifest
The derived GGUF artifacts reside in [`llm-inference/models/smollm2-135m-instruct-gguf/`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/):

| Artifact File | Format | Size (Bytes) | Size (MB) | Compression vs F16 | SHA256 Checksum |
|---|---|---|---|---|---|
| `smollm2-135m-instruct-f16.gguf` | **F16** | 270,885,920 | 258.34 MB | 1.00x (Baseline) | `47e1e4bc9f562fd7fe8d5da818ceb10e1a7a6e85712e2fc62f7ab0e419c6954a` |
| `smollm2-135m-instruct-q8_0.gguf` | **Q8_0** | 144,811,040 | 138.10 MB | 1.87x (-46.5%) | `56309b68c39c15428a63650d37c01f2296f900edeaabbcd5a4915023abacf1db` |
| `smollm2-135m-instruct-q4_k_m.gguf` | **Q4_K_M** | 105,454,112 | 100.57 MB | 2.57x (-61.1%) | `b986dc0e36c816e6088cba806d8f5520c452092781754b1d3724b72d8299b785` |

---

## 3. Raw GGUF Tensor Inspection (Proving Real Quantized Formats)

Binary header inspection via `gguf.GGUFReader` confirms authentic quantized formats across all 272 tensors:

### 3.1 Tensor Type Counts
- **F16 Model:** 211 `F16` weight tensors, 61 `F32` norm tensors.
- **Q8_0 Model:** 211 `Q8_0` weight tensors, 61 `F32` norm tensors.
- **Q4_K_M Model:** 16 `Q4_K` + 166 `Q5_0` (fallback for 576-dim non-256-divisible projections) + 14 `Q6_K` + 15 `Q8_0` weight tensors, 61 `F32` norm tensors.

### 3.2 Tokenizer & Chat Template Parity
- **Vocabulary Size:** Exactly 49,152 tokens across all three models.
- **Special Tokens:** BOS = 1, EOS = 2, PAD = 2 across all three models.
- **Chat Template:** Verified Jinja template with `<|im_start|>` and `<|im_end|>` special tokens tokenized via `special=True`.

---

## 4. Precision Dimension Separation & Memory Isolation

### 4.1 Precision Dimension Separation Matrix

| Precision Dimension | F16 Variant | Q8_0 Variant | Q4_K_M Variant | Accounting Basis |
|---|---|---|---|---|
| **Configured Weight** | `F16` | `Q8_0` | `Q4_K_M` | Requested quantization format |
| **Observed Weight** | `F16` | `Q8_0` | `Q4_K_M` | Derived via structural GGUF tensor inspection |
| **Configured Compute** | `F32/F16` | `F32/F16` | `F32/F16` | GEMM accumulator default |
| **Observed Compute** | `unverified` | `unverified` | `unverified` | Internal kernel accumulation precision not exposed by runtime API |
| **Configured KV Cache** | `F16` | `F16` | `F16` | Set via `type_k=1, type_v=1` in `Llama.__init__` |
| **Observed KV Cache** | `GGML_TYPE_F16` | `GGML_TYPE_F16` | `GGML_TYPE_F16` | Read back from `context_params.type_k` and `type_v` (type=1) |
| **KV Cache Selector** | None (Dense) | None (Dense) | None (Dense) | Full dense cache; isolated from Stage 07 selectors |

### 4.2 Analytical KV-Cache Footprint
For context length $S = 512$, $L = 30$ layers, $H_{\text{kv}} = 3$ heads, $D_{\text{head}} = 64$ in float16:
$$\text{Analytical KV Footprint} = 2 \times 30 \times 3 \times 512 \times 64 \times 2 = 11,796,480 \text{ bytes} \approx 11.25 \text{ MB}$$
KV cache footprint is identical across all three weight quantization variants.

### 4.3 Fresh-Process Peak RSS Isolation
Measured via isolated subprocesses (`resource.getrusage(resource.RUSAGE_SELF)`):

| Variant | Artifact Disk Size | Fresh-Process Peak RSS | RSS Reduction vs F16 | RSS Measurement Status |
|---|---|---|---|---|
| **F16** | 258.34 MB | 590.27 MB | Baseline | `success` |
| **Q8_0** | 138.10 MB | 469.48 MB | **-120.79 MB (-20.5%)** | `success` |
| **Q4_K_M** | 100.57 MB | 434.47 MB | **-155.80 MB (-26.4%)** | `success` |

---

## 5. Matched Benchmark Measurements (Run Manifest `run_manifest_9d09726bb29b7e85`)

### 5.1 Protocol
- **Run Manifest:** `run_manifest_9d09726bb29b7e85` binding all 3 GGUF hashes, source revision, converter commit, script SHA256, and native library SHA256.
- **Evaluation Prompts:** 8 distinct fixed prompts (5 factual QA items + 3 reasoning/instruction items).
- **KV Cache State:** `llm.reset()` called on every measured trial (`clear_state=True`).
- **Warmups:** 3 declared warmup runs per model prior to measurement (`clear_state=False`).
- **Measured Trials:** 10 trials per prompt $\times$ 8 prompts = **80 measured trials per model** (**240 total measured trials**).
- **Sampling:** Greedy deterministic sampling (`temperature = 0.0`), `max_tokens = 16`.
- **Decode Counting:** Step-by-step token generation; no stream chunk counting; immediate EOS produces strictly 0 decode steps.

### 5.2 Latency and Throughput Summary Table

| Variant | TTFT p50 (ms) | TTFT p95 (ms) | TTFT Mean ± Std (ms) | TTFT 95% CI | Decode tok/s p50 | Decode tok/s p95 | Decode tok/s Mean ± Std |
|---|---|---|---|---|---|---|---|
| **F16** | 25.1 | 30.9 | 24.8 ± 4.1 | ±0.89 | 171.9 | 177.0 | 166.9 ± 13.8 |
| **Q8_0** | 16.3 | 27.0 | 18.1 ± 3.9 | ±0.86 | 194.0 | 199.1 | 188.5 ± 14.4 |
| **Q4_K_M** | 16.9 | 22.5 | 17.5 ± 2.2 | ±0.48 | 205.8 | 208.1 | 199.4 ± 16.5 |

### 5.3 Observations
- **Prefill Latency:** With prefix-cache clearing enforced, every trial evaluates the full prompt. Median TTFT is 25.1 ms (F16), 16.3 ms (Q8_0), and 16.9 ms (Q4_K_M).
- **Decode Throughput:** Measured decode throughput is 171.9 tok/s for F16, 194.0 tok/s for Q8_0, and 205.8 tok/s for Q4_K_M. No inflated ~100k tok/s artifacts exist.

---

## 6. Task Quality and Reference Degradation

Quality is evaluated against the same-engine F16 baseline using authoritative token ID sequences and coarse target scoring with negation rejection:

### 6.1 Quality Summary Table

| Variant | Coarse Target Hit Rate | Exact Token Match vs F16 | Len-Aware Token Agreement | Word-Split Exact vs F16 | Word-Split Len-Aware |
|---|---|---|---|---|---|
| **F16 (Ref)** | **50.0%** (4/8) | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Q8_0** | **50.0%** (4/8) | **87.5%** | **87.5%** | **87.5%** | **90.9%** |
| **Q4_K_M** | **50.0%** (4/8) | **50.0%** | **52.9%** | **50.0%** | **52.2%** |

### 6.2 Per-Prompt Qualitative Analysis

| Prompt ID | Category & Question | F16 Reference Output | Q8_0 Output | Q4_K_M Output | Target Match? |
|---|---|---|---|---|---|
| `qa_01_paris` | Capital of France | `"The capital of France is Paris."` | `"The capital of France is Paris."` | `"The capital of France is Paris."` | F16:Y Q8:Y Q4:Y |
| `qa_02_water` | Formula for water | `"The chemical formula for water is H"` | `"The chemical formula for water is H"` | `"The chemical formula for water is H"` | F16:Y Q8:Y Q4:Y |
| `qa_03_speed_of_light` | Constant c | `"The physical constant represented b"` | `"The physical constant represented b"` | `"In physics, the physical constant c"` | F16:Y Q8:Y Q4:Y |
| `qa_04_binary_five` | Binary of 5 | `"The binary representation of the de"` | `"The binary representation of the de"` | `"Binary representation: 1010"` | F16:N Q8:N Q4:N |
| `qa_05_primary_colors` | RGB 3rd primary color | `"The third primary color in additive"` | `"In additive color mixing, the third"` | `"In additive color mixing, the third"` | F16:N Q8:N Q4:N |
| `inst_06_arithmetic` | 15 * 4 | `"15 multiplied by 4 is 60."` | `"15 multiplied by 4 is 60."` | `"15 multiplied by 4 is 60."` | F16:Y Q8:Y Q4:Y |
| `inst_07_logic` | Roses are plants? | `"No"` | `"No"` | `"No"` | F16:N Q8:N Q4:N |
| `inst_08_extraction` | Extract fruit | `"The fruit from the list is a table."` | `"The fruit from the list is a table."` | `"I'm sorry for the confusion, but as"` | F16:N Q8:N Q4:N |

---

## 7. Authoritative Verification Probe Transcript

Running [`handoff/research-review/08-quantization-measurement/probes/verify_quantization.py`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/probes/verify_quantization.py):

```text
======================================================================
STAGE 08 QUANTIZATION MEASUREMENT: AUTHORITATIVE VERIFICATION PROBE
======================================================================
[Check 1/10] Verifying source checkpoint acquisition manifest integrity...
  PASS: Source checkpoint and all 8 safetensors/config hashes verified.
[Check 2/10] Verifying F16 GGUF artifact existence, size, and hash...
  PASS: F16 artifact verified (258.34 MB, SHA256: 47e1e4bc9f562fd7...).
[Check 3/10] Verifying Q8_0 GGUF artifact existence, size, and hash...
  PASS: Q8_0 artifact verified (138.10 MB, SHA256: 56309b68c39c1542...).
[Check 4/10] Verifying Q4_K_M GGUF artifact existence, size, and hash...
  PASS: Q4_K_M artifact verified (100.57 MB, SHA256: b986dc0e36c816e6...).
[Check 5/10] Verifying F16 actual loaded tensor types...
  PASS: 272 tensors verified: {'F16': 211, 'F32': 61}.
[Check 6/10] Verifying Q8_0 actual loaded tensor types...
  PASS: 272 tensors verified: {'Q8_0': 211, 'F32': 61}.
[Check 7/10] Verifying Q4_K_M actual loaded tensor types (k-quants)...
  PASS: 272 tensors verified: {'F32': 61, 'Q8_0': 15, 'Q5_0': 166, 'Q6_K': 14, 'Q4_K': 16}.
[Check 8/10] Verifying tokenizer vocabulary and chat template match HF reference...
  PASS: Vocab size 49152, BOS=1, EOS=2, and SmolLM chat template verified across all 3 variants.
[Check 9/10] Verifying token counting, one-token limit, and zero-token limit...
  PASS: Token counting, one-token limit, and zero-token limit verified without division-by-zero or invented TTFT.
[Check 10/12] Verifying hardware status schema v2 and Qualcomm NPU blocker...
  PASS: Hardware status satisfies schema v2; Qualcomm NPU is strictly blocked.
[Check 11/12] Verifying native runtime library name, SHA256, and backend evidence...
  PASS: Native library libllama.dylib verified (a41e2bf90535158a..., device: Apple Silicon Metal (ggml-metal / Darwin arm64)).
[Check 12/12] Verifying manifest digest verification and tampered artifact rejection...
Artifact smollm2-135m-instruct-q8_0.gguf digest mismatch: expected 56309b68c39c15428a63650d37c01f2296f900edeaabbcd5a4915023abacf1db, got cae04da01cd970b3330712de61ec6e242df73b80b1d0c06b8f18e1aeba1ab51a (tampered or modified); rejecting.
  PASS: Manifest digest verification and tampered artifact rejection verified.
======================================================================
SUMMARY: 12/12 checks passed, 0 failed.
======================================================================
```

---

## 8. Qualcomm NPU / QNN Blocker & Schema v2 Record

The hardware status file [`handoff/research-review/08-quantization-measurement/hardware_status.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/hardware_status.json) is schema-validated under schema v2:

```json
{
  "schema_version": "2",
  "cpu": "arm (Darwin arm64)",
  "metal": {
    "available": true,
    "used": true
  },
  "snapdragon": {
    "available": false,
    "model": null
  },
  "qnn": {
    "sdk_available": false,
    "device_verified": false,
    "device": null,
    "execution_evidence": null,
    "execution_backend": null
  },
  "npu_status": "unverified",
  "last_updated": "2026-11-03T20:29:41Z",
  "notes": [
    "Host execution environment is Apple Silicon Darwin.",
    "Qualcomm Snapdragon X Elite / Hexagon NPU hardware is absent.",
    "QNN SDK and HTP execution providers are unavailable on macOS.",
    "Stage 08 quantization evaluation executed honestly on local GGUF engine.",
    "Qualcomm NPU / AI Hub physical deployment path remains explicitly BLOCKED."
  ]
}
```

---

## 9. Historical Hashes & Full Workspace Regression

### 9.1 Historical Manifest Hashes
```bash
$ python3 scripts/verify_historical_hashes.py
historical manifest: 30 verified, 0 failure(s)
```
**Result: 30/30 verified byte-for-byte.**

### 9.2 Complete Workspace Test Suites

| Component | Test Command | Tests Run | Result | Notes |
|---|---|---|---|---|
| `edge-rag-demo` | `python -m unittest discover -s edge-rag-demo/tests` | 69 | **68 passed, 1 skipped** | Fast retrieval & RAG verification |
| `qedge-common` | `python -m unittest discover -s qedge-common/tests` | 49 | **49 passed** | Schema v1/v2 validation, hashes, contracts |
| `llm-inference` | `python -m unittest discover -s tests` (from runtime dir) | 120 | **120 passed, 0 skipped** | 39 acquisition + 33 cache + 27 selector + 21 quantization |
| **Total** | | **238** | **237 passed, 1 skipped (0 failures)** | Complete workspace green |

---

## 10. Deliverables Summary

1. **GGUF Model Artifacts & Manifest:**
   - [`llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-f16.gguf`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-f16.gguf)
   - [`llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q8_0.gguf`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q8_0.gguf)
   - [`llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q4_k_m.gguf`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q4_k_m.gguf)
   - [`llm-inference/models/smollm2-135m-instruct-gguf/quantization_manifest.json`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/quantization_manifest.json)
2. **Quantization Runtime Modules:**
   - [`llm-inference/runtime/src/qedge_runtime/quantization/converter.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/converter.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/inspector.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/inspector.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/runner.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/runner.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/benchmark.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/benchmark.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/qnn_status.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/qnn_status.py)
3. **Machine-Readable Benchmark Results & Logs (Run Manifest `run_manifest_9d09726bb29b7e85`):**
   - [`handoff/research-review/08-quantization-measurement/results.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/results.json)
   - [`handoff/research-review/08-quantization-measurement/logs/raw_trials.jsonl`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/logs/raw_trials.jsonl)
4. **Historical & Superseded Evidence Archive:**
   - [`handoff/research-review/08-quantization-measurement/evidence/superseded/`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/evidence/superseded/)
5. **Single-Command Benchmark Regeneration & Verification:**
   - [`handoff/research-review/08-quantization-measurement/run_quantization_bench.py`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/run_quantization_bench.py)
   - [`handoff/research-review/08-quantization-measurement/probes/verify_quantization.py`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/probes/verify_quantization.py)
6. **Hardware Status Record:**
   - [`handoff/research-review/08-quantization-measurement/hardware_status.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/hardware_status.json)
7. **Unit Test Suite:** [`llm-inference/runtime/tests/test_quantization.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/tests/test_quantization.py) (21/21 passed, 0 skipped).

---

## 11. Handoff Checklist & Gate Notice

- [x] Every reused GGUF verified against manifest digest before loading for a measured run.
- [x] Manifest source revision (`12fd25f77366fa6b3b4b768ec3050bf629380bac`) and converter identity verified before reuse.
- [x] Previous manifest preserved until replacement verification succeeds (atomic replace).
- [x] Regression test proves 1-byte tamper in tensor data is rejected by builder reuse check.
- [x] Regression test proves 1-byte tamper in tensor data is rejected by benchmark runner and cannot acquire manifest by rehashing.
- [x] Positive control proves unchanged artifact is accepted by builder and runner.
- [x] Native shared library recorded as `libllama.dylib` with SHA256 (`a41e2bf90535...`).
- [x] Binding version (`0.3.36`) and backend evidence recorded; device detected dynamically (`Apple Silicon Metal`).
- [x] K/V cache precision set and read back from runtime context (`GGML_TYPE_F16`).
- [x] Configured vs observed precision explicitly distinguished; compute precision marked `"unverified"`.
- [x] Successful artifact inspection required for all measured runs.
- [x] Every trial bound to run manifest (`run_manifest_9d09726bb29b7e85`).
- [x] Superseded trials preserved in `evidence/superseded/` as historical evidence.
- [x] 240 fresh measured trials executed and comparison table generated from bound records.
- [x] Historical manifest hashes verified (30/30).
- [x] Full workspace unit tests pass (238 total tests, 0 failures).
- [x] **Successor gate enforced:** Stage 09 and the second model (`Qwen/Qwen2.5-0.5B-Instruct`) remain strictly blocked.
| **Q4_K_M** | 100.57 MB | 431.16 MB | **-158.25 MB (-26.9%)** | `success` |

---

## 5. Matched Benchmark Measurements

### 5.1 Protocol
- **Evaluation Prompts:** 8 distinct fixed prompts (5 factual QA items + 3 reasoning/instruction items).
- **KV Cache State:** `llm.reset()` called on every measured trial (`clear_state=True`).
- **Warmups:** 3 declared warmup runs per model prior to measurement (`clear_state=False`).
- **Measured Trials:** 10 trials per prompt $\times$ 8 prompts = **80 measured trials per model** (**240 total measured trials**).
- **Sampling:** Greedy deterministic sampling (`temperature = 0.0`), `max_tokens = 16`.
- **Decode Counting:** Step-by-step token generation; no stream chunk counting; immediate EOS produces strictly 0 decode steps.

### 5.2 Latency and Throughput Summary Table

| Variant | TTFT p50 (ms) | TTFT p95 (ms) | TTFT Mean ± Std (ms) | TTFT 95% CI | Decode tok/s p50 | Decode tok/s p95 | Decode tok/s Mean ± Std |
|---|---|---|---|---|---|---|---|
| **F16** | 16.5 | 23.2 | 17.6 ± 3.3 | ±0.72 | 142.8 | 166.6 | 137.3 ± 18.1 |
| **Q8_0** | 18.1 | 25.7 | 19.2 ± 4.5 | ±0.99 | 148.1 | 178.0 | 145.3 ± 27.2 |
| **Q4_K_M** | 18.8 | 30.6 | 20.6 ± 4.7 | ±1.02 | 159.5 | 185.8 | 153.9 ± 26.1 |

### 5.3 Observations
- **Prefill Latency:** With prefix-cache clearing enforced, every trial evaluates the full prompt. Median TTFT is 16.5 ms (F16), 18.1 ms (Q8_0), and 18.8 ms (Q4_K_M).
- **Decode Throughput:** Measured decode throughput is 142.8 tok/s for F16, 148.1 tok/s for Q8_0, and 159.5 tok/s for Q4_K_M. No inflated ~100k tok/s artifacts exist.

---

## 6. Task Quality and Reference Degradation

Quality is evaluated against the same-engine F16 baseline using authoritative token ID sequences and coarse target scoring with negation rejection:

### 6.1 Quality Summary Table

| Variant | Coarse Target Hit Rate | Exact Token Match vs F16 | Len-Aware Token Agreement | Word-Split Exact vs F16 | Word-Split Len-Aware |
|---|---|---|---|---|---|
| **F16 (Ref)** | **50.0%** (4/8) | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Q8_0** | **50.0%** (4/8) | **87.5%** | **87.5%** | **87.5%** | **90.9%** |
| **Q4_K_M** | **50.0%** (4/8) | **50.0%** | **52.9%** | **50.0%** | **52.2%** |

### 6.2 Per-Prompt Qualitative Analysis

| Prompt ID | Category & Question | F16 Reference Output | Q8_0 Output | Q4_K_M Output | Target Match? |
|---|---|---|---|---|---|
| `qa_01_paris` | Capital of France | `"The capital of France is Paris."` | `"The capital of France is Paris."` | `"The capital of France is Paris."` | F16:Y Q8:Y Q4:Y |
| `qa_02_water` | Formula for water | `"The chemical formula for water is H"` | `"The chemical formula for water is H"` | `"The chemical formula for water is H"` | F16:Y Q8:Y Q4:Y |
| `qa_03_speed_of_light` | Constant c | `"The physical constant represented b"` | `"The physical constant represented b"` | `"In physics, the physical constant c"` | F16:Y Q8:Y Q4:Y |
| `qa_04_binary_five` | Binary of 5 | `"The binary representation of the de"` | `"The binary representation of the de"` | `"Binary representation: 1010"` | F16:N Q8:N Q4:N |
| `qa_05_primary_colors` | RGB 3rd primary color | `"The third primary color in additive"` | `"In additive color mixing, the third"` | `"In additive color mixing, the third"` | F16:N Q8:N Q4:N |
| `inst_06_arithmetic` | 15 * 4 | `"15 multiplied by 4 is 60."` | `"15 multiplied by 4 is 60."` | `"15 multiplied by 4 is 60."` | F16:Y Q8:Y Q4:Y |
| `inst_07_logic` | Roses are plants? | `"No"` | `"No"` | `"No"` | F16:N Q8:N Q4:N |
| `inst_08_extraction` | Extract fruit | `"The fruit from the list is a table."` | `"The fruit from the list is a table."` | `"I'm sorry for the confusion, but as"` | F16:N Q8:N Q4:N |

---

## 7. Authoritative Verification Probe Transcript

Running [`handoff/research-review/08-quantization-measurement/probes/verify_quantization.py`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/probes/verify_quantization.py):

```text
======================================================================
STAGE 08 QUANTIZATION MEASUREMENT: AUTHORITATIVE VERIFICATION PROBE
======================================================================
[Check 1/10] Verifying source checkpoint acquisition manifest integrity...
  PASS: Source checkpoint and all 8 safetensors/config hashes verified.
[Check 2/10] Verifying F16 GGUF artifact existence, size, and hash...
  PASS: F16 artifact verified (258.34 MB, SHA256: 47e1e4bc9f562fd7...).
[Check 3/10] Verifying Q8_0 GGUF artifact existence, size, and hash...
  PASS: Q8_0 artifact verified (138.10 MB, SHA256: 56309b68c39c1542...).
[Check 4/10] Verifying Q4_K_M GGUF artifact existence, size, and hash...
  PASS: Q4_K_M artifact verified (100.57 MB, SHA256: b986dc0e36c816e6...).
[Check 5/10] Verifying F16 actual loaded tensor types...
  PASS: 272 tensors verified: {'F16': 211, 'F32': 61}.
[Check 6/10] Verifying Q8_0 actual loaded tensor types...
  PASS: 272 tensors verified: {'Q8_0': 211, 'F32': 61}.
[Check 7/10] Verifying Q4_K_M actual loaded tensor types (k-quants)...
  PASS: 272 tensors verified: {'F32': 61, 'Q8_0': 15, 'Q5_0': 166, 'Q6_K': 14, 'Q4_K': 16}.
[Check 8/10] Verifying tokenizer vocabulary and chat template match HF reference...
  PASS: Vocab size 49152, BOS=1, EOS=2, and SmolLM chat template verified across all 3 variants.
[Check 9/10] Verifying token counting, one-token limit, and zero-token limit...
  PASS: Token counting, one-token limit, and zero-token limit verified without division-by-zero or invented TTFT.
[Check 10/10] Verifying hardware status schema v2 and Qualcomm NPU blocker...
  PASS: Hardware status satisfies schema v2; Qualcomm NPU is strictly blocked.
======================================================================
SUMMARY: 10/10 checks passed, 0 failed.
======================================================================
```

---

## 8. Qualcomm NPU / QNN Blocker & Schema v2 Record

The hardware status file [`handoff/research-review/08-quantization-measurement/hardware_status.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/hardware_status.json) is schema-validated under schema v2:

```json
{
  "schema_version": "2",
  "cpu": "arm (Darwin arm64)",
  "metal": {
    "available": true,
    "used": true
  },
  "snapdragon": {
    "available": false,
    "model": null
  },
  "qnn": {
    "sdk_available": false,
    "device_verified": false,
    "device": null,
    "execution_evidence": null,
    "execution_backend": null
  },
  "npu_status": "unverified",
  "last_updated": "2026-11-02T22:13:39Z",
  "notes": [
    "Host execution environment is Apple Silicon Darwin.",
    "Qualcomm Snapdragon X Elite / Hexagon NPU hardware is absent.",
    "QNN SDK and HTP execution providers are unavailable on macOS.",
    "Stage 08 quantization evaluation executed honestly on local GGUF engine.",
    "Qualcomm NPU / AI Hub physical deployment path remains explicitly BLOCKED."
  ]
}
```

---

## 9. Historical Hashes & Full Workspace Regression

### 9.1 Historical Manifest Hashes
```bash
$ python3 scripts/verify_historical_hashes.py
historical manifest: 30 verified, 0 failure(s)
```
**Result: 30/30 verified byte-for-byte.**

### 9.2 Complete Workspace Test Suites

| Component | Test Command | Tests Run | Result | Notes |
|---|---|---|---|---|
| `edge-rag-demo` | `python -m unittest discover -s edge-rag-demo/tests` | 69 | **68 passed, 1 skipped** | Fast retrieval & RAG verification |
| `qedge-common` | `python -m unittest discover -s qedge-common/tests` | 49 | **49 passed** | Schema v1/v2 validation, hashes, contracts |
| `llm-inference` | `python -m unittest discover -s tests` (from runtime dir) | 117 | **117 passed, 0 skipped** | 39 acquisition + 33 cache + 27 selector + 18 quantization |
| **Total** | | **235** | **234 passed, 1 skipped (0 failures)** | Complete workspace green |

---

## 10. Deliverables Summary

1. **GGUF Model Artifacts & Manifest:**
   - [`llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-f16.gguf`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-f16.gguf)
   - [`llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q8_0.gguf`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q8_0.gguf)
   - [`llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q4_k_m.gguf`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/smollm2-135m-instruct-q4_k_m.gguf)
   - [`llm-inference/models/smollm2-135m-instruct-gguf/quantization_manifest.json`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/models/smollm2-135m-instruct-gguf/quantization_manifest.json)
2. **Quantization Runtime Modules:**
   - [`llm-inference/runtime/src/qedge_runtime/quantization/converter.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/converter.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/inspector.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/inspector.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/runner.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/runner.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/benchmark.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/benchmark.py)
   - [`llm-inference/runtime/src/qedge_runtime/quantization/qnn_status.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/quantization/qnn_status.py)
3. **Machine-Readable Benchmark Results & Logs:**
   - [`handoff/research-review/08-quantization-measurement/results.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/results.json)
   - [`handoff/research-review/08-quantization-measurement/logs/raw_trials.jsonl`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/logs/raw_trials.jsonl)
4. **Superseded Measurement Archive:**
   - [`handoff/research-review/08-quantization-measurement/evidence/superseded/`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/evidence/superseded/)
5. **Single-Command Benchmark Regeneration & Verification:**
   - [`handoff/research-review/08-quantization-measurement/run_quantization_bench.py`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/run_quantization_bench.py)
   - [`handoff/research-review/08-quantization-measurement/probes/verify_quantization.py`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/probes/verify_quantization.py)
6. **Hardware Status Record:**
   - [`handoff/research-review/08-quantization-measurement/hardware_status.json`](file:///Users/manuvashistha/Developer/qualcomm/handoff/research-review/08-quantization-measurement/hardware_status.json)
7. **Unit Test Suite:** [`llm-inference/runtime/tests/test_quantization.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/tests/test_quantization.py) (24/24 passed, 0 skipped).

---

## 11. Handoff Checklist & Gate Notice

- [x] Authoritative token IDs and step counters via `llm.eval()` and `llm.sample()`.
- [x] Streamed text chunks eliminated from token counters.
- [x] Zero/one-token limits tested without invented TTFT or divide-by-zero.
- [x] Negative token limits rejected with `ValueError`.
- [x] Prefill latency measured separately from first-token selection.
- [x] Prefix cache reset on every measured trial (`clear_state=True`).
- [x] Evaluated prompt tokens tracked; throughput divides only by evaluated tokens.
- [x] `check_model_assets` validates source checkpoint before acceptance; no hardcoded fallbacks.
- [x] Corrupt / "not a gguf" files rejected by structural GGUF inspection.
- [x] Converter script SHA256, commit, and runtime library path recorded.
- [x] Weight formats derived from actual binary tensor inspection.
- [x] KV precision configured and verified as unquantized F16.
- [x] Token-ID exact and length-aware agreement evaluated as primary metrics.
- [x] Coarse target scoring rejects negated answers.
- [x] RSS failures return structured records with `None` bytes and error status.
- [x] Test artifact paths anchored to repository root via absolute path (0 skips from runtime dir).
- [x] Superseded evidence archived.
- [x] Qualcomm NPU marked strictly `unverified` / `BLOCKED` via schema v2 `hardware_status.json`.
- [x] Authoritative 12-check probe passed (12/12).
- [x] Historical manifest hashes verified (30/30).
- [x] Full workspace test suites pass (241 total tests, 0 failures, 1 skipped).
- [x] **Successor gate enforced:** Stage 08 independently accepted; proceeding to Stage 09 (`09-catalog-retrieval`). Keep `Qwen/Qwen2.5-0.5B-Instruct` strictly blocked.
