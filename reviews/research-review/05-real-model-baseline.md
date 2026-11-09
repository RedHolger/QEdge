# Review: 05-real-model-baseline

Date: 31 October 2026  
Reviewed stage: `05-real-model-baseline`  
Review type: Independent review (evaluating initial implementation, settings validation repairs, provenance repairs, and reproduction probes)  
Review scope: Stage 05 reference model runtime under `llm-inference/runtime`, primary checkpoint `HuggingFaceTB/SmolLM2-135M-Instruct` pinned to commit `12fd25f77366fa6b3b4b768ec3050bf629380bac`, license recording (`Apache-2.0`), complete artifact hash verification across all 8 execution-affecting files, checkpoint architecture extraction (30 layers, 9 Q heads, 3 KV heads, head dimension 64, RoPE theta 100k, vocab 49,152), explicit model acquisition tool with download estimation, offline local-path execution mode, real prefill and autoregressive decode loop with full KV cache, strict monotonic position IDs, attention masks, output bounds (`max_new_tokens`), early EOS handling with multiple EOS ID support and non-falsy zero ID checks, state isolation without request leakage, greedy token parity (100%) and numerical logits agreement with documented tolerances (`atol=1e-4` on CPU float32, `atol=0.15` on MPS bfloat16) against unmodified `transformers` `model.generate()`, separate measurement boundaries (loading, formatting, tokenization, prefill, first-token selection, TTFT through first token, decode, detokenization) with accelerator synchronization, contract v2 `benchmark_result` emission with outputs separated from numeric metrics, factual execution metadata (seed policy, template digest identity, actual model parameter dtype), and expanded regression testbed (39 tests).

---

## Verdict

**ACCEPTED (bounded, pinned SmolLM2 baseline)**

The Stage 05 real-model baseline repairs satisfy all requirements specified in `docs/research-review/prompts/05-real-model-baseline.md` and subsequent review findings. Independent verification confirms:

- **Runtime Tests:** **39 / 39 passed**, including CPU float32 and Apple Silicon MPS bfloat16 parity.
- **Provenance & Manifest Integrity:** Manifest hash verification validates actual file digests against trusted expectations. Manifests with forged hashes (e.g., all zeros) are strictly rejected.
- **Loader Override Protection:** Extraneous execution-affecting loader assets (e.g., `chat_template.jinja`, unverified weight or config files) are rejected by provenance checks and cause `ModelRunner` to fail with `ValueError`.
- **Factual Reproducibility Metadata:** `seed_policy` is reported as `"none_deterministic_greedy"` with `seed=null` when unseeded; explicit integer seeds apply `torch.manual_seed()` and report factually. Template identity is derived from the loaded template's SHA256 digest (`sha256:872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788`). Actual dtype is derived directly from loaded model parameters.
- **Offline Generation:** Verified clean offline generation with network egress guarded and blocked at the socket layer.
- **Contract Emission:** Emits valid contract v2 `benchmark_result` with tokens and text held strictly in `outputs` and metrics limited to scalar numbers/nulls.
- **Suite Integrity:**
  - `llm-ctest`: **3 / 3 passed** (`llm_kv_cache`, `kernel_safety`, `qnn_stub_cpu`).
  - `edge-tests`: **68 passed, 0 failed, 1 expected skip** (`RealCatalogIntegrationTests.test_catalog_loads`).
  - `qedge-common-tests`: **49 passed, 0 failed, 0 skipped**.
  - `historical-hashes`: **30 / 30 verified**.
  - `schema-hashes`: **7 / 7 verified byte-for-byte**.

---

## Limits and Scope Boundaries

1. **Pinned SmolLM2 Reference Runtime Only:**
   This acceptance applies strictly to the tested, bounded reference runtime for `HuggingFaceTB/SmolLM2-135M-Instruct` under `llm-inference/runtime`.
2. **Second Model Deferral:**
   Per instructions, the secondary model (`Qwen/Qwen2.5-0.5B-Instruct`) remains deferred.
3. **No Cache Compression or Downstream Execution Claims:**
   This acceptance establishes no cache compression policy performance, QNN execution, or Snapdragon NPU deployment claims.
