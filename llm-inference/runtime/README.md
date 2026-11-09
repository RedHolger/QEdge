# QEdge Real-Model Reference Runtime

Separately packaged Python reference runtime for on-device language models, implementing real prefill and autoregressive decode with full KV cache, verified upstream parity, and schema-validated benchmark telemetry.

## Supported Model

- **Primary Checkpoint:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Pinned Git Revision:** `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- **License:** Apache-2.0
- **Architecture Settings (loaded from checkpoint):**
  - Architecture: `LlamaForCausalLM`
  - Layers: 30
  - Hidden size: 576
  - Intermediate size: 1536
  - Attention heads: 9
  - Key-Value heads: 3 (Grouped-Query Attention: 3 query heads per KV head)
  - Head dimension: 64
  - Max position embeddings: 8192
  - RoPE theta: 100,000.0
  - Vocab size: 49,152
  - Special tokens: BOS = 1, EOS = 2, PAD = 2

## Quick Start

```bash
# 1. Setup virtual environment and dependencies
make setup

# 2. Explicit model acquisition (~271.5 MB download)
make acquire

# 3. Run regression and parity test suite
make test

# 4. Interactive generation demo
make demo

# 5. Schema v2 contract benchmark
make bench
```

## Architecture & Implementation Principles

1. **Explicit Model Acquisition:**
   - Models are never downloaded implicitly during library import or test execution.
   - `python -m qedge_runtime.acquisition` estimates download size, fetches pinned revisions, verifies SHA256 hashes, and writes an `acquisition_manifest.json`.
   - In offline environments, the runtime loads strictly from the local path (`local_files_only=True`).

2. **Real Full-Cache Generation Loop:**
   - Real multi-token prefill with full causal attention.
   - Token-by-token autoregressive decoding updating `past_key_values` at each step.
   - Monotonic absolute position IDs and attention masks.
   - Strict output limit enforcement (`max_new_tokens`) and early EOS stopping.
   - Clean state isolation: no cross-request cache leakage.

3. **Greedy Parity & Logits Agreement:**
   - Evaluated against unmodified `transformers` `model.generate()`.
   - Verified 100% token-by-token greedy match across diverse prompts.
   - Numerical logits agreement within documented tolerances:
     - CPU `float32`: `atol=1e-4, rtol=1e-4`
     - Apple Silicon MPS `bfloat16`: `atol=0.15, rtol=1e-2` (reflecting 7-bit mantissa precision at logit scale ~30)

4. **Telemetry & Measurement Boundaries:**
   - Loading, tokenization, prefill, TTFT, and decode are timed separately with accelerator synchronization (`torch.mps.synchronize()`).
   - Emits contract v2 `benchmark_result` records validated via `qedge-common`.
   - Generated text and token IDs are held strictly in `outputs`, never conflated with numeric `metrics`.
   - No cache compression, no QNN execution, and no false hardware claims.
