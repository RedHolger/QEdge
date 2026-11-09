# Research review 05 real model baseline handoff

Date: 31 October 2026  
Scope: Stage 05 (`05-real-model-baseline`) post-repair handoff for the separately packaged real-model reference runtime under `llm-inference/runtime`, primary checkpoint `HuggingFaceTB/SmolLM2-135M-Instruct` pinned to git revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, license recording (`Apache-2.0`), canonical artifact hashes across all 8 execution-affecting files, checkpoint architecture extraction (30 layers, 9 Q heads, 3 KV heads, head dimension 64, RoPE theta 100k, vocab 49,152), explicit model acquisition tool with download size estimation, offline local-path execution mode, real prefill and autoregressive decode loop with full KV cache, strict monotonic position IDs, attention masks, output bounds (`max_new_tokens`), early EOS handling with multiple EOS ID support and non-falsy zero ID checks, state isolation without request leakage, greedy token parity (100%) and numerical logits agreement with documented tolerances (`atol=1e-4` on CPU float32, `atol=0.15` on MPS bfloat16) against unmodified `transformers` `model.generate()`, separate measurement boundaries (loading, formatting, tokenization, prefill, first-token selection, TTFT through first token, decode, detokenization) with accelerator synchronization, contract v2 `benchmark_result` emission with outputs separated from numeric metrics, factual execution metadata (seed policy, template digest identity, actual model parameter dtype), and expanded regression testbed (39 tests).  
Prerequisites: `00-rebaseline` (ACCEPTED), `01-kernel-safety` (ACCEPTED), `02-contracts` (ACCEPTED), `03-evidence-store` (ACCEPTED), `04-assistant-correctness` (ACCEPTED bounded scope).  
All 7 baseline v1 JSON schemas, 30 historical manifest artifacts, and Stage 01 kernel safety implementations in `llm-inference` were strictly preserved.

---

## Verdict

**Self-assessment: COMPLETE & READY FOR RE-REVIEW. Independent review: PENDING.**

This handoff records author self-assessment following implementation of bounded provenance and metadata repairs addressing independent review findings. The separately packaged reference runtime under `llm-inference/runtime` implements full-cache prefill and autoregressive decoding for the primary checkpoint `HuggingFaceTB/SmolLM2-135M-Instruct`. Exact greedy token parity (100%) and numerical logits agreement are confirmed against unmodified upstream `transformers` generation across diverse prompts. Per instructions, the secondary model (`Qwen/Qwen2.5-0.5B-Instruct`) remains deferred until this baseline is independently reviewed and accepted. No cache compression, QNN execution, or NPU hardware claims are made or implied.

---

## Review Findings and Implemented Repairs

The independent review identified incomplete provenance verification and metadata fidelity gaps; all were reproduced prior to repair and verified resolved with dedicated regression tests:

### 1. Verified Manifest Hash Values, Not Merely Presence
- **Defect:** In an earlier revision, manifests where `artifact_hashes` values were replaced with 64 zeros were accepted as verified because verification only checked that the required keys existed in the dictionary.
- **Repair:**
  - [`check_model_assets()`](../../llm-inference/runtime/src/qedge_runtime/acquisition.py) computes actual SHA256 digests for all 8 required files on disk, then validates each recorded manifest digest:
    1. Recorded digest format must be a valid 64-character lowercase hex string (rejecting malformed hashes).
    2. Recorded digest must match the actual file digest on disk (`rec_h.lower() == act_h.lower()`).
    3. Recorded digest must match the trusted expected hash for the pinned checkpoint (`rec_h.lower() == exp_h.lower()`).
    4. Rejects manifests with unknown extra files or missing required files.
  - Retains support for manifest-free canonical assets **only if** their complete execution identity is established directly through trusted expected hashes (`verify_trusted_hashes=True`).
  - Added negative regression tests: `test_rejection_of_zeros_in_manifest_hashes`, `test_manifest_free_canonical_assets`.

### 2. Closed Gap Between Verified Assets and Loader Override Files
- **Defect:** An unverified `chat_template.jinja` placed alongside canonical assets was accepted by `check_model_assets`, and `AutoTokenizer.from_pretrained()` silently loaded it, hijacking the prompt format.
- **Repair:**
  - `check_model_assets()` enforces strict asset isolation by scanning directory entries:
    - Allows verified required execution assets (`REQUIRED_ASSET_FILES`).
    - Allows `acquisition_manifest.json` and hidden cache/vcs files (`.*`).
    - Allows harmless documentation files (`README*`, `LICENSE*`, `NOTICE*`, `CHANGELOG*`, `CITATION*`, `COPYING*`).
    - **Rejects any unverified loader override asset or extra execution-affecting file** (`.jinja`, extra `.json` configs, `.safetensors`, `.bin`, `.model`, `.py`, or subdirectories) with an explicit error.
  - In [`runner.py`](../../llm-inference/runtime/src/qedge_runtime/runner.py), `ModelRunner.__init__`:
    - Checks provenance via `check_model_assets()` and raises `ValueError` on any unverified directory.
    - Inspects the loaded tokenizer's `chat_template` and computes its SHA256 digest (`template_digest`), enforcing exact agreement with the trusted canonical template digest (`SMOLLM2_135M_EXPECTED_TEMPLATE_DIGEST = "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"`).
  - Added regression test `test_loader_override_cannot_change_formatted_prompt_while_verified` demonstrating that an extra template cannot alter formatted input while provenance is reported as verified.
  - Added tests `test_rejection_of_unexpected_loader_override_assets`, `test_rejection_of_extra_unverified_model_weights`, and control test `test_allow_harmless_documentation`.

### 3. Factual Reproducibility Metadata
- **Defect:** `runner.py` previously reported `"seed": 42` without actually setting a seed, reported a fixed string label for chat template identity, and used the requested dtype alias rather than the actual model parameter dtype.
- **Repair:**
  - **Seed Policy:** Greedy decoding defaults to `"seed_policy": "none_deterministic_greedy"` with `"seed": null`. When an explicit integer seed is passed (`runner.generate(..., seed=123)`), it sets `torch.manual_seed(seed)` and records `"seed_policy": "explicit"`, `"seed": seed`. Non-integer/bool seeds are rejected.
  - **Template Identity:** Derived directly from the effective loaded template string:
    - In chat template mode: `"effective_template_digest": "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"`, `"template_identity": "sha256:872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788"`, `"raw_prompt_mode": false`.
    - In raw prompt mode (`use_chat_template=False`): `"raw_prompt_mode": true`, `"effective_template_digest": null`, `"template_identity": "raw_prompt"`.
  - **Actual Dtype:** Derived directly from loaded model parameters (`next(iter(self.model.parameters())).dtype`), recording both `"requested_dtype"` and `"actual_dtype"`.

---

## Commands and Results

All commands were run from the workspace root against isolated environments. Dated logs are preserved in `handoff/research-review/05-real-model-baseline/logs/`.

| Command | Exit | Passed | Failed | Skipped | Status | Evidence |
|---|---:|---:|---:|---:|---|---|
| `runtime-tests` | 0 | 39 | 0 | 0 | PASS (39 passed, up from 31) | [log](05-real-model-baseline/logs/runtime-tests.log) |
| `stage05-postrepair-probe` | 0 | 6 | 0 | 0 | PASS (0/6 active defects remaining) | [log](05-real-model-baseline/logs/2026-10-30-stage05-postrepair.log) |
| `provenance-gaps-probe` | 0 | 2 | 0 | 0 | PASS (both provenance gaps resolved) | [log](05-real-model-baseline/logs/2026-10-30-provenance-gaps-postrepair.log) |
| `parity-probe` | 0 | 10 | 0 | 0 | PASS (10/10 checks clean) | [log](05-real-model-baseline/logs/parity-probe.log) |
| `offline-mode-probe` | 0 | 1 | 0 | 0 | PASS (network blocked at socket) | [log](05-real-model-baseline/logs/offline-mode.log) |
| `benchmark` | 0 | — | — | — | PASS (emits validated contract v2 record) | [log](05-real-model-baseline/logs/benchmark.log) |
| `cli` | 0 | — | — | — | PASS (valid stdout JSON, stderr logging) | [log](05-real-model-baseline/logs/cli.log) |
| `llm-ctest` | 0 | 3 | 0 | 0 | PASS (CTest 3/3) | [log](05-real-model-baseline/logs/llm-ctest.log) |
| `edge-tests` | 0 | 68 | 0 | 1 | PASS (1 expected skip) | [log](05-real-model-baseline/logs/edge-tests.log) |
| `qedge-common-tests` | 0 | 49 | 0 | 0 | PASS | [log](05-real-model-baseline/logs/qedge-common-tests.log) |
| `historical-hashes` | 0 | 30 | 0 | 0 | PASS (30/30 verified) | [log](05-real-model-baseline/logs/historical-hashes.log) |
| `schema-hashes` | 0 | 7 | 0 | 0 | PASS (all 7 v1 schemas byte-for-byte) | [log](05-real-model-baseline/logs/schema-hashes.log) |

*Test count breakdown in `runtime-tests` (39 total):*
- `test_acquisition.py`: 11 tests (+5 new tests for manifest hash verification, zero-hash rejection, loader override rejection, extra weight rejection, harmless doc allowance, manifest-free canonical assets).
- `test_benchmark_contracts.py`: 5 tests (real benchmark validation, smoke validation, zero-output metric validation, missing asset error without smoke fallback, measured run rejection without model).
- `test_config.py`: 5 tests (pinned metadata, from checkpoint dict, architecture defaults, chat prompt formatting, sha256 computation).
- `test_parity_real.py`: 4 tests (greedy token parity + logits agreement, zero/one token upstream parity, MPS parity, explicit Hub revision propagation).
- `test_runner_unit.py`: 14 tests (+3 new tests: `test_explicit_seed_setting_and_factual_metadata`, `test_factual_template_identity_and_raw_prompt_mode`, and `test_loader_override_cannot_change_formatted_prompt_while_verified`).

---

## Pinned Model Checkpoint & Verified Artifacts

All 8 execution-affecting files are verified against trusted SHA256 hashes:

| File | Size (Bytes) | Trusted Expected SHA256 |
|---|---:|---|
| `config.json` | 861 | `8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a` |
| `generation_config.json` | 132 | `87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013` |
| `merges.txt` | 466,391 | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| `model.safetensors` | 269,060,552 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| `special_tokens_map.json` | 655 | `2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52` |
| `tokenizer.json` | 2,104,556 | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `tokenizer_config.json` | 3,764 | `4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df` |
| `vocab.json` | 800,662 | `82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79` |

- **Effective Loaded Chat Template Digest:** `872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788`
- **Model ID:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Pinned Git Commit SHA:** `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- **License:** `Apache-2.0`
- **Extracted Checkpoint Architecture:**
  - `model_type`: `llama` (`LlamaForCausalLM`)
  - `vocab_size`: 49,152
  - `hidden_size`: 576
  - `intermediate_size`: 1,536
  - `num_hidden_layers`: 30
  - `num_attention_heads`: 9
  - `num_key_value_heads`: 3 (Grouped-Query Attention: 3 query heads per KV head)
  - `head_dim`: 64 (`hidden_size // num_attention_heads = 576 // 9 = 64`)
  - `max_position_embeddings`: 8,192
  - `rope_theta`: 100,000.0
  - `bos_token_id`: 1
  - `eos_token_id`: 2
  - `pad_token_id`: 2
  - `tie_word_embeddings`: true
  - `torch_dtype`: `bfloat16`

---

## Upstream Parity & Tolerances

Evaluated against unmodified `transformers` `model.generate(..., do_sample=False, output_scores=True)`:

| Prompt | Custom Tokens | Upstream Reference Tokens | Token Parity | Max Logit Abs Diff | Logits Status |
|---|---|---|:---:|---:|:---:|
| `What is 2 + 2?` | `[34, 1232, 216, 34, ...]` (16 tok) | `[34, 1232, 216, 34, ...]` (16 tok) | **EXACT (100%)** | `0.0000515` | PASS (`atol=1e-4`) |
| `What is the capital of France?` | `[504, 3575, 282, 4649, ...]` (8 tok, EOS) | `[504, 3575, 282, 4649, ...]` (8 tok, EOS) | **EXACT (100%)** | `0.0000381` | PASS (`atol=1e-4`) |
| `Name the primary colors in order.` | `[57, 4390, 22657, 314, ...]` (16 tok) | `[57, 4390, 22657, 314, ...]` (16 tok) | **EXACT (100%)** | `0.0000476` | PASS (`atol=1e-4`) |
| `Write a Python function to square a number.` | `[1041, 253, 5606, 1284, ...]` (16 tok) | `[1041, 253, 5606, 1284, ...]` (16 tok) | **EXACT (100%)** | `0.0000542` | PASS (`atol=1e-4`) |

- **Documented Tolerances:**
  - **CPU `float32`:** `atol=1e-4, rtol=1e-4`. Max observed logit difference across all test prompts is $5.15 \times 10^{-5}$.
  - **Apple Silicon MPS `bfloat16`:** `atol=0.15, rtol=1e-2`. Reflects the 7-bit mantissa machine epsilon ($2^{-7} \approx 0.0078$) at logit magnitudes $\sim 33.0$ ($33.0 \times 2^{-8} = 0.125$). Verified exact greedy token parity on MPS.

---

## Limitations & Deferred Work

1. **Secondary Model (`Qwen/Qwen2.5-0.5B-Instruct`):** Deferral explicitly maintained; to be introduced only after Stage 05 baseline passes independent review.
2. **KV Cache Compression Policies:** Full cache (100% budget) is maintained; eviction policies (FIFO, LRU, KeyDiff, attention-value decay) belong strictly to **Stage 06 (`06-cache-contract`)** and **Stage 07 (`07-research-selectors`)**.
3. **Hexagon NPU / Snapdragon Execution:** No QNN execution is claimed; device verification belongs strictly to **Stage 11 (`11-qualcomm-deployment`)**.

---

## Next Steps

Work stops here for independent review of Stage 05 provenance repairs. Stage 06 (`06-cache-contract`) is NOT started and remains blocked until Stage 05 acceptance is formally recorded.
