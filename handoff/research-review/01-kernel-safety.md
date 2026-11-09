# 01-kernel-safety handoff

Date: 29 October 2026  
Workspace: `/Users/manuvashistha/Developer/qualcomm`  
Stage: `01-kernel-safety`  
Prerequisite review: `reviews/research-review/00-rebaseline.md` — **ACCEPTED**  
Nested repository: `llm-inference` at `edf35b722c1eeb690602e840ff5052c4f2cb170c` (dirty; no commit created)

## Self-review verdict

**Self-review verdict: ACCEPTED. Independent acceptance is still pending.** This is not independent acceptance. Stages 02–04 remain blocked until this handoff is reviewed.

## Baseline authorization

The 00 rebaseline review verified:

- 45/45 baseline artifact hashes.
- 46/46 audited source hashes at that time.
- 201/201 protected non-handoff files.
- 30/30 historical hashes.
- Release CTest 2/2 passing, and ASan/UBSan CTest 0/2 aborting.
- The distinction between unsanitized Release success and sanitizer failure.

The 00 review also corrected the earlier overbroad statement about gaps: **G01–G12 are all still present, but only G01 and G07–G10 were dynamically reproduced.** G02–G05 and G12 were source-inspection findings; G06 and G11 were inventory/missing-capability findings. No gap was credited as fixed by the baseline.

## Observable failure and repair scope

### Attention

`CPUBackend::dot_product_attention` treated the first tensor axis as `batch` even though the layout is `[heads, sequence, dimension]`.

- **GQA/MQA failure:** with q `[4,1,32]` and k/v `[2,1,32]`, the old loop indexed k/v with q-head indices 0–3, reading past the 256-byte k/v region. This produced the ASan heap-buffer-overflow in `run_tests` and `run_qnn_tests`. The benchmark used q `[32,1,64]` with a one-head retrieval placeholder, producing the same class of overflow.
- **`llm_infer` mask failure:** the old mask path added a flat one-to-one mask. `llm_infer` supplies `[1,4,4]` (16 floats) against scores `[8,4,4]` (128 floats), so it read past the mask after 16 entries. This is a separate cause from the GQA head indexing.
- The old value multiplication also used the whole score matrix against a single KV-head layout instead of mapping each query head to its KV head and using that KV head's V values.
- The old fully masked case had no defined behavior.

### RoPE

The old implementation applied one `pos` to every token, hard-coded base `10000`, and used exponent `j/head_dim` instead of `2j/head_dim`. It had no per-token positions, no configurable base, and no explicit split-half/interleaved convention.

### Benchmark retrieval

The benchmark allocated a one-head placeholder, never checked `retrieve()`'s return value, and could pass the placeholder to attention. The placeholder is now allocated with all KV heads and attention is skipped on a cache miss.

### Metal

The old Metal attention returned `matmul(q,k)` — raw QK scores, ignoring V, softmax, and mask. This stage uses an explicit CPU fallback for attention and fused attention and labels it in `MetalBackend::name()` as `Metal (MPS/GPU; attention CPU fallback)`. No GPU-attention claim is supported.

### Smallest coherent interface change

The RoPE interface now takes explicit per-sequence positions and `RopeConfig`:

```cpp
enum class RopeStyle { SplitHalf, Interleaved };

struct RopeConfig {
  uint32_t head_dim = 0;
  float base = 10000.0f;
  RopeStyle style = RopeStyle::SplitHalf;
};

virtual Tensor rope(const Tensor& x,
                    const std::vector<uint32_t>& positions,
                    const RopeConfig& config) = 0;
```

The legacy `rope(x, pos, head_dim)` overload is retained as a convenience for contiguous positions starting at `pos`; it uses base 10000 and split-half style. All in-repository backends implement the new virtual method. This is an intentional source/interface change, not a silent layout change.

Attention layout is explicitly documented as `[heads, sequence, dimension]`, not batch-first. Masks are additive F32 and may be:

- `[q_len, kv_len]` (broadcast over heads), or
- `[1, q_len, kv_len]` (broadcast over heads), or
- `[q_heads, q_len, kv_len]` (per-head).

Mask values `<= -1e30f` are treated as forbidden. A row with no admissible key produces an all-zero output row; it does not divide by zero or emit NaNs. Invalid attention/RoPE shapes and dtypes throw `std::invalid_argument` in Release as well as Debug.

## Changed files

### Stage 01 source changes

| File | Change |
| --- | --- |
| `llm-inference/include/backend/backend.h` | Explicit `[heads, sequence, dimension]` contract; `RopeStyle`/`RopeConfig`; per-position RoPE virtual; legacy contiguous convenience overload. |
| `llm-inference/include/backend/cpu_backend.h` | New RoPE override; expose the legacy overload. |
| `llm-inference/src/backend/cpu_backend.cpp` | Attention validation, GQA/MQA KV-head mapping, per-head V, 2-D/3-D mask broadcast, sentinel handling, zero output for fully masked rows, stable softmax; corrected RoPE base/exponent/positions/styles. |
| `llm-inference/include/backend/metal_backend.h` | Explicitly labelled CPU attention fallback; new RoPE signature. |
| `llm-inference/src/backend/metal_backend.mm` | Attention and fused attention delegate to `CPUBackend::attention`; RoPE delegates with position/config. |
| `llm-inference/include/backend/qnn_stub_backend.h` | New RoPE signature; expose the legacy overload. |
| `llm-inference/src/backend/qnn_stub_backend.cpp` | RoPE delegate passes positions/config; attention remains a CPU delegate. |
| `llm-inference/src/benchmark/benchmark.cpp` | Allocate `[n_kv_heads,1,head_dim]` k/v retrieval buffers; skip attention on a failed retrieval; widen the backend-name column for the fallback label. |
| `llm-inference/tests/test_attention.cpp` | New independent scalar-reference regression suite (42 checks). |
| `llm-inference/CMakeLists.txt` | Add `run_kernel_tests` and the `kernel_safety` CTest entry; update CTest status message. |

### Pre-existing dirty state preserved

The nested repository was already dirty before this stage. `README.md`, `src/backend/backend.cpp`, `src/main.cpp`, `tests/test_kv_cache.cpp`, and the QNN/runtime/legacy files remain present. `tests/test_kv_cache.cpp` still contains one pre-existing trailing-whitespace line at 195; I did not modify it to avoid an unrelated cleanup. The full dirty `git diff --check` therefore exits 2, while the check restricted to the stage-touched files exits 0.

## Commands and exits

Build trees:

- ASan/UBSan Debug: `/tmp/qedge-research-asan-rr01-20261029`
- Release: `/tmp/qedge-research-release-rr01-20261029`

| Command | Exit | Result |
| --- | --- | --- |
| `cmake -S llm-inference -B /tmp/qedge-research-asan-rr01-20261029 -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON` | 0 | configured |
| `cmake --build /tmp/qedge-research-asan-rr01-20261029 -j 4` | 0 | built |
| `ctest --test-dir /tmp/qedge-research-asan-rr01-20261029 --output-on-failure` | 0 | 3/3 passed |
| `/tmp/qedge-research-asan-rr01-20261029/run_tests` | 0 | 21/21 passed |
| `/tmp/qedge-research-asan-rr01-20261029/run_kernel_tests` | 0 | 42/42 passed |
| `/tmp/qedge-research-asan-rr01-20261029/run_qnn_tests` | 0 | 12/12 passed |
| `/tmp/qedge-research-asan-rr01-20261029/llm_qnn_bench` | 0 | completed; no sanitizer markers |
| `/tmp/qedge-research-asan-rr01-20261029/llm_infer` | 0 | completed; no sanitizer markers |
| `cmake -S llm-inference -B /tmp/qedge-research-release-rr01-20261029 -DCMAKE_BUILD_TYPE=Release -DENABLE_SANITIZERS=OFF` | 0 | configured |
| `cmake --build /tmp/qedge-research-release-rr01-20261029 -j 4` | 0 | built |
| `ctest --test-dir /tmp/qedge-research-release-rr01-20261029 --output-on-failure` | 0 | 3/3 passed |
| `/tmp/qedge-research-release-rr01-20261029/run_tests` | 0 | 21/21 passed |
| `/tmp/qedge-research-release-rr01-20261029/run_kernel_tests` | 0 | 42/42 passed |
| `/tmp/qedge-research-release-rr01-20261029/run_qnn_tests` | 0 | 12/12 passed |
| `/tmp/qedge-research-release-rr01-20261029/llm_qnn_bench` | 0 | completed |
| `/tmp/qedge-research-release-rr01-20261029/llm_infer` | 0 | completed |
| `python3.11 scripts/verify_historical_hashes.py` | 0 | 30 verified, 0 failures |
| Preservation audit over `preservation.json` | 0 | 201 checked; 9 intentional in-scope changes; 0 unexpected |
| `git -C llm-inference diff --check -- <stage files>` | 0 | no whitespace errors in stage files |
| `git -C llm-inference diff --check` (whole dirty tree) | 2 | one pre-existing trailing-whitespace line in `tests/test_kv_cache.cpp:195` |

Sanitizer log search for `AddressSanitizer`, `UndefinedBehaviorSanitizer`, `runtime error:`, `LeakSanitizer`, `heap-buffer-overflow`, `SEGV`, or `ABORTING` found **no markers** in the ASan CTest or entry-point logs.

The only build warning is the pre-existing Apple Accelerate `cblas_sgemm` deprecation warning; it is not introduced by this stage.

## Numerical checks

`run_kernel_tests` uses a scalar reference implemented directly from the formulas; it does not call `CPUBackend` for the expected values. Absolute comparison tolerance is `1e-5`.

| Category | Checks | Result |
| --- | --- | --- |
| MHA with q_len=2, kv_len=3, distinct heads | 2 | pass |
| MQA with 4 query heads sharing one KV head | 2 | pass |
| GQA 3:1, 4:1, 7:1 mapping | 6 | pass |
| Rank-3 broadcast causal, rank-2 broadcast, per-head mask | 3 | pass |
| Fully masked row/all rows zero and finite | 4 | pass |
| Invalid attention shapes/dtypes/masks (Release exceptions) | 10 | pass |
| RoPE bases 10000/100000/1000000, non-contiguous positions, interleaved, contiguous legacy, rank-2 | 9 | pass |
| Invalid RoPE inputs (Release exceptions) | 6 | pass |
| **Total** | **42** | **42/42 pass in Release and ASan/UBSan** |

The tests target the original defects specifically:

- GQA/MQA shapes would overflow or mis-map under the old implementation.
- `[1,Q,K]` broadcast masks would be flat-indexed beyond their allocation under the old mask loop.
- Fully masked rows would become uniform under the old softmax rather than zero.
- Repeated-position/fixed-base/wrong-exponent RoPE would differ from the reference.
- Invalid inputs would not reliably throw in a Release/NDEBUG build.

## Artifacts

Logs and evidence are under `handoff/research-review/01-kernel-safety/logs/`:

- `asan-configure.log`, `asan-build.log`, `asan-ctest.log`
- `asan-run-tests.log`, `asan-kernel-tests.log`, `asan-run-qnn-tests.log`
- `asan-qnn-bench.log`, `asan-llm-infer.log`
- `release-configure.log`, `release-build.log`, `release-ctest.log`
- `release-run-tests.log`, `release-kernel-tests.log`, `release-run-qnn-tests.log`
- `release-qnn-bench.log`, `release-llm-infer.log`
- `preservation-audit.log`, `historical-hashes.log`, `changed-audited-source.log`
- `stage-diff-check.log`, `diff-check.log`, `new-test-whitespace.log`
- `kernel-tests.log`

Machine-readable summary: `handoff/research-review/01-kernel-safety/results.json`.  
Pre-edit failure analysis: `handoff/research-review/01-kernel-safety/PLAN.md`.

## Preservation evidence

- Baseline protected files: 201 checked, 0 missing.
- Intentional in-scope protected changes: 9 (`CMakeLists.txt`, `backend.h`, `cpu_backend.h`, `metal_backend.h`, `qnn_stub_backend.h`, `cpu_backend.cpp`, `metal_backend.mm`, `qnn_stub_backend.cpp`, `benchmark.cpp`).
- Unexpected protected changes: 0.
- Audited source files changed since baseline: 8. The new test file is untracked and not part of the 46-file baseline.
- Historical manifest: 30/30 verified after the stage edits.
- No model, catalog, checkpoint, or external asset was downloaded or used.

## Limitations and remaining defects

1. **This is a self-review.** No independent reviewer has accepted the stage. G01–G03 are repair candidates, not independently verified fixes.
2. **No real model/runtime has been tested.** The runtime path and model catalog remain unavailable; all attention and RoPE checks use synthetic tensors.
3. **Metal attention is CPU fallback.** The label reports this, but it is not a GPU attention implementation. Metal `matmul` remains MPS-backed; attention and fused attention are CPU.
4. **QNN remains a CPU stub.** No NPU or Snapdragon device measurement exists, and no NPU performance is claimed.
5. **G04 is only partially addressed.** Unsafe retrieval/placeholder use is fixed, but RNG is still reseeded per tensor, the workload is synthetic, attention uses a single retrieved token, and the attention timing field measures cache retrieval rather than a full attention kernel. Benchmark numbers remain synthetic, not model throughput.
6. **G05 is untouched.** LRU/KeyDiff/QUOKA policy semantics were not redesigned.
7. **G06 and G11 are untouched.** No integrated model runtime or neural/image retrieval quality evaluation was added.
8. **G12 is untouched.** The CPU-stub energy model and its timing-scope limitation remain.
9. **Validation scope is bounded.** Runtime validation was added for attention and RoPE. Other kernels (`matmul`, `softmax_inplace`, `rms_norm`, `silu_mul`, `add`) retain their pre-existing assertion-based checks.
10. **Mask semantics are bounded.** Only rank-2 and rank-3 additive F32 masks are supported; there is no 4-D batched mask, boolean mask, or checkpoint-specific mask inference. The forbidden sentinel is `<= -1e30f`.
11. **RoPE convention is selected explicitly.** The caller chooses split-half or interleaved style and base; no checkpoint autodetection was added.
12. **Release CTest is not a memory-safety gate.** The sanitizer build is the safety evidence; Release success is compatibility/NDEBUG evidence.

## Claim changes

- **Previous:** Metal returned raw QK scores as attention. **Now:** Metal attention/fused attention use an explicitly labelled CPU fallback; no GPU-attention claim.
- **Previous:** CPU attention could be described as MHA-only or unsafe for GQA/MQA. **Now:** GQA/MQA KV-head mapping and per-head V are implemented and covered by independent checks.
- **Previous:** RoPE was fixed-base, single-position, wrong-exponent. **Now:** per-token positions, configurable base, split-half/interleaved style, and correct `2j/head_dim` frequency are implemented.
- **Previous:** benchmark retrieval failure could feed an unfilled one-head placeholder to attention. **Now:** the placeholder is full KV-head shaped and attention is skipped on a miss.
- **Unchanged:** all QNN numbers are CPU-stub synthetic estimates, not NPU measurements.
- **Unchanged:** the C++ benchmark is not model inference and its timings are not throughput claims for real hardware.
- **Not claimed:** G04 benchmark redesign, G05 cache-policy repair, G06/G11 model/retrieval integration, G12 measured energy, or any NPU/GPU performance result.

## Next prerequisites

1. Independent review of this handoff and the stage-01 evidence.
2. If accepted, proceed to the next authorized stage only; stages 02–04 are not started here.
3. Real-model and device work remains blocked on approved runtime/model/device evidence.
4. Do not treat the synthetic benchmark numbers or the Metal CPU fallback as hardware performance evidence.
