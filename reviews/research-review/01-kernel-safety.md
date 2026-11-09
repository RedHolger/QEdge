# Review: 01-kernel-safety

Date: 29 October 2026  
Reviewed stage: `01-kernel-safety`  
Review type: independent review (not the implementation's self-review)  
Review scope: stage 01 only

## Verdict

**ACCEPTED**

The stage implements the requested attention, masking, RoPE, failed-retrieval,
and Metal-fallback repairs. The numerical references are independent of the
production implementation, the debug sanitizer and Release test runs reproduce,
and the preservation checks pass. No stage-01 defect requiring repair was found.

This acceptance is limited to stage 01. It does not accept G04-G06/G11/G12 or
the application as a whole, and it does not turn synthetic or CPU-stub results
into model, GPU, or NPU evidence.

## Independently reproduced evidence

All commands below were run against fresh build trees under `/tmp`, not the
implementation's build trees. Review logs are under
`reviews/research-review/01-kernel-safety/`.

### Debug ASan/UBSan

| Command | Exit | Result |
| --- | --- | --- |
| `cmake -S llm-inference -B /tmp/qedge-review01-20261029-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON` | 0 | configured |
| `cmake --build /tmp/qedge-review01-20261029-asan -j 4` | 0 | built |
| `ctest --test-dir /tmp/qedge-review01-20261029-asan --output-on-failure` | 0 | **3/3 passed** |
| `/tmp/qedge-review01-20261029-asan/run_tests` | 0 | **21/21 passed** |
| `/tmp/qedge-review01-20261029-asan/run_kernel_tests` | 0 | **42/42 passed** |
| `/tmp/qedge-review01-20261029-asan/run_qnn_tests` | 0 | **12/12 passed** |
| `/tmp/qedge-review01-20261029-asan/llm_qnn_bench` | 0 | completed; no sanitizer runtime markers |
| `/tmp/qedge-review01-20261029-asan/llm_infer` | 0 | completed; no sanitizer runtime markers |
| Strict rerun with `ASAN_OPTIONS=halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` for `run_kernel_tests`, `llm_infer`, `llm_qnn_bench` | 0, 0, 0 | no runtime sanitizer markers |

The review binaries are linked with ASan and contain UBSan handler symbols;
`otool` shows `libclang_rt.asan_osx_dynamic.dylib`, and `nm` shows the UBSan
handlers. LeakSanitizer is not available on this macOS target, so this review
does not claim leak detection. The sanitizer result is for ASan/UBSan only.

### Release / NDEBUG

| Command | Exit | Result |
| --- | --- | --- |
| `cmake -S llm-inference -B /tmp/qedge-review01-20261029-release -DCMAKE_BUILD_TYPE=Release -DENABLE_SANITIZERS=OFF` | 0 | configured |
| `cmake --build /tmp/qedge-review01-20261029-release -j 4` | 0 | built |
| `ctest --test-dir /tmp/qedge-review01-20261029-release --output-on-failure` | 0 | **3/3 passed** |
| `/tmp/qedge-review01-20261029-release/run_tests` | 0 | **21/21 passed** |
| `/tmp/qedge-review01-20261029-release/run_kernel_tests` | 0 | **42/42 passed** |
| `/tmp/qedge-review01-20261029-release/run_qnn_tests` | 0 | **12/12 passed** |
| `/tmp/qedge-review01-20261029-release/llm_qnn_bench` | 0 | completed |
| `/tmp/qedge-review01-20261029-release/llm_infer` | 0 | completed |

The Release `run_kernel_tests` flags include `-DNDEBUG`, `-O3`, and
`-ffast-math`. The validation assertions in the tests therefore exercise the
explicit `std::invalid_argument` paths, not debug `assert` behavior.

### Preservation and historical evidence

| Check | Result |
| --- | --- |
| `python3.11 scripts/verify_historical_hashes.py` | exit 0; **30/30 verified, 0 failures** |
| Preservation audit over `handoff/research-review/00-rebaseline/preservation.json` | exit 0; **201 protected files, 0 missing, 9 expected changes, 0 unexpected changes** |
| Audited 46-file source manifest | 38 unchanged, **8 expected changed**, 0 missing |
| `git -C llm-inference diff --check -- <stage files>` | exit 0 |
| `git -C llm-inference diff --check` (whole dirty tree) | exit 2; only `tests/test_kv_cache.cpp:195` trailing whitespace |
| `tests/test_kv_cache.cpp` SHA-256 vs baseline | **match** (`1cea...`) |

The nine expected protected changes are the declared stage set:
`CMakeLists.txt`, `backend.h`, `cpu_backend.h`, `metal_backend.h`,
`qnn_stub_backend.h`, `cpu_backend.cpp`, `metal_backend.mm`,
`qnn_stub_backend.cpp`, and `benchmark.cpp`. The new
`tests/test_attention.cpp` is untracked and outside the 201 protected files.

The trailing-whitespace defect at `tests/test_kv_cache.cpp:195` is pre-existing:
the file's current hash matches its baseline preservation hash, and the file is
outside the stage change set. I did not clean it. No historical artifact was
overwritten.

## Source review

### Attention and mask semantics

In `llm-inference/src/backend/cpu_backend.cpp`:

- `require_f32_rank3()` and `validate_attention()` reject non-F32 inputs,
  non-3-D Q/K/V tensors, zero dimensions, non-divisible Q/K head counts,
  mismatched K/V head, sequence, and dimension shapes, and invalid rank-2 or
  rank-3 masks. These checks are explicit throws, so they are active in Release.
- Each query head maps to `kv_head = h / (q_heads / kv_heads)`. This is the
  requested `h_kv = h_q / (Hq/Hkv)` convention. The 3:1, 4:1, and 7:1 tests
  exercise the mapping.
- Mask access supports `[q_len, kv_len]`, `[1, q_len, kv_len]`, and
  `[q_heads, q_len, kv_len]`; the first and second broadcast correctly.
- The output is allocated as `[q_heads, q_len, head_dim]`, zero-initialized,
  and each query head accumulates against the value rows of its mapped KV head.
  Distinct per-head V values are exercised.
- The softmax subtracts the row maximum. Bias values `<= -1e30f` are excluded
  before exponentiation. A row with no admissible key leaves the zero output
  row untouched; if the denominator is not positive for another numerical
  reason, the row also remains defined as zero.
- `fused_qk_v_attention` uses the same validated implementation with no mask.

The old flat mask indexing and query-head indexing into a smaller K/V head set
are gone. The sanitized entry points that previously reached the overflow now
complete without runtime markers.

### RoPE

`CPUBackend::rope()` now accepts explicit per-sequence positions and
`RopeConfig{head_dim, base, style}`:

- Split-half pairs `(j, j + head_dim/2)`; interleaved pairs
  `(2j, 2j + 1)`.
- Frequency exponent is `2j/head_dim`, with configurable base.
- The token position is selected from the supplied position vector, not reused
  for every sequence element.
- Rank-2 `[sequence, dimension]` and rank-3 `[heads, sequence, dimension]`
  layouts are supported.
- The legacy `rope(x, pos, head_dim)` overload expands `pos` to a contiguous
  position run and uses base 10000/split-half.

The tests cover non-contiguous positions, bases 10000, 100000, and 1000000,
both styles, the legacy overload, and rank 2. The scalar reference uses the same
mathematical formulas but does not call `CPUBackend`; the production code and
reference are separate implementations.

### Benchmark retrieval and Metal reporting

In `llm-inference/src/benchmark/benchmark.cpp`, retrieval placeholders are
allocated as `[n_kv_heads, 1, head_dim]`. `cache.retrieve()` is checked, and
attention is invoked only when `hit && layer == 0`. A miss therefore cannot feed
an unfilled one-head placeholder into a multi-head attention call.

`MetalBackend::name()` reports
`Metal (MPS/GPU; attention CPU fallback)`. Its attention and fused-attention
paths delegate to `CPUBackend::attention`; they no longer return raw QK scores.
The output of `llm_infer` shows this label. This is a documented CPU fallback,
not a GPU-attention implementation.

## Numerical-reference independence

`llm-inference/tests/test_attention.cpp` implements scalar attention and RoPE
references that do not call `CPUBackend` or reuse its output. The reference:

- computes QK scores, scaling, softmax, and V accumulation directly;
- uses the documented group-major GQA mapping;
- indexes rank-2/rank-3 masks independently;
- writes explicit split-half/interleaved RoPE pairs.

The reference necessarily shares the documented mask sentinel (`-1e30f`) and
the documented head ordering; those are contract choices, not implementation
calls. I also manually checked the standard `2j/head_dim` frequency and the
`h / (Hq/Hkv)` mapping against the formulas. The tests are independent in the
relevant sense: a different implementation path produces the expected values.

Test coverage matches the acceptance list: MHA, MQA, GQA 3:1/4:1/7:1, distinct
per-head V, `q_len > 1` and `kv_len > 1`, causal/no-mask/broadcast/per-head
masks, fully masked rows, invalid attention shapes/dtypes/masks, and
multi-position RoPE with the three required bases. All 42 checks pass in both
Debug ASan/UBSan and Release/NDEBUG.

## Gap status after this stage

| Gap | Status after stage 01 |
| --- | --- |
| G01 (CPU attention/head/mask safety) | Repaired in scope; independent scalar tests and sanitized entry points pass. |
| G02 (Metal raw-QK attention) | Replaced by an explicitly labelled CPU attention fallback; no GPU-attention claim. |
| G03 (RoPE positions/base/exponent/convention) | Repaired in scope; tested for positions, bases, and both pair styles. |
| G04 (benchmark realism/timing) | Only the unsafe retrieval/placeholder path is repaired. RNG is still reseeded per tensor, the workload is synthetic, attention uses one retrieved token, and `avg_attention_us` still measures retrieval rather than a full attention kernel. |
| G05 (cache-policy semantics) | Untouched. LRU/KeyDiff/QUOKA semantics remain as previously reported. |
| G06 (real runtime/model) | Untouched; no model/runtime integration. |
| G07-G10 (assistant/evidence contracts) | Outside stage 01; unchanged from the accepted baseline. |
| G11 (neural/image retrieval evaluation) | Untouched. |
| G12 (QNN energy scope) | Untouched; QNN remains a CPU stub and no NPU measurement is claimed. |

This distinction is important: passing stage-01 tests does not close G04-G06,
G11, or G12, and it does not certify application contracts or model behavior.

## Remaining limitations / follow-up

- **No real model or NPU.** All kernel checks use synthetic tensors; QNN is a
  CPU delegate and Metal attention is a CPU fallback. No Snapdragon, GPU, model
  throughput, or measured-energy claim is supported.
- **G04 remains partial.** The benchmark's timing field remains a cache
  retrieval measurement, and its workload remains synthetic.
- **Tensor slice offsets remain a pre-existing cross-cutting issue.**
  `Tensor::clone()` and some raw-data operations do not honor `Tensor::offset_`;
  a RoPE/attention call on a sliced tensor can therefore read the wrong region.
  No stage-01 test or stage-01 call site passes a sliced tensor. This is not a
  regression introduced by stage 01 and is outside the G01-G03 repair scope, but
  it should be resolved before relying on sliced model tensors.
- **Validation is bounded to attention and RoPE.** Other kernels retain their
  pre-existing assertion-based checks.
- **Mask semantics are bounded.** The documented contract is additive F32
  rank-2/rank-3 masks with `<= -1e30f` forbidden; there is no boolean mask,
  4-D batch mask, or automatic checkpoint-specific mask inference.
- **Release is not the memory-safety gate.** Release confirms NDEBUG
  compatibility and explicit validation; the ASan/UBSan build is the safety
  evidence.

## Decision

**ACCEPTED for stage 01.** The stage has no blocking defect in its declared
scope. The implementation handoff's self-review was not treated as independent
acceptance; this review is the independent acceptance.

The next authorized action is `02-contracts`, after which the workflow must stop
for review. This acceptance does not authorize stages 03-04, real-model/NPU
claims, or closure of the remaining gaps listed above.
