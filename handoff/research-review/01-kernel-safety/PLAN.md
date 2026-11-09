# 01-kernel-safety pre-edit record

Date: 29 October 2026

## Observable failures

1. `CPUBackend::dot_product_attention` names its first tensor axis `batch`, but the documented layout is `[heads, sequence, dimension]`. For GQA, `Q` has more heads than `K/V`; indexing `K` and `V` with the query-head index reads past their buffers. The sanitizer traces show:
   - `run_tests`: q `[4,1,32]`, k/v `[2,1,32]`; heap read immediately after the 256-byte K/V region.
   - `run_qnn_tests`: same 2:1 shape mismatch.
   - `llm_qnn_bench`: q `[32,1,64]`, retrieved k/v `[1,1,64]` because the benchmark allocated a one-head placeholder; heap read after the 2048-byte region.
2. `CPUBackend::dot_product_attention` adds the mask with a flat one-to-one loop. `llm_infer` supplies a broadcast mask `[1,4,4]` (16 floats) for scores `[8,4,4]` (128 floats), so it reads past the mask after 16 entries. This is the `llm_infer` mask-path cause, distinct from the GQA head-index cause.
3. The same function multiplies the whole score matrix by K/V as if all query heads shared one KV head. It does not map each query head to its KV head, does not use per-head V, and has no defined fully-masked behavior.
4. `CPUBackend::rope` applies one `pos` to every token, uses `base=10000` unconditionally, and uses exponent `j/head_dim` instead of `2j/head_dim`. It has no per-token positions or configurable base.
5. `benchmark.cpp` constructs a one-head k/v placeholder and calls attention even when `retrieve` returns false. This creates an invalid attention shape and can index an unfilled placeholder.
6. `MetalBackend::attention` returns raw QK matmul and ignores V, softmax and mask; `fused_qk_v_attention` returns scores rather than token outputs.

## Reference and test strategy

- Add an independent scalar attention reference in `tests/test_attention.cpp`, not implemented by calling `CPUBackend`.
- Cover MHA, MQA, GQA 3:1/4:1/7:1, distinct per-head V, q_len/kv_len > 1, head-broadcast and 2-D masks, per-head masks, fully masked rows, invalid shapes/dtypes/masks, and multi-position RoPE with bases 10000/100000/1000000.
- Add the test as a CTest target and run it in Release and Debug ASan/UBSan.
- Run both sanitized entry points. Metal will use an explicitly labelled CPU attention fallback; no claim of GPU attention.
- Existing historical logs and unrelated source remain untouched.

## Smallest coherent scope

- `llm-inference/include/backend/backend.h`: explicit `[heads, sequence, dimension]` contract; `RopeConfig`/`RopeStyle`; per-position RoPE interface with a legacy convenience overload.
- `llm-inference/include/backend/cpu_backend.h`: updated RoPE override.
- `llm-inference/src/backend/cpu_backend.cpp`: attention validation, GQA mapping, per-head V, mask broadcast/fully-masked semantics; corrected RoPE.
- `llm-inference/include/backend/metal_backend.h` and `src/backend/metal_backend.mm`: explicit labelled CPU fallback for attention and updated RoPE signature.
- `llm-inference/include/backend/qnn_stub_backend.h` and `src/backend/qnn_stub_backend.cpp`: updated RoPE signature; attention remains a CPU delegate.
- `llm-inference/src/benchmark/benchmark.cpp`: multi-head k/v buffer and no attention on retrieval failure.
- `llm-inference/tests/test_attention.cpp`: new independent regression tests.
- `llm-inference/CMakeLists.txt`: add only the new test target/CTest entry.
