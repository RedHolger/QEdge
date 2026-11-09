# C++ kernel lab

The C++ code remains at this project root during Stage 01. It is a **kernel
lab**, not the product runtime:

- it runs synthetic, deterministic tensors, not a tokenizer/model checkpoint;
- `llm_infer` reports cache bookkeeping and synthetic attention timing, not
  generation throughput;
- `llm_qnn_bench` uses `QnnStubBackend`, which delegates to the CPU and prints
  modelled values; it is not a Snapdragon/QNN/NPU measurement;
- no machine-specific home-directory path is needed to build or test it.

## Current mapping

| Path | Role |
|---|---|
| `src/core`, `include/core` | tensor/config utilities |
| `src/kv_cache`, `include/kv_cache` | KV-cache policies (lab implementations) |
| `src/backend`, `include/backend` | CPU, experimental Metal, and QNN CPU stub |
| `src/benchmark` | synthetic timing and power-report code |
| `tests/test_kv_cache.cpp` | legacy KV-cache tests |
| `tests/test_qnn_stub.cpp` | QNN stub CPU tests |

## Commands

From this directory:

```bash
make setup    # cmake --preset release
make test     # build + ctest --preset release
make demo     # synthetic kernel-lab demo
make bench    # QNN CPU-stub report; not an NPU result
```

The `debug` preset enables ASan/UBSan and exists to expose kernel defects for a
later stage; it is not part of the default Stage 01 gate. The real Python model
runner and tokenizer will live under `runtime/` from Stage 02.
