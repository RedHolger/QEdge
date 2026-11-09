# QEdge Runtime — kernel lab and foundations

`llm-inference` is the QEdge Runtime workstream. Stage 01 establishes the C++
build and test gate; the real model runtime begins in Stage 02.

**Current status:** this is a synthetic C++ kernel lab, not a working LLM
inference engine. No model or tokenizer is loaded. The QNN path is a CPU stub
and **no Snapdragon/NPU measurement exists**.

## What runs today

- CPU and experimental Metal tensor kernels;
- FIFO/LRU/KeyDiff/QUOKA KV-cache lab implementations;
- synthetic cache/attention microbenchmarks;
- a QNN CPU-stub report that prints modelled, not measured, values;
- CTest registration for the legacy C++ tests.

These are useful for kernel regression and interface experiments. They are not
evidence of model throughput, generation quality, energy efficiency, or NPU
execution.

## Commands

From the workspace root:

```bash
make setup
make test
make demo
make bench
```

From this directory:

```bash
make setup    # cmake --preset release
make test     # build + ctest --preset release
make demo     # synthetic kernel-lab demo
make bench    # QNN CPU-stub report
```

Requirements: macOS 14+, CMake 3.23+, Apple Clang. The current lab links
Accelerate and Metal; the Python packages remain the portable path.

## Build presets

| Preset | Purpose |
|---|---|
| `release` | default Stage 01 build and CTest gate |
| `debug` | Debug + ASan/UBSan; intended to expose known kernel defects later |

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
cmake --install build/release --prefix /tmp/qedge-runtime-install
```

The install step copies the kernel-lab binaries and headers outside the source
tree; it does not install a model or claim NPU support.

## What is not claimed

- no model was loaded and no tokens were generated from a checkpoint;
- the printed `tok/s` is synthetic microbenchmark bookkeeping;
- `QnnStubBackend` executes on the CPU, not Hexagon;
- the power report is modelled/estimated, not a device power measurement;
- the Metal backend is incomplete; it is not a production inference path;
- there is no Snapdragon, QNN SDK, AI Hub, or NPU result.

`docs/CLAIMS.md` is the live claims ledger. Stage 00's old README and its
synthetic/NPU statements are frozen in [`legacy/`](legacy/). The Stage 01
acceptance criteria are in the workspace [`qedge-build-kit/`](../qedge-build-kit/).

## Layout

```text
llm-inference/
├── src/, include/, metal/   # current C++ kernel lab
├── tests/                   # legacy C++ tests, registered with CTest
├── cpp/README.md            # kernel-lab boundary and mapping
├── runtime/README.md        # Stage 02 placeholder for the real model path
├── legacy/                  # frozen Stage 00 README and retired claims
├── CMakeLists.txt
├── CMakePresets.json
└── Makefile
```

Stage 02 adds the Python model runner, tokenizer, reference generation path,
and the first real generation measurements. Until then, treat every number from
the C++ binaries as synthetic.
