# QEdge Runtime and kernel lab

The Python [reference runtime](runtime/README.md) loads a pinned real SmolLM2
checkpoint and supports CPU generation, bounded-cache experiments and GGUF variants.
The optional ONNX path is a full-prefix CPU baseline without KV-cache I/O.
Install its additional dependencies explicitly with the `onnx` package extra.
Model acquisition is separate from offline inference and tests.

The C++ binaries are a synthetic kernel lab, not model-throughput measurements.
QNN is a CPU stub. Metal attention has an explicitly labeled CPU fallback.

On macOS with Apple frameworks:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The Debug preset enables ASan/UBSan. Stage 00 sanitizer failures are retained as
historical baseline evidence; current checks are in the Stage 12 handoff.

Qualcomm execution/performance remains BLOCKED. A provisional target runbook
is in `runtime/src/qedge_runtime/qualcomm/device_runbook.md`. Provider registration
and caller-supplied evidence cannot establish HTP execution. See the workspace
[release guide](../docs/RELEASE.md) for scope and remaining gates.
