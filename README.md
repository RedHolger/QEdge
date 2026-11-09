# QEdge validated local CPU portfolio

QEdge Assist is an installable evidence-grounded local assistant with find/check
modes, typed requirements, deterministic decisions, authorized versioned citations,
and constrained optional generation. QEdge Runtime provides a separate real-model
CPU reference, cache experiments, GGUF quantization, and an ONNX CPU export.

This is a portfolio release, not a production deployment or a verified NPU product.
Human retrieval quality and independent generalization evaluation remain pending.

## Install once, then run offline

Python 3.11 and `uv` are required. From this workspace, explicitly install/build:

```sh
sh scripts/bootstrap_portfolio.sh
```

This step acquires Python dependencies. It downloads no model or licensed catalog.
The following installed commands use only packaged authored fixtures and work from
any directory (use the absolute path to the executable when outside this workspace):

```sh
.release/venv/bin/qedge-portfolio demo
.release/venv/bin/qedge-portfolio benchmark
.release/venv/bin/qedge-portfolio evaluate
```

The benchmark measures fixture assistant latency, not model or NPU throughput.
Evaluation reports 80 previously exercised regression cases and ten scenario paths.
Image refusal is tested; image execution is not established by this demo.

For your own catalog:

```sh
.release/venv/bin/edge-rag-demo --catalog /path/to/catalog.jsonl --mode find "oak chair under 60cm wide"
.release/venv/bin/edge-rag-demo --catalog /path/to/catalog.jsonl --mode check --item-id ITEM_ID "width <= 60cm"
```

## Release evidence and limitations

- [Five-minute demo, capability matrix and rollback](docs/RELEASE.md)
- [Claims and evidence](docs/CLAIMS.md)
- [Technical walkthrough](docs/ARCHITECTURE.md)
- [Release handoff](handoff/research-review/12-release-review.md)
- [Model runtime](llm-inference/runtime/README.md)

C++ kernel tests require macOS 14+ and Apple frameworks. QNN remains a CPU stub
in the C++ lab. Metal attention uses a CPU fallback. The ONNX graph is a dynamic,
full-prefix CPU baseline with no KV-cache I/O, not a device-ready HTP artifact.
Stage 11 device execution and performance remain BLOCKED pending hardware and
version-specific graph preparation. No energy or Snapdragon performance is claimed.

Historical Stage 00/01 evidence and manifests are retained unchanged. Superseded
workspace documentation is archived under the Stage 12 handoff. Existing `make`
commands remain development conveniences and may install dependencies; use the
installed commands above for the offline release workflow.
