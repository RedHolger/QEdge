# Stage 05 — QNN / AI Hub Deployment Path

## Goal

Build a real ONNX → QNN deployment path for the runtime, with a complete device runbook and honest hardware status. Do not fake an NPU result if no device is available.

## Read first

- `02-TECHNICAL-SPEC.md` section 2.5
- `06-QUALCOMM-ALIGNMENT.md` section 5
- `handoff/04.md`

## Tasks

1. **ONNX export**
   - Export SmolLM2-135M-Instruct with past key/values.
   - Use `optimum` or an equivalent supported exporter.
   - Validate the exported graph on CPU with ONNX Runtime.
   - Compare a short greedy generation with the PyTorch reference.
   - Record opset, input/output names, and dynamic axes.

2. **ONNX Runtime backend**
   - Add `runtime/backends/ort_backend.py`.
   - CPU execution provider must work.
   - QNN execution provider must be selectable.
   - If QNN EP is unavailable on the host, fail with a clear message, not a silent fallback.
   - Record the provider actually used.

3. **QNN adapter**
   - Add `qnn/run_ort_qnn.py`.
   - Support:
     - `--provider qnn`
     - `--device <snapdragon>`
     - `--htp-performance-mode`
     - `--profiling-level`
   - Document the required QNN SDK/ORT versions.

4. **AI Hub path**
   - Add `qnn/compile_aihub.py` or a documented Workbench flow.
   - Include exact commands, model ID, device, and expected artifacts.
   - Do not require a paid service.
   - If access is unavailable, record the blocker.

5. **Device runbook**
   - `qnn/device_runbook.md`:
     - supported devices;
     - OS/driver requirements;
     - Python/ORT/QNN versions;
     - environment variables;
     - compile and run commands;
     - how to collect logs;
     - known failure modes.

6. **Hardware status**
   - `qnn/hardware_status.json`.
   - If no device: `status: unverified`.
   - If device: include log paths, device model, OS, driver, and measurements.

7. **Profiling**
   - On device: capture per-op or per-stage timings.
   - On host: capture CPU/QNN EP provider and fallback warnings.
   - Clearly separate host CPU results from device results.

## Acceptance

- ONNX export runs locally;
- ONNX Runtime CPU path runs and matches the PyTorch output within tolerance;
- QNN EP path compiles/imports or has a specific blocker;
- device runbook is complete enough for a third party;
- `hardware_status.json` is accurate;
- no NPU performance claim without device logs.

## Required handoff

`llm-inference/handoff/05.md`:

- export command;
- ORT CPU parity;
- QNN status;
- device logs (if any);
- hardware status;
- blockers;
- "stopped here."

## Guardrails

- Do not invent device logs.
- Do not use AI Hub numbers without recording the job/device.
- Do not change the model to make export easier without documenting it.
- Do not present a CPU fallback as QNN execution.

## Stop

Stop after the handoff. Ask for review.
