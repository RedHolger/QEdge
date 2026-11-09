# Stage 11 — Hardware and Advanced Stretch (Optional)

## Goal

Use real Snapdragon hardware, a larger catalog, or advanced inference techniques only after Tier 1 and Tier 2 are accepted. This stage is optional and must not be started just to make a slide look better.

## Read first

- `06-QUALCOMM-ALIGNMENT.md`
- `handoff/10.md`
- `qnn/hardware_status.json`

## Options (choose one or two, not all)

### A. Measured Snapdragon run

- Snapdragon X Elite laptop or Snapdragon 8 Elite phone.
- QNN SDK / ONNX Runtime QNN EP.
- Compile the model for HTP.
- Run the same prompts and cache budgets.
- Record:
  - device model;
  - OS/driver;
  - QNN/ORT versions;
  - model hash;
  - quantization;
  - TTFT, prefill/decode tok/s;
  - peak memory;
  - power/thermal trace if available.
- Update `hardware_status.json` and the claims ledger.

### B. AI Hub Workbench run

- Use a free/authorized Workbench job.
- Record job ID, device, model, and artifact paths.
- Do not call it a local measurement.
- Include the compile/profile logs.

### C. 5,000-product catalog

- Only if disk/RAM and license allow.
- Re-run retrieval evaluation.
- Report the scaling effect.
- Do not inflate metrics with duplicate families.

### D. Advanced inference

- INT4 group-wise quantization.
- Paged attention / block allocator.
- Speculative decoding.
- Continuous batching.
- Each requires a correctness reference and a measured comparison.

## Acceptance

- real device logs for any device claim;
- exact versions and hashes;
- before/after table;
- no fabricated numbers;
- claims ledger updated;
- if the option is unavailable, do not start it.

## Required handoff

`handoff/11.md`:

- what was run;
- device/job details;
- measurements;
- comparison to local baseline;
- limitations;
- "stopped here."

## Guardrails

- Do not use a cloud CPU as "Snapdragon."
- Do not use a vendor marketing number as your measurement.
- Do not change the model and claim a speedup.
- Do not start this stage if stage 10 is not accepted.

## Stop

Stop after the handoff. Ask for review.
