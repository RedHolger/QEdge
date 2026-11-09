# Validate an actual Qualcomm execution path

Prompt ID: `11-qualcomm-deployment`  
Existing plan: Original Stage 05 plus optional Stage 11  
Prerequisites: 05-real-model-baseline and 08-quantization-measurement

## Execution contract

Work in the existing QEdge workspace. Read `docs/research-review/ANALYSIS.md`, current source, and applicable repository instructions first. Preserve unrelated and concurrent edits. Treat the Word document's code as unvalidated reference material. Work only on this prompt and its explicit prerequisite repairs; do not silently implement the entire roadmap.

Before editing, state the observable failing behavior and the test/reference that will distinguish a real fix. Prefer small coherent changes. Run relevant regression and behavior tests. Do not weaken a test to conceal a defect, fabricate assets/labels/results, or overwrite historical evidence. Missing hardware/data does not become a passing test.

Write a handoff under `handoff/research-review/` named after this prompt, containing changed files, commands and exit codes, passed/failed/skipped counts, artifact paths, claim changes, remaining defects, and next prerequisites. Produce a review verdict of ACCEPTED, REPAIR REQUIRED, or BLOCKED with evidence. Stop after this bounded stage for review; do not imply that a generated handoff is an independent review. If an input is unavailable, complete independent work and state the exact remaining dependency.

## Task

First inventory the real available target: SoC/device, OS/ABI, firmware/driver, QAIRT/QNN and ONNX Runtime versions, licenses, and device access. Write a compatibility matrix from primary documentation for those exact versions. No universal HTP ID, VTCM size, four-shard layout, or backend library filename. Choose the simplest supported route: ONNX Runtime QNN or a verified Genie/AI Hub artifact, not simultaneous speculative implementations.

Export a real supported graph and verify ONNX CPU numerical behavior first. Define prefill/decode interfaces, tokenization, KV ownership, positions and shape specialization. Cache surgery supported in PyTorch is not automatically available inside a serialized graph; document where selection executes, whether KV tensors cross processors, and the copy/synchronization cost.

On hardware, verify successful HTP inference with device/session logs and graph placement/profile evidence. Registered providers are insufficient. Configure partition/runtime fallback appropriately for the pinned runtime; where CPU wrapper operators are required, report them rather than claim zero CPU. Bind hardware status to hashed run artifacts, never to the mere presence of an environment variable.

Acceptance for a completed device run: CPU/target parity tolerance, actual generated output, graph assignment, model/quantization/context hashes, cold/warm latency, sustained behavior/thermal conditions, memory and transfer costs, repeatability, and failures. VTCM residency requires allocator/profile evidence, not byte arithmetic. No device: finish export/CPU checks and the device runbook, output unverified hardware status, and mark device performance BLOCKED. Do not block unrelated local product work or invent results.
