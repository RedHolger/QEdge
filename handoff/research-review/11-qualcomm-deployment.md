# Stage 11 Handoff: Qualcomm Execution Path Validation (Repaired)

**Prompt ID:** `11-qualcomm-deployment`  
**Date:** 2026-11-07  
**Self-Review Verdict:** `ACCEPTED` (Repaired export provenance validation, active CPU parity under python -O, and strict hardware governance; device execution BLOCKED / NPU UNVERIFIED)  
**Historical Hashes:** 30/30 Verified (`scripts/verify_historical_hashes.py`)  
**Hardware Status:** `npu_status="UNVERIFIED"`, `device_execution_status="BLOCKED"`, `device_performance_status="BLOCKED"`  
**Retrieval Gate Status:** Software repairs ACCEPTED (105/105 tests, 49/49 schema tests); Human retrieval-quality gate remains **BLOCKED** pending annotation of the 1,009 active review queue candidates  
**Secondary Model Status:** `Qwen/Qwen2.5-0.5B-Instruct` remains **DEFERRED / BLOCKED**  
**Generalization Evaluation:** `PENDING`

---

## 1. Executive Summary & Review Findings Addressed

Following a second independent review, Stage 11 was marked `REPAIR REQUIRED` with four specific findings:
1. **Hardware Evidence Bypass:** A caller-supplied dictionary `{"verified_htp_placement": true, "artifact_sha256": "..."}` was accepted by the runner without placement logs, profiler artifacts, or device evidence.
2. **Export Provenance Gaps:** Small identity graphs with wrong source revisions and missing opset metadata bypassed validation.
3. **Assert-Based Parity Checks:** Parity checks used Python `assert`, which compiles away under `python -O`, and token matching counters incremented unconditionally.
4. **Provisional Runbook Tightening:** Unsubstantiated universal TOPS and throughput claims were present in the provisional runbook and compatibility matrix.

All four findings have been addressed and verified with behavioral and negative tests:

1. **Closed Hardware Evidence Bypass ([`runner.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/qualcomm/runner.py)):**
   - Caller-supplied booleans or model checksums are **never** treated as execution proof.
   - Hardware status strictly remains **`UNVERIFIED`** under all local execution conditions.
   - Any caller-supplied evidence dictionary is preserved and returned strictly as `unvalidated_hardware_evidence`.
   - The runner separates CPU-only sessions from mixed/QNN sessions:
     * In a CPU-only session (`registered_providers == ["CPUExecutionProvider"]`): `session_type = "cpu_only"`, `executed_placement = "CPU (CPUExecutionProvider)"`, `fallback_detected = False`.
     * In an unverified mixed/QNN session: `session_type = "unverified_qnn"`, `executed_placement = "UNKNOWN (placement logs not verified)"`, and `fallback_detected = "UNKNOWN (node placement unverified)"`. Neither CPU placement nor absence of fallback is inferred.
   - Regression tests verify that both CPU and QNN sessions supplied with fabricated evidence strictly preserve `hardware_status="UNVERIFIED"`.

2. **Completed Full Export-Provenance and Schema-Consistency Validation ([`provenance.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/qualcomm/provenance.py)):**
   - Created shared validation module `validate_onnx_export_provenance()` enforcing the complete manifest contract and graph consistency:
     * Model identity (`HuggingFaceTB/SmolLM2-135M-Instruct`).
     * Pinned source revision (`12fd25f77366fa6b3b4b768ec3050bf629380bac`).
     * Artifact SHA256 and byte size match against the file on disk.
     * Default-domain opset import (`""` or `"ai.onnx"`) in the ONNX graph header matches manifest `opset_version` and requested opset.
     * ONNX filename match: manifest `onnx_file` must strictly match the resolved model filename on disk.
     * Graph input interface: exactly 1 input named `input_ids`, data type `INT64`, rank 2 (`[batch, seq]`).
     * Graph output interface: exactly 1 output named `logits`, data type `FLOAT32`, rank 3 (`[batch, seq, vocab=49152]`).
     * Deep comparison of `input_schema`: manifest keys must match graph inputs, declared data types must match graph element types (`int64`), and declared shape ranks must match graph ranks (rank 2).
     * Deep comparison of `output_schema`: manifest keys must match graph outputs, non-empty schema required, declared data types must match graph element types (`float32`), shape ranks must match graph ranks (rank 3), and vocab dimensions must match `EXPECTED_VOCAB_SIZE` (49152).
     * Execution mode and KV cache consistency: rejects contradictory declarations (`cached_decode`, `use_cache=True`) for graphs lacking KV cache tensors.
     * Graph structural integrity via `onnx.checker.check_model`.
   - Shared between `export_smollm2_to_onnx()` reuse decisions and `ONNXModelRunner` verified initialization.
   - Preserved the original export manifest created on 2026-11-05 as immutable historical evidence without inventing historical tool versions or source hashes.
   - Added `close()`, `__enter__`, `__exit__`, and `__del__` lifecycle management on `ONNXModelRunner` for deterministic release of native session handles.
   - Tested against comprehensive negative probes: wrong revision (`"wrong-revision"`), missing opset, incompatible input name (`"X"`), wrong model ID, contradictory `onnx_file`, wrong input name (`"wrong_input"`), wrong input dtype (`"float16"`), wrong input rank (`[99]`), empty `output_schema` (`{}`), wrong output name (`"wrong_output"`), wrong output dtype (`"int64"`), wrong output rank (`[1, 4]`), wrong vocab size (`32000`), `cached_decode`, `use_cache=True`, combined contradictory probe, and runner instantiation with `require_verified=True` rejecting contradictory manifests with `ValueError`.

3. **Active Parity Checks Under Optimized Python (`python -O`) ([`export.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/qualcomm/export.py)):**
   - Replaced all Python `assert` statements in `verify_onnx_cpu_numerical_parity` and `verify_onnx_cpu_autoregressive_parity` with explicit `ParityVerificationError` exceptions.
   - Added validation rejecting non-finite logit differences (`np.isfinite`), empty prompt lists, non-positive step counts, and zero-step evaluations.
   - Matching step counter (`total_matching_tokens`) increments **only** upon verified token identity (`pt_next == ort_next`).
   - Exercised a deliberate parity failure under both normal Python and `python -O` (via subprocess exit code 42), demonstrating active enforcement under compiler optimization.

4. **Tightened Provisional Runbook ([`device_runbook.md`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/qualcomm/device_runbook.md) & [`compatibility.py`](file:///Users/manuvashistha/Developer/qualcomm/llm-inference/runtime/src/qedge_runtime/qualcomm/compatibility.py)):**
   - Excised all universal TOPS claims (removed "25 TOPS", "45 TOPS") from runbook and compatibility matrices.
   - Removed universal throughput claims; cited primary documentation for SoC architectures (HTP v73, v75, v79).
   - Retained explicit labeling: **The current dynamic CPU graph is NOT ready for direct HTP staging**.
   - Preserved the 5-step pre-staging checklist (shape specialization, KV cache I/O, op verification, quantization, CPU parity).
   - Marked all unexecuted staging commands and profiler criteria as unverified provisional templates.

---

## 2. Categorized Evidence Summary

### 2.1 Measured Evidence (Local Host Execution)

| Item | Measured Metric | Tolerance / Baseline | Verdict |
|---|---|---|---|
| **ONNX Artifact Digest** | `smollm2-135m-instruct.onnx` SHA256: `59594bf...e167` | Preserved original manifest | **MATCH** |
| **ONNX File Size** | 652,561,201 bytes | Preserved original manifest | **MATCH** |
| **Actual Graph Default Opset** | Opset 17 (from ONNX default domain `ai.onnx` import) | Requested opset: 17 | **MATCH** |
| **Single Forward-Pass Max Diff** | `0.0007988` (tokens `[1, 2, 3, 4]`) | $\le 1.0 \times 10^{-3}$ | **PASS** |
| **Single Forward-Pass Mean Diff** | `0.0001188` (tokens `[1, 2, 3, 4]`) | $\le 2.0 \times 10^{-4}$ | **PASS** |
| **Single Forward-Pass Argmax Tokens** | `[28, 198, 376, 33]` (both PyTorch and ONNX) | 100% token match | **PASS** |
| **Autoregressive Evaluated Steps** | 12 greedy decode steps across 3 real tokenized prompts | 12 steps | - |
| **Autoregressive Max Logit Diff** | `0.0001030` across all steps | $\le 1.0 \times 10^{-3}$ | **PASS** |
| **Autoregressive Mean Logit Diff** | `1.445 \times 10^{-5}$` across all steps | $\le 2.0 \times 10^{-4}$ | **PASS** |
| **Autoregressive Token Agreement** | **100.0% (12 / 12 steps match PyTorch CPU)** | 100.0% | **PASS** |
| **Parity Active Under `python -O`**| Subprocess exit code `42` on deliberate tolerance breach | `ParityVerificationError` active | **PASS** |

### 2.2 Derived Evidence (Architecture Parameter Calculations)

Formulas derived for SmolLM2-135M (30 layers, 3 KV heads, 64 head dim):
- $\text{One-Way Payload}(T, B_{\text{elem}}) = 2 \times 30 \times 3 \times T \times 64 \times B_{\text{elem}}$
- $\text{Round-Trip Traffic}(T, B_{\text{elem}}) = 2 \times \text{One-Way Payload}$

| Token Count ($T$) | Element Format | One-Way Payload | Round-Trip Traffic |
|---|---|---|---|
| **512 tokens** | FP16 ($B=2$ bytes) | **11,796,480 bytes (11.25 MiB)** | **23,592,960 bytes (22.50 MiB)** |
| **512 tokens** | INT8 ($B=1$ byte) | **5,898,240 bytes (5.625 MiB)** | **11,796,480 bytes (11.25 MiB)** |
| **2048 tokens** | FP16 ($B=2$ bytes) | **47,185,920 bytes (45.00 MiB)** | **94,371,840 bytes (90.00 MiB)** |
| **2048 tokens** | INT8 ($B=1$ byte) | **23,592,960 bytes (22.50 MiB)** | **47,185,920 bytes (45.00 MiB)** |

### 2.3 Assumed Evidence (Theoretical Analytical Transfer Model)

Formula: $t_{\text{transfer}} = \frac{\text{RoundTripBytes}}{B_{\text{bus}} \times 10^9} + t_{\text{driver}}$ (for 512 tokens FP16 round-trip: 23,592,960 bytes).

| Assumed Effective Bus Bandwidth ($B_{\text{bus}}$) | Assumed Driver Overhead ($t_{\text{driver}}$) | Estimated Transfer Latency | Classification |
|---|---|---|---|
| 2.0 GB/s | 0.5 ms | **12.3 ms** | Theoretical model (NOT measured) |
| 4.0 GB/s | 0.5 ms | **6.4 ms** | Theoretical model (NOT measured) |
| 8.0 GB/s | 0.3 ms | **3.25 ms** | Theoretical model (NOT measured) |

### 2.4 Unverified Hardware Claims (Hardware Absent)

| Claim / Metric | Status | Evidence Requirement |
|---|---|---|
| **Snapdragon Device Execution** | `BLOCKED` | Physical Snapdragon SoC reachable via ADB |
| **Hexagon HTP Offload** | `UNVERIFIED` | Verified execution log confirming `QnnHtpContext` and node placement |
| **Measured NPU Latency & Energy** | `UNAVAILABLE` | Physical device benchmark logs with battery / thermal profiling |
| **VTCM SRAM Residency** | `UNVERIFIED` | Binary QNN profiler log parsed with `qnn-profile-viewer` showing 0 DDR spills |
| **FastRPC Shared Memory Throughput**| `UNVERIFIED` | Kernel driver `/dev/adsprpc-smd` hardware benchmark |

---

## 3. Hardware & Software Inventory Table

| Component | Status | Details / Classification |
|---|---|---|
| **Host Workstation** | Installed | Apple M2 (`darwin-arm64`, macOS) |
| **Python Virtualenv** | Installed | `llm-inference/runtime/.venv` (Python 3.11.11, PyTorch 2.15.0) |
| **Host ADB Tool** | Installed | `/opt/homebrew/bin/adb` (version 1.0.41) |
| **Host ONNX Libraries** | Installed | `onnx==1.23.2`, `onnxruntime==1.31.1` (Providers: `CoreMLExecutionProvider`, `CPUExecutionProvider`) |
| **Model Checkpoint** | Installed | HuggingFaceTB SmolLM2-135M-Instruct (`12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0) |
| **Snapdragon SoC Target** | **MISSING** | `adb devices` returned 0 devices; no hardware attached |
| **Qualcomm FastRPC Driver**| **MISSING** | `/dev/adsprpc-smd` absent on host |
| **Hexagon DSP Firmware** | **MISSING** | DSP firmware partitions absent on host |
| **Qualcomm QNN Native Libs**| **MISSING** | `libQnnHtp.so`, `libQnnSystem.so` absent on host |
| **QNNExecutionProvider** | **MISSING on Host**| Not built for `darwin-arm64` macOS wheels |

---

## 4. Changed and Created Files

| File | Status | Description |
|---|---|---|
| `llm-inference/runtime/src/qedge_runtime/qualcomm/provenance.py` | Created | Comprehensive export provenance validation module; validates manifest contract, actual graph default opset, and graph interfaces (`input_ids` INT64 rank 2, `logits` FLOAT32 rank 3); defines `ProvenanceValidationError` and `ParityVerificationError`. |
| `llm-inference/runtime/src/qedge_runtime/qualcomm/runner.py` | Repaired | Closed hardware evidence bypass (hardware status remains strictly `UNVERIFIED`); validates full export provenance before session creation; separates `cpu_only` from `unverified_qnn` / `mixed_or_accelerator` sessions. |
| `llm-inference/runtime/src/qedge_runtime/qualcomm/export.py` | Repaired | Replaced Python `assert` with active `ParityVerificationError` checks active under `python -O`; enforces finite metric checks and conditional token matching counts; rejects empty evaluations; shares provenance validation for reuse. |
| `llm-inference/runtime/src/qedge_runtime/qualcomm/compatibility.py` | Repaired | Removed unverified TOPS numbers; labeled provisional target profile; detailed 5-step pre-staging preparation checklist. |
| `llm-inference/runtime/src/qedge_runtime/qualcomm/device_runbook.md` | Repaired | Labeled as provisional template; removed universal TOPS and throughput claims; warned dynamic CPU graph is not ready for HTP staging; marked unexecuted commands and criteria as unverified. |
| `llm-inference/runtime/src/qedge_runtime/qualcomm/cache_analysis.py` | Repaired | Derived byte counts strictly from model config; marked physical latencies unmeasured; exposed analytical transfer model with explicit assumptions. |
| `llm-inference/runtime/tests/test_qualcomm_deployment.py` | Expanded / Repaired | Added negative provenance probes (wrong revision, missing opset, interface mismatch, wrong model ID); added CPU and QNN fabricated-evidence bypass tests; added active parity verification test under `python -O` (11 tests passed). |
| `llm-inference/models/smollm2-135m-instruct-onnx/export_manifest.json` | Preserved | Preserved original historical export manifest from 2026-11-05 intact. |
| `handoff/research-review/11-qualcomm-deployment/results.json` | Regenerated | Restructured into measured, derived, assumed, and unverified sections; recorded hardware governance and negative probe outcomes. |
| `handoff/research-review/11-qualcomm-deployment.md` | Updated | This handoff artifact detailing all Stage 11 repairs. |

---

## 5. Verification Commands and Exit Codes

All commands executed from repository root:

```bash
# 1. Historical manifest checksum verification (30/30 verified)
python3 scripts/verify_historical_hashes.py
# Exit Code: 0
# Output: historical manifest: 30 verified, 0 failure(s)

# 2. Stage 11 Repaired Qualcomm deployment unit tests (11 passed)
PYTHONPATH=qedge-common/src:llm-inference/runtime/src llm-inference/runtime/.venv/bin/python -m unittest llm-inference/runtime/tests/test_qualcomm_deployment.py
# Exit Code: 0
# Output: Ran 11 tests in 40.304s -- OK

# 3. Active parity verification under optimized Python (exit code 42 on deliberate breach)
PYTHONPATH=qedge-common/src:llm-inference/runtime/src llm-inference/runtime/.venv/bin/python -m unittest llm-inference/runtime/tests/test_qualcomm_deployment.py -k test_parity_checks_active_under_normal_and_optimized_python
# Exit Code: 0
# Output: Ran 1 test in 19.415s -- OK

# 4. Shared schema tests in qedge-common (49 passed)
PYTHONPATH=qedge-common/src edge-rag-demo/.venv/bin/python -m unittest discover qedge-common/tests
# Exit Code: 0
# Output: Ran 49 tests in 0.161s -- OK

# 5. edge-rag-demo test suite (133 passed, 1 skipped)
PYTHONPATH=qedge-common/src:edge-rag-demo/src edge-rag-demo/.venv/bin/python -m unittest discover edge-rag-demo/tests
# Exit Code: 0
# Output: Ran 134 tests in 1.265s -- OK (skipped=1)
# Note: "Ran 134, skipped=1" denotes 133 passed and 1 skipped.
```

---

## 6. Preserved Boundaries & Remaining Gates

1. **Hardware Execution Status:**
   - Qualcomm NPU hardware offload and device performance remain strictly **BLOCKED** and **UNVERIFIED**.
   - No hardware benchmark claims or synthetic throughput figures are reported.
2. **Human Retrieval Quality Gate:**
   - Software repairs for catalog retrieval are accepted (105/105 tests).
   - Human retrieval quality remains **BLOCKED** pending annotation of the 1,009 active review queue candidates and completion of the 120-query evaluation.
3. **Secondary Model Validation:**
   - `Qwen/Qwen2.5-0.5B-Instruct` remains **DEFERRED / BLOCKED** per the execution contract.
4. **Generalization Evaluation:**
   - Independent generalization benchmark remains **PENDING**.
5. **Historical Hashes:**
   - 30/30 historical manifest hashes verified byte-for-byte.

---

## 7. Conclusion

Stage 11 bounded repairs are complete and verified. Bounded self-review verdict: `ACCEPTED`.
We stop here for independent review before proceeding to Stage 12.

## Final dimension repair — 9 November 2026

The implementing reviewer completed the remaining dimension-consistency repair.
Every manifest input/output axis now matches the physical graph's fixed value or
symbolic name exactly. Positive integer sizes and nonempty whitespace-free symbols
are accepted only in their corresponding graph axis kind. Boolean, object, negative,
unknown and fixed/dynamic contradictions are rejected. Historical manifests and
model artifacts were not rewritten.

New tests use disposable small ONNX graphs and exercise the shared validator,
verified runner loading before session construction, and the export reuse/rebuild
boundary. They do not re-export large models for each malformed field. The existing
full suite still exercises real checkpoint export, reuse and CPU parity.

Validation: 2 new dimension tests passed; full runtime 137/137 passed, exit 0.
Current historical hashes: 30/30. Evidence is in
[Stage 12 logs](12-release-review/logs/runtime-tests.log) and
[dimension tests](12-release-review/logs/dimensions.log).
No native process failure was observed in this run. The earlier exit-139 cause is
not established by that observation.

Final bounded verdict: **ACCEPTED for local CPU export/provenance validation**.
This is implementing-agent verification, not a separate independent approval.
**Target graph preparation, device execution and device performance remain BLOCKED**;
NPU remains UNVERIFIED. A provisional runbook is not a completed device deployment.
