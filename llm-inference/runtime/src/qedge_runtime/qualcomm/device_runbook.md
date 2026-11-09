# Qualcomm Snapdragon Device Execution Runbook (Stage 11 - Repaired)

**Document Status:** PROVISIONAL RUNBOOK TEMPLATE (Pending Physical Qualcomm Hardware and SDK Access)  
**Host Environment:** Apple Silicon Mac (`darwin-arm64`, macOS)  
**Hardware Status:** `UNVERIFIED` (No Snapdragon device reachable via ADB)  
**Device Execution Status:** `BLOCKED`  
**Device Performance Status:** `BLOCKED`  

---

## 1. Critical Preparation Prerequisites (Do NOT Stage Dynamic CPU Graph)

> [!WARNING]
> **Artifact Readiness Notice:** The currently exported artifact (`smollm2-135m-instruct.onnx`) is a **dynamic-shape, full-prefix CPU baseline graph** (`use_cache=False`).
> According to official [ONNX Runtime QNN EP Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html), the **Hexagon HTP backend strictly requires static/fixed shapes**. Staging the dynamic CPU graph directly to HTP will trigger graph compilation failure or immediate CPU fallback.

Before staging any artifact to physical hardware, the following 5 preparation steps must be completed:

1. **Shape Specialization:**
   - Specialize inputs to fixed dimensions:
     * Prefill graph: fixed shape `[batch=1, seq_len=P]` (e.g. $P=128$).
     * Decode graph: fixed shape `[batch=1, seq_len=1]`.
   - Dynamic axes (`batch_size`, `seq_len`) must be replaced with static integers or explicit profile shapes.
2. **Stateful / I/O KV Cache Interface:**
   - Migrate from prefix recomputation (`use_cache=False`) to explicit key/value tensor inputs and outputs:
     * Inputs: `past_key_values.{layer}.key`, `past_key_values.{layer}.value` of shape `[1, kv_heads=3, max_seq_len, head_dim=64]`.
     * Outputs: `present_key_values.{layer}.key`, `present_key_values.{layer}.value`.
3. **Op Package Verification:**
   - Validate every operator in the specialized graph against the Qualcomm HTP Op Package using the QNN SDK tool `qnn-op-check` or ONNX QNN op mapping table.
4. **Quantization (W8A16 or W4A16):**
   - Hexagon Vector Extensions (HVX) are optimized for INT8/INT4 math.
   - FP32 is rejected by HTP. FP16 is supported with flag `enable_htp_fp16_precision: 1`, but achieves lower throughput than INT8.
   - Run activation calibration and weight quantization using Qualcomm AIMET or QNN Model Tools.
5. **Pre-Staging Parity Check on CPU:**
   - Verify that the specialized, quantized graph achieves acceptable numerical tolerance on CPU before transferring to device.

---

## 2. Target Profile (Provisional Specification)

| Specification | Provisional Target Profile | Alternative Target Profile |
|---|---|---|
| **SoC Model** | Qualcomm Snapdragon 8 Gen 3 (SM8650) | Snapdragon 8 Gen 2 (SM8550) or 8 Elite (SM8750) |
| **HTP Architecture** | Hexagon HTP v75 | Hexagon HTP v73 or v79 |
| **Operating System** | Android 14+ (`arm64-v8a`) | Linux `aarch64` or Windows 11 ARM64 (`arm64x`) |
| **Qualcomm SDK** | QNN SDK v2.18+ or QAIRT v2.28+ | Minimum version matching target SoC |
| **Inference Engine** | ONNX Runtime v1.20+ built with QNN EP | ONNX Runtime v1.31+ with QNN EP |
| **Backend Library** | `libQnnHtp.so` (Android/Linux) | `QnnHtp.dll` (Windows ARM64) |
| **Skeleton Library** | `libQnnHtpV75Skel.so` (matches v75 HTP) | `libQnnHtpV73Skel.so` (v73) or `libQnnHtpV79Skel.so` (v79) |

*Note:* `soc_model` is an integer platform identifier defined in QNN SDK headers (`QnnHtpDevice.h`). Setting `soc_model: 0` enables QNN auto-detection from system properties (`ro.board.platform`). It should not be hard-coded to a single magic constant across different SoC families.

---

## 3. Device Staging via ADB (Provisional Template)

> [!NOTE]
> The commands below represent the executable staging sequence for physical Snapdragon hardware once accessible. They are currently **UNEXECUTED PROVISIONAL TEMPLATES** because no physical hardware is attached.

```bash
# 1. Verify ADB connection and detect actual SoC platform
adb devices -l
# Detect target SoC to choose the matching skeleton library (v73, v75, or v79)
SOC_PLATFORM=$(adb shell getprop ro.board.platform | tr -d '\r')
echo "Detected SoC: $SOC_PLATFORM"

# 2. Create execution directory on device
adb shell "mkdir -p /data/local/tmp/qedge/models /data/local/tmp/qedge/qnn_libs"

# 3. Push specialized and quantized ONNX model + export manifest
adb push models/smollm2-135m-specialized/smollm2-135m-htp.onnx /data/local/tmp/qedge/models/
adb push models/smollm2-135m-specialized/export_manifest.json /data/local/tmp/qedge/models/

# 4. Push matching QNN SDK libraries from $QNN_SDK_ROOT
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/qedge/qnn_libs/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/qedge/qnn_libs/

# Push skeleton matching detected HTP generation (e.g. v75 for SM8650)
if [ "$SOC_PLATFORM" = "sm8650" ]; then
    adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so /data/local/tmp/qedge/qnn_libs/
elif [ "$SOC_PLATFORM" = "sm8550" ]; then
    adb push $QNN_SDK_ROOT/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so /data/local/tmp/qedge/qnn_libs/
elif [ "$SOC_PLATFORM" = "sm8750" ]; then
    adb push $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so /data/local/tmp/qedge/qnn_libs/
fi

# 5. Push ONNX Runtime performance test binary
adb push $ORT_BUILD_DIR/onnxruntime_perf_test /data/local/tmp/qedge/
adb shell "chmod +x /data/local/tmp/qedge/onnxruntime_perf_test"
```

---

## 4. On-Device Execution & Profiling (Provisional Template)

### 4.1 Environment Setup

```bash
adb shell
cd /data/local/tmp/qedge

export LD_LIBRARY_PATH=/data/local/tmp/qedge/qnn_libs:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/qedge/qnn_libs;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp"
```

### 4.2 Benchmark Command with Profiler Options

Execute inference targeting Hexagon HTP with profile generation enabled:

```bash
/data/local/tmp/qedge/onnxruntime_perf_test \
  -e qnn \
  -m /data/local/tmp/qedge/models/smollm2-135m-htp.onnx \
  -o 3 \
  -r 50 \
  -w 10 \
  --qnn_options "backend_path:libQnnHtp.so,htp_performance_mode:burst,soc_model:0,htp_graph_finalization_optimization_mode:3" \
  --profile_file_prefix /data/local/tmp/qedge/qnn_profile \
  > /data/local/tmp/qedge/execution_log.txt 2>&1
```

*Notes on Options:*
- `backend_path:libQnnHtp.so`: Selects Hexagon HTP backend.
- `soc_model:0`: Directs QNN runtime to auto-detect SoC from hardware properties.
- `htp_performance_mode:burst`: Locks clock frequencies to maximum burst state during benchmarking.
- `--profile_file_prefix`: Enables QNN binary profiling for execution time and memory analysis.

---

## 5. Verification Criteria for Hardware Acceptance (Unverified - Pending Device Run)

To transition hardware status from `UNVERIFIED` to `VERIFIED_HTP`, the execution run must satisfy all 4 criteria below. Mere presence of environment variables or registered execution providers is strictly rejected.

### 5.1 Node Placement & Graph Assignment Verification
- Inspect execution log (`execution_log.txt`) and QNN session initialization.
- Confirm successful creation of `QnnHtpContext`:
  ```text
  [INFO] QNN execution provider initialized successfully.
  [INFO] Backend loaded: libQnnHtp.so (Hexagon HTP backend).
  [INFO] Finalized HTP context.
  ```
- Any operator assigned to `CPUExecutionProvider` or inserted CPU wrapper must be reported. If the entire graph falls back to CPU, offload is **REJECTED**.

### 5.2 Parity Acceptance Tolerance
- Target HTP FP16 / W8A16 logits must be evaluated against the CPU float32 baseline:
  * Maximum Absolute Difference: $\le 1.5 \times 10^{-2}$ (due to quantization / FP16 rounding).
  * Mean Absolute Difference: $\le 2.0 \times 10^{-3}$.
  * Greedy Next-Token Agreement: $\ge 98.0\%$ across evaluated test sequences.

### 5.3 Cold/Warm Latency & Thermal Logging
- Measure cold start latency (first token, including graph finalization and FastRPC initialization).
- Measure warm decode latency over 50 consecutive iterations.
- Record battery / SoC package temperature before and after the run to evaluate thermal throttling.

### 5.4 VTCM SRAM Residency Verification via Profiler
- Tightly Coupled Memory (VTCM) residency cannot be proven through theoretical byte arithmetic.
- Pull and parse the binary QNN profiler log:
  ```bash
  adb pull /data/local/tmp/qedge/qnn_profile* ./logs/
  python3 $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
    --input ./logs/qnn_profile.raw \
    --output ./logs/qnn_profile.csv
  ```
- Inspect `TCM_ALLOC` and `VTCM_SPILL` metrics to verify zero DDR spilling during the decode loop.

---

## 6. Current Environment Status

- **Host Device:** Apple M2 (`darwin-arm64`, macOS).
- **Qualcomm Snapdragon Hardware:** None attached via ADB (`adb devices` returned 0 devices).
- **Device Execution Status:** `BLOCKED` (hardware unavailable).
- **Device Performance Status:** `BLOCKED` (hardware unavailable).
- **NPU Status:** `UNVERIFIED` (CPU baseline export verified).
