"""Qualcomm deployment compatibility matrix and route selection (Stage 11 - Repaired).

Defines:
- Single selected deployment route: ONNX Runtime with QNN Execution Provider (QNN EP).
- Provisional target profile (clearly labeled provisional pending physical hardware access).
- Multi-generation target matrix (SM8550, SM8650, SM8750, Snapdragon X Elite) citing primary documentation.
- Mandatory pre-device deployment preparation checklist:
  1. Shape specialization (static shapes for HTP).
  2. KV cache tensor interface (eliminating prefix recomputation).
  3. Op package verification.
  4. Quantization to W8A16 / W4A16.
  5. Pre-staging CPU parity on fixed shapes.
"""
from __future__ import annotations

from typing import Any


def get_qualcomm_compatibility_matrix() -> dict[str, Any]:
    """Return the verified Qualcomm deployment compatibility matrix and preparation specifications."""
    return {
        "selected_route": {
            "name": "ONNX Runtime with QNN Execution Provider (QNN EP)",
            "target_backend": "HTP (Hexagon Tensor Processor via libQnnHtp.so / QnnHtp.dll)",
            "primary_documentation": [
                "https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html",
                "https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk",
                "https://aihub.qualcomm.com/models/smollm",
            ],
            "rationale": (
                "ONNX Runtime QNN EP provides a single, officially supported deployment path "
                "supporting standard ONNX computational graphs, versioned partition fallback controls, "
                "and native Hexagon HTP offload without requiring multiple speculative backends."
            ),
        },
        "provisional_target_profile": {
            "status": "PROVISIONAL (pending physical hardware and SDK access)",
            "soc": "Snapdragon 8 Gen 3 (SM8650)",
            "npu_architecture": "Hexagon HTP v75",
            "target_os": "Android 14+ (arm64-v8a)",
            "target_qnn_sdk_version": "2.18+ or 2.28+",
            "target_precision": "W8A16 or W4A16 (preferred for HVX vector units)",
            "backend_library": "libQnnHtp.so",
            "skeleton_library": "libQnnHtpV75Skel.so",
        },
        "target_hardware_matrix": [
            {
                "soc": "Snapdragon 8 Gen 2 (SM8550)",
                "htp_architecture": "v73",
                "supported_os": ["Android 13+", "Linux aarch64"],
                "min_qnn_sdk_version": "2.14.0",
                "backend_library": "libQnnHtp.so",
                "skel_library": "libQnnHtpV73Skel.so",
                "preferred_precision": "W8A16 or FP16",
            },
            {
                "soc": "Snapdragon 8 Gen 3 (SM8650)",
                "htp_architecture": "v75",
                "supported_os": ["Android 14+", "Linux aarch64"],
                "min_qnn_sdk_version": "2.18.0",
                "backend_library": "libQnnHtp.so",
                "skel_library": "libQnnHtpV75Skel.so",
                "preferred_precision": "W8A16, W4A16, or FP16",
            },
            {
                "soc": "Snapdragon X Elite (X1E-80-100 / X1E-84-100)",
                "htp_architecture": "v73 / v75 variant",
                "supported_os": ["Windows 11 ARM64 (Build 26100+)"],
                "min_qnn_sdk_version": "2.20.0",
                "backend_library": "QnnHtp.dll",
                "skel_library": "QnnHtpV73Skel.dll / QnnHtpV75Skel.dll",
                "preferred_precision": "W8A16, W4A16, or FP16",
            },
            {
                "soc": "Snapdragon 8 Elite (SM8750)",
                "htp_architecture": "v79",
                "supported_os": ["Android 15+"],
                "min_qnn_sdk_version": "2.28.0",
                "backend_library": "libQnnHtp.so",
                "skel_library": "libQnnHtpV79Skel.so",
                "preferred_precision": "W8A16, W4A16, or FP16",
            },
        ],
        "qnn_ep_option_specifications": {
            "backend_path": "Absolute path to backend library (e.g. libQnnHtp.so on Android, QnnHtp.dll on Windows).",
            "soc_model": (
                "Integer SoC platform identifier specified in QNN headers (QnnHtpDevice.h). "
                "Default is 0 (auto-detect by QNN runtime from ro.board.platform). Not a universal constant."
            ),
            "enable_htp_fp16_precision": "Set to '1' for FP16 execution mode; default is '0' (INT8 quantized mode).",
            "htp_graph_finalization_optimization_mode": "Optimization level 0-3 (level 3 performs max graph optimization).",
        },
        "pre_staging_preparation_checklist": [
            {
                "step": "1. Shape Specialization",
                "status": "REQUIRED BEFORE STAGING (current export has dynamic shapes)",
                "detail": (
                    "ONNX Runtime QNN EP documentation explicitly states that the Hexagon HTP backend "
                    "requires static/fixed input shapes or explicit profile shapes. The current export "
                    "has dynamic axes [batch_size, seq_len] which works on CPU but will fail or trigger "
                    "CPU fallback on HTP. Static shapes (e.g. prompt length 128, decode length 1) must be specialized."
                ),
            },
            {
                "step": "2. KV Cache Tensor Interface",
                "status": "REQUIRED BEFORE STAGING (current export uses use_cache=False)",
                "detail": (
                    "The current exported graph recomputes the entire prefix on every decode step. "
                    "For efficient hardware deployment, explicit past_key_values and present_key_values "
                    "stateful or I/O tensors must be wired in the ONNX graph."
                ),
            },
            {
                "step": "3. Operator Verification",
                "status": "REQUIRED BEFORE STAGING",
                "detail": (
                    "All operators in the specialized graph must be validated against the target HTP Op Package "
                    "using QNN tool 'qnn-op-check' or ONNX Runtime QNN supported op list."
                ),
            },
            {
                "step": "4. Quantization to W8A16 / W4A16",
                "status": "RECOMMENDED / REQUIRED FOR PEAK EFFICIENCY",
                "detail": (
                    "Hexagon Vector eXtensions (HVX) deliver peak throughput on INT8/INT4 tensors. "
                    "FP32 is unsupported on HTP; FP16 is supported with flag enable_htp_fp16_precision: 1 "
                    "but has lower throughput than W8A16. Quantization requires calibration data via AIMET or QNN tools."
                ),
            },
            {
                "step": "5. Pre-Staging Parity Verification on CPU",
                "status": "MANDATORY BEFORE DEVICE EXECUTION",
                "detail": (
                    "Specialized and quantized graphs must achieve acceptable numerical parity on CPU "
                    "against the float32 reference baseline before being pushed to device."
                ),
            },
        ],
        "current_export_readiness_status": {
            "graph_type": "Dynamic-shape CPU baseline graph",
            "is_ready_for_direct_htp_staging": False,
            "readiness_verdict": "NOT READY FOR DIRECT HTP STAGING (Requires shape specialization and KV cache interface)",
        },
    }
