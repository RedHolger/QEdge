/**
 * qnn_types.h — Qualcomm AI Engine Direct SDK stub types and profiling macros.
 *
 * PURPOSE: This file stubs out the Qualcomm QNN SDK types and profiling macros
 * so QnnStubBackend compiles on any Mac without the actual Qualcomm AI Engine
 * Direct SDK installed.
 *
 * TO PORT TO REAL HARDWARE: Replace this file with the actual QNN SDK headers:
 *   #include "QnnInterface.h"
 *   #include "QnnProfile.h"
 *   #include "QnnGraph.h"
 *   #include "System/QnnSystemInterface.h"
 * And link against:
 *   libQnn.so / QnnHtp.dll / QnnCpu.dll (from Qualcomm AI Engine Direct SDK)
 *
 * QAIRT SDK download: https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk
 */

#pragma once
#include <chrono>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <atomic>

// ---------------------------------------------------------------------------
// Stub: QNN Op name constants
// These match actual QNN SDK op names in QnnOpDef.h
// ---------------------------------------------------------------------------
#define QNN_OP_MAT_MUL          "QNN_OP_MAT_MUL"
#define QNN_OP_ELEMENT_WISE_ADD "QNN_OP_ELEMENT_WISE_ADD"
#define QNN_OP_ELEMENT_WISE_MUL "QNN_OP_ELEMENT_WISE_MUL"
#define QNN_OP_SOFTMAX          "QNN_OP_SOFTMAX"
#define QNN_OP_LAYER_NORM       "QNN_OP_LAYER_NORM"      // maps to rms_norm
#define QNN_OP_GELU             "QNN_OP_GELU"            // maps to silu_mul
#define QNN_OP_TRANSPOSE        "QNN_OP_TRANSPOSE"
#define QNN_OP_GATHER           "QNN_OP_GATHER"          // maps to embedding_lookup
#define QNN_OP_MULTI_HEAD_ATTN  "QNN_OP_MULTI_HEAD_ATTENTION"
#define QNN_OP_ROPE             "QNN_OP_ROTARY_EMBEDDING" // maps to rope

// ---------------------------------------------------------------------------
// Stub: QNN profile context (real SDK: QnnProfile_Handle_t)
// ---------------------------------------------------------------------------
struct QnnProfileStats {
    uint64_t call_count   = 0;
    double   total_us     = 0.0;
    double   total_flops  = 0.0;
    double   total_bytes  = 0.0;
};

// Per-op profile accumulator — thread-local start time
struct QnnProfileContext {
    std::unordered_map<std::string, QnnProfileStats> ops;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
    std::string _current_op;
};

// ---------------------------------------------------------------------------
// Stub profiling macros (real SDK: Qnn_ErrorHandle_t QnnProfile_getEventData)
// ---------------------------------------------------------------------------
#define QNN_PROFILE_OP_START(ctx, op_name, flops, bytes) \
    do { \
        (ctx)._current_op = (op_name); \
        (ctx).ops[(op_name)].total_flops += (flops); \
        (ctx).ops[(op_name)].total_bytes += (bytes); \
        (ctx)._start = std::chrono::high_resolution_clock::now(); \
    } while(0)

#define QNN_PROFILE_OP_END(ctx) \
    do { \
        auto _end = std::chrono::high_resolution_clock::now(); \
        double _us = std::chrono::duration<double, std::micro>(_end - (ctx)._start).count(); \
        auto& _s = (ctx).ops[(ctx)._current_op]; \
        _s.total_us += _us; \
        _s.call_count++; \
    } while(0)

// ---------------------------------------------------------------------------
// QnnConfig — Qualcomm target hardware configuration
// Real SDK equivalent: Qnn_DeviceConfig_t / QnnHtp_DeviceConfig_t
// ---------------------------------------------------------------------------
struct QnnConfig {
    std::string target_chip = "synthetic-cpu-stub";  // no device is selected
    double      npu_tops    = 0.0;                  // no device projection in Stage 01
    uint32_t    sram_mb     = 4;                     // placeholder SRAM model
    bool        use_fp16    = true;                  // future device option
    bool        verbose     = false;
};
