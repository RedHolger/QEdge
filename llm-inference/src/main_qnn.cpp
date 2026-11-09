/**
 * main_qnn.cpp — CPU-stub power-model report entry point.
 *
 * Runs all 4 KV-cache policies (FIFO, LRU, KeyDiff, QUOKA) against QnnStubBackend
 * and prints the per-policy power report table (tok/s, Peak MB, Hit%, Est.mJ, tok/mJ).
 * Nothing in this path executes on an NPU.
 */
#include "backend/qnn_stub_backend.h"
#include "benchmark/benchmark.h"
#include "benchmark/power_report.h"
#include "core/config.h"
#include <iostream>

int main() {
    std::cout << "\n";
    std::cout << "  QEdge Runtime — QnnStub CPU fallback report (Stage 01)\n";
    std::cout << "  This is not an NPU run. Snapdragon/QNN hardware: unverified.\n";
    std::cout << "  Numbers below are synthetic CPU-stub estimates, not device power.\n";
    std::cout << std::endl;

    ModelConfig model_cfg;
    model_cfg.num_layers         = 22;
    model_cfg.num_heads          = 32;
    model_cfg.num_kv_heads       = 8;
    model_cfg.head_dim           = 64;
    model_cfg.hidden_size        = 2048;
    model_cfg.intermediate_size  = 5632;
    model_cfg.kv_cache_capacity  = 256;

    RuntimeConfig runtime_cfg;
    runtime_cfg.num_bench_tokens = 64;
    runtime_cfg.num_warmup       = 4;

    run_qnn_power_benchmark(model_cfg, runtime_cfg);

    std::cout << "  QNN interface mapping (design only; not executed):\n";
    std::cout << "  ─────────────────────────────────────────────────────────\n";
    std::cout << "  matmul           → QNN_OP_MAT_MUL\n";
    std::cout << "  rope             → QNN_OP_ROTARY_EMBEDDING\n";
    std::cout << "  softmax          → QNN_OP_SOFTMAX\n";
    std::cout << "  rms_norm         → QNN_OP_LAYER_NORM\n";
    std::cout << "  silu_mul         → QNN_OP_GELU\n";
    std::cout << "  add              → QNN_OP_ELEMENT_WISE_ADD\n";
    std::cout << "  attention        → QNN_OP_MULTI_HEAD_ATTENTION\n";
    std::cout << "  fused_qk_v_attn  → QNN_OP_MULTI_HEAD_ATTENTION\n";
    std::cout << "  ─────────────────────────────────────────────────────────\n\n";

    return 0;
}
