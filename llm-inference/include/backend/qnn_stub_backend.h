/**
 * qnn_stub_backend.h — QnnStubBackend: CPU-backed QNN interface stub.
 *
 * Implements the Backend interface by delegating computation to CPUBackend
 * while wrapping every op in QNN SDK-style profiling macros and accumulating
 * a per-op PowerReport.
 *
 * This demonstrates the integration point for a real HexagonNPUBackend:
 * replace the CPU delegate calls with actual QNN graph execution via
 * QnnGraph_execute() / QnnContext_create() once Snapdragon hardware is available.
 * No op in this file runs on an NPU.
 *
 * QAIRT backend interface: https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk
 */

#pragma once
#include "backend/backend.h"
#include "backend/cpu_backend.h"
#include "backend/qnn_types.h"
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

// Power report for one benchmark run across all policies
struct PowerReport {
    std::string policy_name;
    double      tokens_per_sec;
    double      total_wall_us;
    double      total_estimated_flops;
    double      total_bytes_moved;
    double      estimated_mj;      // bytes*0.5pJ + flops*0.3pJ → mJ
    double      tokens_per_mj;     // throughput / (estimated_mj / total_wall_us * 1e6)
    size_t      peak_memory_bytes;
    float       hit_rate;
};

class QnnStubBackend : public Backend {
public:
    explicit QnnStubBackend(const QnnConfig& cfg = QnnConfig{});
    ~QnnStubBackend() override;

    // --- Backend interface ---
    BackendType type()         const override { return BackendType::QNN; }
    std::string name()         const override { return "QnnStub (CPU delegate)"; }
    bool        is_available() const override { return true; }

    Tensor matmul(const Tensor& a, const Tensor& b) override;
    using  Backend::rope;
    Tensor rope(const Tensor& x, const std::vector<uint32_t>& positions,
                const RopeConfig& config) override;
    void   softmax_inplace(Tensor& x) override;
    Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) override;
    Tensor silu_mul(const Tensor& gate, const Tensor& up) override;
    Tensor add(const Tensor& a, const Tensor& b) override;
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                     const Tensor* mask = nullptr) override;
    Tensor fused_qk_v_attention(const Tensor& q, const Tensor& k, const Tensor& v) override;

    // --- QNN profiling ---
    void print_profile() const;
    void reset_profile();

    // Build a PowerReport given external throughput + cache stats
    PowerReport make_power_report(const std::string& policy_name,
                                   double tokens_per_sec,
                                   size_t peak_memory_bytes,
                                   float  hit_rate) const;

    const QnnConfig&        config()  const { return cfg_; }
    const QnnProfileContext& profile() const { return profile_; }

private:
    QnnConfig        cfg_;
    CPUBackend       cpu_;
    QnnProfileContext profile_;

    // Estimate FLOPs for common ops
    static double flops_matmul(const Tensor& a, const Tensor& b);
    static double flops_attention(const Tensor& q);
    static double bytes_tensor(const Tensor& t);
};
