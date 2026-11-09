#include "backend/qnn_stub_backend.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

QnnStubBackend::QnnStubBackend(const QnnConfig& cfg) : cfg_(cfg) {
    if (cfg_.verbose) {
        std::cout << "[QnnStub] Initialised target=" << cfg_.target_chip
                  << " tops=" << cfg_.npu_tops
                  << " sram=" << cfg_.sram_mb << "MB\n";
    }
}

QnnStubBackend::~QnnStubBackend() {
    if (cfg_.verbose) {
        print_profile();
    }
}

// ---------------------------------------------------------------------------
// Static helpers: FLOPs / bytes estimation
// ---------------------------------------------------------------------------

double QnnStubBackend::flops_matmul(const Tensor& a, const Tensor& b) {
    // 2*M*N*K for a matrix multiply (one mul + one add per element)
    if (a.shape().size() < 2 || b.shape().size() < 2) return 0.0;
    double M = static_cast<double>(a.shape()[a.shape().size()-2]);
    double K = static_cast<double>(a.shape()[a.shape().size()-1]);
    double N = static_cast<double>(b.shape()[b.shape().size()-1]);
    return 2.0 * M * N * K;
}

double QnnStubBackend::flops_attention(const Tensor& q) {
    // Q·K^T: (seq, head_dim) x (head_dim, seq) = 2*seq*seq*head_dim
    if (q.shape().size() < 3) return 0.0;
    double heads   = static_cast<double>(q.shape()[0]);
    double seq     = static_cast<double>(q.shape()[1]);
    double hdim    = static_cast<double>(q.shape()[2]);
    return 4.0 * heads * seq * seq * hdim; // Q·K^T + softmax·V
}

double QnnStubBackend::bytes_tensor(const Tensor& t) {
    return static_cast<double>(t.size_bytes());
}

// ---------------------------------------------------------------------------
// Backend ops — delegate to CPU, wrap in QNN profiling macros
// ---------------------------------------------------------------------------

Tensor QnnStubBackend::matmul(const Tensor& a, const Tensor& b) {
    double flops = flops_matmul(a, b);
    double bytes = bytes_tensor(a) + bytes_tensor(b);
    QNN_PROFILE_OP_START(profile_, QNN_OP_MAT_MUL, flops, bytes);
    Tensor r = cpu_.matmul(a, b);
    QNN_PROFILE_OP_END(profile_);
    return r;
}

Tensor QnnStubBackend::rope(const Tensor& x, const std::vector<uint32_t>& positions,
                            const RopeConfig& config) {
    // RoPE: ~4 flops per element (cos/sin lookup + 2 muls + 2 adds → amortised)
    double flops = 4.0 * static_cast<double>(x.numel());
    double bytes = bytes_tensor(x) * 2.0;
    QNN_PROFILE_OP_START(profile_, QNN_OP_ROPE, flops, bytes);
    Tensor r = cpu_.rope(x, positions, config);
    QNN_PROFILE_OP_END(profile_);
    return r;
}

void QnnStubBackend::softmax_inplace(Tensor& x) {
    // Softmax: ~5 flops per element (max, exp, sum, divide)
    double flops = 5.0 * static_cast<double>(x.numel());
    double bytes = bytes_tensor(x) * 2.0;
    QNN_PROFILE_OP_START(profile_, QNN_OP_SOFTMAX, flops, bytes);
    cpu_.softmax_inplace(x);
    QNN_PROFILE_OP_END(profile_);
}

Tensor QnnStubBackend::rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    // RMSNorm: ~4 flops per element (square, mean, rsqrt, mul)
    double flops = 4.0 * static_cast<double>(x.numel());
    double bytes = bytes_tensor(x) + bytes_tensor(weight);
    QNN_PROFILE_OP_START(profile_, QNN_OP_LAYER_NORM, flops, bytes);
    Tensor r = cpu_.rms_norm(x, weight, eps);
    QNN_PROFILE_OP_END(profile_);
    return r;
}

Tensor QnnStubBackend::silu_mul(const Tensor& gate, const Tensor& up) {
    // SiLU·mul: ~4 flops per element (sigmoid approx + mul + elementwise mul)
    double flops = 4.0 * static_cast<double>(gate.numel());
    double bytes = bytes_tensor(gate) + bytes_tensor(up);
    QNN_PROFILE_OP_START(profile_, QNN_OP_GELU, flops, bytes);
    Tensor r = cpu_.silu_mul(gate, up);
    QNN_PROFILE_OP_END(profile_);
    return r;
}

Tensor QnnStubBackend::add(const Tensor& a, const Tensor& b) {
    double flops = static_cast<double>(a.numel());
    double bytes = bytes_tensor(a) + bytes_tensor(b);
    QNN_PROFILE_OP_START(profile_, QNN_OP_ELEMENT_WISE_ADD, flops, bytes);
    Tensor r = cpu_.add(a, b);
    QNN_PROFILE_OP_END(profile_);
    return r;
}

Tensor QnnStubBackend::attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor* mask) {
    double flops = flops_attention(q);
    double bytes = bytes_tensor(q) + bytes_tensor(k) + bytes_tensor(v);
    QNN_PROFILE_OP_START(profile_, QNN_OP_MULTI_HEAD_ATTN, flops, bytes);
    Tensor r = cpu_.attention(q, k, v, mask);
    QNN_PROFILE_OP_END(profile_);
    return r;
}

Tensor QnnStubBackend::fused_qk_v_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
    double flops = flops_attention(q);
    double bytes = bytes_tensor(q) + bytes_tensor(k) + bytes_tensor(v);
    QNN_PROFILE_OP_START(profile_, QNN_OP_MULTI_HEAD_ATTN, flops, bytes);
    Tensor r = cpu_.fused_qk_v_attention(q, k, v);
    QNN_PROFILE_OP_END(profile_);
    return r;
}

// ---------------------------------------------------------------------------
// Profile printing
// ---------------------------------------------------------------------------

void QnnStubBackend::print_profile() const {
    std::cout << "\n[QnnStub] Per-op profile (" << cfg_.target_chip << "):\n";
    std::cout << std::left
              << std::setw(28) << "Op"
              << std::setw(10) << "Calls"
              << std::setw(14) << "Total (ms)"
              << std::setw(16) << "Avg (us)"
              << std::setw(16) << "GFLOPs"
              << std::setw(12) << "MB moved"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    double grand_flops = 0.0, grand_bytes = 0.0, grand_us = 0.0;
    for (const auto& [op, s] : profile_.ops) {
        double avg_us = s.call_count > 0 ? s.total_us / s.call_count : 0.0;
        std::cout << std::left
                  << std::setw(28) << op
                  << std::setw(10) << s.call_count
                  << std::setw(14) << std::fixed << std::setprecision(2) << s.total_us / 1000.0
                  << std::setw(16) << std::fixed << std::setprecision(2) << avg_us
                  << std::setw(16) << std::fixed << std::setprecision(4) << s.total_flops / 1e9
                  << std::setw(12) << std::fixed << std::setprecision(2) << s.total_bytes / 1e6
                  << "\n";
        grand_flops += s.total_flops;
        grand_bytes += s.total_bytes;
        grand_us    += s.total_us;
    }
    std::cout << std::string(96, '-') << "\n";
    std::cout << "  Total GFLOPs: " << std::fixed << std::setprecision(4) << grand_flops / 1e9
              << "  |  Total MB moved: " << grand_bytes / 1e6
              << "  |  Total wall: " << grand_us / 1000.0 << " ms\n";

    std::cout << "  CPU-stub wall time: " << std::fixed << std::setprecision(3)
              << grand_us / 1000.0 << " ms (no NPU projection or device claim)\n";
    std::cout << "\n";
}

void QnnStubBackend::reset_profile() {
    profile_.ops.clear();
}

// ---------------------------------------------------------------------------
// PowerReport construction
// ---------------------------------------------------------------------------

PowerReport QnnStubBackend::make_power_report(const std::string& policy_name,
                                               double tokens_per_sec,
                                               size_t peak_memory_bytes,
                                               float  hit_rate) const {
    PowerReport pr;
    pr.policy_name        = policy_name;
    pr.tokens_per_sec     = tokens_per_sec;
    pr.peak_memory_bytes  = peak_memory_bytes;
    pr.hit_rate           = hit_rate;

    // Accumulate totals across all ops
    for (const auto& [op, s] : profile_.ops) {
        pr.total_wall_us          += s.total_us;
        pr.total_estimated_flops  += s.total_flops;
        pr.total_bytes_moved      += s.total_bytes;
    }

    // Energy model:
    //   DRAM read/write energy: ~0.5 pJ/byte  → bytes * 0.5e-12 J = bytes * 0.5e-9 mJ
    //   NPU op energy:          ~0.3 pJ/flop  → flops * 0.3e-12 J = flops * 0.3e-9 mJ
    double mj_dram  = pr.total_bytes_moved  * 0.5e-9;
    double mj_flops = pr.total_estimated_flops * 0.3e-9;
    pr.estimated_mj = mj_dram + mj_flops;

    // tokens_per_mj = tokens/sec / (mJ/sec)
    // mJ/sec = estimated_mj / (total_wall_us / 1e6)
    double wall_sec = pr.total_wall_us / 1e6;
    double mj_per_sec = (wall_sec > 0) ? pr.estimated_mj / wall_sec : 0.0;
    pr.tokens_per_mj = (mj_per_sec > 0) ? tokens_per_sec / mj_per_sec : 0.0;

    return pr;
}
