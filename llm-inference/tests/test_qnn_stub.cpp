/**
 * test_qnn_stub.cpp — Unit tests for QnnStubBackend + PowerReport
 *
 * 3 new tests (total suite = 21/21 with existing 18):
 *   1. test_qnn_stub_implements_all_ops
 *   2. test_power_report_is_positive
 *   3. test_qnn_stub_matches_cpu_output
 */
#include "backend/qnn_stub_backend.h"
#include "backend/cpu_backend.h"
#include "benchmark/benchmark.h"
#include "kv_cache/policy.h"
#include "core/config.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <random>

static int tests_passed = 0;
static int tests_failed = 0;

#define QNN_TEST(name, expr) do { \
  if (!(expr)) { \
    std::cerr << "  FAIL: " << name << " (" << #expr << ")\n"; \
    tests_failed++; \
  } else { \
    std::cout << "  PASS: " << name << "\n"; \
    tests_passed++; \
  } \
} while(0)

static ModelConfig small_cfg() {
    ModelConfig cfg;
    cfg.num_layers        = 2;
    cfg.num_heads         = 4;
    cfg.num_kv_heads      = 2;
    cfg.head_dim          = 32;
    cfg.kv_cache_capacity = 16;
    cfg.hidden_size       = 128;
    cfg.intermediate_size = 256;
    return cfg;
}

static Tensor rand_tensor(const std::vector<size_t>& shape, float seed = 1.0f) {
    Tensor t(DType::F32, shape);
    std::mt19937 rng(static_cast<uint32_t>(seed * 1234));
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    float* d = t.data_f32();
    for (size_t i = 0; i < t.numel(); ++i) d[i] = dist(rng);
    return t;
}

static float max_diff(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    float mx = 0.0f;
    const float* da = a.data_f32();
    const float* db = b.data_f32();
    for (size_t i = 0; i < a.numel(); ++i)
        mx = std::max(mx, std::abs(da[i] - db[i]));
    return mx;
}

// ---------------------------------------------------------------------------
// Test 1: QnnStubBackend implements all ops without throwing
// ---------------------------------------------------------------------------
static void test_qnn_stub_implements_all_ops() {
    std::cout << "Test: QnnStub implements all ops\n";
    QnnConfig cfg;
    cfg.verbose = false;
    QnnStubBackend qnn(cfg);

    Tensor a = rand_tensor({8, 16});
    Tensor b = rand_tensor({16, 8});
    Tensor q = rand_tensor({4, 1, 32});
    Tensor k = rand_tensor({2, 1, 32});
    Tensor v = rand_tensor({2, 1, 32});
    Tensor w = rand_tensor({128});   // weight for rms_norm
    Tensor x = rand_tensor({1, 128});
    Tensor gate = rand_tensor({1, 64});
    Tensor up   = rand_tensor({1, 64});

    bool ok = true;
    try {
        /* matmul    */ auto r1 = qnn.matmul(a, b);
        /* rope      */ auto r2 = qnn.rope(q, 0, 32);
        /* softmax   */ Tensor s = rand_tensor({1, 8}); qnn.softmax_inplace(s);
        /* rms_norm  */ auto r4 = qnn.rms_norm(x, w, 1e-6f);
        /* silu_mul  */ auto r5 = qnn.silu_mul(gate, up);
        /* add       */ auto r6 = qnn.add(a, a);
        /* attention */ auto r7 = qnn.attention(q, k, v, nullptr);
        /* fused_attn*/ auto r8 = qnn.fused_qk_v_attention(q, k, v);
    } catch (...) {
        ok = false;
    }

    QNN_TEST("all 8 ops succeed without exception", ok);
    QNN_TEST("profile has entries after ops", !qnn.profile().ops.empty());
    QNN_TEST("matmul op recorded", qnn.profile().ops.count(QNN_OP_MAT_MUL) > 0);
    QNN_TEST("softmax op recorded", qnn.profile().ops.count(QNN_OP_SOFTMAX) > 0);
}

// ---------------------------------------------------------------------------
// Test 2: PowerReport fields are all positive after a benchmark run
// ---------------------------------------------------------------------------
static void test_power_report_is_positive() {
    std::cout << "\nTest: PowerReport fields are positive\n";
    ModelConfig cfg = small_cfg();
    RuntimeConfig rcfg;
    rcfg.num_bench_tokens = 10;
    rcfg.num_warmup       = 2;
    rcfg.policy           = EvictionPolicy::QUOKA;

    QnnConfig qnn_cfg;
    qnn_cfg.verbose = false;
    QnnStubBackend qnn(qnn_cfg);
    qnn.reset_profile();

    Benchmark bench(cfg, rcfg);
    auto cache = create_kv_cache(cfg, EvictionPolicy::QUOKA);
    BenchResult br = bench.run_single(qnn, *cache);

    PowerReport pr = qnn.make_power_report("QUOKA", br.tokens_per_sec,
                                            cache->stats().peak_memory_bytes,
                                            cache->hit_rate());

    QNN_TEST("total_wall_us > 0",         pr.total_wall_us > 0.0);
    QNN_TEST("total_estimated_flops > 0", pr.total_estimated_flops > 0.0);
    QNN_TEST("total_bytes_moved > 0",     pr.total_bytes_moved > 0.0);
    QNN_TEST("estimated_mj > 0",          pr.estimated_mj > 0.0);
    QNN_TEST("tokens_per_mj > 0",         pr.tokens_per_mj > 0.0);
}

// ---------------------------------------------------------------------------
// Test 3: QnnStub matches CpuBackend output within 1e-4 tolerance
// ---------------------------------------------------------------------------
static void test_qnn_stub_matches_cpu_output() {
    std::cout << "\nTest: QnnStub matches CPUBackend output (matmul + softmax)\n";
    QnnStubBackend qnn;
    CPUBackend     cpu;

    // matmul
    Tensor a = rand_tensor({4, 8}, 1.0f);
    Tensor b = rand_tensor({8, 4}, 2.0f);
    Tensor qnn_mm  = qnn.matmul(a, b);
    Tensor cpu_mm  = cpu.matmul(a, b);
    float diff_mm = max_diff(qnn_mm, cpu_mm);
    QNN_TEST("matmul output matches CPU within 1e-4", diff_mm < 1e-4f);

    // softmax
    Tensor sx_qnn = rand_tensor({1, 16}, 3.0f);
    Tensor sx_cpu = sx_qnn.clone();
    qnn.softmax_inplace(sx_qnn);
    cpu.softmax_inplace(sx_cpu);
    float diff_sm = max_diff(sx_qnn, sx_cpu);
    QNN_TEST("softmax output matches CPU within 1e-4", diff_sm < 1e-4f);

    // attention
    Tensor q = rand_tensor({2, 1, 16}, 4.0f);
    Tensor k = rand_tensor({2, 1, 16}, 5.0f);
    Tensor v = rand_tensor({2, 1, 16}, 6.0f);
    Tensor attn_qnn = qnn.attention(q, k, v, nullptr);
    Tensor attn_cpu = cpu.attention(q, k, v, nullptr);
    float diff_attn = max_diff(attn_qnn, attn_cpu);
    QNN_TEST("attention output matches CPU within 1e-4", diff_attn < 1e-4f);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "\nQNN Stub Unit Tests\n";
    std::cout << "═══════════════════\n\n";

    test_qnn_stub_implements_all_ops();
    test_power_report_is_positive();
    test_qnn_stub_matches_cpu_output();

    std::cout << "\n─────────────────────────────────────\n";
    std::cout << "  Passed: " << tests_passed << " / " << (tests_passed + tests_failed) << "\n";
    std::cout << "  Failed: " << tests_failed << "\n";

    return tests_failed == 0 ? 0 : 1;
}
