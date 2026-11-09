#include "kv_cache/kv_cache.h"
#include "kv_cache/policy.h"
#include "core/config.h"
#include <cassert>
#include <iostream>
#include <cmath>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
  if (!(expr)) { \
    std::cerr << "  FAIL: " << name << " (" << #expr << ")\n"; \
    tests_failed++; \
  } else { \
    std::cout << "  PASS: " << name << "\n"; \
    tests_passed++; \
  } \
} while(0)

static void test_basic_store_retrieve() {
  std::cout << "Test: Basic Store & Retrieve\n";
  ModelConfig cfg;
  cfg.num_layers = 2;
  cfg.num_kv_heads = 4;
  cfg.head_dim = 64;
  cfg.kv_cache_capacity = 32;

  FIFOKVCache cache(cfg, EvictionPolicy::FIFO);

  Tensor key(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  Tensor value(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  key.fill(1.0f);
  value.fill(2.0f);

  cache.store(0, 0, key, value);
  TEST("size after 1 store", cache.size() == 1);

  Tensor k_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  Tensor v_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  bool found = cache.retrieve(0, 0, k_out, v_out);
  TEST("retrieve token 0 layer 0", found);
  TEST("key value correct", std::abs(k_out.data_f32()[0] - 1.0f) < 1e-5);
  TEST("value value correct", std::abs(v_out.data_f32()[0] - 2.0f) < 1e-5);

  bool not_found = cache.retrieve(0, 999, k_out, v_out);
  TEST("retrieve nonexistent", !not_found);

  bool wrong_layer = cache.retrieve(1, 0, k_out, v_out);
  TEST("wrong layer miss", !wrong_layer);

  cache.clear();
  TEST("clear resets size", cache.size() == 0);
}

static void test_fifo_eviction() {
  std::cout << "Test: FIFO Eviction\n";
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.num_kv_heads = 2;
  cfg.head_dim = 8;
  cfg.kv_cache_capacity = 4;  // 4 entries total, all in 1 layer

  FIFOKVCache cache(cfg, EvictionPolicy::FIFO);

  for (uint32_t t = 0; t < 6; ++t) {
    Tensor key(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    Tensor value(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    key.fill((float)t);
    value.fill((float)t * 10);
    cache.store(0, t, key, value);
  }

  TEST("size capped at capacity", cache.size() <= cfg.kv_cache_capacity);
  TEST("evictions occurred", cache.stats().evictions == 2);

  Tensor k_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  Tensor v_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});

  bool old_evicted = cache.retrieve(0, 0, k_out, v_out);
  TEST("token 0 evicted (FIFO)", !old_evicted);

  bool recent_present = cache.retrieve(0, 5, k_out, v_out);
  TEST("token 5 present (FIFO)", recent_present);
  TEST("token 5 value correct", k_out.data_f32()[0] == 5.0f);
}

static void test_lru_eviction() {
  std::cout << "Test: LRU Eviction\n";
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.num_kv_heads = 2;
  cfg.head_dim = 8;
  cfg.kv_cache_capacity = 4;

  LRUKVCache cache(cfg, EvictionPolicy::LRU);

  for (uint32_t t = 0; t < 4; ++t) {
    Tensor key(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    Tensor value(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    key.fill((float)t);
    value.fill((float)t * 10);
    cache.store(0, t, key, value);
  }

  // Access token 1 twice to boost its score
  Tensor k_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  Tensor v_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  cache.retrieve(0, 1, k_out, v_out);
  cache.retrieve(0, 1, k_out, v_out);

  // Store 2 more tokens, forcing evictions (total capacity 4)
  for (uint32_t t = 4; t < 6; ++t) {
    Tensor key(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    Tensor value(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    key.fill((float)t);
    value.fill((float)t * 10);
    cache.store(0, t, key, value);
  }

  TEST("evictions occurred", cache.stats().evictions == 2);

  bool least_used_evicted = cache.retrieve(0, 0, k_out, v_out);
  TEST("token 0 (least used after access) evicted", !least_used_evicted);

  bool most_used_present = cache.retrieve(0, 1, k_out, v_out);
  TEST("token 1 (most accessed) present", most_used_present);
}

static void test_keydiff_eviction() {
  std::cout << "Test: KeyDiff Eviction\n";
  ModelConfig cfg;
  cfg.num_layers = 1;
  cfg.num_kv_heads = 1;
  cfg.head_dim = 16;
  cfg.kv_cache_capacity = 4;

  KeyDiffKVCache cache(cfg, EvictionPolicy::KeyDiff);

  // Store tokens with engineered similarities
  for (uint32_t t = 0; t < 6; ++t) {
    Tensor key(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    Tensor value(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
    float* kd = key.data_f32();
    // Token 0,1,2 are similar (sin of nearby values)
    // Token 3,4,5 are also similar but different group
    for (size_t i = 0; i < cfg.head_dim; ++i) {
      float phase = (t < 3) ? 0.1f : 1.0f;
      kd[i] = std::sin(phase * (float)(t * cfg.head_dim + i));
    }
    value.fill((float)t);
    cache.store(0, t, key, value);
  }

  // KeyDiff evicts tokens most similar to newest (token 5)
  TEST("evictions occurred", cache.stats().evictions == 2);

  // Verify size is within capacity
  Tensor k_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  Tensor v_out(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
  TEST("cache size at capacity", cache.size() <= cfg.kv_cache_capacity);

  // Verify newest token is always present (never evict the ref token)
  bool t5_present = cache.retrieve(0, 5, k_out, v_out);
  TEST("newest token (5) present", t5_present);
}

#ifdef ENABLE_QNN_STUB
#include "backend/qnn_stub_backend.h"
#include "backend/cpu_backend.h"
#include "benchmark/benchmark.h"
#include <random>

static Tensor rand_tensor_test(const std::vector<size_t>& shape, float seed = 1.0f) {
  Tensor t(DType::F32, shape);
  std::mt19937 rng(static_cast<uint32_t>(seed * 1234));
  std::uniform_real_distribution<float> dist(0.1f, 1.0f);
  float* d = t.data_f32();
  for (size_t i = 0; i < t.numel(); ++i) d[i] = dist(rng);
  return t;
}

static float max_diff_test(const Tensor& a, const Tensor& b) {
  assert(a.numel() == b.numel());
  float mx = 0.0f;
  const float* da = a.data_f32();
  const float* db = b.data_f32();
  for (size_t i = 0; i < a.numel(); ++i)
    mx = std::max(mx, std::abs(da[i] - db[i]));
  return mx;
}

static void test_qnn_stub_suite() {
  std::cout << "Test: QnnStub Integration (CPU delegate)\n";
  
  // 1. test_qnn_stub_implements_all_ops
  bool ops_ok = false;
  try {
    QnnConfig cfg;
    cfg.verbose = false;
    QnnStubBackend qnn(cfg);
    Tensor a = rand_tensor_test({8, 16});
    Tensor b = rand_tensor_test({16, 8});
    Tensor q = rand_tensor_test({4, 1, 32});
    Tensor k = rand_tensor_test({2, 1, 32});
    Tensor v = rand_tensor_test({2, 1, 32});
    Tensor w = rand_tensor_test({128});
    Tensor x = rand_tensor_test({1, 128});
    Tensor gate = rand_tensor_test({1, 64});
    Tensor up = rand_tensor_test({1, 64});

    auto r1 = qnn.matmul(a, b);
    auto r2 = qnn.rope(q, 0, 32);
    Tensor s = rand_tensor_test({1, 8}); qnn.softmax_inplace(s);
    auto r4 = qnn.rms_norm(x, w, 1e-6f);
    auto r5 = qnn.silu_mul(gate, up);
    auto r6 = qnn.add(a, a);
    auto r7 = qnn.attention(q, k, v, nullptr);
    auto r8 = qnn.fused_qk_v_attention(q, k, v);
    ops_ok = (!qnn.profile().ops.empty() &&
              qnn.profile().ops.count(QNN_OP_MAT_MUL) > 0 &&
              qnn.profile().ops.count(QNN_OP_SOFTMAX) > 0);
  } catch (...) {
    ops_ok = false;
  }
  TEST("QnnStub implements all ops and profiles calls", ops_ok);

  // 2. test_power_report_is_positive
  bool power_ok = false;
  try {
    ModelConfig cfg;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 32;
    cfg.kv_cache_capacity = 16;
    cfg.hidden_size = 128;
    cfg.intermediate_size = 256;

    RuntimeConfig rcfg;
    rcfg.num_bench_tokens = 10;
    rcfg.num_warmup = 2;
    rcfg.policy = EvictionPolicy::QUOKA;

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
    power_ok = (pr.total_wall_us > 0.0 &&
                pr.total_estimated_flops > 0.0 &&
                pr.total_bytes_moved > 0.0 &&
                pr.estimated_mj > 0.0 &&
                pr.tokens_per_mj > 0.0);
  } catch (...) {
    power_ok = false;
  }
  TEST("QnnStub PowerReport fields are all positive", power_ok);

  // 3. test_qnn_stub_matches_cpu_output
  bool match_ok = false;
  try {
    QnnStubBackend qnn;
    CPUBackend cpu;
    Tensor a = rand_tensor_test({4, 8}, 1.0f);
    Tensor b = rand_tensor_test({8, 4}, 2.0f);
    Tensor qnn_mm = qnn.matmul(a, b);
    Tensor cpu_mm = cpu.matmul(a, b);
    float diff_mm = max_diff_test(qnn_mm, cpu_mm);

    Tensor sx_qnn = rand_tensor_test({1, 16}, 3.0f);
    Tensor sx_cpu = sx_qnn.clone();
    qnn.softmax_inplace(sx_qnn);
    cpu.softmax_inplace(sx_cpu);
    float diff_sm = max_diff_test(sx_qnn, sx_cpu);

    Tensor q = rand_tensor_test({2, 1, 16}, 4.0f);
    Tensor k = rand_tensor_test({2, 1, 16}, 5.0f);
    Tensor v = rand_tensor_test({2, 1, 16}, 6.0f);
    Tensor attn_qnn = qnn.attention(q, k, v, nullptr);
    Tensor attn_cpu = cpu.attention(q, k, v, nullptr);
    float diff_attn = max_diff_test(attn_qnn, attn_cpu);

    match_ok = (diff_mm < 1e-4f && diff_sm < 1e-4f && diff_attn < 1e-4f);
  } catch (...) {
    match_ok = false;
  }
  TEST("QnnStub output matches CPUBackend within 1e-4", match_ok);
}
#endif

int main() {
  std::cout << "\nKV Cache Unit Tests\n";
  std::cout << "═══════════════════\n\n";

  test_basic_store_retrieve();
  std::cout << "\n";
  test_fifo_eviction();
  std::cout << "\n";
  test_lru_eviction();
  std::cout << "\n";
  test_keydiff_eviction();
#ifdef ENABLE_QNN_STUB
  std::cout << "\n";
  test_qnn_stub_suite();
#endif

  std::cout << "\n─────────────────────────────────\n";
  std::cout << "  Passed: " << tests_passed << " / " << (tests_passed + tests_failed) << "\n";
  std::cout << "  Failed: " << tests_failed << "\n";

  return tests_failed > 0 ? 1 : 0;
}
