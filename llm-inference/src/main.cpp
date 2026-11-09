#include "backend/backend.h"
#include "backend/cpu_backend.h"
#include "backend/metal_factory.h"
#include "kv_cache/kv_cache.h"
#include "kv_cache/policy.h"
#include "benchmark/benchmark.h"
#ifdef ENABLE_QNN_STUB
#include "benchmark/power_report.h"
#endif
#include <iostream>
#include <iomanip>
#include <random>
#include <memory>
#include <vector>

static void print_banner() {
  std::cout << "\n";
  std::cout << "  QEdge Runtime — C++ kernel lab (Stage 01)\n";
  std::cout << "  Synthetic cache and attention microbenchmark; no model is loaded.\n";
  std::cout << "  QNN fallback is a CPU stub; Snapdragon/NPU status: unverified.\n";
  std::cout << std::endl;
}

static void demo_kv_cache() {
  std::cout << "── KV Cache Demo ──────────────────────────────────────────────\n";

  ModelConfig cfg;
  cfg.num_layers = 4;
  cfg.num_heads = 8;
  cfg.num_kv_heads = 4;
  cfg.head_dim = 64;
  cfg.kv_cache_capacity = 32;

  std::vector<std::pair<EvictionPolicy, std::string>> policies = {
    {EvictionPolicy::FIFO, "FIFO"},
    {EvictionPolicy::LRU, "LRU"},
    {EvictionPolicy::KeyDiff, "KeyDiff"},
    {EvictionPolicy::QUOKA, "QUOKA"},
  };

  for (auto& [policy, name] : policies) {
    auto cache = create_kv_cache(cfg, policy);
    std::cout << "  " << name << ": ";

    // Store 20 tokens across layers
    for (uint32_t t = 0; t < 20; ++t) {
      for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        Tensor key(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
        Tensor value(DType::F32, {cfg.num_kv_heads, 1, cfg.head_dim});
        // Fill with deterministic data for KeyDiff/QUOKA comparisons
        float* kd = key.data_f32();
        float* vd = value.data_f32();
        for (size_t i = 0; i < key.numel(); ++i) {
          kd[i] = std::sin((float)(t * cfg.head_dim + i) * 0.1f);
          vd[i] = std::cos((float)(t * cfg.head_dim + i) * 0.1f);
        }
        cache->store(l, t, key, value);
      }
    }

    // Retrieve ~15 random past tokens
    uint32_t hits = 0, attempts = 0;
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
      for (uint32_t t = 0; t < 15; ++t) {
        uint32_t past = (t * 7 + 3) % 20;
        Tensor k_out(DType::F32, {1, 1, cfg.head_dim});
        Tensor v_out(DType::F32, {1, 1, cfg.head_dim});
        if (cache->retrieve(l, past, k_out, v_out)) hits++;
        attempts++;
      }
    }

    std::cout << "size=" << cache->size()
              << " hits=" << hits << "/" << attempts
              << " hit_rate=" << (100.0f * hits / attempts) << "%"
              << " evictions=" << cache->stats().evictions
              << std::endl;
  }
  std::cout << std::endl;
}

static void demo_attention() {
  std::cout << "── Attention Kernel Demo ──────────────────────────────────────\n";

  auto cpu = create_backend(BackendType::CPU);
  std::cout << "  Backend: " << cpu->name() << "\n";

  size_t n_heads = 8, seq_len = 4, head_dim = 64;

  Tensor q(DType::F32, {n_heads, seq_len, head_dim});
  Tensor k(DType::F32, {n_heads, seq_len, head_dim});
  Tensor v(DType::F32, {n_heads, seq_len, head_dim});

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (size_t i = 0; i < q.numel(); ++i) q.data_f32()[i] = dist(rng);
  for (size_t i = 0; i < k.numel(); ++i) k.data_f32()[i] = dist(rng);
  for (size_t i = 0; i < v.numel(); ++i) v.data_f32()[i] = dist(rng);

  // Create causal mask
  Tensor mask(DType::F32, {1, seq_len, seq_len});
  mask.fill(0.0f);
  float* md = mask.data_f32();
  for (size_t i = 0; i < seq_len; ++i)
    for (size_t j = i + 1; j < seq_len; ++j)
      md[i * seq_len + j] = -1e30f;
  auto result = cpu->attention(q, k, v, &mask);

  std::cout << "  Attention output shape: [";
  for (size_t i = 0; i < result.shape().size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << result.shape()[i];
  }
  std::cout << "]\n";

  // Print first head's output for verification
  std::cout << "  First head, first token (first 8 dims): ";
  for (size_t d = 0; d < 8; ++d) {
    std::cout << std::fixed << std::setprecision(4) << result.at({0, 0, d}) << " ";
  }
  std::cout << "\n" << std::endl;
}

int main(int argc, char** argv) {
  print_banner();

#ifdef ENABLE_QNN_STUB
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--backend" && i + 1 < argc && std::string(argv[i + 1]) == "qnn") {
      ModelConfig model_cfg;
      model_cfg.num_layers = 8;
      model_cfg.num_heads = 32;
      model_cfg.num_kv_heads = 8;
      model_cfg.head_dim = 64;
      model_cfg.hidden_size = 2048;
      model_cfg.kv_cache_capacity = 512;

      RuntimeConfig runtime_cfg;
      runtime_cfg.num_bench_tokens = 64;
      runtime_cfg.num_warmup = 3;

      run_qnn_power_benchmark(model_cfg, runtime_cfg);
      return 0;
    }
  }
#endif

  demo_kv_cache();
  demo_attention();

  // Full benchmark
  ModelConfig model_cfg;
  model_cfg.num_layers = 8;
  model_cfg.num_heads = 32;
  model_cfg.num_kv_heads = 8;
  model_cfg.head_dim = 64;
  model_cfg.hidden_size = 2048;
  model_cfg.kv_cache_capacity = 512;

  RuntimeConfig runtime_cfg;
  runtime_cfg.num_bench_tokens = 64;
  runtime_cfg.num_warmup = 3;

  Benchmark bench(model_cfg, runtime_cfg);

  std::vector<BenchResult> results;

  // CPU backend
  auto cpu_backend = create_backend(BackendType::CPU);
  auto cpu_results = bench.run_all_policies(*cpu_backend);
  results.insert(results.end(), cpu_results.begin(), cpu_results.end());

  // Metal backend (if available)
  auto metal_backend = create_metal_backend();
  if (metal_backend) {
    auto metal_results = bench.run_all_policies(*metal_backend);
    results.insert(results.end(), metal_results.begin(), metal_results.end());
  }

  bench.print_report(results);
  bench.print_comparison(results);

  return 0;
}
