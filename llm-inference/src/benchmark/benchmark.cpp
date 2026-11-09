#include "benchmark/benchmark.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>

Benchmark::Benchmark(const ModelConfig& model_cfg, const RuntimeConfig& runtime_cfg)
  : model_cfg_(model_cfg), runtime_cfg_(runtime_cfg) {}

Tensor Benchmark::generate_random_tensor(DType dtype, const std::vector<size_t>& shape) {
  Tensor t(dtype, shape);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  float* data = t.data_f32();
  for (size_t i = 0; i < t.numel(); ++i) data[i] = dist(rng);
  return t;
}

Tensor Benchmark::generate_causal_mask(size_t seq_len) {
  Tensor mask(DType::F32, {1, seq_len, seq_len});
  mask.fill(0.0f);
  float* d = mask.data_f32();
  for (size_t i = 0; i < seq_len; ++i) {
    for (size_t j = i + 1; j < seq_len; ++j) {
      d[i * seq_len + j] = -1e30f;
    }
  }
  return mask;
}

double Benchmark::measure_us(std::function<void()> fn, int warmup, int repeats) {
  for (int i = 0; i < warmup; ++i) fn();
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count() / repeats;
}

BenchResult Benchmark::run_single(Backend& backend, KVCache& cache) {
  cache.clear();
  BenchResult r;
  r.backend_name = backend.name();
  switch (runtime_cfg_.policy) {
    case EvictionPolicy::FIFO:   r.policy_name = "FIFO"; break;
    case EvictionPolicy::LRU:    r.policy_name = "LRU"; break;
    case EvictionPolicy::KeyDiff: r.policy_name = "KeyDiff"; break;
    case EvictionPolicy::QUOKA:  r.policy_name = "QUOKA"; break;
  }

  uint32_t n_heads = model_cfg_.num_heads;
  uint32_t n_kv_heads = model_cfg_.num_kv_heads;
  uint32_t head_dim = model_cfg_.head_dim;
  uint32_t n_layers = model_cfg_.num_layers;
  uint32_t n_tokens = runtime_cfg_.num_bench_tokens;

  // Prefill: generate first token's KV cache entries for all layers
  Tensor q = generate_random_tensor(DType::F32, {n_heads, 1, head_dim});
  Tensor k = generate_random_tensor(DType::F32, {n_kv_heads, 1, head_dim});
  Tensor v = generate_random_tensor(DType::F32, {n_kv_heads, 1, head_dim});

  auto decode_start = std::chrono::high_resolution_clock::now();
  uint64_t total_attention_us = 0;

  for (uint32_t t = 0; t < n_tokens; ++t) {
    for (uint32_t layer = 0; layer < n_layers; ++layer) {
      // Randomize keys for realistic eviction pressure
      Tensor k_tok = generate_random_tensor(DType::F32, {n_kv_heads, 1, head_dim});
      Tensor v_tok = generate_random_tensor(DType::F32, {n_kv_heads, 1, head_dim});

      cache.store(layer, t, k_tok, v_tok);

      // Simulate attention retrieval. Use the full KV-head shape so a hit is
      // a valid GQA/MQA attention input; on a miss the placeholder is not fed
      // to attention as if it were the retrieved cache entry.
      Tensor k_ret(DType::F32, {n_kv_heads, 1, head_dim});
      Tensor v_ret(DType::F32, {n_kv_heads, 1, head_dim});

      // Retrieve a random past token to generate cache activity
      if (t > 0) {
        uint32_t past = std::min((uint32_t)(t * 0.7f), t - 1);
        auto attn_start = std::chrono::high_resolution_clock::now();
        const bool hit = cache.retrieve(layer, past, k_ret, v_ret);
        auto attn_end = std::chrono::high_resolution_clock::now();
        total_attention_us += std::chrono::duration<double, std::micro>(attn_end - attn_start).count();

        // Run attention kernel on CPU backend for measurement only after a
        // successful retrieval.
        if (hit && layer == 0) {
          Tensor scores = backend.attention(q, k_ret, v_ret, nullptr);
        }
      }
    }
  }

  auto decode_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

  r.avg_token_ms = total_ms / n_tokens;
  r.tokens_per_sec = 1000.0 / r.avg_token_ms;
  r.cache_hit_rate = cache.hit_rate();
  r.evictions = cache.stats().evictions;
  r.peak_memory_mb = cache.stats().peak_memory_bytes / (1024.0 * 1024.0);
  r.avg_attention_us = total_attention_us / (n_tokens * n_layers);

  return r;
}

std::vector<BenchResult> Benchmark::run_all_policies(Backend& backend) {
  std::vector<BenchResult> results;
  std::vector<EvictionPolicy> policies = {
    EvictionPolicy::FIFO, EvictionPolicy::LRU,
    EvictionPolicy::KeyDiff, EvictionPolicy::QUOKA
  };

  for (auto policy : policies) {
    runtime_cfg_.policy = policy;
    auto cache = create_kv_cache(model_cfg_, policy);
    auto r = run_single(backend, *cache);
    results.push_back(r);
  }

  return results;
}

std::vector<BenchResult> Benchmark::run_all_backends() {
  std::vector<BenchResult> results;

  auto cpu_backend = create_backend(BackendType::CPU);
  auto cpu_results = run_all_policies(*cpu_backend);
  results.insert(results.end(), cpu_results.begin(), cpu_results.end());

  auto metal_backend = create_backend(BackendType::Metal);
  if (metal_backend->type() == BackendType::Metal) {
    auto metal_results = run_all_policies(*metal_backend);
    results.insert(results.end(), metal_results.begin(), metal_results.end());
  }

  return results;
}

void Benchmark::print_report(const std::vector<BenchResult>& results) {
  std::cout << "\n";
  std::cout << "+--------------------------------------------------------------------------------+\n";
  std::cout << "|      KERNEL LAB: synthetic KV-cache microbenchmark (not model inference)      |\n";
  std::cout << "+--------------------------------------------------------------------------------+\n";

  auto& cfg = runtime_cfg_;
  std::cout << "  Model: " << model_cfg_.num_layers << "L, "
            << model_cfg_.hidden_size << " hidden, "
            << model_cfg_.num_heads << " heads, "
            << model_cfg_.head_dim << " head_dim\n";
  std::cout << "  NOTE: synthetic random tensors and cache bookkeeping; not model throughput.\n";
  std::cout << "  Cache: " << model_cfg_.kv_cache_capacity << " slots, "
            << "Tokens: " << cfg.num_bench_tokens << "\n\n";

  std::cout << std::left << std::setw(40) << "Backend"
            << std::setw(14) << "Policy"
            << std::setw(16) << "Avg Token (ms)"
            << std::setw(16) << "Tok/s"
            << std::setw(14) << "Hit Rate"
            << std::setw(12) << "Evicts"
            << std::setw(14) << "Peak Mem (MB)" << "\n";
  std::cout << std::string(126, '-') << "\n";

  for (const auto& r : results) {
    std::cout << std::left << std::setw(40) << r.backend_name
              << std::setw(14) << r.policy_name
              << std::setw(16) << std::fixed << std::setprecision(3) << r.avg_token_ms
              << std::setw(16) << std::fixed << std::setprecision(1) << r.tokens_per_sec
              << std::setw(14) << std::fixed << std::setprecision(3) << r.cache_hit_rate
              << std::setw(12) << r.evictions
              << std::setw(14) << std::fixed << std::setprecision(2) << r.peak_memory_mb << "\n";
  }
  std::cout << std::string(106, '-') << "\n\n";
}

void Benchmark::print_comparison(const std::vector<BenchResult>& results) {
  std::cout << "Synthetic comparison (not a model benchmark):\n";
  auto best_hit = std::max_element(results.begin(), results.end(),
    [](const BenchResult& a, const BenchResult& b) { return a.cache_hit_rate < b.cache_hit_rate; });
  auto best_speed = std::max_element(results.begin(), results.end(),
    [](const BenchResult& a, const BenchResult& b) { return a.tokens_per_sec < b.tokens_per_sec; });

  if (best_hit != results.end())
    std::cout << "  Best hit rate:  " << best_hit->policy_name
              << " (" << best_hit->backend_name << ") — "
              << std::fixed << std::setprecision(3) << best_hit->cache_hit_rate * 100 << "%\n";
  if (best_speed != results.end())
    std::cout << "  Best speed:     " << best_speed->policy_name
              << " (" << best_speed->backend_name << ") — "
              << std::fixed << std::setprecision(1) << best_speed->tokens_per_sec << " tok/s\n";
  std::cout << "\n";
}
