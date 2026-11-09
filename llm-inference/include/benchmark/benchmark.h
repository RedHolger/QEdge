#pragma once
#include "core/config.h"
#include "core/tensor.h"
#include "backend/backend.h"
#include "kv_cache/kv_cache.h"
#include <string>
#include <vector>
#include <chrono>

struct BenchResult {
  std::string backend_name;
  std::string policy_name;
  double avg_token_ms;
  double tokens_per_sec;
  double cache_hit_rate;
  uint64_t evictions;
  size_t peak_memory_mb;
  double avg_attention_us;
};

class Benchmark {
public:
  Benchmark(const ModelConfig& model_cfg, const RuntimeConfig& runtime_cfg);

  BenchResult run_single(Backend& backend, KVCache& cache);

  std::vector<BenchResult> run_all_policies(Backend& backend);
  std::vector<BenchResult> run_all_backends();

  void print_report(const std::vector<BenchResult>& results);
  void print_comparison(const std::vector<BenchResult>& results);

private:
  ModelConfig model_cfg_;
  RuntimeConfig runtime_cfg_;

  Tensor generate_random_tensor(DType dtype, const std::vector<size_t>& shape);
  Tensor generate_causal_mask(size_t seq_len);
  double measure_us(std::function<void()> fn, int warmup, int repeats);
};
