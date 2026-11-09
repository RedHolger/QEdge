#pragma once
#include <cstdint>
#include <string>

struct ModelConfig {
  uint32_t vocab_size = 32000;
  uint32_t hidden_size = 2048;
  uint32_t num_layers = 22;
  uint32_t num_heads = 32;
  uint32_t num_kv_heads = 8;
  uint32_t head_dim = 64;
  uint32_t intermediate_size = 5632;
  float norm_eps = 1e-6;
  uint32_t max_seq_len = 4096;
  uint32_t kv_cache_capacity = 2048;
};

enum class EvictionPolicy {
  FIFO,
  LRU,
  KeyDiff,
  QUOKA
};

struct RuntimeConfig {
  EvictionPolicy policy = EvictionPolicy::KeyDiff;
  bool use_metal = true;
  uint32_t num_warmup = 5;
  uint32_t num_bench_tokens = 128;
  float kv_cache_eviction_ratio = 0.3f;
  std::string model_path;
};
