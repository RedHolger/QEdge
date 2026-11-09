#include "kv_cache/kv_cache.h"
#include "kv_cache/policy.h"
#include <algorithm>
#include <chrono>
#include <cmath>

KVCache::KVCache(const ModelConfig& config, EvictionPolicy policy)
  : config_(config), policy_(policy) {
  uint32_t total_layers = config_.num_layers;
  caches_.resize(total_layers);
  for (auto& cl : caches_) {
    cl.max_per_layer = config_.kv_cache_capacity / total_layers;
    if (cl.max_per_layer < 4) cl.max_per_layer = 4;
  }
  max_entries_ = caches_.size() * caches_[0].max_per_layer;
}

void KVCache::store(uint32_t layer, uint32_t token_pos,
                    const Tensor& key, const Tensor& value) {
  auto& cl = caches_[layer];
  auto start = std::chrono::high_resolution_clock::now();

  if (cl.entries.size() >= cl.max_per_layer) {
    uint32_t victim = select_eviction_candidate(layer);
    stats_.evictions++;
    cl.entries.erase(cl.entries.begin() + victim);
    total_entries_--;
  }

  KVCacheEntry entry;
  entry.token_pos = token_pos;
  entry.key = key.clone();
  entry.value = value.clone();
  entry.last_access_score = 1.0f;
  cl.entries.push_back(std::move(entry));
  total_entries_++;

  size_t mem = total_entries_ * key.size_bytes() * 2;
  if (mem > stats_.peak_memory_bytes) stats_.peak_memory_bytes = mem;
}

bool KVCache::retrieve(uint32_t layer, uint32_t token_pos,
                       Tensor& key_out, Tensor& value_out) {
  auto start = std::chrono::high_resolution_clock::now();
  auto& cl = caches_[layer];
  for (auto& entry : cl.entries) {
    if (entry.token_pos == token_pos) {
      entry.last_access_score += 1.0f;
      key_out = entry.key.clone();
      value_out = entry.value.clone();
      stats_.hits++;
      return true;
    }
  }
  stats_.misses++;
  return false;
}

float KVCache::hit_rate() const {
  uint64_t total = stats_.hits + stats_.misses;
  return total == 0 ? 0.0f : (float)stats_.hits / total;
}

void KVCache::clear() {
  for (auto& cl : caches_) cl.entries.clear();
  total_entries_ = 0;
  stats_ = CacheStats{};
}

uint32_t KVCache::select_eviction_layer() {
  uint32_t worst = 0;
  float best = 1.0f;
  for (uint32_t i = 1; i < caches_.size(); ++i) {
    float hr = (float)caches_[i].entries.size() / caches_[i].max_per_layer;
    if (hr < best) { best = hr; worst = i; }
  }
  return worst;
}

void KVCache::evict_one(uint32_t layer) {
  if (caches_[layer].entries.empty()) return;
  uint32_t victim = select_eviction_candidate(layer);
  caches_[layer].entries.erase(caches_[layer].entries.begin() + victim);
  total_entries_--;
  stats_.evictions++;
}

// -- FIFO --

uint32_t FIFOKVCache::select_eviction_candidate(uint32_t layer) {
  return 0;
}

// -- LRU --

uint32_t LRUKVCache::select_eviction_candidate(uint32_t layer) {
  auto& entries = caches_[layer].entries;
  uint32_t victim = 0;
  float min_score = entries[0].last_access_score;
  for (uint32_t i = 1; i < entries.size(); ++i) {
    if (entries[i].last_access_score < min_score) {
      min_score = entries[i].last_access_score;
      victim = i;
    }
  }
  return victim;
}

// -- KeyDiff: evict entry whose key is most similar to the newest key --

uint32_t KeyDiffKVCache::select_eviction_candidate(uint32_t layer) {
  auto& entries = caches_[layer].entries;
  if (entries.size() < 2) return 0;
  const float* ref_key = entries.back().key.data_f32();
  size_t key_len = entries.back().key.numel();

  uint32_t victim = 0;
  float max_sim = cosine_similarity(ref_key, entries[0].key.data_f32(), key_len);

  for (uint32_t i = 1; i < entries.size() - 1; ++i) {
    float sim = cosine_similarity(ref_key, entries[i].key.data_f32(), key_len);
    if (sim > max_sim) {
      max_sim = sim;
      victim = i;
    }
  }
  return victim;
}

// -- QUOKA: evict entry with lowest query-key relevance (use avg of all entries as proxy) --

uint32_t QUOKAKVCache::select_eviction_candidate(uint32_t layer) {
  auto& entries = caches_[layer].entries;
  if (entries.size() < 2) return 0;
  size_t key_len = entries[0].key.numel();

  std::vector<float> avg_key(key_len, 0.0f);
  for (const auto& e : entries) {
    const float* k = e.key.data_f32();
    for (size_t i = 0; i < key_len; ++i) avg_key[i] += k[i];
  }
  for (size_t i = 0; i < key_len; ++i) avg_key[i] /= entries.size();

  uint32_t victim = 0;
  float min_rel = cosine_similarity(avg_key.data(), entries[0].key.data_f32(), key_len);
  for (uint32_t i = 1; i < entries.size(); ++i) {
    float rel = cosine_similarity(avg_key.data(), entries[i].key.data_f32(), key_len);
    if (rel < min_rel) {
      min_rel = rel;
      victim = i;
    }
  }
  return victim;
}

// -- Factory --

std::unique_ptr<KVCache> create_kv_cache(const ModelConfig& config, EvictionPolicy policy) {
  switch (policy) {
    case EvictionPolicy::FIFO:   return std::make_unique<FIFOKVCache>(config, policy);
    case EvictionPolicy::LRU:    return std::make_unique<LRUKVCache>(config, policy);
    case EvictionPolicy::KeyDiff: return std::make_unique<KeyDiffKVCache>(config, policy);
    case EvictionPolicy::QUOKA:  return std::make_unique<QUOKAKVCache>(config, policy);
  }
  return std::make_unique<FIFOKVCache>(config, policy);
}
