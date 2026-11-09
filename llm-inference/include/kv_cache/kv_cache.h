#pragma once
#include "core/tensor.h"
#include "core/config.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>

struct KVCacheEntry {
  uint32_t token_pos;
  Tensor key;
  Tensor value;
  float last_access_score;  // for LRU
};

class KVCache {
public:
  KVCache(const ModelConfig& config, EvictionPolicy policy);
  virtual ~KVCache() = default;

  void store(uint32_t layer, uint32_t token_pos,
             const Tensor& key, const Tensor& value);
  bool retrieve(uint32_t layer, uint32_t token_pos,
                Tensor& key_out, Tensor& value_out);

  void clear();
  size_t size() const { return total_entries_; }
  size_t capacity() const { return max_entries_; }
  float hit_rate() const;

  struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    size_t peak_memory_bytes = 0;
    double avg_retrieve_us = 0.0;
  };
  const CacheStats& stats() const { return stats_; }

protected:
  virtual uint32_t select_eviction_layer();
  virtual uint32_t select_eviction_candidate(uint32_t layer) = 0;

  struct PerLayerCache {
    std::deque<KVCacheEntry> entries;
    uint32_t max_per_layer;
  };

  std::vector<PerLayerCache> caches_;
  ModelConfig config_;
  EvictionPolicy policy_;
  size_t total_entries_ = 0;
  size_t max_entries_;
  mutable CacheStats stats_;

  void evict_one(uint32_t layer);
};

class FIFOKVCache : public KVCache {
public:
  using KVCache::KVCache;
protected:
  uint32_t select_eviction_candidate(uint32_t layer) override;
};

class LRUKVCache : public KVCache {
public:
  using KVCache::KVCache;
protected:
  uint32_t select_eviction_candidate(uint32_t layer) override;
};

class KeyDiffKVCache : public KVCache {
public:
  using KVCache::KVCache;
protected:
  uint32_t select_eviction_candidate(uint32_t layer) override;
};

class QUOKAKVCache : public KVCache {
public:
  using KVCache::KVCache;
protected:
  uint32_t select_eviction_candidate(uint32_t layer) override;
};

std::unique_ptr<KVCache> create_kv_cache(
    const ModelConfig& config, EvictionPolicy policy);
