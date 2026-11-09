#pragma once
#include "core/tensor.h"

inline float cosine_similarity(const float* a, const float* b, size_t n) {
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  float denom = std::sqrt(na) * std::sqrt(nb);
  return (denom < 1e-8f) ? 0.0f : dot / denom;
}

inline float l2_distance(const float* a, const float* b, size_t n) {
  float dist = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = a[i] - b[i];
    dist += d * d;
  }
  return std::sqrt(dist);
}
