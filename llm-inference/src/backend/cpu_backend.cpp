#include "backend/cpu_backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Additive mask biases at or below this sentinel are treated as forbidden.
constexpr float kMaskedScore = -1.0e30f;

[[noreturn]] void invalid_argument(const std::string& message) {
  throw std::invalid_argument("attention: " + message);
}

void require_f32_rank3(const Tensor& t, const char* name) {
  if (t.dtype() != DType::F32)
    invalid_argument(std::string(name) + " must be F32");
  if (t.rank() != 3)
    invalid_argument(std::string(name) + " must have rank 3 [heads, sequence, dimension]");
  for (size_t d = 0; d < 3; ++d) {
    if (t.shape()[d] == 0)
      invalid_argument(std::string(name) + " dimensions must be nonzero");
  }
}

void validate_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                        const Tensor* mask) {
  require_f32_rank3(q, "q");
  require_f32_rank3(k, "k");
  require_f32_rank3(v, "v");

  const size_t q_heads = q.shape()[0];
  const size_t kv_heads = k.shape()[0];
  if (q_heads % kv_heads != 0)
    invalid_argument("q heads must be divisible by k/v heads");
  if (v.shape()[0] != kv_heads)
    invalid_argument("k and v must have the same head count");
  if (k.shape()[1] != v.shape()[1] || k.shape()[2] != v.shape()[2])
    invalid_argument("k and v must have the same sequence and dimension");
  if (q.shape()[2] != k.shape()[2])
    invalid_argument("q, k and v must have the same head dimension");

  if (mask) {
    if (mask->dtype() != DType::F32)
      invalid_argument("mask must be F32");
    if (mask->rank() != 2 && mask->rank() != 3)
      invalid_argument("mask must be rank 2 [q_len, kv_len] or rank 3 [1|q_heads, q_len, kv_len]");
    if (mask->rank() == 2) {
      if (mask->shape()[0] != q.shape()[1] || mask->shape()[1] != k.shape()[1])
        invalid_argument("rank-2 mask shape must be [q_len, kv_len]");
    } else {
      if ((mask->shape()[0] != 1 && mask->shape()[0] != q_heads) ||
          mask->shape()[1] != q.shape()[1] || mask->shape()[2] != k.shape()[1])
        invalid_argument("rank-3 mask shape must be [1|q_heads, q_len, kv_len]");
    }
  }
}

float mask_at(const Tensor& mask, size_t head, size_t q_index, size_t k_index) {
  const float* data = mask.data_f32();
  if (mask.rank() == 2)
    return data[q_index * mask.shape()[1] + k_index];
  const size_t mask_head = mask.shape()[0] == 1 ? 0 : head;
  return data[(mask_head * mask.shape()[1] + q_index) * mask.shape()[2] + k_index];
}

}  // namespace

static void sgemm_f32(bool ta, bool tb, int m, int n, int k,
                      float alpha, const float* a, int lda,
                      const float* b, int ldb,
                      float beta, float* c, int ldc) {
  cblas_sgemm(CblasRowMajor,
              ta ? CblasTrans : CblasNoTrans,
              tb ? CblasTrans : CblasNoTrans,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

Tensor CPUBackend::matmul(const Tensor& a, const Tensor& b) {
  assert(a.dtype() == DType::F32 && b.dtype() == DType::F32);
  assert(a.shape().size() == 2 && b.shape().size() == 2);
  assert(a.shape()[1] == b.shape()[0]);

  size_t M = a.shape()[0];
  size_t K = a.shape()[1];
  size_t N = b.shape()[1];

  Tensor c(DType::F32, {M, N});

  sgemm_f32(false, false, (int)M, (int)N, (int)K, 1.0f,
            a.data_f32(), (int)K,
            b.data_f32(), (int)N,
            0.0f, c.data_f32(), (int)N);

  return c;
}

Tensor CPUBackend::rope(const Tensor& x, const std::vector<uint32_t>& positions,
                        const RopeConfig& config) {
  if (x.dtype() != DType::F32)
    throw std::invalid_argument("rope: x must be F32");
  if (x.rank() != 2 && x.rank() != 3)
    throw std::invalid_argument("rope: x must be rank 2 [sequence, dimension] or rank 3 [heads, sequence, dimension]");
  if (config.head_dim == 0 || config.head_dim % 2 != 0)
    throw std::invalid_argument("rope: head_dim must be nonzero and even");
  if (x.shape().back() != config.head_dim)
    throw std::invalid_argument("rope: x dimension must equal config.head_dim");
  if (!(config.base > 0.0f))
    throw std::invalid_argument("rope: base must be positive");

  const size_t seq_len = x.shape()[x.rank() - 2];
  if (seq_len == 0 || x.shape()[0] == 0)
    throw std::invalid_argument("rope: sequence and head dimensions must be nonzero");
  if (positions.size() != seq_len)
    throw std::invalid_argument("rope: one position is required for each sequence element");

  Tensor out = x.clone();
  const size_t head_dim = config.head_dim;
  const size_t half = head_dim / 2;
  const size_t total = out.numel();
  float* data = out.data_f32();

  for (size_t idx = 0; idx < total; idx += head_dim) {
    const size_t token_index = (idx / head_dim) % seq_len;
    const uint32_t token_pos = positions[token_index];
    for (size_t j = 0; j < half; ++j) {
      const size_t i0 = config.style == RopeStyle::SplitHalf ? j : 2 * j;
      const size_t i1 = config.style == RopeStyle::SplitHalf ? j + half : 2 * j + 1;
      const float theta =
          static_cast<float>((2.0 * static_cast<double>(j)) / static_cast<double>(head_dim));
      const float inv_freq = 1.0f / std::pow(config.base, theta);
      const float angle = static_cast<float>(token_pos) * inv_freq;
      const float cos_a = std::cos(angle);
      const float sin_a = std::sin(angle);

      const float x0 = data[idx + i0];
      const float x1 = data[idx + i1];
      data[idx + i0] = x0 * cos_a - x1 * sin_a;
      data[idx + i1] = x0 * sin_a + x1 * cos_a;
    }
  }
  return out;
}

void CPUBackend::softmax_inplace(Tensor& x) {
  assert(x.dtype() == DType::F32);
  assert(x.shape().size() >= 1);
  size_t outer = x.numel() / x.shape().back();
  size_t dim = x.shape().back();

  float* data = x.data_f32();
  for (size_t i = 0; i < outer; ++i) {
    float* row = data + i * dim;
    float max_val = *std::max_element(row, row + dim);
    float sum = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
      row[j] = std::exp(row[j] - max_val);
      sum += row[j];
    }
    float inv_sum = 1.0f / sum;
    for (size_t j = 0; j < dim; ++j) row[j] *= inv_sum;
  }
}

Tensor CPUBackend::rms_norm(const Tensor& x, const Tensor& weight, float eps) {
  assert(x.dtype() == DType::F32 && weight.dtype() == DType::F32);
  Tensor out = x.clone();
  size_t dim = x.shape().back();
  size_t outer = x.numel() / dim;

  const float* w = weight.data_f32();
  float* data = out.data_f32();

  for (size_t i = 0; i < outer; ++i) {
    float* row = data + i * dim;
    float ss = 0.0f;
    for (size_t j = 0; j < dim; ++j) ss += row[j] * row[j];
    float rms = std::sqrt(ss / dim + eps);
    float inv_rms = 1.0f / rms;
    for (size_t j = 0; j < dim; ++j) row[j] = row[j] * inv_rms * w[j];
  }
  return out;
}

Tensor CPUBackend::silu_mul(const Tensor& gate, const Tensor& up) {
  assert(gate.dtype() == DType::F32 && up.dtype() == DType::F32);
  assert(gate.numel() == up.numel());
  Tensor out(DType::F32, gate.shape());
  size_t n = gate.numel();
  const float* g = gate.data_f32();
  const float* u = up.data_f32();
  float* d = out.data_f32();
  for (size_t i = 0; i < n; ++i) {
    float sig = 1.0f / (1.0f + std::exp(-g[i]));
    d[i] = g[i] * sig * u[i];
  }
  return out;
}

Tensor CPUBackend::add(const Tensor& a, const Tensor& b) {
  assert(a.numel() == b.numel());
  Tensor out = a.clone();
  size_t n = a.numel();
  const float* bd = b.data_f32();
  float* od = out.data_f32();
  for (size_t i = 0; i < n; ++i) od[i] += bd[i];
  return out;
}

Tensor CPUBackend::dot_product_attention(const Tensor& q, const Tensor& k,
                                          const Tensor& v, const Tensor* mask) {
  validate_attention(q, k, v, mask);

  const size_t q_heads = q.shape()[0];
  const size_t q_len = q.shape()[1];
  const size_t head_dim = q.shape()[2];
  const size_t kv_heads = k.shape()[0];
  const size_t kv_len = k.shape()[1];
  const size_t group_size = q_heads / kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  Tensor out(DType::F32, {q_heads, q_len, head_dim});
  out.zero();

  const float* qd = q.data_f32();
  const float* kd = k.data_f32();
  const float* vd = v.data_f32();
  float* od = out.data_f32();

  std::vector<float> scores(kv_len);
  std::vector<float> probabilities(kv_len);
  std::vector<unsigned char> valid(kv_len);

  for (size_t h = 0; h < q_heads; ++h) {
    const size_t kv_head = h / group_size;
    for (size_t qi = 0; qi < q_len; ++qi) {
      bool any_valid = false;
      float row_max = 0.0f;

      for (size_t ki = 0; ki < kv_len; ++ki) {
        float sum = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
          sum += qd[(h * q_len + qi) * head_dim + d] *
                 kd[(kv_head * kv_len + ki) * head_dim + d];
        }

        float score = sum * scale;
        valid[ki] = 1;
        if (mask) {
          const float bias = mask_at(*mask, h, qi, ki);
          if (bias <= kMaskedScore) {
            valid[ki] = 0;
          } else {
            score += bias;
          }
        }
        scores[ki] = score;
        if (valid[ki] && (!any_valid || score > row_max)) {
          row_max = score;
          any_valid = true;
        }
      }

      // A row with no admissible key produces a defined all-zero output
      // instead of dividing by a zero softmax denominator or producing NaNs.
      if (!any_valid) continue;

      float denominator = 0.0f;
      for (size_t ki = 0; ki < kv_len; ++ki) {
        if (!valid[ki]) {
          probabilities[ki] = 0.0f;
          continue;
        }
        const float e = std::exp(scores[ki] - row_max);
        probabilities[ki] = e;
        denominator += e;
      }
      if (!(denominator > 0.0f)) continue;

      const float inverse = 1.0f / denominator;
      float* output_row = od + (h * q_len + qi) * head_dim;
      for (size_t ki = 0; ki < kv_len; ++ki) {
        if (!valid[ki]) continue;
        const float probability = probabilities[ki] * inverse;
        const float* value_row = vd + (kv_head * kv_len + ki) * head_dim;
        for (size_t d = 0; d < head_dim; ++d)
          output_row[d] += probability * value_row[d];
      }
    }
  }

  return out;
}

Tensor CPUBackend::attention(const Tensor& q, const Tensor& k,
                             const Tensor& v, const Tensor* mask) {
  return dot_product_attention(q, k, v, mask);
}

Tensor CPUBackend::fused_qk_v_attention(const Tensor& q, const Tensor& k,
                                         const Tensor& v) {
  return dot_product_attention(q, k, v, nullptr);
}
