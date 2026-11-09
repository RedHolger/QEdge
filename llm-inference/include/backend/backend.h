#pragma once
#include "core/tensor.h"
#include "core/config.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class BackendType {
  CPU,
  Metal,
  QNN     // Qualcomm AI Engine Direct (Hexagon NPU) — see qnn_stub_backend.h
};

enum class RopeStyle {
  SplitHalf,   // rotate (j, j + head_dim/2) pairs (LLaMA/GPT-NeoX style)
  Interleaved  // rotate (2j, 2j + 1) pairs (GPT-J style)
};

struct RopeConfig {
  uint32_t head_dim = 0;
  float base = 10000.0f;
  RopeStyle style = RopeStyle::SplitHalf;
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual BackendType type() const = 0;
  virtual std::string name() const = 0;

  virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;

  // Apply rotary embedding to every [head, sequence, dimension] slice. The
  // sequence axis uses the corresponding explicit token position; positions
  // may be non-contiguous. The legacy single-position overload treats pos as
  // the first position of a contiguous run.
  virtual Tensor rope(const Tensor& x, const std::vector<uint32_t>& positions,
                      const RopeConfig& config) = 0;
  Tensor rope(const Tensor& x, uint32_t pos, uint32_t head_dim) {
    size_t seq_len = x.rank() >= 2 ? x.shape()[x.rank() - 2] : 0;
    std::vector<uint32_t> positions(seq_len);
    for (size_t i = 0; i < seq_len; ++i)
      positions[i] = pos + static_cast<uint32_t>(i);
    return rope(x, positions, RopeConfig{head_dim});
  }

  virtual void softmax_inplace(Tensor& x) = 0;
  virtual Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) = 0;
  virtual Tensor silu_mul(const Tensor& gate, const Tensor& up) = 0;
  virtual Tensor add(const Tensor& a, const Tensor& b) = 0;

  // Attention tensors use [heads, sequence, dimension]; the first axis is not
  // a batch axis. Q may have more heads than K/V for GQA/MQA. Mask is additive
  // F32 broadcast as either [q_len, kv_len] or [1|q_heads, q_len, kv_len].
  virtual Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                           const Tensor* mask = nullptr) = 0;
  virtual Tensor fused_qk_v_attention(const Tensor& q, const Tensor& k, const Tensor& v) = 0;

  virtual bool is_available() const = 0;
};

std::unique_ptr<Backend> create_backend(BackendType type);
