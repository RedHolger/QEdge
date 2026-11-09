#pragma once
#include "backend/backend.h"
#include <Accelerate/Accelerate.h>

class CPUBackend : public Backend {
public:
  BackendType type() const override { return BackendType::CPU; }
  std::string name() const override { return "CPU (Accelerate/BLAS)"; }

  Tensor matmul(const Tensor& a, const Tensor& b) override;
  using Backend::rope;
  Tensor rope(const Tensor& x, const std::vector<uint32_t>& positions,
              const RopeConfig& config) override;
  void softmax_inplace(Tensor& x) override;
  Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) override;
  Tensor silu_mul(const Tensor& gate, const Tensor& up) override;
  Tensor add(const Tensor& a, const Tensor& b) override;
  Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor* mask = nullptr) override;
  Tensor fused_qk_v_attention(const Tensor& q, const Tensor& k, const Tensor& v) override;

  bool is_available() const override { return true; }

private:
  Tensor dot_product_attention(const Tensor& q, const Tensor& k,
                               const Tensor& v, const Tensor* mask);
};
