#pragma once
#include "backend/backend.h"
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

class MetalBackend : public Backend {
public:
  MetalBackend();
  ~MetalBackend() override;

  BackendType type() const override { return BackendType::Metal; }
  std::string name() const override {
    return "Metal (MPS/GPU; attention CPU fallback)";
  }

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

  bool is_available() const override {
    return device_ != nil && command_queue_ != nil;
  }

private:
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> command_queue_ = nil;

  id<MTLBuffer> tensor_to_buffer(const Tensor& t);
  Tensor buffer_to_tensor(id<MTLBuffer> buf, DType dtype, std::vector<size_t> shape);

  MPSMatrixDescriptor* make_matrix_desc(size_t rows, size_t cols, MPSDataType dtype);
};

std::unique_ptr<Backend> create_metal_backend();
