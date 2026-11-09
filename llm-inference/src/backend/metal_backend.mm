#import "backend/metal_backend.h"
#include "backend/cpu_backend.h"
#include "backend/metal_factory.h"
#include <cassert>

MetalBackend::MetalBackend() {
  device_ = MTLCreateSystemDefaultDevice();
  if (device_) {
    command_queue_ = [device_ newCommandQueue];
  }
}

MetalBackend::~MetalBackend() {
  [command_queue_ release];
  [device_ release];
}

MPSMatrixDescriptor* MetalBackend::make_matrix_desc(size_t rows, size_t cols, MPSDataType dtype) {
  return [MPSMatrixDescriptor
      matrixDescriptorWithRows:(NSUInteger)rows
      columns:(NSUInteger)cols
      rowBytes:(NSUInteger)cols * 4
      dataType:dtype];
}

id<MTLBuffer> MetalBackend::tensor_to_buffer(const Tensor& t) {
  NSUInteger len = t.size_bytes();
  id<MTLBuffer> buf = [device_ newBufferWithBytes:t.data_raw()
                                           length:len
                                          options:MTLResourceStorageModeShared];
  return [buf autorelease];
}

Tensor MetalBackend::buffer_to_tensor(id<MTLBuffer> buf, DType dtype,
                                       std::vector<size_t> shape) {
  Tensor t(dtype, shape);
  std::memcpy(t.data_raw(), [buf contents], t.size_bytes());
  return t;
}

Tensor MetalBackend::matmul(const Tensor& a, const Tensor& b) {
  assert(a.dtype() == DType::F32 && b.dtype() == DType::F32);
  assert(a.shape().size() == 2 && b.shape().size() == 2);
  assert(a.shape()[1] == b.shape()[0]);

  size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];

  id<MTLBuffer> bufA = tensor_to_buffer(a);
  id<MTLBuffer> bufB = tensor_to_buffer(b);
  id<MTLBuffer> bufC = [device_ newBufferWithLength:(NSUInteger)(M * N * 4)
                                            options:MTLResourceStorageModeShared];

  MPSMatrixDescriptor* descA = make_matrix_desc(M, K, MPSDataTypeFloat32);
  MPSMatrixDescriptor* descB = make_matrix_desc(K, N, MPSDataTypeFloat32);
  MPSMatrixDescriptor* descC = make_matrix_desc(M, N, MPSDataTypeFloat32);

  MPSMatrixMultiplication* mmul = [[MPSMatrixMultiplication alloc]
      initWithDevice:device_
      transposeLeft:false transposeRight:false
      resultRows:(NSUInteger)M resultColumns:(NSUInteger)N
      interiorColumns:(NSUInteger)K alpha:1.0 beta:0.0];

  MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
  MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
  MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

  id<MTLCommandBuffer> cmdBuf = [command_queue_ commandBuffer];
  [mmul encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
  [cmdBuf commit];
  [cmdBuf waitUntilCompleted];

  Tensor result = buffer_to_tensor(bufC, DType::F32, {M, N});

  [matC release];
  [matB release];
  [matA release];
  [mmul release];
  [descC release];
  [descB release];
  [descA release];
  [bufC release];

  return result;
}

Tensor MetalBackend::rope(const Tensor& x, const std::vector<uint32_t>& positions,
                          const RopeConfig& config) {
  CPUBackend cpu;
  return cpu.rope(x, positions, config);
}

void MetalBackend::softmax_inplace(Tensor& x) {
  CPUBackend cpu;
  cpu.softmax_inplace(x);
}

Tensor MetalBackend::rms_norm(const Tensor& x, const Tensor& weight, float eps) {
  CPUBackend cpu;
  return cpu.rms_norm(x, weight, eps);
}

Tensor MetalBackend::silu_mul(const Tensor& gate, const Tensor& up) {
  CPUBackend cpu;
  return cpu.silu_mul(gate, up);
}

Tensor MetalBackend::add(const Tensor& a, const Tensor& b) {
  CPUBackend cpu;
  return cpu.add(a, b);
}

// MPS has no fused attention primitive in this lab. Keep the Metal backend's
// GPU matmul path, but make attention an explicit CPU fallback rather than
// returning raw QK scores as if they were attention.
Tensor MetalBackend::attention(const Tensor& q, const Tensor& k,
                                const Tensor& v, const Tensor* mask) {
  CPUBackend cpu;
  return cpu.attention(q, k, v, mask);
}

Tensor MetalBackend::fused_qk_v_attention(const Tensor& q, const Tensor& k,
                                           const Tensor& v) {
  CPUBackend cpu;
  return cpu.attention(q, k, v, nullptr);
}

std::unique_ptr<Backend> create_metal_backend() {
  auto backend = std::make_unique<MetalBackend>();
  if (backend->is_available()) return backend;
  return nullptr;
}
