#include "core/tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::I32: return 4;
    case DType::I8:  return 1;
  }
  return 4;
}

Tensor::Tensor(DType dtype, std::vector<size_t> shape)
  : dtype_(dtype), shape_(std::move(shape)) {
  data_.resize(numel() * dtype_size(dtype_));
}

Tensor::Tensor(DType dtype, std::vector<size_t> shape, std::vector<float> data)
  : dtype_(dtype), shape_(std::move(shape)) {
  assert(dtype == DType::F32);
  data_.resize(data.size() * sizeof(float));
  std::memcpy(data_.data(), data.data(), data.size() * sizeof(float));
}

size_t Tensor::numel() const {
  if (shape_.empty()) return 0;
  return std::accumulate(shape_.begin(), shape_.end(), (size_t)1, std::multiplies<size_t>());
}

size_t Tensor::size_bytes() const {
  return numel() * dtype_size(dtype_);
}

size_t Tensor::stride(size_t dim) const {
  size_t s = 1;
  for (size_t i = dim + 1; i < shape_.size(); ++i) s *= shape_[i];
  return s;
}

float Tensor::at(std::initializer_list<size_t> indices) const {
  assert(indices.size() == shape_.size());
  size_t offset = offset_;
  size_t i = 0;
  for (auto idx : indices) {
    offset += idx * stride(i++);
  }
  switch (dtype_) {
    case DType::F32: return reinterpret_cast<const float*>(data_.data())[offset];
    default: return 0.0f;
  }
}

float& Tensor::at(std::initializer_list<size_t> indices) {
  assert(indices.size() == shape_.size());
  size_t offset = offset_;
  size_t i = 0;
  for (auto idx : indices) {
    offset += idx * stride(i++);
  }
  assert(dtype_ == DType::F32);
  return reinterpret_cast<float*>(data_.data())[offset];
}

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
  assert(dim < shape_.size() && start <= end && end <= shape_[dim]);
  Tensor t;
  t.dtype_ = dtype_;
  t.shape_ = shape_;
  t.shape_[dim] = end - start;
  t.offset_ = offset_ + start * stride(dim);
  t.data_ = data_;
  return t;
}

Tensor Tensor::clone() const {
  Tensor t(dtype_, shape_);
  std::memcpy(t.data_.data(), data_.data(), size_bytes());
  return t;
}

void Tensor::fill(float v) {
  assert(dtype_ == DType::F32);
  std::fill(data_f32(), data_f32() + numel(), v);
}

void Tensor::zero() {
  std::memset(data_.data(), 0, size_bytes());
}

void Tensor::copy_from(const Tensor& other) {
  assert(dtype_ == other.dtype_ && shape_ == other.shape_);
  std::memcpy(data_.data(), other.data_.data(), size_bytes());
}
