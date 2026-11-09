#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cassert>
#include <cstring>

enum class DType : uint8_t {
  F32,
  F16,
  I32,
  I8
};

size_t dtype_size(DType dtype);

class Tensor {
public:
  Tensor() = default;
  Tensor(DType dtype, std::vector<size_t> shape);
  Tensor(DType dtype, std::vector<size_t> shape, std::vector<float> data);

  DType dtype() const { return dtype_; }
  const std::vector<size_t>& shape() const { return shape_; }
  size_t rank() const { return shape_.size(); }
  size_t numel() const;
  size_t size_bytes() const;
  bool empty() const { return data_.empty(); }

  float* data_f32() { return reinterpret_cast<float*>(data_.data()); }
  const float* data_f32() const { return reinterpret_cast<const float*>(data_.data()); }
  uint8_t* data_raw() { return data_.data(); }
  const uint8_t* data_raw() const { return data_.data(); }

  float at(std::initializer_list<size_t> indices) const;
  float& at(std::initializer_list<size_t> indices);

  Tensor slice(size_t dim, size_t start, size_t end) const;
  Tensor clone() const;

  void fill(float v);
  void zero();
  void copy_from(const Tensor& other);

  size_t stride(size_t dim) const;

private:
  DType dtype_ = DType::F32;
  std::vector<size_t> shape_;
  std::vector<uint8_t> data_;
  size_t offset_ = 0;
};
