/**
 * Independent numerical regression tests for attention and RoPE.
 *
 * The reference implementations below are scalar loops written directly from
 * the mathematical definition. They do not call CPUBackend or reuse its
 * implementation. Tests use runtime checks only, so they remain meaningful
 * with NDEBUG and with the Release -ffast-math flags used by this project.
 */
#include "backend/backend.h"
#include "backend/cpu_backend.h"
#include "core/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(name, expr) do { \
  if (!(expr)) { \
    std::cerr << "  FAIL: " << name << "\n"; \
    tests_failed++; \
  } else { \
    std::cout << "  PASS: " << name << "\n"; \
    tests_passed++; \
  } \
} while (0)

static Tensor sequence_tensor(const std::vector<size_t>& shape, float seed) {
  Tensor t(DType::F32, shape);
  float* data = t.data_f32();
  for (size_t i = 0; i < t.numel(); ++i)
    data[i] = std::sin(seed + static_cast<float>(i + 1) * 0.37f) * 1.3f;
  return t;
}

static float reference_mask_value(const Tensor* mask, size_t head,
                                  size_t q_index, size_t k_index) {
  if (!mask) return 0.0f;
  const float* data = mask->data_f32();
  if (mask->rank() == 2)
    return data[q_index * mask->shape()[1] + k_index];
  const size_t mask_head = mask->shape()[0] == 1 ? 0 : head;
  return data[(mask_head * mask->shape()[1] + q_index) * mask->shape()[2] + k_index];
}

static std::vector<float> reference_attention(const Tensor& q, const Tensor& k,
                                               const Tensor& v, const Tensor* mask) {
  const size_t q_heads = q.shape()[0];
  const size_t q_len = q.shape()[1];
  const size_t head_dim = q.shape()[2];
  const size_t kv_heads = k.shape()[0];
  const size_t kv_len = k.shape()[1];
  const size_t group_size = q_heads / kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> out(q_heads * q_len * head_dim, 0.0f);
  std::vector<float> scores(kv_len);
  std::vector<unsigned char> valid(kv_len);

  for (size_t h = 0; h < q_heads; ++h) {
    const size_t kv_head = h / group_size;
    for (size_t qi = 0; qi < q_len; ++qi) {
      bool any_valid = false;
      float row_max = 0.0f;
      for (size_t ki = 0; ki < kv_len; ++ki) {
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; ++d)
          dot += q.data_f32()[(h * q_len + qi) * head_dim + d] *
                 k.data_f32()[(kv_head * kv_len + ki) * head_dim + d];
        float score = dot * scale;
        valid[ki] = 1;
        if (mask) {
          const float bias = reference_mask_value(mask, h, qi, ki);
          if (bias <= -1.0e30f) {
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
      if (!any_valid) continue;

      float denominator = 0.0f;
      for (size_t ki = 0; ki < kv_len; ++ki) {
        if (!valid[ki]) continue;
        const float e = std::exp(scores[ki] - row_max);
        scores[ki] = e;
        denominator += e;
      }
      if (!(denominator > 0.0f)) continue;

      for (size_t ki = 0; ki < kv_len; ++ki) {
        if (!valid[ki]) continue;
        const float probability = scores[ki] / denominator;
        for (size_t d = 0; d < head_dim; ++d)
          out[(h * q_len + qi) * head_dim + d] +=
              probability * v.data_f32()[(kv_head * kv_len + ki) * head_dim + d];
      }
    }
  }
  return out;
}

static bool compare_output(const Tensor& actual, const std::vector<float>& expected,
                           float tolerance = 1.0e-5f, const char* label = "") {
  if (actual.numel() != expected.size()) {
    std::cerr << "  shape/value-count mismatch for " << label << "\n";
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual.data_f32()[i] - expected[i]) > tolerance) {
      std::cerr << "  value mismatch for " << label << " at " << i
                << ": actual=" << actual.data_f32()[i]
                << " expected=" << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

static bool throws_invalid(std::function<void()> fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

static void test_mha_multi_token() {
  std::cout << "Test: MHA with multiple query and KV tokens\n";
  CPUBackend cpu;
  Tensor q = sequence_tensor({3, 2, 4}, 0.2f);
  Tensor k = sequence_tensor({3, 3, 4}, 1.1f);
  Tensor v = sequence_tensor({3, 3, 4}, 2.3f);
  Tensor out = cpu.attention(q, k, v, nullptr);
  CHECK("MHA output shape", out.shape() == std::vector<size_t>({3, 2, 4}));
  CHECK("MHA independent reference", compare_output(out, reference_attention(q, k, v, nullptr), 1.0e-5f, "MHA"));
}

static void test_mqa() {
  std::cout << "Test: MQA with four query heads sharing one KV head\n";
  CPUBackend cpu;
  Tensor q = sequence_tensor({4, 2, 3}, 0.4f);
  Tensor k = sequence_tensor({1, 3, 3}, 1.7f);
  Tensor v = sequence_tensor({1, 3, 3}, 2.9f);
  Tensor out = cpu.attention(q, k, v, nullptr);
  CHECK("MQA output shape", out.shape() == std::vector<size_t>({4, 2, 3}));
  CHECK("MQA independent reference", compare_output(out, reference_attention(q, k, v, nullptr), 1.0e-5f, "MQA"));
}

static void run_gqa_case(size_t q_heads, size_t kv_heads) {
  const std::string label = "GQA " + std::to_string(q_heads / kv_heads) + ":1";
  CPUBackend cpu;
  Tensor q = sequence_tensor({q_heads, 2, 4}, 0.3f + static_cast<float>(q_heads));
  Tensor k = sequence_tensor({kv_heads, 3, 4}, 1.4f + static_cast<float>(kv_heads));
  Tensor v = sequence_tensor({kv_heads, 3, 4}, 2.6f + static_cast<float>(kv_heads));
  Tensor out = cpu.attention(q, k, v, nullptr);
  CHECK(label + " output shape",
        out.shape() == std::vector<size_t>({q_heads, 2, 4}));
  CHECK(label + " independent reference",
        compare_output(out, reference_attention(q, k, v, nullptr), 1.0e-5f, label.c_str()));
}

static void test_gqa_ratios() {
  std::cout << "Test: GQA 3:1, 4:1 and 7:1 head mapping\n";
  run_gqa_case(6, 2);
  run_gqa_case(8, 2);
  run_gqa_case(7, 1);
}

static void test_causal_and_broadcast_masks() {
  std::cout << "Test: causal and broadcast masks\n";
  CPUBackend cpu;
  Tensor q = sequence_tensor({2, 3, 3}, 0.5f);
  Tensor k = sequence_tensor({2, 4, 3}, 1.6f);
  Tensor v = sequence_tensor({2, 4, 3}, 2.7f);

  Tensor mask3(DType::F32, {1, 3, 4});
  mask3.fill(0.0f);
  for (size_t qi = 0; qi < 3; ++qi)
    for (size_t ki = 0; ki < 4; ++ki)
      if (ki > qi) mask3.data_f32()[qi * 4 + ki] = -1.0e30f;

  Tensor out3 = cpu.attention(q, k, v, &mask3);
  CHECK("rank-3 broadcast causal mask reference",
        compare_output(out3, reference_attention(q, k, v, &mask3), 1.0e-5f, "causal-broadcast"));

  // Build the equivalent rank-2 mask explicitly.
  Tensor mask2d(DType::F32, {3, 4});
  std::memcpy(mask2d.data_f32(), mask3.data_f32(), mask3.size_bytes());
  Tensor out2 = cpu.attention(q, k, v, &mask2d);
  CHECK("rank-2 broadcast mask reference",
        compare_output(out2, reference_attention(q, k, v, &mask2d), 1.0e-5f, "causal-rank2"));

  Tensor maskPerHead(DType::F32, {2, 3, 4});
  maskPerHead.fill(0.0f);
  for (size_t h = 0; h < 2; ++h)
    for (size_t qi = 0; qi < 3; ++qi)
      for (size_t ki = 0; ki < 4; ++ki)
        if ((h == 0 && ki > qi) || (h == 1 && ki < qi))
          maskPerHead.data_f32()[(h * 3 + qi) * 4 + ki] = -1.0e30f;
  Tensor outPerHead = cpu.attention(q, k, v, &maskPerHead);
  CHECK("per-head mask reference",
        compare_output(outPerHead, reference_attention(q, k, v, &maskPerHead), 1.0e-5f, "per-head"));
}

static void test_fully_masked_rows() {
  std::cout << "Test: fully masked rows produce defined zero output\n";
  CPUBackend cpu;
  Tensor q = sequence_tensor({2, 2, 2}, 0.8f);
  Tensor k = sequence_tensor({2, 2, 2}, 1.9f);
  Tensor v = sequence_tensor({2, 2, 2}, 3.1f);
  Tensor mask(DType::F32, {1, 2, 2});
  mask.fill(0.0f);
  // First query row is fully forbidden; second row is fully admissible.
  mask.data_f32()[0] = -1.0e30f;
  mask.data_f32()[1] = -1.0e30f;
  Tensor out = cpu.attention(q, k, v, &mask);
  const std::vector<float> expected = reference_attention(q, k, v, &mask);
  CHECK("fully masked reference", compare_output(out, expected, 1.0e-5f, "fully-masked"));
  CHECK("fully masked row is exactly zero", out.data_f32()[0] == 0.0f && out.data_f32()[1] == 0.0f);

  Tensor allMasked(DType::F32, {1, 2, 2});
  allMasked.fill(-1.0e30f);
  Tensor allOut = cpu.attention(q, k, v, &allMasked);
  bool allFinite = true;
  for (size_t i = 0; i < allOut.numel(); ++i)
    if (!std::isfinite(allOut.data_f32()[i])) allFinite = false;
  CHECK("all rows fully masked stay finite", allFinite);
  bool allZero = true;
  for (size_t i = 0; i < allOut.numel(); ++i)
    if (allOut.data_f32()[i] != 0.0f) allZero = false;
  CHECK("all rows fully masked are zero", allZero);
}

static void test_invalid_attention_inputs() {
  std::cout << "Test: invalid attention shapes and dtypes throw in Release\n";
  CPUBackend cpu;
  Tensor q = sequence_tensor({2, 2, 3}, 0.2f);
  Tensor k = sequence_tensor({2, 2, 3}, 0.3f);
  Tensor v = sequence_tensor({2, 2, 3}, 0.4f);

  CHECK("rank-2 q rejected", throws_invalid([&] { cpu.attention(sequence_tensor({2, 3}, 0.2f), k, v, nullptr); }));
  CHECK("non-F32 q rejected", throws_invalid([&] { cpu.attention(Tensor(DType::F16, {2, 2, 3}), k, v, nullptr); }));
  CHECK("q-head divisibility rejected", throws_invalid([&] {
    cpu.attention(sequence_tensor({3, 2, 3}, 0.2f), sequence_tensor({2, 2, 3}, 0.3f),
                  sequence_tensor({2, 2, 3}, 0.4f), nullptr);
  }));
  CHECK("k/v head mismatch rejected", throws_invalid([&] {
    cpu.attention(q, k, sequence_tensor({1, 2, 3}, 0.4f), nullptr);
  }));
  CHECK("k/v sequence mismatch rejected", throws_invalid([&] {
    cpu.attention(q, k, sequence_tensor({2, 3, 3}, 0.4f), nullptr);
  }));
  CHECK("head-dim mismatch rejected", throws_invalid([&] {
    cpu.attention(q, sequence_tensor({2, 2, 4}, 0.3f), sequence_tensor({2, 2, 4}, 0.4f), nullptr);
  }));
  CHECK("zero dimension rejected", throws_invalid([&] {
    cpu.attention(Tensor(DType::F32, {2, 0, 3}), k, v, nullptr);
  }));
  CHECK("bad rank-2 mask shape rejected", throws_invalid([&] {
    Tensor badMask(DType::F32, {1, 2});
    cpu.attention(q, k, v, &badMask);
  }));
  CHECK("bad rank-3 mask head count rejected", throws_invalid([&] {
    Tensor badMask(DType::F32, {3, 2, 2});
    cpu.attention(q, k, v, &badMask);
  }));
  CHECK("non-F32 mask rejected", throws_invalid([&] {
    Tensor badMask(DType::F16, {1, 2, 2});
    cpu.attention(q, k, v, &badMask);
  }));
}

static std::vector<float> reference_rope(const Tensor& x,
                                         const std::vector<uint32_t>& positions,
                                         const RopeConfig& config) {
  const size_t head_dim = config.head_dim;
  const size_t half = head_dim / 2;
  const size_t seq_len = x.shape()[x.rank() - 2];
  std::vector<float> out(x.data_f32(), x.data_f32() + x.numel());
  for (size_t block = 0; block < x.numel() / head_dim; ++block) {
    const size_t token_index = block % seq_len;
    const uint32_t token_pos = positions[token_index];
    for (size_t j = 0; j < half; ++j) {
      const size_t i0 = config.style == RopeStyle::SplitHalf ? j : 2 * j;
      const size_t i1 = config.style == RopeStyle::SplitHalf ? j + half : 2 * j + 1;
      const float theta = static_cast<float>((2.0 * static_cast<double>(j)) /
                                             static_cast<double>(head_dim));
      const float inv_freq = 1.0f / std::pow(config.base, theta);
      const float angle = static_cast<float>(token_pos) * inv_freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const size_t base = block * head_dim;
      const float x0 = out[base + i0];
      const float x1 = out[base + i1];
      out[base + i0] = x0 * c - x1 * s;
      out[base + i1] = x0 * s + x1 * c;
    }
  }
  return out;
}

static void test_rope_positions_and_bases() {
  std::cout << "Test: RoPE positions and bases\n";
  CPUBackend cpu;
  Tensor x = sequence_tensor({2, 3, 4}, 0.6f);
  const std::vector<uint32_t> non_contiguous = {0, 2, 5};
  for (float base : {10000.0f, 100000.0f, 1000000.0f}) {
    RopeConfig config{4, base, RopeStyle::SplitHalf};
    Tensor out = cpu.rope(x, non_contiguous, config);
    const std::string label = "RoPE split-half base " + std::to_string(static_cast<int>(base));
    CHECK(label + " output shape", out.shape() == x.shape());
    CHECK(label + " independent reference",
          compare_output(out, reference_rope(x, non_contiguous, config), 1.0e-5f, label.c_str()));
  }

  RopeConfig interleaved{4, 10000.0f, RopeStyle::Interleaved};
  Tensor outInterleaved = cpu.rope(x, std::vector<uint32_t>{1, 3, 7}, interleaved);
  CHECK("RoPE interleaved reference",
        compare_output(outInterleaved,
                       reference_rope(x, std::vector<uint32_t>{1, 3, 7}, interleaved),
                       1.0e-5f, "interleaved"));

  Tensor outContiguous = cpu.rope(x, 4, 4);
  RopeConfig contiguousConfig{4, 10000.0f, RopeStyle::SplitHalf};
  CHECK("RoPE contiguous convenience positions",
        compare_output(outContiguous,
                       reference_rope(x, std::vector<uint32_t>{4, 5, 6}, contiguousConfig),
                       1.0e-5f, "contiguous"));

  Tensor x2 = sequence_tensor({3, 4}, 1.2f);
  Tensor out2 = cpu.rope(x2, non_contiguous, contiguousConfig);
  CHECK("RoPE rank-2 reference",
        compare_output(out2, reference_rope(x2, non_contiguous, contiguousConfig),
                       1.0e-5f, "rank2"));
}

static void test_invalid_rope_inputs() {
  std::cout << "Test: invalid RoPE inputs throw in Release\n";
  CPUBackend cpu;
  Tensor x = sequence_tensor({2, 3, 4}, 0.1f);
  CHECK("RoPE odd head_dim rejected", throws_invalid([&] {
    cpu.rope(x, std::vector<uint32_t>{0, 1, 2}, RopeConfig{3, 10000.0f, RopeStyle::SplitHalf});
  }));
  CHECK("RoPE position count rejected", throws_invalid([&] {
    cpu.rope(x, std::vector<uint32_t>{0, 1}, RopeConfig{4, 10000.0f, RopeStyle::SplitHalf});
  }));
  CHECK("RoPE dimension mismatch rejected", throws_invalid([&] {
    cpu.rope(x, std::vector<uint32_t>{0, 1, 2}, RopeConfig{8, 10000.0f, RopeStyle::SplitHalf});
  }));
  CHECK("RoPE nonpositive base rejected", throws_invalid([&] {
    cpu.rope(x, std::vector<uint32_t>{0, 1, 2}, RopeConfig{4, 0.0f, RopeStyle::SplitHalf});
  }));
  CHECK("RoPE rank-1 rejected", throws_invalid([&] {
    cpu.rope(sequence_tensor({4}, 0.1f), std::vector<uint32_t>{0}, RopeConfig{4, 10000.0f, RopeStyle::SplitHalf});
  }));
  CHECK("RoPE non-F32 rejected", throws_invalid([&] {
    cpu.rope(Tensor(DType::F16, {2, 3, 4}), std::vector<uint32_t>{0, 1, 2},
             RopeConfig{4, 10000.0f, RopeStyle::SplitHalf});
  }));
}

int main() {
  std::cout << "\nKernel Safety Unit Tests\n";
  std::cout << "════════════════════════════════════════\n\n";

  test_mha_multi_token();
  std::cout << "\n";
  test_mqa();
  std::cout << "\n";
  test_gqa_ratios();
  std::cout << "\n";
  test_causal_and_broadcast_masks();
  std::cout << "\n";
  test_fully_masked_rows();
  std::cout << "\n";
  test_invalid_attention_inputs();
  std::cout << "\n";
  test_rope_positions_and_bases();
  std::cout << "\n";
  test_invalid_rope_inputs();

  std::cout << "\n────────────────────────────────────────\n";
  std::cout << "  Passed: " << tests_passed << " / " << (tests_passed + tests_failed) << "\n";
  std::cout << "  Failed: " << tests_failed << "\n";
  return tests_failed == 0 ? 0 : 1;
}
