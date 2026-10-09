#pragma once

#include <ggml.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "tensor_test_support.h"
#include "../compute_models/d5_gguf_test_support.h"
#include "../../src/compute_route_vulkan/vulkan_program_internal.h"

namespace vulkan_gguf_test {
namespace c = hyle::compute;
namespace v = c::vulkan;

inline constexpr std::array kTypes{
    c::GgufQuantizedType::kQ3K,   c::GgufQuantizedType::kQ4K,
    c::GgufQuantizedType::kQ5K,   c::GgufQuantizedType::kQ6K,
    c::GgufQuantizedType::kQ8_0,  c::GgufQuantizedType::kIq3Xxs,
    c::GgufQuantizedType::kIq3S,  c::GgufQuantizedType::kIq4Nl,
    c::GgufQuantizedType::kIq4Xs, c::GgufQuantizedType::kMxfp4,
    c::GgufQuantizedType::kQ4_0, c::GgufQuantizedType::kQ4_1,
    c::GgufQuantizedType::kQ2K, c::GgufQuantizedType::kIq2Xxs,
    c::GgufQuantizedType::kIq2Xs, c::GgufQuantizedType::kIq2S,
    c::GgufQuantizedType::kIq1S, c::GgufQuantizedType::kIq1M};

inline constexpr std::array kEmbeddingTypes{
    c::GgufQuantizedType::kQ2K, c::GgufQuantizedType::kQ3K,
    c::GgufQuantizedType::kQ4_0, c::GgufQuantizedType::kQ4_1,
    c::GgufQuantizedType::kQ4K, c::GgufQuantizedType::kQ5K,
    c::GgufQuantizedType::kQ6K, c::GgufQuantizedType::kQ8_0};

inline constexpr std::array kGroupedTypes{
    c::GgufQuantizedType::kIq4Xs, c::GgufQuantizedType::kQ4_0,
    c::GgufQuantizedType::kQ4_1, c::GgufQuantizedType::kQ2K,
    c::GgufQuantizedType::kIq2Xxs, c::GgufQuantizedType::kIq2Xs,
    c::GgufQuantizedType::kIq2S, c::GgufQuantizedType::kIq1S,
    c::GgufQuantizedType::kIq1M};

inline bool iq1_delta() {
#if defined(_WIN32)
  char *value = nullptr; std::size_t size = 0;
  const bool selected = _dupenv_s(&value, &size, "HYLE_VULKAN_IQ1_DELTA") == 0 && value && value[0] == '1';
  std::free(value); return selected;
#else
  const auto *value = std::getenv("HYLE_VULKAN_IQ1_DELTA"); return value && value[0] == '1';
#endif
}

inline std::span<const c::GgufQuantizedType> selected_matrix_types(bool grouped = false) {
  static constexpr std::array only{c::GgufQuantizedType::kIq1S};
  if (iq1_delta()) { std::puts("VULKAN_NUMERIC_DELTA types=IQ1_S"); return only; }
  return grouped ? std::span<const c::GgufQuantizedType>(kGroupedTypes)
                 : std::span<const c::GgufQuantizedType>(kTypes);
}

inline ggml_type native_type(c::GgufQuantizedType type) {
  switch (type) {
  case c::GgufQuantizedType::kQ4_0:
    return GGML_TYPE_Q4_0;
  case c::GgufQuantizedType::kQ4_1:
    return GGML_TYPE_Q4_1;
  case c::GgufQuantizedType::kQ2K:
    return GGML_TYPE_Q2_K;
  case c::GgufQuantizedType::kIq2Xxs:
    return GGML_TYPE_IQ2_XXS;
  case c::GgufQuantizedType::kIq2Xs:
    return GGML_TYPE_IQ2_XS;
  case c::GgufQuantizedType::kIq2S:
    return GGML_TYPE_IQ2_S;
  case c::GgufQuantizedType::kIq1S:
    return GGML_TYPE_IQ1_S;
  case c::GgufQuantizedType::kIq1M:
    return GGML_TYPE_IQ1_M;
  case c::GgufQuantizedType::kQ3K:
    return GGML_TYPE_Q3_K;
  case c::GgufQuantizedType::kQ4K:
    return GGML_TYPE_Q4_K;
  case c::GgufQuantizedType::kQ5K:
    return GGML_TYPE_Q5_K;
  case c::GgufQuantizedType::kQ6K:
    return GGML_TYPE_Q6_K;
  case c::GgufQuantizedType::kQ8_0:
    return GGML_TYPE_Q8_0;
  case c::GgufQuantizedType::kIq3Xxs:
    return GGML_TYPE_IQ3_XXS;
  case c::GgufQuantizedType::kIq3S:
    return GGML_TYPE_IQ3_S;
  case c::GgufQuantizedType::kIq4Nl:
    return GGML_TYPE_IQ4_NL;
  case c::GgufQuantizedType::kIq4Xs:
    return GGML_TYPE_IQ4_XS;
  case c::GgufQuantizedType::kMxfp4:
    return GGML_TYPE_MXFP4;
  }
  return GGML_TYPE_COUNT;
}

struct Packed final {
  std::vector<std::byte> bytes;
  std::vector<float> decoded;
};

inline Packed pack(c::GgufQuantizedType type, std::span<const float> values,
                   int rows, int width, std::size_t bytes) {
  const auto native = native_type(type);
  Packed result{std::vector<std::byte>(bytes),
                std::vector<float>(static_cast<std::size_t>(rows * width))};
  std::vector<float> importance(width, 1.0F);
  const auto written = ggml_quantize_chunk(
      native, values.data(), result.bytes.data(), 0, rows, width,
      ggml_quantize_requires_imatrix(native) ? importance.data() : nullptr);
  EXPECT_EQ(written, bytes);
  const auto *traits = ggml_get_type_traits(native);
  EXPECT_NE(traits, nullptr);
  EXPECT_NE(traits->to_float, nullptr);
  if (traits == nullptr || traits->to_float == nullptr)
    return result;
  const auto row_bytes = ggml_row_size(native, width);
  for (int row = 0; row < rows; ++row)
    traits->to_float(result.bytes.data() + row * row_bytes,
                     result.decoded.data() + row * width, width);
  return result;
}

inline std::vector<float> weights(int rows, int width, int seed = 0) {
  std::vector<float> result(static_cast<std::size_t>(rows * width));
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] =
        0.25F * std::sin(0.031F * static_cast<float>(i + 1 + seed * 17));
  return result;
}

inline std::vector<double> activations(int rows, int width, int replay = 1) {
  std::vector<double> result(static_cast<std::size_t>(rows * width));
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = replay * 0.2 * std::cos(0.017 * static_cast<double>(i + 3));
  return result;
}

inline std::vector<double> matmul(std::span<const double> activation,
                                  std::span<const float> decoded, int rows,
                                  int columns, int width) {
  std::vector<double> result(static_cast<std::size_t>(rows * columns));
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < columns; ++column) {
      double sum = 0;
      for (int inner = 0; inner < width; ++inner)
        sum +=
            activation[row * width + inner] * decoded[column * width + inner];
      result[row * columns + column] = sum;
    }
  return result;
}

inline std::vector<double> grouped_matmul(std::span<const double> activation,
                                          std::span<const float> decoded,
                                          std::span<const std::int32_t> ids,
                                          int columns, int width) {
  std::vector<double> result(ids.size() * static_cast<std::size_t>(columns));
  for (std::size_t token = 0; token < ids.size(); ++token)
    for (int column = 0; column < columns; ++column) {
      double sum = 0;
      const auto weight_row = (ids[token] * columns + column) * width;
      for (int inner = 0; inner < width; ++inner)
        sum += activation[token * width + inner] * decoded[weight_row + inner];
      result[token * columns + column] = sum;
    }
  return result;
}

inline void expect_close(const v::Context &context, const v::Tensor &actual,
                         std::span<const double> expected) {
  const auto observed = vulkan_tensor_test::values(context, actual);
  ASSERT_EQ(observed.size(), expected.size());
  for (std::size_t i = 0; i < observed.size(); ++i)
    EXPECT_NEAR(observed[i], expected[i],
                3.0e-3 + 3.0e-4 * std::abs(expected[i]))
        << i;
}

// The original Vulkan MMVQ program also rounds Q8_1 activation d/s to F16.
// Fix that program reference from its cold device facts, before observing
// any result; the nominal decoded-weight model accuracy remains independent.
inline bool iq1s_mmvq_reference(const v::Context &context) {
  if (context.environment().vendor_id != 0x10de) return false;
  const auto &native = v::detail::Access::runtime(context)->device->provider.native();
  auto get = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
      native.get_instance_proc_addr(native.instance, "vkGetPhysicalDeviceFeatures2"));
  if (!get) return false;
  VkPhysicalDeviceShaderIntegerDotProductFeatures dot{};
  dot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
  VkPhysicalDeviceFeatures2 features{}; features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features.pNext = &dot; get(native.physical, &features);
  return dot.shaderIntegerDotProduct == VK_TRUE;
}

inline std::vector<double> iq1s_q8_program(std::span<const double> activation,
    const Packed &packed, int rows, int columns, int width,
    std::span<const std::int32_t> experts = {}) {
  std::vector<float> input(activation.begin(), activation.end());
  std::vector<std::byte> encoded(rows * (width / 32) * 36);
  ggml_get_type_traits(GGML_TYPE_Q8_1)->from_float_ref(input.data(), encoded.data(), rows * width);
  auto half = [](const std::byte *p) {
    ggml_fp16_t value; std::memcpy(&value, p, sizeof(value)); return ggml_fp16_to_fp32(value);
  };
  std::vector<double> result(rows * columns);
  for (int token = 0; token < rows; ++token) for (int row = 0; row < columns; ++row) {
    const auto weight_row = (experts.empty() ? 0 : experts[token]) * columns + row;
    double total = 0;
    for (int group = 0; group < width / 32; ++group) {
      const auto *a = packed.bytes.data() + (weight_row * (width / 256) + group / 8) * 50;
      std::uint16_t high; std::memcpy(&high, a + 34 + (group % 8) * 2, sizeof(high));
      const float dl = half(a) * (2 * ((high >> 12) & 7) + 1);
      if (dl == 0) continue;
      const float delta = (high & 0x8000) ? -.125F : .125F;
      const auto *b = encoded.data() + (token * (width / 32) + group) * 36;
      int dot = 0;
      for (int i = 0; i < 32; ++i) {
        const auto pos = weight_row * width + group * 32 + i;
        const int grid = int(std::round(packed.decoded[pos] / dl - delta));
        std::int8_t value; std::memcpy(&value, b + 4 + i, sizeof(value));
        dot += (grid + 1) * value;
      }
      total += dl * (half(b) * dot + (delta - 1) * half(b + 2));
    }
    result[token * columns + row] = total;
  }
  return result;
}

inline void expect_native_projection(const v::Context &context, const v::Tensor &actual,
    c::GgufQuantizedType type, std::span<const double> activation, const Packed &packed,
    int rows, int columns, int width, std::span<const std::int32_t> experts = {}) {
  const auto nominal = experts.empty() ? matmul(activation, packed.decoded, rows, columns, width)
      : grouped_matmul(activation, packed.decoded, experts, columns, width);
  if (type != c::GgufQuantizedType::kIq1S || !iq1s_mmvq_reference(context)) {
    expect_close(context, actual, nominal); return;
  }
  const auto native = iq1s_q8_program(activation, packed, rows, columns, width, experts);
  const auto observed = vulkan_tensor_test::values(context, actual);
  ASSERT_EQ(observed.size(), native.size());
  std::vector<float> normalized_actual, normalized_nominal;
  double peak = 0;
  for (auto value : nominal) peak = std::max(peak, std::abs(value));
  ASSERT_GT(peak, 0);
  for (std::size_t i = 0; i < observed.size(); ++i) {
    EXPECT_NEAR(observed[i], native[i], 1e-5 + 2e-6 * std::abs(native[i])) << i;
    normalized_actual.push_back(float(observed[i] / peak));
    normalized_nominal.push_back(float(nominal[i] / peak));
  }
  // Established common GGUF model gate, normalized so zero cannot hide.
  d5_gguf_test::expect_quantized_chain_matmul_close(normalized_actual, normalized_nominal);
}

struct QuantizerCleanup final {
  ~QuantizerCleanup() { ggml_quantize_free(); }
};

} // namespace vulkan_gguf_test
