#pragma once

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4201) // Original ggml block ABI uses anonymous unions.
#endif
#include <ggml-quants.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

#include "hyle/compute/low_precision.h"

namespace cuda_iq1_fixture {
namespace c = hyle::compute;
struct Format { c::GgufQuantizedType type; ggml_type native; };
inline constexpr std::array formats{
    Format{c::GgufQuantizedType::kIq1Xs, GGML_TYPE_IQ1_XS},
    Format{c::GgufQuantizedType::kIq1Xxs, GGML_TYPE_IQ1_XXS},
    Format{c::GgufQuantizedType::kIq1Xxxs, GGML_TYPE_IQ1_XXXS}};

// Legal original block bytes, not a HYLE encoder. Original CPU dequantizers
// provide the oracle; no CPU execution backend is linked or used by the route.
template<class Block> std::vector<std::byte> blocks(std::int64_t rows, std::int64_t k) {
  std::vector<std::byte> result(rows * (k / 256) * sizeof(Block));
  for (std::int64_t row = 0; row < rows; ++row)
    for (std::int64_t group = 0; group < k / 256; ++group) {
      Block block{};
      block.d = ggml_fp32_to_fp16(float(1 + row % 3) / 256.0F);
      std::fill(std::begin(block.qs), std::end(block.qs),
                static_cast<unsigned char>(0x12 + (row + group) % 5));
      std::memcpy(result.data() + (row * (k / 256) + group) * sizeof(Block),
                  &block, sizeof(block));
    }
  return result;
}
inline std::vector<std::byte> weights(Format format, std::int64_t rows, std::int64_t k) {
  switch (format.native) {
  case GGML_TYPE_IQ1_XS: return blocks<block_iq1_xs>(rows, k);
  case GGML_TYPE_IQ1_XXS: return blocks<block_iq1_xxs>(rows, k);
  default: return blocks<block_iq1_xxxs>(rows, k);
  }
}
inline std::vector<float> sums(Format format, std::span<const std::byte> packed,
                               std::int64_t rows, std::int64_t k) {
  std::vector<float> decoded(k), result(rows);
  const auto stride = ggml_row_size(format.native, k);
  for (std::int64_t row = 0; row < rows; ++row) {
    ggml_get_type_traits(format.native)->to_float(packed.data() + row * stride,
                                                  decoded.data(), k);
    for (auto value : decoded) result[row] += value * 0.25F;
  }
  return result;
}

// Original CUDA IQ1 dot: d1q * (ds.x * sumi + ds.y * delta).
// Our legal blocks have zero scale/sign nibbles, hence delta = -1 + 0.125.
// q8_1 rounds d and the original (unquantized) 32-value sum independently.
inline float original_cuda_uniform_dot(float decoded_quarter_sum,
                                        std::int64_t row, std::int64_t k,
                                        float activation) {
  const float weight_scale = float(1 + row % 3) / 256.0F;
  const float quantized_value = ggml_fp16_to_fp32(
      ggml_fp32_to_fp16(activation / 127.0F)) * 127.0F;
  const float partial_sum = ggml_fp16_to_fp32(ggml_fp32_to_fp16(activation * 32));
  return decoded_quarter_sum * (quantized_value / 0.25F) +
      weight_scale * (-1.0F + 0.125F) * (partial_sum - quantized_value * 32) *
      float(k / 32);
}
} // namespace cuda_iq1_fixture
