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
} // namespace cuda_iq1_fixture
