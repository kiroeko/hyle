#pragma once

#include <ggml-cpu.h>
#include <ggml-quants.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "hyle/compute/low_precision.h"

namespace cpu_expanded_gguf_fixture {
namespace c = hyle::compute;
struct Case { c::GgufQuantizedType format; ggml_type native; };
inline constexpr std::array cases{
    Case{c::GgufQuantizedType::kQ4_0, GGML_TYPE_Q4_0},
    Case{c::GgufQuantizedType::kQ4_1, GGML_TYPE_Q4_1},
    Case{c::GgufQuantizedType::kQ2K, GGML_TYPE_Q2_K},
    Case{c::GgufQuantizedType::kIq2Xxs, GGML_TYPE_IQ2_XXS},
    Case{c::GgufQuantizedType::kIq2Xs, GGML_TYPE_IQ2_XS},
    Case{c::GgufQuantizedType::kIq2S, GGML_TYPE_IQ2_S},
    Case{c::GgufQuantizedType::kIq1S, GGML_TYPE_IQ1_S},
    Case{c::GgufQuantizedType::kIq1M, GGML_TYPE_IQ1_M},
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS) || defined(HYLE_GGML_CUDA_IQ1_EXTENSIONS)
    Case{c::GgufQuantizedType::kIq1Xxxs, GGML_TYPE_IQ1_XXXS},
    Case{c::GgufQuantizedType::kIq1Xxs, GGML_TYPE_IQ1_XXS},
    Case{c::GgufQuantizedType::kIq1Xs, GGML_TYPE_IQ1_XS},
#endif
};
inline constexpr std::int64_t k = 256, n = 4, experts = 2;

template<class Block, class Initialize>
void block(std::byte *destination, Initialize initialize) {
  Block value{};
  initialize(value);
  std::memcpy(destination, &value, sizeof(value));
}

// Valid pre-encoded bytes, not a HYLE quantizer. Original block structs bind
// each field; original decoders/dot callbacks provide the numerical reference.
inline std::vector<std::byte> weights(Case item, std::int64_t rows) {
#if defined(HYLE_C5_GGUF_DOT_REFERENCE)
  ggml_quantize_init(item.native);
#else
  ggml_cpu_init();
#endif
  const auto width = ggml_blck_size(item.native);
  const auto bytes = ggml_type_size(item.native);
  std::vector<std::byte> result(static_cast<std::size_t>(rows) * ggml_row_size(item.native, k));
  for (std::int64_t row = 0; row < rows; ++row)
    for (std::int64_t column = 0; column < k / width; ++column) {
      auto *p = result.data() + static_cast<std::size_t>(row * (k / width) + column) * bytes;
      const auto scale = ggml_fp32_to_fp16(float(row + 1) / 4);
      switch (item.native) {
      case GGML_TYPE_Q4_0:
        block<block_q4_0>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x31); }); break;
      case GGML_TYPE_Q4_1:
        block<block_q4_1>(p, [&](auto &b) { b.d = scale; b.m = ggml_fp32_to_fp16(-0.25f); std::fill(std::begin(b.qs), std::end(b.qs), 0x65); }); break;
      case GGML_TYPE_Q2_K:
        block<block_q2_K>(p, [&](auto &b) { b.d = scale; b.dmin = ggml_fp32_to_fp16(0.125f); std::fill(std::begin(b.scales), std::end(b.scales), 0x21); std::fill(std::begin(b.qs), std::end(b.qs), 0xe4); }); break;
      case GGML_TYPE_IQ2_XXS:
        block<block_iq2_xxs>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x0101); }); break;
      case GGML_TYPE_IQ2_XS:
        block<block_iq2_xs>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x0101); std::fill(std::begin(b.scales), std::end(b.scales), 0x21); }); break;
      case GGML_TYPE_IQ2_S:
        block<block_iq2_s>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x12); std::fill(std::begin(b.scales), std::end(b.scales), 0x21); }); break;
      case GGML_TYPE_IQ1_S:
        block<block_iq1_s>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x12); }); break;
      case GGML_TYPE_IQ1_M:
        block<block_iq1_m>(p, [&](auto &b) {
          std::fill(std::begin(b.qs), std::end(b.qs), 0x12);
          const std::array<std::uint16_t, 4> sc{
              static_cast<std::uint16_t>((scale & 0x000f) << 12),
              static_cast<std::uint16_t>((scale & 0x00f0) << 8),
              static_cast<std::uint16_t>((scale & 0x0f00) << 4),
              static_cast<std::uint16_t>(scale & 0xf000)};
          std::memcpy(b.scales, sc.data(), sizeof(sc));
        }); break;
#if defined(HYLE_GGML_CPU_IQ1_EXTENSIONS) || defined(HYLE_GGML_CUDA_IQ1_EXTENSIONS)
      case GGML_TYPE_IQ1_XXXS:
        block<block_iq1_xxxs>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x12); }); break;
      case GGML_TYPE_IQ1_XXS:
        block<block_iq1_xxs>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x12); }); break;
      case GGML_TYPE_IQ1_XS:
        block<block_iq1_xs>(p, [&](auto &b) { b.d = scale; std::fill(std::begin(b.qs), std::end(b.qs), 0x12); }); break;
#endif
      default: throw std::invalid_argument("unlisted CPU expanded format");
      }
    }
  return result;
}

inline std::vector<float> decode(Case item, std::span<const std::byte> packed) {
  const auto rows = packed.size() / ggml_row_size(item.native, k);
  std::vector<float> result(rows * static_cast<std::size_t>(k));
  ggml_get_type_traits(item.native)->to_float(packed.data(), result.data(),
                                             static_cast<std::int64_t>(result.size()));
  return result;
}

inline float dot(Case item, const std::byte *row, std::span<const float> activation) {
  const auto *weight = ggml_get_type_traits_cpu(item.native);
  const auto dtype = weight->vec_dot_type;
  const auto bytes = ggml_row_size(dtype, k);
  std::vector<std::uint64_t> encoded((bytes + 7) / 8);
  ggml_get_type_traits_cpu(dtype)->from_float(activation.data(), encoded.data(), k);
  float result = 0;
  weight->vec_dot(static_cast<int>(k), &result, 0, row, 0, encoded.data(), 0, 1);
  return result;
}
} // namespace cpu_expanded_gguf_fixture
