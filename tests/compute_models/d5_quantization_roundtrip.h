#pragma once

#include "d5_fp8_weight_preparation.h"
#include "d5_moe_execution_fragments.h"

namespace d5_quantization_roundtrip {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
using Format = c::model_ops::BlockQuantizationFormat;

// Definition-based reference: a sorted format table and nearest-even distances,
// independent of the original TileLang casts and bit-manipulated scale formula.
inline std::vector<float> reference(const std::vector<float> &x, bool fp4,
                                     std::size_t fp8_group = 64) {
  const std::size_t group = fp4 ? 32 : fp8_group;
  const float maximum = fp4 ? 6.0F : 448.0F;
  const float floor = fp4 ? 6.0F * std::ldexp(1.0F, -126) : 1.0e-4F;
  std::vector<float> table;
  if (fp4) table = {0, .5F, 1, 1.5F, 2, 3, 4, 6};
  else for (unsigned i = 0; i < 127; ++i)
    table.push_back(d5_fp8_weight_preparation::decode(static_cast<std::uint8_t>(i)));
  std::vector<float> result(x.size());
  for (std::size_t base = 0; base < x.size(); base += group) {
    float maximum_input = floor;
    for (std::size_t i = 0; i < group; ++i)
      maximum_input = std::max(maximum_input, std::abs(x[base + i]));
    const float scaled = maximum_input * (1.0F / maximum);
    const float scale = static_cast<float>(std::exp2(std::ceil(std::log2(double(scaled)))));
    for (std::size_t i = 0; i < group; ++i) {
      const float value = x[base + i] / scale;
      const float magnitude = std::min(std::abs(value), maximum);
      const auto upper = std::lower_bound(table.begin(), table.end(), magnitude);
      const auto hi = std::size_t(upper - table.begin());
      const auto lo = hi == 0 ? 0 : hi - 1;
      const float dl = magnitude - table[lo], du = table[hi] - magnitude;
      const auto chosen = dl < du || (dl == du && lo % 2 == 0) ? lo : hi;
      result[base + i] = d5_dynamic_fp8_fragments::bfloat(std::copysign(table[chosen], value) * scale);
    }
  }
  return result;
}

inline sem::Fragment fragment(bool fp4, std::int64_t rows, std::int64_t width,
                               std::size_t fp8_group = 64) {
  sem::Fragment f{.profile = d5_moe_execution_fragments::cases::kCases[3].profile,
                  .region = 0x070908};
  const auto input = f.input(sem::dense({rows, width}, c::DataType::kBFloat16));
  const auto output = f.one(c::model_ops::block_quantization_roundtrip(
      fp4 ? Format::kFloat4E2M1Block32 : (fp8_group == 128 ? Format::kFloat8E4M3Block128
                                                        : Format::kFloat8E4M3Block64)), {input});
  f.outputs = {d5_dynamic_fp8_fragments::visible(f, output, true)};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<sem::q::BF16> encoded;
    std::vector<float> x;
    for (std::int64_t i = 0; i < rows * width; ++i) {
      const float value = std::sin(float(i + replay * 7) * .13F) * float(1 + i % 19) / 7;
      encoded.push_back(sem::q::encode(value));
      x.push_back(float(sem::q::decode(encoded.back())));
    }
    f.replays.push_back({{sem::bytes(encoded)}, {reference(x, fp4, fp8_group)}});
  }
  return f;
}
} // namespace d5_quantization_roundtrip
