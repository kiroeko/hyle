#pragma once

#include "d5_dynamic_fp8_fragments.h"

namespace d5_fp8_weight_preparation {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;

// Independent E4M3FN definition, including signed zero and the two NaN codes.
inline float decode(std::uint8_t code) {
  const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
  if (exponent == 15 && mantissa == 7)
    return std::numeric_limits<float>::quiet_NaN();
  const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -9)
      : std::ldexp(1.0F + float(mantissa) / 8, int(exponent) - 7);
  return std::copysign(magnitude, (code & 128) ? -1.0F : 1.0F);
}

inline sem::Fragment decode_weights(std::int64_t n, std::int64_t k, bool bf16) {
  sem::Fragment f{.profile = d5_official_fp8_fragments::kGlmTailCase.profile,
                  .region = 0x0e0604};
  const auto input = c::float8_weight_description(k, n, c::Float8ScaleMode::kFloat32Block128).value();
  const auto w = f.input(input);
  auto result = f.one(c::model_ops::float8_weight_decode(
      bf16 ? c::DataType::kBFloat16 : c::DataType::kFloat32), {w});
  f.outputs = {d5_dynamic_fp8_fragments::visible(f, result, bf16)};
  f.exact_outputs = {0};
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<std::byte> storage(input.required_bytes().value());
    std::vector<float> expected(n * k);
    const auto groups = k / 128;
    std::vector<float> scales(((n + 127) / 128) * groups);
    for (std::size_t block = 0; block < scales.size(); ++block)
      scales[block] = float(int((block * 3 + replay) % 13) - 6) / 1024.0F;
    std::memcpy(storage.data() + n * k, scales.data(), scales.size() * sizeof(float));
    for (std::int64_t row = 0; row < n; ++row)
      for (std::int64_t col = 0; col < k; ++col) {
        const auto i = row * k + col;
        auto code = std::uint8_t((i * 29 + replay * 37) % 256);
        // NaN handling is checked directly at the Provider boundary; the
        // ordinary model-fragment contract requires finite outputs.
        if ((code & 127) == 127) --code;
        storage[i] = std::byte{code};
        float value = decode(code) * scales[(row / 128) * groups + col / 128];
        expected[i] = bf16 ? d5_dynamic_fp8_fragments::bfloat(value) : value;
      }
    f.replays.push_back({{std::move(storage)}, {std::move(expected)}});
  }
  return f;
}
} // namespace d5_fp8_weight_preparation
