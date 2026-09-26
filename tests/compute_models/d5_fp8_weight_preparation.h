#pragma once

#include "d5_dynamic_fp8_fragments.h"

namespace d5_fp8_weight_preparation {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;

template <class Buffers>
void check_constants(const sem::Fragment &cold, const Buffers &buffers) {
  ASSERT_EQ(buffers.size(), cold.outputs.size());
  ASSERT_EQ(cold.replays.size(), 1u);
  for (std::size_t port = 0; port < buffers.size(); ++port) {
    ASSERT_EQ(cold.description(cold.outputs[port]).data_type(), c::DataType::kBFloat16);
    std::vector<sem::q::BF16> actual(buffers[port].size() / 2);
    ASSERT_TRUE(buffers[port].view().read(std::as_writable_bytes(std::span(actual))).ok());
    const auto &expected = cold.replays[0].outputs[port];
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      ASSERT_EQ(float(sem::q::decode(actual[i])), expected[i]) << "cache " << port << " at " << i;
  }
}

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
