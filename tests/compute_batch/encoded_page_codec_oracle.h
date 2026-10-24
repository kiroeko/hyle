#pragma once

#include "../compute_provider_native/half_test_support.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace encoded_page_test {

enum class Codec { kQ8, kE4M3 };

// Independent scalar/byte oracle. The K/V scales deliberately differ so that
// a paired producer must keep both the output order and each role's codec.
struct CodecOracle final {
  Codec codec;

  [[nodiscard]] const char *name() const { return codec == Codec::kQ8 ? "Q8" : "E4"; }

  [[nodiscard]] static double e4_value(unsigned code) {
    const auto exponent = (code & 127u) >> 3, mantissa = code & 7u;
    const auto magnitude = exponent
        ? std::ldexp(1. + double(mantissa) / 8., int(exponent) - 7)
        : std::ldexp(double(mantissa) / 8., -6);
    return (code & 128u) ? -magnitude : magnitude;
  }

  [[nodiscard]] std::vector<std::uint8_t> bytes(
      float value, std::size_t elements, bool is_value = false) const {
    if (codec == Codec::kE4M3) {
      const float target = value / (is_value ? 2.F : .5F);
      unsigned selected = 0;
      double error = std::numeric_limits<double>::infinity();
      for (unsigned code = 0; code <= 126; ++code) {
        const auto delta = std::abs(std::abs(target) - e4_value(code));
        if (delta < error || (delta == error && !(code & 1u))) {
          error = delta;
          selected = code;
        }
      }
      const auto code = std::uint8_t(selected | (std::signbit(target) ? 128u : 0u));
      return std::vector<std::uint8_t>(elements, code);
    }
    const float scale = std::abs(value) / 127.F;
    const auto bits = half_test::encode<hyle::simd::float16_t>(scale).bits;
    const auto code = std::uint8_t(int(std::round(value * (scale ? 1.F / scale : 0.F))));
    std::vector<std::uint8_t> result(elements / 32 * 34);
    for (std::size_t block = 0; block < elements / 32; ++block) {
      result[block * 34] = std::uint8_t(bits);
      result[block * 34 + 1] = std::uint8_t(bits >> 8);
      std::fill_n(result.begin() + block * 34 + 2, 32, code);
    }
    return result;
  }

  [[nodiscard]] double decoded(float value, bool is_value = false) const {
    if (codec == Codec::kE4M3)
      return e4_value(bytes(value, 1, is_value)[0]) * (is_value ? 2. : .5);
    const float scale = std::abs(value) / 127.F;
    const auto code = std::round(value * (scale ? 1.F / scale : 0.F));
    return half_test::decode(half_test::encode<hyle::simd::float16_t>(scale)) * code;
  }
};

} // namespace encoded_page_test
