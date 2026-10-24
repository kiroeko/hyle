#pragma once

#include "../compute_provider_native/half_test_support.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ct_packed_test {
using Type = hyle::compute::DataType;
using F16 = hyle::simd::float16_t;
using BF16 = hyle::simd::bfloat16_t;

// The software half encoder searches adjacent representable numbers. Neither
// this oracle nor the bit writer calls a product CT decoder/packed loader.
inline double round(Type type, double x) {
  if (type == Type::kFloat16) return half_test::decode(half_test::encode<F16>(x));
  if (type == Type::kBFloat16) return half_test::decode(half_test::encode<BF16>(x));
  return static_cast<float>(x);
}

struct Fixture final {
  std::int64_t rows, depth, group = 32;
  bool asymmetric;
  std::vector<std::int32_t> codes, zeros;
  std::vector<int> logical_codes, logical_zeros;
  std::vector<double> scales;

  Fixture(bool asym, unsigned seed = 0, std::int64_t n = 9, std::int64_t k = 64,
          std::int64_t group_width = 32)
      : rows(n), depth(k), group(group_width), asymmetric(asym),
        codes(static_cast<std::size_t>(n * k / 8)),
        zeros(static_cast<std::size_t>(((n + 7) / 8) * (k / group)),
              std::bit_cast<std::int32_t>(std::uint32_t{0xdeadbeef})),
        logical_codes(static_cast<std::size_t>(n * k)),
        logical_zeros(static_cast<std::size_t>(n * (k / group))),
        scales(logical_zeros.size()) {
    const auto put = [](std::int32_t &word, unsigned lane, unsigned value) {
      auto bits = std::bit_cast<std::uint32_t>(word);
      for (unsigned b = 0; b < 4; ++b) {
        const auto mask = std::uint32_t{1} << (lane * 4 + b);
        bits = (bits & ~mask) | (((value >> b) & 1U) ? mask : 0U);
      }
      word = std::bit_cast<std::int32_t>(bits);
    };
    constexpr std::array scale_values{0.10009765625, 0.0999755859375,
        -0.10009765625, 0.5, 0.03125, -0.0, 0.0, 1.00390625};
    for (std::int64_t n0 = 0; n0 < rows; ++n0) {
      for (std::int64_t g = 0; g < depth / group; ++g) {
        const auto i = static_cast<std::size_t>(n0 * (depth / group) + g);
        int z = asymmetric ? int((n0 * 7 + g * 3 + seed + 2) % 16) - 8 : 0;
        if (asymmetric && n0 == 0) z = 2;
        logical_zeros[i] = z;
        scales[i] = scale_values[(i + seed) % scale_values.size()];
        put(zeros[static_cast<std::size_t>((n0 / 8) * (depth / group) + g)],
            unsigned(n0 % 8), unsigned(z + 8));
      }
      for (std::int64_t k0 = 0; k0 < depth; ++k0) {
        int q = int((n0 * 13 + k0 * 5 + seed * 3) % 16) - 8;
        if (n0 == 0 && (k0 == 0 || k0 == 32)) q = asymmetric ? 5 : 3;
        logical_codes[static_cast<std::size_t>(n0 * depth + k0)] = q;
        put(codes[static_cast<std::size_t>(n0 * (depth / 8) + k0 / 8)],
            unsigned(k0 % 8), unsigned(q + 8));
      }
    }
    // Both source-round witnesses exist independently of the seed.
    scales[0] = scale_values[0];
    scales[1] = scale_values[1];
  }

  std::vector<double> decoded(Type source, Type decode, Type operand) const {
    std::vector<double> result(logical_codes.size());
    for (std::int64_t n = 0; n < rows; ++n)
      for (std::int64_t k = 0; k < depth; ++k) {
        const auto i = static_cast<std::size_t>(n * depth + k);
        const auto g = static_cast<std::size_t>(n * (depth / group) + k / group);
        const auto difference = round(source,
            round(source, logical_codes[i]) - round(source, logical_zeros[g]));
        result[i] = round(operand, round(decode,
            round(source, difference * round(source, scales[g]))));
      }
    return result;
  }

  std::vector<double> activation(std::int64_t m, Type type, unsigned seed = 0,
                                 bool witness = false) const {
    std::vector<double> result(static_cast<std::size_t>(m * depth));
    for (std::int64_t row = 0; row < m; ++row)
      for (std::int64_t k = 0; k < depth; ++k)
        result[static_cast<std::size_t>(row * depth + k)] = witness
            ? (k == (row % 2 ? 32 : 0) ? 1.0 : 0.0)
            : round(type, double(int((row * 7 + k * 3 + seed) % 11) - 5) / 16);
    return result;
  }
};
} // namespace ct_packed_test
