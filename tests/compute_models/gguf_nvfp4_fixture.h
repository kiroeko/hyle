#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nvfp4_test {

// Definition-based writer and oracle: expected values use the original
// logical lane/code, never unpack the bytes consumed by the implementation.
struct Fixture final {
  std::int64_t rows, columns, experts;
  unsigned seed;
  std::vector<std::uint8_t> bytes;
  std::vector<float> decoded; // [expert,row,K]

  static float scale_value(std::uint8_t code) {
    if (code == 0 || code == 0x7f) return 0.0f;
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const float value = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
    return value * 0.5f;
  }

  explicit Fixture(std::int64_t n = 5, std::int64_t k = 128,
                   std::int64_t e = 1, unsigned seed_value = 0,
                   std::span<const std::uint8_t> scale_codes = {})
      : rows(n), columns(k), experts(e), seed(seed_value),
        bytes(static_cast<std::size_t>(e * n * (k / 64) * 36), 0),
        decoded(static_cast<std::size_t>(e * n * k)) {
    constexpr std::array<std::uint8_t, 8> normal_scales{
        0x38, 0x30, 0x40, 0x28, 0x01, 0x08, 0x00, 0x7f};
    constexpr std::array<int, 16> doubled_e2m1{
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    if (scale_codes.empty()) scale_codes = normal_scales;
    for (std::int64_t expert = 0; expert < e; ++expert)
      for (std::int64_t row = 0; row < n; ++row)
        for (std::int64_t block = 0; block < k / 64; ++block) {
          const auto packed = static_cast<std::size_t>(((expert * n + row) * (k / 64) + block) * 36);
          for (int subgroup = 0; subgroup < 4; ++subgroup) {
            const auto scale_index = static_cast<std::size_t>(expert * 11 + row * 3 + block * 5 + subgroup + seed);
            const auto scale = scale_codes[scale_index % scale_codes.size()];
            bytes[packed + static_cast<std::size_t>(subgroup)] = scale;
            for (int lane = 0; lane < 16; ++lane) {
              const auto code = static_cast<unsigned>(lane * 7 + subgroup * 3 + block * 11 + row * 5 + expert * 13 + seed) % 16;
              const auto byte = packed + 4 + static_cast<std::size_t>(subgroup * 8 + lane % 8);
              bytes[byte] |= static_cast<std::uint8_t>(code << (lane < 8 ? 0 : 4));
              const auto logical = static_cast<std::size_t>((expert * n + row) * k + block * 64 + subgroup * 16 + lane);
              decoded[logical] = static_cast<float>(doubled_e2m1[code]) * scale_value(scale);
            }
          }
        }
  }

  // Each 32-value activation block has one +/-127 and zeros. The original
  // Q8 activation quantization is exact, isolating the packed weight mapping.
  std::vector<float> activations(std::int64_t tokens) const {
    std::vector<float> result(static_cast<std::size_t>(tokens * columns), 0.0f);
    for (std::int64_t token = 0; token < tokens; ++token)
      for (std::int64_t block = 0; block < columns / 32; ++block) {
        const auto lane = (token * 5 + block * 3 + seed) % 32;
        result[static_cast<std::size_t>(token * columns + block * 32 + lane)] =
            (token + block) % 2 ? -127.0f : 127.0f;
      }
    return result;
  }
};

} // namespace nvfp4_test
