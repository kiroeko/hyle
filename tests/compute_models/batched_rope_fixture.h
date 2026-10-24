#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/compute/operation.h"

namespace batched_rope_test {
// B=2, S=3, H=2, D=8, rotary D=4. Other depth elements stay untouched.
inline constexpr std::size_t kElements = 96;
inline constexpr std::array<std::int32_t, 6> kPositions{0, 3, 19, 31, 7, 2};
inline constexpr std::array<float, 6> kFrequencies{0.5F, 0.25F, 0.125F, 0.75F, 0.25F, 0.5F};
inline std::array<double, kElements> data(int seed) {
  std::array<double, kElements> values{};
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = (static_cast<int>((i * 7 + seed) % 29) - 14) / 8.0;
  return values;
}
inline std::vector<double> reference(std::span<const double> input,
    std::span<const std::int32_t> positions, bool per_request,
    bool per_token_frequency, hyle::compute::RopeLayout layout) {
  std::vector<double> result(input.begin(), input.end());
  for (std::size_t b = 0; b < 2; ++b)
    for (std::size_t s = 0; s < 3; ++s)
      for (std::size_t h = 0; h < 2; ++h)
        for (std::size_t p = 0; p < 2; ++p) {
          const auto first = ((b * 3 + s) * 2 + h) * 8 +
              (layout == hyle::compute::RopeLayout::kInterleaved ? 2 * p : p);
          const auto second = first +
              (layout == hyle::compute::RopeLayout::kInterleaved ? 1 : 2);
          const float phase = static_cast<float>(positions[(per_request ? b * 3 : 0) + s]) *
              kFrequencies[(per_token_frequency ? s * 2 : 0) + p];
          const double cosine = std::cos(static_cast<double>(phase));
          const double sine = std::sin(static_cast<double>(phase));
          result[first] = input[first] * cosine - input[second] * sine;
          result[second] = input[first] * sine + input[second] * cosine;
        }
  return result;
}
} // namespace batched_rope_test
