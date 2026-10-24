#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ct_asymmetric_test {

// Independent writer scatters individual bits. The scalar oracle uses the
// original logical signed values, never extracts the packed product inputs.
struct Fixture final {
  static constexpr std::int64_t rows = 9, columns = 64, group = 32;
  static constexpr std::int64_t groups = columns / group;
  std::vector<std::int32_t> codes, zero_points;
  std::vector<double> scales;
  std::vector<int> logical_codes, logical_zero_points;

  explicit Fixture(unsigned seed = 0)
      : scales(rows * groups), logical_codes(rows * columns),
        logical_zero_points(rows * groups) {
    std::vector<std::uint32_t> words(rows * columns / 8);
    // Padded output lanes contain real nonzero garbage, not a safe zero tail.
    std::vector<std::uint32_t> zeros(((rows + 7) / 8) * groups, 0xdeadbeefU);
    constexpr std::array<double, 8> source_scales{
        0.10009765625, 0.0999755859375, 0.5, -0.10009765625,
        1.00390625, -0.0, 0.0, 0.03125};
    const auto scatter = [](std::uint32_t &word, unsigned nibble, unsigned value) {
      for (unsigned bit = 0; bit < 4; ++bit) {
        const auto mask = std::uint32_t{1} << (4 * nibble + bit);
        word = (word & ~mask) | (((value >> bit) & 1U) ? mask : 0U);
      }
    };
    for (std::int64_t row = 0; row < rows; ++row) {
      for (std::int64_t g = 0; g < groups; ++g) {
        const auto index = static_cast<std::size_t>(row * groups + g);
        int zero = static_cast<int>((row * 7 + g * 3 + 2 + seed) % 16) - 8;
        if (row == 0) zero = 2;
        logical_zero_points[index] = zero;
        scales[index] = source_scales[(index + seed) % source_scales.size()];
        scatter(zeros[static_cast<std::size_t>((row / 8) * groups + g)],
                static_cast<unsigned>(row % 8), static_cast<unsigned>(zero + 8));
      }
      for (std::int64_t column = 0; column < columns; ++column) {
        const auto index = static_cast<std::size_t>(row * columns + column);
        int value = static_cast<int>((row * 13 + column * 5 + seed * 3) % 16) - 8;
        // BF16 and F16 source-rounding witnesses in different groups.
        if (row == 0 && (column == 0 || column == group)) value = 5;
        logical_codes[index] = value;
        scatter(words[index / 8], static_cast<unsigned>(column % 8),
                static_cast<unsigned>(value + 8));
      }
    }
    for (auto word : words) codes.push_back(std::bit_cast<std::int32_t>(word));
    for (auto word : zeros) zero_points.push_back(std::bit_cast<std::int32_t>(word));
  }

  // Caller supplies an independent codec for the source scale type. Optional
  // requested output conversion belongs after this typed dequantized result.
  template<class Round> std::vector<double> expected(Round round) const {
    std::vector<double> result(logical_codes.size());
    for (std::int64_t row = 0; row < rows; ++row)
      for (std::int64_t column = 0; column < columns; ++column) {
        const auto index = static_cast<std::size_t>(row * columns + column);
        const auto g = static_cast<std::size_t>(row * groups + column / group);
        const auto difference = round(round(logical_codes[index]) -
                                      round(logical_zero_points[g]));
        result[index] = round(difference * round(scales[g]));
      }
    return result;
  }
};

} // namespace ct_asymmetric_test
