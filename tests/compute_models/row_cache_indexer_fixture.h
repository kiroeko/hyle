#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "row_cache_fixture.h"

namespace row_cache_indexer_test {
namespace c = hyle::compute;

// Independent E2M1 table and BF16 nearest-even oracle. Inputs are the actual
// post-Hadamard/QAT boundary, not a replacement for the model projections.
struct Fixture final {
  std::int64_t queries, rows;
  std::array<c::TensorDesc, 4> descriptions;
  std::vector<std::uint16_t> query, weights;
  std::vector<std::uint8_t> bank;
  std::vector<std::int32_t> ranges;

  explicit Fixture(std::int64_t length = 513, std::int64_t tokens = 3)
      : queries(tokens), rows(length), descriptions{
          c::TensorDesc::dense(c::DataType::kBFloat16,
              std::array{tokens, std::int64_t{64}, std::int64_t{128}}).value(),
          c::row_cache_carrier_description(c::RowCacheEncoding::float4_block32(), length).value(),
          c::TensorDesc::dense(c::DataType::kBFloat16,
              std::array{tokens, std::int64_t{64}}).value(),
          c::TensorDesc::dense(c::DataType::kInt32,
              std::array{tokens, std::int64_t{2}}).value()},
        query(std::size_t(tokens) * 64 * 128), weights(std::size_t(tokens) * 64),
        bank(std::size_t(length) * 68), ranges(std::size_t(tokens) * 2) {
    update(0);
  }

  static double code_value(unsigned code) {
    constexpr std::array<double, 8> table{0, .5, 1, 1.5, 2, 3, 4, 6};
    return (code & 8 ? -1 : 1) * table[code & 7];
  }
  double key(std::int64_t row, std::int64_t channel) const {
    const auto packed = bank[std::size_t(row) * 68 + std::size_t(channel) / 2];
    const unsigned code = (packed >> ((channel & 1) * 4)) & 15;
    const auto exponent = bank[std::size_t(row) * 68 + 64 + std::size_t(channel) / 32];
    const auto scale = exponent == 255 ? std::numeric_limits<double>::quiet_NaN()
                                      : std::ldexp(1.0, int(exponent) - 127);
    return row_cache_test::decode(row_cache_test::bfloat(float(code_value(code) * scale)));
  }
  void update(int phase) {
    std::fill(bank.begin(), bank.end(), std::uint8_t{0});
    for (std::int64_t row = 0; row < rows; ++row) {
      for (std::int64_t d = 0; d < 128; ++d) {
        const auto code = unsigned((row * 11 + d * 3 + phase * 5) % 16);
        bank[std::size_t(row) * 68 + std::size_t(d) / 2] |=
            std::uint8_t(code << ((d & 1) * 4));
      }
      for (int group = 0; group < 4; ++group)
        bank[std::size_t(row) * 68 + 64 + std::size_t(group)] =
            std::uint8_t(125 + (row + group + phase) % 3);
    }
    for (std::int64_t t = 0; t < queries; ++t) {
      for (std::int64_t head = 0; head < 64; ++head) {
        for (std::int64_t d = 0; d < 128; ++d) {
          const auto code = unsigned((t * 7 + head * 5 + d * 3 + phase * 11) % 16);
          query[std::size_t((t * 64 + head) * 128 + d)] = row_cache_test::bfloat(
              code_value(code) * std::ldexp(1.0, -4 + int((head + d / 32 + phase) % 2)));
        }
        weights[std::size_t(t * 64 + head)] = row_cache_test::bfloat(
            double((head * 7 + t * 3 + phase * 5) % 17 - 8) / 512.0);
      }
      const auto begin = rows == 0 ? 0 : (t == 1 ? (std::min)(std::int64_t{3}, rows) : 0);
      const auto end = t == 2 ? begin : (phase && rows > begin ? rows - 1 : rows);
      ranges[std::size_t(t) * 2] = std::int32_t(begin);
      ranges[std::size_t(t) * 2 + 1] = std::int32_t(end);
    }
  }

  std::vector<float> expected(bool bfloat_boundaries = true) const {
    std::vector<float> result(std::size_t(queries * rows),
                              -std::numeric_limits<float>::infinity());
    const auto round = [=](double x) {
      return bfloat_boundaries ? row_cache_test::decode(row_cache_test::bfloat(float(x)))
                               : double(float(x));
    };
    for (std::int64_t t = 0; t < queries; ++t)
      for (std::int64_t row = ranges[std::size_t(t) * 2];
           row < ranges[std::size_t(t) * 2 + 1]; ++row) {
        double score = 0;
        for (std::int64_t head = 0; head < 64; ++head) {
          double dot = 0;
          for (std::int64_t d = 0; d < 128; ++d)
            dot += row_cache_test::decode(query[std::size_t((t * 64 + head) * 128 + d)]) * key(row, d);
          const auto positive = (std::max)(round(dot), 0.0);
          score += round(positive * row_cache_test::decode(weights[std::size_t(t * 64 + head)]));
        }
        result[std::size_t(t * rows + row)] = float(round(score));
      }
    return result;
  }
};
} // namespace row_cache_indexer_test
