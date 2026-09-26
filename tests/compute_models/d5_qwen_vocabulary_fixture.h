#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "../compute/workload_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "d5_qwen38_profile_cases.h"

namespace d5_qwen_vocabulary_fixture {

using BF16 = hyle::simd::bfloat16_t;

struct ProfileCase final {
  const hyle::compute::ModelArtifactProfileDescription *profile;
  std::int64_t hidden;
};

inline constexpr ProfileCase kOfficial2B{&workload_test::kModel, 2048};
inline constexpr ProfileCase kOfficial27B{
    d5_qwen38_profile_cases::kOfficial27B.profile,
    d5_qwen38_profile_cases::kOfficial27B.hidden};
inline constexpr ProfileCase kOfficial24T{
    d5_qwen38_profile_cases::kOfficial24T.profile,
    d5_qwen38_profile_cases::kOfficial24T.hidden};

// All Qwen3.5-2B and Qwen3.8 target profiles have this logical vocabulary.
// A shard limits resident weight storage; every row must still be executed.
inline constexpr std::int64_t kVocabularyRows = 248320;
inline constexpr std::int64_t kShardRows = 16384;
inline constexpr std::array<std::int64_t, 6> kGatherIds{
    0, 16385, 131071, 248317, 248318, 248319};

struct Shard final {
  std::int64_t first;
  std::int64_t rows;
};

inline std::vector<Shard> shards(std::int64_t vocabulary_rows = kVocabularyRows) {
  std::vector<Shard> result;
  for (std::int64_t first = 0; first < vocabulary_rows;
       first += kShardRows)
    result.push_back(
        {first, (std::min)(kShardRows, vocabulary_rows - first)});
  return result;
}

inline std::optional<Shard> shard_for(
    std::int64_t global_row, std::int64_t vocabulary_rows = kVocabularyRows) {
  if (global_row < 0 || global_row >= vocabulary_rows)
    return std::nullopt;
  const auto first = (global_row / kShardRows) * kShardRows;
  return Shard{first, (std::min)(kShardRows, vocabulary_rows - first)};
}

inline BF16 encode(double value) { return half_test::encode<BF16>(value); }
inline double decode(BF16 value) { return half_test::decode(value); }

inline std::size_t source_column(std::int64_t row, std::int64_t hidden) {
  return static_cast<std::size_t>((row * 131 + 17) % hidden);
}

inline BF16 coefficient(std::int64_t row) {
  const auto magnitude = 1 + row % 7;
  const auto sign = row % 2 == 0 ? 1 : -1;
  return encode(static_cast<double>(sign * magnitude) / 16.0);
}

inline std::vector<BF16> activation(std::int64_t hidden, int seed) {
  std::vector<BF16> result(static_cast<std::size_t>(hidden));
  for (std::size_t column = 0; column < result.size(); ++column)
    result[column] = encode(
        static_cast<double>(static_cast<int>((column * 11 + seed * 19) % 47) -
                            23) /
        64.0);
  return result;
}

// Output-major BF16 payload with one nonzero per full-width row. The backend
// still executes its normal dense MatMul, while the independent oracle is O(V).
inline std::vector<BF16> head_weight(Shard shard, std::int64_t hidden) {
  std::vector<BF16> result(
      static_cast<std::size_t>(shard.rows * hidden), encode(0.0));
  for (std::int64_t local = 0; local < shard.rows; ++local) {
    const auto row = shard.first + local;
    result[static_cast<std::size_t>(local * hidden) +
           source_column(row, hidden)] = coefficient(row);
  }
  return result;
}

inline std::vector<BF16> expected_logits(std::span<const BF16> activation,
                                         Shard shard) {
  std::vector<BF16> result(static_cast<std::size_t>(shard.rows));
  const auto hidden = static_cast<std::int64_t>(activation.size());
  for (std::int64_t local = 0; local < shard.rows; ++local) {
    const auto row = shard.first + local;
    result[static_cast<std::size_t>(local)] = encode(
        decode(activation[source_column(row, hidden)]) *
        decode(coefficient(row)));
  }
  return result;
}

inline std::vector<BF16> expected_embedding(std::int64_t global_row,
                                            std::int64_t hidden) {
  std::vector<BF16> result(static_cast<std::size_t>(hidden), encode(0.0));
  result[source_column(global_row, hidden)] = coefficient(global_row);
  return result;
}

} // namespace d5_qwen_vocabulary_fixture
