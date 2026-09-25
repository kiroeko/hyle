#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "d5_qwen_vocabulary_fixture.h"

namespace d5_qwen_vocabulary_test {

namespace fixture = d5_qwen_vocabulary_fixture;

// The route supplies its native prepare, weight upload and execution. The
// traversal, full logical output and independent oracle stay identical.
template <class Prepare, class UploadWeight, class Execute>
void check_head(std::int64_t hidden, Prepare prepare,
                UploadWeight upload_weight, Execute execute) {
  auto full = prepare(fixture::kShardRows);
  auto tail = prepare(fixture::kVocabularyRows % fixture::kShardRows);
  const std::array activations{fixture::activation(hidden, 3),
                               fixture::activation(hidden, 17)};
  std::array<std::vector<double>, 2> logical_logits{
      std::vector<double>(fixture::kVocabularyRows),
      std::vector<double>(fixture::kVocabularyRows)};
  std::int64_t visited = 0;
  for (const auto shard : fixture::shards()) {
    const auto values = fixture::head_weight(shard, hidden);
    auto native_weight = upload_weight(shard, values);
    auto &call = shard.rows == fixture::kShardRows ? full : tail;
    for (std::size_t replay = 0; replay < activations.size(); ++replay) {
      const auto actual = execute(call, activations[replay], native_weight);
      ASSERT_EQ(actual.size(), static_cast<std::size_t>(shard.rows));
      std::ranges::copy(actual, logical_logits[replay].begin() + shard.first);
    }
    visited += shard.rows;
  }
  ASSERT_EQ(visited, fixture::kVocabularyRows);
  for (std::size_t replay = 0; replay < activations.size(); ++replay)
    for (const auto shard : fixture::shards()) {
      const auto expected = fixture::expected_logits(activations[replay], shard);
      const auto actual = std::span(logical_logits[replay])
                              .subspan(shard.first, shard.rows);
      ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto reference = fixture::decode(expected[i]);
        ASSERT_NEAR(actual[i], reference,
                    std::abs(reference) * 0.012 + 2.0e-5)
            << "global vocabulary row " << shard.first + i;
      }
    }
}

template <class Prepare, class UploadTable, class Execute>
void check_gather(std::int64_t hidden, Prepare prepare,
                  UploadTable upload_table, Execute execute) {
  auto full = prepare(fixture::kShardRows);
  auto tail = prepare(fixture::kVocabularyRows % fixture::kShardRows);
  // Each program sees changed indices; the final three IDs exercise the
  // non-uniform tail shard and both ends of the global vocabulary domain.
  for (const auto global_id : fixture::kGatherIds) {
    ASSERT_GE(global_id, 0);
    ASSERT_LT(global_id, fixture::kVocabularyRows);
    const auto shard = fixture::shard_for(global_id);
    const auto table = fixture::head_weight(shard, hidden);
    auto native_table = upload_table(shard, table);
    auto &call = shard.rows == fixture::kShardRows ? full : tail;
    const auto actual = execute(call, global_id - shard.first, native_table);
    const auto expected = fixture::expected_embedding(global_id, hidden);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t column = 0; column < actual.size(); ++column)
      ASSERT_NEAR(actual[column], fixture::decode(expected[column]), 1.0e-6)
          << "global row " << global_id << " column " << column;
  }
  EXPECT_EQ(fixture::shard_for(fixture::kVocabularyRows - 1).first,
            fixture::kShardRows * 15);
}

} // namespace d5_qwen_vocabulary_test
