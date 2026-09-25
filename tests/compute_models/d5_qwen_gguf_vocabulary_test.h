#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "d5_qwen_gguf_vocabulary_fixture.h"

namespace d5_qwen_gguf_vocabulary_test {
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace fixture = d5_qwen_gguf_vocabulary_fixture;
namespace vocab = d5_qwen_vocabulary_fixture;

// Native route adapters own prepare, upload, execution, and checked failure.
// The same complete logical vocabulary and decoded GGUF oracle are used by
// every Direct/Graph route and every exact embedding/output-head storage role.
template <class Prepare, class Upload, class Execute>
void check_head(const cases::Case &profile, Prepare prepare, Upload upload,
                Execute execute) {
  fixture::SparseRows rows(profile.output, profile.hidden);
  auto full = prepare(vocab::kShardRows);
  auto tail = prepare(vocab::kVocabularyRows % vocab::kShardRows);
  auto activations = std::array{fixture::activation(profile.hidden, 3),
                                fixture::activation(profile.hidden, 17)};
  std::array<std::vector<float>, 2> logical{
      std::vector<float>(vocab::kVocabularyRows),
      std::vector<float>(vocab::kVocabularyRows)};
  for (std::size_t replay = 0; replay < activations.size(); ++replay) {
    if (!fixture::is_quantized(profile.output)) {
      const auto encoded = fixture::encode_dense(profile.output,
                                                  activations[replay]);
      activations[replay] = fixture::decode_values(profile.output, encoded);
    }
  }
  std::int64_t visited = 0;
  for (const auto shard : vocab::shards()) {
    auto weight = upload(shard, rows.shard(shard));
    auto &call = shard.rows == vocab::kShardRows ? full : tail;
    for (std::size_t replay = 0; replay < activations.size(); ++replay) {
      const auto actual = execute(call, activations[replay], weight, shard);
      ASSERT_EQ(actual.size(), static_cast<std::size_t>(shard.rows));
      std::ranges::copy(actual, logical[replay].begin() + shard.first);
    }
    visited += shard.rows;
  }
  ASSERT_EQ(visited, vocab::kVocabularyRows);
  for (std::size_t replay = 0; replay < activations.size(); ++replay) {
    for (const auto shard : vocab::shards()) {
      auto expected = rows.logits(activations[replay], shard);
      if (!fixture::is_quantized(profile.output)) {
        const auto encoded = fixture::encode_dense(profile.output, expected);
        expected = fixture::decode_values(profile.output, encoded);
      }
      const auto actual = std::span(logical[replay]).subspan(shard.first,
                                                              shard.rows);
      if (fixture::is_quantized(profile.output)) {
        // ggml also quantizes the F32 activation for its native Q8 dot path.
        // Use the same decoded-weight oracle and error contract as the other
        // GGUF projection consumers, over every output row of every shard.
        d5_gguf_test::expect_quantized_matmul_close(actual, expected);
        continue;
      }
      for (std::size_t local = 0; local < actual.size(); ++local) {
        ASSERT_NEAR(actual[local], expected[local],
                    std::abs(expected[local]) * 0.012F + 2.0e-5F)
            << profile.name << " global vocabulary row "
            << shard.first + static_cast<std::int64_t>(local);
      }
    }
  }
}

template <class Prepare, class Upload, class Execute, class Reject>
void check_gather(const cases::Case &profile, Prepare prepare, Upload upload,
                  Execute execute, Reject reject,
                  float absolute_tolerance = 1.0e-6F,
                  float relative_tolerance = 0.0F) {
  fixture::SparseRows rows(profile.embedding, profile.hidden);
  auto full = prepare(vocab::kShardRows);
  auto tail = prepare(vocab::kVocabularyRows % vocab::kShardRows);
  for (const auto global_id : vocab::kGatherIds) {
    const auto located = vocab::shard_for(global_id);
    ASSERT_TRUE(located);
    const auto shard = *located;
    auto table = upload(shard, rows.shard(shard));
    auto &call = shard.rows == vocab::kShardRows ? full : tail;
    const auto actual = execute(call, global_id - shard.first, table, shard);
    const auto expected = rows.row(global_id);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t column = 0; column < actual.size(); ++column)
      ASSERT_NEAR(actual[column], expected[column],
                  absolute_tolerance +
                      relative_tolerance * std::abs(expected[column]))
          << profile.name << " global row " << global_id
          << " column " << column;
  }
  EXPECT_FALSE(vocab::shard_for(-1));
  EXPECT_FALSE(vocab::shard_for(vocab::kVocabularyRows));
  const auto tail_shard = vocab::shard_for(vocab::kVocabularyRows - 1).value();
  auto table = upload(tail_shard, rows.shard(tail_shard));
  reject(tail, tail_shard.rows, table, tail_shard);
}
} // namespace d5_qwen_gguf_vocabulary_test
