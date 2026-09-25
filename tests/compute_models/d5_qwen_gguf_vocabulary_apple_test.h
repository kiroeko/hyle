#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <mlx/array.h>

#include "../compute_route_apple/tensor_test_support.h"
#include "d5_qwen_gguf_vocabulary_test.h"
#include "hyle/compute/model_ops.h"

namespace d5_qwen_gguf_vocabulary_apple_test {
namespace c = hyle::compute;
namespace mx = mlx::core;
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace fixture = d5_qwen_gguf_vocabulary_fixture;
namespace check = d5_qwen_gguf_vocabulary_test;
namespace vocab = d5_qwen_vocabulary_fixture;

inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

template <class Prepare, class UploadDense, class UploadPacked, class Run>
void check_head(const cases::Case &profile, Prepare prepare,
                UploadDense upload_dense, UploadPacked upload_packed,
                Run run) {
  const auto type = fixture::data_type(profile.output);
  const auto activation_desc = dense(type, {1, profile.hidden});
  const auto workload = d5_qwen35_fixture::workload(
      *profile.profile, c::WorkloadPhase::kDecode, 1);
  const auto make_head = [&](std::int64_t count) {
    return prepare(workload, c::Operation(c::MatMul{}), activation_desc,
                   fixture::head_description(profile.output, profile.hidden,
                                             count));
  };
  const auto upload_weight = [&](vocab::Shard shard,
                                 std::span<const std::byte> bytes) {
    if (fixture::is_quantized(profile.output))
      return upload_packed(bytes);
    return upload_dense(fixture::head_description(profile.output,
                                                  profile.hidden, shard.rows),
                        bytes);
  };
  const auto execute = [&](auto &call, std::span<const float> input,
                           const mx::array &weight, vocab::Shard shard) {
    const auto bytes = fixture::is_quantized(profile.output)
                           ? std::vector<std::byte>(
                                 std::as_bytes(input).begin(),
                                 std::as_bytes(input).end())
                           : fixture::encode_dense(profile.output, input);
    const auto activation = upload_dense(activation_desc, bytes);
    const auto outputs = run(call, activation, weight);
    EXPECT_EQ(outputs.size(), 1U);
    const auto observed = apple_tensor_test::values(outputs.at(0));
    EXPECT_EQ(observed.size(), static_cast<std::size_t>(shard.rows));
    std::vector<float> result(observed.size());
    std::ranges::transform(observed, result.begin(), [](double value) {
      return static_cast<float>(value);
    });
    return result;
  };
  check::check_head(profile, make_head, upload_weight, execute);
}

template <class Prepare, class UploadDense, class UploadPacked, class Run>
void check_gather(const cases::Case &profile, Prepare prepare,
                  UploadDense upload_dense, UploadPacked upload_packed,
                  Run run) {
  const auto ids_desc = dense(c::DataType::kInt32, {1});
  const auto workload = d5_qwen35_fixture::workload(
      *profile.profile, c::WorkloadPhase::kDecode, 1);
  const auto make_gather = [&](std::int64_t count) {
    return prepare(workload, c::Operation(c::Gather{}),
                   fixture::table_description(profile.embedding,
                                              profile.hidden, count),
                   ids_desc);
  };
  const auto upload_table = [&](vocab::Shard shard,
                                std::span<const std::byte> bytes) {
    if (fixture::is_quantized(profile.embedding))
      return upload_packed(bytes);
    return upload_dense(fixture::table_description(profile.embedding,
                                                   profile.hidden, shard.rows),
                        bytes);
  };
  std::optional<mx::array> retained;
  const auto execute = [&](auto &call, std::int64_t local_id,
                           const mx::array &table, vocab::Shard) {
    const std::array index{static_cast<std::int32_t>(local_id)};
    const auto ids = upload_dense(ids_desc, std::as_bytes(std::span(index)));
    const auto outputs = run(call, table, ids);
    EXPECT_EQ(outputs.size(), 1U);
    retained = outputs.at(0);
    const auto observed = apple_tensor_test::values(*retained);
    std::vector<float> result(observed.size());
    std::ranges::transform(observed, result.begin(), [](double value) {
      return static_cast<float>(value);
    });
    return result;
  };
  const auto reject = [&](auto &call, std::int64_t local_id,
                          const mx::array &table, vocab::Shard) {
    ASSERT_TRUE(retained.has_value());
    const auto before = apple_tensor_test::bits(*retained);
    const std::array index{static_cast<std::int32_t>(local_id)};
    const auto ids = upload_dense(ids_desc, std::as_bytes(std::span(index)));
    auto invocation = call.program.create_invocation().value();
    const std::array inputs{table, ids};
    ASSERT_TRUE(invocation.bind(inputs).ok());
    auto failed = invocation.submit();
    if (failed) {
      const auto result = failed->wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsOutOfRange());
    } else {
      EXPECT_TRUE(failed.error().IsOutOfRange());
    }
    EXPECT_EQ(apple_tensor_test::bits(*retained), before);
  };
  const auto absolute_tolerance =
      profile.embedding == cases::Storage::kQ4K ? 5.0e-4F : 1.0e-6F;
  check::check_gather(profile, make_gather, upload_table, execute, reject,
                      absolute_tolerance);
}
} // namespace d5_qwen_gguf_vocabulary_apple_test
