#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "../compute_route_vulkan/tensor_test_support.h"
#include "d5_qwen_gguf_vocabulary_test.h"
#include "hyle/compute/model_ops.h"

namespace d5_qwen_gguf_vocabulary_vulkan_test {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace fixture = d5_qwen_gguf_vocabulary_fixture;
namespace check = d5_qwen_gguf_vocabulary_test;
namespace vocab = d5_qwen_vocabulary_fixture;

inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

template <class Prepare, class Run>
void check_head(const v::Context &context, const cases::Case &profile,
                Prepare prepare, Run run,
                std::span<const std::vector<float>> observed_inputs = {},
                std::span<const std::vector<float>> reference_inputs = {}) {
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
    const auto desc = fixture::head_description(profile.output,
                                                 profile.hidden, shard.rows);
    EXPECT_EQ(bytes.size(), desc.required_bytes().value());
    return context.upload(desc, bytes).value();
  };
  const auto execute = [&](auto &call, std::span<const float> input,
                           const v::Tensor &weight, vocab::Shard shard) {
    const auto bytes = fixture::is_quantized(profile.output)
                           ? std::vector<std::byte>(
                                 std::as_bytes(input).begin(),
                                 std::as_bytes(input).end())
                           : fixture::encode_dense(profile.output, input);
    const auto activation = context.upload(activation_desc, bytes).value();
    const auto outputs = run(call, activation, weight);
    EXPECT_EQ(outputs.size(), 1U);
    const auto observed = vulkan_tensor_test::values(context, outputs.at(0));
    EXPECT_EQ(observed.size(), static_cast<std::size_t>(shard.rows));
    std::vector<float> result(observed.size());
    std::ranges::transform(observed, result.begin(), [](double value) {
      return static_cast<float>(value);
    });
    return result;
  };
  check::check_head(profile, make_head, upload_weight, execute,
                    observed_inputs, reference_inputs);
}

template <class Prepare, class Run>
void check_gather(const v::Context &context, const cases::Case &profile,
                  Prepare prepare, Run run) {
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
    const auto desc = fixture::table_description(profile.embedding,
                                                  profile.hidden, shard.rows);
    EXPECT_EQ(bytes.size(), desc.required_bytes().value());
    return context.upload(desc, bytes).value();
  };
  v::Tensor retained;
  const auto execute = [&](auto &call, std::int64_t local_id,
                           const v::Tensor &table, vocab::Shard) {
    const std::array index{static_cast<std::int32_t>(local_id)};
    const auto ids = context.upload(ids_desc, std::as_bytes(std::span(index)))
                         .value();
    const auto outputs = run(call, table, ids);
    EXPECT_EQ(outputs.size(), 1U);
    retained = outputs.at(0);
    const auto observed = vulkan_tensor_test::values(context, retained);
    std::vector<float> result(observed.size());
    std::ranges::transform(observed, result.begin(), [](double value) {
      return static_cast<float>(value);
    });
    return result;
  };
  const auto reject = [&](auto &call, std::int64_t local_id,
                          const v::Tensor &table, vocab::Shard) {
    const auto before = vulkan_tensor_test::bytes(context, retained);
    const std::array index{static_cast<std::int32_t>(local_id)};
    const auto ids = context.upload(ids_desc, std::as_bytes(std::span(index)))
                         .value();
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
    EXPECT_EQ(vulkan_tensor_test::bytes(context, retained), before);
  };
  const auto absolute_tolerance =
      fixture::is_quantized(profile.embedding) ? 3.0e-3F : 1.0e-6F;
  const auto relative_tolerance =
      fixture::is_quantized(profile.embedding) ? 3.0e-4F : 0.0F;
  check::check_gather(profile, make_gather, upload_table, execute, reject,
                      absolute_tolerance, relative_tolerance);
}
} // namespace d5_qwen_gguf_vocabulary_vulkan_test
