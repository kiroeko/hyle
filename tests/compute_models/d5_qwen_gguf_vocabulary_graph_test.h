#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "d5_qwen_gguf_vocabulary_test.h"
#include "hyle/compute/provider.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/graph.h"
#include "hyle/mem/buffer.h"

namespace d5_qwen_gguf_vocabulary_graph_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace fixture = d5_qwen_gguf_vocabulary_fixture;
namespace check = d5_qwen_gguf_vocabulary_test;
namespace vocab = d5_qwen_vocabulary_fixture;

inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

inline mem::Buffer buffer(mem::MemoryResourceRef resource, std::size_t bytes) {
  return mem::Buffer::create(resource, bytes, 256).value();
}

inline std::vector<std::byte> read(const mem::Buffer &source,
                                   std::size_t bytes) {
  std::vector<std::byte> result(bytes);
  EXPECT_TRUE(source.view().subview(0, bytes).value()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

inline void expect_mature(const g::CompiledGraph &compiled,
                          std::span<const c::ProviderId> providers,
                          const c::WorkloadContext &workload) {
  ASSERT_GT(compiled.report.step_count(), 0U);
  EXPECT_EQ(compiled.executable.workload(), workload);
  for (std::size_t step = 0; step < compiled.report.step_count(); ++step) {
    const auto selected = compiled.report.step(step).value();
    EXPECT_TRUE(std::ranges::find(providers, selected.provider.id) !=
                providers.end())
        << selected.provider.name;
  }
}

struct Call final {
  g::CompiledGraph compiled;
  g::Invocation invocation;
};

template <class Compile>
void check_head(const c::Context &context, mem::MemoryResourceRef resource,
                std::span<const c::ProviderId> providers,
                const cases::Case &profile, Compile compile,
                std::span<const std::vector<float>> observed_inputs = {},
                std::span<const std::vector<float>> reference_inputs = {}) {
  const auto type = fixture::data_type(profile.output);
  const auto activation_desc = dense(type, {1, profile.hidden});
  const auto workload = d5_qwen35_fixture::workload(
      *profile.profile, c::WorkloadPhase::kDecode, 1);
  const auto max_weight_desc = fixture::head_description(
      profile.output, profile.hidden, vocab::kShardRows);
  auto weight = buffer(resource, max_weight_desc.required_bytes().value());
  auto activation = buffer(resource, activation_desc.required_bytes().value());
  const auto max_output_desc = dense(type, {1, vocab::kShardRows});
  auto output = buffer(resource, max_output_desc.required_bytes().value());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};

  const auto make_head = [&](std::int64_t count) {
    const auto weight_desc = fixture::head_description(
        profile.output, profile.hidden, count);
    const auto output_desc = dense(type, {1, count});
    auto builder = g::GraphBuilder::create().value();
    const auto input_id = builder.input(activation_desc).value();
    const auto weight_id = builder.input(weight_desc).value();
    const auto result_id =
        builder.add(c::Operation(c::MatMul{}), input_id, weight_id).value();
    EXPECT_TRUE(builder.publish(result_id).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = compile(logical, workload);
    expect_mature(compiled, providers, workload);
    auto invocation = compiled.executable.create_invocation().value();
    const auto weight_view = weight.view()
                                 .subview(0, weight_desc.required_bytes().value())
                                 .value();
    const auto output_view = output.view()
                                 .subview(0, output_desc.required_bytes().value())
                                 .value();
    EXPECT_TRUE(invocation
                    .bind(input_id, c::ConstTensorView::create(
                                        activation_desc, activation.view())
                                        .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(weight_id,
                          c::ConstTensorView::create(weight_desc, weight_view)
                              .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(result_id,
                          c::TensorView::create(output_desc, output_view)
                              .value())
                    .ok());
    EXPECT_TRUE(invocation.prepare_replay(queues).ok());
    return Call{std::move(compiled), std::move(invocation)};
  };
  const auto upload_weight = [&](vocab::Shard shard,
                                 std::span<const std::byte> bytes) {
    const auto desc = fixture::head_description(
        profile.output, profile.hidden, shard.rows);
    EXPECT_EQ(bytes.size(), desc.required_bytes().value());
    EXPECT_TRUE(weight.view().subview(0, bytes.size()).value().write(bytes).ok());
    return 0;
  };
  const auto execute = [&](Call &call, std::span<const float> input,
                           int, vocab::Shard shard) {
    const auto bytes = fixture::is_quantized(profile.output)
                           ? std::vector<std::byte>(
                                 std::as_bytes(input).begin(),
                                 std::as_bytes(input).end())
                           : fixture::encode_dense(profile.output, input);
    EXPECT_TRUE(activation.view().write(bytes).ok());
    auto done = call.invocation.submit(queue);
    EXPECT_TRUE(done) << done.error().message();
    if (done) {
      EXPECT_TRUE(done->wait().ok());
    }
    return fixture::decode_values(
        profile.output,
        read(output, dense(type, {1, shard.rows}).required_bytes().value()));
  };
  check::check_head(profile, make_head, upload_weight, execute,
                    observed_inputs, reference_inputs);
}

template <class Compile>
void check_gather(const c::Context &context, mem::MemoryResourceRef resource,
                  std::span<const c::ProviderId> providers,
                  const cases::Case &profile, Compile compile) {
  const auto type = fixture::data_type(profile.embedding);
  const auto ids_desc = dense(c::DataType::kInt32, {1});
  const auto output_desc = dense(type, {1, profile.hidden});
  const auto workload = d5_qwen35_fixture::workload(
      *profile.profile, c::WorkloadPhase::kDecode, 1);
  const auto max_table_desc = fixture::table_description(
      profile.embedding, profile.hidden, vocab::kShardRows);
  auto table = buffer(resource, max_table_desc.required_bytes().value());
  auto ids = buffer(resource, ids_desc.required_bytes().value());
  auto output = buffer(resource, output_desc.required_bytes().value());
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};

  const auto make_gather = [&](std::int64_t count) {
    const auto table_desc = fixture::table_description(
        profile.embedding, profile.hidden, count);
    auto builder = g::GraphBuilder::create().value();
    const auto table_id = builder.input(table_desc).value();
    const auto ids_id = builder.input(ids_desc).value();
    const auto output_id =
        builder.add(c::Operation(c::Gather{}), table_id, ids_id).value();
    EXPECT_TRUE(builder.publish(output_id).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = compile(logical, workload);
    expect_mature(compiled, providers, workload);
    auto invocation = compiled.executable.create_invocation().value();
    const auto table_view = table.view()
                                .subview(0, table_desc.required_bytes().value())
                                .value();
    EXPECT_TRUE(invocation
                    .bind(table_id,
                          c::ConstTensorView::create(table_desc, table_view)
                              .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(ids_id,
                          c::ConstTensorView::create(ids_desc, ids.view())
                              .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(output_id,
                          c::TensorView::create(output_desc, output.view())
                              .value())
                    .ok());
    EXPECT_TRUE(invocation.prepare_replay(queues).ok());
    return Call{std::move(compiled), std::move(invocation)};
  };
  const auto upload_table = [&](vocab::Shard shard,
                                std::span<const std::byte> bytes) {
    const auto desc = fixture::table_description(
        profile.embedding, profile.hidden, shard.rows);
    EXPECT_EQ(bytes.size(), desc.required_bytes().value());
    EXPECT_TRUE(table.view().subview(0, bytes.size()).value().write(bytes).ok());
    return 0;
  };
  const auto execute = [&](Call &call, std::int64_t local_id, int,
                           vocab::Shard) {
    const std::array index{static_cast<std::int32_t>(local_id)};
    EXPECT_TRUE(ids.view().write(std::as_bytes(std::span(index))).ok());
    auto done = call.invocation.submit(queue);
    EXPECT_TRUE(done) << done.error().message();
    if (done) {
      EXPECT_TRUE(done->wait().ok());
    }
    return fixture::decode_values(profile.embedding,
                                  read(output, output.size()));
  };
  const auto reject = [&](Call &call, std::int64_t local_id, int,
                          vocab::Shard) {
    std::vector<std::byte> sentinel(output.size(), std::byte{0x5a});
    ASSERT_TRUE(output.view().write(sentinel).ok());
    const std::array index{static_cast<std::int32_t>(local_id)};
    ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(index))).ok());
    auto failed = call.invocation.submit(queue);
    if (failed) {
      EXPECT_TRUE(failed->wait().IsOutOfRange());
    }
    else
      EXPECT_TRUE(failed.error().IsOutOfRange());
    EXPECT_EQ(read(output, output.size()), sentinel);
  };
  check::check_gather(profile, make_gather, upload_table, execute, reject);
}
} // namespace d5_qwen_gguf_vocabulary_graph_test
