#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

#include "d5_qwen_vocabulary_fixture.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/graph.h"
#include "hyle/mem/buffer.h"

namespace d5_qwen_vocabulary_graph_test {

namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
namespace fixture = d5_qwen_vocabulary_fixture;

inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

template <class Compile, class CheckProvider>
void check_gather(const c::Context &context, mem::MemoryResourceRef resource,
                  const c::WorkloadContext &workload, std::int64_t hidden,
                  Compile compile, CheckProvider check_provider) {
  const auto ids_desc = dense(c::DataType::kInt32, {1});
  const auto output_desc = dense(c::DataType::kBFloat16, {1, hidden});
  struct GraphCall final {
    g::CompiledGraph compiled;
    g::ValueId table;
    g::ValueId ids;
    g::ValueId output;
  };
  const auto make_graph = [&](std::int64_t rows) {
    const auto table_desc = dense(c::DataType::kBFloat16, {rows, hidden});
    auto builder = g::GraphBuilder::create().value();
    const auto table = builder.input(table_desc).value();
    const auto ids = builder.input(ids_desc).value();
    const auto output = builder.add(c::Operation(c::Gather{}), table, ids).value();
    EXPECT_TRUE(builder.publish(output).ok());
    auto logical = std::move(builder).finish().value();
    auto compiled = compile(logical, workload);
    check_provider(compiled);
    EXPECT_EQ(compiled.executable.workload(), workload);
    return GraphCall{std::move(compiled), table, ids, output};
  };
  auto full = make_graph(fixture::kShardRows);
  auto tail = make_graph(fixture::kVocabularyRows % fixture::kShardRows);

  const auto max_table_desc =
      dense(c::DataType::kBFloat16, {fixture::kShardRows, hidden});
  auto table_buffer =
      mem::Buffer::create(resource, max_table_desc.required_bytes().value(), 256)
          .value();
  auto ids_buffer = mem::Buffer::create(resource, sizeof(std::int32_t), 256).value();
  auto output_buffer =
      mem::Buffer::create(resource, output_desc.required_bytes().value(), 256)
          .value();

  auto full_invocation = full.compiled.executable.create_invocation().value();
  auto tail_invocation = tail.compiled.executable.create_invocation().value();
  const auto bind = [&](g::Invocation &invocation, const GraphCall &call,
                        std::int64_t rows) {
    const auto table_desc = dense(c::DataType::kBFloat16, {rows, hidden});
    const auto table_view = table_buffer.view()
                                .subview(0, table_desc.required_bytes().value())
                                .value();
    EXPECT_TRUE(invocation
                    .bind(call.table,
                          c::ConstTensorView::create(table_desc, table_view)
                              .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(call.ids,
                          c::ConstTensorView::create(ids_desc, ids_buffer.view())
                              .value())
                    .ok());
    EXPECT_TRUE(invocation
                    .bind(call.output,
                          c::TensorView::create(output_desc, output_buffer.view())
                              .value())
                    .ok());
  };
  bind(full_invocation, full, fixture::kShardRows);
  bind(tail_invocation, tail,
       fixture::kVocabularyRows % fixture::kShardRows);
  auto queue = context.create_queue().value();
  const std::array queues{
      g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(full_invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(tail_invocation.prepare_replay(queues).ok());

  for (const auto global_id : fixture::kGatherIds) {
    const auto shard = fixture::shard_for(global_id);
    const auto table = fixture::head_weight(shard, hidden);
    const auto table_view = table_buffer.view()
                                .subview(0, std::as_bytes(std::span(table)).size())
                                .value();
    ASSERT_TRUE(table_view.write(std::as_bytes(std::span(table))).ok());
    const std::array index{
        static_cast<std::int32_t>(global_id - shard.first)};
    ASSERT_TRUE(ids_buffer.view().write(std::as_bytes(std::span(index))).ok());
    auto &invocation = shard.rows == fixture::kShardRows ? full_invocation
                                                        : tail_invocation;
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::vector<fixture::BF16> actual(static_cast<std::size_t>(hidden));
    ASSERT_TRUE(output_buffer.view()
                    .read(std::as_writable_bytes(std::span(actual)))
                    .ok());
    const auto expected = fixture::expected_embedding(global_id, hidden);
    for (std::size_t column = 0; column < actual.size(); ++column)
      ASSERT_EQ(actual[column].bits, expected[column].bits)
          << "global row " << global_id << " column " << column;
  }
}

} // namespace d5_qwen_vocabulary_graph_test
