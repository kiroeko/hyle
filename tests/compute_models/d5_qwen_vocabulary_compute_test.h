#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

#include "d5_qwen_vocabulary_test.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"

namespace d5_qwen_vocabulary_compute_test {

namespace c = hyle::compute;
namespace mem = hyle::mem;
namespace fixture = d5_qwen_vocabulary_fixture;

struct Call final {
  c::PreparedOperation prepared;
  mem::Buffer workspace;
  c::OperationInvocation invocation;
};

inline c::TensorDesc dense(c::DataType type,
                           std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

template <class CheckProvider>
void check_gather(const c::Context &context, mem::MemoryResourceRef resource,
                  const c::WorkloadContext &workload, std::int64_t hidden,
                  CheckProvider check_provider) {
  auto queue = context.create_queue().value();
  const auto ids_desc = dense(c::DataType::kInt32, {1});
  const auto output_desc = dense(c::DataType::kBFloat16, {1, hidden});
  const auto prepare = [&](std::int64_t rows) {
    const auto table_desc = dense(c::DataType::kBFloat16, {rows, hidden});
    const std::array inputs{table_desc, ids_desc};
    const std::array outputs{output_desc};
    auto prepared = c::prepare(context, c::Operation(c::Gather{}), inputs,
                               outputs, resource, workload)
                        .value();
    check_provider(prepared);
    EXPECT_EQ(prepared.workload(), workload);
    const auto requirement = prepared.workspace_requirement();
    auto workspace =
        requirement.size == 0
            ? mem::Buffer{}
            : mem::Buffer::create(resource, requirement.size,
                                  requirement.alignment)
                  .value();
    auto invocation = prepared.create_invocation(workspace.view()).value();
    return Call{std::move(prepared), std::move(workspace),
                std::move(invocation)};
  };
  const auto upload_table = [&](fixture::Shard,
                                std::span<const fixture::BF16> values) {
    auto table = mem::Buffer::create(resource, std::as_bytes(values).size(),
                                      256)
                     .value();
    EXPECT_TRUE(table.view().write(std::as_bytes(values)).ok());
    return table;
  };
  const auto execute = [&](Call &call, std::int64_t local_id,
                           const mem::Buffer &table) {
    const auto rows = static_cast<std::int64_t>(
        table.size() / (static_cast<std::size_t>(hidden) *
                        sizeof(fixture::BF16)));
    const auto table_desc = dense(c::DataType::kBFloat16, {rows, hidden});
    auto ids = mem::Buffer::create(resource, sizeof(std::int32_t), 256).value();
    const std::array index{static_cast<std::int32_t>(local_id)};
    EXPECT_TRUE(ids.view().write(std::as_bytes(std::span(index))).ok());
    auto output =
        mem::Buffer::create(resource, output_desc.required_bytes().value(), 256)
            .value();
    const std::array inputs{
        c::ConstTensorView::create(table_desc, table.view()).value(),
        c::ConstTensorView::create(ids_desc, ids.view()).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, output.view()).value()};
    auto done = call.invocation.submit(queue, inputs, outputs);
    EXPECT_TRUE(done) << done.error().message();
    if (done) {
      EXPECT_TRUE(done->wait().ok());
    }
    std::vector<fixture::BF16> raw(static_cast<std::size_t>(hidden));
    EXPECT_TRUE(output.view().read(std::as_writable_bytes(std::span(raw))).ok());
    std::vector<double> result(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i)
      result[i] = fixture::decode(raw[i]);
    return result;
  };
  d5_qwen_vocabulary_test::check_gather(
      hidden, prepare, upload_table, execute,
      [](Call &, std::int64_t, const mem::Buffer &) {
        // This wrapper checks the checked-Provider failure below.
      });

  // The first global ID after the vocabulary maps to the tail shard's first
  // invalid local row. Bypass the consumer's global guard here to exercise the
  // checked Provider failure and output publication boundary as well.
  const auto tail_shard = fixture::shard_for(fixture::kVocabularyRows - 1).value();
  auto tail = prepare(tail_shard.rows);
  const auto table_values = fixture::head_weight(tail_shard, hidden);
  auto table = upload_table(tail_shard, table_values);
  const auto table_desc = dense(c::DataType::kBFloat16,
                                {tail_shard.rows, hidden});
  auto ids = mem::Buffer::create(resource, sizeof(std::int32_t), 256).value();
  const std::array invalid{static_cast<std::int32_t>(tail_shard.rows)};
  ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(invalid))).ok());
  auto output =
      mem::Buffer::create(resource, output_desc.required_bytes().value(), 256)
          .value();
  const std::vector<fixture::BF16> sentinel(static_cast<std::size_t>(hidden),
                                            fixture::encode(0.375));
  ASSERT_TRUE(output.view().write(std::as_bytes(std::span(sentinel))).ok());
  const std::array inputs{
      c::ConstTensorView::create(table_desc, table.view()).value(),
      c::ConstTensorView::create(ids_desc, ids.view()).value()};
  const std::array outputs{
      c::TensorView::create(output_desc, output.view()).value()};
  auto failure = tail.invocation.submit(queue, inputs, outputs);
  if (failure) {
    EXPECT_TRUE(failure->wait().IsOutOfRange());
  }
  else
    EXPECT_TRUE(failure.error().IsOutOfRange());
  std::vector<fixture::BF16> retained(sentinel.size());
  ASSERT_TRUE(output.view()
                  .read(std::as_writable_bytes(std::span(retained)))
                  .ok());
  for (std::size_t column = 0; column < sentinel.size(); ++column)
    EXPECT_EQ(retained[column].bits, sentinel[column].bits) << column;
}

} // namespace d5_qwen_vocabulary_compute_test
