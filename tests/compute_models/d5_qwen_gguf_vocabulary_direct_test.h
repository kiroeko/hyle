#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "d5_qwen_gguf_vocabulary_test.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/provider.h"
#include "hyle/mem/buffer.h"

namespace d5_qwen_gguf_vocabulary_direct_test {
namespace c = hyle::compute;
namespace mem = hyle::mem;
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace fixture = d5_qwen_gguf_vocabulary_fixture;
namespace check = d5_qwen_gguf_vocabulary_test;
namespace vocab = d5_qwen_vocabulary_fixture;

c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

inline mem::Buffer buffer(mem::MemoryResourceRef resource, std::size_t bytes) {
  return mem::Buffer::create(resource, bytes, 256).value();
}

inline mem::Buffer upload(mem::MemoryResourceRef resource,
                          std::span<const std::byte> bytes) {
  auto result = buffer(resource, bytes.size());
  EXPECT_TRUE(result.view().write(bytes).ok());
  return result;
}

inline std::vector<std::byte> read(const mem::Buffer &source, std::size_t bytes) {
  std::vector<std::byte> result(bytes);
  EXPECT_TRUE(source.view().subview(0, bytes).value()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

struct Call final {
  c::PreparedOperation prepared;
  mem::Buffer workspace;
  c::OperationInvocation invocation;
};

inline Call prepare(const c::Context &context, mem::MemoryResourceRef resource,
                    std::span<const c::ProviderId> providers,
                    c::Operation operation,
                    std::span<const c::TensorDesc> inputs,
                    std::span<const c::TensorDesc> outputs,
                    const c::WorkloadContext &workload) {
  auto prepared = c::prepare(context, std::move(operation), inputs, outputs,
                             resource, workload)
                      .value();
  EXPECT_TRUE(std::ranges::find(providers, prepared.provider().id) !=
              providers.end())
      << prepared.provider().name;
  EXPECT_EQ(prepared.workload(), workload);
  const auto need = prepared.workspace_requirement();
  auto workspace = need.size == 0
                       ? mem::Buffer{}
                       : mem::Buffer::create(resource, need.size,
                                             need.alignment)
                             .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  return {std::move(prepared), std::move(workspace), std::move(invocation)};
}

inline void check_head(const c::Context &context,
                       mem::MemoryResourceRef resource,
                       std::span<const c::ProviderId> providers,
                       const cases::Case &profile,
                       std::span<const std::vector<float>> observed_inputs = {},
                       std::span<const std::vector<float>> reference_inputs = {}) {
  const auto type = fixture::data_type(profile.output);
  const auto activation_desc = dense(type, {1, profile.hidden});
  const auto workload = d5_qwen35_fixture::workload(
      *profile.profile, c::WorkloadPhase::kDecode, 1);
  auto queue = context.create_queue().value();
  const auto make_head = [&](std::int64_t count) {
    const auto weight_desc =
        fixture::head_description(profile.output, profile.hidden, count);
    const std::array inputs{activation_desc, weight_desc};
    const std::array outputs{dense(type, {1, count})};
    return prepare(context, resource, providers, c::Operation(c::MatMul{}),
                   inputs, outputs, workload);
  };
  const auto upload_weight = [resource](vocab::Shard,
                                        std::span<const std::byte> bytes) {
    return upload(resource, bytes);
  };
  const auto execute = [&](Call &call, std::span<const float> activation,
                           const mem::Buffer &weight, vocab::Shard shard) {
    const auto weight_desc = fixture::head_description(
        profile.output, profile.hidden, shard.rows);
    const auto output_desc = dense(type, {1, shard.rows});
    auto activation_bytes = fixture::is_quantized(profile.output)
                                ? std::vector<std::byte>(
                                      std::as_bytes(activation).begin(),
                                      std::as_bytes(activation).end())
                                : fixture::encode_dense(profile.output,
                                                        activation);
    auto input = upload(resource, activation_bytes);
    auto output = buffer(resource, output_desc.required_bytes().value());
    const std::array inputs{
        c::ConstTensorView::create(activation_desc, input.view()).value(),
        c::ConstTensorView::create(weight_desc, weight.view()).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, output.view()).value()};
    auto done = call.invocation.submit(queue, inputs, outputs);
    EXPECT_TRUE(done) << done.error().message();
    if (done) {
      EXPECT_TRUE(done->wait().ok());
    }
    return fixture::decode_values(
        profile.output, read(output, output_desc.required_bytes().value()));
  };
  check::check_head(profile, make_head, upload_weight, execute,
                    observed_inputs, reference_inputs);
}

inline void check_gather(const c::Context &context,
                         mem::MemoryResourceRef resource,
                         std::span<const c::ProviderId> providers,
                         const cases::Case &profile) {
  const auto type = fixture::data_type(profile.embedding);
  const auto ids_desc = dense(c::DataType::kInt32, {1});
  const auto output_desc = dense(type, {1, profile.hidden});
  const auto workload = d5_qwen35_fixture::workload(
      *profile.profile, c::WorkloadPhase::kDecode, 1);
  auto queue = context.create_queue().value();
  const auto make_gather = [&](std::int64_t count) {
    const auto table_desc = fixture::table_description(
        profile.embedding, profile.hidden, count);
    const std::array inputs{table_desc, ids_desc};
    const std::array outputs{output_desc};
    return prepare(context, resource, providers, c::Operation(c::Gather{}),
                   inputs, outputs, workload);
  };
  const auto upload_table = [resource](vocab::Shard,
                                       std::span<const std::byte> bytes) {
    return upload(resource, bytes);
  };
  const auto execute = [&](Call &call, std::int64_t local_id,
                           const mem::Buffer &table, vocab::Shard shard) {
    const auto table_desc = fixture::table_description(
        profile.embedding, profile.hidden, shard.rows);
    const std::array index{static_cast<std::int32_t>(local_id)};
    auto ids = upload(resource, std::as_bytes(std::span(index)));
    auto output = buffer(resource, output_desc.required_bytes().value());
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
    return fixture::decode_values(
        profile.embedding, read(output, output_desc.required_bytes().value()));
  };
  const auto reject = [&](Call &call, std::int64_t local_id,
                          const mem::Buffer &table, vocab::Shard shard) {
    const auto table_desc = fixture::table_description(
        profile.embedding, profile.hidden, shard.rows);
    const std::array index{static_cast<std::int32_t>(local_id)};
    auto ids = upload(resource, std::as_bytes(std::span(index)));
    auto output = buffer(resource, output_desc.required_bytes().value());
    std::vector<std::byte> sentinel(output.size(), std::byte{0x5a});
    ASSERT_TRUE(output.view().write(sentinel).ok());
    const std::array inputs{
        c::ConstTensorView::create(table_desc, table.view()).value(),
        c::ConstTensorView::create(ids_desc, ids.view()).value()};
    const std::array outputs{
        c::TensorView::create(output_desc, output.view()).value()};
    auto failed = call.invocation.submit(queue, inputs, outputs);
    if (failed) {
      EXPECT_TRUE(failed->wait().IsOutOfRange());
    }
    else
      EXPECT_TRUE(failed.error().IsOutOfRange());
    EXPECT_EQ(read(output, output.size()), sentinel);
  };
  check::check_gather(profile, make_gather, upload_table, execute, reject);
}

} // namespace d5_qwen_gguf_vocabulary_direct_test
