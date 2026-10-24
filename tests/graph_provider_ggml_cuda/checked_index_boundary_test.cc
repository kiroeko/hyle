#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/compute/low_precision.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

#include <array>
#include <cstdint>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

TEST(GgmlCudaGraph, ProducedCheckedIndicesKeepTheirNativeValidationBoundary) {
  // Concat is a supported I32 operation, but its result must remain an input
  // boundary of the checked reader. A region cannot validate that result as
  // an original input if it also swallows the index producer.
  auto context = c::Context::create(c::cuda::device(0).value(),
      g::ggml_cuda::provider()).value();
  auto device = c::StorageOwner::adopt(context, c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value()).value();
  auto managed = c::StorageOwner::adopt(context, c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kManaged).value()).value();
  const auto table_desc = c::gguf_weight_description(c::GgufQuantizedType::kQ8_0,
      c::DataType::kFloat32, 32, 4).value().transposed().value();
  const auto index_desc = dense(c::DataType::kInt32, {1});
  const auto output_desc = dense(c::DataType::kFloat32, {2, 32});
  auto builder = g::GraphBuilder::create().value();
  const auto table = builder.input(table_desc).value();
  const auto first = builder.input(index_desc).value();
  const auto second = builder.input(index_desc).value();
  const auto indices = builder.result(builder.add_node(c::Operation(c::Concat{.axis = 0}),
      std::array{first, second}).value()).value();
  const auto gathered = builder.add(c::Gather{.bounds = c::IndexBoundsPolicy::kChecked},
      table, indices).value();
  const auto result = builder.add(c::Unary{c::UnaryKind::kNeg}, gathered).value();
  ASSERT_TRUE(builder.publish(gathered).ok());
  ASSERT_TRUE(builder.publish(result).ok());
  const auto graph = std::move(builder).finish().value();

  // The host-visible managed alternative creates real resource choices.
  // Retrying the same invalid fused reader over those choices used to spend
  // the whole search budget before reaching the valid ordinary leaf route.
  const std::array domains{
      g::StorageDomain{.resource = device.ref(), .owner = device},
      g::StorageDomain{.resource = managed.ref(), .owner = managed}};
  const std::array targets{g::ExecutionTarget{
      .source = g::ExecutionSource::acquire(context).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  constexpr std::size_t search_budget = 32;
  auto compiled = g::compile(graph,
      {.storage_domains = domains, .targets = targets, .lanes = lanes},
      {.max_route_search_states = search_budget});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_FALSE(compiled->report.search().route_search_limit_reached);
  EXPECT_LT(compiled->report.search().route_search_states, search_budget);
  // A resource-qualification decline is counted as a route rejection too.
  // It must not cause unsupported native artifacts to be prepared repeatedly.
  EXPECT_LE(compiled->report.search().placement_attempts, graph.node_count());

  std::vector<float> values(4 * 32);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) / 4.F;
  // Four original Q8_0 blocks: exact binary16 scale 0.25 and signed codes
  // 0..127. The independent decoded table above has no quantizer tolerance.
  std::vector<std::byte> encoded(4 * 34);
  for (std::size_t row = 0; row < 4; ++row) {
    encoded[row * 34 + 1] = std::byte{0x34};
    for (std::size_t column = 0; column < 32; ++column)
      encoded[row * 34 + 2 + column] = static_cast<std::byte>(row * 32 + column);
  }
  ASSERT_EQ(encoded.size(), table_desc.required_bytes().value());
  auto table_bytes = upload<std::byte>(device.ref(), encoded);
  auto left = upload<std::int32_t>(device.ref(), std::array<std::int32_t, 1>{3});
  auto right = upload<std::int32_t>(device.ref(), std::array<std::int32_t, 1>{1});
  auto selected_output = output_buffer(device.ref(), output_desc);
  auto output = output_buffer(device.ref(), output_desc);
  auto queue = context.create_queue().value();
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  for (const bool invalid : {false, true}) {
    SCOPED_TRACE(invalid ? "invalid produced index" : "valid produced indices");
    const std::array<std::int32_t, 1> changed{invalid ? 4 : 1};
    ASSERT_TRUE(right.view().write(std::as_bytes(std::span(changed))).ok());
    const std::vector<float> untouched(2 * 32, 1337.F);
    ASSERT_TRUE(selected_output.view().write(std::as_bytes(std::span(untouched))).ok());
    ASSERT_TRUE(output.view().write(std::as_bytes(std::span(untouched))).ok());
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(table, c::ConstTensorView::create(table_desc, table_bytes.view()).value()).ok());
    ASSERT_TRUE(invocation.bind(first, c::ConstTensorView::create(index_desc, left.view()).value()).ok());
    ASSERT_TRUE(invocation.bind(second, c::ConstTensorView::create(index_desc, right.view()).value()).ok());
    ASSERT_TRUE(invocation.bind(gathered, c::TensorView::create(output_desc, selected_output.view()).value()).ok());
    ASSERT_TRUE(invocation.bind(result, c::TensorView::create(output_desc, output.view()).value()).ok());
    ASSERT_TRUE(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kCommands}).ok());
    auto done = invocation.submit(queues);
    ASSERT_TRUE(done) << done.error().message();
    const auto status = done->wait();
    if (invalid) {
      EXPECT_TRUE(status.IsOutOfRange()) << status.message();
      EXPECT_EQ(half_test::read<float>(selected_output), untouched);
      // Checked Gather protects its own output. An independent later node's
      // borrowed caller buffer is not a transactional whole-Graph result.
      EXPECT_FALSE(invocation.output(gathered));
      EXPECT_FALSE(invocation.output(result));
      auto retired = invocation.submit(queues);
      ASSERT_FALSE(retired);
      EXPECT_TRUE(retired.error().IsFailedPrecondition());
    } else {
      ASSERT_TRUE(status.ok()) << status.message();
      const auto actual = half_test::read<float>(output);
      const auto selected = half_test::read<float>(selected_output);
      ASSERT_EQ(actual.size(), 2U * 32U);
      ASSERT_EQ(selected.size(), actual.size());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(selected[i], values[(i < 32 ? 3 : 1) * 32 + i % 32]) << i;
        EXPECT_EQ(actual[i], -values[(i < 32 ? 3 : 1) * 32 + i % 32]) << i;
      }
    }
  }
}
} // namespace
