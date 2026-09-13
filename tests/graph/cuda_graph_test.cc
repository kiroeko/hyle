#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"
#include "test_environment.h"

namespace {

using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::Queue;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::graph::CompileEnvironment;
using hyle::graph::CompileOptions;
using hyle::graph::ExternalResourceContract;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::NodePlacement;
using hyle::graph::PlacementStrength;
using hyle::graph::QueueBinding;
using hyle::graph::ReplayMode;
using hyle::graph::ReplayRoute;
using hyle::graph::ValueId;
using hyle::graph::test::Environment;
using hyle::graph::test::ExecutionSpec;
using hyle::mem::Buffer;

TEST(GraphCudaReplay, CapturesEligibleSegmentsAndReplaysAHybridPlan) {
  auto cuda_device = hyle::compute::cuda::device(0);
  ASSERT_TRUE(cuda_device.has_value()) << cuda_device.error().message();
  Context cpu_context = hyle::compute::cpu::create_context().value();
  Context cuda_context = Context::create(*cuda_device).value();
  auto pinned = hyle::compute::cuda::MemoryResource::create(
                    cuda_context, hyle::compute::cuda::MemoryKind::kPinnedHost)
                    .value();
  auto device = hyle::compute::cuda::MemoryResource::create(
                    cuda_context, hyle::compute::cuda::MemoryKind::kDevice)
                    .value();

  const std::array dimensions{std::int64_t{8}};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, dimensions).value();
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(description).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  const Environment environment{{
      ExecutionSpec{.context = &cpu_context, .storage = pinned.ref()},
      ExecutionSpec{.context = &cuda_context, .storage = device.ref()},
  }};
  const std::array placements{NodePlacement{
      .node = graph.node_id(0).value(),
      .target = environment.target(0),
      .strength = PlacementStrength::kRequired,
  }};
  const std::array resources{
      ExternalResourceContract{.value = input,
                               .storage = environment.storage(1)},
      ExternalResourceContract{.value = output,
                               .storage = environment.storage(1)},
  };
  auto compiled_result =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources});
  ASSERT_TRUE(compiled_result.has_value()) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  ASSERT_EQ(compiled.report.target_count(), 2u);
  ASSERT_EQ(compiled.report.storage_domain_count(), 2u);
  ASSERT_EQ(compiled.report.lane_count(), 2u);
  EXPECT_FALSE(compiled.report.lane(0)->capture_supported);
  EXPECT_TRUE(compiled.report.lane(1)->capture_supported);
  EXPECT_EQ(compiled.report.step_count(), 3u);
  EXPECT_EQ(compiled.report.segment_count(), 3u);
  EXPECT_EQ(compiled.report.command_segment_count(), 3u);
  EXPECT_EQ(compiled.report.capturable_segment_count(), 2u);
  EXPECT_EQ(compiled.report.physical_value_count(), 4u);
  EXPECT_FALSE(compiled.report.segment(1)->capture_supported);
  EXPECT_TRUE(compiled.report.segment(0)->capture_supported);
  EXPECT_TRUE(compiled.report.segment(2)->capture_supported);
  ASSERT_EQ(compiled.report.segment(2)->dependency_count, 2u);
  EXPECT_EQ(compiled.report.segment_dependency(2, 0)->source_segment, 0u);
  EXPECT_EQ(compiled.report.segment_dependency(2, 0)->route,
            hyle::graph::CompileDependencyRoute::kOrderedQueue);
  EXPECT_EQ(compiled.report.segment_dependency(2, 1)->source_segment, 1u);
  EXPECT_EQ(compiled.report.segment_dependency(2, 1)->route,
            hyle::graph::CompileDependencyRoute::kSourceCompletesInline);

  const std::size_t bytes = description.required_bytes().value();
  Buffer input_buffer = Buffer::create(device.ref(), bytes, 256).value();
  Buffer output_buffer = Buffer::create(device.ref(), bytes, 256).value();
  TensorView input_view =
      TensorView::create(description, input_buffer.view()).value();
  TensorView output_view =
      TensorView::create(description, output_buffer.view()).value();
  const std::array<float, 8> values{-4.0F, -3.0F, -2.0F, -1.0F,
                                    1.0F,  2.0F,  3.0F,  4.0F};
  ASSERT_TRUE(input_buffer.view().write(std::as_bytes(std::span(values))).ok());

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input, input_view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(output, output_view).ok());
  Queue cpu_queue = cpu_context.create_queue().value();
  Queue cuda_queue = cuda_context.create_queue().value();
  const std::array queues{
      QueueBinding{.lane = environment.lane(0), .queue = &cpu_queue},
      QueueBinding{.lane = environment.lane(1), .queue = &cuda_queue},
  };

  EXPECT_TRUE(
      invocation.prepare_replay(queues, {.mode = ReplayMode::kRequireCapture})
          .IsUnimplemented());
  ASSERT_TRUE(
      invocation.prepare_replay(queues, {.mode = ReplayMode::kPreferCapture})
          .ok());
  EXPECT_EQ(invocation.replay_route(), ReplayRoute::kHybrid);

  for (int replay = 0; replay < 2; ++replay) {
    auto completion = invocation.submit(queues);
    ASSERT_TRUE(completion.has_value()) << completion.error().message();
    ASSERT_TRUE(completion->wait().ok());
    std::array<float, 8> result{};
    ASSERT_TRUE(output_buffer.view()
                    .read(std::as_writable_bytes(std::span(result)))
                    .ok());
    for (std::size_t index = 0; index < values.size(); ++index)
      EXPECT_EQ(result[index], -values[index]);
  }

  ASSERT_TRUE(invocation.bind(input, input_view.as_const()).ok());
  EXPECT_EQ(invocation.replay_route(), ReplayRoute::kUnprepared);
  EXPECT_TRUE(invocation.submit(queues).error().IsFailedPrecondition());
}

} // namespace
