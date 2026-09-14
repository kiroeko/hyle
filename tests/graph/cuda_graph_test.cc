#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

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
using hyle::graph::ExecutionLane;
using hyle::graph::ExecutionLaneId;
using hyle::graph::ExecutionLaneRole;
using hyle::graph::ExecutionTarget;
using hyle::graph::ExecutionTargetId;
using hyle::graph::ExternalResourceContract;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::NodePlacement;
using hyle::graph::PlacementStrength;
using hyle::graph::QueueBinding;
using hyle::graph::ReplayMode;
using hyle::graph::ReplayRoute;
using hyle::graph::StorageDomain;
using hyle::graph::StorageDomainId;
using hyle::graph::ValueId;
using hyle::mem::Buffer;

TEST(GraphCudaReplay, MaterializesConstantsThroughACompletedDeviceCopy) {
  auto device = hyle::compute::cuda::device(0).value();
  auto context = Context::create(device).value();
  auto target = hyle::compute::cuda::MemoryResource::create(
                    context, hyle::compute::cuda::MemoryKind::kDevice)
                    .value();
  for (auto source_kind : {hyle::compute::cuda::MemoryKind::kPinnedHost,
                           hyle::compute::cuda::MemoryKind::kDevice}) {
    auto source_resource =
        hyle::compute::cuda::MemoryResource::create(context, source_kind)
            .value();
    std::array<float, 257> values{};
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(i);
    auto source =
        Buffer::create(source_resource.ref(), sizeof(values), 256).value();
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
    auto description =
        TensorDesc::dense(DataType::kFloat32,
                          std::array{static_cast<std::int64_t>(values.size())})
            .value();
    auto builder = GraphBuilder::create().value();
    auto constant = builder.constant(description).value();
    ASSERT_TRUE(builder.publish(constant).ok());
    auto graph = std::move(builder).finish().value();
    const std::array constants{hyle::graph::ConstantBinding{
        constant,
        hyle::compute::ConstTensorView::create(description, source.view())
            .value(),
        hyle::graph::ConstantPolicy::kMaterialize}};
    const std::array storage{StorageDomain{target.ref()}};
    const std::array targets{ExecutionTarget{&context, StorageDomainId{0}}};
    auto compiled = hyle::graph::compile(
        graph, {.storage_domains = storage, .targets = targets},
        {.constants = constants});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_TRUE(source.reset().ok());
    EXPECT_EQ(compiled->executable.output_ownership().value(),
              hyle::graph::OutputOwnership::kExecutable);
    EXPECT_EQ(compiled->report.materialized_constant_bytes(), sizeof(values));
    auto run = compiled->executable.create_invocation().value();
    compiled->executable = {};
    ASSERT_TRUE(run.submit()->wait().ok());
    std::array<float, 257> actual{};
    ASSERT_TRUE(run.output(constant)
                    ->storage()
                    .read(std::as_writable_bytes(std::span(actual)))
                    .ok());
    EXPECT_EQ(actual, values);
  }
}

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

  const std::array storage_domains{
      StorageDomain{.resource = pinned.ref()},
      StorageDomain{.resource = device.ref()},
  };
  const std::array targets{
      ExecutionTarget{.context = &cpu_context,
                      .default_storage = StorageDomainId{0}},
      ExecutionTarget{.context = &cuda_context,
                      .default_storage = StorageDomainId{1}},
  };
  // The CUDA target has separate compute and transfer lanes backed by the
  // same device storage. This graph emits only transfers on CUDA, so the
  // unused compute lane must neither require a Queue nor duplicate storage.
  const std::array lanes{
      ExecutionLane{.target = ExecutionTargetId{0},
                    .role = ExecutionLaneRole::kCompute},
      ExecutionLane{.target = ExecutionTargetId{1},
                    .role = ExecutionLaneRole::kCompute},
      ExecutionLane{.target = ExecutionTargetId{1},
                    .role = ExecutionLaneRole::kTransfer},
  };
  const CompileEnvironment environment{
      .storage_domains = storage_domains,
      .targets = targets,
      .lanes = lanes,
  };
  const std::array placements{NodePlacement{
      .node = graph.node_id(0).value(),
      .target = ExecutionTargetId{0},
      .strength = PlacementStrength::kRequired,
  }};
  const std::array resources{
      ExternalResourceContract{.value = input,
                               .storage = StorageDomainId{1}},
      ExternalResourceContract{.value = output,
                               .storage = StorageDomainId{1}},
  };
  auto compiled_result =
      hyle::graph::compile(graph, environment,
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources});
  ASSERT_TRUE(compiled_result.has_value()) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  ASSERT_EQ(compiled.report.target_count(), 2u);
  ASSERT_EQ(compiled.report.storage_domain_count(), 2u);
  ASSERT_EQ(compiled.report.lane_count(), 3u);
  EXPECT_FALSE(compiled.report.lane(0)->capture_supported);
  EXPECT_TRUE(compiled.report.lane(1)->capture_supported);
  EXPECT_TRUE(compiled.report.lane(2)->capture_supported);
  EXPECT_TRUE(compiled.report.lane(0)->queue_required);
  EXPECT_FALSE(compiled.report.lane(1)->queue_required);
  EXPECT_TRUE(compiled.report.lane(2)->queue_required);
  EXPECT_EQ(compiled.executable.required_queue_count(), 2u);
  EXPECT_EQ(compiled.report.step_count(), 3u);
  EXPECT_EQ(compiled.report.segment_count(), 3u);
  EXPECT_EQ(compiled.report.command_segment_count(), 3u);
  EXPECT_EQ(compiled.report.capturable_segment_count(), 2u);
  EXPECT_EQ(compiled.report.physical_value_count(), 4u);
  EXPECT_FALSE(compiled.report.segment(1)->capture_supported);
  EXPECT_TRUE(compiled.report.segment(0)->capture_supported);
  EXPECT_TRUE(compiled.report.segment(2)->capture_supported);
  EXPECT_EQ(compiled.report.segment(0)->lane, ExecutionLaneId{2});
  EXPECT_EQ(compiled.report.segment(1)->lane, ExecutionLaneId{0});
  EXPECT_EQ(compiled.report.segment(2)->lane, ExecutionLaneId{2});
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
      QueueBinding{.lane = ExecutionLaneId{0}, .queue = &cpu_queue},
      QueueBinding{.lane = ExecutionLaneId{2}, .queue = &cuda_queue},
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
