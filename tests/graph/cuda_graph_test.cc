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

TEST(GraphCudaReplay, MetadataViewAliasesDeviceStorageWithoutAKernelProvider) {
  auto device = hyle::compute::cuda::device(0).value();
  auto context = Context::create(device).value();
  auto cpu = Context::create(hyle::compute::cpu::device()).value();
  auto resource = hyle::compute::cuda::MemoryResource::create(
                      context, hyle::compute::cuda::MemoryKind::kDevice)
                      .value();
  ASSERT_FALSE(resource.ref().properties().has(
      hyle::mem::MemoryProperty::kHostAccessible));
  auto source = Buffer::create(resource.ref(), 8 * sizeof(float), 256).value();
  const std::array<float, 8> values{0, 1, 2, 3, 4, 5, 6, 7};
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
  auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{8})
          .value();
  auto builder = GraphBuilder::create().value();
  const auto start = builder.dimension({0, 5, 1}).value();
  const auto input = builder.input(description).value();
  const auto output =
      builder
          .add(hyle::graph::ViewSpec::slice(0, hyle::graph::Dimension(start),
                                            hyle::graph::Dimension::fixed(3))
                   .value(),
               input)
          .value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  // Neither a CUDA target nor a CPU metadata target needs a native Provider
  // or host access to the aliased bytes. Their real default storage is valid.
  for (const bool on_cpu : {false, true}) {
    const std::array storage{
        StorageDomain{resource.ref()},
        StorageDomain{
            hyle::compute::cpu::memory_resource(cpu.device()).value()}};
    const std::array targets{ExecutionTarget{
        on_cpu ? &cpu : &context, StorageDomainId{on_cpu ? 1u : 0u}}};
    const std::array external{
        ExternalResourceContract{input, StorageDomainId{0}}};
    auto compiled = hyle::graph::compile(
        graph, {.storage_domains = storage, .targets = targets},
        {.shape_policy = hyle::graph::CompileShapePolicy::kDeclaredDomain,
         .external_resources = external});
    ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
    EXPECT_EQ(compiled->report.copy_bytes(), 0u);
    EXPECT_EQ(compiled->report.step(0)->kind,
              hyle::graph::CompileStepKind::kView);
    EXPECT_FALSE(compiled->report.step(0)->provider.valid());
    EXPECT_EQ(compiled->executable.required_queue_count(), 0u);
    for (std::int64_t offset : {0, 2, 5}) {
      const std::array shape{hyle::graph::ShapeBinding{start, offset}};
      auto created = compiled->executable.create_invocation(shape);
      ASSERT_TRUE(created) << created.error().message();
      auto run = std::move(*created);
      ASSERT_TRUE(
          run.bind(input,
                   TensorView::create(description, source.view()).value())
              .ok());
      ASSERT_TRUE(run.submit()->wait().ok());
      auto result = run.output(output).value();
      EXPECT_EQ(result.storage().resource(), resource.ref());
      EXPECT_EQ(result.storage().offset(),
                static_cast<std::size_t>(offset) * sizeof(float));
      std::array<float, 3> actual{};
      ASSERT_TRUE(result.storage()
                      .read(std::as_writable_bytes(std::span(actual)))
                      .ok());
      EXPECT_EQ(actual, (std::array<float, 3>{static_cast<float>(offset),
                                              static_cast<float>(offset + 1),
                                              static_cast<float>(offset + 2)}));
    }
  }
}

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

TEST(GraphCudaReplay, RejectsCpuCudaComputePlacement) {
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
  ASSERT_FALSE(compiled_result);
  EXPECT_TRUE(compiled_result.error().IsInvalidArgument());
  EXPECT_NE(cpu_context.device(), cuda_context.device());
}

TEST(GraphCudaReplay, CaptureBudgetPreservesCommandsAndReportsUnknownStorage) {
  auto context =
      Context::create(hyle::compute::cuda::device(0).value()).value();
  auto pinned = hyle::compute::cuda::MemoryResource::create(
                    context, hyle::compute::cuda::MemoryKind::kPinnedHost)
                    .value();
  auto device = hyle::compute::cuda::MemoryResource::create(
                    context, hyle::compute::cuda::MemoryKind::kDevice)
                    .value();
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{8})
          .value();
  auto builder = GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto output =
      builder
          .add(hyle::graph::ViewSpec::slice(0, hyle::graph::Dimension::fixed(0),
                                            hyle::graph::Dimension::fixed(8))
                   .value(),
               input)
          .value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array storage{StorageDomain{pinned.ref()},
                           StorageDomain{device.ref()}};
  const std::array targets{ExecutionTarget{&context, StorageDomainId{1}}};
  const std::array external{
      ExternalResourceContract{input, StorageDomainId{0}}};
  const std::array placements{hyle::graph::ValuePlacement{
      output, StorageDomainId{1}, PlacementStrength::kRequired}};
  const CompileEnvironment environment{.storage_domains = storage,
                                       .targets = targets};
  auto source = Buffer::create(pinned.ref(), 8 * sizeof(float), 256).value();
  const auto input_view =
      TensorView::create(description, source.view()).value();
  auto queue = context.create_queue().value();
  const std::array queues{QueueBinding{ExecutionLaneId{0}, &queue}};

  for (const bool allow_unknown : {false, true}) {
    auto compiled = hyle::graph::compile(
        graph, environment,
        {.invocation_storage = hyle::graph::InvocationStorageLimits{
             .max_bytes = 1U << 20,
             .unknown = allow_unknown
                            ? hyle::graph::UnknownStoragePolicy::kAllow
                            : hyle::graph::UnknownStoragePolicy::kReject},
         .value_placements = placements,
         .external_resources = external});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->executable.required_queue_count(), 1U);
    ASSERT_GT(compiled->report.capturable_segment_count(), 0U);
    ASSERT_EQ(compiled->report.copy_bytes(), 8 * sizeof(float));
    {
      auto invocation = compiled->executable.create_invocation().value();
      ASSERT_TRUE(invocation.bind(input, input_view.as_const()).ok());
      ASSERT_TRUE(invocation.prepare_replay(queues).ok());
      EXPECT_EQ(invocation.replay_route(), ReplayRoute::kCommands);
      EXPECT_FALSE(
          compiled->executable.invocation_storage_usage()->has_unknown_storage);
      const auto before =
          compiled->executable.invocation_storage_usage()->live_bytes;
      const auto prepared = invocation.prepare_replay(
          queues, {.mode = ReplayMode::kRequireCapture});
      if (allow_unknown) {
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        EXPECT_EQ(invocation.replay_route(), ReplayRoute::kCaptured);
        EXPECT_TRUE(compiled->executable.invocation_storage_usage()
                        ->has_unknown_storage);
      } else {
        EXPECT_TRUE(prepared.IsResourceExhausted());
        EXPECT_EQ(invocation.replay_route(), ReplayRoute::kCommands);
        EXPECT_EQ(compiled->executable.invocation_storage_usage()->live_bytes,
                  before);
        EXPECT_FALSE(compiled->executable.invocation_storage_usage()
                         ->has_unknown_storage);
      }
      // No graph work runs while recording. Every replay must read the new
      // bytes, including the still-valid commands after a rejected capture.
      for (int replay = 0; replay < 3; ++replay) {
        if (!allow_unknown && replay == 1) {
          ASSERT_TRUE(
              invocation
                  .prepare_replay(queues, {.mode = ReplayMode::kPreferCapture})
                  .ok());
          EXPECT_EQ(invocation.replay_route(), ReplayRoute::kCommands);
        }
        std::array<float, 8> values{};
        for (std::size_t index = 0; index < values.size(); ++index)
          values[index] = static_cast<float>(index + 10 * replay);
        ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
        auto completion = invocation.submit(queues);
        ASSERT_TRUE(completion) << completion.error().message();
        ASSERT_TRUE(completion->wait().ok());
        auto result = invocation.output(output);
        ASSERT_TRUE(result) << result.error().message();
        EXPECT_EQ(result->storage().resource(), device.ref());
        std::array<float, 8> actual{};
        ASSERT_TRUE(result->storage()
                        .read(std::as_writable_bytes(std::span(actual)))
                        .ok());
        EXPECT_EQ(actual, values);
      }
    }
    EXPECT_EQ(compiled->executable.invocation_storage_usage()->live_bytes, 0U);
    EXPECT_EQ(
        compiled->executable.invocation_storage_usage()->has_unknown_storage,
        allow_unknown);
  }
}

} // namespace
