#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/compute/cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"
#include "test_environment.h"

namespace {

using hyle::compute::AttentionLayout;
using hyle::compute::Binary;
using hyle::compute::BinaryKind;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::GatedActivation;
using hyle::compute::GatedActivationKind;
using hyle::compute::GatedInputKind;
using hyle::compute::MatMul;
using hyle::compute::Queue;
using hyle::compute::RmsNorm;
using hyle::compute::ScaledDotProductAttention;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::compute::View;
using hyle::graph::CompiledGraph;
using hyle::graph::CompileOptions;
using hyle::graph::ConstantBinding;
using hyle::graph::Dimension;
using hyle::graph::Executable;
using hyle::graph::ExternalResourceContract;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::Invocation;
using hyle::graph::NodePlacement;
using hyle::graph::OutputOwnership;
using hyle::graph::PlacementStrength;
using hyle::graph::QueueBinding;
using hyle::graph::ShapeBinding;
using hyle::graph::TensorSpec;
using hyle::graph::ValueId;
using hyle::graph::test::Environment;
using hyle::graph::test::ExecutionSpec;
using hyle::mem::AllocatorRef;
using hyle::mem::Buffer;

TensorDesc row(std::int64_t size) {
  const std::array shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array shape{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

struct LogicalGraph final {
  Graph graph{};
  std::vector<ValueId> inputs{};
  std::vector<ValueId> outputs{};
  std::vector<ValueId> intermediates{};
};

LogicalGraph make_rms_graph(TensorDesc description, bool weighted = false) {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  result.inputs.push_back(builder.input(description).value());
  ValueId output;
  if (weighted) {
    const TensorDesc weight = row(description.dimensions().back());
    result.inputs.push_back(builder.input(weight).value());
    output = builder.add(RmsNorm{}, result.inputs[0], result.inputs[1]).value();
  } else {
    output = builder.add(RmsNorm{}, result.inputs[0]).value();
  }
  EXPECT_TRUE(builder.publish(output).ok());
  result.outputs.push_back(output);
  result.graph = std::move(builder).finish().value();
  return result;
}

LogicalGraph make_transformer_graph(std::size_t rows, std::size_t hidden,
                                    std::size_t ffn) {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  const TensorDesc activation = matrix(static_cast<std::int64_t>(rows),
                                       static_cast<std::int64_t>(hidden));
  const TensorDesc norm_weight = row(static_cast<std::int64_t>(hidden));
  const TensorDesc stored_weight = matrix(static_cast<std::int64_t>(2 * ffn),
                                          static_cast<std::int64_t>(hidden));
  result.inputs.push_back(builder.input(activation).value());
  result.inputs.push_back(builder.input(norm_weight).value());
  result.inputs.push_back(
      builder.input(stored_weight.transposed().value()).value());
  const ValueId normalized =
      builder.add(RmsNorm{}, result.inputs[0], result.inputs[1]).value();
  const ValueId projected =
      builder.add(MatMul{}, normalized, result.inputs[2]).value();
  const ValueId activated = builder.add(GatedActivation{}, projected).value();
  result.intermediates = {normalized, projected};
  result.outputs.push_back(activated);
  EXPECT_TRUE(builder.publish(activated).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

LogicalGraph make_branch_graph() {
  GraphBuilder builder = GraphBuilder::create().value();
  LogicalGraph result;
  result.inputs.push_back(builder.input(matrix(2, 3)).value());
  result.inputs.push_back(builder.input(matrix(3, 2)).value());
  result.inputs.push_back(builder.input(matrix(3, 1)).value());
  result.outputs.push_back(
      builder.add(MatMul{}, result.inputs[0], result.inputs[1]).value());
  result.outputs.push_back(
      builder.add(MatMul{}, result.inputs[0], result.inputs[2]).value());
  EXPECT_TRUE(builder.publish(result.outputs[0]).ok());
  EXPECT_TRUE(builder.publish(result.outputs[1]).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

struct OwnedTensor final {
  Buffer buffer{};
  TensorView view{};
};

class HostResource final : public hyle::mem::MemoryResource {
public:
  std::string_view name() const noexcept override { return "graph.test.host"; }

  hyle::mem::MemoryProperties properties() const noexcept override {
    return hyle::mem::MemoryProperties{
        .flags = hyle::mem::MemoryProperty::kHostAccessible |
                 hyle::mem::MemoryProperty::kHostDeviceCoherent,
        .domain = hyle::mem::MemoryDomainId{this, 0},
    };
  }

  hyle::Result<hyle::mem::ResourceAllocation>
  allocate(std::size_t size, std::size_t alignment) noexcept override {
    return hyle::mem::system_resource().resource().allocate(size, alignment);
  }

  void deallocate(hyle::mem::ResourceAllocation allocation) noexcept override {
    hyle::mem::system_resource().resource().deallocate(allocation);
  }

  hyle::Result<hyle::mem::ResourceHostMapping>
  map_host(void *native_handle, std::size_t offset, std::size_t size,
           hyle::mem::HostAccess) noexcept override {
    return hyle::mem::ResourceHostMapping{
        size == 0 ? nullptr
                  : static_cast<void *>(
                        static_cast<std::byte *>(native_handle) + offset),
        nullptr,
    };
  }
};

OwnedTensor make_tensor(TensorDesc description,
                        hyle::mem::MemoryResourceRef resource) {
  const std::size_t bytes = description.required_bytes().value();
  OwnedTensor result;
  result.buffer = Buffer::create(resource, bytes, 64).value();
  result.view = TensorView::create(description, result.buffer.view()).value();
  return result;
}

OwnedTensor make_tensor(TensorDesc description) {
  return make_tensor(description, hyle::mem::system_resource());
}

void write(OwnedTensor &tensor, std::span<const float> values) {
  ASSERT_TRUE(tensor.buffer.view().write(std::as_bytes(values)).ok());
}

std::vector<float> read(const OwnedTensor &tensor, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(tensor.buffer.view()
                  .read(std::as_writable_bytes(std::span(result)))
                  .ok());
  return result;
}

std::vector<float> read(ConstTensorView tensor, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(
      tensor.storage().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

std::vector<float> expected_transformer(std::span<const float> input,
                                        std::span<const float> norm_weight,
                                        std::span<const float> stored_matrix,
                                        std::size_t rows, std::size_t hidden,
                                        std::size_t ffn) {
  std::vector<float> normalized(rows * hidden);
  for (std::size_t r = 0; r < rows; ++r) {
    float sum = 0.0F;
    for (std::size_t k = 0; k < hidden; ++k) {
      const float value = input[r * hidden + k];
      sum += value * value;
    }
    const float scale =
        1.0F / std::sqrt(sum / static_cast<float>(hidden) + RmsNorm{}.epsilon);
    for (std::size_t k = 0; k < hidden; ++k) {
      normalized[r * hidden + k] =
          input[r * hidden + k] * scale * norm_weight[k];
    }
  }

  std::vector<float> projected(rows * 2 * ffn);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t n = 0; n < 2 * ffn; ++n) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < hidden; ++k) {
        sum += normalized[r * hidden + k] * stored_matrix[n * hidden + k];
      }
      projected[r * 2 * ffn + n] = sum;
    }
  }

  std::vector<float> output(rows * ffn);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t n = 0; n < ffn; ++n) {
      const float gate = projected[r * 2 * ffn + n];
      const float value = projected[r * 2 * ffn + ffn + n];
      output[r * ffn + n] = (gate / (1.0F + std::exp(-gate))) * value;
    }
  }
  return output;
}

void expect_near(std::span<const float> actual,
                 std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 5.0e-5F + 5.0e-6F)
        << "index " << index;
  }
}

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> deallocations{0};
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> fail_on_attempt{
      (std::numeric_limits<std::size_t>::max)()};
  std::atomic<bool> fail{false};
};

bool should_fail(AllocatorProbe &probe) noexcept {
  if (probe.fail.load())
    return true;
  const std::size_t attempt = probe.attempts.fetch_add(1);
  return attempt == probe.fail_on_attempt.load();
}

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (should_fail(probe))
    return nullptr;
  void *result = probe.backing.allocate(size, alignment);
  if (result != nullptr)
    ++probe.allocations;
  return result;
}

void probe_free(void *opaque, void *pointer, std::size_t size,
                std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (pointer != nullptr)
    ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *opaque, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (should_fail(probe))
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_free, &probe_reallocate, &probe);
}

TEST(GraphCompile, CompilesEveryNodeAndReportsItsWorkspaceHonestly) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(logical.graph, context).value();

  EXPECT_TRUE(compiled.executable.valid());
  EXPECT_EQ(compiled.executable.input_count(), 3u);
  EXPECT_EQ(compiled.executable.output_count(), 1u);
  EXPECT_EQ(compiled.executable.node_count(), 3u);
  EXPECT_EQ(compiled.executable.input_id(1).value(), logical.inputs[1]);
  EXPECT_EQ(compiled.executable.output_id().value(), logical.outputs[0]);
  EXPECT_EQ(compiled.report.backend(), hyle::compute::cpu::kBackendId);
  EXPECT_EQ(compiled.report.provider().id, hyle::compute::native::kProviderId);
  EXPECT_EQ(compiled.report.node_count(), 3u);
  EXPECT_GT(compiled.report.workspace_bytes(), 0u);
  EXPECT_GT(compiled.report.workspace_backing_count(), 0u);
  EXPECT_EQ(compiled.report.workspace_bytes(),
            compiled.executable.workspace_bytes());
  EXPECT_EQ(compiled.report.workspace_backing_count(),
            compiled.executable.workspace_backing_count());
  EXPECT_EQ(compiled.report.segment_count(), 1u);
  EXPECT_EQ(compiled.report.command_segment_count(), 1u);
  EXPECT_EQ(compiled.report.delegated_segment_count(), 0u);
  EXPECT_EQ(compiled.report.serialization_edge_count(), 0u);
  EXPECT_EQ(compiled.report.segment_dependency_count(), 0u);
  ASSERT_EQ(compiled.report.lane_count(), 1u);
  EXPECT_EQ(compiled.report.lane(0)->runtime_mode,
            hyle::compute::ProviderRuntimeMode::kSharedQueue);
  EXPECT_EQ(compiled.report.copy_bytes(), 0u);
  EXPECT_EQ(compiled.report.synchronization_count(), 0u);
  EXPECT_FALSE(compiled.report.mixed_providers());

  Executable copied = compiled.executable;
  Executable moved = std::move(copied);
  EXPECT_FALSE(copied.valid()); // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(moved.create_invocation().has_value());
}

TEST(GraphCompile, LargeTargetSetKeepsDeterministicPreferredOrdinal) {
  constexpr std::size_t kTargetCount = 33;
  const LogicalGraph logical = make_rms_graph(row(8));
  AllocatorProbe probe;
  std::array<Context, kTargetCount> contexts{};
  std::array<hyle::graph::ExecutionTarget, kTargetCount> targets{};
  const std::array storage_domains{hyle::graph::StorageDomain{
      .resource = hyle::mem::system_resource(),
  }};
  for (std::size_t index = 0; index < kTargetCount; ++index) {
    contexts[index] = hyle::compute::cpu::create_context(
                          {.metadata_allocator = probe_allocator(probe)})
                          .value();
    targets[index] = hyle::graph::ExecutionTarget{
        .context = &contexts[index],
        .default_storage = hyle::graph::StorageDomainId{0},
    };
  }
  const std::array placements{NodePlacement{
      .node = logical.graph.node_id().value(),
      .target = hyle::graph::ExecutionTargetId{static_cast<std::uint32_t>(
          kTargetCount - 1)},
      .strength = PlacementStrength::kPreferred,
  }};
  const hyle::graph::CompileEnvironment environment{
      .storage_domains = storage_domains,
      .targets = targets,
  };

  auto compiled =
      hyle::graph::compile(logical.graph, environment,
                           CompileOptions{.node_placements = placements});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  const auto step = compiled->report.step(0);
  ASSERT_TRUE(step.has_value()) << step.error().message();
  EXPECT_EQ(step->target, hyle::graph::ExecutionTargetId{
                              static_cast<std::uint32_t>(kTargetCount - 1)});
}

TEST(GraphCompile, SpecializesBoundedShapesWithExactCollisionSafeIdentity) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({.minimum = 1, .maximum = 4}).value();
  const TensorSpec activation =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const auto input_id = builder.input(activation).value();
  const auto output_id = builder.add(RmsNorm{}, activation, input_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();

  EXPECT_TRUE(hyle::graph::compile(graph, context).error().IsInvalidArgument());
  const std::array outside{ShapeBinding{rows, 5}};
  EXPECT_TRUE(
      hyle::graph::compile(graph, context, CompileOptions{.shape = outside})
          .error()
          .IsOutOfRange());

  const std::array decode_shape{ShapeBinding{rows, 1}};
  CompiledGraph decode =
      hyle::graph::compile(graph, context,
                           CompileOptions{.shape = decode_shape})
          .value();
  const auto executable_signature = decode.executable.specialization();
  const auto report_signature = decode.report.specialization();
  ASSERT_TRUE(executable_signature.valid());
  EXPECT_EQ(executable_signature.binding_count(), 1u);
  EXPECT_EQ(executable_signature.binding().value(), decode_shape[0]);
  EXPECT_EQ(executable_signature, report_signature);
  EXPECT_EQ(executable_signature.hash(), report_signature.hash());

  CompiledGraph same =
      hyle::graph::compile(graph, context,
                           CompileOptions{.shape = decode_shape})
          .value();
  EXPECT_EQ(executable_signature, same.executable.specialization());
  const std::array prefill_shape{ShapeBinding{rows, 4}};
  CompiledGraph prefill =
      hyle::graph::compile(graph, context,
                           CompileOptions{.shape = prefill_shape})
          .value();
  EXPECT_NE(executable_signature, prefill.executable.specialization());

  OwnedTensor input = make_tensor(matrix(1, 4));
  OwnedTensor output = make_tensor(matrix(1, 4));
  write(input, std::array<float, 4>{1, -2, 3, -4});
  Invocation invocation = decode.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(GraphCompile, RejectsSymbolicOutputThatFailsConcreteInference) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto rows = builder.dimension({.minimum = 1, .maximum = 4}).value();
  const TensorSpec input_spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(4)})
          .value();
  const TensorSpec wrong_output =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(rows), Dimension::fixed(5)})
          .value();
  const auto input = builder.input(input_spec).value();
  const auto output = builder.add(RmsNorm{}, wrong_output, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();
  const std::array shape{ShapeBinding{rows, 2}};
  EXPECT_TRUE(
      hyle::graph::compile(graph, context, CompileOptions{.shape = shape})
          .error()
          .IsInvalidArgument());
}

TEST(GraphCompile, PlacesAcrossTargetsWithoutFalseStorageCopies) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{
      ExecutionSpec{.context = &first, .storage = host},
      ExecutionSpec{.context = &second, .storage = host},
  }};
  const std::array placements{
      NodePlacement{.node = logical.graph.node_id(0).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = logical.graph.node_id(1).value(),
                    .target = environment.target(1),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = logical.graph.node_id(2).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
  };
  const std::array resources{
      ExternalResourceContract{.value = logical.inputs[0],
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = logical.inputs[1],
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = logical.inputs[2],
                               .storage = environment.storage(1)},
      ExternalResourceContract{.value = logical.outputs[0],
                               .storage = environment.storage(0)},
  };
  CompiledGraph compiled =
      hyle::graph::compile(logical.graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  EXPECT_EQ(compiled.executable.target_count(), 2u);
  EXPECT_EQ(compiled.executable.storage_domain_count(), 1u);
  EXPECT_EQ(compiled.executable.lane_count(), 2u);
  EXPECT_EQ(compiled.report.step_count(), 3u);
  EXPECT_EQ(compiled.report.fallback_operation_count(), 3u);
  EXPECT_EQ(compiled.report.copy_bytes(), 0u);
  EXPECT_TRUE(compiled.report.lane(0)->queue_required);
  EXPECT_TRUE(compiled.report.lane(1)->queue_required);
  EXPECT_FALSE(compiled.report.lane(0)->asynchronous_queue);
  EXPECT_FALSE(compiled.report.lane(1)->asynchronous_queue);
  EXPECT_EQ(compiled.report.synchronization_count(), 0u);
  for (std::size_t segment = 0; segment < compiled.report.segment_count();
       ++segment) {
    const auto segment_report = compiled.report.segment(segment).value();
    for (std::size_t dependency = 0;
         dependency < segment_report.dependency_count; ++dependency) {
      EXPECT_NE(compiled.report.segment_dependency(segment, dependency)->route,
                hyle::graph::CompileDependencyRoute::kQueueBindingDependent);
    }
  }

  OwnedTensor input = make_tensor(matrix(2, 4));
  OwnedTensor norm_weight = make_tensor(row(4));
  OwnedTensor matrix_weight = make_tensor(matrix(8, 4).transposed().value());
  OwnedTensor output = make_tensor(matrix(2, 4));
  const std::array<float, 8> input_values{1, -2, 3, -4, 0.5F, 1, -1, 2};
  const std::array<float, 4> norm_values{1, 0.5F, -1, 2};
  std::array<float, 32> matrix_values{};
  for (std::size_t index = 0; index < matrix_values.size(); ++index)
    matrix_values[index] = static_cast<float>(index % 7) * 0.1F - 0.2F;
  write(input, input_values);
  write(norm_weight, norm_values);
  write(matrix_weight, matrix_values);

  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array wrong_queues{
      QueueBinding{.lane = environment.lane(0), .queue = &first_queue},
      QueueBinding{.lane = environment.lane(1), .queue = &first_queue},
  };
  EXPECT_TRUE(invocation.prepare_replay(wrong_queues).IsInvalidArgument());
  const std::array queues{
      QueueBinding{.lane = environment.lane(0), .queue = &first_queue},
      QueueBinding{.lane = environment.lane(1), .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto replay = invocation.replay_report().value();
  EXPECT_EQ(replay.device_wait_dependency_count, 0u);
  EXPECT_EQ(replay.host_wait_dependency_count, 0u);
  EXPECT_GT(replay.synchronous_source_dependency_count, 0u);
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(output, 8), expected_transformer(input_values, norm_values,
                                                    matrix_values, 2, 4, 4));
}

TEST(GraphCompile,
     SchedulesInterleavedIndependentBranchesIntoTwoQueuePrefixes) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(4)).value();
  const ValueId second_input = builder.input(row(4)).value();
  const ValueId first_middle =
      builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  const ValueId second_middle =
      builder.add(Unary{UnaryKind::kNeg}, second_input).value();
  const ValueId first_output =
      builder.add(Unary{UnaryKind::kNeg}, first_middle).value();
  const ValueId second_output =
      builder.add(Unary{UnaryKind::kNeg}, second_middle).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  Graph graph = std::move(builder).finish().value();

  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{
      ExecutionSpec{.context = &first, .storage = host},
      ExecutionSpec{.context = &second, .storage = host},
  }};
  const std::array placements{
      NodePlacement{.node = graph.node_id(0).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(1).value(),
                    .target = environment.target(1),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(2).value(),
                    .target = environment.target(0),
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(3).value(),
                    .target = environment.target(1),
                    .strength = PlacementStrength::kRequired},
  };
  const std::array resources{
      ExternalResourceContract{.value = first_input,
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = second_input,
                               .storage = environment.storage(1)},
      ExternalResourceContract{.value = first_output,
                               .storage = environment.storage(0)},
      ExternalResourceContract{.value = second_output,
                               .storage = environment.storage(1)},
  };
  CompiledGraph compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  ASSERT_EQ(compiled.report.step_count(), 4u);
  ASSERT_EQ(compiled.report.segment_count(), 2u);
  EXPECT_EQ(compiled.report.segment_dependency_count(), 0u);
  EXPECT_EQ(compiled.report.step(0)->target, environment.target(0));
  EXPECT_EQ(compiled.report.step(1)->target, environment.target(0));
  EXPECT_EQ(compiled.report.step(2)->target, environment.target(1));
  EXPECT_EQ(compiled.report.step(3)->target, environment.target(1));
  EXPECT_EQ(compiled.report.step(0)->node_begin, 0u);
  EXPECT_EQ(compiled.report.step(1)->node_begin, 2u);
  EXPECT_EQ(compiled.report.step(2)->node_begin, 1u);
  EXPECT_EQ(compiled.report.step(3)->node_begin, 3u);

  OwnedTensor first_source = make_tensor(row(4));
  OwnedTensor second_source = make_tensor(row(4));
  OwnedTensor first_destination = make_tensor(row(4));
  OwnedTensor second_destination = make_tensor(row(4));
  write(first_source, std::array<float, 4>{1, 2, 3, 4});
  write(second_source, std::array<float, 4>{5, 6, 7, 8});
  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(first_output, first_destination.view).ok());
  ASSERT_TRUE(invocation.bind(second_output, second_destination.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      QueueBinding{.lane = environment.lane(0), .queue = &first_queue},
      QueueBinding{.lane = environment.lane(1), .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(first_destination, 4), std::array<float, 4>{1, 2, 3, 4});
  expect_near(read(second_destination, 4), std::array<float, 4>{5, 6, 7, 8});
}

TEST(GraphCompile, MakesWorkspaceSerializationAnExplicitBudgetTradeoff) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId first_input = builder.input(row(8)).value();
  const ValueId second_input = builder.input(row(8)).value();
  const ValueId first_middle =
      builder.add(Unary{UnaryKind::kNeg}, first_input).value();
  const ValueId second_middle =
      builder.add(Unary{UnaryKind::kNeg}, second_input).value();
  const ValueId first_output =
      builder.add(Unary{UnaryKind::kNeg}, first_middle).value();
  const ValueId second_output =
      builder.add(Unary{UnaryKind::kNeg}, second_middle).value();
  ASSERT_TRUE(builder.publish(first_output).ok());
  ASSERT_TRUE(builder.publish(second_output).ok());
  Graph graph = std::move(builder).finish().value();

  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const std::array targets{
      hyle::graph::ExecutionTarget{.context = &first,
                                   .default_storage =
                                       hyle::graph::StorageDomainId{0}},
      hyle::graph::ExecutionTarget{.context = &second,
                                   .default_storage =
                                       hyle::graph::StorageDomainId{0}},
  };
  const std::array lanes{
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{0},
                                 .role =
                                     hyle::graph::ExecutionLaneRole::kCompute},
      hyle::graph::ExecutionLane{.target = hyle::graph::ExecutionTargetId{1},
                                 .role =
                                     hyle::graph::ExecutionLaneRole::kCompute},
  };
  const std::array placements{
      NodePlacement{.node = graph.node_id(0).value(),
                    .target = hyle::graph::ExecutionTargetId{0},
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(1).value(),
                    .target = hyle::graph::ExecutionTargetId{1},
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(2).value(),
                    .target = hyle::graph::ExecutionTargetId{0},
                    .strength = PlacementStrength::kRequired},
      NodePlacement{.node = graph.node_id(3).value(),
                    .target = hyle::graph::ExecutionTargetId{1},
                    .strength = PlacementStrength::kRequired},
  };
  const std::array resources{
      ExternalResourceContract{.value = first_input,
                               .storage = hyle::graph::StorageDomainId{0}},
      ExternalResourceContract{.value = second_input,
                               .storage = hyle::graph::StorageDomainId{0}},
      ExternalResourceContract{.value = first_output,
                               .storage = hyle::graph::StorageDomainId{0}},
      ExternalResourceContract{.value = second_output,
                               .storage = hyle::graph::StorageDomainId{0}},
  };
  auto compile_with_budget = [&](std::size_t budget,
                                 hyle::graph::WorkspaceReusePolicy policy) {
    const std::array storage_domains{hyle::graph::StorageDomain{
        .resource = host, .workspace_budget_bytes = budget}};
    return hyle::graph::compile(graph,
                                hyle::graph::CompileEnvironment{
                                    .storage_domains = storage_domains,
                                    .targets = targets,
                                    .lanes = lanes,
                                },
                                CompileOptions{.node_placements = placements,
                                               .external_resources = resources,
                                               .workspace_reuse = policy});
  };

  auto concurrent = compile_with_budget(
      32, hyle::graph::WorkspaceReusePolicy::kExistingOrderOnly);
  ASSERT_FALSE(concurrent.has_value());
  EXPECT_TRUE(concurrent.error().IsResourceExhausted());

  auto impossible = compile_with_budget(
      31, hyle::graph::WorkspaceReusePolicy::kAllowSerialization);
  ASSERT_FALSE(impossible.has_value());
  EXPECT_TRUE(impossible.error().IsResourceExhausted());

  auto unconstrained = compile_with_budget(
      64, hyle::graph::WorkspaceReusePolicy::kAllowSerialization);
  ASSERT_TRUE(unconstrained.has_value()) << unconstrained.error().message();
  EXPECT_EQ(unconstrained->report.workspace_bytes(), 64u);
  EXPECT_EQ(unconstrained->report.serialization_edge_count(), 0u);
  EXPECT_EQ(unconstrained->report.segment_dependency_count(), 0u);

  auto serialized = compile_with_budget(
      32, hyle::graph::WorkspaceReusePolicy::kAllowSerialization);
  ASSERT_TRUE(serialized.has_value()) << serialized.error().message();
  EXPECT_EQ(serialized->report.workspace_bytes(), 32u);
  EXPECT_EQ(serialized->report.serialization_edge_count(), 1u);
  EXPECT_EQ(serialized->report.segment_dependency_count(), 1u);

  OwnedTensor first_source = make_tensor(row(8));
  OwnedTensor second_source = make_tensor(row(8));
  OwnedTensor first_destination = make_tensor(row(8));
  OwnedTensor second_destination = make_tensor(row(8));
  const std::array<float, 8> first_values{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<float, 8> second_values{9, 10, 11, 12, 13, 14, 15, 16};
  write(first_source, first_values);
  write(second_source, second_values);
  Invocation invocation = serialized->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(first_input, first_source.view).ok());
  ASSERT_TRUE(invocation.bind(second_input, second_source.view).ok());
  ASSERT_TRUE(invocation.bind(first_output, first_destination.view).ok());
  ASSERT_TRUE(invocation.bind(second_output, second_destination.view).ok());
  Queue first_queue = first.create_queue().value();
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      QueueBinding{.lane = hyle::graph::ExecutionLaneId{0},
                   .queue = &first_queue},
      QueueBinding{.lane = hyle::graph::ExecutionLaneId{1},
                   .queue = &second_queue},
  };
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(first_destination, 8), first_values);
  expect_near(read(second_destination, 8), second_values);
}

TEST(GraphCompile, BindsBorrowedConstantsOnceForRepeatedInvocations) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input_id = builder.input(row(4)).value();
  const auto weight_id = builder.constant(row(4)).value();
  const auto output_id = builder.add(RmsNorm{}, input_id, weight_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();

  OwnedTensor input = make_tensor(row(4));
  OwnedTensor weight = make_tensor(row(4));
  OwnedTensor output = make_tensor(row(4));
  const std::array<float, 4> input_values{1, -2, 3, -4};
  const std::array<float, 4> weight_values{1, 0.5F, -1, 2};
  write(input, input_values);
  write(weight, weight_values);
  const std::array bindings{ConstantBinding{weight_id, weight.view.as_const()}};
  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled =
      hyle::graph::compile(graph, context,
                           CompileOptions{.constants = bindings})
          .value();
  EXPECT_EQ(compiled.report.constant_count(), 1u);
  EXPECT_EQ(compiled.report.borrowed_constant_count(), 1u);
  EXPECT_EQ(compiled.report.packed_constant_count(), 0u);
  EXPECT_EQ(compiled.report.packed_constant_bytes(), 0u);

  Invocation invocation = compiled.executable.create_invocation().value();
  EXPECT_TRUE(invocation.bind(weight_id, weight.view).IsInvalidArgument());
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());

  float square_sum = 0.0F;
  for (float value : input_values)
    square_sum += value * value;
  const float factor = 1.0F / std::sqrt(square_sum / 4.0F + RmsNorm{}.epsilon);
  std::array<float, 4> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index)
    expected[index] = input_values[index] * factor * weight_values[index];
  expect_near(read(output, 4), expected);

  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.bind(input_id, input.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(GraphCompile, RejectsMissingForeignAndMismatchedConstantBindings) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input_id = builder.input(row(4)).value();
  const auto constant_id = builder.constant(row(4)).value();
  const auto output_id = builder.add(RmsNorm{}, input_id, constant_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();
  EXPECT_TRUE(hyle::graph::compile(graph, context).error().IsInvalidArgument());

  OwnedTensor correct = make_tensor(row(4));
  OwnedTensor wrong = make_tensor(row(3));
  const std::array wrong_shape{
      ConstantBinding{constant_id, wrong.view.as_const()}};
  EXPECT_TRUE(hyle::graph::compile(graph, context,
                                   CompileOptions{.constants = wrong_shape})
                  .error()
                  .IsInvalidArgument());

  GraphBuilder foreign_builder = GraphBuilder::create().value();
  const auto foreign = foreign_builder.constant(row(4)).value();
  ASSERT_TRUE(foreign_builder.publish(foreign).ok());
  Graph foreign_graph = std::move(foreign_builder).finish().value();
  static_cast<void>(foreign_graph);
  const std::array foreign_binding{
      ConstantBinding{foreign, correct.view.as_const()}};
  EXPECT_TRUE(hyle::graph::compile(graph, context,
                                   CompileOptions{.constants = foreign_binding})
                  .error()
                  .IsInvalidArgument());
}

TEST(GraphCompile, ExecutesPublishedConstantWithoutRuntimeRebinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto constant_id = builder.constant(row(4)).value();
  ASSERT_TRUE(builder.publish(constant_id).ok());
  Graph graph = std::move(builder).finish().value();
  OwnedTensor constant = make_tensor(row(4));
  const std::array binding{
      ConstantBinding{constant_id, constant.view.as_const()}};
  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled =
      hyle::graph::compile(graph, context, CompileOptions{.constants = binding})
          .value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kSourceAlias);
  EXPECT_EQ(compiled.executable.required_queue_count(), 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.submit()->wait().ok());
  EXPECT_TRUE(invocation.output(constant_id).has_value());
  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.submit()->wait().ok());
}

TEST(GraphCompile, APublishedInputAcceptsTheNaturalMutableInputBinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(row(4)).value();
  ASSERT_TRUE(builder.publish(input).ok());
  Graph graph = std::move(builder).finish().value();
  Context context = hyle::compute::cpu::create_context().value();
  auto compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kSourceAlias);
  EXPECT_EQ(compiled.executable.required_queue_count(), 0u);

  auto invocation = compiled.executable.create_invocation().value();
  OwnedTensor storage = make_tensor(row(4));
  ASSERT_TRUE(invocation.bind(input, storage.view).ok());
  ASSERT_TRUE(invocation.submit()->wait().ok());
  EXPECT_TRUE(invocation.output(input).has_value());
}

TEST(Invocation, RunsNaturalTransformerFragmentAndReusesIt) {
  constexpr std::size_t kRows = 2;
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kFfn = 4;
  const LogicalGraph logical = make_transformer_graph(kRows, kHidden, kFfn);
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();

  OwnedTensor input = make_tensor(matrix(kRows, kHidden));
  OwnedTensor norm_weight = make_tensor(row(kHidden));
  const TensorDesc stored_weight = matrix(2 * kFfn, kHidden);
  OwnedTensor matrix_weight = make_tensor(stored_weight.transposed().value());
  OwnedTensor output = make_tensor(matrix(kRows, kFfn));
  const std::vector<float> input_values{1, -2, 3, -4, 2, 1, -1, 0.5F};
  const std::vector<float> norm_values{1, 0.5F, 1.5F, -1};
  std::vector<float> matrix_values(2 * kFfn * kHidden);
  for (std::size_t index = 0; index < matrix_values.size(); ++index) {
    matrix_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3) * 0.125F;
  }
  write(input, input_values);
  write(norm_weight, norm_values);
  write(matrix_weight, matrix_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());

  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  ASSERT_TRUE(completion->wait().ok());
  const std::vector<float> expected = expected_transformer(
      input_values, norm_values, matrix_values, kRows, kHidden, kFfn);
  expect_near(read(output, kRows * kFfn), expected);

  ASSERT_TRUE(invocation.reset().ok());
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], matrix_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
}

TEST(Invocation, RunsBroadcastMatMulIntoSeparateGatedActivation) {
  constexpr std::array<std::int64_t, 4> left_shape{2, 1, 1, 2};
  constexpr std::array<std::int64_t, 4> right_shape{1, 2, 2, 4};
  constexpr std::array<std::int64_t, 4> output_shape{2, 2, 1, 4};
  const TensorDesc left_desc =
      TensorDesc::dense(DataType::kFloat32, left_shape).value();
  const TensorDesc right_desc =
      TensorDesc::dense(DataType::kFloat32, right_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();

  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId left_id = builder.input(left_desc).value();
  const ValueId right_id = builder.input(right_desc).value();
  const ValueId value_id = builder.input(output_desc).value();
  const ValueId product_id = builder.add(MatMul{}, left_id, right_id).value();
  const ValueId output_id =
      builder
          .add(GatedActivation{
                   .activation = GatedActivationKind::kGeluTanh,
                   .inputs = GatedInputKind::kSeparate},
               product_id, value_id)
          .value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  Graph graph = std::move(builder).finish().value();

  const std::array<float, 4> left_values{1, 2, 3, -1};
  const std::array<float, 16> right_values{
      1, 0, -1, 2, 0.5F, 2, 1, -1,
      2, 1, 0, -1, 1, -2, 3, 0.5F,
  };
  std::array<float, 16> value_values{};
  for (std::size_t index = 0; index < value_values.size(); ++index)
    value_values[index] = 0.25F * static_cast<float>(index + 1);

  std::array<float, 16> expected{};
  for (std::size_t left_batch = 0; left_batch < 2; ++left_batch) {
    for (std::size_t right_batch = 0; right_batch < 2; ++right_batch) {
      for (std::size_t column = 0; column < 4; ++column) {
        float product = 0.0F;
        for (std::size_t inner = 0; inner < 2; ++inner) {
          product += left_values[left_batch * 2 + inner] *
                     right_values[(right_batch * 2 + inner) * 4 + column];
        }
        const std::size_t index = (left_batch * 2 + right_batch) * 4 + column;
        const float activated =
            0.5F * product *
            (1.0F + std::tanh(0.79788456080286535588F * product *
                              std::fma(0.044715F, product * product, 1.0F)));
        expected[index] = activated * value_values[index];
      }
    }
  }

  OwnedTensor left = make_tensor(left_desc);
  OwnedTensor right = make_tensor(right_desc);
  OwnedTensor value = make_tensor(output_desc);
  OwnedTensor output = make_tensor(output_desc);
  write(left, left_values);
  write(right, right_values);
  write(value, value_values);

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(graph, context)
                              ->executable.create_invocation()
                              .value();
  ASSERT_TRUE(invocation.bind(left_id, left.view).ok());
  ASSERT_TRUE(invocation.bind(right_id, right.view).ok());
  ASSERT_TRUE(invocation.bind(value_id, value.view).ok());
  ASSERT_TRUE(invocation.bind(output_id, output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output, expected.size()), expected);
}

TEST(Invocation, ExecutesSharedInputBranchesWithIndependentOutputs) {
  const LogicalGraph logical = make_branch_graph();
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor input = make_tensor(matrix(2, 3));
  OwnedTensor first_weight = make_tensor(matrix(3, 2));
  OwnedTensor second_weight = make_tensor(matrix(3, 1));
  OwnedTensor first_output = make_tensor(matrix(2, 2));
  OwnedTensor second_output = make_tensor(matrix(2, 1));
  const std::vector<float> input_values{1, 2, 3, 4, 5, 6};
  const std::vector<float> first_values{1, 0, 0, 1, 1, 1};
  const std::vector<float> second_values{1, 2, 3};
  write(input, input_values);
  write(first_weight, first_values);
  write(second_weight, second_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], first_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], second_weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], first_output.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[1], second_output.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(first_output, 4), std::array{4.0F, 5.0F, 10.0F, 11.0F});
  expect_near(read(second_output, 2), std::array{14.0F, 32.0F});
}

TEST(Invocation, RejectsForeignInternalConstOutputAndWrongShapeBindings) {
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  const LogicalGraph other = make_rms_graph(matrix(2, 4));
  Context context = hyle::compute::cpu::create_context().value();
  Invocation invocation = hyle::graph::compile(logical.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor activation = make_tensor(matrix(2, 4));
  OwnedTensor wrong = make_tensor(matrix(1, 4));
  OwnedTensor output = make_tensor(matrix(2, 4));

  EXPECT_TRUE(
      invocation.bind(other.inputs[0], activation.view).IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(logical.intermediates[0], activation.view)
                  .IsInvalidArgument());
  EXPECT_TRUE(invocation.bind(logical.outputs[0], output.view.as_const())
                  .IsInvalidArgument());
  EXPECT_TRUE(
      invocation.bind(logical.inputs[0], wrong.view).IsInvalidArgument());
  Queue queue = context.create_queue().value();
  EXPECT_TRUE(invocation.submit(queue).error().IsFailedPrecondition());
}

TEST(GraphCompile, ReportsInvalidClosedAndMetadataAllocationFailures) {
  Context context = hyle::compute::cpu::create_context().value();
  EXPECT_TRUE(
      hyle::graph::compile(Graph{}, context).error().IsInvalidArgument());

  const LogicalGraph logical = make_rms_graph(row(8));
  ASSERT_TRUE(context.close().ok());
  EXPECT_TRUE(hyle::graph::compile(logical.graph, context)
                  .error()
                  .IsFailedPrecondition());

  AllocatorProbe probe;
  Context oom_context = hyle::compute::cpu::create_context(
                            {.metadata_allocator = probe_allocator(probe)})
                            .value();
  probe.fail = true;
  auto oom = hyle::graph::compile(logical.graph, oom_context);
  ASSERT_FALSE(oom.has_value());
  EXPECT_TRUE(oom.error().IsResourceExhausted());

  probe.fail = false;
  Executable executable =
      hyle::graph::compile(logical.graph, oom_context)->executable;
  probe.fail = true;
  auto invocation = executable.create_invocation();
  ASSERT_FALSE(invocation.has_value());
  EXPECT_TRUE(invocation.error().IsResourceExhausted());
}

TEST(GraphCompile, DefersEncodedWeightExecutionToAFutureProvider) {
  constexpr StorageFormatId codec{0x48594c45U, 43U};
  const StorageFormat blocked =
      StorageFormat::blocked(codec, 32, 18, 2).value();
  const std::array<std::int64_t, 2> stored_shape{8, 64};
  const TensorDesc weight =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, blocked)
          ->transposed()
          .value();

  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(matrix(2, 64)).value();
  const ValueId encoded_weight = builder.input(weight).value();
  const ValueId output = builder.add(MatMul{}, input, encoded_weight).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  auto compiled = hyle::graph::compile(graph, context);
  ASSERT_FALSE(compiled.has_value());
  EXPECT_TRUE(compiled.error().IsUnimplemented());
}

TEST(GraphCompile, EveryMetadataAllocationFailureRollsBackAndCanRetry) {
  AllocatorProbe probe;
  Context context = hyle::compute::cpu::create_context(
                        {.metadata_allocator = probe_allocator(probe)})
                        .value();
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);

  probe.attempts = 0;
  {
    auto successful = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(successful.has_value());
  }
  const std::size_t compile_attempts = probe.attempts.load();
  ASSERT_GT(compile_attempts, 1u);

  for (std::size_t failure = 0; failure < compile_attempts; ++failure) {
    probe.attempts = 0;
    probe.fail_on_attempt = failure;
    auto failed = hyle::graph::compile(logical.graph, context);
    ASSERT_FALSE(failed.has_value()) << "allocation attempt " << failure;
    EXPECT_TRUE(failed.error().IsResourceExhausted())
        << "allocation attempt " << failure;

    probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
    probe.attempts = 0;
    auto retried = hyle::graph::compile(logical.graph, context);
    ASSERT_TRUE(retried.has_value()) << "allocation attempt " << failure;
  }
}

TEST(Invocation, SubmitDoesNotAllocatePlanMetadata) {
  AllocatorProbe probe;
  Context context = hyle::compute::cpu::create_context(
                        {.metadata_allocator = probe_allocator(probe)})
                        .value();
  const LogicalGraph logical = make_transformer_graph(2, 4, 4);
  Executable executable =
      hyle::graph::compile(logical.graph, context)->executable;
  Invocation invocation = executable.create_invocation().value();
  OwnedTensor input = make_tensor(matrix(2, 4));
  OwnedTensor norm = make_tensor(row(4));
  OwnedTensor weights = make_tensor(matrix(8, 4).transposed().value());
  OwnedTensor output = make_tensor(matrix(2, 4));
  std::vector<float> input_values(8, 1.0F);
  std::vector<float> norm_values(4, 1.0F);
  std::vector<float> weight_values(32, 0.25F);
  write(input, input_values);
  write(norm, norm_values);
  write(weights, weight_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], weights.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  Queue queue = context.create_queue().value();
  const std::size_t before = probe.allocations.load();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(probe.allocations.load(), before);
}

TEST(Invocation, DistinctInvocationsRunConcurrentlyFromOneExecutable) {
  constexpr std::size_t kCount = 256;
  const LogicalGraph logical = make_rms_graph(row(kCount));
  Context context = hyle::compute::cpu::create_context().value();
  Executable executable =
      hyle::graph::compile(logical.graph, context)->executable;
  std::atomic<int> failures{0};

  auto worker = [&](float value) {
    Queue queue = context.create_queue().value();
    Invocation invocation = executable.create_invocation().value();
    OwnedTensor input = make_tensor(row(kCount));
    OwnedTensor output = make_tensor(row(kCount));
    std::vector<float> values(kCount, value);
    write(input, values);
    if (!invocation.bind(logical.inputs[0], input.view).ok() ||
        !invocation.bind(logical.outputs[0], output.view).ok()) {
      ++failures;
      return;
    }
    auto completion = invocation.submit(queue);
    if (!completion || !completion->wait().ok())
      ++failures;
  };

  std::thread first(worker, 1.0F);
  std::thread second(worker, 2.0F);
  first.join();
  second.join();
  EXPECT_EQ(failures.load(), 0);
}

TEST(Invocation, PlanAndRunStateRetainTheirContextOwnedArtifacts) {
  AllocatorProbe probe;
  Executable retained;
  {
    Context context = hyle::compute::cpu::create_context(
                          {.metadata_allocator = probe_allocator(probe)})
                          .value();
    const LogicalGraph logical = make_rms_graph(row(8));
    retained = hyle::graph::compile(logical.graph, context)->executable;
  }
  EXPECT_TRUE(retained.valid());
  Invocation invocation = retained.create_invocation().value();
  retained = Executable{};
  EXPECT_TRUE(invocation.valid());
}

TEST(Invocation, ZeroExtentIntermediateWorkspaceRemainsAValidRun) {
  const LogicalGraph logical = make_transformer_graph(0, 4, 4);
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  CompiledGraph compiled = hyle::graph::compile(logical.graph, context).value();
  EXPECT_EQ(compiled.report.node_count(), 3u);

  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input = make_tensor(matrix(0, 4));
  OwnedTensor norm = make_tensor(row(4));
  OwnedTensor weight = make_tensor(matrix(8, 4).transposed().value());
  OwnedTensor output = make_tensor(matrix(0, 4));
  const std::array<float, 4> norm_values{1, 1, 1, 1};
  const std::array<float, 32> weight_values{};
  write(norm, norm_values);
  write(weight, weight_values);
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[1], norm.view).ok());
  ASSERT_TRUE(invocation.bind(logical.inputs[2], weight.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().ok());
}

TEST(Invocation, PublishedIntermediateCanFeedALaterNode) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId first = builder.add(RmsNorm{}, input).value();
  const ValueId second = builder.add(RmsNorm{}, first).value();
  ASSERT_TRUE(builder.publish(first).ok());
  ASSERT_TRUE(builder.publish(second).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.workspace_bytes(), 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input_tensor = make_tensor(description);
  OwnedTensor first_tensor = make_tensor(description);
  OwnedTensor second_tensor = make_tensor(description);
  const std::array<float, 4> values{1, 2, 3, 4};
  write(input_tensor, values);
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(first, first_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(second, second_tensor.view).ok());
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  EXPECT_TRUE(completion->wait().ok());
}

TEST(Invocation, AllowsOnlyLifetimeSafeExactExternalInPlaceAlias) {
  const TensorDesc description = row(4);
  const LogicalGraph safe = make_rms_graph(description);
  Context context = hyle::compute::cpu::create_context().value();
  Queue queue = context.create_queue().value();
  Invocation invocation = hyle::graph::compile(safe.graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor shared = make_tensor(description);
  const std::array<float, 4> values{1, 2, 3, 4};
  write(shared, values);
  ASSERT_TRUE(invocation.bind(safe.inputs[0], shared.view).ok());
  ASSERT_TRUE(invocation.bind(safe.outputs[0], shared.view).ok());
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());

  Buffer partial_storage =
      Buffer::create(hyle::mem::system_resource(), 5 * sizeof(float), 64)
          .value();
  TensorView partial_input =
      TensorView::create(
          description,
          partial_storage.view().subview(0, 4 * sizeof(float)).value())
          .value();
  TensorView partial_output =
      TensorView::create(description,
                         partial_storage.view()
                             .subview(sizeof(float), 4 * sizeof(float))
                             .value())
          .value();
  ASSERT_TRUE(invocation.reset().ok());
  ASSERT_TRUE(invocation.bind(safe.inputs[0], partial_input).ok());
  EXPECT_TRUE(
      invocation.bind(safe.outputs[0], partial_output).IsInvalidArgument());
}

TEST(GraphCompile, ReusesInternalStorageForLifetimeSafeInPlaceOperations) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId negated = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const ValueId magnitude =
      builder.add(Unary{UnaryKind::kAbs}, negated).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, magnitude).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.workspace_bytes(),
            description.required_bytes().value());
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input_tensor = make_tensor(description);
  OwnedTensor output_tensor = make_tensor(description);
  const std::array<float, 4> values{-1, 2, -3, 4};
  write(input_tensor, values);
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 4), std::array<float, 4>{-1, -2, -3, -4});
}

TEST(Invocation, ResolvesAnInternalSliceViewWithoutMaterializingIt) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = matrix(2, 3);
  const ValueId input = builder.input(input_description).value();
  const ValueId slice =
      builder.add(View::slice(0, 1, 1).value(), input).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, slice).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.report.workspace_bytes(), 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor input_tensor = make_tensor(input_description);
  OwnedTensor output_tensor = make_tensor(matrix(1, 3));
  const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  write(input_tensor, values);
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 3), std::array<float, 3>{-4, -5, -6});
}

TEST(Invocation, OwnsAPublishedViewRootedInPrivateWorkspace) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = row(6);
  const ValueId input = builder.input(input_description).value();
  const ValueId internal = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const ValueId output =
      builder.add(View::slice(0, 2, 3).value(), internal).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  auto compile_result = hyle::graph::compile(graph, context);
  ASSERT_TRUE(compile_result.has_value()) << compile_result.error().message();
  CompiledGraph compiled = std::move(*compile_result);
  auto ownership = compiled.executable.output_ownership();
  ASSERT_TRUE(ownership.has_value()) << ownership.error().message();
  EXPECT_EQ(*ownership, OutputOwnership::kInvocation);
  EXPECT_EQ(compiled.report.invocation_owned_output_count(), 1u);
  EXPECT_EQ(compiled.report.workspace_bytes(),
            input_description.required_bytes().value());

  auto created = compiled.executable.create_invocation();
  ASSERT_TRUE(created.has_value()) << created.error().message();
  Invocation invocation = std::move(*created);
  EXPECT_TRUE(invocation.output(output).error().IsFailedPrecondition());
  OwnedTensor input_tensor = make_tensor(input_description);
  OwnedTensor rejected_output = make_tensor(row(3));
  write(input_tensor, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  EXPECT_TRUE(
      invocation.bind(output, rejected_output.view).IsFailedPrecondition());
  Queue queue = context.create_queue().value();
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value()) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  auto owned = invocation.output(output);
  ASSERT_TRUE(owned.has_value()) << owned.error().message();
  EXPECT_EQ(read(*owned, 3), (std::vector<float>{-3, -4, -5}));

  ASSERT_TRUE(invocation.reset().ok());
  EXPECT_TRUE(invocation.output(output).error().IsFailedPrecondition());
}

TEST(Invocation, PublishedViewReusesItsSourceBinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = matrix(2, 3);
  const ValueId input = builder.input(input_description).value();
  const ValueId output =
      builder.add(View::slice(0, 1, 1).value(), input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kSourceAlias);
  ASSERT_EQ(compiled.report.step_count(), 1u);
  EXPECT_EQ(compiled.report.segment_count(), 0u);
  EXPECT_EQ(compiled.report.command_segment_count(), 0u);
  EXPECT_EQ(compiled.executable.required_queue_count(), 0u);
  EXPECT_FALSE(
      compiled.executable.requires_queue(hyle::graph::ExecutionLaneId{0})
          .value());
  EXPECT_EQ(compiled.report.step(0)->dispatch_count, 0u);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor storage = make_tensor(input_description);
  write(storage, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  const std::size_t row_bytes = 3 * sizeof(float);
  TensorView second_row =
      TensorView::create(
          matrix(1, 3),
          storage.buffer.view().subview(row_bytes, row_bytes).value())
          .value();
  ASSERT_TRUE(invocation.bind(input, storage.view).ok());
  EXPECT_TRUE(invocation.bind(output, second_row).IsFailedPrecondition());
  ASSERT_TRUE(invocation.submit()->wait().ok());
  auto derived = invocation.output(output);
  ASSERT_TRUE(derived.has_value()) << derived.error().message();
  EXPECT_EQ(read(*derived, 3), (std::vector<float>{4, 5, 6}));

  ASSERT_TRUE(invocation.reset().ok());
  TensorView wrong =
      TensorView::create(matrix(1, 3),
                         storage.buffer.view().subview(0, row_bytes).value())
          .value();
  ASSERT_TRUE(invocation.bind(input, storage.view).ok());
  EXPECT_TRUE(invocation.bind(output, wrong).IsFailedPrecondition());
}

TEST(Invocation, RequiresQueuesOnlyForLanesThatEmitRuntimeCommands) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context unused = hyle::compute::cpu::create_context().value();
  Context active = hyle::compute::cpu::create_context().value();
  const auto host = hyle::mem::system_resource();
  const Environment environment{{
      ExecutionSpec{.context = &unused, .storage = host},
      ExecutionSpec{.context = &active, .storage = host},
  }};
  const std::array placements{NodePlacement{
      .node = graph.node_id().value(),
      .target = environment.target(1),
      .strength = PlacementStrength::kRequired,
  }};
  const std::array resources{
      ExternalResourceContract{.value = input,
                               .storage = environment.storage(1)},
      ExternalResourceContract{.value = output,
                               .storage = environment.storage(1)},
  };
  CompiledGraph compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();

  ASSERT_EQ(compiled.report.lane_count(), 2u);
  EXPECT_FALSE(compiled.report.lane(0)->queue_required);
  EXPECT_EQ(compiled.report.lane(0)->completion_slots_per_invocation, 0u);
  EXPECT_TRUE(compiled.report.lane(1)->queue_required);
  EXPECT_EQ(compiled.report.lane(1)->completion_slots_per_invocation, 1u);
  EXPECT_EQ(compiled.executable.required_queue_count(), 1u);
  EXPECT_FALSE(compiled.executable.requires_queue(environment.lane(0)).value());
  EXPECT_TRUE(compiled.executable.requires_queue(environment.lane(1)).value());
  EXPECT_TRUE(
      compiled.executable.requires_queue(hyle::graph::ExecutionLaneId{2})
          .error()
          .IsOutOfRange());

  OwnedTensor source = make_tensor(description);
  OwnedTensor destination = make_tensor(description);
  write(source, std::array<float, 4>{1, 2, 3, 4});
  Invocation invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input, source.view).ok());
  ASSERT_TRUE(invocation.bind(output, destination.view).ok());
  Queue unused_queue = unused.create_queue().value();
  const std::array wrong_binding{
      QueueBinding{.lane = environment.lane(0), .queue = &unused_queue}};
  EXPECT_TRUE(invocation.prepare_replay(wrong_binding).IsInvalidArgument());

  Queue active_queue = active.create_queue().value();
  ASSERT_TRUE(invocation.submit(active_queue)->wait().ok());
  expect_near(read(destination, 4), std::array<float, 4>{-1, -2, -3, -4});
}

TEST(Invocation, CrossStoragePublishedViewUsesItsExplicitOutputBinding) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc input_description = matrix(2, 3);
  const ValueId input = builder.input(input_description).value();
  const ValueId output =
      builder.add(View::slice(0, 1, 1).value(), input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  HostResource first_storage;
  HostResource second_storage;
  const Environment environment{{
      ExecutionSpec{.context = &first,
                    .storage = hyle::mem::MemoryResourceRef(first_storage)},
      ExecutionSpec{.context = &second,
                    .storage = hyle::mem::MemoryResourceRef(second_storage)},
  }};
  const std::array placements{NodePlacement{
      .node = graph.node_id(0).value(),
      .target = environment.target(0),
      .strength = PlacementStrength::kRequired,
  }};
  const std::array resources{ExternalResourceContract{
      .value = output, .storage = environment.storage(1)}};
  auto compiled =
      hyle::graph::compile(graph, environment.environment(),
                           CompileOptions{.node_placements = placements,
                                          .external_resources = resources})
          .value();
  EXPECT_EQ(compiled.executable.output_ownership().value(),
            OutputOwnership::kCaller);
  ASSERT_EQ(compiled.report.step_count(), 2u);
  ASSERT_EQ(compiled.executable.required_queue_count(), 1u);
  EXPECT_FALSE(compiled.report.lane(0)->queue_required);
  EXPECT_TRUE(compiled.report.lane(1)->queue_required);
  EXPECT_EQ(compiled.report.step(1)->kind, hyle::graph::CompileStepKind::kCopy);
  EXPECT_EQ(compiled.report.step(1)->synchronization_count, 0u);
  EXPECT_EQ(compiled.report.synchronization_count(), 0u);

  OwnedTensor source = make_tensor(input_description,
                                   hyle::mem::MemoryResourceRef(first_storage));
  OwnedTensor destination =
      make_tensor(matrix(1, 3), hyle::mem::MemoryResourceRef(second_storage));
  write(source, std::array<float, 6>{1, 2, 3, 4, 5, 6});
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input, source.view).ok());
  ASSERT_TRUE(invocation.bind(output, destination.view).ok());
  Queue second_queue = second.create_queue().value();
  const std::array queues{
      QueueBinding{.lane = environment.lane(1), .queue = &second_queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  ASSERT_TRUE(invocation.submit(queues)->wait().ok());
  expect_near(read(destination, 3), std::array<float, 3>{4, 5, 6});
}

TEST(GraphCompile, DoesNotInPlaceABroadcastProviderRoute) {
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId input = builder.input(row(4)).value();
  const ValueId scalar = builder.input(row(1)).value();
  const ValueId first = builder.add(Unary{UnaryKind::kNeg}, input).value();
  const ValueId sum =
      builder.add(Binary{BinaryKind::kAdd}, first, scalar).value();
  const ValueId output = builder.add(Unary{UnaryKind::kNeg}, sum).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  Invocation invocation = hyle::graph::compile(graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor input_tensor = make_tensor(row(4));
  OwnedTensor scalar_tensor = make_tensor(row(1));
  OwnedTensor output_tensor = make_tensor(row(4));
  write(input_tensor, std::array<float, 4>{1, 2, 3, 4});
  write(scalar_tensor, std::array<float, 1>{10});
  ASSERT_TRUE(invocation.bind(input, input_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(scalar, scalar_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 4), std::array<float, 4>{-9, -8, -7, -6});
}

TEST(Invocation, RejectsExternalAliasWhileTheInputRemainsLive) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc description = row(4);
  const ValueId input = builder.input(description).value();
  const ValueId early = builder.add(RmsNorm{}, input).value();
  const ValueId late = builder.add(RmsNorm{}, input).value();
  ASSERT_TRUE(builder.publish(early).ok());
  ASSERT_TRUE(builder.publish(late).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  Invocation invocation = hyle::graph::compile(graph, context)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor shared = make_tensor(description);
  ASSERT_TRUE(invocation.bind(input, shared.view).ok());
  EXPECT_TRUE(invocation.bind(early, shared.view).IsInvalidArgument());
}

TEST(Invocation, RejectsAQueueFromAnotherContext) {
  const LogicalGraph logical = make_rms_graph(row(4));
  Context first = hyle::compute::cpu::create_context().value();
  Context second = hyle::compute::cpu::create_context().value();
  Queue wrong_queue = second.create_queue().value();
  Invocation invocation = hyle::graph::compile(logical.graph, first)
                              ->executable.create_invocation()
                              .value();
  OwnedTensor input = make_tensor(row(4));
  OwnedTensor output = make_tensor(row(4));
  ASSERT_TRUE(invocation.bind(logical.inputs[0], input.view).ok());
  ASSERT_TRUE(invocation.bind(logical.outputs[0], output.view).ok());
  EXPECT_TRUE(invocation.submit(wrong_queue).error().IsInvalidArgument());
}

TEST(GraphCompile, ExecutesNativeAttentionWithPlannedInvocationWorkspace) {
  constexpr std::array<std::int64_t, 4> query_shape{1, 1, 1, 1};
  constexpr std::array<std::int64_t, 4> key_shape{1, 1, 3, 1};
  const TensorDesc query_desc =
      TensorDesc::dense(DataType::kFloat32, query_shape).value();
  const TensorDesc key_desc =
      TensorDesc::dense(DataType::kFloat32, key_shape).value();
  GraphBuilder builder = GraphBuilder::create().value();
  const ValueId query = builder.input(query_desc).value();
  const ValueId key = builder.input(key_desc).value();
  const ValueId value = builder.input(key_desc).value();
  const ValueId output =
      builder
          .add(
              ScaledDotProductAttention{
                  .layout = AttentionLayout::kBatchHeadSequenceDepth,
                  .causal = true},
              query, key, value)
          .value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();

  Context context = hyle::compute::cpu::create_context().value();
  CompiledGraph compiled = hyle::graph::compile(graph, context).value();
  EXPECT_GT(compiled.report.workspace_bytes(), 0U);
  Invocation invocation = compiled.executable.create_invocation().value();
  OwnedTensor query_tensor = make_tensor(query_desc);
  OwnedTensor key_tensor = make_tensor(key_desc);
  OwnedTensor value_tensor = make_tensor(key_desc);
  OwnedTensor output_tensor = make_tensor(query_desc);
  write(query_tensor, std::array<float, 1>{0});
  write(key_tensor, std::array<float, 3>{0, 0, 0});
  write(value_tensor, std::array<float, 3>{1, 2, 6});
  ASSERT_TRUE(invocation.bind(query, query_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(key, key_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(value, value_tensor.view).ok());
  ASSERT_TRUE(invocation.bind(output, output_tensor.view).ok());
  Queue queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output_tensor, 1), std::array<float, 1>{3});
}

} // namespace
