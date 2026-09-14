#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::GatedActivation;
using hyle::compute::GatedActivationKind;
using hyle::compute::GatedInputKind;
using hyle::compute::MatMul;
using hyle::compute::ParallelExecutorRef;
using hyle::compute::Provider;
using hyle::compute::RmsNorm;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::graph::CompileOptions;
using hyle::graph::CompileFallbackReason;
using hyle::graph::CompileStepKind;
using hyle::graph::ConstantBinding;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::RegionSelectionPolicy;
using hyle::graph::ValueId;
using hyle::mem::Buffer;

TensorDesc row(std::int64_t size) {
  const std::array dimensions{size};
  return TensorDesc::dense(DataType::kFloat32, dimensions).value();
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array dimensions{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, dimensions).value();
}

struct OwnedTensor final {
  Buffer buffer{};
  TensorView view{};
};

OwnedTensor make_tensor(TensorDesc description) {
  OwnedTensor result;
  result.buffer = Buffer::create(hyle::mem::system_resource(),
                                 description.required_bytes().value(), 64)
                      .value();
  result.view = TensorView::create(description, result.buffer.view()).value();
  return result;
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

struct GatedMlp final {
  Graph graph{};
  ValueId activation{};
  ValueId norm_weight{};
  ValueId normalized{};
  ValueId gate_weight{};
  ValueId up_weight{};
  ValueId down_weight{};
  ValueId gate{};
  ValueId output{};
};

enum class MlpOrder { kCanonical, kInterleaved, kUpFirst, kControlReentry };

GatedMlp make_gated_mlp(std::size_t rows, std::size_t inner, std::size_t hidden,
                        bool publish_gate = false, bool with_rms_norm = false,
                        bool consume_gate = false,
                        MlpOrder order = MlpOrder::kCanonical) {
  GraphBuilder builder = GraphBuilder::create().value();
  GatedMlp result;
  result.activation = builder.input(matrix(rows, inner)).value();
  ValueId projected_input = result.activation;
  if (with_rms_norm) {
    result.norm_weight = builder.constant(row(inner)).value();
    result.normalized =
        builder.add(RmsNorm{}, result.activation, result.norm_weight).value();
    projected_input = result.normalized;
  }
  result.gate_weight = builder.constant(matrix(inner, hidden)).value();
  result.up_weight = builder.constant(matrix(inner, hidden)).value();
  result.down_weight = builder.constant(matrix(hidden, inner)).value();
  ValueId up;
  if (order == MlpOrder::kUpFirst)
    up = builder.add(MatMul{}, projected_input, result.up_weight).value();
  const auto gate_node =
      builder
          .add_node(hyle::compute::Operation(MatMul{}),
                    std::array{projected_input, result.gate_weight})
          .value();
  result.gate = builder.result(gate_node).value();
  hyle::graph::NodeId side_node;
  if (order != MlpOrder::kCanonical) {
    side_node = builder
                    .add_node(hyle::compute::Operation(RmsNorm{}),
                              std::array{projected_input})
                    .value();
    if (order == MlpOrder::kControlReentry)
      EXPECT_TRUE(builder.depends_on(side_node, gate_node).ok());
  }
  if (order != MlpOrder::kUpFirst)
    up = builder.add(MatMul{}, projected_input, result.up_weight).value();
  const auto gated_node =
      builder
          .add_node(hyle::compute::Operation(GatedActivation{
                        .activation = GatedActivationKind::kSilu,
                        .inputs = GatedInputKind::kSeparate}),
                    std::array{result.gate, up})
          .value();
  if (order == MlpOrder::kControlReentry)
    EXPECT_TRUE(builder.depends_on(gated_node, side_node).ok());
  const ValueId gated = builder.result(gated_node).value();
  result.output = builder.add(MatMul{}, gated, result.down_weight).value();
  if (consume_gate) {
    const ValueId diagnostic = builder.add(RmsNorm{}, result.gate).value();
    EXPECT_TRUE(builder.publish(diagnostic).ok());
  }
  if (publish_gate)
    EXPECT_TRUE(builder.publish(result.gate).ok());
  EXPECT_TRUE(builder.publish(result.output).ok());
  result.graph = std::move(builder).finish().value();
  return result;
}

std::vector<float> matmul(std::span<const float> left,
                          std::span<const float> right, std::size_t rows,
                          std::size_t inner, std::size_t columns) {
  std::vector<float> output(rows * columns, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t k = 0; k < inner; ++k) {
        output[row * columns + column] +=
            left[row * inner + k] * right[k * columns + column];
      }
    }
  }
  return output;
}

std::vector<float>
reference(std::span<const float> activation, std::span<const float> gate_weight,
          std::span<const float> up_weight, std::span<const float> down_weight,
          std::size_t rows, std::size_t inner, std::size_t hidden) {
  std::vector<float> gate =
      matmul(activation, gate_weight, rows, inner, hidden);
  const std::vector<float> up =
      matmul(activation, up_weight, rows, inner, hidden);
  for (std::size_t index = 0; index < gate.size(); ++index)
    gate[index] = gate[index] / (1.0F + std::exp(-gate[index])) * up[index];
  return matmul(gate, down_weight, rows, hidden, inner);
}

std::vector<float> rms_norm(std::span<const float> activation,
                            std::span<const float> weight, std::size_t rows,
                            std::size_t inner) {
  std::vector<float> output(activation.size());
  for (std::size_t row_index = 0; row_index < rows; ++row_index) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < inner; ++column) {
      const float value = activation[row_index * inner + column];
      sum += value * value;
    }
    const float scale =
        1.0F / std::sqrt(sum / static_cast<float>(inner) + RmsNorm{}.epsilon);
    for (std::size_t column = 0; column < inner; ++column) {
      output[row_index * inner + column] =
          activation[row_index * inner + column] * scale * weight[column];
    }
  }
  return output;
}

std::array<Provider, 3> provider_sequence() {
  return {hyle::graph::onednn::provider(), hyle::compute::onednn::provider(),
          hyle::compute::native::provider()};
}

struct ParallelProbe final {
  std::atomic<std::size_t> episodes{0};
  std::atomic<std::size_t> tasks{0};
};

void probe_parallel_for(void *opaque, std::size_t count,
                        hyle::compute::ParallelTaskFn task,
                        void *task_context) noexcept {
  auto &probe = *static_cast<ParallelProbe *>(opaque);
  ++probe.episodes;
  probe.tasks += count;
  for (std::size_t index = 0; index < count; ++index)
    task(task_context, index, count);
}

bool probe_in_parallel(void *) noexcept { return true; }

struct AllocatorProbe final {
  hyle::mem::AllocatorRef backing = hyle::mem::system_allocator();
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> fail_on_attempt{
      (std::numeric_limits<std::size_t>::max)()};
};

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  const std::size_t attempt = probe.attempts.fetch_add(1);
  if (attempt == probe.fail_on_attempt.load())
    return nullptr;
  return probe.backing.allocate(size, alignment);
}

void probe_free(void *opaque, void *pointer, std::size_t size,
                std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *opaque, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  const std::size_t attempt = probe.attempts.fetch_add(1);
  if (attempt == probe.fail_on_attempt.load())
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

hyle::mem::AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return {&probe_allocate, &probe_free, &probe_reallocate, &probe};
}

void expect_near(std::span<const float> actual,
                 std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 8.0e-5F + 8.0e-6F)
        << "index " << index;
  }
}

void run_gated_mlp_shape(std::size_t rows, std::size_t inner,
                         std::size_t hidden,
                         MlpOrder order = MlpOrder::kCanonical) {
  const GatedMlp logical =
      make_gated_mlp(rows, inner, hidden, false, false, false, order);
  OwnedTensor activation = make_tensor(matrix(rows, inner));
  OwnedTensor gate_weight = make_tensor(matrix(inner, hidden));
  OwnedTensor up_weight = make_tensor(matrix(inner, hidden));
  OwnedTensor down_weight = make_tensor(matrix(hidden, inner));
  OwnedTensor output = make_tensor(matrix(rows, inner));
  std::vector<float> activation_values(rows * inner);
  std::vector<float> gate_values(inner * hidden);
  std::vector<float> up_values(inner * hidden);
  std::vector<float> down_values(hidden * inner);
  for (std::size_t index = 0; index < activation_values.size(); ++index)
    activation_values[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125F;
  for (std::size_t index = 0; index < gate_values.size(); ++index)
    gate_values[index] =
        static_cast<float>(static_cast<int>(index % 11) - 5) * 0.046875F;
  for (std::size_t index = 0; index < up_values.size(); ++index)
    up_values[index] =
        static_cast<float>(static_cast<int>(index % 13) - 6) * 0.0390625F;
  for (std::size_t index = 0; index < down_values.size(); ++index)
    down_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3) * 0.0546875F;
  write(activation, activation_values);
  write(gate_weight, gate_values);
  write(up_weight, up_values);
  write(down_weight, down_values);
  // Region boundaries are keyed by ValueId, not by binding insertion order.
  const std::array constants{
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
  };
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants})
                      .value();
  ASSERT_EQ(compiled.report.region_count(),
            order == MlpOrder::kControlReentry ? 0u : 1u);
  if (order == MlpOrder::kInterleaved || order == MlpOrder::kUpFirst) {
    ASSERT_EQ(compiled.report.fallback_operation_count(), 1u);
    for (std::size_t index = 0; index < compiled.report.step_count(); ++index) {
      const auto step = compiled.report.step(index).value();
      if (step.kind == CompileStepKind::kProviderRegion)
        EXPECT_FALSE(step.nodes_contiguous);
    }
  }
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.activation, activation.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  expect_near(read(output, rows * inner),
              reference(activation_values, gate_values, up_values, down_values,
                        rows, inner, hidden));
}

TEST(OneDnnRegionProvider, ExecutesExactGatedMlpAndReportsItsCosts) {
  constexpr std::size_t kRows = 17;
  constexpr std::size_t kInner = 5;
  constexpr std::size_t kHidden = 7;
  const GatedMlp logical = make_gated_mlp(kRows, kInner, kHidden);

  OwnedTensor activation = make_tensor(matrix(kRows, kInner));
  OwnedTensor gate_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor up_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor down_weight = make_tensor(matrix(kHidden, kInner));
  OwnedTensor output = make_tensor(matrix(kRows, kInner));
  std::vector<float> activation_values(kRows * kInner);
  std::vector<float> gate_values(kInner * kHidden);
  std::vector<float> up_values(kInner * kHidden);
  std::vector<float> down_values(kHidden * kInner);
  for (std::size_t index = 0; index < activation_values.size(); ++index)
    activation_values[index] =
        static_cast<float>(static_cast<int>(index % 13) - 6) * 0.0625F;
  for (std::size_t index = 0; index < gate_values.size(); ++index)
    gate_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3) * 0.125F;
  for (std::size_t index = 0; index < up_values.size(); ++index)
    up_values[index] =
        static_cast<float>(static_cast<int>(index % 5) - 2) * 0.1875F;
  for (std::size_t index = 0; index < down_values.size(); ++index)
    down_values[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) * 0.09375F;
  write(activation, activation_values);
  write(gate_weight, gate_values);
  write(up_weight, up_values);
  write(down_weight, down_values);

  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  const auto providers = provider_sequence();
  Context context = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                                   {.providers = providers})
                        .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants})
                      .value();
  ASSERT_EQ(compiled.report.region_count(), 1U);
  ASSERT_EQ(compiled.report.step_count(), 1U);
  EXPECT_EQ(compiled.report.provider().id, hyle::graph::onednn::kProviderId);
  EXPECT_FALSE(compiled.report.mixed_providers());
  EXPECT_EQ(compiled.report.copy_bytes(), 0U);
  EXPECT_EQ(compiled.report.synchronization_count(), 0U);
  auto step = compiled.report.step(0);
  ASSERT_TRUE(step.has_value());
  EXPECT_EQ(step->node_count, 4U);
  EXPECT_EQ(step->dispatch_count, 2U);
  EXPECT_TRUE(step->runtime.blocks_host);
  EXPECT_FALSE(step->runtime.uses_private_threads);
  EXPECT_TRUE(step->runtime.supports_concurrent_invocations);
  EXPECT_EQ(step->runtime.replay_class,
            hyle::compute::ProviderReplayClass::kPrepared);
  EXPECT_GT(step->workspace.size, 2 * kRows * kHidden * sizeof(float));
  EXPECT_EQ(step->workspace.alignment, 64U);

  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.activation, activation.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  auto completion = invocation.submit(queue);
  ASSERT_TRUE(completion.has_value());
  const hyle::Status terminal = completion->wait();
  ASSERT_TRUE(terminal.ok()) << terminal.message();
  expect_near(read(output, kRows * kInner),
              reference(activation_values, gate_values, up_values, down_values,
                        kRows, kInner, kHidden));
}

TEST(OneDnnRegionProvider, SupportsDecodeAndNonTileMultipleShapes) {
  run_gated_mlp_shape(1, 5, 7);
  run_gated_mlp_shape(4, 17, 31);
  run_gated_mlp_shape(16, 33, 65);
}

TEST(OneDnnRegionProvider, MatchesDataflowWithUnrelatedAndReorderedNodes) {
  run_gated_mlp_shape(4, 17, 31, MlpOrder::kInterleaved);
  run_gated_mlp_shape(4, 17, 31, MlpOrder::kUpFirst);
}

TEST(OneDnnRegionProvider, DoesNotFuseAcrossAControlDependencyReentry) {
  run_gated_mlp_shape(4, 17, 31, MlpOrder::kControlReentry);
}

TEST(OneDnnRegionProvider, DoesNotInterceptSingleOperationPreparation) {
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto prepared = hyle::compute::prepare(context, MatMul{}, matrix(16, 5),
                                         matrix(5, 7), matrix(16, 7));
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_NE(prepared->provider().id, hyle::graph::onednn::kProviderId);
}

TEST(OneDnnRegionProvider, RejectsPublishedIntermediateWithoutForcedFusion) {
  const GatedMlp logical = make_gated_mlp(17, 5, 7, true);
  OwnedTensor gate_weight = make_tensor(matrix(5, 7));
  OwnedTensor up_weight = make_tensor(matrix(5, 7));
  OwnedTensor down_weight = make_tensor(matrix(7, 5));
  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 0U);
  EXPECT_EQ(compiled->report.fallback_operation_count(), 4U);
  for (std::size_t index = 0; index < compiled->report.step_count(); ++index) {
    const auto step = compiled->report.step(index).value();
    EXPECT_EQ(step.kind, CompileStepKind::kOperationFallback);
    EXPECT_EQ(step.fallback_reason, CompileFallbackReason::kNoEligibleRegion);
    EXPECT_NE(step.provider.id, hyle::graph::onednn::kProviderId);
  }
}

TEST(OneDnnRegionProvider, ExtraIntermediateConsumerIsNotForcedIntoRegion) {
  const GatedMlp logical = make_gated_mlp(17, 5, 7, false, false, true);
  OwnedTensor gate_weight = make_tensor(matrix(5, 7));
  OwnedTensor up_weight = make_tensor(matrix(5, 7));
  OwnedTensor down_weight = make_tensor(matrix(7, 5));
  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 0U);
  EXPECT_EQ(compiled->report.fallback_operation_count(), 5U);
  for (std::size_t index = 0; index < compiled->report.step_count(); ++index) {
    const auto step = compiled->report.step(index).value();
    EXPECT_EQ(step.kind, CompileStepKind::kOperationFallback);
    EXPECT_EQ(step.fallback_reason, CompileFallbackReason::kNoEligibleRegion);
    EXPECT_NE(step.provider.id, hyle::graph::onednn::kProviderId);
  }
}

TEST(OneDnnRegionProvider,
     StrictBenefitPolicyFallsBackWhenPortableEstimateIsUnknown) {
  const GatedMlp logical = make_gated_mlp(17, 5, 7);
  OwnedTensor gate_weight = make_tensor(matrix(5, 7));
  OwnedTensor up_weight = make_tensor(matrix(5, 7));
  OwnedTensor down_weight = make_tensor(matrix(7, 5));
  const std::array constants{
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
  };
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto compiled = hyle::graph::compile(
      logical.graph, context,
      CompileOptions{.constants = constants,
                     .region_selection =
                         RegionSelectionPolicy::kRequireEstimatedBenefit});
  ASSERT_TRUE(compiled.has_value()) << compiled.error().message();
  EXPECT_EQ(compiled->report.region_count(), 0U);
  EXPECT_EQ(compiled->report.fallback_operation_count(), 4U);
  for (std::size_t index = 0; index < compiled->report.step_count(); ++index) {
    const auto step = compiled->report.step(index).value();
    EXPECT_EQ(step.kind, CompileStepKind::kOperationFallback);
    EXPECT_EQ(step.fallback_reason, CompileFallbackReason::kNoEligibleRegion);
    EXPECT_NE(step.provider.id, hyle::graph::onednn::kProviderId);
  }
}

TEST(OneDnnRegionProvider, ComposesAfterRmsNormAsOneMixedExecutionPlan) {
  constexpr std::size_t kRows = 16;
  constexpr std::size_t kInner = 5;
  constexpr std::size_t kHidden = 7;
  const GatedMlp logical = make_gated_mlp(kRows, kInner, kHidden, false, true);
  OwnedTensor activation = make_tensor(matrix(kRows, kInner));
  OwnedTensor norm_weight = make_tensor(row(kInner));
  OwnedTensor gate_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor up_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor down_weight = make_tensor(matrix(kHidden, kInner));
  OwnedTensor output = make_tensor(matrix(kRows, kInner));
  std::vector<float> activation_values(kRows * kInner);
  std::vector<float> norm_values(kInner);
  std::vector<float> gate_values(kInner * kHidden);
  std::vector<float> up_values(kInner * kHidden);
  std::vector<float> down_values(kHidden * kInner);
  for (std::size_t index = 0; index < activation_values.size(); ++index)
    activation_values[index] =
        static_cast<float>(static_cast<int>(index % 11) - 5) * 0.09375F;
  for (std::size_t index = 0; index < norm_values.size(); ++index)
    norm_values[index] = 0.75F + static_cast<float>(index) * 0.125F;
  for (std::size_t index = 0; index < gate_values.size(); ++index)
    gate_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3) * 0.0625F;
  for (std::size_t index = 0; index < up_values.size(); ++index)
    up_values[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) * 0.078125F;
  for (std::size_t index = 0; index < down_values.size(); ++index)
    down_values[index] =
        static_cast<float>(static_cast<int>(index % 5) - 2) * 0.15625F;
  write(activation, activation_values);
  write(norm_weight, norm_values);
  write(gate_weight, gate_values);
  write(up_weight, up_values);
  write(down_weight, down_values);
  const std::array constants{
      ConstantBinding{logical.norm_weight, norm_weight.view.as_const()},
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants})
                      .value();
  ASSERT_EQ(compiled.report.step_count(), 2U);
  EXPECT_EQ(compiled.report.region_count(), 1U);
  EXPECT_TRUE(compiled.report.mixed_providers());
  EXPECT_EQ(compiled.report.step(0)->provider.id,
            hyle::compute::native::kProviderId);
  EXPECT_EQ(compiled.report.step(1)->provider.id,
            hyle::graph::onednn::kProviderId);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.activation, activation.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  const std::vector<float> normalized =
      rms_norm(activation_values, norm_values, kRows, kInner);
  expect_near(read(output, kRows * kInner),
              reference(normalized, gate_values, up_values, down_values, kRows,
                        kInner, kHidden));
}

TEST(OneDnnRegionProvider, WarmSubmitDoesNotGrowContextMetadata) {
  constexpr std::size_t kRows = 16;
  constexpr std::size_t kInner = 5;
  constexpr std::size_t kHidden = 7;
  const GatedMlp logical = make_gated_mlp(kRows, kInner, kHidden);
  OwnedTensor activation = make_tensor(matrix(kRows, kInner));
  OwnedTensor gate_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor up_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor down_weight = make_tensor(matrix(kHidden, kInner));
  OwnedTensor output = make_tensor(matrix(kRows, kInner));
  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  AllocatorProbe probe;
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(),
                      {.metadata_allocator = probe_allocator(probe),
                       .providers = providers})
          .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants})
                      .value();
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.activation, activation.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  const std::size_t before = probe.attempts.load();
  for (std::size_t iteration = 0; iteration < 8; ++iteration)
    ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(probe.attempts.load(), before);
}

TEST(OneDnnRegionProvider, BorrowedNestedExecutorRunsTwoEpisodes) {
  constexpr std::size_t kRows = 2;
  constexpr std::size_t kInner = 129;
  constexpr std::size_t kHidden = 257;
  const GatedMlp logical = make_gated_mlp(kRows, kInner, kHidden);
  OwnedTensor activation = make_tensor(matrix(kRows, kInner));
  OwnedTensor gate_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor up_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor down_weight = make_tensor(matrix(kHidden, kInner));
  OwnedTensor output = make_tensor(matrix(kRows, kInner));
  write(activation, std::vector<float>(kRows * kInner, 0.03125F));
  write(gate_weight, std::vector<float>(kInner * kHidden, 0.015625F));
  write(up_weight, std::vector<float>(kInner * kHidden, -0.0078125F));
  write(down_weight, std::vector<float>(kHidden * kInner, 0.00390625F));
  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  ParallelProbe probe;
  const ParallelExecutorRef executor(&probe, 4, &probe_parallel_for,
                                     &probe_in_parallel);
  const std::array<Provider, 3> providers{
      hyle::graph::onednn::provider({.executor = executor}),
      hyle::compute::onednn::provider({.executor = executor}),
      hyle::compute::native::provider({.executor = executor}),
  };
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants})
                      .value();
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(logical.activation, activation.view).ok());
  ASSERT_TRUE(invocation.bind(logical.output, output.view).ok());
  auto queue = context.create_queue().value();
  ASSERT_TRUE(invocation.submit(queue)->wait().ok());
  EXPECT_EQ(probe.episodes.load(), 2U);
  EXPECT_GT(probe.tasks.load(), 2U);
}

TEST(OneDnnRegionProvider, SharedExecutableSupportsConcurrentInvocations) {
  constexpr std::size_t kRows = 16;
  constexpr std::size_t kInner = 5;
  constexpr std::size_t kHidden = 7;
  const GatedMlp logical = make_gated_mlp(kRows, kInner, kHidden);
  OwnedTensor first_activation = make_tensor(matrix(kRows, kInner));
  OwnedTensor second_activation = make_tensor(matrix(kRows, kInner));
  OwnedTensor gate_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor up_weight = make_tensor(matrix(kInner, kHidden));
  OwnedTensor down_weight = make_tensor(matrix(kHidden, kInner));
  OwnedTensor first_output = make_tensor(matrix(kRows, kInner));
  OwnedTensor second_output = make_tensor(matrix(kRows, kInner));
  std::vector<float> first_values(kRows * kInner, 0.25F);
  std::vector<float> second_values(kRows * kInner, -0.375F);
  std::vector<float> gate_values(kInner * kHidden, 0.125F);
  std::vector<float> up_values(kInner * kHidden, -0.0625F);
  std::vector<float> down_values(kHidden * kInner, 0.1875F);
  write(first_activation, first_values);
  write(second_activation, second_values);
  write(gate_weight, gate_values);
  write(up_weight, up_values);
  write(down_weight, down_values);
  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(), {.providers = providers})
          .value();
  auto executable = hyle::graph::compile(logical.graph, context,
                                         CompileOptions{.constants = constants})
                        ->executable;
  auto first = executable.create_invocation().value();
  auto second = executable.create_invocation().value();
  ASSERT_TRUE(first.bind(logical.activation, first_activation.view).ok());
  ASSERT_TRUE(first.bind(logical.output, first_output.view).ok());
  ASSERT_TRUE(second.bind(logical.activation, second_activation.view).ok());
  ASSERT_TRUE(second.bind(logical.output, second_output.view).ok());
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  std::atomic<bool> first_ok{false};
  std::atomic<bool> second_ok{false};
  std::thread first_thread([&] {
    auto completion = first.submit(first_queue);
    first_ok = completion.has_value() && completion->wait().ok();
  });
  std::thread second_thread([&] {
    auto completion = second.submit(second_queue);
    second_ok = completion.has_value() && completion->wait().ok();
  });
  first_thread.join();
  second_thread.join();
  ASSERT_TRUE(first_ok.load());
  ASSERT_TRUE(second_ok.load());
  expect_near(read(first_output, kRows * kInner),
              reference(first_values, gate_values, up_values, down_values,
                        kRows, kInner, kHidden));
  expect_near(read(second_output, kRows * kInner),
              reference(second_values, gate_values, up_values, down_values,
                        kRows, kInner, kHidden));
}

TEST(OneDnnRegionProvider, EveryCompileAllocationFailureRollsBackAndRetries) {
  const GatedMlp logical = make_gated_mlp(16, 5, 7);
  OwnedTensor gate_weight = make_tensor(matrix(5, 7));
  OwnedTensor up_weight = make_tensor(matrix(5, 7));
  OwnedTensor down_weight = make_tensor(matrix(7, 5));
  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  AllocatorProbe probe;
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(),
                      {.metadata_allocator = probe_allocator(probe),
                       .providers = providers})
          .value();
  probe.attempts = 0;
  {
    auto successful = hyle::graph::compile(
        logical.graph, context, CompileOptions{.constants = constants});
    ASSERT_TRUE(successful.has_value()) << successful.error().message();
  }
  const std::size_t compile_attempts = probe.attempts.load();
  ASSERT_GT(compile_attempts, 4U);

  for (std::size_t failure = 0; failure < compile_attempts; ++failure) {
    probe.attempts = 0;
    probe.fail_on_attempt = failure;
    auto failed = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants});
    ASSERT_FALSE(failed.has_value()) << "allocation attempt " << failure;
    EXPECT_TRUE(failed.error().IsResourceExhausted())
        << "allocation attempt " << failure << ": " << failed.error().message();

    probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
    probe.attempts = 0;
    auto retried = hyle::graph::compile(logical.graph, context,
                                        CompileOptions{.constants = constants});
    ASSERT_TRUE(retried.has_value())
        << "retry after allocation attempt " << failure << ": "
        << retried.error().message();
  }
}

TEST(OneDnnRegionProvider,
     EveryInvocationAllocationFailureRollsBackAndRetries) {
  const GatedMlp logical = make_gated_mlp(16, 5, 7);
  OwnedTensor gate_weight = make_tensor(matrix(5, 7));
  OwnedTensor up_weight = make_tensor(matrix(5, 7));
  OwnedTensor down_weight = make_tensor(matrix(7, 5));
  const std::array constants{
      ConstantBinding{logical.gate_weight, gate_weight.view.as_const()},
      ConstantBinding{logical.up_weight, up_weight.view.as_const()},
      ConstantBinding{logical.down_weight, down_weight.view.as_const()},
  };
  AllocatorProbe probe;
  const auto providers = provider_sequence();
  Context context =
      Context::create(hyle::compute::cpu::device(),
                      {.metadata_allocator = probe_allocator(probe),
                       .providers = providers})
          .value();
  auto compiled = hyle::graph::compile(logical.graph, context,
                                       CompileOptions{.constants = constants})
                      .value();

  probe.attempts = 0;
  {
    auto successful = compiled.executable.create_invocation();
    ASSERT_TRUE(successful.has_value()) << successful.error().message();
  }
  const std::size_t invocation_attempts = probe.attempts.load();
  ASSERT_GT(invocation_attempts, 1U);

  for (std::size_t failure = 0; failure < invocation_attempts; ++failure) {
    probe.attempts = 0;
    probe.fail_on_attempt = failure;
    auto failed = compiled.executable.create_invocation();
    ASSERT_FALSE(failed.has_value()) << "allocation attempt " << failure;
    EXPECT_TRUE(failed.error().IsResourceExhausted())
        << "allocation attempt " << failure << ": " << failed.error().message();

    probe.fail_on_attempt = (std::numeric_limits<std::size_t>::max)();
    probe.attempts = 0;
    auto retried = compiled.executable.create_invocation();
    ASSERT_TRUE(retried.has_value())
        << "retry after allocation attempt " << failure << ": "
        << retried.error().message();
  }
}

} // namespace
