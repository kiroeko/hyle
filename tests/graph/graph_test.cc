#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/graph/graph.h"

namespace {

using hyle::Status;
using hyle::compute::AliasPolicy;
using hyle::compute::DataType;
using hyle::compute::EffectKind;
using hyle::compute::GatedActivation;
using hyle::compute::LayerNorm;
using hyle::compute::MatMul;
using hyle::compute::Operation;
using hyle::compute::OperationId;
using hyle::compute::OperationKind;
using hyle::compute::OperationTraits;
using hyle::compute::PrecisionPolicy;
using hyle::compute::RmsNorm;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::TopK;
using hyle::graph::Dimension;
using hyle::graph::DimensionBounds;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::GraphBuilderOptions;
using hyle::graph::TensorSpec;
using hyle::graph::TensorSpecKind;
using hyle::graph::ValueKind;
using hyle::mem::AllocatorRef;

TensorDesc row(std::int64_t size) {
  const std::array shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array shape{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::size_t allocations = 0;
  std::size_t deallocations = 0;
  bool fail = false;
};

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (probe.fail)
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
  if (probe.fail)
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_free, &probe_reallocate, &probe);
}

struct ExternalEffect final {
  static constexpr OperationId operation_id{0x48594c45U, 0x45464654U};
  static constexpr std::string_view operation_name = "test-external-effect";

  OperationTraits operation_traits() const noexcept {
    return {{0, 0},
            0,
            AliasPolicy::kNone,
            EffectKind::kExternalSideEffect,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            false};
  }

  Status
  infer_output_descriptions(std::span<const TensorDesc> inputs,
                            std::span<TensorDesc> outputs) const noexcept {
    return inputs.empty() && outputs.empty()
               ? Status::Ok()
               : Status::InvalidArgument("effect operation has no tensors");
  }
};

TEST(GraphBuilder, BuildsNaturalMultiNodeTransformerFragment) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc activation = matrix(2, 4);
  const TensorDesc norm_weight = row(4);
  const TensorDesc stored_gate_up = matrix(8, 4);
  const TensorDesc logical_gate_up = stored_gate_up.transposed().value();

  const auto input = builder.input(activation).value();
  const auto weight = builder.input(norm_weight).value();
  const auto matrix_weight = builder.input(logical_gate_up).value();
  const auto normalized = builder.add(RmsNorm{}, input, weight).value();
  const auto projected =
      builder.add(MatMul{}, normalized, matrix_weight).value();
  const auto activated = builder.add(GatedActivation{}, projected).value();
  ASSERT_TRUE(builder.publish(activated).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 3u);
  EXPECT_EQ(graph.node_count(), 3u);
  EXPECT_EQ(graph.output_count(), 1u);
  EXPECT_EQ(graph.value_count(), 6u);
  EXPECT_EQ(graph.input_id(0).value(), input);
  EXPECT_EQ(graph.input_id(1).value(), weight);
  EXPECT_EQ(graph.input_id(2).value(), matrix_weight);
  EXPECT_EQ(graph.output_id().value(), activated);
  EXPECT_EQ(graph.output_description().value(), activation);
  EXPECT_EQ(graph.value_description(projected).value(), matrix(2, 8));

  const auto norm_node = graph.node_id(0).value();
  const auto matmul_node = graph.node_id(1).value();
  const auto activation_node = graph.node_id(2).value();
  EXPECT_EQ(graph.operation(norm_node)->kind(), OperationKind::kRmsNorm);
  EXPECT_EQ(graph.operation(matmul_node)->kind(), OperationKind::kMatMul);
  EXPECT_EQ(graph.operation(activation_node)->kind(),
            OperationKind::kGatedActivation);
  EXPECT_EQ(graph.node_input_count(norm_node).value(), 2u);
  EXPECT_EQ(graph.node_input(norm_node, 0).value(), input);
  EXPECT_EQ(graph.node_input(norm_node, 1).value(), weight);
  EXPECT_EQ(graph.result(norm_node).value(), normalized);
  EXPECT_EQ(graph.result(matmul_node).value(), projected);
  EXPECT_EQ(graph.result(activation_node).value(), activated);
  EXPECT_EQ(graph.producer(normalized).value(), norm_node);
  EXPECT_EQ(input.kind(), ValueKind::kInput);
  EXPECT_EQ(normalized.kind(), ValueKind::kNodeOutput);
}

TEST(GraphBuilder, BuildsBoundedSymbolicGraphWithExplicitOutputSpecs) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto batch = builder.dimension({.minimum = 1, .maximum = 64}).value();
  const auto tokens =
      builder.dimension({.minimum = 8, .maximum = 4096, .multiple_of = 8})
          .value();
  const std::array activation_shape{Dimension(batch), Dimension(tokens),
                                    Dimension::fixed(128)};
  const TensorSpec activation =
      TensorSpec::dense(DataType::kFloat16, activation_shape).value();
  const auto input = builder.input(activation).value();
  const auto output = builder.add(RmsNorm{}, activation, input).value();
  ASSERT_TRUE(builder.publish(output).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.dimension_count(), 2u);
  EXPECT_EQ(graph.dimension_id(0).value(), batch);
  EXPECT_EQ(graph.dimension_id(1).value(), tokens);
  EXPECT_EQ(graph.dimension_bounds(tokens).value(),
            (DimensionBounds{.minimum = 8, .maximum = 4096, .multiple_of = 8}));
  EXPECT_EQ(graph.input_spec().value(), activation);
  EXPECT_EQ(graph.output_spec().value(), activation);
  EXPECT_EQ(graph.value_spec(output)->kind(), TensorSpecKind::kDense);
  EXPECT_TRUE(graph.input_description().error().IsFailedPrecondition());
  EXPECT_TRUE(graph.output_description().error().IsFailedPrecondition());
}

TEST(GraphBuilder, RejectsInvalidAndForeignSymbolicDimensions) {
  GraphBuilder first = GraphBuilder::create().value();
  EXPECT_TRUE(first.dimension({.minimum = 0, .maximum = 4})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(first.dimension({.minimum = 9, .maximum = 15, .multiple_of = 8})
                  .error()
                  .IsInvalidArgument());

  GraphBuilder second = GraphBuilder::create().value();
  const auto foreign = second.dimension({.minimum = 1, .maximum = 8}).value();
  const std::array shape{Dimension(foreign), Dimension::fixed(4)};
  const TensorSpec spec = TensorSpec::dense(DataType::kFloat32, shape).value();
  EXPECT_TRUE(first.input(spec).error().IsInvalidArgument());

  const auto local = first.dimension({.minimum = 1, .maximum = 8}).value();
  const TensorSpec local_spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(local), Dimension::fixed(4)})
          .value();
  const auto input = first.input(local_spec).value();
  EXPECT_TRUE(first.add(RmsNorm{}, input).error().IsFailedPrecondition());
}

TEST(GraphBuilder, FixedSpecsShareConcreteValidationAndInference) {
  const TensorSpec fixed =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension::fixed(2), Dimension::fixed(4)})
          .value();
  EXPECT_FALSE(fixed.is_symbolic());
  EXPECT_EQ(fixed.concrete_description().value(), matrix(2, 4));

  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(fixed).value();
  const auto output = builder.add(RmsNorm{}, input).value();
  EXPECT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.output_description().value(), matrix(2, 4));

  GraphBuilder mismatch = GraphBuilder::create().value();
  const auto mismatch_input = mismatch.input(fixed).value();
  const TensorSpec wrong =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension::fixed(2), Dimension::fixed(5)})
          .value();
  EXPECT_TRUE(mismatch.add(RmsNorm{}, wrong, mismatch_input)
                  .error()
                  .IsInvalidArgument());
}

TEST(GraphBuilder, SupportsBranchesSharedValuesAndMultipleOutputs) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(matrix(2, 4)).value();
  const auto first_weight = builder.input(matrix(4, 3)).value();
  const auto second_weight = builder.input(matrix(4, 5)).value();
  const auto first = builder.add(MatMul{}, input, first_weight).value();
  const auto second = builder.add(MatMul{}, input, second_weight).value();
  ASSERT_TRUE(builder.publish(first).ok());
  ASSERT_TRUE(builder.publish(second).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 3u);
  EXPECT_EQ(graph.node_count(), 2u);
  EXPECT_EQ(graph.output_count(), 2u);
  EXPECT_EQ(graph.node_input(graph.node_id(0).value(), 0).value(), input);
  EXPECT_EQ(graph.node_input(graph.node_id(1).value(), 0).value(), input);
  EXPECT_EQ(graph.output_id(0).value(), first);
  EXPECT_EQ(graph.output_id(1).value(), second);
  EXPECT_EQ(graph.output_description(0).value(), matrix(2, 3));
  EXPECT_EQ(graph.output_description(1).value(), matrix(2, 5));
}

TEST(GraphBuilder, RepresentsConstantsGeneralArityAndMultiResultNodes) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(matrix(2, 4)).value();
  const auto scale = builder.constant(row(4)).value();
  const auto bias = builder.constant(row(4)).value();
  const std::array normalization_inputs{input, scale, bias};
  const auto normalization =
      builder.add_node(Operation(LayerNorm{}), normalization_inputs).value();
  const auto normalized = builder.result(normalization).value();
  const std::array top_k_inputs{normalized};
  const auto top_k =
      builder.add_node(Operation(TopK{.count = 2}), top_k_inputs).value();
  const auto values = builder.result(top_k, 0).value();
  const auto indices = builder.result(top_k, 1).value();
  EXPECT_TRUE(builder.result(top_k, 2).error().IsOutOfRange());
  ASSERT_TRUE(builder.publish(values).ok());
  ASSERT_TRUE(builder.publish(indices).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 1u);
  EXPECT_EQ(graph.constant_count(), 2u);
  EXPECT_EQ(graph.node_count(), 2u);
  EXPECT_EQ(graph.value_count(), 6u);
  EXPECT_EQ(graph.constant_id(0).value(), scale);
  EXPECT_EQ(graph.constant_id(1).value(), bias);
  EXPECT_EQ(graph.constant_description(0).value(), row(4));
  EXPECT_EQ(scale.kind(), ValueKind::kConstant);
  EXPECT_EQ(graph.node_input_count(normalization).value(), 3u);
  EXPECT_EQ(graph.node_output_count(normalization).value(), 1u);
  EXPECT_EQ(graph.node_output_count(top_k).value(), 2u);
  EXPECT_EQ(graph.result(top_k, 0).value(), values);
  EXPECT_EQ(graph.result(top_k, 1).value(), indices);
  EXPECT_EQ(graph.producer(values).value(), top_k);
  EXPECT_TRUE(graph.producer(scale).error().IsFailedPrecondition());
}

TEST(GraphBuilder, PublishesInputAndConstantWithoutManufacturingNodes) {
  GraphBuilder input_builder = GraphBuilder::create().value();
  const auto input = input_builder.input(row(4)).value();
  ASSERT_TRUE(input_builder.publish(input).ok());
  Graph input_graph = std::move(input_builder).finish().value();
  EXPECT_EQ(input_graph.node_count(), 0u);
  EXPECT_EQ(input_graph.output_id().value(), input);

  GraphBuilder constant_builder = GraphBuilder::create().value();
  const auto constant = constant_builder.constant(row(4)).value();
  ASSERT_TRUE(constant_builder.publish(constant).ok());
  Graph constant_graph = std::move(constant_builder).finish().value();
  EXPECT_EQ(constant_graph.node_count(), 0u);
  EXPECT_EQ(constant_graph.constant_count(), 1u);
  EXPECT_EQ(constant_graph.output_id().value(), constant);
}

TEST(GraphBuilder, RequiresExplicitOrderingBetweenExternalEffects) {
  constexpr std::array<hyle::graph::ValueId, 0> no_inputs{};
  GraphBuilder unordered = GraphBuilder::create().value();
  ASSERT_TRUE(
      unordered.add_node(Operation::custom(ExternalEffect{}), no_inputs));
  ASSERT_TRUE(
      unordered.add_node(Operation::custom(ExternalEffect{}), no_inputs));
  EXPECT_TRUE(std::move(unordered).finish().error().IsFailedPrecondition());

  GraphBuilder builder = GraphBuilder::create().value();
  const auto first =
      builder.add_node(Operation::custom(ExternalEffect{}), no_inputs).value();
  const auto second =
      builder.add_node(Operation::custom(ExternalEffect{}), no_inputs).value();
  EXPECT_TRUE(builder.depends_on(first, second).IsInvalidArgument());
  ASSERT_TRUE(builder.depends_on(second, first).ok());
  ASSERT_TRUE(builder.depends_on(second, first).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.node_count(), 2u);
  EXPECT_EQ(graph.output_count(), 0u);
  EXPECT_EQ(graph.node_output_count(first).value(), 0u);
  EXPECT_EQ(graph.node_dependency_count(first).value(), 0u);
  EXPECT_EQ(graph.node_dependency_count(second).value(), 1u);
  EXPECT_EQ(graph.node_dependency(second, 0).value(), first);
  EXPECT_TRUE(graph.node_dependency(second, 1).error().IsOutOfRange());
}

TEST(GraphBuilder, PreservesProviderNeutralBlockedWeightMetadata) {
  constexpr StorageFormatId codec{0x48594c45U, 41U};
  const StorageFormat blocked =
      StorageFormat::blocked(codec, 32, 18, 2).value();
  const TensorDesc activation = matrix(2, 64);
  const std::array<std::int64_t, 2> stored_shape{8, 64};
  const TensorDesc weight =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, blocked)
          ->transposed()
          .value();

  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(activation).value();
  const auto encoded_weight = builder.input(weight).value();
  const auto product = builder.add(MatMul{}, input, encoded_weight).value();
  ASSERT_TRUE(builder.publish(product).ok());
  Graph graph = std::move(builder).finish().value();

  EXPECT_EQ(graph.value_description(encoded_weight).value(), weight);
  EXPECT_EQ(graph.value_description(product).value(), matrix(2, 8));
}

TEST(GraphBuilder, RejectsForeignIdsAndShapeErrorsWithoutLogicalMutation) {
  GraphBuilder first = GraphBuilder::create().value();
  GraphBuilder second = GraphBuilder::create().value();
  const auto input = first.input(matrix(2, 4)).value();
  const auto foreign = second.input(matrix(2, 4)).value();

  auto cross = first.add(RmsNorm{}, foreign);
  ASSERT_FALSE(cross.has_value());
  EXPECT_TRUE(cross.error().IsInvalidArgument());

  const auto wrong_weight = first.input(row(5)).value();
  auto mismatch = first.add(RmsNorm{}, input, wrong_weight);
  ASSERT_FALSE(mismatch.has_value());
  EXPECT_TRUE(mismatch.error().IsInvalidArgument());

  auto bad_epsilon = first.add(
      RmsNorm{.epsilon = (std::numeric_limits<float>::quiet_NaN)()}, input);
  ASSERT_FALSE(bad_epsilon.has_value());
  EXPECT_TRUE(bad_epsilon.error().IsInvalidArgument());

  const auto recovered = first.add(RmsNorm{}, input).value();
  EXPECT_TRUE(first.publish(recovered).ok());
  EXPECT_EQ(std::move(first).finish()->node_count(), 1u);
}

TEST(GraphBuilder, OutputRoleAndPublicationAreExplicit) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(row(4)).value();
  EXPECT_TRUE(builder.publish(input).ok());
  const auto result = builder.add(RmsNorm{}, input).value();
  EXPECT_TRUE(builder.publish(result).ok());
  EXPECT_TRUE(builder.publish(result).IsAlreadyExists());

  GraphBuilder unfinished = GraphBuilder::create().value();
  EXPECT_TRUE(std::move(unfinished).finish().error().IsFailedPrecondition());
}

TEST(GraphBuilder, MoveAndGraphCopiesPreserveOriginAndLifetime) {
  GraphBuilder original = GraphBuilder::create().value();
  const auto input = original.input(row(8)).value();
  GraphBuilder moved = std::move(original);
  EXPECT_FALSE(original.valid()); // NOLINT(bugprone-use-after-move)
  const auto output = moved.add(RmsNorm{}, input).value();
  ASSERT_TRUE(moved.publish(output).ok());
  Graph first = std::move(moved).finish().value();
  Graph copy = first;
  Graph relocated = std::move(first);
  EXPECT_FALSE(first.valid()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(copy.output_id().value(), output);
  EXPECT_EQ(relocated.input_id().value(), input);
}

TEST(GraphBuilder, AllocationFailureDoesNotPublishPartialLogicalState) {
  AllocatorProbe probe;
  probe.fail = true;
  auto unavailable = GraphBuilder::create(
      GraphBuilderOptions{.metadata_allocator = probe_allocator(probe)});
  ASSERT_FALSE(unavailable.has_value());
  EXPECT_TRUE(unavailable.error().IsResourceExhausted());

  probe.fail = false;
  GraphBuilder builder =
      GraphBuilder::create(
          GraphBuilderOptions{.metadata_allocator = probe_allocator(probe)})
          .value();
  std::array<hyle::graph::ValueId, 8> inputs{};
  for (auto &input : inputs)
    input = builder.input(row(4)).value();
  probe.fail = true;
  auto ninth = builder.input(row(4));
  ASSERT_FALSE(ninth.has_value());
  EXPECT_TRUE(ninth.error().IsResourceExhausted());

  probe.fail = false;
  ninth = builder.input(row(4));
  ASSERT_TRUE(ninth.has_value());
  const auto output = builder.add(RmsNorm{}, inputs[0]).value();
  ASSERT_TRUE(builder.publish(output).ok());

  probe.fail = true;
  auto failed_finish = std::move(builder).finish();
  ASSERT_FALSE(failed_finish.has_value());
  EXPECT_TRUE(failed_finish.error().IsResourceExhausted());
  EXPECT_TRUE(builder.valid());

  probe.fail = false;
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 9u);
  EXPECT_EQ(graph.node_count(), 1u);
}

TEST(Graph, InvalidOutOfRangeAndForeignQueriesAreRecoverable) {
  Graph invalid;
  EXPECT_TRUE(invalid.input_id().error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.node_id().error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.output_id().error().IsFailedPrecondition());

  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(row(4)).value();
  const auto builder_node =
      builder.add_node(Operation(RmsNorm{}), std::array{input}).value();
  const auto output = builder.result(builder_node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_TRUE(graph.input_id(1).error().IsOutOfRange());
  EXPECT_TRUE(graph.node_id(1).error().IsOutOfRange());
  const auto node = graph.node_id().value();
  EXPECT_TRUE(graph.node_input(node, 1).error().IsOutOfRange());
  EXPECT_TRUE(graph.producer(input).error().IsFailedPrecondition());
  EXPECT_TRUE(graph.output_id(1).error().IsOutOfRange());

  GraphBuilder other = GraphBuilder::create().value();
  const auto foreign = other.input(row(4)).value();
  const auto foreign_output = other.add(RmsNorm{}, foreign).value();
  ASSERT_TRUE(other.publish(foreign_output).ok());
  Graph other_graph = std::move(other).finish().value();
  const auto foreign_node = other_graph.node_id().value();
  EXPECT_TRUE(graph.value_description(foreign).error().IsInvalidArgument());
  EXPECT_TRUE(graph.operation(foreign_node).error().IsInvalidArgument());
  EXPECT_TRUE(graph.producer(foreign_output).error().IsInvalidArgument());
}

} // namespace
