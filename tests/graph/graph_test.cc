#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include "hyle/graph/graph.h"

namespace {

using hyle::Status;
using hyle::compute::DataType;
using hyle::compute::MatMul;
using hyle::compute::Operation;
using hyle::compute::OperationKind;
using hyle::compute::RmsNorm;
using hyle::compute::SiluAndMul;
using hyle::compute::TensorDesc;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::GraphBuilderOptions;
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
  const auto activated = builder.add(SiluAndMul{}, projected).value();
  ASSERT_TRUE(builder.output(activated).ok());

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
            OperationKind::kSiluAndMul);
  EXPECT_EQ(graph.node_input_count(norm_node).value(), 2u);
  EXPECT_EQ(graph.node_input(norm_node, 0).value(), input);
  EXPECT_EQ(graph.node_input(norm_node, 1).value(), weight);
  EXPECT_EQ(graph.node_output(norm_node).value(), normalized);
  EXPECT_EQ(graph.node_output(matmul_node).value(), projected);
  EXPECT_EQ(graph.node_output(activation_node).value(), activated);
  EXPECT_EQ(graph.producer(normalized).value(), norm_node);
  EXPECT_EQ(input.kind(), ValueKind::kInput);
  EXPECT_EQ(normalized.kind(), ValueKind::kNodeOutput);
}

TEST(GraphBuilder, SupportsBranchesSharedValuesAndMultipleOutputs) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(matrix(2, 4)).value();
  const auto first_weight = builder.input(matrix(4, 3)).value();
  const auto second_weight = builder.input(matrix(4, 5)).value();
  const auto first = builder.add(MatMul{}, input, first_weight).value();
  const auto second = builder.add(MatMul{}, input, second_weight).value();
  ASSERT_TRUE(builder.output(first).ok());
  ASSERT_TRUE(builder.output(second).ok());

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
  EXPECT_TRUE(first.output(recovered).ok());
  EXPECT_EQ(std::move(first).finish()->node_count(), 1u);
}

TEST(GraphBuilder, OutputRoleAndPublicationAreExplicit) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(row(4)).value();
  EXPECT_TRUE(builder.output(input).IsInvalidArgument());
  const auto result = builder.add(RmsNorm{}, input).value();
  EXPECT_TRUE(builder.output(result).ok());
  EXPECT_TRUE(builder.output(result).IsAlreadyExists());

  GraphBuilder unfinished = GraphBuilder::create().value();
  EXPECT_TRUE(std::move(unfinished).finish().error().IsFailedPrecondition());
}

TEST(GraphBuilder, MoveAndGraphCopiesPreserveOriginAndLifetime) {
  GraphBuilder original = GraphBuilder::create().value();
  const auto input = original.input(row(8)).value();
  GraphBuilder moved = std::move(original);
  EXPECT_FALSE(original.valid()); // NOLINT(bugprone-use-after-move)
  const auto output = moved.add(RmsNorm{}, input).value();
  ASSERT_TRUE(moved.output(output).ok());
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
  ASSERT_TRUE(builder.output(output).ok());

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
  const auto output =
      builder.add(Operation(RmsNorm{}), std::array{input}).value();
  ASSERT_TRUE(builder.output(output).ok());
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
  ASSERT_TRUE(other.output(foreign_output).ok());
  Graph other_graph = std::move(other).finish().value();
  const auto foreign_node = other_graph.node_id().value();
  EXPECT_TRUE(graph.value_description(foreign).error().IsInvalidArgument());
  EXPECT_TRUE(graph.operation(foreign_node).error().IsInvalidArgument());
  EXPECT_TRUE(graph.producer(foreign_output).error().IsInvalidArgument());
}

} // namespace
