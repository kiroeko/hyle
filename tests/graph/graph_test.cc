#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include "hyle/graph/graph.h"

namespace {

using hyle::compute::DataType;
using hyle::compute::Operation;
using hyle::compute::OperationKind;
using hyle::compute::RmsNorm;
using hyle::compute::TensorDesc;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::ValueKind;
using hyle::Status;

TensorDesc row(std::int64_t size) {
  const std::array<std::int64_t, 1> shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

TEST(GraphBuilder, PublishesAnImmutableTypedOneOpGraph) {
  GraphBuilder builder;
  auto input = builder.input(row(32));
  ASSERT_TRUE(input.has_value());
  EXPECT_EQ(input->kind(), ValueKind::kInput);
  EXPECT_EQ(input->ordinal(), 0u);

  auto output = builder.add(RmsNorm{.epsilon = 1.0e-6F}, *input);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->kind(), ValueKind::kOutput);
  EXPECT_EQ(output->ordinal(), 0u);
  ASSERT_TRUE(builder.output(*output).ok());

  auto finished = std::move(builder).finish();
  ASSERT_TRUE(finished.has_value()) << finished.error().message();
  Graph graph = *finished;
  EXPECT_TRUE(graph.valid());
  EXPECT_EQ(graph.input_count(), 1u);
  EXPECT_EQ(graph.node_count(), 1u);
  EXPECT_EQ(graph.output_count(), 1u);
  EXPECT_EQ(graph.input_id().value(), *input);
  EXPECT_EQ(graph.output_id().value(), *output);
  EXPECT_EQ(graph.input_description().value(), row(32));
  EXPECT_EQ(graph.output_description().value(), row(32));
  EXPECT_EQ(graph.operation()->kind(), OperationKind::kRmsNorm);
  ASSERT_NE(graph.operation()->rms_norm(), nullptr);
  EXPECT_FLOAT_EQ(graph.operation()->rms_norm()->epsilon, 1.0e-6F);
  EXPECT_TRUE(graph.node_id()->valid());
}

TEST(GraphBuilder, WeightedFormRequiresMatchingDeclaredInputs) {
  GraphBuilder builder;
  auto input = builder.input(row(16)).value();
  auto weight = builder.input(row(16)).value();
  auto output = builder.add(RmsNorm{}, input, weight);
  ASSERT_TRUE(output.has_value());
  ASSERT_TRUE(builder.output(*output).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 2u);
  EXPECT_EQ(graph.input_id(0).value(), input);
  EXPECT_EQ(graph.input_id(1).value(), weight);
}

TEST(GraphBuilder, RejectsCrossBuilderAndWrongRoleIdsWithoutMutation) {
  GraphBuilder first;
  GraphBuilder second;
  auto first_input = first.input(row(8)).value();
  auto second_input = second.input(row(8)).value();

  auto cross = first.add(RmsNorm{}, second_input);
  ASSERT_FALSE(cross.has_value());
  EXPECT_TRUE(cross.error().IsInvalidArgument());

  auto output = first.add(RmsNorm{}, first_input);
  ASSERT_TRUE(output.has_value());
  Status wrong_role = first.output(first_input);
  EXPECT_TRUE(wrong_role.IsInvalidArgument());
  EXPECT_TRUE(first.output(*output).ok());
  EXPECT_TRUE(std::move(first).finish().has_value());
}

TEST(GraphBuilder, MovePreservesOriginAndInvalidatesTheSource) {
  GraphBuilder original;
  auto input = original.input(row(8)).value();
  GraphBuilder moved = std::move(original);
  EXPECT_FALSE(original.valid());  // NOLINT(bugprone-use-after-move)
  auto output = moved.add(RmsNorm{}, input);
  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(moved.output(*output).ok());
  EXPECT_TRUE(std::move(moved).finish().has_value());
}

TEST(GraphBuilder, FiniteLimitsAndOrderingAreExplicit) {
  GraphBuilder no_node;
  auto unfinished = std::move(no_node).finish();
  ASSERT_FALSE(unfinished.has_value());
  EXPECT_TRUE(unfinished.error().IsFailedPrecondition());

  GraphBuilder builder;
  auto input = builder.input(row(4)).value();
  auto output = builder.add(RmsNorm{}, input).value();
  auto late_input = builder.input(row(4));
  ASSERT_FALSE(late_input.has_value());
  EXPECT_TRUE(late_input.error().IsFailedPrecondition());
  auto second_node = builder.add(RmsNorm{}, input);
  ASSERT_FALSE(second_node.has_value());
  EXPECT_TRUE(second_node.error().IsAlreadyExists());
  EXPECT_TRUE(builder.output(output).ok());
  EXPECT_TRUE(builder.output(output).IsAlreadyExists());

  GraphBuilder too_many;
  ASSERT_TRUE(too_many.input(row(4)).has_value());
  ASSERT_TRUE(too_many.input(row(4)).has_value());
  auto third = too_many.input(row(4));
  ASSERT_FALSE(third.has_value());
  EXPECT_TRUE(third.error().IsUnimplemented());
}

TEST(GraphBuilder, ValidatesShapeAndOperationFactsBeforeCommit) {
  GraphBuilder mismatch;
  auto input = mismatch.input(row(4)).value();
  auto weight = mismatch.input(row(5)).value();
  auto rejected = mismatch.add(RmsNorm{}, input, weight);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());

  GraphBuilder epsilon;
  auto epsilon_input = epsilon.input(row(4)).value();
  rejected = epsilon.add(
      RmsNorm{.epsilon = (std::numeric_limits<float>::quiet_NaN)()},
      epsilon_input);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  auto recovered = epsilon.add(RmsNorm{}, epsilon_input);
  ASSERT_TRUE(recovered.has_value());
}

TEST(Graph, InvalidAndOutOfRangeQueriesAreRecoverable) {
  Graph invalid;
  EXPECT_TRUE(invalid.input_id().error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.node_id().error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.output_id().error().IsFailedPrecondition());

  GraphBuilder builder;
  auto input = builder.input(row(4)).value();
  auto output = builder.add(Operation(RmsNorm{}), std::array{input}).value();
  ASSERT_TRUE(builder.output(output).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_TRUE(graph.input_id(1).error().IsOutOfRange());
  EXPECT_TRUE(graph.node_id(1).error().IsOutOfRange());
  EXPECT_TRUE(graph.output_id(1).error().IsOutOfRange());
}

}  // namespace
