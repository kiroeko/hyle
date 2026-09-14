#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "shape_program.h"

namespace {

using namespace hyle::compute::detail;

ShapeDimension derived(ShapeDimensionKind kind, ShapeOperand left,
                       ShapeOperand right, ProviderShapeRange left_bounds,
                       ProviderShapeRange right_bounds) {
  return {infer_shape_range(kind, left_bounds, right_bounds).value(), kind,
          left, right};
}

TEST(ShapeProgram, EvaluatesFiniteChainAndGuardsInCallerScratch) {
  std::array<ShapeDimension, 5> dimensions{};
  dimensions[0].bounds = {0, 4, 1};
  dimensions[1].bounds = {0, 8, 2};
  dimensions[2] = derived(ShapeDimensionKind::kMultiply, ShapeOperand::value(0),
                          ShapeOperand::value(1), dimensions[0].bounds,
                          dimensions[1].bounds);
  dimensions[3] =
      derived(ShapeDimensionKind::kExactDivide, ShapeOperand::value(2),
              ShapeOperand::fixed(2), dimensions[2].bounds, {2, 2, 1});
  dimensions[4] =
      derived(ShapeDimensionKind::kAdd, ShapeOperand::value(3),
              ShapeOperand::fixed(1), dimensions[3].bounds, {1, 1, 1});
  const std::array guards{
      ShapeGuard{ShapeGuardKind::kLessEqual, ShapeOperand::value(2),
                 ShapeOperand::fixed(16)},
      ShapeGuard{ShapeGuardKind::kEqual, ShapeOperand::value(0),
                 ShapeOperand::fixed(2)}};
  const ShapeProgramView program{dimensions, guards};
  ASSERT_TRUE(validate_shape_program(program).ok());
  EXPECT_EQ(program.input_count(), 2u);
  EXPECT_FALSE(program.exact());
  std::array<std::int64_t, 5> values{2, 6, -99, -99, -99};
  ASSERT_TRUE(evaluate_shape_program(program, values).ok());
  EXPECT_EQ(values, (std::array<std::int64_t, 5>{2, 6, 12, 6, 7}));
  values[0] = 4;
  EXPECT_TRUE(evaluate_shape_program(program, values).IsInvalidArgument());
  values[0] = -1;
  EXPECT_TRUE(evaluate_shape_program(program, values).IsOutOfRange());
  EXPECT_TRUE(evaluate_shape_program(program, {}).IsInvalidArgument());
}

TEST(ShapeProgram, RejectsMalformedProgramsWithoutReadingInvalidReferences) {
  std::array<ShapeDimension, 2> dimensions{};
  dimensions[0].bounds = {1, 4, 1};
  dimensions[1] =
      derived(ShapeDimensionKind::kAdd, ShapeOperand::value(0),
              ShapeOperand::fixed(1), dimensions[0].bounds, {1, 1, 1});
  const auto original = dimensions[1];
  const ShapeProgramView program{dimensions};
  ASSERT_TRUE(validate_shape_program(program).ok());
  dimensions[1].left = ShapeOperand::value(1);
  EXPECT_TRUE(validate_shape_program(program).IsInvalidArgument());
  dimensions[1].left = ShapeOperand::value(17);
  EXPECT_TRUE(validate_shape_program(program).IsInvalidArgument());
  dimensions[1] = original;
  dimensions[1].right = ShapeOperand::fixed(-1);
  EXPECT_TRUE(validate_shape_program(program).IsInvalidArgument());
  dimensions[1] = original;
  dimensions[1].left.constant = 1;
  EXPECT_TRUE(validate_shape_program(program).IsInvalidArgument());
  dimensions[1] = original;
  dimensions[1].bounds.maximum = 7;
  EXPECT_TRUE(validate_shape_program(program).IsInvalidArgument());
  dimensions[1] = original;
  dimensions[1].kind = static_cast<ShapeDimensionKind>(255);
  EXPECT_TRUE(validate_shape_program(program).IsInvalidArgument());
  dimensions[1] = original;
  dimensions[0].left = ShapeOperand::value(0);
  EXPECT_TRUE(validate_shape_program(program).IsInvalidArgument());
  dimensions[0].left = {};
  std::array guards{ShapeGuard{ShapeGuardKind::kEqual, ShapeOperand::value(2),
                               ShapeOperand::fixed(1)}};
  EXPECT_TRUE(validate_shape_program({dimensions, guards}).IsInvalidArgument());
  guards[0].left = ShapeOperand::fixed(2);
  EXPECT_TRUE(validate_shape_program({dimensions, guards}).IsInvalidArgument());
  guards[0].left = ShapeOperand::fixed(1);
  guards[0].kind = static_cast<ShapeGuardKind>(255);
  EXPECT_TRUE(validate_shape_program({dimensions, guards}).IsInvalidArgument());
}

TEST(ShapeProgram, ChecksOverflowAndZeroWithoutUndefinedIntegerArithmetic) {
  constexpr auto maximum = (std::numeric_limits<std::int64_t>::max)();
  const ProviderShapeRange large{maximum - 1, maximum, 1};
  EXPECT_TRUE(infer_shape_range(ShapeDimensionKind::kAdd, large, {1, 1, 1})
                  .error()
                  .IsOutOfRange());
  EXPECT_TRUE(infer_shape_range(ShapeDimensionKind::kMultiply, large, {2, 2, 1})
                  .error()
                  .IsOutOfRange());
  auto zero =
      infer_shape_range(ShapeDimensionKind::kMultiply, large, {0, 0, maximum});
  ASSERT_TRUE(zero.has_value());
  EXPECT_TRUE(zero->exact() && zero->contains(0));
  auto identity = infer_shape_range(ShapeDimensionKind::kAdd, large, {0, 0, 1});
  ASSERT_TRUE(identity.has_value());
  EXPECT_TRUE(identity->contains(maximum));
  EXPECT_FALSE(
      infer_shape_range(ShapeDimensionKind::kExactDivide, large, {0, 0, 1}));
  EXPECT_FALSE(infer_shape_range(ShapeDimensionKind::kExactDivide, {3, 3, 1},
                                 {2, 2, 1}));
  EXPECT_FALSE(
      infer_shape_range(ShapeDimensionKind::kBroadcast, {2, 2, 1}, {3, 3, 1}));
  auto empty_broadcast =
      infer_shape_range(ShapeDimensionKind::kBroadcast, {0, 0, 1}, {1, 1, 1});
  ASSERT_TRUE(empty_broadcast.has_value());
  EXPECT_TRUE(empty_broadcast->exact() && empty_broadcast->contains(0));
}

// Independent point oracle: small integers need no checked arithmetic and do
// not reuse the production interval/guard implementation.
std::optional<std::int64_t> oracle(ShapeDimensionKind kind, std::int64_t left,
                                   std::int64_t right) {
  switch (kind) {
  case ShapeDimensionKind::kAdd:
    return left + right;
  case ShapeDimensionKind::kMultiply:
    return left * right;
  case ShapeDimensionKind::kExactDivide:
    if (right != 0 && left % right == 0)
      return left / right;
    return {};
  case ShapeDimensionKind::kBroadcast:
    if (left == right)
      return left;
    if (left == 1)
      return right;
    if (right == 1)
      return left;
    return {};
  default:
    return {};
  }
}

TEST(ShapeProgram, SmallDomainEnumerationNeverExcludesALegalDerivedValue) {
  std::vector<ProviderShapeRange> ranges;
  for (std::int64_t minimum = 0; minimum <= 3; ++minimum) {
    for (std::int64_t maximum = minimum; maximum <= 4; ++maximum) {
      for (std::int64_t multiple = 1; multiple <= 3; ++multiple) {
        const ProviderShapeRange range{minimum, maximum, multiple};
        if (range.valid())
          ranges.push_back(range);
      }
    }
  }
  for (const auto left : ranges) {
    for (const auto right : ranges) {
      for (const auto kind :
           {ShapeDimensionKind::kAdd, ShapeDimensionKind::kMultiply,
            ShapeDimensionKind::kExactDivide, ShapeDimensionKind::kBroadcast}) {
        const auto inferred = infer_shape_range(kind, left, right);
        for (std::int64_t lhs = left.minimum; lhs <= left.maximum; ++lhs) {
          if (!left.contains(lhs))
            continue;
          for (std::int64_t rhs = right.minimum; rhs <= right.maximum; ++rhs) {
            if (!right.contains(rhs))
              continue;
            const auto expected = oracle(kind, lhs, rhs);
            if (expected) {
              ASSERT_TRUE(inferred.has_value())
                  << static_cast<int>(kind) << ':' << lhs << ',' << rhs;
              ASSERT_TRUE(inferred->contains(*expected));
            }
            if (!inferred)
              continue;
            const std::array dimensions{
                ShapeDimension{.bounds = left}, ShapeDimension{.bounds = right},
                ShapeDimension{*inferred, kind, ShapeOperand::value(0),
                               ShapeOperand::value(1)}};
            std::array<std::int64_t, 3> values{lhs, rhs, -1};
            const auto status = evaluate_shape_program({dimensions}, values);
            ASSERT_EQ(status.ok(), expected.has_value());
            if (expected)
              ASSERT_EQ(values[2], *expected);
          }
        }
      }
    }
  }
}

} // namespace
