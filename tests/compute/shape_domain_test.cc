#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

#include "shape_domain.h"

namespace {

using hyle::compute::DataType;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::detail::bind_shape_domain;
using hyle::compute::detail::ProviderShapeDomain;
using hyle::compute::detail::ProviderShapeRange;
using hyle::compute::detail::ProviderTensorAxes;
using hyle::compute::detail::ProviderTensorLayout;
using hyle::compute::detail::ShapeDimension;
using hyle::compute::detail::validate_shape_binding;
using hyle::compute::detail::validate_shape_domain;

TEST(ProviderShapeDomain, RangesAdmitMultiplesWithoutEndpointOverflow) {
  constexpr ProviderShapeRange rounded{3, 11, 4};
  static_assert(rounded.valid() && rounded.first() == 4 && rounded.last() == 8);
  static_assert(!rounded.exact() && rounded.contains(4) && rounded.contains(8));
  EXPECT_FALSE(rounded.contains(3));
  EXPECT_FALSE(rounded.contains(12));
  constexpr ProviderShapeRange singleton{3, 7, 4};
  static_assert(singleton.exact() && singleton.first() == 4);
  constexpr auto maximum = (std::numeric_limits<std::int64_t>::max)();
  constexpr ProviderShapeRange extreme{maximum - 1, maximum, maximum};
  static_assert(extreme.valid() && extreme.first() == maximum &&
                extreme.exact());
  constexpr ProviderShapeRange zero{0, 0, maximum};
  static_assert(zero.valid() && zero.contains(0) && zero.exact());
  constexpr std::array invalid{
      ProviderShapeRange{-1, 8, 1}, ProviderShapeRange{8, 2, 1},
      ProviderShapeRange{1, 8, 0},  ProviderShapeRange{1, 8, -1},
      ProviderShapeRange{5, 7, 4},  ProviderShapeRange{maximum, maximum, 2}};
  for (const auto range : invalid) {
    EXPECT_FALSE(range.valid());
    EXPECT_FALSE(range.exact());
    EXPECT_FALSE(range.contains(0));
  }
}

TEST(ProviderShapeDomain,
     RepeatedIndicesShareValuesAndEqualRangesStayIndependent) {
  const std::array dimensions{ShapeDimension{.bounds = {1, 8, 2}},
                              ShapeDimension{.bounds = {1, 8, 2}}};
  const std::array<std::int64_t, 2> shape{8, 8};
  const std::array upper{TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<ProviderTensorAxes, 1> input_axes{};
  input_axes[0].values[0] = 0;
  input_axes[0].values[1] = 1;
  std::array<ProviderTensorAxes, 1> output_axes{};
  output_axes[0].values[0] = 1;
  output_axes[0].values[1] = 0;
  const ProviderShapeDomain domain{dimensions, input_axes, output_axes};
  ASSERT_TRUE(validate_shape_domain(domain, upper, upper).ok());
  std::array<TensorDesc, 1> inputs{}, outputs{};
  for (std::int64_t rows = 2; rows <= 8; rows += 2) {
    for (std::int64_t columns = 2; columns <= 8; columns += 2) {
      std::array point{rows, columns};
      ASSERT_TRUE(
          bind_shape_domain(domain, upper, upper, point, inputs, outputs).ok());
      const std::array expected_input{rows, columns};
      const std::array expected_output{columns, rows};
      EXPECT_EQ(inputs[0],
                TensorDesc::dense(DataType::kFloat32, expected_input).value());
      EXPECT_EQ(outputs[0],
                TensorDesc::dense(DataType::kFloat32, expected_output).value());
    }
  }
  std::array<std::int64_t, 2> nonmultiple{3, 4};
  EXPECT_TRUE(
      bind_shape_domain(domain, upper, upper, nonmultiple, inputs, outputs)
          .IsOutOfRange());
  std::array<std::int64_t, 2> outside{10, 4};
  EXPECT_TRUE(bind_shape_domain(domain, upper, upper, outside, inputs, outputs)
                  .IsOutOfRange());
}

TEST(ProviderShapeDomain,
     DerivedAxesAndGuardsDoNotRequireAnAdmittedEnvelopePoint) {
  using namespace hyle::compute::detail;
  const ProviderShapeRange left{0, 4, 1}, right{0, 8, 1};
  const std::array dimensions{
      ShapeDimension{.bounds = left}, ShapeDimension{.bounds = right},
      ShapeDimension{
          infer_shape_range(ShapeDimensionKind::kAdd, left, right).value(),
          ShapeDimensionKind::kAdd, ShapeOperand::value(0),
          ShapeOperand::value(1)}};
  const std::array guards{ShapeGuard{ShapeGuardKind::kLessEqual,
                                     ShapeOperand::value(1),
                                     ShapeOperand::value(0)}};
  const auto desc = [](std::int64_t rows) {
    return TensorDesc::dense(DataType::kFloat32,
                             std::array{rows, std::int64_t{32}})
        .value();
  };
  const std::array upper_inputs{desc(4), desc(8)};
  const std::array upper_outputs{desc(12)};
  std::array<ProviderTensorAxes, 2> input_axes{};
  std::array<ProviderTensorAxes, 1> output_axes{};
  input_axes[0].values[0] = 0;
  input_axes[1].values[0] = 1;
  output_axes[0].values[0] = 2;
  const ProviderShapeDomain domain{dimensions, input_axes, output_axes, guards};
  ASSERT_TRUE(validate_shape_domain(domain, upper_inputs, upper_outputs).ok());
  std::array<TensorDesc, 2> inputs{};
  std::array<TensorDesc, 1> outputs{};
  std::array<std::int64_t, 3> values{2, 1, -1};
  ASSERT_TRUE(bind_shape_domain(domain, upper_inputs, upper_outputs, values,
                                inputs, outputs)
                  .ok());
  EXPECT_EQ(values[2], 3);
  EXPECT_EQ(inputs[0], desc(2));
  EXPECT_EQ(inputs[1], desc(1));
  EXPECT_EQ(outputs[0], desc(3));
  values = {4, 8, -1};
  EXPECT_TRUE(bind_shape_domain(domain, upper_inputs, upper_outputs, values,
                                inputs, outputs)
                  .IsInvalidArgument());
}

TEST(ProviderShapeDomain, RejectsMalformedMapsBeforeBinding) {
  std::array dimensions{ShapeDimension{.bounds = {1, 8, 1}}};
  const std::array<std::int64_t, 1> shape{8};
  const std::array upper{TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].values[0] = 0;
  ProviderShapeDomain domain{dimensions, axes, {}};
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).ok());
  EXPECT_TRUE(validate_shape_domain(domain, {}, {}).IsInvalidArgument());
  axes[0].values[0] = 1;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  axes[0].values[0] = 0;
  axes[0].values[1] = 0;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  axes[0].values[1] = hyle::compute::detail::kFixedShapeAxis;
  dimensions[0].bounds.maximum = 7;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  dimensions[0].bounds = ProviderShapeRange{9, 8, 1};
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  dimensions[0].bounds = ProviderShapeRange{1, 8, 1};
  const std::array<TensorDesc, 1> invalid{};
  EXPECT_TRUE(validate_shape_domain(domain, invalid, {}).IsInvalidArgument());
  std::array<TensorDesc, 1> concrete{};
  EXPECT_TRUE(bind_shape_domain(domain, upper, {}, {}, concrete, {})
                  .IsInvalidArgument());
  std::array<std::int64_t, 1> point{4};
  EXPECT_TRUE(
      bind_shape_domain(domain, upper, {}, point, {}, {}).IsInvalidArgument());
}

TEST(ProviderShapeDomain, DenseBindingIncludesEmptyTensorsAndInPlaceScratch) {
  const std::array dimensions{ShapeDimension{.bounds = {0, 8, 1}}};
  const std::array<std::int64_t, 2> shape{8, 3};
  std::array upper{TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].values[0] = 0;
  const ProviderShapeDomain domain{dimensions, axes, {}};
  std::array<std::int64_t, 1> empty{0};
  ASSERT_TRUE(bind_shape_domain(domain, upper, {}, empty, upper, {}).ok());
  EXPECT_EQ(upper[0].dimensions()[0], 0);
  EXPECT_EQ(upper[0].required_bytes().value(), 0u);
  EXPECT_TRUE(upper[0].is_dense());
  EXPECT_TRUE(bind_shape_domain({}, upper, {}, {}, upper, {}).ok());
  EXPECT_EQ(upper[0].dimensions()[0], 0);
}

TEST(ProviderShapeDomain, BlockedDimensionsRequireWholeBlocksAcrossTheDomain) {
  const auto format =
      StorageFormat::blocked(StorageFormatId{0x48594c45U, 11U}, 32, 18, 2)
          .value();
  const std::array<std::int64_t, 2> shape{2, 128};
  const std::array upper{
      TensorDesc::blocked_dense(DataType::kFloat32, shape, format).value()};
  std::array dimensions{ShapeDimension{.bounds = {32, 128, 32}}};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].values[1] = 0;
  const ProviderShapeDomain domain{dimensions, axes, {}};
  std::array<TensorDesc, 1> concrete{};
  std::array<std::int64_t, 1> point{64};
  ASSERT_TRUE(bind_shape_domain(domain, upper, {}, point, concrete, {}).ok());
  const std::array<std::int64_t, 2> expected{2, 64};
  EXPECT_EQ(
      concrete[0],
      TensorDesc::blocked_dense(DataType::kFloat32, expected, format).value());
  dimensions[0].bounds.multiple_of = 16;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
}

TEST(ProviderShapeDomain,
     ExactStridesAndOpaqueStorageArePreservedNotRecreated) {
  const std::array<std::int64_t, 2> shape{8, 3};
  const std::array<std::int64_t, 2> strides{16, 4};
  const auto opaque =
      StorageFormat::opaque(StorageFormatId{0x48594c45U, 19U}, 16).value();
  const std::array upper{
      TensorDesc::strided_bytes(DataType::kFloat32, shape, strides).value(),
      TensorDesc::opaque(DataType::kFloat32, shape, opaque, 256).value()};
  std::array dimensions{ShapeDimension{.bounds = {8, 8, 1}}};
  std::array<ProviderTensorAxes, 2> axes{};
  axes[0].values[0] = axes[1].values[0] = 0;
  const ProviderShapeDomain domain{dimensions, axes, {}};
  std::array<TensorDesc, 2> concrete{};
  std::array<std::int64_t, 1> point{8};
  ASSERT_TRUE(bind_shape_domain(domain, upper, {}, point, concrete, {}).ok());
  EXPECT_EQ(concrete, upper);
  dimensions[0].bounds.minimum = 1;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsUnimplemented());
  const ProviderShapeDomain opaque_domain{
      dimensions, std::span(axes).last(1), {}};
  EXPECT_TRUE(validate_shape_domain(opaque_domain, std::span(upper).last(1), {})
                  .IsUnimplemented());
}

TEST(ProviderShapeDomain, SuppliedDenseLayoutsMatchAutomaticBinding) {
  const std::array dimensions{ShapeDimension{.bounds = {0, 4, 1}},
                              ShapeDimension{.bounds = {0, 3, 1}}};
  const auto dense = [](std::int64_t rows, std::int64_t columns) {
    return TensorDesc::dense(DataType::kFloat32, std::array{rows, columns})
        .value();
  };
  const std::array upper_inputs{dense(4, 3)}, upper_outputs{dense(3, 4)};
  std::array<ProviderTensorAxes, 1> in_axes{}, out_axes{};
  in_axes[0].values[0] = out_axes[0].values[1] = 0;
  in_axes[0].values[1] = out_axes[0].values[0] = 1;
  const ProviderShapeDomain domain{dimensions, in_axes, out_axes};
  for (std::int64_t rows = 0; rows <= 4; ++rows) {
    for (std::int64_t columns = 0; columns <= 3; ++columns) {
      std::array point{rows, columns};
      std::array<TensorDesc, 1> inputs{}, outputs{};
      ASSERT_TRUE(bind_shape_domain(domain, upper_inputs, upper_outputs, point,
                                    inputs, outputs)
                      .ok());
      EXPECT_TRUE(validate_shape_binding(domain, upper_inputs, upper_outputs,
                                         point, inputs, outputs)
                      .ok());
      inputs[0] = TensorDesc::strided_bytes(DataType::kFloat32,
                                            std::array{rows, columns},
                                            std::array<std::int64_t, 2>{64, 4})
                      .value();
      EXPECT_TRUE(validate_shape_binding(domain, upper_inputs, upper_outputs,
                                         point, inputs, outputs)
                      .IsInvalidArgument());
    }
  }
}

TEST(ProviderShapeDomain, FixedExtentsDoNotMakeAStridedFamilyExact) {
  const auto upper_value =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{3, 4})
          .value();
  const std::array upper{upper_value};
  std::array<ProviderTensorAxes, 1> maps{};
  maps[0].layout = ProviderTensorLayout::kBoundedStrided;
  maps[0].maximum_bytes = 96;
  const ProviderShapeDomain domain{{}, maps, {}};
  EXPECT_FALSE(domain.empty());
  EXPECT_FALSE(domain.exact());
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).ok());
  std::array<TensorDesc, 1> automatic{};
  EXPECT_TRUE(bind_shape_domain(domain, upper, {}, {}, automatic, {})
                  .IsFailedPrecondition());
  for (const auto strides :
       {std::array<std::int64_t, 2>{16, 4}, std::array<std::int64_t, 2>{4, 12},
        std::array<std::int64_t, 2>{32, 4},
        std::array<std::int64_t, 2>{0, 4}}) {
    const std::array concrete{
        TensorDesc::strided_bytes(DataType::kFloat32, upper_value.dimensions(),
                                  strides)
            .value()};
    EXPECT_TRUE(
        validate_shape_binding(domain, upper, {}, {}, concrete, {}).ok());
  }
}

TEST(ProviderShapeDomain,
     StridedFamilyChecksReachableSpanIncludingEmptyAndZeroStrides) {
  const std::array dimensions{ShapeDimension{.bounds = {0, 3, 1}},
                              ShapeDimension{.bounds = {0, 4, 1}}};
  const std::array upper{
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{3, 4})
          .value()};
  std::array<ProviderTensorAxes, 1> maps{};
  maps[0].values[0] = 0;
  maps[0].values[1] = 1;
  maps[0].layout = ProviderTensorLayout::kBoundedStrided;
  maps[0].maximum_bytes = 128;
  const ProviderShapeDomain domain{dimensions, maps, {}};
  for (std::int64_t rows = 0; rows <= 3; ++rows) {
    for (std::int64_t columns = 0; columns <= 4; ++columns) {
      for (std::int64_t row_stride : {0, 4, 16, 64}) {
        for (std::int64_t column_stride : {0, 4, 16}) {
          std::array point{rows, columns};
          const std::array concrete{
              TensorDesc::strided_bytes(DataType::kFloat32, point,
                                        std::array{row_stride, column_stride})
                  .value()};
          const auto status =
              validate_shape_binding(domain, upper, {}, point, concrete, {});
          const auto span =
              rows == 0 || columns == 0
                  ? 0
                  : (rows - 1) * row_stride + (columns - 1) * column_stride + 4;
          EXPECT_EQ(status.ok(), span <= 128);
          if (span > 128)
            EXPECT_TRUE(status.IsOutOfRange());
        }
      }
    }
  }
  std::array<std::int64_t, 2> point{3, 4};
  const auto opaque = StorageFormat::opaque(StorageFormatId{123, 1}, 4).value();
  for (const auto wrong :
       {TensorDesc{}, TensorDesc::dense(DataType::kInt32, point).value(),
        TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{12})
            .value(),
        TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{3, 3})
            .value(),
        TensorDesc::opaque(DataType::kFloat32, point, opaque, 48).value()}) {
    EXPECT_TRUE(
        validate_shape_binding(domain, upper, {}, point, std::array{wrong}, {})
            .IsInvalidArgument());
  }
  EXPECT_TRUE(validate_shape_binding(domain, upper, {}, point, {}, {})
                  .IsInvalidArgument());
  EXPECT_TRUE(validate_shape_binding(domain, upper, {}, {}, upper, {})
                  .IsInvalidArgument());
}

TEST(ProviderShapeDomain, RejectsMalformedLayoutFamiliesAndFalseBounds) {
  const std::array<std::int64_t, 1> shape{4};
  std::array upper{TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<ProviderTensorAxes, 1> maps{};
  const ProviderShapeDomain domain{{}, maps, {}};
  maps[0].maximum_bytes = 16;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  maps[0].layout = static_cast<ProviderTensorLayout>(255);
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  maps[0].layout = ProviderTensorLayout::kBoundedStrided;
  maps[0].maximum_bytes = 15;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  maps[0].maximum_bytes = 16;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).ok());
  const auto format =
      StorageFormat::blocked(StorageFormatId{123, 2}, 4, 4, 1).value();
  upper[0] =
      TensorDesc::blocked_dense(DataType::kFloat32, shape, format).value();
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsUnimplemented());
  maps[0].maximum_bytes = 0;
  upper[0] =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{0})
          .value();
  EXPECT_TRUE(validate_shape_binding(domain, upper, {}, {}, upper, {}).ok());
}

TEST(ProviderShapeDomain, SuppliedLayoutsEvaluateDerivedAxesAndGuards) {
  using namespace hyle::compute::detail;
  const ProviderShapeRange left{0, 4, 1}, right{0, 3, 1};
  const std::array dimensions{
      ShapeDimension{.bounds = left}, ShapeDimension{.bounds = right},
      ShapeDimension{
          infer_shape_range(ShapeDimensionKind::kMultiply, left, right).value(),
          ShapeDimensionKind::kMultiply, ShapeOperand::value(0),
          ShapeOperand::value(1)}};
  const std::array guards{ShapeGuard{ShapeGuardKind::kLessEqual,
                                     ShapeOperand::value(0),
                                     ShapeOperand::value(1)}};
  const std::array upper{
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{12, 4})
          .value()};
  const std::array concrete{
      TensorDesc::strided_bytes(DataType::kFloat32,
                                std::array<std::int64_t, 2>{6, 4},
                                std::array<std::int64_t, 2>{32, 4})
          .value()};
  std::array<ProviderTensorAxes, 1> maps{};
  maps[0].values[0] = 2;
  maps[0].layout = ProviderTensorLayout::kBoundedStrided;
  maps[0].maximum_bytes = 256;
  const ProviderShapeDomain domain{dimensions, maps, {}, guards};
  std::array<std::int64_t, 3> point{2, 3, -99};
  EXPECT_TRUE(
      validate_shape_binding(domain, upper, {}, point, concrete, {}).ok());
  EXPECT_EQ(point[2], 6);
  point = {4, 3, -99};
  EXPECT_TRUE(validate_shape_binding(domain, upper, {}, point, concrete, {})
                  .IsInvalidArgument());
  point = {2, 4, -99};
  EXPECT_TRUE(validate_shape_binding(domain, upper, {}, point, concrete, {})
                  .IsOutOfRange());
}

TEST(ProviderShapeDomain, SuppliedExactDescriptionsDoNotChangeLayout) {
  const std::array<std::int64_t, 2> shape{1, 4};
  const std::array upper{
      TensorDesc::strided_bytes(DataType::kFloat32, shape,
                                std::array<std::int64_t, 2>{128, 4})
          .value()};
  EXPECT_TRUE(validate_shape_binding({}, upper, upper, {}, upper, upper).ok());
  const std::array changed{
      TensorDesc::dense(DataType::kFloat32, shape).value()};
  EXPECT_TRUE(validate_shape_binding({}, upper, upper, {}, changed, upper)
                  .IsInvalidArgument());
  EXPECT_TRUE(validate_shape_binding({}, upper, upper, {}, upper, changed)
                  .IsInvalidArgument());
}

} // namespace
