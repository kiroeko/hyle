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
  const std::array ranges{ProviderShapeRange{1, 8, 2},
                          ProviderShapeRange{1, 8, 2}};
  const std::array<std::int64_t, 2> shape{8, 8};
  const std::array upper{TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<ProviderTensorAxes, 1> input_axes{};
  input_axes[0].symbols[0] = 0;
  input_axes[0].symbols[1] = 1;
  std::array<ProviderTensorAxes, 1> output_axes{};
  output_axes[0].symbols[0] = 1;
  output_axes[0].symbols[1] = 0;
  const ProviderShapeDomain domain{ranges, input_axes, output_axes};
  ASSERT_TRUE(validate_shape_domain(domain, upper, upper).ok());
  std::array<TensorDesc, 1> inputs{}, outputs{};
  for (std::int64_t rows = 2; rows <= 8; rows += 2) {
    for (std::int64_t columns = 2; columns <= 8; columns += 2) {
      const std::array point{rows, columns};
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
  const std::array<std::int64_t, 2> nonmultiple{3, 4};
  EXPECT_TRUE(
      bind_shape_domain(domain, upper, upper, nonmultiple, inputs, outputs)
          .IsOutOfRange());
  const std::array<std::int64_t, 2> outside{10, 4};
  EXPECT_TRUE(bind_shape_domain(domain, upper, upper, outside, inputs, outputs)
                  .IsOutOfRange());
}

TEST(ProviderShapeDomain, RejectsMalformedMapsBeforeBinding) {
  std::array ranges{ProviderShapeRange{1, 8, 1}};
  const std::array<std::int64_t, 1> shape{8};
  const std::array upper{TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].symbols[0] = 0;
  ProviderShapeDomain domain{ranges, axes, {}};
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).ok());
  EXPECT_TRUE(validate_shape_domain(domain, {}, {}).IsInvalidArgument());
  axes[0].symbols[0] = 1;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  axes[0].symbols[0] = 0;
  axes[0].symbols[1] = 0;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  axes[0].symbols[1] = hyle::compute::detail::kFixedShapeAxis;
  ranges[0].maximum = 7;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  ranges[0] = ProviderShapeRange{9, 8, 1};
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsInvalidArgument());
  ranges[0] = ProviderShapeRange{1, 8, 1};
  const std::array<TensorDesc, 1> invalid{};
  EXPECT_TRUE(validate_shape_domain(domain, invalid, {}).IsInvalidArgument());
  std::array<TensorDesc, 1> concrete{};
  EXPECT_TRUE(bind_shape_domain(domain, upper, {}, {}, concrete, {})
                  .IsInvalidArgument());
  const std::array<std::int64_t, 1> point{4};
  EXPECT_TRUE(
      bind_shape_domain(domain, upper, {}, point, {}, {}).IsInvalidArgument());
}

TEST(ProviderShapeDomain, DenseBindingIncludesEmptyTensorsAndInPlaceScratch) {
  const std::array ranges{ProviderShapeRange{0, 8, 1}};
  const std::array<std::int64_t, 2> shape{8, 3};
  std::array upper{TensorDesc::dense(DataType::kFloat32, shape).value()};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].symbols[0] = 0;
  const ProviderShapeDomain domain{ranges, axes, {}};
  const std::array<std::int64_t, 1> empty{0};
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
  std::array ranges{ProviderShapeRange{32, 128, 32}};
  std::array<ProviderTensorAxes, 1> axes{};
  axes[0].symbols[1] = 0;
  const ProviderShapeDomain domain{ranges, axes, {}};
  std::array<TensorDesc, 1> concrete{};
  const std::array<std::int64_t, 1> point{64};
  ASSERT_TRUE(bind_shape_domain(domain, upper, {}, point, concrete, {}).ok());
  const std::array<std::int64_t, 2> expected{2, 64};
  EXPECT_EQ(
      concrete[0],
      TensorDesc::blocked_dense(DataType::kFloat32, expected, format).value());
  ranges[0].multiple_of = 16;
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
  std::array ranges{ProviderShapeRange{8, 8, 1}};
  std::array<ProviderTensorAxes, 2> axes{};
  axes[0].symbols[0] = axes[1].symbols[0] = 0;
  const ProviderShapeDomain domain{ranges, axes, {}};
  std::array<TensorDesc, 2> concrete{};
  const std::array<std::int64_t, 1> point{8};
  ASSERT_TRUE(bind_shape_domain(domain, upper, {}, point, concrete, {}).ok());
  EXPECT_EQ(concrete, upper);
  ranges[0].minimum = 1;
  EXPECT_TRUE(validate_shape_domain(domain, upper, {}).IsUnimplemented());
  const ProviderShapeDomain opaque_domain{ranges, std::span(axes).last(1), {}};
  EXPECT_TRUE(validate_shape_domain(opaque_domain, std::span(upper).last(1), {})
                  .IsUnimplemented());
}

} // namespace
