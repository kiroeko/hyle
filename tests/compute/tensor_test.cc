#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/compute/tensor.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::Buffer;
using hyle::mem::system_resource;

TEST(TensorDesc, DenseOwnsCanonicalShapeStrideAndByteFacts) {
  const std::array<std::int64_t, 3> shape{2, 3, 4};
  auto result = TensorDesc::dense(DataType::kFloat32, shape);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  const TensorDesc description = *result;

  EXPECT_EQ(description.data_type(), DataType::kFloat32);
  EXPECT_EQ(description.rank(), 3u);
  EXPECT_TRUE(std::ranges::equal(description.dimensions(), shape));
  const std::array<std::int64_t, 3> expected_strides{12, 4, 1};
  EXPECT_TRUE(
      std::ranges::equal(description.element_strides(), expected_strides));
  EXPECT_EQ(description.element_count().value(), 24u);
  EXPECT_EQ(description.required_bytes().value(), 96u);
  EXPECT_TRUE(description.is_dense());
}

TEST(TensorDesc, ScalarsAndZeroExtentsHaveExplicitStorageSemantics) {
  TensorDesc scalar = TensorDesc::dense(DataType::kFloat64, {}).value();
  EXPECT_EQ(scalar.rank(), 0u);
  EXPECT_EQ(scalar.element_count().value(), 1u);
  EXPECT_EQ(scalar.required_bytes().value(), sizeof(double));
  EXPECT_TRUE(scalar.is_dense());

  const std::array<std::int64_t, 3> shape{9, 0, 7};
  TensorDesc empty = TensorDesc::dense(DataType::kFloat32, shape).value();
  EXPECT_EQ(empty.element_count().value(), 0u);
  EXPECT_EQ(empty.required_bytes().value(), 0u);
  EXPECT_TRUE(empty.is_dense());
}

TEST(TensorDesc, StridedAndBroadcastDescriptionsReportPhysicalExtent) {
  const std::array<std::int64_t, 2> shape{2, 3};
  const std::array<std::int64_t, 2> strides{0, 1};
  TensorDesc broadcast =
      TensorDesc::strided(DataType::kFloat32, shape, strides).value();
  EXPECT_EQ(broadcast.element_count().value(), 6u);
  EXPECT_EQ(broadcast.required_bytes().value(), 3u * sizeof(float));
  EXPECT_FALSE(broadcast.is_dense());
}

TEST(TensorDesc, RejectsInvalidAndUnrepresentableSchemas) {
  const std::array<std::int64_t, 1> one{1};
  auto unknown = TensorDesc::dense(DataType::kUnknown, one);
  ASSERT_FALSE(unknown.has_value());
  EXPECT_TRUE(unknown.error().IsInvalidArgument());

  const std::array<std::int64_t, 1> negative{-1};
  auto negative_result = TensorDesc::dense(DataType::kFloat32, negative);
  ASSERT_FALSE(negative_result.has_value());
  EXPECT_TRUE(negative_result.error().IsInvalidArgument());

  const std::array<std::int64_t, 9> too_many{};
  auto rank = TensorDesc::dense(DataType::kFloat32, too_many);
  ASSERT_FALSE(rank.has_value());
  EXPECT_TRUE(rank.error().IsInvalidArgument());

  auto mismatch = TensorDesc::strided(DataType::kFloat32, one, {});
  ASSERT_FALSE(mismatch.has_value());
  EXPECT_TRUE(mismatch.error().IsInvalidArgument());

  const std::array<std::int64_t, 1> negative_stride{-1};
  auto stride =
      TensorDesc::strided(DataType::kFloat32, one, negative_stride);
  ASSERT_FALSE(stride.has_value());
  EXPECT_TRUE(stride.error().IsInvalidArgument());

  const std::array<std::int64_t, 2> huge{
      (std::numeric_limits<std::int64_t>::max)(),
      (std::numeric_limits<std::int64_t>::max)()};
  auto overflow = TensorDesc::dense(DataType::kFloat64, huge);
  ASSERT_FALSE(overflow.has_value());
  EXPECT_TRUE(overflow.error().IsOutOfRange());
}

TEST(TensorView, BorrowsAndNarrowsMemStorageWithoutTakingOwnership) {
  const std::array<std::int64_t, 1> shape{4};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer buffer = Buffer::create(system_resource(), 64).value();

  TensorView mutable_view =
      TensorView::create(description, buffer.view()).value();
  EXPECT_EQ(mutable_view.storage().size(), 4u * sizeof(float));
  EXPECT_EQ(mutable_view.description(), description);
  ConstTensorView const_view = mutable_view.as_const();
  EXPECT_EQ(const_view.storage().native_handle(), buffer.native_handle());
  EXPECT_EQ(const_view.storage().size(), 4u * sizeof(float));

  auto direct_const = ConstTensorView::create(description, buffer.view());
  ASSERT_TRUE(direct_const.has_value());
  EXPECT_EQ(direct_const->description(), description);
}

TEST(TensorView, RejectsMissingOrShortStorageAndSupportsOwnedEmptyBuffers) {
  const std::array<std::int64_t, 1> shape{4};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  auto missing = TensorView::create(description, {});
  ASSERT_FALSE(missing.has_value());
  EXPECT_TRUE(missing.error().IsInvalidArgument());

  Buffer short_buffer = Buffer::create(system_resource(), 15).value();
  auto short_view = TensorView::create(description, short_buffer.view());
  ASSERT_FALSE(short_view.has_value());
  EXPECT_TRUE(short_view.error().IsOutOfRange());

  const std::array<std::int64_t, 1> empty_shape{0};
  const TensorDesc empty_desc =
      TensorDesc::dense(DataType::kFloat32, empty_shape).value();
  Buffer empty_buffer = Buffer::create(system_resource(), 0).value();
  auto empty_view = TensorView::create(empty_desc, empty_buffer.view());
  ASSERT_TRUE(empty_view.has_value());
  EXPECT_TRUE(empty_view->storage().empty());
}

}  // namespace
