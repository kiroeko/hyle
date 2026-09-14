#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/compute/tensor_desc.h"

namespace {

using hyle::compute::data_type_size;
using hyle::compute::DataType;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::TensorShape;

TEST(ComputeOpsTensorDesc, CoversEveryPublicScalarWidth) {
  EXPECT_EQ(data_type_size(DataType::kBool), 1u);
  EXPECT_EQ(data_type_size(DataType::kFloat8E4M3FN), 1u);
  EXPECT_EQ(data_type_size(DataType::kFloat8E5M2), 1u);
  EXPECT_EQ(data_type_size(DataType::kFloat16), 2u);
  EXPECT_EQ(data_type_size(DataType::kBFloat16), 2u);
  EXPECT_EQ(data_type_size(DataType::kFloat32), 4u);
  EXPECT_EQ(data_type_size(DataType::kFloat64), 8u);
  EXPECT_EQ(data_type_size(DataType::kInt8), 1u);
  EXPECT_EQ(data_type_size(DataType::kUInt8), 1u);
  EXPECT_EQ(data_type_size(DataType::kInt16), 2u);
  EXPECT_EQ(data_type_size(DataType::kUInt16), 2u);
  EXPECT_EQ(data_type_size(DataType::kInt32), 4u);
  EXPECT_EQ(data_type_size(DataType::kUInt32), 4u);
  EXPECT_EQ(data_type_size(DataType::kInt64), 8u);
  EXPECT_EQ(data_type_size(DataType::kUInt64), 8u);
  EXPECT_EQ(data_type_size(DataType::kUnknown), 0u);
}

TEST(ComputeOpsTensorDesc, OwnsPortableShapeAndStrideFactsWithoutStorage) {
  constexpr std::array<std::int64_t, 3> shape{2, 3, 4};
  auto result = TensorDesc::dense(DataType::kFloat32, shape);
  ASSERT_TRUE(result.has_value()) << result.error().message();

  const TensorDesc description = *result;
  constexpr std::array<std::int64_t, 3> expected_strides{48, 16, 4};
  EXPECT_TRUE(std::ranges::equal(description.dimensions(), shape));
  EXPECT_TRUE(std::ranges::equal(description.byte_strides(), expected_strides));
  EXPECT_EQ(description.element_count(), 24u);
  EXPECT_EQ(description.required_bytes(), 96u);
  EXPECT_TRUE(description.is_dense());
}

TEST(ComputeOpsTensorDesc, KeepsBlockedEncodingSeparateFromLogicalType) {
  constexpr StorageFormatId codec{0x48594c45U, 7U};
  const StorageFormat q4 = StorageFormat::blocked(codec, 32, 18, 2).value();
  constexpr std::array<std::int64_t, 2> shape{7, 64};
  const TensorDesc description =
      TensorDesc::blocked_dense(DataType::kFloat32, shape, q4).value();

  EXPECT_EQ(description.data_type(), DataType::kFloat32);
  EXPECT_EQ(description.storage_format(), q4);
  EXPECT_EQ(description.element_count(), 448u);
  EXPECT_EQ(description.required_bytes(), 252u);
  EXPECT_EQ(description.required_alignment(), 2u);
}

TEST(ComputeOpsTensorDesc, RejectsSchemasThatCannotBeRepresented) {
  constexpr std::array<std::int64_t, 1> shape{4};
  EXPECT_TRUE(
      TensorDesc::dense(DataType::kUnknown, shape).error().IsInvalidArgument());

  constexpr std::array<std::int64_t, 9> excessive_rank{};
  EXPECT_TRUE(TensorDesc::dense(DataType::kFloat32, excessive_rank)
                  .error()
                  .IsInvalidArgument());
}

TEST(ComputeOpsTensorDesc,
     ZeroExtentMakesElementCountZeroBeforeMultiplication) {
  constexpr std::array<std::int64_t, 3> dimensions{
      (std::numeric_limits<std::int64_t>::max)(),
      (std::numeric_limits<std::int64_t>::max)(), 0};
  auto shape = TensorShape::create(dimensions);
  ASSERT_TRUE(shape.has_value()) << shape.error().message();
  auto count = shape->element_count();
  ASSERT_TRUE(count.has_value()) << count.error().message();
  EXPECT_EQ(*count, 0u);
}

TEST(ComputeOpsTensorDesc, TransposeSwapsOnlyMatrixAxesForBatchedLayouts) {
  constexpr StorageFormatId codec{0x48594c45U, 23U};
  const StorageFormat q4 = StorageFormat::blocked(codec, 32, 18, 2).value();
  constexpr std::array<std::int64_t, 4> stored_shape{2, 3, 5, 64};
  const TensorDesc stored =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, q4).value();

  const TensorDesc logical = stored.transposed().value();
  constexpr std::array<std::int64_t, 4> logical_shape{2, 3, 64, 5};
  constexpr std::array<std::int64_t, 4> logical_strides{540, 180, 18, 36};
  EXPECT_TRUE(std::ranges::equal(logical.dimensions(), logical_shape));
  EXPECT_TRUE(std::ranges::equal(logical.byte_strides(), logical_strides));
  EXPECT_EQ(logical.block_axis(), 2u);
  EXPECT_EQ(logical.required_bytes(), stored.required_bytes());

  constexpr std::array<std::int64_t, 1> vector_shape{7};
  const TensorDesc vector =
      TensorDesc::dense(DataType::kFloat32, vector_shape).value();
  EXPECT_TRUE(vector.transposed().error().IsInvalidArgument());
}

} // namespace
