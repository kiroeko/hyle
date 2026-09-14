#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/compute/tensor.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::StorageFormatKind;
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
  const std::array<std::int64_t, 3> expected_strides{48, 16, 4};
  EXPECT_TRUE(std::ranges::equal(description.byte_strides(), expected_strides));
  EXPECT_EQ(description.element_count().value(), 24u);
  EXPECT_EQ(description.required_bytes().value(), 96u);
  EXPECT_TRUE(description.is_dense());
}

TEST(TensorDesc, EqualityUsesEveryLogicalShapeAndStrideFact) {
  const std::array<std::int64_t, 2> shape{2, 3};
  const TensorDesc dense = TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc same = TensorDesc::dense(DataType::kFloat32, shape).value();
  const std::array<std::int64_t, 2> padded_strides{16, 4};
  const TensorDesc padded =
      TensorDesc::strided_bytes(DataType::kFloat32, shape, padded_strides)
          .value();
  const std::array<std::int64_t, 2> other_shape{3, 2};
  const TensorDesc other =
      TensorDesc::dense(DataType::kFloat32, other_shape).value();

  EXPECT_EQ(dense, same);
  EXPECT_NE(dense, padded);
  EXPECT_NE(dense, other);
  EXPECT_NE(dense, TensorDesc::dense(DataType::kFloat64, shape).value());
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

  const std::array<std::int64_t, 3> extreme_empty{
      0, (std::numeric_limits<std::int64_t>::max)(),
      (std::numeric_limits<std::int64_t>::max)()};
  TensorDesc safe_empty =
      TensorDesc::dense(DataType::kFloat64, extreme_empty).value();
  const std::array<std::int64_t, 3> zero_strides{};
  EXPECT_TRUE(std::ranges::equal(safe_empty.byte_strides(), zero_strides));
  EXPECT_EQ(safe_empty.element_count().value(), 0u);
  EXPECT_EQ(safe_empty.required_bytes().value(), 0u);
  EXPECT_TRUE(safe_empty.is_dense());
}

TEST(TensorDesc, StridedAndBroadcastDescriptionsReportPhysicalExtent) {
  const std::array<std::int64_t, 2> shape{2, 3};
  const std::array<std::int64_t, 2> strides{0, 4};
  TensorDesc broadcast =
      TensorDesc::strided_bytes(DataType::kFloat32, shape, strides).value();
  EXPECT_EQ(broadcast.element_count().value(), 6u);
  EXPECT_EQ(broadcast.required_bytes().value(), 3u * sizeof(float));
  EXPECT_FALSE(broadcast.is_dense());
}

TEST(TensorDesc, PermutationDescribesZeroCopyLogicalViews) {
  const std::array<std::int64_t, 3> shape{2, 3, 4};
  const TensorDesc dense = TensorDesc::dense(DataType::kFloat32, shape).value();
  const std::array<std::size_t, 3> axes{2, 0, 1};
  const TensorDesc permuted = dense.permuted(axes).value();
  const std::array<std::int64_t, 3> expected_shape{4, 2, 3};
  const std::array<std::int64_t, 3> expected_strides{4, 48, 16};
  EXPECT_TRUE(std::ranges::equal(permuted.dimensions(), expected_shape));
  EXPECT_TRUE(std::ranges::equal(permuted.byte_strides(), expected_strides));
  EXPECT_EQ(permuted.required_bytes(), dense.required_bytes());
  EXPECT_FALSE(permuted.is_dense());

  const std::array<std::int64_t, 2> matrix_shape{7, 11};
  const TensorDesc matrix =
      TensorDesc::dense(DataType::kFloat32, matrix_shape).value();
  const TensorDesc transposed = matrix.transposed().value();
  const std::array<std::int64_t, 2> transposed_shape{11, 7};
  const std::array<std::int64_t, 2> transposed_strides{4, 44};
  EXPECT_TRUE(std::ranges::equal(transposed.dimensions(), transposed_shape));
  EXPECT_TRUE(
      std::ranges::equal(transposed.byte_strides(), transposed_strides));
  EXPECT_EQ(transposed.required_bytes(), matrix.required_bytes());
}

TEST(TensorDesc, RejectsInvalidPermutationsAndRankOneTranspose) {
  const TensorDesc invalid;
  EXPECT_TRUE(invalid.permuted({}).error().IsFailedPrecondition());

  const std::array<std::int64_t, 3> shape{2, 3, 4};
  const TensorDesc tensor =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const std::array<std::size_t, 2> short_axes{0, 1};
  EXPECT_TRUE(tensor.permuted(short_axes).error().IsInvalidArgument());
  const std::array<std::size_t, 3> duplicate_axes{0, 0, 2};
  EXPECT_TRUE(tensor.permuted(duplicate_axes).error().IsInvalidArgument());
  const std::array<std::size_t, 3> out_of_range_axes{0, 1, 3};
  EXPECT_TRUE(tensor.permuted(out_of_range_axes).error().IsInvalidArgument());
  const TensorDesc batched_transpose = tensor.transposed().value();
  const std::array<std::int64_t, 3> expected_shape{2, 4, 3};
  EXPECT_TRUE(
      std::ranges::equal(batched_transpose.dimensions(), expected_shape));

  const std::array<std::int64_t, 1> vector_shape{4};
  const TensorDesc vector =
      TensorDesc::dense(DataType::kFloat32, vector_shape).value();
  EXPECT_TRUE(vector.transposed().error().IsInvalidArgument());
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

  auto mismatch = TensorDesc::strided_bytes(DataType::kFloat32, one, {});
  ASSERT_FALSE(mismatch.has_value());
  EXPECT_TRUE(mismatch.error().IsInvalidArgument());

  const std::array<std::int64_t, 1> negative_stride{-1};
  auto stride =
      TensorDesc::strided_bytes(DataType::kFloat32, one, negative_stride);
  ASSERT_FALSE(stride.has_value());
  EXPECT_TRUE(stride.error().IsInvalidArgument());

  const std::array<std::int64_t, 2> huge{
      (std::numeric_limits<std::int64_t>::max)(),
      (std::numeric_limits<std::int64_t>::max)()};
  auto overflow = TensorDesc::dense(DataType::kFloat64, huge);
  ASSERT_FALSE(overflow.has_value());
  EXPECT_TRUE(overflow.error().IsOutOfRange());

  const std::array<std::int64_t, 2> huge_strides{
      (std::numeric_limits<std::int64_t>::max)(),
      (std::numeric_limits<std::int64_t>::max)()};
  auto strided_overflow =
      TensorDesc::strided_bytes(DataType::kUInt8, huge, huge_strides);
  ASSERT_FALSE(strided_overflow.has_value());
  EXPECT_TRUE(strided_overflow.error().IsOutOfRange());
}

TEST(StorageFormat, ValidatesStableBlockedAndOpaqueIdentities) {
  constexpr StorageFormatId codec{0x48594c45U, 7U};
  StorageFormat blocked = StorageFormat::blocked(codec, 32, 18, 2).value();
  EXPECT_EQ(blocked.kind(), StorageFormatKind::kBlocked);
  EXPECT_EQ(blocked.id(), codec);
  EXPECT_EQ(blocked.elements_per_block(), 32u);
  EXPECT_EQ(blocked.bytes_per_block(), 18u);
  EXPECT_EQ(blocked.required_alignment(), 2u);

  StorageFormat opaque = StorageFormat::opaque(codec, 64).value();
  EXPECT_EQ(opaque.kind(), StorageFormatKind::kOpaque);
  EXPECT_EQ(opaque.required_alignment(), 64u);
  EXPECT_NE(blocked, opaque);

  EXPECT_TRUE(
      StorageFormat::blocked({}, 32, 18, 2).error().IsInvalidArgument());
  EXPECT_TRUE(
      StorageFormat::blocked(codec, 0, 18, 2).error().IsInvalidArgument());
  EXPECT_TRUE(
      StorageFormat::blocked(codec, 32, 18, 3).error().IsInvalidArgument());
  EXPECT_TRUE(StorageFormat::opaque(codec, 0).error().IsInvalidArgument());
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    const std::size_t above_u32 =
        static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)()) +
        1;
    EXPECT_TRUE(
        StorageFormat::blocked(codec, above_u32, 18, 2).error().IsOutOfRange());
  }
}

TEST(TensorDesc, BlockedLayoutKeepsLogicalShapeAndPhysicalBlockFactsApart) {
  constexpr StorageFormatId codec{0x48594c45U, 11U};
  const StorageFormat q4 = StorageFormat::blocked(codec, 32, 18, 2).value();
  const std::array<std::int64_t, 2> stored_shape{7, 64};
  const TensorDesc stored =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, q4).value();

  EXPECT_EQ(stored.data_type(), DataType::kFloat32);
  EXPECT_EQ(stored.storage_format(), q4);
  EXPECT_FALSE(stored.is_plain());
  EXPECT_EQ(stored.block_axis(), 1u);
  const std::array<std::int64_t, 2> stored_strides{36, 18};
  EXPECT_TRUE(std::ranges::equal(stored.byte_strides(), stored_strides));
  EXPECT_EQ(stored.element_count(), 448u);
  EXPECT_EQ(stored.required_bytes(), 252u);
  EXPECT_EQ(stored.required_alignment(), 2u);
  EXPECT_TRUE(stored.is_dense());

  const TensorDesc logical = stored.transposed().value();
  const std::array<std::int64_t, 2> logical_shape{64, 7};
  const std::array<std::int64_t, 2> logical_strides{18, 36};
  EXPECT_TRUE(std::ranges::equal(logical.dimensions(), logical_shape));
  EXPECT_TRUE(std::ranges::equal(logical.byte_strides(), logical_strides));
  EXPECT_EQ(logical.block_axis(), 0u);
  EXPECT_EQ(logical.required_bytes(), stored.required_bytes());
  EXPECT_FALSE(logical.is_dense());

  const std::array<std::int64_t, 2> indivisible{7, 65};
  EXPECT_TRUE(TensorDesc::blocked_dense(DataType::kFloat32, indivisible, q4)
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(TensorDesc::blocked_strided(DataType::kFloat32, stored_shape, q4,
                                          2, stored_strides)
                  .error()
                  .IsInvalidArgument());

  const StorageFormat other_id =
      StorageFormat::blocked(StorageFormatId{0x48594c45U, 12U}, 32, 18, 2)
          .value();
  EXPECT_NE(stored, TensorDesc::blocked_dense(DataType::kFloat32, stored_shape,
                                              other_id)
                        .value());
}

TEST(TensorDesc, OpaqueLayoutExposesOnlyIdentityAlignmentAndExactExtent) {
  constexpr StorageFormatId codec{0x48594c45U, 19U};
  const StorageFormat packed = StorageFormat::opaque(codec, 16).value();
  const std::array<std::int64_t, 2> shape{3, 7};
  const TensorDesc description =
      TensorDesc::opaque(DataType::kFloat32, shape, packed, 127).value();

  EXPECT_EQ(description.storage_format(), packed);
  EXPECT_TRUE(description.byte_strides().empty());
  EXPECT_FALSE(description.block_axis().has_value());
  EXPECT_EQ(description.element_count(), 21u);
  EXPECT_EQ(description.required_bytes(), 127u);
  EXPECT_EQ(description.required_alignment(), 16u);
  EXPECT_FALSE(description.is_dense());
  EXPECT_TRUE(description.transposed().error().IsUnimplemented());

  EXPECT_TRUE(TensorDesc::opaque(DataType::kFloat32, shape, StorageFormat{}, 1)
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(TensorDesc::opaque(DataType::kFloat32, shape, packed, 0)
                  .error()
                  .IsInvalidArgument());

  const std::array<std::int64_t, 2> empty_shape{3, 0};
  EXPECT_EQ(TensorDesc::opaque(DataType::kFloat32, empty_shape, packed, 0)
                ->required_bytes()
                .value(),
            0u);
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

TEST(TensorView, RejectsObviouslyMisalignedPhysicalSubviewOffsets) {
  constexpr StorageFormatId codec{0x48594c45U, 23U};
  const StorageFormat packed = StorageFormat::opaque(codec, 16).value();
  const std::array<std::int64_t, 1> shape{8};
  const TensorDesc description =
      TensorDesc::opaque(DataType::kFloat32, shape, packed, 64).value();
  Buffer buffer = Buffer::create(system_resource(), 128, 64).value();

  auto misaligned_storage = buffer.view().subview(1, 64).value();
  auto misaligned = TensorView::create(description, misaligned_storage);
  ASSERT_FALSE(misaligned.has_value());
  EXPECT_TRUE(misaligned.error().IsInvalidArgument());

  auto aligned_storage = buffer.view().subview(16, 96).value();
  TensorView aligned = TensorView::create(description, aligned_storage).value();
  EXPECT_EQ(aligned.storage().offset(), 16u);
  EXPECT_EQ(aligned.storage().size(), 64u);
}

} // namespace
