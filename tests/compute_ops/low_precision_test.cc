#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/operation.h"

namespace {
using namespace hyle::compute;

TEST(ComputeOpsLowPrecision, Float8PhysicalSizesAndLogicalOrientation) {
  const auto block = Float8ScaleMode::kFloat32Block128;
  const auto group = Float8ScaleMode::kUnsignedExponent8Group32;
  const auto activation = float8_activation_description(3, 256, block);
  const auto weight = float8_weight_description(256, 128, block);
  ASSERT_TRUE(activation);
  ASSERT_TRUE(weight);
  EXPECT_EQ(activation->data_type(), DataType::kFloat32);
  EXPECT_EQ(*activation->required_bytes(), 3u * 256 + 3u * 2 * 4);
  EXPECT_EQ(*weight->required_bytes(), 128u * 256 + 1u * 2 * 4);
  EXPECT_EQ(weight->dimensions()[0], 256);
  EXPECT_EQ(weight->dimensions()[1], 128);
  EXPECT_EQ(weight->required_alignment(), 4u);
  EXPECT_TRUE(weight->byte_strides().empty());
  EXPECT_NE(activation->storage_format().id(), weight->storage_format().id());
  const auto ue8_activation = float8_activation_description(3, 64, group);
  const auto ue8_weight = float8_weight_description(64, 5, group);
  ASSERT_TRUE(ue8_activation);
  ASSERT_TRUE(ue8_weight);
  EXPECT_EQ(*ue8_activation->required_bytes(), 3u * 64 + 3u * 2);
  EXPECT_EQ(*ue8_weight->required_bytes(), 5u * 64 + 5u * 2);
  EXPECT_NE(weight->storage_format().id(), ue8_weight->storage_format().id());
  const auto output = infer_single_output(Operation(MatMul{}),
                                          std::array{*activation, *weight});
  ASSERT_TRUE(output);
  EXPECT_EQ(output->dimensions()[0], 3);
  EXPECT_EQ(output->dimensions()[1], 128);
  EXPECT_TRUE(output->is_dense());
  EXPECT_EQ(output->data_type(), DataType::kFloat32);
}

TEST(ComputeOpsLowPrecision, Mxfp4BlockAxisFollowsLogicalK) {
  const auto weight = mxfp4_weight_description(64, 5);
  ASSERT_TRUE(weight);
  EXPECT_EQ(weight->data_type(), DataType::kFloat16);
  EXPECT_EQ(*weight->required_bytes(), 170u);
  ASSERT_TRUE(weight->block_axis());
  EXPECT_EQ(*weight->block_axis(), 0u);
  EXPECT_EQ(weight->byte_strides()[0], 17);
  EXPECT_EQ(weight->byte_strides()[1], 34);
  const auto physical = weight->transposed();
  ASSERT_TRUE(physical);
  EXPECT_TRUE(physical->is_dense());
  EXPECT_EQ(physical->dimensions()[0], 5);
  EXPECT_EQ(physical->dimensions()[1], 64);
  const auto activation =
      TensorDesc::dense(DataType::kFloat16, std::array<std::int64_t, 2>{3, 64});
  ASSERT_TRUE(activation);
  const auto output = infer_single_output(Operation(MatMul{}),
                                          std::array{*activation, *weight});
  ASSERT_TRUE(output);
  EXPECT_EQ(output->data_type(), DataType::kFloat16);
  EXPECT_EQ(output->dimensions()[0], 3);
  EXPECT_EQ(output->dimensions()[1], 5);
}

TEST(ComputeOpsLowPrecision, RejectsIncompleteBlocksAndOverflow) {
  const auto block = Float8ScaleMode::kFloat32Block128;
  const auto group = Float8ScaleMode::kUnsignedExponent8Group32;
  EXPECT_FALSE(float8_activation_description(0, 128, block));
  EXPECT_FALSE(float8_activation_description(-1, 128, block));
  EXPECT_FALSE(float8_activation_description(1, 129, block));
  EXPECT_FALSE(float8_weight_description(128, 1, block));
  EXPECT_FALSE(float8_weight_description(33, 1, group));
  EXPECT_FALSE(
      float8_weight_description(32, 1, static_cast<Float8ScaleMode>(255)));
  EXPECT_FALSE(mxfp4_weight_description(31, 1));
  EXPECT_FALSE(mxfp4_weight_description(32, 0));
  constexpr auto maximum = (std::numeric_limits<std::int64_t>::max)();
  EXPECT_FALSE(float8_activation_description(maximum, 128, block));
  EXPECT_FALSE(float8_weight_description(maximum - 31, 128, group));
  EXPECT_FALSE(mxfp4_weight_description(maximum - 31, maximum));
}
} // namespace
