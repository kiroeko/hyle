#include <gtest/gtest.h>

#include <algorithm>
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

TEST(ComputeOpsLowPrecision, Float8WeightTailKeepsCeilingScaleRows) {
  for (std::int64_t n : {1, 127, 128, 129, 576}) {
    const auto weight = float8_weight_description(6144, n,
        Float8ScaleMode::kFloat32Block128);
    ASSERT_TRUE(weight);
    EXPECT_EQ(*weight->required_bytes(),
              std::size_t(n * 6144 + ((n + 127) / 128) * 48 * 4));
    const auto activation = float8_activation_description(3, 6144,
        Float8ScaleMode::kFloat32Block128).value();
    const auto output = infer_single_output(Operation(MatMul{}),
                                            std::array{activation, *weight});
    ASSERT_TRUE(output);
    EXPECT_EQ(output->dimensions()[1], n);
  }
}

TEST(ComputeOpsLowPrecision, GgufCatalogPreservesExactNativeBlocks) {
  struct Case final {
    GgufQuantizedType type;
    StorageFormatId id;
    std::size_t bytes;
  };
  constexpr std::array cases{Case{GgufQuantizedType::kQ3K, kGgufQ3K, 110},
                             Case{GgufQuantizedType::kQ4K, kGgufQ4K, 144},
                             Case{GgufQuantizedType::kQ5K, kGgufQ5K, 176},
                             Case{GgufQuantizedType::kQ6K, kGgufQ6K, 210},
                             Case{GgufQuantizedType::kIq3Xxs, kGgufIq3Xxs, 98},
                             Case{GgufQuantizedType::kIq3S, kGgufIq3S, 110},
                             Case{GgufQuantizedType::kIq4Xs, kGgufIq4Xs, 136}};
  for (const auto &[type, id, bytes] : cases) {
    const auto weight =
        gguf_weight_description(type, DataType::kFloat32, 256, 3);
    ASSERT_TRUE(weight);
    EXPECT_EQ(weight->storage_format().id(), id);
    EXPECT_EQ(*weight->required_bytes(), bytes * 3);
    EXPECT_EQ(gguf_quantized_type(id), type);
    EXPECT_EQ(gguf_quantized_type(weight->storage_format()), type);
    EXPECT_EQ(weight->block_axis(), 0u);
  }
  const auto q8 = gguf_weight_description(GgufQuantizedType::kQ8_0,
                                          DataType::kBFloat16, 64, 3);
  const auto iq4 = gguf_weight_description(GgufQuantizedType::kIq4Nl,
                                           DataType::kFloat16, 64, 3);
  ASSERT_TRUE(q8);
  ASSERT_TRUE(iq4);
  EXPECT_EQ(*q8->required_bytes(), 204u);
  EXPECT_EQ(*iq4->required_bytes(), 108u);
  const auto malformed = StorageFormat::blocked(kGgufQ4K, 256, 143).value();
  EXPECT_FALSE(gguf_quantized_type(malformed));
}

TEST(ComputeOpsLowPrecision, GgufGroupedWeightsKeepExpertRowAndKOrder) {
  const auto weight = gguf_grouped_weight_description(
      GgufQuantizedType::kIq4Xs, DataType::kFloat32, 4, 3, 256);
  ASSERT_TRUE(weight);
  EXPECT_TRUE(weight->is_dense());
  constexpr std::array<std::int64_t, 3> dimensions{4, 3, 256};
  EXPECT_TRUE(std::ranges::equal(weight->dimensions(), dimensions));
  EXPECT_EQ(weight->block_axis(), 2u);
  EXPECT_EQ(weight->byte_strides()[2], 136);
  EXPECT_EQ(weight->byte_strides()[1], 136);
  EXPECT_EQ(weight->byte_strides()[0], 408);
  EXPECT_EQ(*weight->required_bytes(), 1632u);
}

TEST(ComputeOpsLowPrecision, RejectsIncompleteBlocksAndOverflow) {
  const auto block = Float8ScaleMode::kFloat32Block128;
  const auto group = Float8ScaleMode::kUnsignedExponent8Group32;
  EXPECT_FALSE(float8_activation_description(0, 128, block));
  EXPECT_FALSE(float8_activation_description(-1, 128, block));
  EXPECT_FALSE(float8_activation_description(1, 129, block));
  EXPECT_FALSE(float8_weight_description(128, 0, block));
  EXPECT_FALSE(float8_weight_description(33, 1, group));
  EXPECT_FALSE(
      float8_weight_description(32, 1, static_cast<Float8ScaleMode>(255)));
  EXPECT_FALSE(mxfp4_weight_description(31, 1));
  EXPECT_FALSE(mxfp4_weight_description(32, 0));
  EXPECT_FALSE(gguf_weight_description(GgufQuantizedType::kQ4K, DataType::kInt8,
                                       256, 1));
  EXPECT_FALSE(gguf_weight_description(GgufQuantizedType::kQ4K,
                                       DataType::kFloat32, 255, 1));
  EXPECT_FALSE(gguf_grouped_weight_description(GgufQuantizedType::kQ8_0,
                                               DataType::kFloat32, 0, 1, 32));
  constexpr auto maximum = (std::numeric_limits<std::int64_t>::max)();
  EXPECT_FALSE(float8_activation_description(maximum, 128, block));
  EXPECT_FALSE(float8_weight_description(maximum - 31, 128, group));
  EXPECT_FALSE(mxfp4_weight_description(maximum - 31, maximum));
}
} // namespace
