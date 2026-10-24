#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>

#include "../../src/compute_ops/view_alias.h"
#include "hyle/compute/operation.h"

namespace {
using namespace hyle::compute;

TensorDesc strided(std::initializer_list<std::int64_t> dims,
                   std::initializer_list<std::int64_t> strides) {
  return TensorDesc::strided_bytes(DataType::kBFloat16,
      std::span(dims.begin(), dims.size()),
      std::span(strides.begin(), strides.size())).value();
}

hyle::Result<TensorDesc> reshape(const TensorDesc &input,
                                std::initializer_list<std::int64_t> dims) {
  const auto shape = TensorShape::create(std::span(dims.begin(), dims.size())).value();
  return infer_single_output(Operation(View::reshape(shape).value()), std::array{input});
}

// Independent row-major logical-index oracle: the two descriptors must name
// the same byte for every element, regardless of their ranks/chunk boundaries.
std::size_t address(const TensorDesc &desc, std::size_t flat) {
  std::size_t result = 0;
  for (std::size_t end = desc.rank(); end != 0; --end) {
    const auto axis = end - 1;
    const auto extent = static_cast<std::size_t>(desc.dimensions()[axis]);
    result += (flat % extent) * static_cast<std::size_t>(desc.byte_strides()[axis]);
    flat /= extent;
  }
  return result;
}

void same_addresses(const TensorDesc &source, const TensorDesc &target) {
  ASSERT_EQ(source.element_count().value(), target.element_count().value());
  EXPECT_EQ(source.required_bytes().value(), target.required_bytes().value());
  for (std::size_t flat = 0; flat < source.element_count().value(); ++flat)
    ASSERT_EQ(address(source, flat), address(target, flat)) << "element " << flat;
}

TEST(StridedReshape, QkvSubrowsPreserveTokenPitchAndSliceOffset) {
  const auto projection = TensorDesc::dense(DataType::kBFloat16,
      std::array<std::int64_t, 3>{1, 3, 6144}).value();
  for (auto start : {0, 2048, 4096}) {
    const auto slice = View::slice(2, start, 2048).value();
    const auto subrow = infer_single_output(Operation(slice), std::array{projection}).value();
    const auto output = reshape(subrow, {1, 3, 16, 128});
    ASSERT_TRUE(output) << output.error().message();
    EXPECT_EQ(output->byte_strides()[1], 12288);
    EXPECT_EQ(output->byte_strides()[2], 256);
    EXPECT_EQ(output->byte_strides()[3], 2);
    EXPECT_FALSE(output->is_dense());
    EXPECT_EQ(detail::view_alias_offset(slice, projection, subrow).value(),
              static_cast<std::size_t>(start) * 2);
    same_addresses(subrow, *output);
    const auto view = View::reshape(TensorShape::create(output->dimensions()).value()).value();
    EXPECT_EQ(detail::view_alias_offset(view, subrow, *output).value(), 0u);
  }
}

TEST(StridedReshape, SplitsAndMergesChunksWithoutCrossingGaps) {
  const auto source = strided({2, 3, 4}, {96, 32, 4});
  const auto output = reshape(source, {6, 2, 2});
  ASSERT_TRUE(output) << output.error().message();
  same_addresses(source, *output);
  EXPECT_EQ(*output, strided({6, 2, 2}, {32, 8, 4}));
  const auto roundtrip = reshape(*output, {2, 3, 4});
  ASSERT_TRUE(roundtrip);
  same_addresses(*output, *roundtrip);
  EXPECT_EQ(*roundtrip, source);
  EXPECT_TRUE(reshape(source, {24}).error().IsUnimplemented());
  EXPECT_TRUE(reshape(strided({3, 2}, {2, 6}), {6}).error().IsUnimplemented());
  EXPECT_TRUE(reshape(strided({1, 3, 8, 256}, {24576, 8192, 1024, 2}),
                      {3, 2048}).error().IsUnimplemented());
}

TEST(StridedReshape, PreservesBroadcastsAndIgnoresSingletonStrides) {
  const auto broadcast = strided({2, 3}, {0, 0});
  const auto flattened = reshape(broadcast, {6});
  ASSERT_TRUE(flattened);
  same_addresses(broadcast, *flattened);
  EXPECT_EQ(flattened->byte_strides()[0], 0);
  EXPECT_TRUE(reshape(strided({2, 3}, {0, 2}), {6}).error().IsUnimplemented());
  const auto source = strided({2, 1, 3, 1}, {24, 999, 8, 100});
  const auto output = reshape(source, {1, 6, 1});
  ASSERT_TRUE(output) << output.error().message();
  same_addresses(source, *output);
  EXPECT_EQ(output->byte_strides()[1], 8);
}

TEST(StridedReshape, HandlesEmptyScalarAndCountMismatch) {
  const auto empty = strided({2, 0, 3}, {50, 20, 2});
  const auto output = reshape(empty, {0, 6});
  ASSERT_TRUE(output);
  EXPECT_EQ(output->element_count().value(), 0u);
  EXPECT_EQ(output->required_bytes().value(), 0u);
  const auto scalar = TensorDesc::dense(DataType::kBFloat16, {}).value();
  const auto ones = reshape(scalar, {1, 1});
  ASSERT_TRUE(ones);
  same_addresses(scalar, *ones);
  const auto back = reshape(strided({1, 1}, {999, 0}), {});
  ASSERT_TRUE(back);
  same_addresses(scalar, *back);
  EXPECT_TRUE(reshape(empty, {6}).error().IsInvalidArgument());
  EXPECT_TRUE(reshape(strided({2, 3}, {20, 2}), {5}).error().IsInvalidArgument());
}

TEST(StridedReshape, DoesNotOverflowUnusedSingletonStrides) {
  if constexpr (sizeof(std::size_t) >= sizeof(std::int64_t)) {
    const auto pitch = (std::numeric_limits<std::int64_t>::max)() / 2;
    const auto source = strided({3}, {pitch});
    const auto output = reshape(source, {1, 3});
    ASSERT_TRUE(output) << output.error().message();
    EXPECT_EQ(output->byte_strides()[1], pitch);
    same_addresses(source, *output);
  }
}
} // namespace
