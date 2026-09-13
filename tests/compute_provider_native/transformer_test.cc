#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::MaskKind;
using hyle::compute::Operation;
using hyle::compute::RoPE;
using hyle::compute::RopeLayout;
using hyle::compute::Softmax;
using hyle::compute::SoftmaxKind;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::TopK;
using hyle::mem::Buffer;

template <class Value, std::size_t Size>
Buffer make_buffer(const std::array<Value, Size> &values) {
  Buffer result = Buffer::create(hyle::mem::system_resource(), sizeof(values),
                                 alignof(Value))
                      .value();
  EXPECT_TRUE(result.view().write(std::as_bytes(std::span(values))).ok());
  return result;
}

template <class Value, std::size_t Size>
std::array<Value, Size> read_buffer(const Buffer &buffer) {
  std::array<Value, Size> result{};
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

void expect_near(std::span<const float> actual,
                 std::span<const float> expected,
                 float tolerance = 3.0e-5F) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_NEAR(actual[index], expected[index], tolerance) << index;
}

std::array<float, 3> reference_softmax(std::array<float, 3> values) {
  const float maximum = (std::max)({values[0], values[1], values[2]});
  float sum = 0.0F;
  for (float &value : values) {
    value = std::exp(value - maximum);
    sum += value;
  }
  for (float &value : values)
    value /= sum;
  return values;
}

TEST(NativeTransformer, ExecutesBothRopeLayoutsAndPreservesTheTail) {
  constexpr float half_pi = 1.57079632679489661923F;
  constexpr float pi = 3.14159265358979323846F;
  constexpr std::array<std::int64_t, 3> data_shape{1, 2, 6};
  constexpr std::array<std::int64_t, 2> position_shape{1, 2};
  constexpr std::array<std::int64_t, 2> frequency_shape{2, 2};
  constexpr std::array<float, 12> values{1, 2, 3, 4, 5, 6,
                                         7, 8, 9, 10, 11, 12};
  constexpr std::array<std::int32_t, 2> positions{0, 1};
  constexpr std::array<float, 4> frequencies{half_pi, pi, half_pi, pi};
  constexpr std::array<float, 12> interleaved_expected{
      1, 2, 3, 4, 5, 6, -8, 7, -9, -10, 11, 12};
  constexpr std::array<float, 12> split_expected{
      1, 2, 3, 4, 5, 6, -9, -8, 7, -10, 11, 12};

  const TensorDesc data_desc =
      TensorDesc::dense(DataType::kFloat32, data_shape).value();
  const TensorDesc position_desc =
      TensorDesc::dense(DataType::kInt32, position_shape).value();
  const TensorDesc frequency_desc =
      TensorDesc::dense(DataType::kFloat32, frequency_shape).value();
  Buffer data_storage = make_buffer(values);
  Buffer position_storage = make_buffer(positions);
  Buffer frequency_storage = make_buffer(frequencies);
  Buffer output_storage = make_buffer(std::array<float, values.size()>{});
  const std::array inputs{
      ConstTensorView::create(data_desc, data_storage.view()).value(),
      ConstTensorView::create(position_desc, position_storage.view()).value(),
      ConstTensorView::create(frequency_desc, frequency_storage.view()).value()};
  const std::array outputs{
      TensorView::create(data_desc, output_storage.view()).value()};
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();

  auto interleaved =
      hyle::compute::prepare(
          context,
          Operation(RoPE{.rotary_dimensions = 4,
                         .sequence_axis = 1,
                         .layout = RopeLayout::kInterleaved}),
          inputs, outputs)
          .value();
  auto invocation = interleaved.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  expect_near(read_buffer<float, values.size()>(output_storage),
              interleaved_expected);

  const TensorView alias_output =
      TensorView::create(data_desc, data_storage.view()).value();
  const std::array alias_outputs{alias_output};
  auto split = hyle::compute::prepare(
                   context,
                   Operation(RoPE{.rotary_dimensions = 4,
                                  .sequence_axis = 1,
                                  .layout = RopeLayout::kHalfSplit}),
                   inputs, alias_outputs)
                   .value();
  EXPECT_TRUE(split.alias_support(0, 0)->supported);
  invocation = split.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, alias_outputs)->wait().ok());
  expect_near(read_buffer<float, values.size()>(data_storage), split_expected);
}

TEST(NativeTransformer, RopeExercisesVectorBlocksAndPairTail) {
  constexpr std::size_t pairs = 65;
  constexpr std::size_t rotary = pairs * 2;
  constexpr std::size_t depth = rotary + 3;
  constexpr std::array<std::int64_t, 3> data_shape{
      1, 1, static_cast<std::int64_t>(depth)};
  constexpr std::array<std::int64_t, 1> position_shape{1};
  constexpr std::array<std::int64_t, 1> frequency_shape{
      static_cast<std::int64_t>(pairs)};
  std::array<float, depth> values{};
  std::array<float, pairs> frequencies{};
  for (std::size_t index = 0; index < depth; ++index)
    values[index] = static_cast<float>(index + 1) * 0.03125F;
  for (std::size_t index = 0; index < pairs; ++index)
    frequencies[index] = (static_cast<float>(index) - 32.0F) * 0.003F;
  constexpr std::array<std::int32_t, 1> positions{3};

  const TensorDesc data_desc =
      TensorDesc::dense(DataType::kFloat32, data_shape).value();
  const TensorDesc position_desc =
      TensorDesc::dense(DataType::kInt32, position_shape).value();
  const TensorDesc frequency_desc =
      TensorDesc::dense(DataType::kFloat32, frequency_shape).value();
  Buffer data_storage = make_buffer(values);
  Buffer position_storage = make_buffer(positions);
  Buffer frequency_storage = make_buffer(frequencies);
  Buffer output_storage = make_buffer(std::array<float, depth>{});
  const std::array inputs{
      ConstTensorView::create(data_desc, data_storage.view()).value(),
      ConstTensorView::create(position_desc, position_storage.view()).value(),
      ConstTensorView::create(frequency_desc, frequency_storage.view()).value()};
  const std::array outputs{
      TensorView::create(data_desc, output_storage.view()).value()};
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();

  for (RopeLayout layout : {RopeLayout::kInterleaved,
                            RopeLayout::kHalfSplit}) {
    std::array<float, depth> expected = values;
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const float angle = 3.0F * frequencies[pair];
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      const std::size_t first =
          layout == RopeLayout::kInterleaved ? pair * 2 : pair;
      const std::size_t second =
          layout == RopeLayout::kInterleaved ? first + 1 : pairs + pair;
      expected[first] = values[first] * cosine - values[second] * sine;
      expected[second] = values[first] * sine + values[second] * cosine;
    }
    auto prepared =
        hyle::compute::prepare(
            context,
            Operation(RoPE{.rotary_dimensions = rotary,
                           .sequence_axis = 1,
                           .layout = layout}),
            inputs, outputs)
            .value();
    auto invocation = prepared.create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
    expect_near(read_buffer<float, depth>(output_storage), expected, 5.0e-5F);
  }
}

TEST(NativeTransformer, RopeMaterializesPlainStridedInputs) {
  constexpr float half_pi = 1.57079632679489661923F;
  constexpr float pi = 3.14159265358979323846F;
  constexpr std::array<std::int64_t, 3> data_shape{1, 2, 6};
  constexpr std::array<std::int64_t, 3> data_strides{64, 32, 4};
  constexpr std::array<std::int64_t, 2> position_shape{1, 2};
  constexpr std::array<std::int64_t, 2> frequency_shape{2, 2};
  constexpr std::array<std::int64_t, 2> frequency_strides{12, 4};
  constexpr std::array<float, 16> padded_values{
      1, 2, 3, 4, 5, 6, 101, 102, 7, 8, 9, 10, 11, 12, 103, 104};
  constexpr std::array<std::int32_t, 2> positions{0, 1};
  constexpr std::array<float, 6> padded_frequencies{
      half_pi, pi, 201, half_pi, pi, 202};
  constexpr std::array<float, 12> expected{
      1, 2, 3, 4, 5, 6, -8, 7, -9, -10, 11, 12};

  const TensorDesc data_desc =
      TensorDesc::strided_bytes(DataType::kFloat32, data_shape, data_strides)
          .value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, data_shape).value();
  const TensorDesc position_desc =
      TensorDesc::dense(DataType::kInt32, position_shape).value();
  const TensorDesc frequency_desc = TensorDesc::strided_bytes(
                                        DataType::kFloat32, frequency_shape,
                                        frequency_strides)
                                        .value();
  Buffer data_storage = make_buffer(padded_values);
  Buffer position_storage = make_buffer(positions);
  Buffer frequency_storage = make_buffer(padded_frequencies);
  Buffer output_storage = make_buffer(std::array<float, expected.size()>{});
  const std::array inputs{
      ConstTensorView::create(data_desc, data_storage.view()).value(),
      ConstTensorView::create(position_desc, position_storage.view()).value(),
      ConstTensorView::create(frequency_desc, frequency_storage.view()).value()};
  const std::array outputs{
      TensorView::create(output_desc, output_storage.view()).value()};
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(
          context,
          Operation(RoPE{.rotary_dimensions = 4,
                         .sequence_axis = 1,
                         .layout = RopeLayout::kInterleaved}),
          inputs, outputs)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  expect_near(read_buffer<float, expected.size()>(output_storage), expected);
}

TEST(NativeTransformer, SoftmaxSupportsBroadcastMasksScaleAndExactAlias) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  constexpr std::array<std::int64_t, 2> mask_shape{1, 3};
  constexpr std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  constexpr std::array<std::uint8_t, 3> mask{1, 0, 1};
  const TensorDesc data_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc mask_desc =
      TensorDesc::dense(DataType::kBool, mask_shape).value();
  Buffer data_storage = make_buffer(values);
  Buffer mask_storage = make_buffer(mask);
  const ConstTensorView data =
      ConstTensorView::create(data_desc, data_storage.view()).value();
  const ConstTensorView mask_view =
      ConstTensorView::create(mask_desc, mask_storage.view()).value();
  const TensorView output =
      TensorView::create(data_desc, data_storage.view()).value();
  const std::array inputs{data, mask_view};
  const std::array outputs{output};
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(
          context,
          Operation(Softmax{.axis = -1,
                            .scale = 0.5F,
                            .mask = MaskKind::kBoolean}),
          inputs, outputs)
          .value();
  EXPECT_TRUE(prepared.alias_support(0, 0)->supported);
  auto invocation_result = prepared.create_invocation();
  ASSERT_TRUE(invocation_result.has_value())
      << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());

  const auto first = reference_softmax(
      {0.5F, -std::numeric_limits<float>::infinity(), 1.5F});
  const auto second = reference_softmax(
      {2.0F, -std::numeric_limits<float>::infinity(), 3.0F});
  const std::array<float, 6> expected{first[0], first[1], first[2],
                                      second[0], second[1], second[2]};
  expect_near(read_buffer<float, values.size()>(data_storage), expected);
}

TEST(NativeTransformer, LogSoftmaxSupportsANonTrailingAxis) {
  constexpr std::array<std::int64_t, 3> shape{1, 3, 2};
  constexpr std::array<float, 6> values{1, 10, 2, 20, 3, 30};
  std::array<float, values.size()> zero{};
  const TensorDesc data_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer input_storage = make_buffer(values);
  Buffer output_storage = make_buffer(zero);
  const std::array inputs{
      ConstTensorView::create(data_desc, input_storage.view()).value()};
  const std::array outputs{
      TensorView::create(data_desc, output_storage.view()).value()};
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(
                      context,
                      Operation(Softmax{.axis = 1,
                                        .kind = SoftmaxKind::kLogSoftmax}),
                      inputs, outputs)
                      .value();
  auto invocation_result = prepared.create_invocation();
  ASSERT_TRUE(invocation_result.has_value())
      << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());

  const float shift0 = 3.0F + std::log(std::exp(-2.0F) + std::exp(-1.0F) + 1.0F);
  const float shift1 =
      30.0F + std::log(std::exp(-20.0F) + std::exp(-10.0F) + 1.0F);
  const std::array<float, 6> expected{1 - shift0, 10 - shift1,
                                      2 - shift0, 20 - shift1,
                                      3 - shift0, 30 - shift1};
  expect_near(read_buffer<float, values.size()>(output_storage), expected);
}

TEST(NativeTransformer, SoftmaxSupportsAdditiveMasksAndUndefinedRows) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  constexpr std::array<std::int64_t, 2> mask_shape{1, 3};
  constexpr std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  const float negative_infinity =
      -std::numeric_limits<float>::infinity();
  const std::array<float, 3> mask{0, negative_infinity, -1};
  const TensorDesc data_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc mask_desc =
      TensorDesc::dense(DataType::kFloat32, mask_shape).value();
  Buffer data_storage = make_buffer(values);
  Buffer mask_storage = make_buffer(mask);
  Buffer output_storage = make_buffer(std::array<float, values.size()>{});
  const std::array inputs{
      ConstTensorView::create(data_desc, data_storage.view()).value(),
      ConstTensorView::create(mask_desc, mask_storage.view()).value()};
  const std::array outputs{
      TensorView::create(data_desc, output_storage.view()).value()};
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(
                      context,
                      Operation(Softmax{.axis = -1,
                                        .mask = MaskKind::kAdditive}),
                      inputs, outputs)
                      .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());

  const auto first = reference_softmax({1, negative_infinity, 2});
  const auto second = reference_softmax({4, negative_infinity, 5});
  const std::array<float, 6> expected{first[0], first[1], first[2],
                                      second[0], second[1], second[2]};
  expect_near(read_buffer<float, values.size()>(output_storage), expected);

  const std::array<float, 3> all_masked{
      negative_infinity, negative_infinity, negative_infinity};
  mask_storage = make_buffer(all_masked);
  const std::array invalid_inputs{
      ConstTensorView::create(data_desc, data_storage.view()).value(),
      ConstTensorView::create(mask_desc, mask_storage.view()).value()};
  prepared = hyle::compute::prepare(
                 context,
                 Operation(Softmax{.axis = -1,
                                   .mask = MaskKind::kAdditive}),
                 invalid_inputs, outputs)
                 .value();
  invocation = prepared.create_invocation().value();
  ASSERT_TRUE(
      invocation.submit(queue, invalid_inputs, outputs)->wait().ok());
  for (float value : read_buffer<float, values.size()>(output_storage))
    EXPECT_TRUE(std::isnan(value));
}

TEST(NativeTransformer, TopKPreservesScoresAndStableIndicesAcrossAxes) {
  constexpr std::array<std::int64_t, 2> shape{2, 5};
  constexpr std::array<std::int64_t, 2> output_shape{2, 2};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 10> values{1, 5, 5, nan, 3, 4, -1, 8, 7, 6};
  std::array<float, 4> zero_values{};
  std::array<std::int32_t, 4> zero_indices{};
  const TensorDesc data_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc value_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();
  const TensorDesc index_desc =
      TensorDesc::dense(DataType::kInt32, output_shape).value();
  Buffer input_storage = make_buffer(values);
  Buffer value_storage = make_buffer(zero_values);
  Buffer index_storage = make_buffer(zero_indices);
  const std::array inputs{
      ConstTensorView::create(data_desc, input_storage.view()).value()};
  const std::array outputs{
      TensorView::create(value_desc, value_storage.view()).value(),
      TensorView::create(index_desc, index_storage.view()).value()};
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared_result = hyle::compute::prepare(
      context,
      Operation(TopK{.count = 2,
                     .axis = 1,
                     .largest = true,
                     .sorted = true,
                     .stable = true}),
      inputs, outputs);
  ASSERT_TRUE(prepared_result.has_value())
      << prepared_result.error().message();
  auto prepared = std::move(*prepared_result);
  const auto requirement = prepared.workspace_requirement();
  Buffer workspace_storage =
      Buffer::create(hyle::mem::system_resource(), requirement.size,
                     requirement.alignment)
          .value();
  auto invocation_result =
      prepared.create_invocation(workspace_storage.view());
  ASSERT_TRUE(invocation_result.has_value())
      << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  EXPECT_EQ((read_buffer<float, 4>(value_storage)),
            (std::array<float, 4>{5, 5, 8, 7}));
  EXPECT_EQ((read_buffer<std::int32_t, 4>(index_storage)),
            (std::array<std::int32_t, 4>{1, 2, 2, 3}));

  constexpr std::array<std::int64_t, 3> middle_shape{1, 3, 2};
  constexpr std::array<std::int64_t, 3> middle_output_shape{1, 2, 2};
  constexpr std::array<float, 6> middle_values{3, 9, 1, 8, 2, 7};
  const TensorDesc middle_desc =
      TensorDesc::dense(DataType::kFloat32, middle_shape).value();
  const TensorDesc middle_value_desc =
      TensorDesc::dense(DataType::kFloat32, middle_output_shape).value();
  const TensorDesc middle_index_desc =
      TensorDesc::dense(DataType::kInt32, middle_output_shape).value();
  input_storage = make_buffer(middle_values);
  value_storage = make_buffer(zero_values);
  index_storage = make_buffer(zero_indices);
  const std::array middle_inputs{
      ConstTensorView::create(middle_desc, input_storage.view()).value()};
  const std::array middle_outputs{
      TensorView::create(middle_value_desc, value_storage.view()).value(),
      TensorView::create(middle_index_desc, index_storage.view()).value()};
  prepared_result = hyle::compute::prepare(
      context,
      Operation(TopK{.count = 2,
                     .axis = 1,
                     .largest = false,
                     .sorted = true,
                     .stable = true}),
      middle_inputs, middle_outputs);
  ASSERT_TRUE(prepared_result.has_value())
      << prepared_result.error().message();
  prepared = std::move(*prepared_result);
  const auto middle_requirement = prepared.workspace_requirement();
  workspace_storage =
      Buffer::create(hyle::mem::system_resource(), middle_requirement.size,
                     middle_requirement.alignment)
          .value();
  invocation_result = prepared.create_invocation(workspace_storage.view());
  ASSERT_TRUE(invocation_result.has_value())
      << invocation_result.error().message();
  invocation = std::move(*invocation_result);
  ASSERT_TRUE(
      invocation.submit(queue, middle_inputs, middle_outputs)->wait().ok());
  EXPECT_EQ((read_buffer<float, 4>(value_storage)),
            (std::array<float, 4>{1, 7, 2, 8}));
  EXPECT_EQ((read_buffer<std::int32_t, 4>(index_storage)),
            (std::array<std::int32_t, 4>{1, 2, 2, 1}));
}

} // namespace
