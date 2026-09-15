#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "../compute/native_context_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::AttentionLayout;
using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::MaskKind;
using hyle::compute::Operation;
using hyle::compute::ParallelTaskFn;
using hyle::compute::RoPE;
using hyle::compute::RopeLayout;
using hyle::compute::ScaledDotProductAttention;
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

void expect_near(std::span<const float> actual, std::span<const float> expected,
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
  constexpr std::array<float, 12> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  constexpr std::array<std::int32_t, 2> positions{0, 1};
  constexpr std::array<float, 4> frequencies{half_pi, pi, half_pi, pi};
  constexpr std::array<float, 12> interleaved_expected{1,  2, 3,  4,   5,  6,
                                                       -8, 7, -9, -10, 11, 12};
  constexpr std::array<float, 12> split_expected{1,  2,  3, 4,   5,  6,
                                                 -9, -8, 7, -10, 11, 12};

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
      ConstTensorView::create(frequency_desc, frequency_storage.view())
          .value()};
  const std::array outputs{
      TensorView::create(data_desc, output_storage.view()).value()};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  auto interleaved =
      hyle::compute::prepare(
          context,
          RoPE{.rotary_dimensions = 4,
               .sequence_axis = 1,
               .layout = RopeLayout::kInterleaved},
          data_desc, position_desc, frequency_desc)
          .value();
  auto invocation = interleaved.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs[0], inputs[1], inputs[2],
                                outputs[0])
                  ->wait()
                  .ok());
  expect_near(read_buffer<float, values.size()>(output_storage),
              interleaved_expected);

  const TensorView alias_output =
      TensorView::create(data_desc, data_storage.view()).value();
  const std::array alias_outputs{alias_output};
  auto split = hyle::compute::prepare(
                   context,
                   RoPE{.rotary_dimensions = 4,
                        .sequence_axis = 1,
                        .layout = RopeLayout::kHalfSplit},
                   data_desc, position_desc, frequency_desc)
                   .value();
  EXPECT_TRUE(split.alias_support(0, 0)->supported);
  invocation = split.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs[0], inputs[1], inputs[2],
                                alias_outputs[0])
                  ->wait()
                  .ok());
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
      ConstTensorView::create(frequency_desc, frequency_storage.view())
          .value()};
  const std::array outputs{
      TensorView::create(data_desc, output_storage.view()).value()};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  for (RopeLayout layout : {RopeLayout::kInterleaved, RopeLayout::kHalfSplit}) {
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
        hyle::compute::prepare(context,
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
  constexpr std::array<float, 16> padded_values{1, 2, 3, 4,  5,  6,  101, 102,
                                                7, 8, 9, 10, 11, 12, 103, 104};
  constexpr std::array<std::int32_t, 2> positions{0, 1};
  constexpr std::array<float, 6> padded_frequencies{half_pi, pi, 201,
                                                    half_pi, pi, 202};
  constexpr std::array<float, 12> expected{1,  2, 3,  4,   5,  6,
                                           -8, 7, -9, -10, 11, 12};

  const TensorDesc data_desc =
      TensorDesc::strided_bytes(DataType::kFloat32, data_shape, data_strides)
          .value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, data_shape).value();
  const TensorDesc position_desc =
      TensorDesc::dense(DataType::kInt32, position_shape).value();
  const TensorDesc frequency_desc =
      TensorDesc::strided_bytes(DataType::kFloat32, frequency_shape,
                                frequency_strides)
          .value();
  Buffer data_storage = make_buffer(padded_values);
  Buffer position_storage = make_buffer(positions);
  Buffer frequency_storage = make_buffer(padded_frequencies);
  Buffer output_storage = make_buffer(std::array<float, expected.size()>{});
  const std::array inputs{
      ConstTensorView::create(data_desc, data_storage.view()).value(),
      ConstTensorView::create(position_desc, position_storage.view()).value(),
      ConstTensorView::create(frequency_desc, frequency_storage.view())
          .value()};
  const std::array outputs{
      TensorView::create(output_desc, output_storage.view()).value()};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(
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
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(
          context,
          Softmax{.axis = -1, .scale = 0.5F, .mask = MaskKind::kBoolean},
          data_desc, mask_desc)
          .value();
  EXPECT_TRUE(prepared.alias_support(0, 0)->supported);
  auto invocation_result = prepared.create_invocation();
  ASSERT_TRUE(invocation_result.has_value())
      << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.submit(queue, inputs[0], inputs[1], outputs[0])
                  ->wait()
                  .ok());

  const auto first =
      reference_softmax({0.5F, -std::numeric_limits<float>::infinity(), 1.5F});
  const auto second =
      reference_softmax({2.0F, -std::numeric_limits<float>::infinity(), 3.0F});
  const std::array<float, 6> expected{first[0],  first[1],  first[2],
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
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(
          context, Softmax{.axis = 1, .kind = SoftmaxKind::kLogSoftmax},
          data_desc)
          .value();
  auto invocation_result = prepared.create_invocation();
  ASSERT_TRUE(invocation_result.has_value())
      << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.submit(queue, inputs[0], outputs[0])->wait().ok());

  const float shift0 =
      3.0F + std::log(std::exp(-2.0F) + std::exp(-1.0F) + 1.0F);
  const float shift1 =
      30.0F + std::log(std::exp(-20.0F) + std::exp(-10.0F) + 1.0F);
  const std::array<float, 6> expected{1 - shift0,  10 - shift1, 2 - shift0,
                                      20 - shift1, 3 - shift0,  30 - shift1};
  expect_near(read_buffer<float, values.size()>(output_storage), expected);
}

TEST(NativeTransformer, SoftmaxSupportsAdditiveMasksAndUndefinedRows) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  constexpr std::array<std::int64_t, 2> mask_shape{1, 3};
  constexpr std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  const float negative_infinity = -std::numeric_limits<float>::infinity();
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
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(
          context, Operation(Softmax{.axis = -1, .mask = MaskKind::kAdditive}),
          inputs, outputs)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());

  const auto first = reference_softmax({1, negative_infinity, 2});
  const auto second = reference_softmax({4, negative_infinity, 5});
  const std::array<float, 6> expected{first[0],  first[1],  first[2],
                                      second[0], second[1], second[2]};
  expect_near(read_buffer<float, values.size()>(output_storage), expected);

  const std::array<float, 3> all_masked{negative_infinity, negative_infinity,
                                        negative_infinity};
  mask_storage = make_buffer(all_masked);
  const std::array invalid_inputs{
      ConstTensorView::create(data_desc, data_storage.view()).value(),
      ConstTensorView::create(mask_desc, mask_storage.view()).value()};
  prepared =
      hyle::compute::prepare(
          context, Operation(Softmax{.axis = -1, .mask = MaskKind::kAdditive}),
          invalid_inputs, outputs)
          .value();
  invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, invalid_inputs, outputs)->wait().ok());
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
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared_result =
      hyle::compute::prepare(context,
                             TopK{.count = 2,
                                  .axis = 1,
                                  .largest = true,
                                  .sorted = true,
                                  .stable = true},
                             data_desc);
  ASSERT_TRUE(prepared_result.has_value()) << prepared_result.error().message();
  auto prepared = std::move(*prepared_result);
  const auto requirement = prepared.workspace_requirement();
  Buffer workspace_storage =
      Buffer::create(hyle::mem::system_resource(), requirement.size,
                     requirement.alignment)
          .value();
  auto invocation_result = prepared.create_invocation(workspace_storage.view());
  ASSERT_TRUE(invocation_result.has_value())
      << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.submit(queue, inputs[0], outputs[0], outputs[1])
                  ->wait()
                  .ok());
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
  prepared_result = hyle::compute::prepare(context,
                                           Operation(TopK{.count = 2,
                                                          .axis = 1,
                                                          .largest = false,
                                                          .sorted = true,
                                                          .stable = true}),
                                           middle_inputs, middle_outputs);
  ASSERT_TRUE(prepared_result.has_value()) << prepared_result.error().message();
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

TEST(NativeTransformer, SdpaExecutesBlockedOnlineSoftmaxInBothLayouts) {
  constexpr std::array<std::int64_t, 4> query_shape{1, 2, 1, 2};
  constexpr std::array<std::int64_t, 4> key_shape{1, 3, 1, 2};
  constexpr std::array<std::int64_t, 4> value_shape{1, 3, 1, 3};
  constexpr std::array<std::int64_t, 4> output_shape{1, 2, 1, 3};
  constexpr std::array<float, 4> query_values{1, 0, 0, 1};
  constexpr std::array<float, 6> key_values{1, 0, 0, 1, 1, 1};
  constexpr std::array<float, 9> value_values{10, 0, 1, 0, 20, 2, 30, 30, 3};
  const TensorDesc query_desc =
      TensorDesc::dense(DataType::kFloat32, query_shape).value();
  const TensorDesc key_desc =
      TensorDesc::dense(DataType::kFloat32, key_shape).value();
  const TensorDesc value_desc =
      TensorDesc::dense(DataType::kFloat32, value_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();
  Buffer query = make_buffer(query_values);
  Buffer key = make_buffer(key_values);
  Buffer value = make_buffer(value_values);
  Buffer output = make_buffer(std::array<float, 6>{});
  const std::array inputs{
      ConstTensorView::create(query_desc, query.view()).value(),
      ConstTensorView::create(key_desc, key.view()).value(),
      ConstTensorView::create(value_desc, value.view()).value()};
  const std::array outputs{
      TensorView::create(output_desc, output.view()).value()};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(
                      context, ScaledDotProductAttention{}, query_desc,
                      key_desc, value_desc)
                      .value();
  const auto requirement = prepared.workspace_requirement();
  EXPECT_GT(requirement.size, 0U);
  Buffer workspace = Buffer::create(hyle::mem::system_resource(),
                                    requirement.size, requirement.alignment)
                         .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  ASSERT_TRUE(invocation.submit(queue, inputs[0], inputs[1], inputs[2],
                                outputs[0])
                  ->wait()
                  .ok());

  const float e = std::exp(1.0F);
  const float first_sum = 2.0F * e + 1.0F;
  const float second_sum = 1.0F + 2.0F * e;
  const std::array<float, 6> expected{
      40.0F * e / first_sum,         (20.0F + 30.0F * e) / first_sum,
      (4.0F * e + 2.0F) / first_sum, (10.0F + 30.0F * e) / second_sum,
      50.0F * e / second_sum,        (1.0F + 5.0F * e) / second_sum};
  expect_near(read_buffer<float, 6>(output), expected, 8.0e-5F);

  constexpr std::array<std::int64_t, 4> head_query_shape{1, 1, 2, 2};
  constexpr std::array<std::int64_t, 4> head_key_shape{1, 1, 3, 2};
  constexpr std::array<std::int64_t, 4> head_value_shape{1, 1, 3, 3};
  constexpr std::array<std::int64_t, 4> head_output_shape{1, 1, 2, 3};
  const TensorDesc head_query_desc =
      TensorDesc::dense(DataType::kFloat32, head_query_shape).value();
  const TensorDesc head_key_desc =
      TensorDesc::dense(DataType::kFloat32, head_key_shape).value();
  const TensorDesc head_value_desc =
      TensorDesc::dense(DataType::kFloat32, head_value_shape).value();
  const TensorDesc head_output_desc =
      TensorDesc::dense(DataType::kFloat32, head_output_shape).value();
  const std::array head_inputs{
      ConstTensorView::create(head_query_desc, query.view()).value(),
      ConstTensorView::create(head_key_desc, key.view()).value(),
      ConstTensorView::create(head_value_desc, value.view()).value()};
  const std::array head_outputs{
      TensorView::create(head_output_desc, output.view()).value()};
  prepared = hyle::compute::prepare(
                 context,
                 Operation(ScaledDotProductAttention{
                     .layout = AttentionLayout::kBatchHeadSequenceDepth}),
                 head_inputs, head_outputs)
                 .value();
  workspace = Buffer::create(hyle::mem::system_resource(),
                             prepared.workspace_requirement().size,
                             prepared.workspace_requirement().alignment)
                  .value();
  invocation = prepared.create_invocation(workspace.view()).value();
  ASSERT_TRUE(invocation.submit(queue, head_inputs, head_outputs)->wait().ok());
  expect_near(read_buffer<float, 6>(output), expected, 8.0e-5F);
}

TEST(NativeTransformer, SdpaKeepsBatchAndHeadRowsSeparateInBothLayouts) {
  constexpr std::array<float, 8> query_values{};
  constexpr std::array<float, 12> key_values{};
  constexpr std::array<float, 12> bshd_values{1, 10, 2, 20, 3, 30,
                                              4, 40, 5, 50, 6, 60};
  constexpr std::array<float, 8> bshd_expected{2, 20, 2, 20, 5, 50, 5, 50};
  constexpr std::array<float, 12> bhsd_values{1, 2, 3, 10, 20, 30,
                                              4, 5, 6, 40, 50, 60};
  constexpr std::array<float, 8> bhsd_expected{2, 2, 20, 20, 5, 5, 50, 50};
  Buffer query = make_buffer(query_values);
  Buffer key = make_buffer(key_values);
  Buffer output = make_buffer(std::array<float, 8>{});
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  auto run = [&](AttentionLayout layout,
                 std::span<const std::int64_t> query_shape,
                 std::span<const std::int64_t> key_shape,
                 const std::array<float, 12> &values,
                 const std::array<float, 8> &expected) {
    const TensorDesc query_desc =
        TensorDesc::dense(DataType::kFloat32, query_shape).value();
    const TensorDesc key_desc =
        TensorDesc::dense(DataType::kFloat32, key_shape).value();
    Buffer value = make_buffer(values);
    const std::array inputs{
        ConstTensorView::create(query_desc, query.view()).value(),
        ConstTensorView::create(key_desc, key.view()).value(),
        ConstTensorView::create(key_desc, value.view()).value()};
    const std::array outputs{
        TensorView::create(query_desc, output.view()).value()};
    auto prepared =
        hyle::compute::prepare(
            context, Operation(ScaledDotProductAttention{.layout = layout}),
            inputs, outputs)
            .value();
    const auto requirement = prepared.workspace_requirement();
    Buffer workspace = Buffer::create(hyle::mem::system_resource(),
                                      requirement.size, requirement.alignment)
                           .value();
    auto invocation = prepared.create_invocation(workspace.view()).value();
    ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
    expect_near(read_buffer<float, 8>(output), expected);
  };

  constexpr std::array<std::int64_t, 4> bshd_query_shape{2, 2, 2, 1};
  constexpr std::array<std::int64_t, 4> bshd_key_shape{2, 3, 2, 1};
  run(AttentionLayout::kBatchSequenceHeadDepth, bshd_query_shape,
      bshd_key_shape, bshd_values, bshd_expected);

  constexpr std::array<std::int64_t, 4> bhsd_query_shape{2, 2, 2, 1};
  constexpr std::array<std::int64_t, 4> bhsd_key_shape{2, 2, 3, 1};
  run(AttentionLayout::kBatchHeadSequenceDepth, bhsd_query_shape,
      bhsd_key_shape, bhsd_values, bhsd_expected);
}

TEST(NativeTransformer, SdpaUsesBottomRightCausalAndExplicitMasks) {
  constexpr std::array<std::int64_t, 4> query_shape{1, 1, 1, 1};
  constexpr std::array<std::int64_t, 4> key_shape{1, 1, 3, 1};
  constexpr std::array<std::int64_t, 4> mask_shape{1, 1, 1, 3};
  constexpr std::array<float, 1> query_values{0};
  constexpr std::array<float, 3> key_values{0, 0, 0};
  constexpr std::array<float, 3> value_values{1, 2, 6};
  const TensorDesc query_desc =
      TensorDesc::dense(DataType::kFloat32, query_shape).value();
  const TensorDesc key_desc =
      TensorDesc::dense(DataType::kFloat32, key_shape).value();
  const TensorDesc bool_mask_desc =
      TensorDesc::dense(DataType::kBool, mask_shape).value();
  const TensorDesc float_mask_desc =
      TensorDesc::dense(DataType::kFloat32, mask_shape).value();
  Buffer query = make_buffer(query_values);
  Buffer key = make_buffer(key_values);
  Buffer value = make_buffer(value_values);
  Buffer output = make_buffer(std::array<float, 1>{});
  const std::array base_inputs{
      ConstTensorView::create(query_desc, query.view()).value(),
      ConstTensorView::create(key_desc, key.view()).value(),
      ConstTensorView::create(key_desc, value.view()).value()};
  const std::array outputs{
      TensorView::create(query_desc, output.view()).value()};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  auto run = [&](Operation operation, auto inputs) {
    auto prepared =
        hyle::compute::prepare(context, operation, inputs, outputs).value();
    const auto requirement = prepared.workspace_requirement();
    Buffer workspace = Buffer::create(hyle::mem::system_resource(),
                                      requirement.size, requirement.alignment)
                           .value();
    auto invocation = prepared.create_invocation(workspace.view()).value();
    EXPECT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  };

  run(Operation(ScaledDotProductAttention{
          .layout = AttentionLayout::kBatchHeadSequenceDepth, .causal = true}),
      base_inputs);
  EXPECT_NEAR((read_buffer<float, 1>(output)[0]), 3.0F, 2.0e-5F);

  Buffer bool_mask = make_buffer(std::array<std::uint8_t, 3>{1, 0, 1});
  const std::array bool_inputs{
      base_inputs[0], base_inputs[1], base_inputs[2],
      ConstTensorView::create(bool_mask_desc, bool_mask.view()).value()};
  const ScaledDotProductAttention bool_operation{
      .layout = AttentionLayout::kBatchHeadSequenceDepth,
      .mask = MaskKind::kBoolean};
  auto bool_prepared =
      hyle::compute::prepare(context, bool_operation, query_desc, key_desc,
                             key_desc, bool_mask_desc)
          .value();
  auto bool_workspace =
      Buffer::create(hyle::mem::system_resource(),
                     bool_prepared.workspace_requirement().size,
                     bool_prepared.workspace_requirement().alignment)
          .value();
  auto bool_invocation =
      bool_prepared.create_invocation(bool_workspace.view()).value();
  ASSERT_TRUE(bool_invocation
                  .submit(queue, bool_inputs[0], bool_inputs[1], bool_inputs[2],
                          bool_inputs[3], outputs[0])
                  ->wait()
                  .ok());
  EXPECT_NEAR((read_buffer<float, 1>(output)[0]), 3.5F, 2.0e-5F);

  const float negative_infinity = -std::numeric_limits<float>::infinity();
  Buffer additive_mask =
      make_buffer(std::array<float, 3>{0, negative_infinity, 0});
  const std::array additive_inputs{
      base_inputs[0], base_inputs[1], base_inputs[2],
      ConstTensorView::create(float_mask_desc, additive_mask.view()).value()};
  run(Operation(ScaledDotProductAttention{
          .layout = AttentionLayout::kBatchHeadSequenceDepth,
          .mask = MaskKind::kAdditive}),
      additive_inputs);
  EXPECT_NEAR((read_buffer<float, 1>(output)[0]), 3.5F, 2.0e-5F);

  bool_mask = make_buffer(std::array<std::uint8_t, 3>{0, 0, 0});
  const std::array invalid_inputs{
      base_inputs[0], base_inputs[1], base_inputs[2],
      ConstTensorView::create(bool_mask_desc, bool_mask.view()).value()};
  run(Operation(ScaledDotProductAttention{
          .layout = AttentionLayout::kBatchHeadSequenceDepth,
          .mask = MaskKind::kBoolean}),
      invalid_inputs);
  EXPECT_TRUE(std::isnan(read_buffer<float, 1>(output)[0]));
}

TEST(NativeTransformer, SdpaSupportsPlainStridedReferenceInputs) {
  constexpr std::array<std::int64_t, 4> query_shape{1, 1, 1, 2};
  constexpr std::array<std::int64_t, 4> key_shape{1, 2, 1, 2};
  constexpr std::array<std::int64_t, 4> query_strides{16, 16, 16, 4};
  constexpr std::array<std::int64_t, 4> key_strides{32, 16, 16, 4};
  constexpr std::array<float, 4> query_values{1, 0, 91, 92};
  constexpr std::array<float, 8> key_values{1, 0, 81, 82, 0, 1, 83, 84};
  constexpr std::array<float, 8> value_values{1, 2, 71, 72, 3, 4, 73, 74};
  const TensorDesc query_desc =
      TensorDesc::strided_bytes(DataType::kFloat32, query_shape, query_strides)
          .value();
  const TensorDesc key_desc =
      TensorDesc::strided_bytes(DataType::kFloat32, key_shape, key_strides)
          .value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, query_shape).value();
  Buffer query = make_buffer(query_values);
  Buffer key = make_buffer(key_values);
  Buffer value = make_buffer(value_values);
  Buffer output = make_buffer(std::array<float, 2>{});
  const std::array inputs{
      ConstTensorView::create(query_desc, query.view()).value(),
      ConstTensorView::create(key_desc, key.view()).value(),
      ConstTensorView::create(key_desc, value.view()).value()};
  const std::array outputs{
      TensorView::create(output_desc, output.view()).value()};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Operation(ScaledDotProductAttention{}),
                             inputs, outputs)
          .value();
  const auto requirement = prepared.workspace_requirement();
  Buffer workspace = Buffer::create(hyle::mem::system_resource(),
                                    requirement.size, requirement.alignment)
                         .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  const float e = std::exp(1.0F);
  const std::array<float, 2> expected{(e + 3.0F) / (e + 1.0F),
                                      (2.0F * e + 4.0F) / (e + 1.0F)};
  expect_near(read_buffer<float, 2>(output), expected, 8.0e-5F);
}

struct AttentionParallelProbe final {
  std::size_t episodes = 0;
  std::size_t tasks = 0;
};

void attention_parallel_for(void *context, std::size_t count,
                            ParallelTaskFn task, void *task_context) noexcept {
  auto &probe = *static_cast<AttentionParallelProbe *>(context);
  ++probe.episodes;
  probe.tasks += count;
  for (std::size_t index = 0; index < count; ++index)
    task(task_context, index, count);
}

TEST(NativeTransformer, SdpaUsesDisjointWorkspacePerExecutorTask) {
  constexpr std::array<std::int64_t, 4> query_shape{1, 8, 4, 64};
  constexpr std::array<std::int64_t, 4> key_shape{1, 64, 4, 64};
  const TensorDesc query_desc =
      TensorDesc::dense(DataType::kFloat32, query_shape).value();
  const TensorDesc key_desc =
      TensorDesc::dense(DataType::kFloat32, key_shape).value();
  Buffer query = Buffer::create(hyle::mem::system_resource(),
                                query_desc.required_bytes().value(), 64)
                     .value();
  Buffer key = Buffer::create(hyle::mem::system_resource(),
                              key_desc.required_bytes().value(), 64)
                   .value();
  Buffer value = Buffer::create(hyle::mem::system_resource(),
                                key_desc.required_bytes().value(), 64)
                     .value();
  Buffer output = Buffer::create(hyle::mem::system_resource(),
                                 query_desc.required_bytes().value(), 64)
                      .value();
  ASSERT_TRUE(query.view().fill(std::byte{0}).ok());
  ASSERT_TRUE(key.view().fill(std::byte{0}).ok());
  ASSERT_TRUE(value.view().fill(std::byte{0}).ok());
  ASSERT_TRUE(output.view().fill(std::byte{0}).ok());
  const std::array inputs{
      ConstTensorView::create(query_desc, query.view()).value(),
      ConstTensorView::create(key_desc, key.view()).value(),
      ConstTensorView::create(key_desc, value.view()).value()};
  const std::array outputs{
      TensorView::create(query_desc, output.view()).value()};

  AttentionParallelProbe probe;
  const hyle::compute::ParallelExecutorRef executor(&probe, 4,
                                                    &attention_parallel_for);
  auto context =
      hyle::compute::test::native_context({.executor = executor}).value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Operation(ScaledDotProductAttention{}),
                             inputs, outputs)
          .value();
  const auto requirement = prepared.workspace_requirement();
  Buffer workspace = Buffer::create(hyle::mem::system_resource(),
                                    requirement.size, requirement.alignment)
                         .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  EXPECT_EQ(probe.episodes, 1U);
  EXPECT_EQ(probe.tasks, 4U);
}

TEST(NativeTransformer, SdpaDefinesAnEmptyKeyRowAsInvalid) {
  constexpr std::array<std::int64_t, 4> query_shape{1, 1, 1, 1};
  constexpr std::array<std::int64_t, 4> key_shape{1, 0, 1, 1};
  constexpr std::array<std::int64_t, 4> value_shape{1, 0, 1, 2};
  constexpr std::array<std::int64_t, 4> output_shape{1, 1, 1, 2};
  const TensorDesc query_desc =
      TensorDesc::dense(DataType::kFloat32, query_shape).value();
  const TensorDesc key_desc =
      TensorDesc::dense(DataType::kFloat32, key_shape).value();
  const TensorDesc value_desc =
      TensorDesc::dense(DataType::kFloat32, value_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();
  Buffer query = make_buffer(std::array<float, 1>{1});
  Buffer key = Buffer::create(hyle::mem::system_resource(), 0).value();
  Buffer value = Buffer::create(hyle::mem::system_resource(), 0).value();
  Buffer output = make_buffer(std::array<float, 2>{4, 5});
  const std::array inputs{
      ConstTensorView::create(query_desc, query.view()).value(),
      ConstTensorView::create(key_desc, key.view()).value(),
      ConstTensorView::create(value_desc, value.view()).value()};
  const std::array outputs{
      TensorView::create(output_desc, output.view()).value()};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Operation(ScaledDotProductAttention{}),
                             inputs, outputs)
          .value();
  const auto requirement = prepared.workspace_requirement();
  Buffer workspace = Buffer::create(hyle::mem::system_resource(),
                                    requirement.size, requirement.alignment)
                         .value();
  auto invocation = prepared.create_invocation(workspace.view()).value();
  ASSERT_TRUE(invocation.submit(queue, inputs, outputs)->wait().ok());
  for (float result : read_buffer<float, 2>(output))
    EXPECT_TRUE(std::isnan(result));
}

TEST(NativeTransformer, SdpaRejectsUnsupportedStorageAndAccumulation) {
  constexpr std::array<std::int64_t, 4> shape{1, 1, 1, 2};
  const TensorDesc f16 = TensorDesc::dense(DataType::kFloat16, shape).value();
  const TensorDesc f32 = TensorDesc::dense(DataType::kFloat32, shape).value();
  auto context = hyle::compute::test::native_context().value();

  const std::array f16_inputs{f16, f16, f16};
  const std::array f16_outputs{f16};
  auto storage = hyle::compute::prepare(
      context, Operation(ScaledDotProductAttention{}), f16_inputs, f16_outputs,
      hyle::mem::system_resource());
  ASSERT_FALSE(storage.has_value());
  EXPECT_TRUE(storage.error().IsUnimplemented());

  const std::array f32_inputs{f32, f32, f32};
  const std::array f32_outputs{f32};
  auto accumulation = hyle::compute::prepare(
      context,
      Operation(
          ScaledDotProductAttention{.accumulation_type = DataType::kFloat64}),
      f32_inputs, f32_outputs, hyle::mem::system_resource());
  ASSERT_FALSE(accumulation.has_value());
  EXPECT_TRUE(accumulation.error().IsUnimplemented());
}

} // namespace
