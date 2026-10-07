#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "../compute/native_context_fixture.h"
#include "half_test_support.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

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

TEST(MatureTransformer, ExecutesBothRopeLayoutsAndPreservesTheTail) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  const auto i64 = TensorDesc::dense(DataType::kInt64, position_shape).value();
  auto wide_positions = hyle::compute::prepare(
      context, RoPE{.rotary_dimensions = 4, .sequence_axis = 1}, data_desc, i64,
      frequency_desc);
  ASSERT_TRUE(wide_positions) << wide_positions.error().message();
  auto queue = context.create_queue().value();

  auto interleaved =
      hyle::compute::prepare(context,
                             RoPE{.rotary_dimensions = 4,
                                  .sequence_axis = 1,
                                  .layout = RopeLayout::kInterleaved},
                             data_desc, position_desc, frequency_desc)
          .value();
  auto invocation = interleaved.create_invocation().value();
  ASSERT_TRUE(
      invocation.submit(queue, inputs[0], inputs[1], inputs[2], outputs[0])
          ->wait()
          .ok());
  expect_near(read_buffer<float, values.size()>(output_storage),
              interleaved_expected);

  const TensorView alias_output =
      TensorView::create(data_desc, data_storage.view()).value();
  const std::array alias_outputs{alias_output};
  auto split = hyle::compute::prepare(context,
                                      RoPE{.rotary_dimensions = 4,
                                           .sequence_axis = 1,
                                           .layout = RopeLayout::kHalfSplit},
                                      data_desc, position_desc, frequency_desc)
                   .value();
  EXPECT_TRUE(split.alias_support(0, 0)->supported);
  invocation = split.create_invocation().value();
  ASSERT_TRUE(
      invocation
          .submit(queue, inputs[0], inputs[1], inputs[2], alias_outputs[0])
          ->wait()
          .ok());
  expect_near(read_buffer<float, values.size()>(data_storage), split_expected);
}

TEST(MatureTransformer, RopeExercisesVectorBlocksAndPairTail) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureTransformer, RopeMaterializesPlainStridedInputs) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureTransformer, SoftmaxSupportsBroadcastMasksScaleAndExactAlias) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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
  ASSERT_TRUE(
      invocation.submit(queue, inputs[0], inputs[1], outputs[0])->wait().ok());

  const auto first =
      reference_softmax({0.5F, -std::numeric_limits<float>::infinity(), 1.5F});
  const auto second =
      reference_softmax({2.0F, -std::numeric_limits<float>::infinity(), 3.0F});
  const std::array<float, 6> expected{first[0],  first[1],  first[2],
                                      second[0], second[1], second[2]};
  expect_near(read_buffer<float, values.size()>(data_storage), expected);
}

TEST(MatureTransformer, LogSoftmaxSupportsANonTrailingAxis) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureTransformer, SoftmaxSupportsAdditiveMasksAndUndefinedRows) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(NativeTransformer, SdpaIsRetiredBeforeExecution) {
  const auto desc = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 4>{1, 1, 1, 64}).value();
  auto context = hyle::compute::test::native_context().value();
  const auto result = hyle::compute::prepare(context,
      ScaledDotProductAttention{}, desc, desc, desc);
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsUnimplemented());
}

template <class Value>
void expect_half_softmax(bool strided, MaskKind mask_kind) {
  constexpr std::size_t width = 129;
  const std::size_t inner = strided ? 3 : 1;
  const std::size_t count = 2 * width * inner;
  const auto type = half_test::data_type<Value>();
  const std::size_t step = strided ? 3 : sizeof(Value);
  const std::array<std::int64_t, 3> shape{2, width,
                                          static_cast<std::int64_t>(inner)};
  const auto input_desc =
      TensorDesc::strided_bytes(
          type, shape,
          std::array<std::int64_t, 3>{
              static_cast<std::int64_t>(width * inner * step),
              static_cast<std::int64_t>(inner * step),
              static_cast<std::int64_t>(step)})
          .value();
  std::vector<Value> source(count);
  std::vector<std::byte> physical((count - 1) * step + sizeof(Value));
  for (std::size_t index = 0; index < count; ++index) {
    source[index] = half_test::encode<Value>(
        static_cast<double>(static_cast<int>(index % 67) - 33) / 8);
    std::memcpy(physical.data() + index * step, &source[index], sizeof(Value));
  }
  std::vector<Value> additive(width);
  std::vector<std::uint8_t> boolean(width);
  for (std::size_t lane = 0; lane < width; ++lane) {
    boolean[lane] = lane % 7 != 0 ? 1 : 0;
    additive[lane] = half_test::encode<Value>(
        boolean[lane] != 0 ? -static_cast<double>(lane % 3) / 8
                           : -std::numeric_limits<double>::infinity());
  }
  const auto mask_desc =
      TensorDesc::dense(mask_kind == MaskKind::kBoolean ? DataType::kBool
                                                        : type,
                        std::array<std::int64_t, 3>{1, width, 1})
          .value();
  Buffer input = half_test::buffer<std::byte>(physical);
  Buffer mask = mask_kind == MaskKind::kBoolean
                    ? half_test::buffer<std::uint8_t>(boolean)
                    : half_test::buffer<Value>(additive);
  Buffer output = half_test::buffer<Value>(source);
  const auto input_view =
      ConstTensorView::create(input_desc, input.view()).value();
  const auto mask_view =
      ConstTensorView::create(mask_desc, mask.view()).value();
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  for (const auto kind : {SoftmaxKind::kSoftmax}) {
    const float scale = mask_kind == MaskKind::kNone ? 1.0F : 0.75F;
    const Softmax operation{
        .axis = 1, .scale = scale, .kind = kind, .mask = mask_kind};
    auto prepared =
        mask_kind == MaskKind::kNone
            ? hyle::compute::prepare(context, operation, input_desc)
            : hyle::compute::prepare(context, operation, input_desc, mask_desc);
    ASSERT_TRUE(prepared.has_value());
    const auto output_view =
        TensorView::create(prepared->output_description().value(),
                           output.view())
            .value();
    const auto requirement = prepared->workspace_requirement();
    EXPECT_EQ(requirement.size, width * sizeof(float));
    EXPECT_FALSE(prepared->create_invocation().has_value());
    Buffer scratch = Buffer::create(hyle::mem::system_resource(),
                                    requirement.size, requirement.alignment)
                         .value();
    auto invocation = prepared->create_invocation(scratch.view()).value();
    const auto run = [&](const TensorView &target) {
      return mask_kind == MaskKind::kNone
                 ? invocation.submit(queue, input_view, target)
                 : invocation.submit(queue, input_view, mask_view, target);
    };
    ASSERT_TRUE(run(output_view)->wait().ok());
    const auto actual = half_test::read<Value>(output);
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t column = 0; column < inner; ++column) {
        std::array<double, width> logits{};
        for (std::size_t lane = 0; lane < width; ++lane) {
          logits[lane] =
              half_test::decode(source[(row * width + lane) * inner + column]) *
              scale;
          if (mask_kind == MaskKind::kBoolean && boolean[lane] == 0)
            logits[lane] = -std::numeric_limits<double>::infinity();
          if (mask_kind == MaskKind::kAdditive)
            logits[lane] += half_test::decode(additive[lane]);
        }
        const double maximum = *std::max_element(logits.begin(), logits.end());
        double sum = 0;
        for (const double value : logits)
          sum += std::exp(value - maximum);
        for (std::size_t lane = 0; lane < width; ++lane) {
          const double expected = kind == SoftmaxKind::kSoftmax
                                      ? std::exp(logits[lane] - maximum) / sum
                                      : logits[lane] - maximum - std::log(sum);
          half_test::expect_near(actual[(row * width + lane) * inner + column],
                                 expected);
        }
      }
    }
    if (!strided) {
      const auto alias = TensorView::create(input_desc, input.view()).value();
      ASSERT_TRUE(run(alias)->wait().ok());
      const auto alias_values = half_test::read<Value>(input);
      for (std::size_t index = 0; index < count; ++index)
        EXPECT_EQ(alias_values[index].bits, actual[index].bits);
      ASSERT_TRUE(input.view().write(physical).ok());
    }
    if (mask_kind != MaskKind::kNone) {
      const std::vector<std::uint8_t> no_boolean(width, 0);
      const std::vector<Value> no_additive(
          width,
          half_test::encode<Value>(-std::numeric_limits<double>::infinity()));
      ASSERT_TRUE(mask.view()
                      .write(mask_kind == MaskKind::kBoolean
                                 ? std::as_bytes(std::span(no_boolean))
                                 : std::as_bytes(std::span(no_additive)))
                      .ok());
      Buffer second_scratch =
          Buffer::create(hyle::mem::system_resource(), requirement.size,
                         requirement.alignment)
              .value();
      auto second_invocation =
          prepared->create_invocation(second_scratch.view()).value();
      Buffer second_output = half_test::buffer<Value>(source);
      const auto target =
          TensorView::create(prepared->output_description().value(),
                             second_output.view())
              .value();
      ASSERT_TRUE(
          second_invocation.submit(queue, input_view, mask_view, target)
              ->wait()
              .ok());
      for (const auto value : half_test::read<Value>(second_output))
        EXPECT_TRUE(std::isnan(half_test::decode(value)));
      const auto retained = half_test::read<Value>(output);
      for (std::size_t index = 0; index < count; ++index)
        EXPECT_EQ(retained[index].bits, actual[index].bits);
      ASSERT_TRUE(mask.view()
                      .write(mask_kind == MaskKind::kBoolean
                                 ? std::as_bytes(std::span(boolean))
                                 : std::as_bytes(std::span(additive)))
                      .ok());
    }
  }
}

TEST(NativeTransformer, HalfSoftmaxUsesDeclaredWorkspaceAndMaskSemantics) {
  for (const bool strided : {false, true})
    expect_half_softmax<hyle::simd::float16_t>(strided, MaskKind::kBoolean);
}

template <class Value> void expect_half_empty_transformer() {
  const auto desc = TensorDesc::dense(half_test::data_type<Value>(),
                                      std::array<std::int64_t, 2>{0, 129})
                        .value();
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  Buffer empty = Buffer::create(hyle::mem::system_resource(), 0).value();
  const auto input = ConstTensorView::create(desc, empty.view()).value();
  const auto boolean = TensorDesc::dense(DataType::kBool, std::array<std::int64_t, 2>{0, 129}).value();
  auto softmax = hyle::compute::prepare(context, Softmax{.mask = MaskKind::kBoolean}, desc, boolean).value();
  EXPECT_EQ(softmax.workspace_requirement().size, 0U);
  auto invocation = softmax.create_invocation().value();
  const auto output = TensorView::create(desc, empty.view()).value();
  const auto mask = ConstTensorView::create(boolean, empty.view()).value();
  ASSERT_TRUE(invocation.submit(queue, input, mask, output)->wait().ok());
}

TEST(NativeTransformer, EmptyHalfWorkloadRequiresNoWorkspace) {
  expect_half_empty_transformer<hyle::simd::float16_t>();
}

} // namespace
