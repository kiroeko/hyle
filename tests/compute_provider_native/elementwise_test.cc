#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::Binary;
using hyle::compute::BinaryKind;
using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::Select;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::mem::Buffer;

constexpr float kSqrt2Inv = 0.70710678118654752440F;
constexpr float kSqrt2OverPi = 0.79788456080286535588F;

float sigmoid(float value) {
  if (std::isnan(value))
    return value;
  const float exponential = std::exp(-std::abs(value));
  return value < 0.0F ? exponential / (1.0F + exponential)
                      : 1.0F / (1.0F + exponential);
}

float unary_reference(UnaryKind kind, float value) {
  switch (kind) {
  case UnaryKind::kAbs:
    return std::fabs(value);
  case UnaryKind::kNeg:
    return -value;
  case UnaryKind::kExp:
    return std::exp(value);
  case UnaryKind::kLog:
    return std::log(value);
  case UnaryKind::kSqrt:
    return std::sqrt(value);
  case UnaryKind::kRsqrt:
    return 1.0F / std::sqrt(value);
  case UnaryKind::kTanh:
    return std::tanh(value);
  case UnaryKind::kSigmoid:
    return sigmoid(value);
  case UnaryKind::kRelu:
    if (std::isnan(value))
      return value;
    return value > 0.0F ? value : 0.0F;
  case UnaryKind::kGelu:
    if (value == -std::numeric_limits<float>::infinity())
      return 0.0F;
    return 0.5F * value * (1.0F + std::erf(value * kSqrt2Inv));
  case UnaryKind::kGeluTanh: {
    if (value == -std::numeric_limits<float>::infinity())
      return 0.0F;
    const float inner =
        kSqrt2OverPi * value * std::fma(0.044715F, value * value, 1.0F);
    return 0.5F * value * (1.0F + std::tanh(inner));
  }
  case UnaryKind::kSilu:
    if (value == -std::numeric_limits<float>::infinity())
      return 0.0F;
    return value * sigmoid(value);
  }
  return std::numeric_limits<float>::quiet_NaN();
}

float binary_reference(BinaryKind kind, float left, float right) {
  switch (kind) {
  case BinaryKind::kAdd:
    return left + right;
  case BinaryKind::kSubtract:
    return left - right;
  case BinaryKind::kMultiply:
    return left * right;
  case BinaryKind::kDivide:
    return left / right;
  case BinaryKind::kMinimum:
    return std::fmin(left, right);
  case BinaryKind::kMaximum:
    return std::fmax(left, right);
  }
  return std::numeric_limits<float>::quiet_NaN();
}

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

void expect_float(float actual, float expected) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual));
  } else if (std::isinf(expected)) {
    EXPECT_EQ(actual, expected);
  } else {
    EXPECT_NEAR(actual, expected, std::abs(expected) * 3.0e-4F + 3.0e-6F);
  }
}

TEST(NativeElementwise, ExecutesEveryDenseUnaryModeAndExactAlias) {
  constexpr std::array<std::int64_t, 1> shape{7};
  constexpr std::array<float, 7> values{-2.0F, -0.0F, 0.25F, 0.5F,
                                        1.0F,  2.0F,  4.0F};
  constexpr std::array kinds{
      UnaryKind::kAbs,  UnaryKind::kNeg,      UnaryKind::kExp,
      UnaryKind::kLog,  UnaryKind::kSqrt,     UnaryKind::kRsqrt,
      UnaryKind::kTanh, UnaryKind::kSigmoid,  UnaryKind::kRelu,
      UnaryKind::kGelu, UnaryKind::kGeluTanh, UnaryKind::kSilu};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer input_storage = make_buffer(values);
  std::array<float, values.size()> zero{};
  Buffer output_storage = make_buffer(zero);
  const ConstTensorView input =
      ConstTensorView::create(description, input_storage.view()).value();
  const TensorView output =
      TensorView::create(description, output_storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();

  for (UnaryKind kind : kinds) {
    auto prepared = hyle::compute::prepare(context, Unary{kind}, description);
    ASSERT_TRUE(prepared.has_value());
    auto invocation = prepared->create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
    const auto actual = read_buffer<float, values.size()>(output_storage);
    for (std::size_t index = 0; index < values.size(); ++index)
      expect_float(actual[index], unary_reference(kind, values[index]));
  }

  ASSERT_TRUE(
      input_storage.view().write(std::as_bytes(std::span(values))).ok());
  const TensorView alias =
      TensorView::create(description, input_storage.view()).value();
  auto prepared =
      hyle::compute::prepare(context, Unary{UnaryKind::kNeg}, description)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, alias.as_const(), alias)->wait().ok());
  const auto actual = read_buffer<float, values.size()>(input_storage);
  for (std::size_t index = 0; index < values.size(); ++index)
    EXPECT_FLOAT_EQ(actual[index], -values[index]);
}

TEST(NativeElementwise, MaterializesUnalignedStridedUnary) {
  constexpr std::array<std::int64_t, 2> shape{2, 2};
  constexpr std::array<std::int64_t, 2> strides{10, 4};
  constexpr std::array<float, 4> values{-1.0F, 2.0F, -3.0F, 4.0F};
  constexpr std::array<std::size_t, 4> offsets{0, 4, 10, 14};
  constexpr std::array<float, 4> expected{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<std::byte, 18> physical{};
  for (std::size_t index = 0; index < values.size(); ++index)
    std::memcpy(physical.data() + offsets[index], &values[index],
                sizeof(float));

  const TensorDesc input_desc =
      TensorDesc::strided_bytes(DataType::kFloat32, shape, strides).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer input_storage = Buffer::create(hyle::mem::system_resource(),
                                        physical.size(), alignof(float))
                             .value();
  ASSERT_TRUE(
      input_storage.view().write(std::span<const std::byte>(physical)).ok());
  std::array<float, expected.size()> zero{};
  Buffer output_storage = make_buffer(zero);
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();

  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Unary{UnaryKind::kAbs}, input_desc)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
  EXPECT_EQ((read_buffer<float, expected.size()>(output_storage)), expected);
}

TEST(NativeElementwise, ExecutesEveryDenseBinaryModeAndAliasesSecondInput) {
  constexpr std::array<std::int64_t, 1> shape{6};
  constexpr std::array<float, 6> left{
      -2.0F, -0.0F, 1.0F, 4.0F, 8.0F, std::numeric_limits<float>::quiet_NaN()};
  constexpr std::array<float, 6> right{1.0F, -3.0F, 2.0F, 0.5F, -2.0F, 7.0F};
  constexpr std::array kinds{BinaryKind::kAdd,      BinaryKind::kSubtract,
                             BinaryKind::kMultiply, BinaryKind::kDivide,
                             BinaryKind::kMinimum,  BinaryKind::kMaximum};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer left_storage = make_buffer(left);
  Buffer right_storage = make_buffer(right);
  std::array<float, left.size()> zero{};
  Buffer output_storage = make_buffer(zero);
  const ConstTensorView left_view =
      ConstTensorView::create(description, left_storage.view()).value();
  const ConstTensorView right_view =
      ConstTensorView::create(description, right_storage.view()).value();
  const TensorView output =
      TensorView::create(description, output_storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();

  for (BinaryKind kind : kinds) {
    auto prepared =
        hyle::compute::prepare(context, Binary{kind}, description, description);
    ASSERT_TRUE(prepared.has_value());
    auto invocation = prepared->create_invocation().value();
    ASSERT_TRUE(
        invocation.submit(queue, left_view, right_view, output)->wait().ok());
    const auto actual = read_buffer<float, left.size()>(output_storage);
    for (std::size_t index = 0; index < left.size(); ++index)
      expect_float(actual[index],
                   binary_reference(kind, left[index], right[index]));
  }

  ASSERT_TRUE(right_storage.view().write(std::as_bytes(std::span(right))).ok());
  const TensorView alias =
      TensorView::create(description, right_storage.view()).value();
  auto prepared = hyle::compute::prepare(context, Binary{BinaryKind::kAdd},
                                         description, description)
                      .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, left_view, alias.as_const(), alias)
                  ->wait()
                  .ok());
  const auto actual = read_buffer<float, left.size()>(right_storage);
  for (std::size_t index = 0; index < left.size(); ++index)
    expect_float(actual[index], left[index] + right[index]);
}

TEST(NativeElementwise, BroadcastsBinaryOperands) {
  constexpr std::array<std::int64_t, 2> left_shape{2, 3};
  constexpr std::array<std::int64_t, 1> right_shape{3};
  constexpr std::array<float, 6> left{1, 2, 3, 4, 5, 6};
  constexpr std::array<float, 3> right{10, 20, 30};
  constexpr std::array<float, 6> expected{10, 40, 90, 40, 100, 180};
  const TensorDesc left_desc =
      TensorDesc::dense(DataType::kFloat32, left_shape).value();
  const TensorDesc right_desc =
      TensorDesc::dense(DataType::kFloat32, right_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, left_shape).value();
  Buffer left_storage = make_buffer(left);
  Buffer right_storage = make_buffer(right);
  std::array<float, expected.size()> zero{};
  Buffer output_storage = make_buffer(zero);
  const ConstTensorView left_view =
      ConstTensorView::create(left_desc, left_storage.view()).value();
  const ConstTensorView right_view =
      ConstTensorView::create(right_desc, right_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();

  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, Binary{BinaryKind::kMultiply},
                                         left_desc, right_desc);
  ASSERT_TRUE(prepared.has_value());
  ASSERT_EQ(prepared->output_description().value(), output_desc);
  auto invocation = prepared->create_invocation().value();
  ASSERT_TRUE(
      invocation.submit(queue, left_view, right_view, output)->wait().ok());
  EXPECT_EQ((read_buffer<float, expected.size()>(output_storage)), expected);
}

TEST(NativeElementwise, BroadcastsSelectWithoutChangingBranchBits) {
  constexpr std::array<std::int64_t, 2> condition_shape{2, 1};
  constexpr std::array<std::int64_t, 2> true_shape{2, 3};
  constexpr std::array<std::int64_t, 1> false_shape{3};
  constexpr std::array<std::uint8_t, 2> condition{1, 0};
  constexpr std::array<std::uint32_t, 6> when_true{10, 11, 12, 20, 21, 22};
  constexpr std::array<std::uint32_t, 3> when_false{30, 31, 32};
  constexpr std::array<std::uint32_t, 6> expected{10, 11, 12, 30, 31, 32};
  const TensorDesc condition_desc =
      TensorDesc::dense(DataType::kBool, condition_shape).value();
  const TensorDesc true_desc =
      TensorDesc::dense(DataType::kUInt32, true_shape).value();
  const TensorDesc false_desc =
      TensorDesc::dense(DataType::kUInt32, false_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kUInt32, true_shape).value();
  Buffer condition_storage = make_buffer(condition);
  Buffer true_storage = make_buffer(when_true);
  Buffer false_storage = make_buffer(when_false);
  std::array<std::uint32_t, expected.size()> zero{};
  Buffer output_storage = make_buffer(zero);
  const ConstTensorView condition_view =
      ConstTensorView::create(condition_desc, condition_storage.view()).value();
  const ConstTensorView true_view =
      ConstTensorView::create(true_desc, true_storage.view()).value();
  const ConstTensorView false_view =
      ConstTensorView::create(false_desc, false_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();

  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, Select{}, condition_desc,
                                         true_desc, false_desc);
  ASSERT_TRUE(prepared.has_value());
  ASSERT_EQ(prepared->output_description().value(), output_desc);
  auto invocation = prepared->create_invocation().value();
  ASSERT_TRUE(
      invocation.submit(queue, condition_view, true_view, false_view, output)
          ->wait()
          .ok());
  EXPECT_EQ((read_buffer<std::uint32_t, expected.size()>(output_storage)),
            expected);
}

TEST(NativeElementwise, SupportsDenseSelectAliasingABranch) {
  constexpr std::array<std::int64_t, 1> shape{4};
  constexpr std::array<std::uint8_t, 4> condition{1, 0, 1, 0};
  constexpr std::array<std::uint32_t, 4> when_true{10, 11, 12, 13};
  constexpr std::array<std::uint32_t, 4> when_false{20, 21, 22, 23};
  constexpr std::array<std::uint32_t, 4> expected{10, 21, 12, 23};
  const TensorDesc condition_desc =
      TensorDesc::dense(DataType::kBool, shape).value();
  const TensorDesc branch_desc =
      TensorDesc::dense(DataType::kUInt32, shape).value();
  Buffer condition_storage = make_buffer(condition);
  Buffer true_storage = make_buffer(when_true);
  Buffer false_storage = make_buffer(when_false);
  const ConstTensorView condition_view =
      ConstTensorView::create(condition_desc, condition_storage.view()).value();
  const TensorView true_alias =
      TensorView::create(branch_desc, true_storage.view()).value();
  const ConstTensorView false_view =
      ConstTensorView::create(branch_desc, false_storage.view()).value();

  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, Select{}, condition_desc,
                                         branch_desc, branch_desc)
                      .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation
                  .submit(queue, condition_view, true_alias.as_const(),
                          false_view, true_alias)
                  ->wait()
                  .ok());
  EXPECT_EQ((read_buffer<std::uint32_t, expected.size()>(true_storage)),
            expected);
}

TEST(NativeElementwise, ExecutesDenseF32SelectAcrossVectorBlocksAndTail) {
  constexpr std::size_t kCount = 19;
  constexpr std::array<std::int64_t, 1> shape{
      static_cast<std::int64_t>(kCount)};
  constexpr std::array<std::uint8_t, kCount> condition{
      0, 1, 2, 0xff, 0, 4, 0, 7, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0};
  std::array<float, kCount> when_true{};
  std::array<float, kCount> when_false{};
  std::array<float, kCount> expected{};
  for (std::size_t i = 0; i < kCount; ++i) {
    when_true[i] = 10.0F + static_cast<float>(i);
    when_false[i] = -10.0F - static_cast<float>(i);
    expected[i] = condition[i] != 0 ? when_true[i] : when_false[i];
  }

  const TensorDesc condition_desc =
      TensorDesc::dense(DataType::kBool, shape).value();
  const TensorDesc branch_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer condition_storage = make_buffer(condition);
  Buffer true_storage = make_buffer(when_true);
  Buffer false_storage = make_buffer(when_false);
  Buffer output_storage = make_buffer(std::array<float, kCount>{});
  const ConstTensorView condition_view =
      ConstTensorView::create(condition_desc, condition_storage.view()).value();
  const ConstTensorView true_view =
      ConstTensorView::create(branch_desc, true_storage.view()).value();
  const ConstTensorView false_view =
      ConstTensorView::create(branch_desc, false_storage.view()).value();
  const TensorView output =
      TensorView::create(branch_desc, output_storage.view()).value();

  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, Select{}, condition_desc,
                                         branch_desc, branch_desc)
                      .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(
      invocation.submit(queue, condition_view, true_view, false_view, output)
          ->wait()
          .ok());
  EXPECT_EQ((read_buffer<float, kCount>(output_storage)), expected);
}

TEST(NativeElementwise, RejectsUnsafeStridedAliasAndUnsupportedArithmeticType) {
  constexpr std::array<std::int64_t, 2> shape{2, 2};
  constexpr std::array<std::int64_t, 2> transposed_strides{4, 8};
  const TensorDesc transposed =
      TensorDesc::strided_bytes(DataType::kFloat32, shape, transposed_strides)
          .value();
  const TensorDesc dense = TensorDesc::dense(DataType::kFloat32, shape).value();
  constexpr std::array<float, 4> values{1, -2, 3, -4};
  Buffer storage = make_buffer(values);
  const ConstTensorView input =
      ConstTensorView::create(transposed, storage.view()).value();
  const TensorView output = TensorView::create(dense, storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Unary{UnaryKind::kAbs}, transposed)
          .value();
  auto invocation = prepared.create_invocation().value();
  auto completion = invocation.submit(queue, input, output);
  ASSERT_FALSE(completion.has_value());
  EXPECT_TRUE(completion.error().IsInvalidArgument());

  const TensorDesc f16 = TensorDesc::dense(DataType::kFloat16, shape).value();
  auto unsupported_unary =
      hyle::compute::prepare(context, Unary{UnaryKind::kSilu}, f16);
  ASSERT_FALSE(unsupported_unary.has_value());
  EXPECT_TRUE(unsupported_unary.error().IsUnimplemented());

  const TensorDesc i32 = TensorDesc::dense(DataType::kInt32, shape).value();
  auto unsupported_binary =
      hyle::compute::prepare(context, Binary{BinaryKind::kAdd}, i32, i32);
  ASSERT_FALSE(unsupported_binary.has_value());
  EXPECT_TRUE(unsupported_binary.error().IsUnimplemented());
}

TEST(NativeElementwise, ExecutesZeroElementFamiliesWithoutTouchingStorage) {
  constexpr std::array<std::int64_t, 2> shape{0, 4};
  const TensorDesc values =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc condition =
      TensorDesc::dense(DataType::kBool, shape).value();
  Buffer empty = Buffer::create(hyle::mem::system_resource(), 0).value();
  const ConstTensorView value_input =
      ConstTensorView::create(values, empty.view()).value();
  const ConstTensorView condition_input =
      ConstTensorView::create(condition, empty.view()).value();
  const TensorView output = TensorView::create(values, empty.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();

  auto unary =
      hyle::compute::prepare(context, Unary{UnaryKind::kExp}, values).value();
  auto unary_invocation = unary.create_invocation().value();
  ASSERT_TRUE(unary_invocation.submit(queue, value_input, output)->wait().ok());

  auto binary =
      hyle::compute::prepare(context, Binary{BinaryKind::kAdd}, values, values)
          .value();
  auto binary_invocation = binary.create_invocation().value();
  ASSERT_TRUE(binary_invocation.submit(queue, value_input, value_input, output)
                  ->wait()
                  .ok());

  auto select =
      hyle::compute::prepare(context, Select{}, condition, values, values)
          .value();
  auto select_invocation = select.create_invocation().value();
  ASSERT_TRUE(
      select_invocation
          .submit(queue, condition_input, value_input, value_input, output)
          ->wait()
          .ok());
}

} // namespace
