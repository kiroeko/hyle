#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

#include "../compute/native_context_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::Axes;
using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::Reduce;
using hyle::compute::ReductionKind;
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

TEST(NativeReduction, ExecutesEveryModeAcrossTheLastAxis) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  constexpr std::array<std::int64_t, 1> output_shape{2};
  constexpr std::array<std::int32_t, 1> axis{1};
  constexpr std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  constexpr std::array kinds{ReductionKind::kSum, ReductionKind::kMean,
                             ReductionKind::kMinimum, ReductionKind::kMaximum};
  constexpr std::array<std::array<float, 2>, 4> expected{
      std::array<float, 2>{6, 15}, std::array<float, 2>{2, 5},
      std::array<float, 2>{1, 4}, std::array<float, 2>{3, 6}};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();
  const Axes axes = Axes::from(axis).value();
  Buffer input_storage = make_buffer(values);
  std::array<float, 2> zero{};
  Buffer output_storage = make_buffer(zero);
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  for (std::size_t index = 0; index < kinds.size(); ++index) {
    auto prepared = hyle::compute::prepare(
        context, Reduce{.kind = kinds[index], .axes = axes}, input_desc);
    ASSERT_TRUE(prepared.has_value());
    ASSERT_EQ(prepared->output_description().value(), output_desc);
    auto invocation = prepared->create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
    EXPECT_EQ((read_buffer<float, 2>(output_storage)), expected[index]);
  }
}

TEST(NativeReduction, ReducesMultipleAxesWithAndWithoutKeptDimensions) {
  constexpr std::array<std::int64_t, 3> shape{2, 2, 2};
  constexpr std::array<std::int64_t, 1> compact_shape{2};
  constexpr std::array<std::int64_t, 3> kept_shape{1, 2, 1};
  constexpr std::array<std::int32_t, 2> axes_values{0, 2};
  constexpr std::array<float, 8> values{1, 2, 3, 4, 5, 6, 7, 8};
  constexpr std::array<float, 2> expected{14, 22};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc compact_desc =
      TensorDesc::dense(DataType::kFloat32, compact_shape).value();
  const TensorDesc kept_desc =
      TensorDesc::dense(DataType::kFloat32, kept_shape).value();
  const Axes axes = Axes::from(axes_values).value();
  Buffer input_storage = make_buffer(values);
  Buffer compact_storage = make_buffer(std::array<float, 2>{});
  Buffer kept_storage = make_buffer(std::array<float, 2>{});
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView compact =
      TensorView::create(compact_desc, compact_storage.view()).value();
  const TensorView kept =
      TensorView::create(kept_desc, kept_storage.view()).value();
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  auto compact_prepared =
      hyle::compute::prepare(context,
                             Reduce{.kind = ReductionKind::kSum, .axes = axes},
                             input_desc)
          .value();
  auto compact_invocation = compact_prepared.create_invocation().value();
  ASSERT_TRUE(compact_invocation.submit(queue, input, compact)->wait().ok());
  EXPECT_EQ((read_buffer<float, 2>(compact_storage)), expected);

  auto kept_prepared =
      hyle::compute::prepare(context,
                             Reduce{.kind = ReductionKind::kSum,
                                    .axes = axes,
                                    .keep_dimensions = true},
                             input_desc)
          .value();
  ASSERT_EQ(kept_prepared.output_description().value(), kept_desc);
  auto kept_invocation = kept_prepared.create_invocation().value();
  ASSERT_TRUE(kept_invocation.submit(queue, input, kept)->wait().ok());
  EXPECT_EQ((read_buffer<float, 2>(kept_storage)), expected);
}

TEST(NativeReduction, HandlesUnalignedStrideNaNAndSignedZero) {
  constexpr std::array<std::int64_t, 2> shape{2, 2};
  constexpr std::array<std::int64_t, 2> strides{10, 4};
  constexpr std::array<std::size_t, 4> offsets{0, 4, 10, 14};
  constexpr std::array<float, 4> values{
      0.0F, -0.0F, 3.0F, std::numeric_limits<float>::quiet_NaN()};
  std::array<std::byte, 18> physical{};
  for (std::size_t index = 0; index < values.size(); ++index)
    std::memcpy(physical.data() + offsets[index], &values[index],
                sizeof(float));
  const TensorDesc input_desc =
      TensorDesc::strided_bytes(DataType::kFloat32, shape, strides).value();
  constexpr std::array<std::int64_t, 0> scalar_shape{};
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, scalar_shape).value();
  Buffer input_storage = Buffer::create(hyle::mem::system_resource(),
                                        physical.size(), alignof(float))
                             .value();
  ASSERT_TRUE(
      input_storage.view().write(std::span<const std::byte>(physical)).ok());
  Buffer output_storage = make_buffer(std::array<float, 1>{});
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  auto prepared =
      hyle::compute::prepare(context, Reduce{.kind = ReductionKind::kMinimum},
                             input_desc)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
  EXPECT_TRUE(std::isnan(read_buffer<float, 1>(output_storage)[0]));

  std::array<float, 37> zeros{};
  zeros[1] = -0.0F;
  zeros[19] = -0.0F;
  constexpr std::array<std::int64_t, 1> zero_shape{37};
  const TensorDesc zeros_desc =
      TensorDesc::dense(DataType::kFloat32, zero_shape).value();
  Buffer zeros_storage = make_buffer(zeros);
  const ConstTensorView zeros_input =
      ConstTensorView::create(zeros_desc, zeros_storage.view()).value();
  prepared = hyle::compute::prepare(
                 context, Reduce{.kind = ReductionKind::kMinimum}, zeros_desc)
                 .value();
  invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, zeros_input, output)->wait().ok());
  EXPECT_TRUE(std::signbit(read_buffer<float, 1>(output_storage)[0]));
}

TEST(NativeReduction, DefinesEmptyReductionIdentities) {
  constexpr std::array<std::int64_t, 2> shape{2, 0};
  constexpr std::array<std::int64_t, 1> output_shape{2};
  constexpr std::array<std::int32_t, 1> axis{1};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();
  const Axes axes = Axes::from(axis).value();
  Buffer input_storage =
      Buffer::create(hyle::mem::system_resource(), 0).value();
  Buffer output_storage = make_buffer(std::array<float, 2>{});
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();

  constexpr std::array kinds{ReductionKind::kSum, ReductionKind::kMean,
                             ReductionKind::kMinimum, ReductionKind::kMaximum};
  for (ReductionKind kind : kinds) {
    auto prepared = hyle::compute::prepare(
                        context, Reduce{.kind = kind, .axes = axes}, input_desc)
                        .value();
    auto invocation = prepared.create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
    const auto actual = read_buffer<float, 2>(output_storage);
    for (float value : actual) {
      if (kind == ReductionKind::kSum)
        EXPECT_EQ(value, 0.0F);
      else if (kind == ReductionKind::kMean)
        EXPECT_TRUE(std::isnan(value));
      else if (kind == ReductionKind::kMinimum)
        EXPECT_EQ(value, std::numeric_limits<float>::infinity());
      else
        EXPECT_EQ(value, -std::numeric_limits<float>::infinity());
    }
  }
}

TEST(NativeReduction, RejectsUnsupportedStorageAndAccumulation) {
  constexpr std::array<std::int64_t, 1> shape{4};
  const TensorDesc f16 = TensorDesc::dense(DataType::kFloat16, shape).value();
  const TensorDesc f32 = TensorDesc::dense(DataType::kFloat32, shape).value();
  auto context = hyle::compute::test::native_context().value();

  auto storage =
      hyle::compute::prepare(context, Reduce{.kind = ReductionKind::kSum}, f16);
  ASSERT_FALSE(storage.has_value());
  EXPECT_TRUE(storage.error().IsUnimplemented());

  auto accumulation =
      hyle::compute::prepare(context,
                             Reduce{.kind = ReductionKind::kSum,
                                    .accumulation_type = DataType::kFloat64},
                             f32);
  ASSERT_FALSE(accumulation.has_value());
  EXPECT_TRUE(accumulation.error().IsUnimplemented());
}

} // namespace
