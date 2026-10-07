#include <gtest/gtest.h>

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

TEST(MatureReduction, ExecutesEveryModeAcrossTheLastAxis) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureReduction, ReducesMultipleAxesWithAndWithoutKeptDimensions) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureReduction, HandlesUnalignedStrideNaNAndSignedZero) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureReduction, DefinesEmptyReductionIdentities) {
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
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureReduction, RejectsUnsupportedStorageAndAccumulation) {
  constexpr std::array<std::int64_t, 1> shape{4};
  const TensorDesc f16 = TensorDesc::dense(DataType::kFloat16, shape).value();
  const TensorDesc f32 = TensorDesc::dense(DataType::kFloat32, shape).value();
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();

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

template <class Value>
void expect_half_reduction(bool general, bool padded, bool last_axis,
                           bool native = false) {
  constexpr std::size_t width = 129;
  const std::size_t groups = general ? 3 : 1;
  std::vector<Value> source(2 * groups * width);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] = half_test::encode<Value>(
        (index % width == 0 ? 1.0 : 1.0 / 4096) *
        static_cast<double>(1 + (index / width) % groups) *
        static_cast<double>(1 + index / (width * groups)));
  const auto type = half_test::data_type<Value>();
  const std::array<std::int64_t, 3> shape{2, static_cast<std::int64_t>(groups),
                                          width};
  const std::size_t step = padded ? 3 : sizeof(Value);
  std::vector<std::byte> physical((source.size() - 1) * step + sizeof(Value));
  for (std::size_t index = 0; index < source.size(); ++index)
    std::memcpy(physical.data() + index * step, &source[index], sizeof(Value));
  const auto input_desc =
      TensorDesc::strided_bytes(
          type, shape,
          std::array<std::int64_t, 3>{
              static_cast<std::int64_t>(groups * width * step),
              static_cast<std::int64_t>(width * step),
              static_cast<std::int64_t>(step)})
          .value();
  Buffer input = half_test::buffer<std::byte>(physical);
  const auto input_view =
      ConstTensorView::create(input_desc, input.view()).value();
  auto context = native
                     ? hyle::compute::test::native_context().value()
                     : hyle::compute::Context::create(
                           hyle::compute::cpu::device(),
                           hyle::compute::aten_cpu::provider())
                           .value();
  auto queue = context.create_queue().value();
  for (const auto kind : {ReductionKind::kSum, ReductionKind::kMean,
                          ReductionKind::kMinimum, ReductionKind::kMaximum}) {
    if (native &&
        !(kind == ReductionKind::kMaximum ||
          (type == DataType::kFloat16 && kind == ReductionKind::kMinimum)))
      continue;
    const Axes axes =
        last_axis ? Axes::from(std::array<std::int32_t, 1>{-1}).value()
        : general ? Axes::from(std::array<std::int32_t, 2>{0, 2}).value()
                  : Axes::all();
    auto prepared =
        hyle::compute::prepare(context,
                               Reduce{.kind = kind,
                                      .axes = axes,
                                      .accumulation_type = DataType::kFloat32},
                               input_desc)
            .value();
    const auto output_desc = prepared.output_description().value();
    std::vector<Value> empty(last_axis ? groups * 2 : groups);
    Buffer output = half_test::buffer<Value>(empty);
    const auto output_view =
        TensorView::create(output_desc, output.view()).value();
    auto invocation = prepared.create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, input_view, output_view)->wait().ok());
    const auto actual = half_test::read<Value>(output);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const double factor = static_cast<double>(index % groups + 1);
      const double row_factor =
          last_axis ? static_cast<double>(index / groups + 1) : 3.0;
      const double expected =
          kind == ReductionKind::kSum ? 1.03125 * row_factor * factor
          : kind == ReductionKind::kMean
              ? 1.03125 * row_factor * factor / (last_axis ? width : 2 * width)
          : kind == ReductionKind::kMinimum
              ? (last_axis ? row_factor : 1.0) * factor / 4096
              : (last_axis ? row_factor : 2.0) * factor;
      EXPECT_EQ(actual[index].bits, half_test::encode<Value>(expected).bits);
    }
  }
}

TEST(MatureReduction, HalfAccumulatesInFloat32AcrossDenseAndStridedAxes) {
  for (const bool general : {false, true})
    for (const bool padded : {false, true})
      for (const bool last : {false, true}) {
        expect_half_reduction<hyle::simd::float16_t>(general, padded, last);
        expect_half_reduction<hyle::simd::bfloat16_t>(general, padded, last);
      }
}

TEST(NativeReduction, QualifiedNarrowExtremaKeepDenseAndOddStrideSemantics) {
  for (const bool general : {false, true})
    for (const bool padded : {false, true})
      for (const bool last : {false, true}) {
        SCOPED_TRACE(general);
        SCOPED_TRACE(padded);
        SCOPED_TRACE(last);
        expect_half_reduction<hyle::simd::float16_t>(general, padded, last, true);
        expect_half_reduction<hyle::simd::bfloat16_t>(general, padded, last, true);
      }
}

template <class Value> void expect_half_reduction_edges() {
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  for (const bool empty : {false, true}) {
    const auto desc =
        TensorDesc::dense(half_test::data_type<Value>(),
                          std::array<std::int64_t, 2>{2, empty ? 0 : 2})
            .value();
    std::vector<Value> source;
    if (!empty)
      source = {
          half_test::encode<Value>(std::numeric_limits<double>::quiet_NaN()),
          half_test::encode<Value>(1.0),
          half_test::encode<Value>(-std::numeric_limits<double>::infinity()),
          half_test::encode<Value>(std::numeric_limits<double>::infinity())};
    Buffer input = half_test::buffer<Value>(source);
    const auto input_view = ConstTensorView::create(desc, input.view()).value();
    for (const auto kind : {ReductionKind::kSum, ReductionKind::kMean,
                            ReductionKind::kMinimum, ReductionKind::kMaximum}) {
      auto prepared =
          hyle::compute::prepare(
              context,
              Reduce{.kind = kind,
                     .axes =
                         Axes::from(std::array<std::int32_t, 1>{-1}).value(),
                     .accumulation_type = DataType::kFloat32},
              desc)
              .value();
      std::vector<Value> zero(2);
      Buffer output = half_test::buffer<Value>(zero);
      const auto target =
          TensorView::create(prepared.output_description().value(),
                             output.view())
              .value();
      auto invocation = prepared.create_invocation().value();
      ASSERT_TRUE(invocation.submit(queue, input_view, target)->wait().ok());
      const auto actual = half_test::read<Value>(output);
      const double identity = kind == ReductionKind::kSum ? 0.0
                              : kind == ReductionKind::kMean
                                  ? std::numeric_limits<double>::quiet_NaN()
                              : kind == ReductionKind::kMinimum
                                  ? std::numeric_limits<double>::infinity()
                                  : -std::numeric_limits<double>::infinity();
      half_test::expect_near(actual[0],
                             empty ? identity
                                   : std::numeric_limits<double>::quiet_NaN());
      const double special = kind == ReductionKind::kMinimum
                                 ? -std::numeric_limits<double>::infinity()
                             : kind == ReductionKind::kMaximum
                                 ? std::numeric_limits<double>::infinity()
                                 : std::numeric_limits<double>::quiet_NaN();
      half_test::expect_near(actual[1], empty ? identity : special);
    }
  }
}

TEST(MatureReduction, HalfPreservesEmptyAndNonFiniteRules) {
  expect_half_reduction_edges<hyle::simd::float16_t>();
  expect_half_reduction_edges<hyle::simd::bfloat16_t>();
}

} // namespace
