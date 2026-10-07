#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
using hyle::compute::LayerNorm;
using hyle::compute::ParallelTaskFn;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::Buffer;

template <std::size_t Size>
Buffer make_buffer(const std::array<float, Size> &values) {
  Buffer result = Buffer::create(hyle::mem::system_resource(), sizeof(values),
                                 alignof(float))
                      .value();
  EXPECT_TRUE(result.view().write(std::as_bytes(std::span(values))).ok());
  return result;
}

template <std::size_t Size>
std::array<float, Size> read_buffer(const Buffer &buffer) {
  std::array<float, Size> result{};
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

void expect_near(std::span<const float> actual,
                 std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    EXPECT_NEAR(actual[index], expected[index], 2.0e-5F) << index;
}

struct ParallelProbe final {
  std::size_t episodes = 0;
  std::size_t tasks = 0;
  bool nested = false;
};

void probe_parallel_for(void *context, std::size_t count, ParallelTaskFn task,
                        void *task_context) noexcept {
  auto &probe = *static_cast<ParallelProbe *>(context);
  ++probe.episodes;
  probe.tasks += count;
  for (std::size_t index = 0; index < count; ++index)
    task(task_context, index, count);
}

bool probe_in_parallel(void *context) noexcept {
  return static_cast<ParallelProbe *>(context)->nested;
}

TEST(MatureLayerNorm, ExecutesNonAffineScaleOnlyAndAffineRows) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  constexpr std::array<std::int64_t, 1> parameter_shape{3};
  constexpr std::array<float, 6> values{1, 2, 3, 2, 4, 6};
  constexpr std::array<float, 3> scale{2, 3, 4};
  constexpr std::array<float, 3> bias{10, 20, 30};
  constexpr float epsilon = 1.0e-5F;
  const float first_root = 1.0F / std::sqrt(2.0F / 3.0F + epsilon);
  const float second_root = 2.0F / std::sqrt(8.0F / 3.0F + epsilon);
  const std::array<float, 6> normalized{-first_root,  0, first_root,
                                        -second_root, 0, second_root};
  const std::array<float, 6> scaled{-2 * first_root,  0, 4 * first_root,
                                    -2 * second_root, 0, 4 * second_root};
  const std::array<float, 6> affine{
      10 - 2 * first_root,  20, 30 + 4 * first_root,
      10 - 2 * second_root, 20, 30 + 4 * second_root};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc parameter_desc =
      TensorDesc::dense(DataType::kFloat32, parameter_shape).value();
  Buffer input_storage = make_buffer(values);
  Buffer scale_storage = make_buffer(scale);
  Buffer bias_storage = make_buffer(bias);
  Buffer output_storage = make_buffer(std::array<float, 6>{});
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const ConstTensorView scale_view =
      ConstTensorView::create(parameter_desc, scale_storage.view()).value();
  const ConstTensorView bias_view =
      ConstTensorView::create(parameter_desc, bias_storage.view()).value();
  const TensorView output =
      TensorView::create(input_desc, output_storage.view()).value();
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const LayerNorm operation{.epsilon = epsilon};

  auto prepared =
      hyle::compute::prepare(context, operation, input_desc).value();
  ASSERT_EQ(prepared.output_description().value(), input_desc);
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
  expect_near(read_buffer<6>(output_storage), normalized);

  prepared =
      hyle::compute::prepare(context, operation, input_desc, parameter_desc)
          .value();
  invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, scale_view, output)->wait().ok());
  expect_near(read_buffer<6>(output_storage), scaled);

  prepared = hyle::compute::prepare(context, operation, input_desc,
                                    parameter_desc, parameter_desc)
                 .value();
  invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, scale_view, bias_view, output)
                  ->wait()
                  .ok());
  expect_near(read_buffer<6>(output_storage), affine);
}

TEST(MatureLayerNorm, SupportsExactInputOutputAlias) {
  constexpr std::array<std::int64_t, 2> shape{1, 3};
  constexpr std::array<float, 3> values{1, 2, 3};
  constexpr float root = 1.2247357F;
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer storage = make_buffer(values);
  const ConstTensorView input =
      ConstTensorView::create(description, storage.view()).value();
  const TensorView output =
      TensorView::create(description, storage.view()).value();
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, LayerNorm{}, description).value();
  auto invocation = prepared.create_invocation().value();

  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());
  expect_near(read_buffer<3>(storage), std::array<float, 3>{-root, 0, root});
}

TEST(MatureLayerNorm, AcceptsAnEmptyLastDimension) {
  constexpr std::array<std::int64_t, 2> shape{3, 0};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer storage = Buffer::create(hyle::mem::system_resource(), 0).value();
  const ConstTensorView input =
      ConstTensorView::create(description, storage.view()).value();
  const TensorView output =
      TensorView::create(description, storage.view()).value();
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, LayerNorm{}, description).value();
  auto invocation = prepared.create_invocation().value();

  EXPECT_TRUE(invocation.submit(queue, input, output)->wait().ok());
}

template <class Value, class Scale, class Bias>
void expect_half_norm(std::size_t hidden, bool rms, std::size_t arity,
                      float epsilon = 1.0e-6F, bool native = false) {
  constexpr std::size_t rows = 3;
  const auto width = static_cast<std::int64_t>(hidden);
  const auto desc = TensorDesc::dense(half_test::data_type<Value>(),
                                      std::array<std::int64_t, 2>{rows, width})
                        .value();
  const auto scale_desc = TensorDesc::dense(half_test::data_type<Scale>(),
                                            std::array<std::int64_t, 1>{width})
                              .value();
  const auto bias_desc = TensorDesc::dense(half_test::data_type<Bias>(),
                                           std::array<std::int64_t, 1>{width})
                             .value();
  std::vector<Value> values(rows * hidden);
  std::vector<Value> other(rows * hidden);
  std::vector<Scale> scale(hidden);
  std::vector<Bias> bias(hidden);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const double value =
        index / hidden == 1
            ? 2.0
            : static_cast<double>(static_cast<int>(index % 97) - 48) / 16;
    values[index] = half_test::encode<Value>(value);
    other[index] = half_test::encode<Value>(value * 0.5 - 0.25);
  }
  for (std::size_t index = 0; index < hidden; ++index) {
    scale[index] =
        half_test::encode<Scale>(1.001 + static_cast<double>(index % 7) / 16);
    bias[index] =
        half_test::encode<Bias>(0.101 + static_cast<double>(index % 5) / 32);
  }
  // An F32 scale cannot be narrowed to F16 before the affine step: this
  // finite parameter is outside F16, while the selected output is finite.
  if constexpr (std::is_same_v<Value, hyle::simd::float16_t> &&
                std::is_same_v<Scale, float>) {
    if (hidden > 1) {
      scale[0] = 100000.0F;
      values[0] = half_test::encode<Value>(0.03125);
      other[0] = half_test::encode<Value>(0.015625);
    }
  }
  Buffer first = half_test::buffer<Value>(values);
  Buffer second = half_test::buffer<Value>(other);
  Buffer scale_buffer = half_test::buffer<Scale>(scale);
  Buffer bias_buffer = half_test::buffer<Bias>(bias);
  const auto first_view = TensorView::create(desc, first.view()).value();
  const auto second_view = TensorView::create(desc, second.view()).value();
  const auto scale_view =
      ConstTensorView::create(scale_desc, scale_buffer.view()).value();
  const auto bias_view =
      ConstTensorView::create(bias_desc, bias_buffer.view()).value();
  auto context = native
                     ? hyle::compute::test::native_context().value()
                     : hyle::compute::Context::create(
                           hyle::compute::cpu::device(),
                           hyle::compute::aten_cpu::provider())
                           .value();
  auto queue = context.create_queue().value();
  auto prepared = [&] {
    if (rms) {
      const hyle::compute::RmsNorm operation{.epsilon = epsilon};
      return arity == 1
                 ? hyle::compute::prepare(context, operation, desc)
                 : hyle::compute::prepare(context, operation, desc, scale_desc);
    }
    const LayerNorm operation{.epsilon = epsilon};
    if (arity == 1)
      return hyle::compute::prepare(context, operation, desc);
    return arity == 2
               ? hyle::compute::prepare(context, operation, desc, scale_desc)
               : hyle::compute::prepare(context, operation, desc, scale_desc,
                                        bias_desc);
  }();
  ASSERT_TRUE(prepared.has_value());
  EXPECT_EQ(prepared->output_description().value(), desc);
  auto first_invocation = prepared->create_invocation().value();
  auto second_invocation = prepared->create_invocation().value();
  const auto run = [&](auto &invocation, const TensorView &view) {
    const std::array inputs{view.as_const(), scale_view, bias_view};
    return invocation.submit(queue, std::span(inputs).first(arity), view)
        ->wait();
  };
  ASSERT_TRUE(run(first_invocation, first_view).ok());
  const auto retained = half_test::read<Value>(first);
  ASSERT_TRUE(run(second_invocation, second_view).ok());
  const auto second_output = half_test::read<Value>(second);
  const auto still_retained = half_test::read<Value>(first);
  for (std::size_t index = 0; index < retained.size(); ++index)
    EXPECT_EQ(still_retained[index].bits, retained[index].bits);
  const auto check = [&](std::span<const Value> source,
                         std::span<const Value> output) {
    for (std::size_t row = 0; hidden != 0 && row < rows; ++row) {
      double mean = 0.0;
      if (!rms) {
        for (std::size_t column = 0; column < hidden; ++column)
          mean += half_test::decode(source[row * hidden + column]);
        mean /= static_cast<double>(hidden);
      }
      double squares = 0.0;
      for (std::size_t column = 0; column < hidden; ++column) {
        const double centered =
            half_test::decode(source[row * hidden + column]) - mean;
        squares += centered * centered;
      }
      const double inverse =
          1.0 / std::sqrt(squares / static_cast<double>(hidden) + epsilon);
      for (std::size_t column = 0; column < hidden; ++column) {
        double expected =
            (half_test::decode(source[row * hidden + column]) - mean) * inverse;
        if (arity > 1)
          expected *= half_test::decode(scale[column]);
        if (arity > 2)
          expected += half_test::decode(bias[column]);
        SCOPED_TRACE(row * hidden + column);
        half_test::expect_near(output[row * hidden + column], expected);
      }
    }
  };
  check(values, retained);
  check(other, second_output);
}

template <class Value> void expect_half_norm_domains() {
  for (const std::size_t hidden : {0, 1, 127, 128, 129, 257, 2048}) {
    SCOPED_TRACE(hidden);
    for (const bool rms : {false, true}) {
      SCOPED_TRACE(rms);
      expect_half_norm<Value, Value, Value>(hidden, rms, 1);
      expect_half_norm<Value, Value, Value>(hidden, rms, 2);
      expect_half_norm<Value, float, Value>(hidden, rms, 2);
      if (!rms) {
        expect_half_norm<Value, Value, Value>(hidden, false, 3);
        expect_half_norm<Value, Value, float>(hidden, false, 3);
        expect_half_norm<Value, float, Value>(hidden, false, 3);
        expect_half_norm<Value, float, float>(hidden, false, 3);
      }
    }
  }
  expect_half_norm<Value, Value, Value>(1, false, 1, 0.0F);
}

TEST(MatureNormalization, HalfStorageKeepsParametersAndIndependentInvocations) {
  expect_half_norm_domains<hyle::simd::float16_t>();
  expect_half_norm_domains<hyle::simd::bfloat16_t>();
}

TEST(NativeRmsNorm, QualifiedBfloatKeepsScaleAndIndependentReplays) {
  using Value = hyle::simd::bfloat16_t;
  for (const std::size_t hidden : {0, 1, 127, 128, 129, 2048, 4096}) {
    SCOPED_TRACE(hidden);
    expect_half_norm<Value, Value, Value>(hidden, true, 1, 1.0e-5F, true);
    expect_half_norm<Value, Value, Value>(hidden, true, 2, 1.0e-5F, true);
    expect_half_norm<Value, float, Value>(hidden, true, 2, 1.0e-5F, true);
  }
}

TEST(MatureLayerNorm, RejectsUnsupportedStorageAndAccumulation) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  const TensorDesc f64 = TensorDesc::dense(DataType::kFloat64, shape).value();
  const TensorDesc f32 = TensorDesc::dense(DataType::kFloat32, shape).value();
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();

  auto storage = hyle::compute::prepare(context, LayerNorm{}, f64);
  ASSERT_FALSE(storage.has_value());
  EXPECT_TRUE(storage.error().IsUnimplemented());

  auto accumulation = hyle::compute::prepare(
      context, LayerNorm{.accumulation_type = DataType::kFloat64}, f32);
  ASSERT_FALSE(accumulation.has_value());
  EXPECT_TRUE(accumulation.error().IsUnimplemented());
  const auto rms_accumulation = hyle::compute::prepare(
      context, hyle::compute::RmsNorm{.accumulation_type = DataType::kFloat64},
      f32);
  ASSERT_FALSE(rms_accumulation.has_value());
  EXPECT_TRUE(rms_accumulation.error().IsUnimplemented());
}

TEST(NativeRmsNorm, PinsExecutorAndUsesOneBoundedSynchronousFanOut) {
  constexpr std::array<std::int64_t, 2> shape{8, 8192};
  const TensorDesc description =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const std::size_t bytes =
      static_cast<std::size_t>(shape[0] * shape[1]) * sizeof(float);
  Buffer input =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  Buffer output =
      Buffer::create(hyle::mem::system_resource(), bytes, alignof(float))
          .value();
  ASSERT_TRUE(input.view().fill(std::byte{0}).ok());
  ASSERT_TRUE(output.view().fill(std::byte{0}).ok());

  ParallelProbe probe;
  const hyle::compute::ParallelExecutorRef executor(
      &probe, 4, &probe_parallel_for, &probe_in_parallel);
  auto context =
      hyle::compute::test::native_context({.executor = executor}).value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, hyle::compute::RmsNorm{}, description).value();
  auto invocation = prepared.create_invocation().value();
  const ConstTensorView input_view =
      ConstTensorView::create(description, input.view()).value();
  const TensorView output_view =
      TensorView::create(description, output.view()).value();

  ASSERT_TRUE(invocation.submit(queue, input_view, output_view)->wait().ok());
  EXPECT_EQ(probe.episodes, 1U);
  EXPECT_EQ(probe.tasks, 4U);

  probe.nested = true;
  ASSERT_TRUE(invocation.submit(queue, input_view, output_view)->wait().ok());
  EXPECT_EQ(probe.episodes, 1U);
  EXPECT_EQ(probe.tasks, 4U);
}

} // namespace
