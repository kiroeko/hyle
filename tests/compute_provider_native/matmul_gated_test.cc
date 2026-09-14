#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::GatedActivation;
using hyle::compute::GatedActivationKind;
using hyle::compute::GatedInputKind;
using hyle::compute::MatMul;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::Buffer;

Buffer make_buffer(std::span<const float> values) {
  Buffer result =
      Buffer::create(hyle::mem::system_resource(),
                     values.size_bytes(), alignof(float))
          .value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());
  return result;
}

std::vector<float> read_buffer(const Buffer &buffer, std::size_t count) {
  std::vector<float> result(count);
  EXPECT_TRUE(
      buffer.view().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}

void expect_matmul(const TensorDesc &left_desc, std::span<const float> left,
                   const TensorDesc &right_desc,
                   std::span<const float> right,
                   std::span<const float> expected) {
  Buffer left_storage = make_buffer(left);
  Buffer right_storage = make_buffer(right);
  Buffer output_storage =
      make_buffer(std::vector<float>(expected.size(), 0.0F));
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared_result =
      hyle::compute::prepare(context, MatMul{}, left_desc, right_desc);
  ASSERT_TRUE(prepared_result.has_value())
      << prepared_result.error().message();
  auto prepared = std::move(*prepared_result);
  auto output_desc = prepared.output_description();
  ASSERT_TRUE(output_desc.has_value()) << output_desc.error().message();
  const auto left_view =
      ConstTensorView::create(left_desc, left_storage.view()).value();
  const auto right_view =
      ConstTensorView::create(right_desc, right_storage.view()).value();
  const auto output_view =
      TensorView::create(*output_desc, output_storage.view()).value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, left_view, right_view, output_view)
                  ->wait()
                  .ok());
  const auto actual = read_buffer(output_storage, expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index],
                std::abs(expected[index]) * 3.0e-5F + 3.0e-6F)
        << index;
  }
}

float activate(GatedActivationKind kind, float value) {
  if (value == -std::numeric_limits<float>::infinity())
    return 0.0F;
  switch (kind) {
  case GatedActivationKind::kSilu: {
    const float exponential = std::exp(-std::abs(value));
    const float sigmoid = value < 0.0F ? exponential / (1.0F + exponential)
                                       : 1.0F / (1.0F + exponential);
    return value * sigmoid;
  }
  case GatedActivationKind::kGelu:
    return 0.5F * value *
           (1.0F + std::erf(value * 0.70710678118654752440F));
  case GatedActivationKind::kGeluTanh:
    return 0.5F * value *
           (1.0F + std::tanh(0.79788456080286535588F * value *
                             std::fma(0.044715F, value * value, 1.0F)));
  }
  return 0.0F;
}

TEST(NativeMatMul, BroadcastsBothBatchOperandsAndSupportsTransposedWeights) {
  constexpr std::size_t first_batches = 2;
  constexpr std::size_t second_batches = 3;
  constexpr std::size_t rows = 2;
  constexpr std::size_t inner = 3;
  constexpr std::size_t columns = 2;
  constexpr std::array<std::int64_t, 4> left_shape{2, 1, 2, 3};
  constexpr std::array<std::int64_t, 4> right_shape{1, 3, 3, 2};
  constexpr std::array<std::int64_t, 4> physical_right_shape{1, 3, 2, 3};
  constexpr std::array<std::int64_t, 4> output_shape{2, 3, 2, 2};

  const std::vector<float> left{1,  2, 3, 4,  5,  6,
                                -1, 0, 2, 0.5F, -2, 3};
  const std::vector<float> right{
      1, 2, 3, 4, 5, 6,
      -1, 1, 2, 0, 0.5F, -2,
      3, -1, 0, 2, 4, 0.25F,
  };
  std::vector<float> expected(first_batches * second_batches * rows * columns,
                              0.0F);
  for (std::size_t first_batch = 0; first_batch < first_batches;
       ++first_batch) {
    for (std::size_t second_batch = 0; second_batch < second_batches;
         ++second_batch) {
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
          float sum = 0.0F;
          for (std::size_t index = 0; index < inner; ++index) {
            sum += left[(first_batch * rows + row) * inner + index] *
                   right[(second_batch * inner + index) * columns + column];
          }
          expected[((first_batch * second_batches + second_batch) * rows +
                    row) *
                       columns +
                   column] = sum;
        }
      }
    }
  }

  std::vector<float> transposed(right.size());
  for (std::size_t batch = 0; batch < second_batches; ++batch) {
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t index = 0; index < inner; ++index) {
        transposed[(batch * columns + column) * inner + index] =
            right[(batch * inner + index) * columns + column];
      }
    }
  }

  const TensorDesc left_desc =
      TensorDesc::dense(DataType::kFloat32, left_shape).value();
  const TensorDesc right_desc =
      TensorDesc::dense(DataType::kFloat32, right_shape).value();
  const TensorDesc transposed_desc =
      TensorDesc::dense(DataType::kFloat32, physical_right_shape)
          ->transposed()
          .value();
  const TensorDesc expected_output =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();
  Buffer left_storage = make_buffer(left);
  Buffer right_storage = make_buffer(right);
  Buffer transposed_storage = make_buffer(transposed);
  Buffer output_storage = make_buffer(std::vector<float>(expected.size()));
  const ConstTensorView left_view =
      ConstTensorView::create(left_desc, left_storage.view()).value();
  const TensorView output_view =
      TensorView::create(expected_output, output_storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();

  const auto run = [&](const TensorDesc &description, const Buffer &storage) {
    auto prepared_result =
        hyle::compute::prepare(context, MatMul{}, left_desc, description);
    ASSERT_TRUE(prepared_result.has_value())
        << prepared_result.error().message();
    auto prepared = std::move(*prepared_result);
    EXPECT_EQ(prepared.output_description().value(), expected_output);
    auto invocation = prepared.create_invocation().value();
    const ConstTensorView right_view =
        ConstTensorView::create(description, storage.view()).value();
    ASSERT_TRUE(invocation.submit(queue, left_view, right_view, output_view)
                    ->wait()
                    .ok());
    const auto actual = read_buffer(output_storage, expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      EXPECT_NEAR(actual[index], expected[index],
                  std::abs(expected[index]) * 3.0e-5F + 3.0e-6F)
          << index;
    }
  };

  run(right_desc, right_storage);
  run(transposed_desc, transposed_storage);
}

TEST(NativeMatMul, AcceptsAnEmptyBroadcastBatch) {
  constexpr std::array<std::int64_t, 3> left_shape{0, 2, 3};
  constexpr std::array<std::int64_t, 3> right_shape{1, 3, 4};
  constexpr std::array<std::int64_t, 3> output_shape{0, 2, 4};
  const TensorDesc left_desc =
      TensorDesc::dense(DataType::kFloat32, left_shape).value();
  const TensorDesc right_desc =
      TensorDesc::dense(DataType::kFloat32, right_shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, output_shape).value();
  Buffer left = Buffer::create(hyle::mem::system_resource(), 0).value();
  Buffer right = make_buffer(std::vector<float>(12, 1.0F));
  Buffer output = Buffer::create(hyle::mem::system_resource(), 0).value();
  const auto left_view =
      ConstTensorView::create(left_desc, left.view()).value();
  const auto right_view =
      ConstTensorView::create(right_desc, right.view()).value();
  const auto output_view =
      TensorView::create(output_desc, output.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();
  auto prepared_result =
      hyle::compute::prepare(context, MatMul{}, left_desc, right_desc);
  ASSERT_TRUE(prepared_result.has_value())
      << prepared_result.error().message();
  auto prepared = std::move(*prepared_result);
  auto invocation = prepared.create_invocation().value();
  EXPECT_TRUE(invocation.submit(queue, left_view, right_view, output_view)
                  ->wait()
                  .ok());

  constexpr auto maximum = (std::numeric_limits<std::int64_t>::max)();
  constexpr std::array<std::int64_t, 4> huge_empty_left_shape{maximum,
                                                              maximum, 0, 1};
  constexpr std::array<std::int64_t, 4> scalar_right_shape{1, 1, 1, 1};
  const TensorDesc huge_empty_left =
      TensorDesc::dense(DataType::kFloat32, huge_empty_left_shape).value();
  const TensorDesc scalar_right =
      TensorDesc::dense(DataType::kFloat32, scalar_right_shape).value();
  auto huge_empty =
      hyle::compute::prepare(context, MatMul{}, huge_empty_left, scalar_right);
  ASSERT_TRUE(huge_empty.has_value()) << huge_empty.error().message();
}

TEST(NativeMatMul, ZeroInnerDimensionProducesZeroWithoutReadingInputs) {
  const auto run = [](std::span<const std::int64_t> left_shape,
                      std::span<const std::int64_t> right_shape,
                      std::span<const std::int64_t> output_shape) {
    const TensorDesc left_desc =
        TensorDesc::dense(DataType::kFloat32, left_shape).value();
    const TensorDesc right_desc =
        TensorDesc::dense(DataType::kFloat32, right_shape).value();
    const TensorDesc output_desc =
        TensorDesc::dense(DataType::kFloat32, output_shape).value();
    Buffer left = Buffer::create(hyle::mem::system_resource(), 0).value();
    Buffer right = Buffer::create(hyle::mem::system_resource(), 0).value();
    const std::size_t output_elements = output_desc.element_count().value();
    Buffer output =
        make_buffer(std::vector<float>(output_elements, 17.0F));
    const auto left_view =
        ConstTensorView::create(left_desc, left.view()).value();
    const auto right_view =
        ConstTensorView::create(right_desc, right.view()).value();
    const auto output_view =
        TensorView::create(output_desc, output.view()).value();
    auto context = hyle::compute::cpu::create_context().value();
    auto queue = context.create_queue().value();
    auto prepared =
        hyle::compute::prepare(context, MatMul{}, left_desc, right_desc)
            .value();
    auto invocation = prepared.create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, left_view, right_view, output_view)
                    ->wait()
                    .ok());
    EXPECT_EQ(read_buffer(output, output_elements),
              std::vector<float>(output_elements, 0.0F));
  };

  constexpr std::array<std::int64_t, 2> matrix_left{2, 0};
  constexpr std::array<std::int64_t, 2> matrix_right{0, 3};
  constexpr std::array<std::int64_t, 2> matrix_output{2, 3};
  run(matrix_left, matrix_right, matrix_output);

  constexpr std::array<std::int64_t, 3> batched_left{2, 2, 0};
  constexpr std::array<std::int64_t, 3> batched_right{1, 0, 3};
  constexpr std::array<std::int64_t, 3> batched_output{2, 2, 3};
  run(batched_left, batched_right, batched_output);
}

TEST(NativeMatMul, BroadcastsAnOperandWithNoExplicitBatchAxes) {
  constexpr std::array<std::int64_t, 2> matrix_left_shape{2, 3};
  constexpr std::array<std::int64_t, 3> batched_right_shape{2, 3, 2};
  constexpr std::array<std::int64_t, 3> batched_left_shape{2, 2, 3};
  constexpr std::array<std::int64_t, 2> matrix_right_shape{3, 2};
  const TensorDesc matrix_left =
      TensorDesc::dense(DataType::kFloat32, matrix_left_shape).value();
  const TensorDesc batched_right =
      TensorDesc::dense(DataType::kFloat32, batched_right_shape).value();
  const TensorDesc batched_left =
      TensorDesc::dense(DataType::kFloat32, batched_left_shape).value();
  const TensorDesc matrix_right =
      TensorDesc::dense(DataType::kFloat32, matrix_right_shape).value();

  const std::vector<float> left{1, 2, 3, 4, 5, 6};
  const std::vector<float> right_batches{
      1, 2, 3, 4, 5, 6,
      -1, 1, 2, 0, 0.5F, -2,
  };
  std::vector<float> expected_left_broadcast(8, 0.0F);
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t column = 0; column < 2; ++column) {
        for (std::size_t inner = 0; inner < 3; ++inner) {
          expected_left_broadcast[(batch * 2 + row) * 2 + column] +=
              left[row * 3 + inner] *
              right_batches[(batch * 3 + inner) * 2 + column];
        }
      }
    }
  }
  expect_matmul(matrix_left, left, batched_right, right_batches,
                expected_left_broadcast);

  const std::vector<float> left_batches{
      1, 2, 3, 4, 5, 6,
      -1, 0, 2, 0.5F, -2, 3,
  };
  const std::vector<float> right{1, 2, 3, 4, 5, 6};
  std::vector<float> expected_right_broadcast(8, 0.0F);
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t column = 0; column < 2; ++column) {
        for (std::size_t inner = 0; inner < 3; ++inner) {
          expected_right_broadcast[(batch * 2 + row) * 2 + column] +=
              left_batches[(batch * 2 + row) * 3 + inner] *
              right[inner * 2 + column];
        }
      }
    }
  }
  expect_matmul(batched_left, left_batches, matrix_right, right,
                expected_right_broadcast);
}

TEST(NativeGatedActivation, ExecutesEveryModeForPackedAndSeparateInputs) {
  constexpr std::size_t rows = 2;
  constexpr std::size_t hidden = 5;
  const std::vector<float> gate{
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      0.5F,
      3,
      2,
      -3,
      1,
      4,
      -0.25F,
  };
  const std::vector<float> value{0.5F, 2, -1, 3, 0.25F,
                                 -2, 1.5F, 4, -0.5F, 8};
  std::vector<float> packed(rows * hidden * 2);
  for (std::size_t row = 0; row < rows; ++row) {
    std::copy_n(gate.data() + row * hidden, hidden,
                packed.data() + row * hidden * 2);
    std::copy_n(value.data() + row * hidden, hidden,
                packed.data() + row * hidden * 2 + hidden);
  }

  const std::array<std::int64_t, 2> separate_shape{
      static_cast<std::int64_t>(rows), static_cast<std::int64_t>(hidden)};
  const std::array<std::int64_t, 2> packed_shape{
      static_cast<std::int64_t>(rows), static_cast<std::int64_t>(hidden * 2)};
  const TensorDesc separate_desc =
      TensorDesc::dense(DataType::kFloat32, separate_shape).value();
  const TensorDesc packed_desc =
      TensorDesc::dense(DataType::kFloat32, packed_shape).value();
  Buffer gate_storage = make_buffer(gate);
  Buffer value_storage = make_buffer(value);
  Buffer packed_storage = make_buffer(packed);
  Buffer output_storage = make_buffer(std::vector<float>(gate.size()));
  const auto gate_view =
      ConstTensorView::create(separate_desc, gate_storage.view()).value();
  const auto value_view =
      ConstTensorView::create(separate_desc, value_storage.view()).value();
  const auto packed_view =
      ConstTensorView::create(packed_desc, packed_storage.view()).value();
  const auto output_view =
      TensorView::create(separate_desc, output_storage.view()).value();
  auto context = hyle::compute::cpu::create_context().value();
  auto queue = context.create_queue().value();

  for (const GatedActivationKind kind : {GatedActivationKind::kSilu,
                                         GatedActivationKind::kGelu,
                                         GatedActivationKind::kGeluTanh}) {
    auto packed_prepared =
        hyle::compute::prepare(
            context,
            GatedActivation{.activation = kind,
                            .inputs = GatedInputKind::kPackedHalves},
            packed_desc)
            .value();
    auto invocation = packed_prepared.create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, packed_view, output_view)->wait().ok());
    const auto packed_actual = read_buffer(output_storage, gate.size());

    auto separate_prepared =
        hyle::compute::prepare(
            context,
            GatedActivation{.activation = kind,
                            .inputs = GatedInputKind::kSeparate},
            separate_desc, separate_desc)
            .value();
    invocation = separate_prepared.create_invocation().value();
    ASSERT_TRUE(invocation.submit(queue, gate_view, value_view, output_view)
                    ->wait()
                    .ok());
    const auto separate_actual = read_buffer(output_storage, gate.size());

    for (std::size_t index = 0; index < gate.size(); ++index) {
      const float expected = activate(kind, gate[index]) * value[index];
      if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(packed_actual[index])) << index;
        EXPECT_TRUE(std::isnan(separate_actual[index])) << index;
        continue;
      }
      if (std::isinf(expected)) {
        EXPECT_EQ(packed_actual[index], expected) << index;
        EXPECT_EQ(separate_actual[index], expected) << index;
        continue;
      }
      const float tolerance = std::abs(expected) * 1.0e-5F + 3.0e-6F;
      EXPECT_NEAR(packed_actual[index], expected, tolerance) << index;
      EXPECT_NEAR(separate_actual[index], expected, tolerance) << index;
    }
  }

  auto packed_prepared =
      hyle::compute::prepare(context, GatedActivation{}, packed_desc).value();
  auto packed_invocation = packed_prepared.create_invocation().value();
  const auto packed_prefix =
      packed_storage.view().subview(0, gate.size() * sizeof(float)).value();
  const auto packed_alias =
      TensorView::create(separate_desc, packed_prefix).value();
  auto rejected = packed_invocation.submit(queue, packed_view, packed_alias);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());

  auto separate_prepared =
      hyle::compute::prepare(
          context,
          GatedActivation{.inputs = GatedInputKind::kSeparate}, separate_desc,
          separate_desc)
          .value();
  auto separate_invocation = separate_prepared.create_invocation().value();
  const auto gate_alias =
      TensorView::create(separate_desc, gate_storage.view()).value();
  rejected =
      separate_invocation.submit(queue, gate_view, value_view, gate_alias);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
}

} // namespace
