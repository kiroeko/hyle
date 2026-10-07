#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "../compute/native_context_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::compute::ConstTensorView;
using hyle::compute::Copy;
using hyle::compute::DataType;
using hyle::compute::Fill;
using hyle::compute::Operation;
using hyle::compute::TensorDesc;
using hyle::compute::TensorShape;
using hyle::compute::TensorView;
using hyle::compute::View;
using hyle::mem::Buffer;

TEST(NativeTransform, ValidatesAndExecutesZeroKernelSliceView) {
  constexpr std::array<std::int64_t, 2> shape{2, 3};
  constexpr std::array<std::int32_t, 6> values{1, 2, 3, 4, 5, 6};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kInt32, shape).value();
  const View slice = View::slice(1, 1, 2).value();
  const std::array input_descriptions{input_desc};
  std::array<TensorDesc, 1> output_descriptions{};
  ASSERT_TRUE(hyle::compute::infer_output_descriptions(
                  Operation(slice), input_descriptions, output_descriptions)
                  .ok());

  Buffer storage =
      Buffer::create(hyle::mem::system_resource(), sizeof(values), 4).value();
  ASSERT_TRUE(storage.view().write(std::as_bytes(std::span(values))).ok());
  const ConstTensorView input =
      ConstTensorView::create(input_desc, storage.view()).value();
  const std::size_t output_bytes =
      output_descriptions[0].required_bytes().value();
  const TensorView output =
      TensorView::create(
          output_descriptions[0],
          storage.view().subview(sizeof(std::int32_t), output_bytes).value())
          .value();

  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, slice, input_desc).value();
  ASSERT_EQ(prepared.output_description().value(), output_descriptions[0]);
  const auto alias = prepared.alias_support(0, 0).value();
  EXPECT_TRUE(alias.supported);
  EXPECT_EQ(alias.byte_offset, sizeof(std::int32_t));
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation
                  .submit(queue, std::span<const ConstTensorView>(&input, 1),
                          std::span<const TensorView>(&output, 1))
                  ->wait()
                  .ok());

  const TensorView wrong =
      TensorView::create(output_descriptions[0],
                         storage.view().subview(0, output_bytes).value())
          .value();
  auto rejected =
      invocation.submit(queue, std::span<const ConstTensorView>(&input, 1),
                        std::span<const TensorView>(&wrong, 1));
  ASSERT_FALSE(rejected.has_value());
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
}

TEST(NativeTransform, ExactStridedCopyRejectsNativeAndUsesMatureMaterialization) {
  constexpr std::array<std::int64_t, 2> shape{2, 2};
  constexpr std::array<std::int64_t, 2> strides{12, 4};
  constexpr std::array<std::int32_t, 6> physical{1, 2, 99, 3, 4, 98};
  constexpr std::array<std::int32_t, 4> expected{1, 2, 3, 4};
  const TensorDesc input_desc =
      TensorDesc::strided_bytes(DataType::kInt32, shape, strides).value();
  const std::array input_descriptions{input_desc};
  std::array<TensorDesc, 1> output_descriptions{};
  ASSERT_TRUE(hyle::compute::infer_output_descriptions(
                  Operation(Copy{}), input_descriptions, output_descriptions)
                  .ok());

  Buffer input_storage =
      Buffer::create(hyle::mem::system_resource(), sizeof(physical), 4).value();
  Buffer output_storage =
      Buffer::create(hyle::mem::system_resource(), sizeof(expected), 4).value();
  ASSERT_TRUE(
      input_storage.view().write(std::as_bytes(std::span(physical))).ok());
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_descriptions[0], output_storage.view()).value();

  auto context = hyle::compute::test::native_context().value();
  const auto rejected = hyle::compute::prepare(context, Copy{}, input_desc);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
  if (!hyle::compute::aten_cpu::provider())
    return;
  auto mature = hyle::compute::Context::create(hyle::compute::cpu::device(),
                                hyle::compute::aten_cpu::provider()).value();
  auto queue = mature.create_queue().value();
  auto prepared = hyle::compute::prepare(mature, Copy{}, input_desc).value();
  ASSERT_EQ(prepared.output_description().value(), output_descriptions[0]);
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation
                  .submit(queue, std::span<const ConstTensorView>(&input, 1),
                          std::span<const TensorView>(&output, 1))
                  ->wait()
                  .ok());

  std::array<std::int32_t, expected.size()> actual{};
  ASSERT_TRUE(output_storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  EXPECT_EQ(actual, expected);
}

TEST(MatureTransform, FillsExactTypedScalarBits) {
  constexpr std::array<std::int64_t, 2> dimensions{2, 3};
  const TensorShape shape = TensorShape::create(dimensions).value();
  const Fill fill{.output_shape = shape,
                  .output_type = DataType::kUInt32,
                  .value =
                      hyle::compute::ScalarValue(std::uint32_t{0x12345678})};
  constexpr std::array<std::uint32_t, 6> expected{
      0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678, 0x12345678};
  const std::array<TensorDesc, 0> input_descriptions{};
  std::array<TensorDesc, 1> output_descriptions{};
  ASSERT_TRUE(hyle::compute::infer_output_descriptions(
                  Operation(fill), input_descriptions, output_descriptions)
                  .ok());

  Buffer storage =
      Buffer::create(hyle::mem::system_resource(), sizeof(expected), 4).value();
  const TensorView output =
      TensorView::create(output_descriptions[0], storage.view()).value();
  auto context = hyle::compute::Context::create(
      hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(context, fill).value();
  ASSERT_EQ(prepared.output_description().value(), output_descriptions[0]);
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, output)->wait().ok());

  std::array<std::uint32_t, expected.size()> actual{};
  ASSERT_TRUE(
      storage.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, expected);
}

} // namespace
