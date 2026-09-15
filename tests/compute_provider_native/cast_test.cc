#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

#include "../compute/native_context_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/dtype.h"

namespace {

using hyle::compute::Cast;
using hyle::compute::ConstTensorView;
using hyle::compute::DataType;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::mem::Buffer;

template <class Value> constexpr DataType data_type() noexcept {
  if constexpr (std::is_same_v<Value, float>)
    return DataType::kFloat32;
  else if constexpr (std::is_same_v<Value, hyle::simd::float16_t>)
    return DataType::kFloat16;
  else if constexpr (std::is_same_v<Value, hyle::simd::bfloat16_t>)
    return DataType::kBFloat16;
  else if constexpr (std::is_same_v<Value, hyle::simd::float8_e4m3_t>)
    return DataType::kFloat8E4M3FN;
  else {
    static_assert(std::is_same_v<Value, hyle::simd::float8_e5m2_t>);
    return DataType::kFloat8E5M2;
  }
}

template <class Value> Value encode(float value) noexcept {
  if constexpr (std::is_same_v<Value, float>)
    return value;
  else if constexpr (std::is_same_v<Value, hyle::simd::float16_t>)
    return hyle::simd::f32_to_f16(value);
  else if constexpr (std::is_same_v<Value, hyle::simd::bfloat16_t>)
    return hyle::simd::f32_to_bf16(value);
  else if constexpr (std::is_same_v<Value, hyle::simd::float8_e4m3_t>)
    return hyle::simd::f32_to_f8e4m3(value);
  else {
    static_assert(std::is_same_v<Value, hyle::simd::float8_e5m2_t>);
    return hyle::simd::f32_to_f8e5m2(value);
  }
}

template <class Value> float decode(Value value) noexcept {
  if constexpr (std::is_same_v<Value, float>)
    return value;
  else if constexpr (std::is_same_v<Value, hyle::simd::float16_t>)
    return hyle::simd::f16_to_f32(value);
  else if constexpr (std::is_same_v<Value, hyle::simd::bfloat16_t>)
    return hyle::simd::bf16_to_f32(value);
  else if constexpr (std::is_same_v<Value, hyle::simd::float8_e4m3_t>)
    return hyle::simd::f8e4m3_to_f32(value);
  else {
    static_assert(std::is_same_v<Value, hyle::simd::float8_e5m2_t>);
    return hyle::simd::f8e5m2_to_f32(value);
  }
}

template <class Source, class Destination> void expect_dense_cast() {
  constexpr std::array<std::int64_t, 1> shape{7};
  constexpr std::array<float, 7> values{-32.0F, -2.0F, -0.0F, 0.25F,
                                        1.0F,   3.5F,  64.0F};
  std::array<Source, values.size()> source{};
  std::array<Destination, values.size()> expected{};
  for (std::size_t index = 0; index < values.size(); ++index) {
    source[index] = encode<Source>(values[index]);
    expected[index] = encode<Destination>(decode(source[index]));
  }

  const TensorDesc input_desc =
      TensorDesc::dense(data_type<Source>(), shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(data_type<Destination>(), shape).value();
  Buffer input_storage = Buffer::create(hyle::mem::system_resource(),
                                        sizeof(source), alignof(Source))
                             .value();
  Buffer output_storage = Buffer::create(hyle::mem::system_resource(),
                                         sizeof(expected), alignof(Destination))
                              .value();
  ASSERT_TRUE(
      input_storage.view().write(std::as_bytes(std::span(source))).ok());
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared = hyle::compute::prepare(
      context, Cast{data_type<Destination>()}, input_desc);
  ASSERT_TRUE(prepared.has_value());
  auto invocation = prepared->create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());

  std::array<Destination, expected.size()> actual{};
  ASSERT_TRUE(output_storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  EXPECT_EQ(std::memcmp(actual.data(), expected.data(), sizeof(expected)), 0);
}

template <class Source> void expect_all_destinations() {
  expect_dense_cast<Source, float>();
  expect_dense_cast<Source, hyle::simd::float16_t>();
  expect_dense_cast<Source, hyle::simd::bfloat16_t>();
  expect_dense_cast<Source, hyle::simd::float8_e4m3_t>();
  expect_dense_cast<Source, hyle::simd::float8_e5m2_t>();
}

TEST(NativeCast, CoversCompleteInferenceStorageConversionGraph) {
  expect_all_destinations<float>();
  expect_all_destinations<hyle::simd::float16_t>();
  expect_all_destinations<hyle::simd::bfloat16_t>();
  expect_all_destinations<hyle::simd::float8_e4m3_t>();
  expect_all_destinations<hyle::simd::float8_e5m2_t>();
}

TEST(NativeCast, ConvertsDenseFloat32ToFloat16Bits) {
  constexpr std::array<std::int64_t, 1> shape{5};
  constexpr std::array<float, 5> values{0.0F, 1.0F, -2.0F, 0.5F, 65504.0F};
  constexpr std::array<std::uint16_t, 5> expected{0x0000, 0x3c00, 0xc000,
                                                  0x3800, 0x7bff};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat16, shape).value();
  Buffer input_storage = Buffer::create(hyle::mem::system_resource(),
                                        sizeof(values), alignof(float))
                             .value();
  Buffer output_storage =
      Buffer::create(hyle::mem::system_resource(), sizeof(expected),
                     alignof(std::uint16_t))
          .value();
  ASSERT_TRUE(
      input_storage.view().write(std::as_bytes(std::span(values))).ok());
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();

  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Cast{DataType::kFloat16}, input_desc)
          .value();
  ASSERT_EQ(prepared.output_description().value(), output_desc);
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());

  std::array<std::uint16_t, expected.size()> actual{};
  ASSERT_TRUE(output_storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  EXPECT_EQ(actual, expected);
}

TEST(NativeCast, MaterializesStridedFloat16ToFloat32) {
  constexpr std::array<std::int64_t, 2> shape{2, 2};
  constexpr std::array<std::int64_t, 2> strides{5, 2};
  constexpr std::array<std::uint16_t, 4> bits{0x3c00, 0x4000, 0x4200, 0x4400};
  constexpr std::array<std::size_t, 4> offsets{0, 2, 5, 7};
  constexpr std::array<float, 4> expected{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<std::byte, 9> physical{};
  for (std::size_t index = 0; index < bits.size(); ++index)
    std::memcpy(physical.data() + offsets[index], &bits[index],
                sizeof(bits[0]));
  const TensorDesc input_desc =
      TensorDesc::strided_bytes(DataType::kFloat16, shape, strides).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kFloat32, shape).value();
  Buffer input_storage =
      Buffer::create(hyle::mem::system_resource(), sizeof(physical),
                     alignof(std::uint16_t))
          .value();
  Buffer output_storage = Buffer::create(hyle::mem::system_resource(),
                                         sizeof(expected), alignof(float))
                              .value();
  ASSERT_TRUE(
      input_storage.view().write(std::span<const std::byte>(physical)).ok());
  const ConstTensorView input =
      ConstTensorView::create(input_desc, input_storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, output_storage.view()).value();

  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Cast{DataType::kFloat32}, input_desc)
          .value();
  auto invocation = prepared.create_invocation().value();
  ASSERT_TRUE(invocation.submit(queue, input, output)->wait().ok());

  std::array<float, expected.size()> actual{};
  ASSERT_TRUE(output_storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  EXPECT_EQ(actual, expected);
}

TEST(NativeCast, RejectsUnfrozenIntegerConversionSemantics) {
  constexpr std::array<std::int64_t, 1> shape{4};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kInt32, shape).value();
  auto context = hyle::compute::test::native_context().value();
  auto prepared =
      hyle::compute::prepare(context, Cast{DataType::kFloat32}, input_desc);
  ASSERT_FALSE(prepared.has_value());
  EXPECT_TRUE(prepared.error().IsUnimplemented());
}

TEST(NativeCast, RejectsOverlappingInputAndOutputStorage) {
  constexpr std::array<std::int64_t, 1> shape{4};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kFloat16, shape).value();
  const TensorDesc output_desc =
      TensorDesc::dense(DataType::kBFloat16, shape).value();
  Buffer storage =
      Buffer::create(hyle::mem::system_resource(), 8, alignof(std::uint16_t))
          .value();
  const ConstTensorView input =
      ConstTensorView::create(input_desc, storage.view()).value();
  const TensorView output =
      TensorView::create(output_desc, storage.view()).value();

  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Cast{DataType::kBFloat16}, input_desc)
          .value();
  auto invocation = prepared.create_invocation().value();
  auto submitted = invocation.submit(queue, input, output);
  ASSERT_FALSE(submitted.has_value());
  EXPECT_TRUE(submitted.error().IsInvalidArgument());
}

} // namespace
