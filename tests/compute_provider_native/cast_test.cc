#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

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

template <class Source, class Destination> void expect_dense_cast(bool native = false) {
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
  auto context = (native ? hyle::compute::test::native_context()
                        : hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider())).value();
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

TEST(NativeCast, ReplaysAllFourQualifiedStoragePairs) {
  expect_dense_cast<hyle::simd::float8_e4m3_t, hyle::simd::float16_t>(true);
  expect_dense_cast<hyle::simd::float8_e5m2_t, hyle::simd::bfloat16_t>(true);
  expect_dense_cast<hyle::simd::float8_e5m2_t, hyle::simd::float8_e4m3_t>(true);
  expect_dense_cast<hyle::simd::float8_e5m2_t, hyle::simd::float8_e5m2_t>(true);
}

TEST(MatureCast, CoversCompleteInferenceStorageConversionGraph) {
  expect_all_destinations<float>();
  expect_all_destinations<hyle::simd::float16_t>();
  expect_all_destinations<hyle::simd::bfloat16_t>();
  expect_all_destinations<hyle::simd::float8_e4m3_t>();
  expect_all_destinations<hyle::simd::float8_e5m2_t>();
}

TEST(MatureCast, ConvertsDenseFloat32ToFloat16Bits) {
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

  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(MatureCast, MaterializesStridedFloat16ToFloat32) {
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

  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
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

TEST(AtenCpuCast, RejectsUnsupportedIntegerConversionCarrier) {
  constexpr std::array<std::int64_t, 1> shape{4};
  const TensorDesc input_desc =
      TensorDesc::dense(DataType::kInt32, shape).value();
  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  auto prepared =
      hyle::compute::prepare(context, Cast{DataType::kFloat32}, input_desc);
  ASSERT_FALSE(prepared.has_value());
  EXPECT_TRUE(prepared.error().IsUnimplemented());
}

TEST(NativeCast, Int32ToFloat32RoundsEvenAcrossDenseAndOddByteStrides) {
  constexpr std::array<std::int64_t,2> shape{2,8}, strides{51,5};
  constexpr std::array<std::int32_t,16> values{
      0,1,-1,129279,16777215,16777216,16777217,16777219,
      -16777217,-16777219,2147483520,2147483583,2147483584,
      std::numeric_limits<std::int32_t>::max(),
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::min()+1};
  // Independent IEEE754 binary32 encodings, including both sides of 2^24
  // midpoint ties and the 2^31 carry. No floating cast constructs this oracle.
  constexpr std::array<std::uint32_t,16> expected{
      0x00000000u,0x3f800000u,0xbf800000u,0x47fc7f80u,
      0x4b7fffffu,0x4b800000u,0x4b800000u,0x4b800002u,
      0xcb800000u,0xcb800002u,0x4effffffu,0x4effffffu,
      0x4f000000u,0x4f000000u,0xcf000000u,0xcf000000u};
  struct RestoreRounding final {
    int mode=std::fegetround();
    ~RestoreRounding() { if (mode!=-1) std::fesetround(mode); }
  } restore_rounding;
  ASSERT_NE(restore_rounding.mode,-1);
  auto context=hyle::compute::test::native_context().value();
  auto queue=context.create_queue().value();
  for (const bool strided:{false,true}) {
    SCOPED_TRACE(strided);
    const auto input_desc=(strided ? TensorDesc::strided_bytes(DataType::kInt32,shape,strides)
                                 : TensorDesc::dense(DataType::kInt32,shape)).value();
    const auto output_desc=TensorDesc::dense(DataType::kFloat32,shape).value();
    auto source=Buffer::create(hyle::mem::system_resource(),input_desc.required_bytes().value(),64).value();
    auto destination=Buffer::create(hyle::mem::system_resource(),sizeof(expected),64).value();
    auto prepared=hyle::compute::prepare(context,Cast{DataType::kFloat32},input_desc);
    ASSERT_TRUE(prepared)<<prepared.error().message();
    EXPECT_EQ(prepared->output_description().value(),output_desc);
    auto invocation=prepared->create_invocation().value();
    const auto input=ConstTensorView::create(input_desc,source.view()).value();
    const auto output=TensorView::create(output_desc,destination.view()).value();
    for (int replay=0;replay<2;++replay) {
      SCOPED_TRACE(replay);
      ASSERT_EQ(std::fesetround(replay ? FE_DOWNWARD : FE_UPWARD),0);
      std::vector<std::byte> physical(source.size(),std::byte{0x5a});
      std::array<std::uint32_t,values.size()> answer{};
      for (std::size_t i=0;i<values.size();++i) {
        const auto from=replay ? values.size()-1-i : i;
        const auto offset=strided ? (i/8)*std::size_t(strides[0])+(i%8)*std::size_t(strides[1])
                                  : i*sizeof(std::int32_t);
        std::memcpy(physical.data()+offset,&values[from],sizeof(values[from]));
        answer[i]=expected[from];
      }
      ASSERT_TRUE(source.view().write(physical).ok());
      if (!replay) {
        const auto aliased=TensorView::create(output_desc,source.view()).value();
        auto rejected=invocation.submit(queue,input,aliased);
        ASSERT_FALSE(rejected);
        EXPECT_TRUE(rejected.error().IsInvalidArgument());
      }
      auto done=invocation.submit(queue,input,output);
      ASSERT_TRUE(done)<<done.error().message();
      ASSERT_TRUE(done->wait().ok());
      std::array<std::uint32_t,expected.size()> actual{};
      ASSERT_TRUE(destination.view().read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual,answer);
      std::vector<std::byte> unchanged(source.size());
      ASSERT_TRUE(source.view().read(unchanged).ok());
      EXPECT_EQ(unchanged,physical);
    }
  }
  // Freezing the integer->float pair does not define float truncation,
  // out-of-range integer results or NaN-to-integer behavior.
  auto reverse=hyle::compute::prepare(context,Cast{DataType::kInt32},
      TensorDesc::dense(DataType::kFloat32,shape).value());
  ASSERT_FALSE(reverse);
  EXPECT_TRUE(reverse.error().IsUnimplemented());
}

TEST(MatureCast, RejectsOverlappingInputAndOutputStorage) {
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

  auto context = hyle::compute::Context::create(hyle::compute::cpu::device(), hyle::compute::aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  auto prepared =
      hyle::compute::prepare(context, Cast{DataType::kBFloat16}, input_desc)
          .value();
  auto invocation = prepared.create_invocation().value();
  auto submitted = invocation.submit(queue, input, output);
  ASSERT_FALSE(submitted.has_value());
  EXPECT_TRUE(submitted.error().IsInvalidArgument());
}


template<class Bits> void expect_boolean_cast(DataType type,
    const std::array<Bits, 8> &values, Bits one) {
  constexpr std::array<std::int64_t, 2> shape{2, 4};
  constexpr std::array<std::uint8_t, 8> truth{0, 0, 1, 1, 1, 1, 1, 1};
  constexpr std::array<std::uint8_t, 8> boolean{0, 1, 2, 255, 1, 0, 0, 1};
  auto context = hyle::compute::test::native_context().value();
  auto queue = context.create_queue().value();
  for (const bool strided : {false, true}) {
    SCOPED_TRACE(strided);
    const auto width = static_cast<std::int64_t>(sizeof(Bits));
    const std::array<std::int64_t, 2> source_strides{4 * (width + 1) + 1, width + 1};
    const auto input_desc = (strided ? TensorDesc::strided_bytes(type, shape, source_strides)
        : TensorDesc::dense(type, shape)).value();
    std::vector<std::byte> physical(input_desc.required_bytes().value(), std::byte{0x5a});
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto offset = strided ? (i / 4) * std::size_t(source_strides[0]) +
          (i % 4) * std::size_t(source_strides[1]) : i * sizeof(Bits);
      std::memcpy(physical.data() + offset, &values[i], sizeof(Bits));
    }
    auto source = Buffer::create(hyle::mem::system_resource(), physical.size(), 64).value();
    auto destination = Buffer::create(hyle::mem::system_resource(), truth.size(), 64).value();
    ASSERT_TRUE(source.view().write(physical).ok());
    auto prepared = hyle::compute::prepare(context, Cast{DataType::kBool}, input_desc);
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto invocation = prepared->create_invocation().value();
    auto done = invocation.submit(queue, ConstTensorView::create(input_desc, source.view()).value(),
        TensorView::create(prepared->output_description().value(), destination.view()).value());
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::array<std::uint8_t, 8> actual{};
    ASSERT_TRUE(destination.view().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, truth);

    const std::array<std::int64_t, 2> bool_strides{11, 2};
    const auto bool_desc = (strided ? TensorDesc::strided_bytes(DataType::kBool, shape, bool_strides)
        : TensorDesc::dense(DataType::kBool, shape)).value();
    std::vector<std::byte> raw(bool_desc.required_bytes().value(), std::byte{0x5a});
    for (std::size_t i = 0; i < boolean.size(); ++i)
      raw[strided ? (i / 4) * 11 + (i % 4) * 2 : i] = std::byte{boolean[i]};
    auto mask = Buffer::create(hyle::mem::system_resource(), raw.size(), 64).value();
    auto numeric = Buffer::create(hyle::mem::system_resource(), 8 * sizeof(Bits), 64).value();
    ASSERT_TRUE(mask.view().write(raw).ok());
    auto reverse = hyle::compute::prepare(context, Cast{type}, bool_desc);
    ASSERT_TRUE(reverse) << reverse.error().message();
    auto reverse_invocation = reverse->create_invocation().value();
    auto reverse_done = reverse_invocation.submit(queue, ConstTensorView::create(bool_desc, mask.view()).value(),
        TensorView::create(reverse->output_description().value(), numeric.view()).value());
    ASSERT_TRUE(reverse_done) << reverse_done.error().message();
    ASSERT_TRUE(reverse_done->wait().ok());
    std::array<Bits, 8> widened{}, expected{};
    for (std::size_t i = 0; i < boolean.size(); ++i) expected[i] = boolean[i] ? one : Bits{0};
    ASSERT_TRUE(numeric.view().read(std::as_writable_bytes(std::span(widened))).ok());
    EXPECT_EQ(widened, expected);
  }
}

TEST(NativeCast, BooleanConversionsKeepSignedZerosTinyValuesAndByteStrides) {
  expect_boolean_cast<std::uint32_t>(DataType::kFloat32,
      {0, 0x80000000u, 1, 0x80000001u, 0x3f800000u, 0xbf800000u, 0x7f800000u, 0x7fc00001u}, 0x3f800000u);
  expect_boolean_cast<std::uint16_t>(DataType::kFloat16,
      {0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7c00, 0x7e01}, 0x3c00);
  expect_boolean_cast<std::uint16_t>(DataType::kBFloat16,
      {0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80, 0x7f80, 0x7fc1}, 0x3f80);
}

} // namespace
