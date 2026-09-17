#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"

namespace {
using namespace hyle::compute;
namespace mem = hyle::mem;

struct Tensor final {
  mem::Buffer storage;
  TensorView view;
  Tensor(const TensorDesc &desc, mem::MemoryResourceRef resource)
      : storage(
            mem::Buffer::create(resource, desc.required_bytes().value(), 256)
                .value()),
        view(TensorView::create(desc, storage.view()).value()) {}
  template <class T> hyle::Status write(std::span<const T> values) {
    const auto &desc = view.description();
    std::vector<std::byte> bytes(storage.size(), std::byte{0x5a});
    for (std::size_t index = 0; index < values.size(); ++index) {
      std::size_t remaining = index, offset = 0;
      for (std::size_t axis = desc.rank(); axis-- > 0;) {
        offset +=
            (remaining % desc.dimensions()[axis]) * desc.byte_strides()[axis];
        remaining /= desc.dimensions()[axis];
      }
      std::memcpy(bytes.data() + offset, &values[index], sizeof(T));
    }
    return storage.view().write(bytes);
  }
  template <class T> std::vector<T> read() const {
    return half_test::read<T>(storage);
  }
};

struct Fixture final {
  Context owner =
      Context::create(cuda::device(0).value(), cuda_native::provider()).value();
  cuda::MemoryResource resource =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  Queue queue = owner.create_queue().value();

  hyle::Status execute(Operation operation,
                       std::span<const ConstTensorView> inputs,
                       TensorView output) {
    auto prepared = prepare(owner, operation, inputs, std::array{output});
    if (!prepared)
      return prepared.error();
    EXPECT_EQ(prepared->workspace_requirement().size, 0U);
    EXPECT_EQ(prepared->invocation_private_bytes(), 0U);
    EXPECT_TRUE(prepared->capture_compatible());
    auto slot = prepared->create_invocation();
    if (!slot)
      return slot.error();
    auto done = slot->submit(queue, inputs, std::array{output});
    return done ? done->wait() : done.error();
  }
};

template <class T> void expect_value(T actual, double expected) {
  if constexpr (std::is_same_v<T, float>) {
    const float rounded = static_cast<float>(expected);
    if (std::isnan(rounded))
      EXPECT_TRUE(std::isnan(actual));
    else if (std::isinf(rounded))
      EXPECT_EQ(actual, rounded);
    else
      EXPECT_NEAR(actual, rounded, std::abs(rounded) * 4e-6 + 2e-7);
  } else {
    half_test::expect_near(actual, expected);
  }
}

double reference(UnaryKind kind, double x) {
  switch (kind) {
  case UnaryKind::kAbs:
    return std::abs(x);
  case UnaryKind::kNeg:
    return -x;
  case UnaryKind::kExp:
    return std::exp(x);
  case UnaryKind::kLog:
    return std::log(x);
  case UnaryKind::kSqrt:
    return std::sqrt(x);
  case UnaryKind::kRsqrt:
    return 1 / std::sqrt(x);
  case UnaryKind::kTanh:
    return std::tanh(x);
  case UnaryKind::kSigmoid:
    return 1 / (1 + std::exp(-x));
  case UnaryKind::kRelu:
    return std::isnan(x) ? x : (x > 0 ? x : 0.0);
  case UnaryKind::kGelu:
    return x == -INFINITY ? 0 : 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
  case UnaryKind::kGeluTanh:
    return x == -INFINITY
               ? 0
               : 0.5 * x *
                     (1 + std::tanh(std::sqrt(2 / 3.141592653589793) *
                                    (x + 0.044715 * x * x * x)));
  case UnaryKind::kSilu:
    return x == -INFINITY ? 0 : x / (1 + std::exp(-x));
  case UnaryKind::kSoftplus:
    if (std::isnan(x))
      return x;
    return x > 100 ? x : std::log1p(std::exp(x));
  }
  return NAN;
}

template <class T> void unary_cases() {
  Fixture fixture;
  constexpr auto type = half_test::data_type<T>();
  for (const std::int64_t count : {0, 1, 127, 129, 2048, 6144}) {
    for (const bool strided : {false, true}) {
      const std::array shape{count};
      const auto output_desc = TensorDesc::dense(type, shape).value();
      const auto input_desc =
          strided ? TensorDesc::strided_bytes(
                        type, shape,
                        std::array{static_cast<std::int64_t>(sizeof(T) + 1)})
                        .value()
                  : output_desc;
      Tensor input(input_desc, fixture.resource.ref()),
          output(output_desc, fixture.resource.ref());
      std::vector<T> values(static_cast<std::size_t>(count));
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] =
            half_test::encode<T>((static_cast<int>(i % 97) - 48) * 0.125);
      const std::array edges{-INFINITY, INFINITY, NAN,   -0.0F,
                             0.0F,      -80.0F,   80.0F, 1e-30F};
      for (std::size_t i = 0; i < std::min(values.size(), edges.size()); ++i)
        values[i] = half_test::encode<T>(edges[i]);
      ASSERT_TRUE(input.write<T>(values).ok());
      for (int mode = 0; mode <= static_cast<int>(UnaryKind::kSoftplus);
           ++mode) {
        SCOPED_TRACE(::testing::Message()
                     << count << '/' << strided << '/' << mode);
        const auto kind = static_cast<UnaryKind>(mode);
        const auto status =
            fixture.execute(Operation(Unary{kind}),
                            std::array{input.view.as_const()}, output.view);
        ASSERT_TRUE(status.ok()) << status.message();
        const auto actual = output.read<T>();
        for (std::size_t i = 0; i < actual.size(); ++i)
          expect_value(actual[i],
                       reference(kind, half_test::decode(values[i])));
      }
    }
  }
}
TEST(CudaTensor, UnaryFloatAndHalfMathIncludesStridesTailsAndSpecialValues) {
  unary_cases<float>();
  unary_cases<hyle::simd::float16_t>();
  unary_cases<hyle::simd::bfloat16_t>();
}

double reference(BinaryKind kind, double x, double y) {
  switch (kind) {
  case BinaryKind::kAdd:
    return x + y;
  case BinaryKind::kSubtract:
    return x - y;
  case BinaryKind::kMultiply:
    return x * y;
  case BinaryKind::kDivide:
    return x / y;
  case BinaryKind::kMinimum:
    return std::fmin(x, y);
  case BinaryKind::kMaximum:
    return std::fmax(x, y);
  }
  return NAN;
}
template <class T> void binary_cases() {
  Fixture fixture;
  constexpr auto type = half_test::data_type<T>();
  constexpr std::int64_t width = 129;
  const auto output_desc =
      TensorDesc::dense(type, std::array<std::int64_t, 3>{2, 3, width}).value();
  Tensor output(output_desc, fixture.resource.ref());
  for (const bool broadcast : {false, true}) {
    const auto a_desc =
        broadcast ? TensorDesc::strided_bytes(
                        type, std::array<std::int64_t, 3>{2, 1, width},
                        std::array<std::int64_t, 3>{width * (sizeof(T) + 1),
                                                    width * (sizeof(T) + 1),
                                                    sizeof(T) + 1})
                        .value()
                  : output_desc;
    const auto b_desc =
        broadcast
            ? TensorDesc::dense(type, std::array<std::int64_t, 2>{3, 1}).value()
            : output_desc;
    Tensor a(a_desc, fixture.resource.ref()), b(b_desc, fixture.resource.ref());
    std::vector<T> av(a_desc.element_count().value()),
        bv(b_desc.element_count().value());
    for (std::size_t i = 0; i < av.size(); ++i)
      av[i] = half_test::encode<T>((static_cast<int>(i % 19) - 9) * 0.25);
    for (std::size_t i = 0; i < bv.size(); ++i)
      bv[i] = half_test::encode<T>((i % 7 + 1) * 0.5);
    av[0] = half_test::encode<T>(NAN);
    av[1] = half_test::encode<T>(INFINITY);
    ASSERT_TRUE(a.write<T>(av).ok());
    ASSERT_TRUE(b.write<T>(bv).ok());
    for (int mode = 0; mode <= static_cast<int>(BinaryKind::kMaximum); ++mode) {
      const auto kind = static_cast<BinaryKind>(mode);
      const auto status = fixture.execute(
          Operation(Binary{kind}),
          std::array{a.view.as_const(), b.view.as_const()}, output.view);
      ASSERT_TRUE(status.ok()) << status.message();
      const auto actual = output.read<T>();
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto ai = broadcast ? (i / (3 * width)) * width + i % width : i;
        const auto bi = broadcast ? (i / width) % 3 : i;
        expect_value(actual[i], reference(kind, half_test::decode(av[ai]),
                                          half_test::decode(bv[bi])));
      }
    }
  }
  // Rank-zero scalar broadcasting and exact in-place mutation reuse the slot.
  const auto scalar_desc =
      TensorDesc::dense(type, std::span<const std::int64_t>{}).value();
  Tensor scalar(scalar_desc, fixture.resource.ref());
  const std::array<T, 1> one{half_test::encode<T>(1)};
  ASSERT_TRUE(scalar.write<T>(one).ok());
  auto op = prepare(fixture.owner, Operation(Binary{}),
                    std::array{output.view.as_const(), scalar.view.as_const()},
                    std::array{output.view});
  ASSERT_TRUE(op.has_value()) << op.error().message();
  EXPECT_TRUE(op->alias_support(0, 0)->supported);
  EXPECT_FALSE(op->alias_support(0, 1)->supported);
  auto slot = op->create_invocation().value();
  std::vector<T> seed(output_desc.element_count().value(),
                      half_test::encode<T>(2));
  ASSERT_TRUE(output.write<T>(seed).ok());
  for (int repeat = 0; repeat < 3; ++repeat) {
    auto done = slot.submit(fixture.queue, output.view.as_const(),
                            scalar.view.as_const(), output.view);
    ASSERT_TRUE(done.has_value());
    ASSERT_TRUE(done->wait().ok());
  }
  for (const auto value : output.read<T>())
    expect_value(value, 5);
}
TEST(CudaTensor, BinaryBroadcastsStridedInputsAndAllowsOnlySafeExactAliases) {
  binary_cases<float>();
  binary_cases<hyle::simd::float16_t>();
  binary_cases<hyle::simd::bfloat16_t>();
}

template <class From, class To> void cast_case(Fixture &fixture) {
  const std::array<std::int64_t, 2> shape{2, 129};
  const auto in_desc =
      TensorDesc::strided_bytes(half_test::data_type<From>(), shape,
                                std::array<std::int64_t, 2>{
                                    129 * (sizeof(From) + 1), sizeof(From) + 1})
          .value();
  const auto out_desc =
      TensorDesc::dense(half_test::data_type<To>(), shape).value();
  Tensor input(in_desc, fixture.resource.ref()),
      output(out_desc, fixture.resource.ref());
  std::vector<From> values(258);
  const std::array<double, 14> edges{
      0,     -0.0,       NAN,           INFINITY,   -INFINITY,
      1e-40, -1e-40,     0x1p-24,       0x1p-14,    65504,
      65520, 1.00390625, 1.00048828125, -1.00390625};
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = half_test::encode<From>(
        i < edges.size() ? edges[i] : (static_cast<int>(i) - 99) * 0.037);
  ASSERT_TRUE(input.write<From>(values).ok());
  const auto status =
      fixture.execute(Operation(Cast{half_test::data_type<To>()}),
                      std::array{input.view.as_const()}, output.view);
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = output.read<To>();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto expected = half_test::encode<To>(half_test::decode(values[i]));
    if (std::isnan(half_test::decode(expected)))
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i])));
    else
      EXPECT_EQ(half_test::decode(actual[i]), half_test::decode(expected));
  }
}
TEST(CudaTensor, FloatStorageConversionsMatchIndependentRounding) {
  Fixture fixture;
  using H = hyle::simd::float16_t;
  using B = hyle::simd::bfloat16_t;
  cast_case<float, float>(fixture);
  cast_case<float, H>(fixture);
  cast_case<float, B>(fixture);
  cast_case<H, float>(fixture);
  cast_case<H, H>(fixture);
  cast_case<H, B>(fixture);
  cast_case<B, float>(fixture);
  cast_case<B, H>(fixture);
  cast_case<B, B>(fixture);
}

TEST(CudaTensor, SelectCopyFillPreserveBitsWithBroadcastMaskAndOddStride) {
  Fixture fixture;
  for (const auto type :
       {DataType::kBool, DataType::kFloat16, DataType::kBFloat16,
        DataType::kFloat32, DataType::kInt32, DataType::kInt64}) {
    const auto width = data_type_size(type);
    const auto dense =
        TensorDesc::dense(type, std::array<std::int64_t, 2>{2, 129}).value();
    const auto strided = TensorDesc::strided_bytes(
                             type, dense.dimensions(),
                             std::array<std::int64_t, 2>{
                                 static_cast<std::int64_t>(129 * (width + 1)),
                                 static_cast<std::int64_t>(width + 1)})
                             .value();
    Tensor input(strided, fixture.resource.ref()),
        output(dense, fixture.resource.ref());
    const auto scalar =
        TensorDesc::dense(type, std::span<const std::int64_t>{}).value();
    Tensor other(scalar, fixture.resource.ref());
    std::vector<std::byte> bytes(input.storage.size(), std::byte{0x55});
    const std::uint64_t bits =
        type == DataType::kBool ? 1 : UINT64_C(0x7f80000180007fa5);
    for (std::size_t i = 0; i < 258; ++i)
      std::memcpy(bytes.data() + i * (width + 1), &bits, width);
    ASSERT_TRUE(input.storage.view().write(bytes).ok());
    ASSERT_TRUE(fixture
                    .execute(Operation(Copy{}),
                             std::array{input.view.as_const()}, output.view)
                    .ok());
    const auto copied = output.read<std::byte>();
    for (std::size_t i = 0; i < 258; ++i)
      EXPECT_EQ(std::memcmp(copied.data() + i * width, &bits, width), 0);
    const Fill fill{TensorShape::create({}).value(), type,
                    ScalarValue::from_bits(type, bits)};
    ASSERT_TRUE(fixture.execute(Operation(fill), {}, other.view).ok());
    Tensor mask(
        TensorDesc::dense(DataType::kBool, std::array<std::int64_t, 2>{2, 1})
            .value(),
        fixture.resource.ref());
    const std::array<std::uint8_t, 2> masks{1, 0};
    ASSERT_TRUE(mask.write<std::uint8_t>(masks).ok());
    std::fill(bytes.begin(), bytes.end(), std::byte{0});
    ASSERT_TRUE(input.storage.view().write(bytes).ok());
    ASSERT_TRUE(
        fixture
            .execute(Operation(Select{}),
                     std::array{mask.view.as_const(), input.view.as_const(),
                                other.view.as_const()},
                     output.view)
            .ok());
    const auto actual = output.read<std::byte>();
    const std::uint64_t zero = 0;
    for (std::size_t i = 0; i < 258; ++i)
      EXPECT_EQ(std::memcmp(actual.data() + i * width, i < 129 ? &zero : &bits,
                            width),
                0);
  }
}

TEST(CudaTensor, ViewUsesRequiredStorageOffsetAndCopyRejectsOverlap) {
  Fixture fixture;
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 2>{3, 7})
          .value();
  Tensor input(desc, fixture.resource.ref());
  const auto slice = View::slice(1, 1, 2).value();
  auto prepared = prepare(
      fixture.owner, Operation(slice), std::array{desc},
      std::array{TensorDesc::strided_bytes(DataType::kInt32,
                                           std::array<std::int64_t, 2>{3, 2},
                                           std::array<std::int64_t, 2>{28, 4})
                     .value()},
      fixture.resource.ref());
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(prepared->alias_support(0, 0)->byte_offset, 4U);
  const auto sliced_desc = prepared->output_description().value();
  const auto output =
      TensorView::create(sliced_desc,
                         input.storage.view()
                             .subview(4, sliced_desc.required_bytes().value())
                             .value())
          .value();
  auto slot = prepared->create_invocation().value();
  auto done = slot.submit(fixture.queue, input.view.as_const(), output);
  ASSERT_TRUE(done.has_value());
  EXPECT_TRUE(done->wait().ok());
  auto wrong = TensorView::create(sliced_desc, input.storage.view()).value();
  auto invalid = slot.submit(fixture.queue, input.view.as_const(), wrong);
  EXPECT_FALSE(invalid.has_value());
  auto copy =
      prepare(fixture.owner, Copy{}, desc, fixture.resource.ref()).value();
  auto copy_slot = copy.create_invocation().value();
  auto overlap =
      copy_slot.submit(fixture.queue, input.view.as_const(), input.view);
  EXPECT_FALSE(overlap.has_value());
  const auto f64 =
      TensorDesc::dense(DataType::kFloat64, std::array<std::int64_t, 1>{3})
          .value();
  EXPECT_FALSE(
      prepare(fixture.owner, Unary{}, f64, fixture.resource.ref()).has_value());
  ASSERT_TRUE(fixture.owner.close().ok());
  EXPECT_FALSE(
      copy_slot.submit(fixture.queue, input.view.as_const(), input.view)
          .has_value());
}
} // namespace
