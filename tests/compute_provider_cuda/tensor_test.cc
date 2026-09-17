#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
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
    if (!done)
      return done.error();
    return done->wait();
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
template <class T, class Weight, class Bias>
void normalization_case(Fixture &fixture, std::int64_t width, bool layer,
                        unsigned parameters, float epsilon, bool inplace) {
  const auto description =
      TensorDesc::dense(half_test::data_type<T>(),
                        std::array<std::int64_t, 2>{3, width})
          .value();
  const auto weight_desc =
      TensorDesc::dense(half_test::data_type<Weight>(), std::array{width})
          .value();
  const auto bias_desc =
      TensorDesc::dense(half_test::data_type<Bias>(), std::array{width})
          .value();
  Tensor input(description, fixture.resource.ref()),
      output(description, fixture.resource.ref());
  Tensor scale(weight_desc, fixture.resource.ref()),
      bias(bias_desc, fixture.resource.ref());
  std::vector<T> values(static_cast<std::size_t>(3 * width));
  std::vector<Weight> weights(static_cast<std::size_t>(width));
  std::vector<Bias> biases(static_cast<std::size_t>(width));
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = i < static_cast<std::size_t>(width)
                             ? 0.75
                             : (static_cast<int>(i % 37) - 18) * 0.1875;
    values[i] = half_test::encode<T>(value);
  }
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = half_test::encode<Weight>(1 + (i % 17) * 0.0031);
    biases[i] = half_test::encode<Bias>((static_cast<int>(i % 7) - 3) * 0.0159);
  }
  ASSERT_TRUE(input.write<T>(values).ok());
  ASSERT_TRUE(scale.write<Weight>(weights).ok());
  ASSERT_TRUE(bias.write<Bias>(biases).ok());
  const std::array all_inputs{input.view.as_const(), scale.view.as_const(),
                              bias.view.as_const()};
  const auto operation =
      layer ? Operation(LayerNorm{epsilon}) : Operation(RmsNorm{epsilon});
  const auto status =
      fixture.execute(operation, std::span(all_inputs).first(1 + parameters),
                      inplace ? input.view : output.view);
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = inplace ? input.read<T>() : output.read<T>();
  if (width == 0) {
    EXPECT_TRUE(actual.empty());
    return;
  }
  for (std::size_t row = 0; row < 3; ++row) {
    double mean = 0;
    if (layer) {
      for (std::int64_t col = 0; col < width; ++col)
        mean += half_test::decode(values[row * width + col]);
      mean /= width;
    }
    double variance = 0;
    for (std::int64_t col = 0; col < width; ++col) {
      const double centered =
          half_test::decode(values[row * width + col]) - mean;
      variance += centered * centered;
    }
    variance /= width;
    const double inverse = 1 / std::sqrt(variance + epsilon);
    for (std::int64_t col = 0; col < width; ++col) {
      double expected =
          (half_test::decode(values[row * width + col]) - mean) * inverse;
      if (parameters > 0)
        expected *= half_test::decode(weights[col]);
      if (parameters > 1)
        expected += half_test::decode(biases[col]);
      expect_value(actual[row * width + col], expected);
    }
  }
}

template <class T> void normalization_cases() {
  Fixture fixture;
  for (const std::int64_t width :
       {0, 1, 127, 128, 129, 2048, 5120, 6144, 7168}) {
    SCOPED_TRACE(width);
    for (const bool inplace : {false, true}) {
      normalization_case<T, T, T>(fixture, width, false, 0, 1e-5F, inplace);
      normalization_case<T, T, T>(fixture, width, false, 1, 1e-5F, inplace);
      normalization_case<T, float, T>(fixture, width, false, 1, 1e-5F, inplace);
      normalization_case<T, T, T>(fixture, width, true, 0, 1e-5F, inplace);
      normalization_case<T, T, T>(fixture, width, true, 1, 1e-5F, inplace);
      normalization_case<T, float, T>(fixture, width, true, 1, 1e-5F, inplace);
      normalization_case<T, T, T>(fixture, width, true, 2, 1e-5F, inplace);
      normalization_case<T, float, T>(fixture, width, true, 2, 1e-5F, inplace);
      normalization_case<T, T, float>(fixture, width, true, 2, 1e-5F, inplace);
      normalization_case<T, float, float>(fixture, width, true, 2, 1e-5F,
                                          inplace);
    }
  }
  normalization_case<T, T, T>(fixture, 129, false, 0, 0, false);
  normalization_case<T, T, T>(fixture, 129, true, 0, 0, false);
}
TEST(CudaTensor,
     NormalizationUsesFloatAccumulationAndIndependentParameterTypes) {
  normalization_cases<float>();
  normalization_cases<hyle::simd::float16_t>();
  normalization_cases<hyle::simd::bfloat16_t>();
}

template <class T> void precise_parameters() {
  Fixture fixture;
  const auto description = TensorDesc::dense(half_test::data_type<T>(),
                                             std::array<std::int64_t, 1>{2})
                               .value();
  const auto parameter_desc =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{2})
          .value();
  Tensor input(description, fixture.resource.ref()),
      output(description, fixture.resource.ref());
  Tensor weight(parameter_desc, fixture.resource.ref()),
      bias(parameter_desc, fixture.resource.ref());
  const std::array<T, 2> values{half_test::encode<T>(-1),
                                half_test::encode<T>(1)};
  const std::array<float, 2> scale{1.0001F, 1.0001F}, offset{1.0F, -1.0F};
  ASSERT_TRUE(input.write<T>(values).ok());
  ASSERT_TRUE(weight.write<float>(scale).ok());
  ASSERT_TRUE(bias.write<float>(offset).ok());
  ASSERT_TRUE(
      fixture
          .execute(Operation(LayerNorm{.epsilon = 0}),
                   std::array{input.view.as_const(), weight.view.as_const(),
                              bias.view.as_const()},
                   output.view)
          .ok());
  const auto actual = output.read<T>();
  EXPECT_EQ(half_test::decode(actual[0]),
            half_test::decode(half_test::encode<T>(1.0F - 1.0001F)));
  EXPECT_EQ(half_test::decode(actual[1]),
            half_test::decode(half_test::encode<T>(1.0001F - 1.0F)));
  EXPECT_NE(half_test::decode(actual[1]), 0.0);
}
TEST(CudaTensor, FloatNormParametersAreNotNarrowedBeforeAffineArithmetic) {
  precise_parameters<hyle::simd::float16_t>();
  precise_parameters<hyle::simd::bfloat16_t>();
}

TEST(CudaTensor, NormalizationHandlesSpecialValuesAndStableVariance) {
  Fixture fixture;
  const auto description =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{4, 128})
          .value();
  Tensor input(description, fixture.resource.ref()),
      output(description, fixture.resource.ref());
  std::vector<float> values(512, 0);
  values[128] = NAN;
  values[256] = INFINITY;
  for (std::size_t i = 384; i < values.size(); ++i)
    values[i] = 10000 + (i % 2 == 0 ? -0.25F : 0.25F);
  ASSERT_TRUE(input.write<float>(values).ok());
  for (const bool layer : {false, true}) {
    const auto op = layer ? Operation(LayerNorm{.epsilon = 0})
                          : Operation(RmsNorm{.epsilon = 0});
    ASSERT_TRUE(
        fixture.execute(op, std::array{input.view.as_const()}, output.view)
            .ok());
    const auto actual = output.read<float>();
    for (std::size_t i = 0; i < 256; ++i)
      EXPECT_TRUE(std::isnan(actual[i]));
    for (std::size_t i = 256; i < 384; ++i) {
      if (layer || i == 256)
        EXPECT_TRUE(std::isnan(actual[i]));
      else
        EXPECT_EQ(actual[i], 0);
    }
    if (layer)
      for (std::size_t i = 384; i < actual.size(); ++i)
        EXPECT_FLOAT_EQ(actual[i], i % 2 == 0 ? -1.0F : 1.0F);
  }
  auto unsupported =
      prepare(fixture.owner, RmsNorm{.accumulation_type = DataType::kFloat64},
              description, fixture.resource.ref());
  ASSERT_FALSE(unsupported.has_value());
  EXPECT_TRUE(unsupported.error().IsUnimplemented());
  const auto strided = TensorDesc::strided_bytes(
                           DataType::kFloat32, std::array<std::int64_t, 1>{128},
                           std::array<std::int64_t, 1>{8})
                           .value();
  auto noncontiguous =
      prepare(fixture.owner, RmsNorm{}, strided, fixture.resource.ref());
  ASSERT_FALSE(noncontiguous.has_value());
  EXPECT_TRUE(noncontiguous.error().IsUnimplemented());
  auto prepared =
      prepare(fixture.owner, RmsNorm{}, description, fixture.resource.ref())
          .value();
  auto invocation = prepared.create_invocation().value();
  auto partial = TensorView::create(description, input.storage.view()).value();
  // A larger backing range allows a genuinely overlapping, non-equal binding.
  auto backing =
      mem::Buffer::create(fixture.resource.ref(), input.storage.size() + 4, 256)
          .value();
  const auto source =
      ConstTensorView::create(
          description, backing.view().subview(0, input.storage.size()).value())
          .value();
  partial =
      TensorView::create(
          description, backing.view().subview(4, input.storage.size()).value())
          .value();
  auto overlap = invocation.submit(fixture.queue, source, partial);
  ASSERT_FALSE(overlap.has_value());
  EXPECT_TRUE(overlap.error().IsInvalidArgument());
}
TensorDesc reduction_desc(DataType type, std::span<const std::int64_t> shape,
                          bool odd_stride) {
  if (!odd_stride)
    return TensorDesc::dense(type, shape).value();
  std::vector<std::int64_t> strides(shape.size());
  auto stride = static_cast<std::int64_t>(data_type_size(type) + 1);
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    strides[axis] = stride;
    stride *= std::max<std::int64_t>(shape[axis], 1);
  }
  return TensorDesc::strided_bytes(type, shape, strides).value();
}

// The reference enumerates input coordinates and groups their values by the
// surviving output coordinates; it does not use the device's row/column map.
template <class T>
void check_reduce(Fixture &fixture, const TensorDesc &description,
                  std::span<const T> values, const Axes &axes, bool keep,
                  ReductionKind kind) {
  const Reduce operation{.kind = kind,
                         .axes = axes,
                         .keep_dimensions = keep,
                         .accumulation_type = DataType::kFloat32};
  auto prepared =
      prepare(fixture.owner, operation, description, fixture.resource.ref());
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  const auto output_desc = prepared->output_description().value();
  EXPECT_EQ(prepared->workspace_requirement().size, 0U);
  EXPECT_EQ(prepared->invocation_private_bytes(), 0U);
  EXPECT_TRUE(prepared->capture_compatible());
  Tensor input(description, fixture.resource.ref()),
      output(output_desc, fixture.resource.ref());
  ASSERT_TRUE(input.write<T>(values).ok());
  std::vector<T> sentinel(*output_desc.element_count(),
                          half_test::encode<T>(37));
  ASSERT_TRUE(output.write<T>(sentinel).ok());
  auto slot = prepared->create_invocation().value();
  auto done = slot.submit(fixture.queue, input.view.as_const(), output.view);
  ASSERT_TRUE(done.has_value()) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  std::array<bool, 8> reduced{};
  for (std::size_t axis = 0; axis < description.rank(); ++axis)
    reduced[axis] = axes.is_all();
  for (const auto requested : axes.values())
    reduced[requested < 0 ? requested + static_cast<int>(description.rank())
                          : requested] = true;
  const double identity = kind == ReductionKind::kMinimum   ? INFINITY
                          : kind == ReductionKind::kMaximum ? -INFINITY
                                                            : 0;
  std::vector<double> expected(sentinel.size(), identity);
  std::vector<std::size_t> counts(expected.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::array<std::size_t, 8> coordinate{};
    std::size_t remaining = index;
    for (std::size_t axis = description.rank(); axis-- > 0;) {
      coordinate[axis] = remaining % description.dimensions()[axis];
      remaining /= description.dimensions()[axis];
    }
    std::size_t output_index = 0;
    for (std::size_t axis = 0; axis < description.rank(); ++axis)
      if (!reduced[axis])
        output_index =
            output_index * description.dimensions()[axis] + coordinate[axis];
    const auto value = half_test::decode(values[index]);
    auto &result = expected[output_index];
    if (std::isnan(result) || std::isnan(value))
      result = NAN;
    else if (kind == ReductionKind::kMinimum)
      result = std::fmin(result, value);
    else if (kind == ReductionKind::kMaximum)
      result = std::fmax(result, value);
    else
      result += value;
    ++counts[output_index];
  }
  const auto actual = output.read<T>();
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (kind == ReductionKind::kMean)
      expected[index] =
          counts[index] == 0 ? NAN : expected[index] / counts[index];
    expect_value(actual[index], expected[index]);
  }
}

template <class T> void reduction_cases() {
  Fixture fixture;
  const std::vector<std::vector<std::int64_t>> shapes{{},
                                                      {0},
                                                      {1},
                                                      {129},
                                                      {2048},
                                                      {8193},
                                                      {2, 0, 3},
                                                      {0, 7},
                                                      {2, 3, 5},
                                                      {2, 3, 129},
                                                      {2, 2, 2, 1, 3, 1, 2, 5}};
  for (const auto &shape : shapes) {
    for (const bool strided : {false, true}) {
      const auto desc =
          reduction_desc(half_test::data_type<T>(), shape, strided);
      std::vector<T> values(desc.element_count().value());
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = half_test::encode<T>(
            (static_cast<int>(i % 29) - 14) * 0.125 + 0.0625);
      std::vector<Axes> choices{Axes::all()};
      if (!shape.empty())
        choices.push_back(Axes::from(std::array<std::int32_t, 1>{-1}).value());
      if (shape.size() > 1)
        choices.push_back(
            Axes::from(std::array<std::int32_t, 2>{0, -1}).value());
      if (shape.size() > 2)
        choices.push_back(Axes::from(std::array<std::int32_t, 1>{1}).value());
      for (const auto &axes : choices) {
        for (const bool keep : {false, true}) {
          for (const auto kind :
               {ReductionKind::kSum, ReductionKind::kMean,
                ReductionKind::kMinimum, ReductionKind::kMaximum}) {
            SCOPED_TRACE(::testing::Message()
                         << "rank=" << shape.size()
                         << " count=" << values.size() << " strided=" << strided
                         << " axes-all=" << axes.is_all() << " keep=" << keep
                         << " kind=" << static_cast<int>(kind));
            check_reduce<T>(fixture, desc, values, axes, keep, kind);
          }
        }
      }
    }
  }
}
TEST(CudaReduction, AxesKeepDimensionsStridesEmptyAndHalfAccumulation) {
  reduction_cases<float>();
  reduction_cases<hyle::simd::float16_t>();
  reduction_cases<hyle::simd::bfloat16_t>();
}

TEST(CudaReduction, NonFiniteValuesPropagateAcrossWarpAndTileBoundaries) {
  Fixture fixture;
  const auto desc = TensorDesc::dense(DataType::kFloat32,
                                      std::array<std::int64_t, 2>{2, 8193})
                        .value();
  std::vector<float> values(*desc.element_count(), 2);
  for (const auto index : {0, 31, 32, 128, 256, 8192}) {
    for (const float value : {NAN, INFINITY, -INFINITY}) {
      values[index] = value;
      for (const auto kind : {ReductionKind::kSum, ReductionKind::kMean,
                              ReductionKind::kMinimum, ReductionKind::kMaximum})
        check_reduce<float>(fixture, desc, values,
                            Axes::from(std::array<std::int32_t, 1>{-1}).value(),
                            false, kind);
      values[index] = 2;
    }
  }
  // Broadcast storage is read once per logical element, not once per address.
  const auto broadcast =
      TensorDesc::strided_bytes(DataType::kFloat32,
                                std::array<std::int64_t, 2>{3, 129},
                                std::array<std::int64_t, 2>{0, 4})
          .value();
  values.assign(*broadcast.element_count(), 1.25F);
  check_reduce<float>(fixture, broadcast, values, Axes::all(), false,
                      ReductionKind::kSum);
}

std::size_t broadcast_index(std::size_t logical,
                            std::span<const std::int64_t> shape,
                            std::span<const std::int64_t> mask) {
  std::size_t result = 0, step = 1;
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    const auto coordinate = logical % shape[axis];
    logical /= shape[axis];
    if (mask[axis] != 1)
      result += coordinate * step;
    step *= mask[axis];
  }
  return result;
}

template <class T>
void check_softmax(Fixture &fixture, std::span<const std::int64_t> shape,
                   int axis, bool strided, bool inplace, SoftmaxKind kind,
                   MaskKind mask_kind, bool scalar_mask, bool special) {
  const auto desc = reduction_desc(half_test::data_type<T>(), shape, strided);
  const auto output_desc =
      TensorDesc::dense(half_test::data_type<T>(), shape).value();
  Tensor input(desc, fixture.resource.ref()),
      output(output_desc, fixture.resource.ref());
  std::vector<T> values(*desc.element_count());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = half_test::encode<T>((static_cast<int>(i % 67) - 33) * 0.3125);
  std::vector<std::int64_t> mask_shape(shape.begin(), shape.end());
  if (scalar_mask)
    std::fill(mask_shape.begin(), mask_shape.end(), 1);
  else
    mask_shape[0] = 1;
  const auto mask_desc = reduction_desc(mask_kind == MaskKind::kBoolean
                                            ? DataType::kBool
                                            : half_test::data_type<T>(),
                                        mask_shape, strided);
  Tensor mask(mask_desc, fixture.resource.ref());
  std::vector<std::uint8_t> boolean(*mask_desc.element_count());
  std::vector<T> additive(boolean.size());
  for (std::size_t i = 0; i < boolean.size(); ++i) {
    boolean[i] = i % 3 == 0 ? 0 : 1;
    additive[i] = half_test::encode<T>(
        i % 3 == 0 ? -INFINITY : (static_cast<int>(i % 7) - 3) * 0.125);
  }
  if (special && !values.empty()) {
    // One ordinary finite row, one all -Inf row, one unmasked +Inf/NaN row;
    // masked NaNs must be ignored. Cover every normalization classification.
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto col = i % shape.back();
      const auto row = i / shape.back();
      double value = row == 1               ? -INFINITY
                     : row == 2 && col == 1 ? INFINITY
                     : row == 3 && col == 1 ? NAN
                     : col == 0             ? NAN
                     : col % 2 == 0         ? 10000
                                            : -10000;
      values[i] = half_test::encode<T>(value);
    }
  }
  ASSERT_TRUE(input.write<T>(values).ok());
  if (mask_kind == MaskKind::kBoolean)
    ASSERT_TRUE(mask.write<std::uint8_t>(boolean).ok());
  else if (mask_kind == MaskKind::kAdditive)
    ASSERT_TRUE(mask.write<T>(additive).ok());
  const float scale = special ? 1 : -0.75F;
  const Softmax op{
      .axis = axis, .scale = scale, .kind = kind, .mask = mask_kind};
  const std::array inputs{input.view.as_const(), mask.view.as_const()};
  const auto status = fixture.execute(
      Operation(op),
      std::span(inputs).first(mask_kind == MaskKind::kNone ? 1 : 2),
      inplace ? input.view : output.view);
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = inplace ? input.read<T>() : output.read<T>();
  const auto normalized_axis =
      axis < 0 ? axis + static_cast<int>(shape.size()) : axis;
  std::size_t outer = 1, inner = 1;
  for (int dim = 0; dim < normalized_axis; ++dim)
    outer *= shape[dim];
  for (std::size_t dim = normalized_axis + 1; dim < shape.size(); ++dim)
    inner *= shape[dim];
  const auto width = static_cast<std::size_t>(shape[normalized_axis]);
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t part = 0; part < inner; ++part) {
      std::vector<double> reference_values(width);
      double maximum = -INFINITY;
      bool invalid = false;
      for (std::size_t col = 0; col < width; ++col) {
        const auto logical = (row * width + col) * inner + part;
        float value =
            static_cast<float>(half_test::decode(values[logical])) * scale;
        if (mask_kind != MaskKind::kNone) {
          const auto index = broadcast_index(logical, shape, mask_shape);
          if (mask_kind == MaskKind::kBoolean) {
            if (boolean[index] == 0)
              value = -INFINITY;
          } else
            value += static_cast<float>(half_test::decode(additive[index]));
        }
        reference_values[col] = value;
        invalid |= std::isnan(value) || value == INFINITY;
        maximum = std::max(maximum, static_cast<double>(value));
      }
      invalid |= maximum == -INFINITY;
      double sum = 0;
      for (const auto value : reference_values)
        sum += std::exp(value - maximum);
      for (std::size_t col = 0; col < width; ++col) {
        const double expected =
            invalid ? NAN
            : kind == SoftmaxKind::kSoftmax
                ? std::exp(reference_values[col] - maximum) / sum
                : (reference_values[col] - maximum) - std::log(sum);
        expect_value(actual[(row * width + col) * inner + part], expected);
      }
    }
  }
}

template <class T> void softmax_cases() {
  Fixture fixture;
  for (const auto kind : {SoftmaxKind::kSoftmax, SoftmaxKind::kLogSoftmax}) {
    for (const std::int64_t width : {0, 1, 127, 129, 2048, 8192, 8193, 32769}) {
      for (const bool inplace : {false, true}) {
        for (const auto mask :
             {MaskKind::kNone, MaskKind::kBoolean, MaskKind::kAdditive}) {
          SCOPED_TRACE(::testing::Message()
                       << "width=" << width << " inplace=" << inplace
                       << " mask=" << static_cast<int>(mask)
                       << " kind=" << static_cast<int>(kind));
          check_softmax<T>(fixture, std::array<std::int64_t, 2>{2, width}, -1,
                           false, inplace, kind, mask, false, false);
        }
      }
    }
    for (const int axis : {0, 1, -1}) {
      for (const bool strided : {false, true}) {
        for (const auto mask :
             {MaskKind::kNone, MaskKind::kBoolean, MaskKind::kAdditive}) {
          SCOPED_TRACE(::testing::Message()
                       << "axis=" << axis << " strided=" << strided
                       << " mask=" << static_cast<int>(mask));
          check_softmax<T>(fixture, std::array<std::int64_t, 3>{2, 129, 3},
                           axis, strided, !strided, kind, mask, false, false);
          check_softmax<T>(fixture, std::array<std::int64_t, 3>{2, 129, 3},
                           axis, strided, !strided, kind, mask, true, false);
        }
      }
    }
    for (const std::int64_t width : {129, 8193})
      for (const auto mask :
           {MaskKind::kNone, MaskKind::kBoolean, MaskKind::kAdditive})
        check_softmax<T>(fixture, std::array<std::int64_t, 2>{4, width}, -1,
                         false, true, kind, mask, false, true);
  }
}
TEST(CudaReduction, SoftmaxFloatAndHalfMasksAxesCacheBoundaryAndLongRows) {
  softmax_cases<float>();
  softmax_cases<hyle::simd::float16_t>();
  softmax_cases<hyle::simd::bfloat16_t>();
}

TEST(CudaReduction, PreparationRejectsPrecisionAndUnsafeAliasesWithoutWriting) {
  Fixture fixture;
  const auto empty_axes = Axes::from(std::span<const std::int32_t>{});
  ASSERT_FALSE(empty_axes.has_value());
  EXPECT_TRUE(empty_axes.error().IsInvalidArgument());
  const auto half = TensorDesc::dense(DataType::kBFloat16,
                                      std::array<std::int64_t, 2>{2, 129})
                        .value();
  auto implicit_half =
      prepare(fixture.owner, Reduce{}, half, fixture.resource.ref());
  ASSERT_FALSE(implicit_half.has_value());
  EXPECT_TRUE(implicit_half.error().IsUnimplemented());
  for (const bool softmax : {false, true}) {
    auto rejected =
        softmax ? prepare(fixture.owner,
                          Softmax{.accumulation_type = DataType::kFloat64},
                          half, fixture.resource.ref())
                : prepare(fixture.owner,
                          Reduce{.accumulation_type = DataType::kFloat64}, half,
                          fixture.resource.ref());
    ASSERT_FALSE(rejected.has_value());
    EXPECT_TRUE(rejected.error().IsUnimplemented());
  }
  const auto desc =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{129})
          .value();
  Tensor input(desc, fixture.resource.ref());
  std::vector<float> values(129, 1);
  ASSERT_TRUE(input.write<float>(values).ok());
  auto reduction =
      prepare(fixture.owner, Reduce{}, desc, fixture.resource.ref()).value();
  auto reduced_output =
      TensorView::create(reduction.output_description().value(),
                         input.storage.view().subview(0, sizeof(float)).value())
          .value();
  auto reduce_slot = reduction.create_invocation().value();
  auto alias =
      reduce_slot.submit(fixture.queue, input.view.as_const(), reduced_output);
  ASSERT_FALSE(alias.has_value());
  EXPECT_TRUE(alias.error().IsInvalidArgument());
  EXPECT_EQ(input.read<float>(), values);
  auto backing =
      mem::Buffer::create(fixture.resource.ref(), 130 * sizeof(float), 256)
          .value();
  auto source =
      ConstTensorView::create(
          desc, backing.view().subview(0, 129 * sizeof(float)).value())
          .value();
  auto destination =
      TensorView::create(
          desc,
          backing.view().subview(sizeof(float), 129 * sizeof(float)).value())
          .value();
  auto softmax =
      prepare(fixture.owner, Softmax{}, desc, fixture.resource.ref()).value();
  auto softmax_slot = softmax.create_invocation().value();
  auto partial = softmax_slot.submit(fixture.queue, source, destination);
  ASSERT_FALSE(partial.has_value());
  EXPECT_TRUE(partial.error().IsInvalidArgument());
}

template <class T> void log_softmax_tail_cases() {
  Fixture fixture;
  for (const std::int64_t width : {129, 8193}) {
    const auto desc = TensorDesc::dense(half_test::data_type<T>(),
                                        std::array<std::int64_t, 2>{3, width})
                          .value();
    Tensor input(desc, fixture.resource.ref());
    std::vector<T> values(*desc.element_count(), half_test::encode<T>(-20));
    values[static_cast<std::size_t>(width) - 1] = half_test::encode<T>(0);
    values[static_cast<std::size_t>(width)] = half_test::encode<T>(0);
    values[static_cast<std::size_t>(width * 2) - 1] = half_test::encode<T>(0);
    for (std::size_t i = static_cast<std::size_t>(width * 2); i < values.size();
         ++i)
      values[i] = half_test::encode<T>(0);
    ASSERT_TRUE(input.write<T>(values).ok());
    const auto status =
        fixture.execute(Operation(Softmax{.kind = SoftmaxKind::kLogSoftmax}),
                        std::array{input.view.as_const()}, input.view);
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = input.read<T>();
    for (std::size_t row = 0; row < 3; ++row) {
      const auto maxima = row == 2 ? static_cast<std::size_t>(width) : row + 1;
      const double log_partition =
          std::log(static_cast<double>(maxima)) +
          std::log1p((width - maxima) * std::exp(-20.0) / maxima);
      for (std::size_t col = 0; col < static_cast<std::size_t>(width); ++col) {
        const auto index = row * width + col;
        expect_value(actual[index],
                     half_test::decode(values[index]) - log_partition);
      }
    }
  }
}
TEST(CudaReduction, LogSoftmaxPreservesNearOneTailsAndCountsEqualMaxima) {
  log_softmax_tail_cases<float>();
  log_softmax_tail_cases<hyle::simd::float16_t>();
  log_softmax_tail_cases<hyle::simd::bfloat16_t>();
}

TEST(CudaReduction, EmptyStorageWithHugeDimensionStillWritesIdentities) {
  Fixture fixture;
  const auto desc = TensorDesc::dense(
      DataType::kFloat32, std::array<std::int64_t, 2>{INT64_MAX, 0});
  ASSERT_TRUE(desc.has_value()) << desc.error().message();
  for (const auto kind : {ReductionKind::kSum, ReductionKind::kMean,
                          ReductionKind::kMinimum, ReductionKind::kMaximum})
    check_reduce<float>(fixture, *desc, {}, Axes::all(), false, kind);
}
template <class T>
void check_topk(Fixture &fixture, const TensorDesc &description,
                std::span<const T> data, TopK operation) {
  auto prepared =
      prepare(fixture.owner, operation, description, fixture.resource.ref());
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_EQ(prepared->workspace_requirement().size, 0U);
  EXPECT_EQ(prepared->invocation_private_bytes(), 0U);
  EXPECT_TRUE(prepared->capture_compatible());
  Tensor input(description, fixture.resource.ref());
  Tensor values(prepared->output_description(0).value(),
                fixture.resource.ref());
  Tensor indices(prepared->output_description(1).value(),
                 fixture.resource.ref());
  ASSERT_TRUE(input.write<T>(data).ok());
  auto slot = prepared->create_invocation().value();
  auto done = slot.submit(fixture.queue, input.view.as_const(), values.view,
                          indices.view);
  ASSERT_TRUE(done.has_value()) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto actual_values = values.read<T>();
  const auto actual_indices = indices.read<std::int32_t>();
  const int axis = operation.axis < 0
                       ? operation.axis + static_cast<int>(description.rank())
                       : operation.axis;
  const auto width = static_cast<std::size_t>(description.dimensions()[axis]);
  std::size_t inner = 1, outer = 1;
  for (int dim = 0; dim < axis; ++dim)
    outer *= description.dimensions()[dim];
  for (std::size_t dim = axis + 1; dim < description.rank(); ++dim)
    inner *= description.dimensions()[dim];
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t part = 0; part < inner; ++part) {
      std::vector<std::int32_t> reference(width);
      std::iota(reference.begin(), reference.end(), 0);
      std::stable_sort(
          reference.begin(), reference.end(), [&](auto left, auto right) {
            const auto a =
                half_test::decode(data[(row * width + left) * inner + part]);
            const auto b =
                half_test::decode(data[(row * width + right) * inner + part]);
            if (std::isnan(a) != std::isnan(b))
              return !std::isnan(a);
            if (std::isnan(a) || a == b)
              return false;
            return operation.largest ? a > b : a < b;
          });
      reference.resize(operation.count);
      std::vector<std::int32_t> actual;
      for (std::size_t col = 0; col < operation.count; ++col) {
        const auto output = (row * operation.count + col) * inner + part;
        const auto selected = actual_indices[output];
        ASSERT_GE(selected, 0);
        ASSERT_LT(static_cast<std::size_t>(selected), width);
        actual.push_back(selected);
        const auto source = (row * width + selected) * inner + part;
        EXPECT_EQ(std::memcmp(&actual_values[output], &data[source], sizeof(T)),
                  0);
      }
      if (!operation.sorted) {
        std::sort(reference.begin(), reference.end());
        std::sort(actual.begin(), actual.end());
      }
      EXPECT_EQ(actual, reference);
    }
  }
}

template <class T> std::vector<T> topk_values(std::size_t count) {
  std::vector<T> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = half_test::encode<T>((static_cast<int>(i % 97) - 48) * 0.1875);
    switch (i % 67) {
    case 0:
      values[i] = half_test::encode<T>(NAN);
      break;
    case 1:
      values[i] = half_test::encode<T>(-INFINITY);
      break;
    case 2:
      values[i] = half_test::encode<T>(INFINITY);
      break;
    case 3:
      values[i] = half_test::encode<T>(-0.0);
      break;
    case 4:
      values[i] = half_test::encode<T>(0.0);
      break;
    }
  }
  return values;
}
template <class T> void topk_cases() {
  Fixture fixture;
  for (const std::int64_t width : {1, 127, 129, 256, 257, 288, 384, 512, 513,
                                   896, 897, 1025, 2049, 4096, 4097, 65537}) {
    for (const bool strided : {false, true}) {
      const auto desc =
          reduction_desc(half_test::data_type<T>(),
                         std::array<std::int64_t, 2>{2, width}, strided);
      const auto values = topk_values<T>(*desc.element_count());
      std::vector<std::size_t> counts{1};
      if (width <= 4096)
        for (const auto k : {6, 8, 10, 16, static_cast<int>(width)})
          if (k <= width && k > 1)
            counts.push_back(k);
      for (const auto k : counts) {
        for (const bool largest : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << "width=" << width << " k=" << k
                       << " largest=" << largest << " strided=" << strided);
          check_topk<T>(fixture, desc, values,
                        TopK{.count = k, .largest = largest});
        }
      }
    }
  }
  for (const auto &shape : std::vector<std::vector<std::int64_t>>{
           {2, 129, 3}, {2, 1, 2, 1, 3, 129, 1, 2}, {0, 129}}) {
    for (const bool strided : {false, true}) {
      const auto desc =
          reduction_desc(half_test::data_type<T>(), shape, strided);
      const auto values = topk_values<T>(*desc.element_count());
      for (const int axis : {0, 1, -1}) {
        const auto width = shape[axis < 0 ? axis + shape.size() : axis];
        if (width == 0)
          continue;
        for (const bool sorted : {false, true})
          for (const bool stable : {false, true})
            check_topk<T>(fixture, desc, values,
                          TopK{.count = static_cast<std::size_t>(width),
                               .axis = axis,
                               .largest = false,
                               .sorted = sorted,
                               .stable = stable});
      }
    }
  }
}
TEST(CudaSelection, TopKValuesAndIndicesUseStableIndependentOrdering) {
  topk_cases<float>();
  topk_cases<hyle::simd::float16_t>();
  topk_cases<hyle::simd::bfloat16_t>();
}

template <class T> void topk_ties() {
  Fixture fixture;
  const auto desc = reduction_desc(half_test::data_type<T>(),
                                   std::array<std::int64_t, 2>{2, 897}, true);
  for (const bool nan : {false, true}) {
    std::vector<T> values(*desc.element_count());
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (nan) {
        if constexpr (std::is_same_v<T, float>)
          values[i] =
              std::bit_cast<float>(0x7fc00001U + static_cast<unsigned>(i));
        else
          values[i] = T::from_bits(static_cast<std::uint16_t>(
              (std::is_same_v<T, hyle::simd::float16_t> ? 0x7e00U : 0x7fc0U) |
              (i % 31 + 1) | (i % 2 ? 0x8000 : 0)));
      } else
        values[i] = half_test::encode<T>(i % 2 ? -0.0 : 0.0);
    }
    for (const auto count : {1U, 16U, 897U})
      for (const bool largest : {false, true})
        check_topk<T>(fixture, desc, values,
                      TopK{.count = count, .largest = largest});
  }
  const auto broadcast =
      TensorDesc::strided_bytes(half_test::data_type<T>(),
                                std::array<std::int64_t, 2>{3, 129},
                                std::array<std::int64_t, 2>{0, sizeof(T) + 1})
          .value();
  std::vector<T> values(*broadcast.element_count());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = half_test::encode<T>(static_cast<int>(i % 129) - 64);
  check_topk<T>(fixture, broadcast, values, TopK{.count = 16});
}
TEST(CudaSelection, TopKTiesSignedZerosAndNanPayloadsSurvivePadding) {
  topk_ties<float>();
  topk_ties<hyle::simd::float16_t>();
  topk_ties<hyle::simd::bfloat16_t>();
}

TEST(CudaSelection, TopKRejectsInvalidCountsCapacityAndEveryOverlap) {
  Fixture fixture;
  const auto desc =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{129})
          .value();
  const auto zero =
      prepare(fixture.owner, TopK{.count = 0}, desc, fixture.resource.ref());
  ASSERT_FALSE(zero.has_value());
  EXPECT_TRUE(zero.error().IsInvalidArgument());
  const auto oversized =
      prepare(fixture.owner, TopK{.count = 130}, desc, fixture.resource.ref());
  ASSERT_FALSE(oversized.has_value());
  EXPECT_TRUE(oversized.error().IsOutOfRange());
  const auto longer =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{4097})
          .value();
  const auto limited =
      prepare(fixture.owner, TopK{.count = 16}, longer, fixture.resource.ref());
  ASSERT_FALSE(limited.has_value());
  EXPECT_TRUE(limited.error().IsUnimplemented());
  const auto empty =
      TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 1>{0})
          .value();
  const auto empty_axis =
      prepare(fixture.owner, TopK{}, empty, fixture.resource.ref());
  ASSERT_FALSE(empty_axis.has_value());
  EXPECT_TRUE(empty_axis.error().IsOutOfRange());
  auto prepared =
      prepare(fixture.owner, TopK{.count = 8}, desc, fixture.resource.ref())
          .value();
  auto slot = prepared.create_invocation().value();
  Tensor input(desc, fixture.resource.ref()),
      values(prepared.output_description(0).value(), fixture.resource.ref()),
      indices(prepared.output_description(1).value(), fixture.resource.ref());
  std::vector<float> original(129, 2);
  ASSERT_TRUE(input.write<float>(original).ok());
  const auto aliased_values =
      TensorView::create(prepared.output_description(0).value(),
                         input.storage.view().subview(0, 32).value())
          .value();
  auto result = slot.submit(fixture.queue, input.view.as_const(),
                            aliased_values, indices.view);
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsInvalidArgument());
  const auto aliased_indices =
      TensorView::create(prepared.output_description(1).value(),
                         input.storage.view().subview(4, 32).value())
          .value();
  result = slot.submit(fixture.queue, input.view.as_const(), values.view,
                       aliased_indices);
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsInvalidArgument());
  const auto overlapping_outputs =
      TensorView::create(prepared.output_description(1).value(),
                         values.storage.view())
          .value();
  result = slot.submit(fixture.queue, input.view.as_const(), values.view,
                       overlapping_outputs);
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsInvalidArgument());
  EXPECT_EQ(input.read<float>(), original);
}
} // namespace
