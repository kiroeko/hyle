#pragma once

#include "../compute_provider_native/half_test_support.h"
#include "test_support.h"

#include <cstring>
#include <limits>

namespace ggml_cuda_tensor_test {
using namespace hyle::compute;
namespace mem = hyle::mem;
using mem::Buffer;

inline TensorDesc dense(DataType type,
                        std::initializer_list<std::int64_t> shape) {
  return TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
inline Buffer output_buffer(mem::MemoryResourceRef resource,
                            const TensorDesc &description) {
  return Buffer::create(resource, description.required_bytes().value(), 256)
      .value();
}
template <class Value>
Buffer upload(mem::MemoryResourceRef resource, std::span<const Value> values) {
  auto result = Buffer::create(resource, values.size_bytes(), 256).value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());
  return result;
}
template <class Op>
hyle::Status execute(Context &context, Op operation,
                     std::span<const ConstTensorView> inputs,
                     TensorView output) {
  auto prepared =
      prepare(context, Operation(operation), inputs, std::span(&output, 1));
  if (!prepared)
    return std::move(prepared).error();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto invocation = prepared->create_invocation();
  if (!invocation)
    return std::move(invocation).error();
  auto queue = context.create_queue();
  if (!queue)
    return std::move(queue).error();
  auto completion = invocation->submit(*queue, inputs, std::span(&output, 1));
  if (!completion)
    return std::move(completion).error();
  return completion->wait();
}

template <class T> void norms(Context &context) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();

  const auto description = dense(half_test::data_type<T>(), {2, 17, 31});
  const auto parameter = dense(DataType::kFloat32, {31});
  std::vector<T> values(description.element_count().value());
  std::vector<float> scales(31), biases(31);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = half_test::encode<T>((static_cast<int>(i % 29) - 14) / 8.0);
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = 0.5F + static_cast<float>(i % 3) / 8.F;
    biases[i] = static_cast<float>(i % 5) / 16.F;
  }
  auto source = upload<T>(resource, values);
  auto scale = upload<float>(resource, scales);
  auto bias = upload<float>(resource, biases);
  auto output = output_buffer(resource, description);
  const std::array inputs{
      ConstTensorView::create(description, source.view()).value(),
      ConstTensorView::create(parameter, scale.view()).value(),
      ConstTensorView::create(parameter, bias.view()).value()};
  for (bool rms : {true, false}) {
    const auto operation = rms ? Operation(RmsNorm{}) : Operation(LayerNorm{});
    auto status =
        execute(context, operation, std::span(inputs).first(rms ? 2 : 3),
                TensorView::create(description, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<T>(output);
    for (std::size_t row = 0; row < 34; ++row) {
      double mean = 0, squares = 0;
      if (!rms) {
        for (std::size_t col = 0; col < 31; ++col)
          mean += half_test::decode(values[row * 31 + col]);
        mean /= 31;
      }
      for (std::size_t col = 0; col < 31; ++col) {
        const double centered =
            half_test::decode(values[row * 31 + col]) - mean;
        squares += centered * centered;
      }
      for (std::size_t col = 0; col < 31; ++col) {
        const auto index = row * 31 + col;
        const double expected = (half_test::decode(values[index]) - mean) /
                                    std::sqrt(squares / 31 + 1.e-5) *
                                    scales[col] +
                                (rms ? 0 : biases[col]);
        half_test::expect_near(actual[index], expected);
      }
    }
  }
}

template <class Value> void check_general_broadcast(Context &context) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();

  using Shape = std::vector<std::int64_t>;
  const std::array<std::pair<Shape, Shape>, 3> shapes{
      {{{2, 1, 3, 1, 2, 1}, {1, 3, 1, 2, 1, 5}},
       {{2, 1, 2, 1, 2, 1, 2, 1}, {1, 2, 1, 2, 1, 2, 1, 3}},
       {{3, 1, 2, 1}, {2, 1, 1, 4, 1, 5}}}};
  for (const auto &[left_shape, right_shape] : shapes) {
    const auto left_desc =
        TensorDesc::dense(half_test::data_type<Value>(), left_shape).value();
    const auto right_desc =
        TensorDesc::dense(half_test::data_type<Value>(), right_shape).value();
    std::vector<Value> left(left_desc.element_count().value()),
        right(right_desc.element_count().value());
    for (std::size_t i = 0; i < left.size(); ++i)
      left[i] = half_test::encode<Value>(0.5 + static_cast<double>(i % 11) / 8);
    for (std::size_t i = 0; i < right.size(); ++i)
      right[i] =
          half_test::encode<Value>(0.75 + static_cast<double>(i % 7) / 8);
    auto left_buffer = upload<Value>(resource, left);
    auto right_buffer = upload<Value>(resource, right);
    const auto left_view =
        ConstTensorView::create(left_desc, left_buffer.view()).value();
    const auto right_view =
        ConstTensorView::create(right_desc, right_buffer.view()).value();
    for (const auto kind : {BinaryKind::kAdd, BinaryKind::kSubtract,
                            BinaryKind::kMultiply, BinaryKind::kDivide}) {
      for (const bool reverse : {false, true}) {
        const std::array descriptions{left_desc, right_desc};
        const auto description =
            infer_single_output(Operation(Binary{kind}), descriptions).value();
        const std::array inputs{reverse ? right_view : left_view,
                                reverse ? left_view : right_view};
        auto output = output_buffer(resource, description);
        const auto status =
            execute(context, Binary{kind}, inputs,
                    TensorView::create(description, output.view()).value());
        ASSERT_TRUE(status.ok()) << status.message();
        const auto actual = half_test::read<Value>(output);
        for (std::size_t i = 0; i < actual.size(); ++i) {
          auto remainder = i;
          std::array<std::size_t, kMaxTensorRank> coordinates{};
          for (std::size_t d = description.rank(); d-- > 0;) {
            const auto extent =
                static_cast<std::size_t>(description.dimensions()[d]);
            coordinates[d] = remainder % extent;
            remainder /= extent;
          }
          const auto offset = [&](const TensorDesc &operand) {
            std::size_t index = 0;
            for (std::size_t d = 0; d < operand.rank(); ++d) {
              const auto extent =
                  static_cast<std::size_t>(operand.dimensions()[d]);
              index =
                  index * extent +
                  (extent == 1
                       ? 0
                       : coordinates[description.rank() - operand.rank() + d]);
            }
            return index;
          };
          const float l =
              static_cast<float>(half_test::decode(left[offset(left_desc)]));
          const float r =
              static_cast<float>(half_test::decode(right[offset(right_desc)]));
          const float a = reverse ? r : l, b = reverse ? l : r;
          const float expected = kind == BinaryKind::kAdd        ? a + b
                                 : kind == BinaryKind::kSubtract ? a - b
                                 : kind == BinaryKind::kMultiply ? a * b
                                                                 : a / b;
          if constexpr (std::is_same_v<Value, float>)
            EXPECT_NEAR(actual[i], expected, 2.e-6F);
          else
            half_test::expect_near(actual[i], expected);
        }
      }
    }
    const auto preserved_left = half_test::read<Value>(left_buffer);
    const auto preserved_right = half_test::read<Value>(right_buffer);
    EXPECT_EQ(std::memcmp(left.data(), preserved_left.data(),
                          left.size() * sizeof(Value)),
              0);
    EXPECT_EQ(std::memcmp(right.data(), preserved_right.data(),
                          right.size() * sizeof(Value)),
              0);
  }
}

template <class Value>
void check_reduction_axes(Context &context,
                          std::initializer_list<std::int64_t> shape,
                          bool strided = false) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();

  auto description = dense(half_test::data_type<Value>(), shape);
  if (strided) {
    std::vector<std::int64_t> strides(description.byte_strides().begin(),
                                      description.byte_strides().end());
    for (auto &stride : strides)
      stride *= 2;
    description = TensorDesc::strided_bytes(half_test::data_type<Value>(),
                                            description.dimensions(), strides)
                      .value();
  }
  const auto size = description.element_count().value();
  const std::size_t spacing = strided ? 2 : 1;
  std::vector<Value> values(size), physical(size * spacing);
  for (std::size_t i = 0; i < size; ++i) {
    values[i] = half_test::encode<Value>(
        static_cast<double>(static_cast<int>(i % 23) - 11) / 32);
    physical[i * spacing] = values[i];
  }
  auto source = upload<Value>(resource, physical);
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  const auto end = std::size_t{1} << description.rank();
  for (std::size_t mask = description.rank() == 0 ? 0 : 1; mask < end; ++mask) {
    if (description.rank() == 6 && mask != 1 && mask != 2 && mask != 3 &&
        mask != 21 && mask != 33 && mask != 42 && mask != 63)
      continue;
    for (const bool keep : {false, true}) {
      std::vector<std::int32_t> requested;
      for (std::size_t d = description.rank(); d-- > 0;)
        if ((mask & (std::size_t{1} << d)) != 0)
          requested.push_back(
              static_cast<std::int32_t>(d) -
              (keep ? 0 : static_cast<std::int32_t>(description.rank())));
      const auto axes = requested.empty() || (mask == end - 1 && keep)
                            ? Axes::all()
                            : Axes::from(requested).value();
      for (const auto kind : {ReductionKind::kSum, ReductionKind::kMean}) {
        SCOPED_TRACE(::testing::Message()
                     << "rank=" << description.rank() << " axes=" << mask
                     << " keep=" << keep);
        const Reduce operation{.kind = kind,
                               .axes = axes,
                               .keep_dimensions = keep,
                               .accumulation_type = DataType::kFloat32};
        const auto out_desc = infer_single_output(Operation(operation),
                                                  std::span(&description, 1))
                                  .value();
        auto output = output_buffer(resource, out_desc);
        const auto status =
            execute(context, operation, std::span(&input, 1),
                    TensorView::create(out_desc, output.view()).value());
        ASSERT_TRUE(status.ok()) << status.message();
        const auto actual = half_test::read<Value>(output);
        std::vector<double> sums(actual.size());
        std::vector<std::size_t> counts(actual.size());
        for (std::size_t i = 0; i < size; ++i) {
          auto remainder = i;
          std::array<std::size_t, kMaxTensorRank> coordinates{};
          for (std::size_t d = description.rank(); d-- > 0;) {
            const auto extent =
                static_cast<std::size_t>(description.dimensions()[d]);
            coordinates[d] = remainder % extent;
            remainder /= extent;
          }
          std::size_t output_index = 0;
          for (std::size_t d = 0; d < description.rank(); ++d)
            if ((mask & (std::size_t{1} << d)) == 0)
              output_index = output_index * static_cast<std::size_t>(
                                                description.dimensions()[d]) +
                             coordinates[d];
          sums[output_index] += half_test::decode(values[i]);
          ++counts[output_index];
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
          const auto expected = kind == ReductionKind::kSum ? sums[i]
                                : counts[i] == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : sums[i] / static_cast<double>(counts[i]);
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(half_test::decode(actual[i])));
          } else if constexpr (std::is_same_v<Value, float>) {
            EXPECT_NEAR(actual[i], expected, 2.e-6);
          } else {
            half_test::expect_near(actual[i], expected);
          }
        }
      }
    }
  }
}

inline void unary_and_cast(Context &context) {
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto resource = memory.ref();

  const std::array values{-std::numeric_limits<float>::infinity(),
                          -4.0F,
                          -0.0F,
                          0.0F,
                          0.125F,
                          2.0F,
                          std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()};
  const auto description = dense(DataType::kFloat32, {8});
  auto source = upload<float>(resource, values);
  auto output = output_buffer(resource, description);
  const auto input =
      ConstTensorView::create(description, source.view()).value();
  for (auto kind :
       {UnaryKind::kAbs, UnaryKind::kNeg, UnaryKind::kExp, UnaryKind::kLog,
        UnaryKind::kSqrt, UnaryKind::kRsqrt, UnaryKind::kTanh}) {
    auto status =
        execute(context, Unary{kind}, std::span(&input, 1),
                TensorView::create(description, output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<float>(output);
    for (std::size_t i = 0; i < values.size(); ++i) {
      double expected = 0;
      switch (kind) {
      case UnaryKind::kAbs:
        expected = std::abs(values[i]);
        break;
      case UnaryKind::kNeg:
        expected = -values[i];
        break;
      case UnaryKind::kExp:
        expected = std::exp(values[i]);
        break;
      case UnaryKind::kLog:
        expected = std::log(values[i]);
        break;
      case UnaryKind::kSqrt:
        expected = std::sqrt(values[i]);
        break;
      case UnaryKind::kRsqrt:
        expected = 1.0F / std::sqrt(values[i]);
        break;
      case UnaryKind::kTanh:
        expected = std::tanh(values[i]);
        break;
      default:
        FAIL();
      }
      half_test::expect_near(actual[i], expected);
    }
  }
  // RNE ties, signed zero, subnormals and infinities use an independent
  // encoder.
  const std::array rounding{
      -0.0F,          1.00048828125F,
      1.00146484375F, 1.00390625F,
      1.01171875F,    0x1p-24F,
      65504.F,        std::numeric_limits<float>::infinity()};
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(rounding))).ok());
  const auto check = [&]<class T>() {
    const auto converted = dense(half_test::data_type<T>(), {8});
    auto cast_output = output_buffer(resource, converted);
    auto status =
        execute(context, Cast{half_test::data_type<T>()}, std::span(&input, 1),
                TensorView::create(converted, cast_output.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = half_test::read<T>(cast_output);
    for (std::size_t i = 0; i < rounding.size(); ++i)
      EXPECT_EQ(actual[i].bits, half_test::encode<T>(rounding[i]).bits);
  };
  check.template operator()<hyle::simd::float16_t>();
  check.template operator()<hyle::simd::bfloat16_t>();
}

} // namespace ggml_cuda_tensor_test
