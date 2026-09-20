#pragma once

#include "tensor_test_support.h"

#include <algorithm>

namespace ggml_cuda_tensor_test {

template <class T>
std::vector<double>
softmax_reference(std::span<const T> values,
                  std::span<const std::int64_t> shape, std::size_t axis,
                  float scale, std::span<const T> mask = {},
                  std::span<const std::int64_t> mask_shape = {}) {
  std::size_t inner = 1;
  for (std::size_t i = axis + 1; i < shape.size(); ++i)
    inner *= static_cast<std::size_t>(shape[i]);
  const auto width = static_cast<std::size_t>(shape[axis]);
  const auto outer = values.size() / (width * inner);
  const auto value = [&](std::size_t index) {
    double result = half_test::decode(values[index]) * scale;
    if (!mask.empty()) {
      std::size_t offset = 0, stride = 1;
      for (std::size_t d = shape.size(); d-- > 0;) {
        const auto coordinate = index % static_cast<std::size_t>(shape[d]);
        index /= static_cast<std::size_t>(shape[d]);
        if (d + mask_shape.size() >= shape.size()) {
          const auto extent = static_cast<std::size_t>(
              mask_shape[d + mask_shape.size() - shape.size()]);
          offset += (extent == 1 ? 0 : coordinate) * stride;
          stride *= extent;
        }
      }
      result += half_test::decode(mask[offset]);
    }
    return result;
  };
  std::vector<double> result(values.size());
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t column = 0; column < inner; ++column) {
      double maximum = -std::numeric_limits<double>::infinity();
      bool nan = false;
      for (std::size_t k = 0; k < width; ++k) {
        const auto v = value((row * width + k) * inner + column);
        nan |= std::isnan(v);
        maximum = std::max(maximum, v);
      }
      double sum = 0;
      for (std::size_t k = 0; k < width; ++k)
        sum += std::exp(value((row * width + k) * inner + column) - maximum);
      for (std::size_t k = 0; k < width; ++k) {
        const auto index = (row * width + k) * inner + column;
        result[index] = nan || !std::isfinite(maximum)
                            ? std::numeric_limits<double>::quiet_NaN()
                            : std::exp(value(index) - maximum) / sum;
      }
    }
  }
  return result;
}

template <class T>
void softmax_case(Context &context, std::vector<std::int64_t> shape,
                  std::int32_t axis, bool masked, bool strided = false,
                  bool inplace = false, float scale = 0.5F,
                  bool special = false, bool all_masked = false) {
  SCOPED_TRACE(::testing::Message()
               << "axis=" << axis << " rank=" << shape.size()
               << " mask=" << masked << " strided=" << strided
               << " inplace=" << inplace << " scale=" << scale);
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto type = half_test::data_type<T>();
  const auto output_desc = TensorDesc::dense(type, shape).value();
  const auto selected = static_cast<std::size_t>(
      axis < 0 ? axis + static_cast<std::int32_t>(shape.size()) : axis);
  auto input_desc = output_desc;
  if (strided) {
    std::vector<std::int64_t> strides(output_desc.byte_strides().begin(),
                                      output_desc.byte_strides().end());
    for (auto &stride : strides)
      stride *= 2;
    input_desc = TensorDesc::strided_bytes(type, shape, strides).value();
  }
  std::vector<T> logical(output_desc.element_count().value());
  for (std::size_t i = 0; i < logical.size(); ++i)
    logical[i] = half_test::encode<T>((static_cast<int>(i % 29) - 14) / 4.0);
  if (special) {
    const auto width = static_cast<std::size_t>(shape.back());
    for (std::size_t i = 0; i < width; ++i)
      logical[i] =
          half_test::encode<T>(-std::numeric_limits<double>::infinity());
    logical[width + 1] =
        half_test::encode<T>(std::numeric_limits<double>::quiet_NaN());
    logical[2 * width] =
        half_test::encode<T>(std::numeric_limits<double>::infinity());
  }
  const auto step = strided ? 2U : 1U;
  std::vector<T> physical(logical.size() * step, half_test::encode<T>(99));
  for (std::size_t i = 0; i < logical.size(); ++i)
    physical[i * step] = logical[i];
  auto input = upload<T>(memory.ref(), physical);
  auto output = output_buffer(memory.ref(), output_desc);
  std::vector<std::int64_t> mask_shape(shape.size(), 1);
  mask_shape[selected] = shape[selected];
  if (shape.size() > 4)
    mask_shape[0] = shape[0];
  const auto mask_desc = TensorDesc::dense(type, mask_shape).value();
  std::vector<T> mask_values(mask_desc.element_count().value());
  for (std::size_t i = 0; i < mask_values.size(); ++i)
    mask_values[i] = half_test::encode<T>(
        i % 3 == 1 ? -std::numeric_limits<double>::infinity()
                   : (static_cast<int>(i % 7) - 3) / 8.0);
  if (all_masked)
    std::ranges::fill(
        mask_values,
        half_test::encode<T>(-std::numeric_limits<double>::infinity()));
  auto mask = upload<T>(memory.ref(), mask_values);
  const std::array inputs{
      ConstTensorView::create(input_desc, input.view()).value(),
      ConstTensorView::create(mask_desc, mask.view()).value()};
  const auto output_view =
      TensorView::create(output_desc, inplace ? input.view() : output.view())
          .value();
  const Softmax operation{.axis = axis,
                          .scale = scale,
                          .mask =
                              masked ? MaskKind::kAdditive : MaskKind::kNone};
  auto status = execute(context, operation,
                        std::span(inputs).first(masked ? 2 : 1), output_view);
  ASSERT_TRUE(status.ok()) << status.message();
  const auto actual = half_test::read<T>(inplace ? input : output);
  const auto expected = softmax_reference<T>(
      logical, shape, selected, scale,
      masked ? std::span<const T>(mask_values) : std::span<const T>{},
      mask_shape);
  for (std::size_t i = 0; i < expected.size(); ++i)
    half_test::expect_near(actual[i], expected[i]);
  if (!inplace) {
    const auto retained = half_test::read<T>(input);
    EXPECT_EQ(std::memcmp(retained.data(), physical.data(),
                          physical.size() * sizeof(T)),
              0);
  }
  const auto retained_mask = half_test::read<T>(mask);
  EXPECT_EQ(std::memcmp(retained_mask.data(), mask_values.data(),
                        mask_values.size() * sizeof(T)),
            0);
}

template <class T> void softmax_batch(Context &context) {
  for (const std::int32_t axis : {0, 1, -1})
    for (const bool masked : {false, true}) {
      softmax_case<T>(context, {2, 3, 17}, axis, masked);
      softmax_case<T>(context, {2, 3, 17}, axis, masked, true);
      softmax_case<T>(context, {2, 3, 17}, axis, masked, false, true);
    }
  softmax_case<T>(context, {2, 1, 3, 2, 1, 7}, 2, true);
  softmax_case<T>(context, {2, 1, 2, 1, 2, 3, 1, 5}, 5, true, false, false,
                  -1.0F);
  softmax_case<T>(context, {4, 35}, -1, false, false, false, 1.0F, true);
  softmax_case<T>(context, {2, 17}, -1, true, false, false, 0.0F);
  softmax_case<T>(context, {3, 1}, -1, false);
  softmax_case<T>(context, {2, 17}, -1, true, false, false, 1.0F, false, true);
  for (const bool masked : {false, true})
    softmax_case<T>(context, {2, 32769}, -1, masked, false, true, 1.0F);
}

} // namespace ggml_cuda_tensor_test
