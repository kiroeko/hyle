#pragma once

#include <numeric>

#include "tensor_test_support.h"

namespace vulkan_reduction_test {
using namespace vulkan_tensor_test;

inline std::vector<double> mask_values(std::span<const std::uint64_t> flags) {
  std::vector<double> result;
  result.reserve(flags.size());
  for (auto flag : flags)
    result.push_back(flag != 0 ? 1.0 : 0.0);
  return result;
}

inline std::vector<double> pattern(std::size_t count, int phase = 0) {
  std::vector<double> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = (static_cast<int>((i * 7 + phase) % 31) - 15) * 0.125;
  return result;
}

// Literal input traversal into mathematical output coordinates, independent
// of the workgroup shape and of the device's reduction address mapping.
inline std::vector<double> reduce_reference(std::span<const double> input,
                                            const c::TensorDesc &description,
                                            c::Reduce operation) {
  const auto shape = description.dimensions();
  std::vector<bool> selected(shape.size(), operation.axes.is_all());
  for (int axis : operation.axes.values())
    selected[axis < 0 ? axis + static_cast<int>(shape.size()) : axis] = true;
  std::size_t count = 1;
  for (std::size_t axis = 0; axis < shape.size(); ++axis)
    if (!selected[axis])
      count *= static_cast<std::size_t>(shape[axis]);
  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const bool minimum = operation.kind == c::ReductionKind::kMinimum;
  const bool maximum = operation.kind == c::ReductionKind::kMaximum;
  std::vector<double> result(count, minimum ? inf : maximum ? -inf : 0);
  std::vector<std::size_t> counts(count);
  for (std::size_t i = 0; i < input.size(); ++i) {
    std::vector<std::size_t> coordinates(shape.size());
    auto remaining = i;
    for (std::size_t axis = shape.size(); axis-- > 0;) {
      coordinates[axis] = remaining % shape[axis];
      remaining /= shape[axis];
    }
    std::size_t destination = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis)
      if (!selected[axis])
        destination = destination * shape[axis] + coordinates[axis];
    auto &accumulator = result[destination];
    const auto value = input[i];
    ++counts[destination];
    if (std::isnan(accumulator) || std::isnan(value))
      accumulator = nan;
    else if ((minimum || maximum) && value == 0 && accumulator == 0)
      accumulator = std::copysign(
          0.0, (minimum ? std::signbit(value) || std::signbit(accumulator)
                        : std::signbit(value) && std::signbit(accumulator))
                   ? -1.0
                   : 1.0);
    else if (minimum)
      accumulator = std::min(accumulator, value);
    else if (maximum)
      accumulator = std::max(accumulator, value);
    else
      accumulator += value;
  }
  if (operation.kind == c::ReductionKind::kMean)
    for (std::size_t i = 0; i < count; ++i)
      result[i] = counts[i] ? result[i] / counts[i] : nan;
  return result;
}

// Independent scalar FP64 formula. Input scale/additive-mask arithmetic is
// explicitly F32, as specified; the axis mask is broadcast to every row.
inline std::vector<double>
probability_reference(std::span<const double> input,
                      const c::TensorDesc &description, c::Softmax operation,
                      std::span<const double> mask = {}) {
  std::vector<double> output(input.size());
  if (input.empty())
    return output;
  const auto shape = description.dimensions();
  const auto axis = operation.axis < 0
                        ? operation.axis + static_cast<int>(shape.size())
                        : operation.axis;
  const auto width = static_cast<std::size_t>(shape[axis]);
  std::size_t inner = 1;
  for (std::size_t dim = axis + 1; dim < shape.size(); ++dim)
    inner *= shape[dim];
  for (std::size_t outer = 0; outer < input.size() / width / inner; ++outer)
    for (std::size_t lane = 0; lane < inner; ++lane) {
      std::vector<double> row(width);
      bool invalid = false;
      for (std::size_t col = 0; col < width; ++col) {
        const float scaled =
            static_cast<float>(input[(outer * width + col) * inner + lane]) *
            operation.scale;
        float value = scaled;
        if (operation.mask == c::MaskKind::kBoolean && mask[col] == 0)
          value = -std::numeric_limits<float>::infinity();
        if (operation.mask == c::MaskKind::kAdditive)
          value = scaled + static_cast<float>(mask[col]);
        row[col] = value;
        invalid |= std::isnan(value) ||
                   value == std::numeric_limits<float>::infinity();
      }
      const auto peak = static_cast<std::size_t>(
          std::max_element(row.begin(), row.end()) - row.begin());
      const auto maximum = row[peak];
      invalid |= !std::isfinite(maximum);
      double tail = 0;
      for (std::size_t col = 0; col < width; ++col)
        if (col != peak)
          tail += std::exp(row[col] - maximum);
      for (std::size_t col = 0; col < width; ++col)
        output[(outer * width + col) * inner + lane] =
            invalid ? std::numeric_limits<double>::quiet_NaN()
            : operation.kind == c::SoftmaxKind::kSoftmax
                ? std::exp(row[col] - maximum) / (1 + tail)
                : row[col] - maximum - std::log1p(tail);
    }
  return output;
}

} // namespace vulkan_reduction_test
