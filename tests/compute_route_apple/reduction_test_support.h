#pragma once

#include <numeric>

#include "tensor_test_support.h"

namespace apple_tensor_test {
inline hyle::Result<std::vector<mx::array>>
execute_all(a::Context &context, c::Operation operation,
            std::span<const mx::array> inputs) {
  std::vector<a::InputSpec> descriptions;
  for (const auto &input : inputs)
    descriptions.push_back({input.shape(), input.dtype()});
  HYLE_ASSIGN_OR_RETURN(auto program, context.prepare(operation, descriptions));
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
  auto outputs = result.get()->arrays();
  return std::vector<mx::array>(outputs.begin(), outputs.end());
}
// Scalar FP64 normalization; inputs are already rounded to their storage
// type. Scale/mask follow the specified F32 input arithmetic, not MLX ops.
inline std::vector<double>
probability_oracle(std::span<const double> data, const mx::Shape &shape,
                   c::Softmax operation, std::span<const double> mask = {}) {
  int axis = operation.axis < 0
                 ? operation.axis + static_cast<int>(shape.size())
                 : operation.axis;
  const std::size_t width = shape[axis];
  std::size_t inner = 1;
  for (std::size_t i = axis + 1; i < shape.size(); ++i)
    inner *= shape[i];
  std::vector<double> output(data.size());
  if (data.empty())
    return output;
  for (std::size_t outer = 0; outer < data.size() / width / inner; ++outer) {
    for (std::size_t lane = 0; lane < inner; ++lane) {
      std::vector<double> row(width);
      double maximum = -std::numeric_limits<double>::infinity();
      bool invalid = false;
      for (std::size_t col = 0; col < width; ++col) {
        auto x =
            static_cast<float>(data[(outer * width + col) * inner + lane]) *
            operation.scale;
        const auto m = mask.empty() ? 0.0 : mask[col];
        if (operation.mask == c::MaskKind::kBoolean && m == 0)
          x = -std::numeric_limits<float>::infinity();
        if (operation.mask == c::MaskKind::kAdditive)
          x = static_cast<float>(x + static_cast<float>(m));
        row[col] = x;
        invalid |= std::isnan(x) || x == std::numeric_limits<float>::infinity();
        maximum = std::max(maximum, row[col]);
      }
      invalid |= !std::isfinite(maximum);
      double total = 0;
      for (double x : row)
        total += std::exp(x - maximum);
      for (std::size_t col = 0; col < width; ++col)
        output[(outer * width + col) * inner + lane] =
            invalid ? std::numeric_limits<double>::quiet_NaN()
            : operation.kind == c::SoftmaxKind::kSoftmax
                ? std::exp(row[col] - maximum) / total
                : row[col] - maximum - std::log(total);
    }
  }
  return output;
}
inline std::vector<std::size_t> top_indices(std::span<const double> row,
                                            std::size_t count, bool largest) {
  std::vector<std::size_t> order(row.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
    const auto a = row[left], b = row[right];
    if (std::isnan(a) != std::isnan(b))
      return !std::isnan(a);
    if (std::isnan(a) || a == b)
      return false;
    return largest ? a > b : a < b;
  });
  order.resize(count);
  return order;
}
} // namespace apple_tensor_test
