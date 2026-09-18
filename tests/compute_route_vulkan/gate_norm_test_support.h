#pragma once

#include "tensor_test_support.h"

namespace vulkan_gate_norm_test {
using namespace vulkan_tensor_test;

inline double activate(c::GatedActivationKind kind, double value) {
  return unary(kind == c::GatedActivationKind::kSilu ? c::UnaryKind::kSilu
               : kind == c::GatedActivationKind::kGelu
                   ? c::UnaryKind::kGelu
                   : c::UnaryKind::kGeluTanh,
               value);
}

// Independent scalar FP64 definition, using already storage-rounded inputs.
inline std::vector<double> norm(std::span<const double> input,
                                std::size_t width, bool layer, double epsilon,
                                std::span<const double> scale = {},
                                std::span<const double> bias = {}) {
  std::vector<double> result(input.size());
  if (width == 0)
    return result;
  for (std::size_t row = 0; row < input.size() / width; ++row) {
    double mean = 0, squares = 0;
    if (layer) {
      for (std::size_t column = 0; column < width; ++column)
        mean += input[row * width + column];
      mean /= static_cast<double>(width);
    }
    for (std::size_t column = 0; column < width; ++column) {
      const auto centered = input[row * width + column] - mean;
      squares += centered * centered;
    }
    const auto inverse =
        1 / std::sqrt(squares / static_cast<double>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      auto value = (input[row * width + column] - mean) * inverse;
      if (!scale.empty())
        value *= scale[column];
      if (!bias.empty())
        value += bias[column];
      result[row * width + column] = value;
    }
  }
  return result;
}
} // namespace vulkan_gate_norm_test
