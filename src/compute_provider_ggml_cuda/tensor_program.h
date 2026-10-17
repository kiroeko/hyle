#pragma once

#include <ggml.h>

#include "program.h"

namespace hyle::compute::ggml_cuda::internal {

struct ScalarConstant final {
  ggml_tensor *tensor;
  float value;
};
struct DenseConstant final {
  ggml_tensor *tensor;
  const void *data;
  std::size_t bytes;
};

struct HadamardFactors final {
  std::int64_t low, high;
  constexpr std::size_t constant_elements() const noexcept {
    return static_cast<std::size_t>(
        low * low + (high != 1 && high != low ? high * high : 0));
  }
};
constexpr HadamardFactors hadamard_factors(std::int64_t width) noexcept {
  return width > 512 ? HadamardFactors{64, width / 64}
                     : HadamardFactors{width, 1};
}

bool tensor_operation_supports(Operation, std::span<const TensorDesc>,
                               std::span<const TensorDesc>) noexcept;
ggml_tensor *create_tensor_input(ggml_context *, const TensorDesc &);
std::vector<ggml_tensor *> lower_tensor_operation(
    ggml_context *, const Recipe &, const Node &, std::span<ggml_tensor *const>,
    std::vector<ScalarConstant> &, std::vector<DenseConstant> &,
    std::span<const float> hadamard);

} // namespace hyle::compute::ggml_cuda::internal
