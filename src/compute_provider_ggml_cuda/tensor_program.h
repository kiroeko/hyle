#pragma once

#include <ggml.h>

#include "program.h"

namespace hyle::compute::ggml_cuda::internal {

struct ScalarConstant final {
  ggml_tensor *tensor;
  float value;
};

bool tensor_operation_supports(Operation, std::span<const TensorDesc>,
                               std::span<const TensorDesc>) noexcept;
ggml_tensor *create_tensor_input(ggml_context *, const TensorDesc &);
ggml_tensor *lower_tensor_operation(ggml_context *, const Recipe &,
                                    const Node &, std::span<ggml_tensor *const>,
                                    std::vector<ScalarConstant> &);

} // namespace hyle::compute::ggml_cuda::internal
