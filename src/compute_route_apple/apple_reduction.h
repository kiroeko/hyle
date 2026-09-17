#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
// Called after the shared schema and native dtype/shape qualification.
Result<std::vector<mlx::core::array>>
lower_reduction(const Operation &operation,
                std::span<const mlx::core::array> inputs,
                const mlx::core::Shape &output_shape, mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
