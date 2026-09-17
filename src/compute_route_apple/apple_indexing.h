#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
Result<mlx::core::array> lower_scatter(const Scatter &operation,
                                       std::span<const mlx::core::array> inputs,
                                       mlx::core::Stream stream);
Result<std::vector<mlx::core::array>>
lower_indexing(const Operation &operation,
               std::span<const mlx::core::array> inputs,
               const mlx::core::Shape &output_shape, mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
