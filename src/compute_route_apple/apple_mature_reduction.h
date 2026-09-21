#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
// Called after common Operation, native shape and floating storage validation.
Result<std::vector<mlx::core::array>> lower_mature_reduction(
    const Operation &operation, std::span<const mlx::core::array> inputs,
    const mlx::core::Shape &output_shape, mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
