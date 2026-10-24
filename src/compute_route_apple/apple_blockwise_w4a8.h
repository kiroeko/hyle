#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
Result<std::vector<mlx::core::array>> lower_blockwise_w4a8(
    std::span<const TensorDesc> descriptions,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream,
    bool grouped);
} // namespace hyle::compute::apple::detail
