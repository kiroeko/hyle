#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
Status qualify_packed_gdn(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept;
Result<std::vector<mlx::core::array>> lower_packed_gdn(
    const Operation &operation, std::span<const mlx::core::array> inputs,
    mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
