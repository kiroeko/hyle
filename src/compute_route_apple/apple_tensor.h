#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
Result<std::vector<mlx::core::array>>
lower_tensor(const Operation &operation,
             std::span<const mlx::core::array> inputs,
             mlx::core::Stream stream) noexcept;
} // namespace hyle::compute::apple::detail
