#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
Result<std::vector<mlx::core::array>>
lower_sampling(const Operation &, std::span<const mlx::core::array>,
                 mlx::core::Stream);
Result<std::vector<mlx::core::array>>
lower_attention_math_block(const Operation &operation,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream) noexcept;
Result<std::vector<mlx::core::array>>
lower_tensor(const Operation &operation,
             std::span<const mlx::core::array> inputs,
             mlx::core::Stream stream) noexcept;
} // namespace hyle::compute::apple::detail
