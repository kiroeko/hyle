#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
bool qualifies_short_gdn(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept;
Result<std::vector<mlx::core::array>> lower_short_gdn(const GatedDeltaNet &operation,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
Status qualify_packed_conv(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept;
Result<std::vector<mlx::core::array>> lower_packed_conv(
    const Operation &operation, std::span<const mlx::core::array> inputs,
    mlx::core::Stream stream);
Status qualify_packed_gdn(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept;
Result<std::vector<mlx::core::array>> lower_packed_gdn(
    const Operation &operation, std::span<const mlx::core::array> inputs,
    mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
