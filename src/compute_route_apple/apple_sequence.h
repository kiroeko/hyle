#pragma once

#include "hyle/compute/apple.h"

#include <array>

namespace hyle::compute::apple::detail {
bool qualifies_short_gdn(const Operation &operation,
    std::span<const TensorDesc> inputs) noexcept;
Result<std::vector<mlx::core::array>> lower_short_gdn(const GatedDeltaNet &operation,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
HYLE_COMPUTE_ROUTE_APPLE_API std::vector<mlx::core::array> lower_bfloat_conv_gdn_inputs(
    const mlx::core::array &samples, const mlx::core::array &weights,
    const mlx::core::array &query_epsilon, const mlx::core::array &key_epsilon,
    const mlx::core::Shape &query_shape, const mlx::core::Shape &value_shape,
    std::array<int,3> channel_offsets, mlx::core::Stream stream);
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
