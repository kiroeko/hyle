#pragma once

#include "hyle/compute/detail/apple_plan.h"
#include "../compute_packed_shared/static_fp8_linear.h"

namespace hyle::compute::apple::detail {
bool is_mlx_float8_cast(const Operation &, std::span<const TensorDesc>) noexcept;
bool is_mlx_exact_binary(const Operation &, std::span<const TensorDesc>) noexcept;
Result<std::vector<mlx::core::array>> lower_exact_binary(
    const Operation &, std::span<const TensorDesc>,
    std::span<const mlx::core::array>, mlx::core::Stream);
Result<std::vector<mlx::core::array>> lower_float8_cast(
    const Operation &, std::span<const TensorDesc>,
    std::span<const mlx::core::array>, mlx::core::Stream);
Result<mlx::core::array> lower_float8_cache_encode(
    const CacheStorage &, const mlx::core::array &, const TensorDesc &,
    mlx::core::Stream);
Result<std::vector<mlx::core::array>> lower_static_fp8(
    const packed::internal::StaticFloat8Linear &, std::span<const TensorDesc>,
    std::span<const mlx::core::array>, mlx::core::Stream);
}
