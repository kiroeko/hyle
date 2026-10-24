#pragma once

#include "hyle/compute/apple.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::apple::detail {
Result<std::vector<mlx::core::array>> lower_nvfp4(
    std::span<const TensorDesc> descriptions,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
Result<std::vector<mlx::core::array>> lower_grouped_nvfp4(
    std::span<const TensorDesc> descriptions,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
Result<std::vector<mlx::core::array>> lower_local_nvfp4(
    const model_ops::LocalNvfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
Result<std::vector<mlx::core::array>> lower_grouped_local_nvfp4(
    const model_ops::GroupedLocalNvfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
Result<std::vector<mlx::core::array>> lower_mxfp4(
    const model_ops::Mxfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
Result<std::vector<mlx::core::array>> lower_grouped_mxfp4(
    const model_ops::GroupedMxfp4Linear &operation,
    std::span<const TensorDesc> descriptions,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
