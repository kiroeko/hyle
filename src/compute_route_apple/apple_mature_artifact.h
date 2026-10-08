#pragma once
#include "hyle/compute/apple.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::apple::detail {
mlx::core::array lower_mature_artifact(model_ops::ArtifactWeightDecode operation,
    std::span<const mlx::core::array> inputs, mlx::core::Shape shape,
    mlx::core::Dtype dtype, mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
