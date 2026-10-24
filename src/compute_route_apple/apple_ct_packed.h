#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::packed::internal { struct ArtifactPackedLinear; }
namespace hyle::compute::apple::detail {
Result<std::vector<mlx::core::array>> lower_ct_packed(
    const packed::internal::ArtifactPackedLinear &, std::span<const TensorDesc>,
    std::span<const mlx::core::array>, mlx::core::Stream);
} // namespace hyle::compute::apple::detail
