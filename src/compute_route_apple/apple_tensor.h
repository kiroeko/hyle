#pragma once

#include "hyle/compute/apple.h"

#include <optional>

namespace hyle::compute::apple::detail {
enum class AttentionPath { kNative, kForcedFused, kMath };
// Cold shape/dtype/capability selection only; never evaluates the arrays.
AttentionPath attention_path(const ScaledDotProductAttention &,
    const mlx::core::array &query, const mlx::core::array &key,
    const mlx::core::array &value, mlx::core::Stream);
Result<std::vector<mlx::core::array>>
lower_sampling(const Operation &, std::span<const mlx::core::array>,
                 mlx::core::Stream);
Result<std::vector<mlx::core::array>>
lower_attention_math_block(const Operation &operation,
    std::span<const mlx::core::array> inputs, mlx::core::Stream stream) noexcept;
// Internal compute operands only. The original input arrays still determine
// inference, logical dtype and final output rounding.
struct PreparedMlxAffineMetadata final {
  std::optional<mlx::core::array> scales;
  std::optional<mlx::core::array> biases;
};
HYLE_COMPUTE_ROUTE_APPLE_API std::optional<mlx::core::array>
lower_short_bfloat_gated_projection(const mlx::core::array &, const mlx::core::array &,
    const mlx::core::array &, mlx::core::Stream);

Result<std::vector<mlx::core::array>>
lower_tensor(const Operation &operation,
             std::span<const mlx::core::array> inputs,
             mlx::core::Stream stream,
             const PreparedMlxAffineMetadata *prepared = nullptr) noexcept;
} // namespace hyle::compute::apple::detail
