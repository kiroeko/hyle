#pragma once

#include "hyle/compute/apple.h"

namespace hyle::compute::apple::detail {
inline constexpr std::int32_t kIndexOutOfRange = 1;
inline constexpr std::int32_t kScanOverflow = 2;

// Registers a lazy GPU status only while a managed native region is traced.
// The returned native dependency preserves reachability without copying data.
Result<mlx::core::array> with_device_status(mlx::core::array output,
                                            mlx::core::array status,
                                            mlx::core::Stream stream);
} // namespace hyle::compute::apple::detail
