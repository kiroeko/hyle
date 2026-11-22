#pragma once

#include <array>
#include <memory>

#include "hyle/compute/apple.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"

namespace hyle::compute::apple::detail {

struct PreparedBFloatProjection final {
  StorageOwner owner;
  ReadLease source;
  ConstTensorView original;
  ResourceBudget::Reservation bytes;
  ResourceBudget::Reservation private_metadata;
  std::array<mlx::core::array, 2> packed;
};

// Available only for the measured short, large dense F32-score geometry.
HYLE_COMPUTE_ROUTE_APPLE_API bool packed_bfloat_projection_eligible(
    const Operation &, const TensorDesc &activation, const TensorDesc &weights) noexcept;

// A null result leaves the original carrier selected. Preparation is scoped
// to the exact original allocation, Runtime and resource budget.
HYLE_COMPUTE_ROUTE_APPLE_API Result<std::shared_ptr<const PreparedBFloatProjection>>
prepare_bfloat_projection(const std::shared_ptr<Runtime> &, ConstTensorView,
                          const StorageOwner &, const ReadLease &,
                          const ResourceBudget &) noexcept;

Result<std::array<mlx::core::array, 2>> prepare_bfloat_projection_on_owner(
    const std::shared_ptr<Runtime> &, const mlx::core::array &) noexcept;
std::array<mlx::core::array, 2> pack_bfloat_projection(
    const mlx::core::array &, mlx::core::Stream);

HYLE_COMPUTE_ROUTE_APPLE_API mlx::core::array lower_packed_bfloat_projection(
    const PreparedBFloatProjection &, const mlx::core::array &weights,
    const mlx::core::array &activation, mlx::core::Stream);

} // namespace hyle::compute::apple::detail
