#pragma once

#include "hyle/compute/apple_weights.h"

namespace hyle::compute::apple::detail {
// Exact metadata and immutable ownership only. No MLX array is constructed and
// pending bytes are not observed; Graph orders the original producer before
// its readers. The retained state prevents selection/backing reuse.
struct WeightBinding final {
  std::shared_ptr<WeightState> owner;
  compute::ConstTensorView tensor;
  mem::AllocatorRef allocator;
};
struct WeightAccess final {
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<WeightBinding>
  binding(const WeightBatch &, std::uint64_t) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<CompletionPoll>
  poll(WeightState *) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API static Status wait(WeightState *) noexcept;
};
} // namespace hyle::compute::apple::detail
