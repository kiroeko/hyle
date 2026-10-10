#pragma once

#include "hyle/compute/cuda_weights.h"

namespace hyle::compute::cuda::detail {

HYLE_COMPUTE_ROUTE_CUDA_API void retain_weight(WeightState *state) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API void release_weight(WeightState *state) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API Status wait_weight(WeightState *state) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::Completion> order_weight(WeightState *state,
                                         compute::Queue &queue) noexcept;

struct WeightAccess final {
  static WeightState *state(const WeightBatch &batch) noexcept {
    return batch.state_;
  }
};

} // namespace hyle::compute::cuda::detail
