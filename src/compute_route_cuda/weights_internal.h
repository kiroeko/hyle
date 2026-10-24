#pragma once

#include "hyle/compute/cuda_weights.h"
#include "hyle/compute/detail/queue_sequence.h"

namespace hyle::compute::cuda::detail {

HYLE_COMPUTE_ROUTE_CUDA_API void retain_weight(WeightState *state) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API void release_weight(WeightState *state) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API Status wait_weight(WeightState *state) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::Completion> order_weight(WeightState *state,
                                         compute::Queue &queue) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API Result<compute::CompletionPoll> poll_weight(WeightState *) noexcept;
// Shared Graph bindings append to an already-admitted sequence. Qualification
// and observations use the original batch's lock, never a borrowed raw cell.
HYLE_COMPUTE_ROUTE_CUDA_API Result<bool> weight_device_route(
    WeightState *, const compute::detail::QueueRef *) noexcept;
HYLE_COMPUTE_ROUTE_CUDA_API compute::detail::EnqueueResult enqueue_weight(
    WeightState *, compute::detail::QueueSequence &) noexcept;
// Only after a successful consuming prefix has actually retired, proving its
// native wait was reached. Reconcile the original cached observation status.
HYLE_COMPUTE_ROUTE_CUDA_API Status consumed_weight_status(WeightState *) noexcept;

struct WeightAccess final {
  static WeightState *state(const WeightBatch &batch) noexcept {
    return batch.state_;
  }
};

} // namespace hyle::compute::cuda::detail
