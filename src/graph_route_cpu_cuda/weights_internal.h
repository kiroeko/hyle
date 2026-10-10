#pragma once

#include "hyle/graph/cpu_cuda.h"

namespace hyle::graph::cpu_cuda::detail {

void retain_weight(WeightState *state) noexcept;
void release_weight(WeightState *state) noexcept;
Status wait_weight(WeightState *state) noexcept;
Result<compute::Completion> order_weight(WeightState *state,
                                         compute::Queue &queue) noexcept;

struct WeightAccess final {
  static WeightState *state(const WeightBatch &batch) noexcept {
    return batch.state_;
  }
};

} // namespace hyle::graph::cpu_cuda::detail
