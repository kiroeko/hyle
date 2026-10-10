#pragma once

#include "hyle/compute/device.h"
#include "hyle/error/status.h"

namespace hyle::sched {
class Event;
}

namespace hyle::graph::cpu_cuda::detail {

// The Event is slot-owned and remains live until the post-callback native
// Completion terminates. Called from an existing Sched fiber: long waits park
// that fiber. The CUDA callback only signals; it never calls CUDA or frees owners.
Status wait_cuda_prefix(compute::Queue &queue, sched::Event &ready) noexcept;

} // namespace hyle::graph::cpu_cuda::detail
