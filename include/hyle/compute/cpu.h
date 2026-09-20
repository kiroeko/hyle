#pragma once

#include <cstdint>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/device.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_ROUTE_CPU_API
#elif defined(HYLE_COMPUTE_ROUTE_CPU_BUILD)
#define HYLE_COMPUTE_ROUTE_CPU_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_ROUTE_CPU_API HYLE_IMPORT
#endif

namespace hyle::compute::cpu {

struct DefaultContextOptions final {
  // Both backings are borrowed through all derived artifacts and invocations.
  mem::AllocatorRef metadata_allocator{};
  // Empty uses the caller thread. Native/oneDNN may split work through this
  // executor; ggml dispatches one complete program through it.
  ParallelExecutorRef executor{};
  // ggml's independent native budget, including the submitting thread. A
  // value above one explicitly prepares persistent workers per invocation.
  // The borrowed executor's concurrency never implicitly creates such a pool.
  std::uint32_t native_threads = 1;
};

// CPU Direct Compute with fixed implementation rules. Prepare
// binds eligible mature oneDNN/ggml programs or the existing Native fallback.
HYLE_COMPUTE_ROUTE_CPU_API Result<Context>
create_context(DefaultContextOptions options = {}) noexcept;

} // namespace hyle::compute::cpu
