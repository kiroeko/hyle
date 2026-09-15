#pragma once

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
  // Empty uses the caller thread. The route never creates a worker pool.
  ParallelExecutorRef executor{};
};

// CPU Direct Compute with development-qualified implementation rules. Prepare
// binds an eligible oneDNN operation or Native implementation without timing.
HYLE_COMPUTE_ROUTE_CPU_API Result<Context>
create_context(DefaultContextOptions options = {}) noexcept;

} // namespace hyle::compute::cpu
