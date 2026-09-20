#pragma once

#include <cstdint>

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_COMPUTE_PROVIDER_GGML_API
#elif defined(HYLE_COMPUTE_PROVIDER_GGML_BUILD)
#define HYLE_COMPUTE_PROVIDER_GGML_API HYLE_EXPORT
#else
#define HYLE_COMPUTE_PROVIDER_GGML_API HYLE_IMPORT
#endif

namespace hyle::compute::ggml_cpu {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4747435055000001ULL};

struct Options final {
  // Borrowed through all derived objects. Dispatches one complete native
  // program; its concurrency does not configure ggml's internal workers.
  ParallelExecutorRef executor{};
  // Includes the submitting thread. Each invocation prepares its own pool;
  // one executes on the submitting thread without creating another worker.
  std::uint32_t native_threads = 1;
};

// Original ggml CPU programs. Preparation owns graph/plan/workspace; replay
// updates bindings and synchronously completes the fixed native program.
// Invocations are independent; one invocation cannot run or close concurrently.
#if defined(HYLE_COMPUTE_GGML_CPU)
HYLE_COMPUTE_PROVIDER_GGML_API Provider provider(Options options = {}) noexcept;
#endif

} // namespace hyle::compute::ggml_cpu
