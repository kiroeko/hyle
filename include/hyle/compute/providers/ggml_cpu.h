#pragma once

#include <cstdint>

#include "hyle/compute/provider.h"
#include "hyle/visibility.h"

namespace hyle::compute::ggml_cpu {

inline constexpr ProviderId kProviderId{0x48594c4500000001ULL,
                                        0x4747435055000001ULL};

struct Options final {
  // Borrowed through all derived objects. Dispatches one complete native
  // program; its concurrency does not configure ggml's internal workers.
  ParallelExecutorRef executor{};
  // Includes the submitting thread. One provider Context lazily owns a shared
  // native team; its complete computes are serialized. One executes on the
  // submitting thread without creating another worker.
  std::uint32_t native_threads = 1;
};

// Original ggml CPU programs and qualified standard index algorithms.
// Preparation owns each program's resources; replay updates bindings and
// synchronously completes the fixed program.
// Invocation graph/workspace state is independent; one invocation cannot run
// or close concurrently. Programs and slots retain their Context native team.
HYLE_COMPUTE_PROVIDER_GGML_API Provider provider(Options options = {}) noexcept;

} // namespace hyle::compute::ggml_cpu
