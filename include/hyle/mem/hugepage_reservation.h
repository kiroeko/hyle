#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/mem/visibility.h"
#include "hyle/platform/numa.h"

namespace hyle::mem {

enum class HugePagePlacement : std::uint8_t {
  kUnconstrained = 0,
  // The allocator asked the OS for this preference, but its API cannot prove
  // that the policy was installed or that physical pages landed there.
  kPreferenceUnverified = 1,
};

// One observable process-global reservation attempt. A partial reservation
// cannot be rolled back because mimalloc immediately publishes the resulting
// arena to the process; callers must therefore inspect both reserved_pages and
// outcome. Invalid arguments remain a Result error because no attempt occurred.
struct HugePageReservationReport {
  std::size_t requested_pages = 0;
  std::size_t reserved_pages = 0;
  int requested_numa_node = platform::kNumaAny;
  HugePagePlacement placement = HugePagePlacement::kUnconstrained;
  Status outcome;

  bool complete() const noexcept {
    return outcome.ok() && reserved_pages == requested_pages;
  }
};

// Reserve contiguous 1 GiB OS pages for the shipping mimalloc heap. This is
// allocator-process configuration, not a HugeMapping owner operation, so it
// remains in the base Mem factor.
HYLE_MEM_API Result<HugePageReservationReport> reserve_huge_1g_pages(
    std::size_t pages, int numa_node = platform::kNumaAny,
    unsigned timeout_ms = 0) noexcept;

}  // namespace hyle::mem
