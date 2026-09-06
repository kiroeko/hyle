#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/mem/visibility.h"
#include "hyle/platform/numa.h"

namespace hyle::mem {

enum class LargePagePlacement : std::uint8_t {
  kUnconstrained = 0,
  // The allocator asked the OS for this preference, but its API cannot prove
  // that the policy was installed or that physical pages landed there.
  kPreferenceUnverified = 1,
};

enum class LargePageGranularity : std::uint8_t {
  // The allocator can prove that the OS large-page route accepted each GiB
  // chunk, but its public API cannot distinguish 1 GiB pages from its 2 MiB
  // fallback (or another native large-page granularity).
  kUnverified = 0,
};

// One observable process-global large-page arena reservation attempt. The
// allocator requests storage in GiB-sized chunks, but that request unit is not
// a claim about the physical OS page size. A partial reservation cannot be
// rolled back because mimalloc immediately publishes the resulting arena to
// the process; callers must inspect both reserved_gib and outcome. Invalid
// arguments remain a Result error because no attempt occurred.
struct LargePageArenaReservationReport {
  std::size_t requested_gib = 0;
  std::size_t reserved_gib = 0;
  int requested_numa_node = platform::kNumaAny;
  LargePagePlacement placement = LargePagePlacement::kUnconstrained;
  LargePageGranularity page_granularity = LargePageGranularity::kUnverified;
  Status outcome;

  bool complete() const noexcept {
    return outcome.ok() && reserved_gib == requested_gib;
  }
};

// Reserve a contiguous, pinned large-page arena for the shipping mimalloc heap.
// `gib` is the requested arena size in GiB-sized chunks; actual OS page
// granularity is deliberately reported as unverified. This is allocator-process
// configuration, not a LargeMapping owner operation, so it remains in base Mem.
HYLE_MEM_API Result<LargePageArenaReservationReport> reserve_large_page_arena(
    std::size_t gib, int numa_node = platform::kNumaAny,
    unsigned timeout_ms = 0) noexcept;

}  // namespace hyle::mem
