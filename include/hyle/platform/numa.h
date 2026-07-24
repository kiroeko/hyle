#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_API
#elif defined(HYLE_PLATFORM_BUILD)
#  define HYLE_PLATFORM_API HYLE_EXPORT
#else
#  define HYLE_PLATFORM_API HYLE_IMPORT
#endif

// Host NUMA topology + placement (mem-06c).
//
// Full placement modes for a standard-library surface (not "wait for a consumer"):
//   Preferred — soft home node (default; never fails the process)
//   Bind      — hard node (Linux MPOL_BIND; may fail if node is tight)
//   Interleave— stripe across all nodes (shared large arenas)
//
// Single-node machines: count==1, current==0; bind/interleave return
// kNotNeeded. No libnuma / hwloc.

namespace hyle::platform {

// "No NUMA constraint" for reserve/bind APIs.
inline constexpr int kNumaAny = -1;

// Placement policy for vm_bind_numa / HugeRegion.
enum class NumaPolicy : std::uint8_t {
  kPreferred = 0,   // MPOL_PREFERRED (soft)
  kBind = 1,        // MPOL_BIND (hard; Linux)
  kInterleave = 2,  // MPOL_INTERLEAVE all nodes (Linux)
};

// Whether a successful NUMA request changed placement. Preferred placement may
// legitimately fall back while leaving a fully usable allocation; callers must
// not confuse that with an applied policy.
enum class NumaPlacement : std::uint8_t {
  kApplied = 0,
  kNotNeeded = 1,   // ANY, empty range, or a single-node machine
  kUnavailable = 2  // soft preferred fallback; allocation remains usable
};

struct NumaReservation {
  void* data = nullptr;
  NumaPlacement placement = NumaPlacement::kNotNeeded;
};

// Number of NUMA nodes with memory/CPUs, always >= 1.
HYLE_PLATFORM_API unsigned numa_node_count() noexcept;

// Home node of the calling thread (0 on failure / single-node / unsupported).
// Always < numa_node_count().
HYLE_PLATFORM_API unsigned numa_current_node() noexcept;

// NUMA node owning logical CPU `cpu` (0 on failure / single-node / unsupported /
// out-of-range cpu). Always < numa_node_count(). Lets a caller that pinned a
// worker to a core (ThreadOpts::affinity) place that worker's arena on the
// core's node via vm_reserve_numa. Not a hot path (a /sys walk on Linux).
HYLE_PLATFORM_API unsigned numa_node_of_cpu(unsigned cpu) noexcept;

// Bind an existing page-aligned mapping.
//   kPreferred + node: soft home (node==kNumaAny -> kNotNeeded)
//   kBind + node:      hard home; invalid/unsupported -> Status error
//   kInterleave:       ignore node; stripe all nodes; unsupported -> error
// Empty range -> kNotNeeded. Soft preferred fallback -> kUnavailable.
HYLE_PLATFORM_API Result<NumaPlacement> vm_bind_numa(
    void* ptr, std::size_t size, int node,
    NumaPolicy policy = NumaPolicy::kPreferred) noexcept;

// Stripe [ptr,ptr+size) across all NUMA nodes (alias for Interleave).
HYLE_PLATFORM_API Result<NumaPlacement> vm_bind_numa_interleave(
    void* ptr, std::size_t size) noexcept;

// Reserve `size` bytes (page-rounded) preferring `node`. node == kNumaAny
// equals vm_reserve. The returned placement distinguishes applied placement
// from a soft preferred fallback. Caller commits/releases `data` with vm_*.
HYLE_PLATFORM_API Result<NumaReservation> vm_reserve_numa(
    std::size_t size, int node) noexcept;

}  // namespace hyle::platform
