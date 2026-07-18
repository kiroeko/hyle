#pragma once

#include <cstddef>
#include <cstdint>

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
// Single-node machines: count==1, current==0; bind/interleave degrade to no-ops
// that still return true. No libnuma / hwloc.

namespace hyle::platform {

// "No NUMA constraint" for reserve/bind APIs.
inline constexpr int kNumaAny = -1;

// Placement policy for vm_bind_numa / HugeRegion.
enum class NumaPolicy : std::uint8_t {
  kPreferred = 0,   // MPOL_PREFERRED (soft)
  kBind = 1,        // MPOL_BIND (hard; Linux)
  kInterleave = 2,  // MPOL_INTERLEAVE all nodes (Linux)
};

// Number of NUMA nodes with memory/CPUs, always >= 1.
HYLE_PLATFORM_API unsigned numa_node_count() noexcept;

// Home node of the calling thread (0 on failure / single-node / unsupported).
// Always < numa_node_count().
HYLE_PLATFORM_API unsigned numa_current_node() noexcept;

// NUMA node owning logical CPU `cpu` (0 on failure / single-node / unsupported /
// out-of-range cpu). Always < numa_node_count(). Lets a caller that pinned a
// worker to a core (ThreadOpts::affinity_cpu) place that worker's arena on the
// core's node via vm_reserve_numa. Not a hot path (a /sys walk on Linux).
HYLE_PLATFORM_API unsigned numa_node_of_cpu(unsigned cpu) noexcept;

// Bind an existing page-aligned mapping.
//   kPreferred + node: soft home (node==kNumaAny -> true no-op)
//   kBind + node:      hard home; invalid node -> false
//   kInterleave:       ignore node; stripe all nodes (Linux); soft true elsewhere
// Empty range -> true. Soft errors never trap.
HYLE_PLATFORM_API bool vm_bind_numa(void* ptr, std::size_t size, int node,
                                    NumaPolicy policy = NumaPolicy::kPreferred) noexcept;

// Stripe [ptr,ptr+size) across all NUMA nodes (alias for Interleave).
HYLE_PLATFORM_API bool vm_bind_numa_interleave(void* ptr, std::size_t size) noexcept;

// Reserve `size` bytes (page-rounded) preferring `node`. node == kNumaAny
// equals vm_reserve. On failure nullptr (see vm_last_error). Caller commits /
// releases with the usual vm_* APIs (same ownership as vm_reserve).
HYLE_PLATFORM_API void* vm_reserve_numa(std::size_t size, int node) noexcept;

}  // namespace hyle::platform
