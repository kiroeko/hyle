#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/error/result.h"
#include "hyle/core/compiler.h"
#include "hyle/platform/visibility.h"

// Host NUMA topology + placement (mem-06c).
//
// Full placement modes for a standard-library surface (not "wait for a consumer"):
//   Preferred — soft home node (default; never fails the process)
//   Bind      — hard node (Linux MPOL_BIND; may fail if node is tight)
//   Interleave— stripe across all nodes (shared large arenas)
//
// Single-node machines: count==1, current is the sole actual node ID;
// bind/interleave return kNotNeeded. No libnuma / hwloc.

namespace hyle::platform {

// "No NUMA constraint" for reserve/bind APIs.
inline constexpr int kNumaAny = -1;

// Placement policy for vm_bind_numa / HugeMapping.
enum class NumaPolicy : std::uint8_t {
  kPreferred = 0,   // MPOL_PREFERRED (soft)
  kBind = 1,        // MPOL_BIND (hard; Linux)
  kInterleave = 2,  // MPOL_INTERLEAVE all nodes (Linux)
};

// Whether the OS accepted a NUMA policy request. kApplied records policy
// installation, not a residency proof: preferred/interleave policies may use
// another node under memory pressure. Linux kBind is the strict-node policy.
// Preferred reservation may also fall back before installing a policy while
// leaving a fully usable allocation.
enum class NumaPlacement : std::uint8_t {
  kApplied = 0,
  kNotNeeded = 1,   // ANY, empty range, or a single-node machine
  kUnavailable = 2  // soft preferred fallback; allocation remains usable
};

// Native NUMA operations available to this process. A capability describes
// the OS primitive, not whether using it is necessary on the current machine:
// a single-node host can support an operation that returns kNotNeeded.
enum class NumaCapability : std::uint32_t {
  kNone = 0,
  kTopology = 1u << 0,
  kPreferredReservation = 1u << 1,
  kBindExisting = 1u << 2,
  kInterleaveExisting = 1u << 3,
  kSelectedInterleave = 1u << 4,
};

HYLE_FORCE_INLINE constexpr NumaCapability operator|(NumaCapability a,
                                                       NumaCapability b) noexcept {
  return static_cast<NumaCapability>(static_cast<std::uint32_t>(a) |
                                     static_cast<std::uint32_t>(b));
}
HYLE_FORCE_INLINE constexpr NumaCapability operator&(NumaCapability a,
                                                       NumaCapability b) noexcept {
  return static_cast<NumaCapability>(static_cast<std::uint32_t>(a) &
                                     static_cast<std::uint32_t>(b));
}
HYLE_FORCE_INLINE constexpr bool has_numa_capability(
    NumaCapability capabilities, NumaCapability capability) noexcept {
  return static_cast<std::uint32_t>(capabilities & capability) != 0;
}

struct NumaReservation {
  void* data = nullptr;
  NumaPlacement placement = NumaPlacement::kNotNeeded;
};

// A committed read-write allocation. `size` is the page-rounded release size;
// pair `data,size` with vm_release, or prefer VmRegion::allocate_numa_interleaved.
// Do not decommit/recommit subranges: that can discard the requested placement.
struct NumaAllocation {
  void* data = nullptr;
  std::size_t size = 0;
  NumaPlacement placement = NumaPlacement::kNotNeeded;
};

// Native feature discovery. This is the preferred runtime boundary; public
// callers normally should not branch on HYLE_OS_* themselves.
HYLE_PLATFORM_API NumaCapability numa_capabilities() noexcept;

// Stable process-lifetime list of actual NUMA node IDs. IDs are not required
// to be dense. Unsupported/single-node hosts expose the logical node {0}.
HYLE_PLATFORM_API std::span<const unsigned> numa_nodes() noexcept;

// True iff `node` is present in numa_nodes().
HYLE_PLATFORM_API bool numa_node_available(unsigned node) noexcept;

// Number of NUMA nodes with memory/CPUs, always >= 1.
HYLE_PLATFORM_API unsigned numa_node_count() noexcept;

// Home node of the calling thread. A single logical-node host returns that node
// without an OS query. On a real multi-node host, discovery failures remain a
// Status instead of silently selecting a different placement.
HYLE_PLATFORM_API Result<unsigned> numa_current_node() noexcept;

// NUMA node owning process-available logical CPU `cpu`. An unavailable CPU is
// OutOfRange; a topology-query failure remains an error. Lets a caller that
// pinned a worker to a core (ThreadOpts::affinity) place that worker's arena on
// the core's node via vm_reserve_numa. Not a hot path (a /sys walk on Linux).
HYLE_PLATFORM_API Result<unsigned> numa_node_of_cpu(unsigned cpu) noexcept;

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

// Allocate one contiguous, committed RW span striped over selected NUMA nodes.
// An empty `nodes` span selects all nodes. Node IDs must be available; stripes
// follow the supplied order and repeated IDs express weighted placement.
// bytes_per_chunk is 0 for one OS page, otherwise it must be a page-size
// multiple. Validation is O(nodes). Setup is O(size/bytes_per_chunk) where the
// OS lacks a whole-range policy (a single node and Linux's ordinary page stripe
// are O(1) in allocation size). Ordinary memory access has no HYLE dispatch or
// metadata cost.
//
// Linux normally applies lazy mbind policies before first touch. For a very
// fine custom pattern, retaining one VMA policy per stripe is unsafe; HYLE
// instead performs O(pages) placement with fixed-size move batches.
// Windows reserves one address range and commits each stripe with a preferred
// node when VirtualAllocExNuma is available. VirtualAlloc2 alone cannot satisfy
// the one-vm_release ownership contract for replaced placeholder stripes, so it
// does not advertise this capability. Interleave/preference is a placement
// policy, not a physical-residency proof; the OS may fall back under pressure.
// A platform without a multi-node primitive returns Unimplemented rather than
// silently omitting the policy.
// Single-node placement is a normal allocation with kNotNeeded.
HYLE_PLATFORM_API Result<NumaAllocation> vm_allocate_numa_interleaved(
    std::size_t size, std::span<const unsigned> nodes = {},
    std::size_t bytes_per_chunk = 0) noexcept;

}  // namespace hyle::platform
