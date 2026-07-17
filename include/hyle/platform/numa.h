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

// Host NUMA topology + soft placement (mem-06c).
//
// Soft preferred placement only — never hard-bind by default. Single-node
// machines report count==1 and current==0; all APIs succeed as no-ops where
// binding is meaningless. No libnuma / hwloc dependency (syscalls + Win API).
//
// Not device/VRAM. Not a MemorySpace. Pair with worker CPU affinity (sched) so
// large host buffers allocated on a pinned worker land on its home node.

namespace hyle::platform {

// "No NUMA constraint" for reserve/bind APIs.
inline constexpr int kNumaAny = -1;

// Number of NUMA nodes with memory/CPUs, always >= 1.
HYLE_PLATFORM_API unsigned numa_node_count() noexcept;

// Home node of the calling thread (0 on failure / single-node / unsupported).
// Always < numa_node_count().
HYLE_PLATFORM_API unsigned numa_current_node() noexcept;

// Soft-bind an existing page-aligned mapping to `node` (preferred).
// node == kNumaAny -> no-op true. Invalid node (>= count) -> false.
// Linux: mbind(MPOL_PREFERRED). Windows: best-effort (often true no-op after
// plain VirtualAlloc — prefer vm_reserve_numa at allocate time on Win).
// Soft: false on failure, never traps. Empty range -> true.
HYLE_PLATFORM_API bool vm_bind_numa(void* ptr, std::size_t size, int node) noexcept;

// Reserve `size` bytes (page-rounded) preferring `node`. node == kNumaAny
// equals vm_reserve. On failure nullptr (see vm_last_error). Caller commits /
// releases with the usual vm_* APIs (same ownership as vm_reserve).
HYLE_PLATFORM_API void* vm_reserve_numa(std::size_t size, int node) noexcept;

}  // namespace hyle::platform
