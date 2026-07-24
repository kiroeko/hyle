#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_MEM_API
#elif defined(HYLE_MEM_BUILD)
#define HYLE_MEM_API HYLE_EXPORT
#else
#define HYLE_MEM_API HYLE_IMPORT
#endif

// Large-region hugepage + NUMA (mem-06b/06c) — complete standard-library
// surface.
//
//   * TLB: large-page align + THP hint (access walk ~ N/2MiB).
//   * NUMA: preferred / bind / interleave on large spans.
//   * 1GiB OS pages: process-start reserve via mimalloc (opt-in, may need
//   privileges).
// Fiber stacks / small objects must NOT use HugeRegion.

namespace hyle::mem {

// ---- process-wide mimalloc policy -------------------------------------------

struct AllocatorHugepagePolicy {
  bool allow_thp = true;
  bool allow_large_os_pages = false;
};

HYLE_MEM_API void
enable_allocator_hugepages(AllocatorHugepagePolicy policy = {}) noexcept;

// 0 = all NUMA nodes (mimalloc default).
HYLE_MEM_API void enable_allocator_numa(unsigned max_nodes = 0) noexcept;

// Reserve `pages` contiguous 1GiB huge OS pages into mimalloc's arenas.
//   numa_node == platform::kNumaAny → interleave across nodes
//   otherwise → pin to that node
// timeout_ms: 0 = mimalloc default wait. Returns true if reservation succeeded
// (or pages==0). Soft failure (no privileges / fragmented RAM) → false; process
// remains usable with normal pages.
HYLE_MEM_API bool reserve_huge_1g_pages(std::size_t pages,
                                        int numa_node = platform::kNumaAny,
                                        unsigned timeout_ms = 0) noexcept;

// ---- HugeRegion -------------------------------------------------------------

enum class HugeFlags : std::uint8_t {
  kNone = 0,
  kThpHint = 1u << 0,
  kPrefault = 1u << 1,
  // After place: hard-bind to node (Linux MPOL_BIND). Default is preferred.
  kNumaBind = 1u << 2,
  // Stripe across all nodes (ignores numa_node). Shared multi-MB arenas.
  kNumaInterleave = 1u << 3,
  kDefault = kThpHint,
};

HYLE_FORCE_INLINE HugeFlags operator|(HugeFlags a, HugeFlags b) noexcept {
  return static_cast<HugeFlags>(static_cast<std::uint8_t>(a) |
                                static_cast<std::uint8_t>(b));
}
HYLE_FORCE_INLINE HugeFlags operator&(HugeFlags a, HugeFlags b) noexcept {
  return static_cast<HugeFlags>(static_cast<std::uint8_t>(a) &
                                static_cast<std::uint8_t>(b));
}
HYLE_FORCE_INLINE bool any(HugeFlags f) noexcept {
  return static_cast<std::uint8_t>(f) != 0;
}

class HugeRegion {
public:
  HugeRegion() noexcept = default;

  // Reserve+commit `bytes` (large-page aligned). NUMA:
  //   kNumaInterleave → stripe all nodes
  //   else reserve_numa(node) + preferred or kNumaBind
  static Result<HugeRegion> allocate(
      std::size_t bytes, HugeFlags flags = HugeFlags::kDefault,
      int numa_node = platform::kNumaAny) noexcept {
    HugeRegion r;
    r.requested_flags_ = flags;
    r.requested_numa_node_ = numa_node;
    if (bytes == 0) {
      return std::unexpected(
          Status::InvalidArgument("HugeRegion allocate: size is zero"));
    }
    const std::size_t aligned = platform::large_page_align_up(bytes);
    if (aligned == 0) {
      return std::unexpected(
          Status::OutOfRange("HugeRegion allocate: size overflow"));
    }
    const bool interleave = any(flags & HugeFlags::kNumaInterleave);
    if (!interleave && numa_node != platform::kNumaAny &&
        (numa_node < 0 ||
         static_cast<unsigned>(numa_node) >= platform::numa_node_count())) {
      return std::unexpected(Status::OutOfRange(
          "HugeRegion allocate: NUMA node index is out of range"));
    }
    if (interleave) {
      HYLE_ASSIGN_OR_RETURN(r.region_, platform::VmRegion::reserve(aligned));
    } else {
      HYLE_ASSIGN_OR_RETURN(r.region_,
                            platform::VmRegion::reserve_numa(aligned, numa_node));
    }
    Status status = r.region_.commit(0, r.region_.size());
    if (!status.ok()) {
      (void)r.region_.release();
      return std::unexpected(
          std::move(status).WithContext("HugeRegion allocate commit"));
    }
    platform::NumaPlacement placement = r.region_.numa_placement();
    if (interleave) {
      // Interleave and hard bind are requested guarantees, unlike preferred
      // placement. Do not return a valid region while silently violating them.
      auto placed = platform::vm_bind_numa_interleave(r.region_.data(),
                                                       r.region_.size());
      if (!placed) {
        Status error = std::move(placed).error();
        (void)r.region_.release();
        return std::unexpected(
            std::move(error).WithContext("HugeRegion allocate interleave"));
      }
      placement = *placed;
      r.accepted_flags_ = r.accepted_flags_ | HugeFlags::kNumaInterleave;
    } else if (numa_node != platform::kNumaAny) {
      const auto pol = any(flags & HugeFlags::kNumaBind)
                           ? platform::NumaPolicy::kBind
                           : platform::NumaPolicy::kPreferred;
      // reserve_numa only requests preferred placement. A hard bind must still
      // go through the hard-policy API; Windows correctly rejects this on a
      // multi-node machine because VirtualAlloc2 preference is not a guarantee.
      if (pol == platform::NumaPolicy::kBind) {
        auto placed = platform::vm_bind_numa(
            r.region_.data(), r.region_.size(), numa_node, pol);
        if (!placed) {
          Status error = std::move(placed).error();
          (void)r.region_.release();
          return std::unexpected(
              std::move(error).WithContext("HugeRegion allocate hard bind"));
        }
        placement = *placed;
      }
      if (pol == platform::NumaPolicy::kBind) {
        r.accepted_flags_ = r.accepted_flags_ | HugeFlags::kNumaBind;
      }
    }
    r.numa_placement_applied_ =
        placement == platform::NumaPlacement::kApplied;
    if (any(flags & HugeFlags::kThpHint)) {
      status = platform::vm_advise_hugepage(r.region_.data(), r.region_.size());
      if (status.ok())
        r.accepted_flags_ = r.accepted_flags_ | HugeFlags::kThpHint;
    }
    if (any(flags & HugeFlags::kPrefault)) {
      auto *p = static_cast<volatile char *>(r.region_.data());
      const std::size_t step = platform::page_size();
      const std::size_t n = r.region_.size();
      for (std::size_t off = 0; off < n; off += step)
        p[off] = p[off];
      r.accepted_flags_ = r.accepted_flags_ | HugeFlags::kPrefault;
    }
    return r;
  }

  static Result<HugeRegion>
  allocate_local(std::size_t bytes,
                 HugeFlags flags = HugeFlags::kDefault) noexcept {
    return allocate(bytes, flags,
                    static_cast<int>(platform::numa_current_node()));
  }

  // Shared multi-MB arena: interleave + THP.
  static Result<HugeRegion>
  allocate_interleaved(std::size_t bytes,
                       HugeFlags flags = HugeFlags::kThpHint) noexcept {
    return allocate(bytes, flags | HugeFlags::kNumaInterleave,
                    platform::kNumaAny);
  }

  HugeRegion(HugeRegion &&o) noexcept
      : region_(std::move(o.region_)), requested_flags_(o.requested_flags_),
        accepted_flags_(o.accepted_flags_),
        requested_numa_node_(o.requested_numa_node_),
        numa_placement_applied_(o.numa_placement_applied_) {
    o.accepted_flags_ = HugeFlags::kNone;
    o.requested_flags_ = HugeFlags::kNone;
    o.requested_numa_node_ = platform::kNumaAny;
    o.numa_placement_applied_ = false;
  }
  HugeRegion &operator=(HugeRegion &&o) noexcept {
    if (this != &o) {
      region_ = std::move(o.region_);
      requested_flags_ = o.requested_flags_;
      accepted_flags_ = o.accepted_flags_;
      requested_numa_node_ = o.requested_numa_node_;
      numa_placement_applied_ = o.numa_placement_applied_;
      o.accepted_flags_ = HugeFlags::kNone;
      o.requested_flags_ = HugeFlags::kNone;
      o.requested_numa_node_ = platform::kNumaAny;
      o.numa_placement_applied_ = false;
    }
    return *this;
  }
  HYLE_DISABLE_COPY(HugeRegion);

  bool valid() const noexcept { return region_.valid(); }
  explicit operator bool() const noexcept { return valid(); }

  void *data() const noexcept { return region_.data(); }
  std::size_t size() const noexcept { return region_.size(); }

  HugeFlags requested_flags() const noexcept { return requested_flags_; }
  // Policies accepted by the OS-facing calls. For THP this means the advice
  // call was accepted, not that the kernel has already promoted every physical
  // page.
  HugeFlags accepted_flags() const noexcept { return accepted_flags_; }
  int requested_numa_node() const noexcept { return requested_numa_node_; }
  bool numa_placement_applied() const noexcept {
    return numa_placement_applied_;
  }

  Status reset() noexcept {
    Status status = region_.release();
    if (!status.ok()) return status;
    accepted_flags_ = HugeFlags::kNone;
    numa_placement_applied_ = false;
    return Status::Ok();
  }

  Status advise_hugepage() noexcept {
    if (!valid()) {
      return Status::FailedPrecondition("HugeRegion advise: region is empty");
    }
    Status status =
        platform::vm_advise_hugepage(region_.data(), region_.size());
    if (status.ok())
      accepted_flags_ = accepted_flags_ | HugeFlags::kThpHint;
    return status;
  }

private:
  platform::VmRegion region_;
  HugeFlags requested_flags_ = HugeFlags::kNone;
  HugeFlags accepted_flags_ = HugeFlags::kNone;
  int requested_numa_node_ = platform::kNumaAny;
  bool numa_placement_applied_ = false;
};

HYLE_FORCE_INLINE Status advise_hugepage_range(void *ptr,
                                               std::size_t size) noexcept {
  return platform::vm_advise_hugepage(ptr, size);
}

HYLE_FORCE_INLINE std::size_t hugepage_region_threshold() noexcept {
  std::size_t lp = platform::large_page_size();
  return lp != 0 ? lp : (static_cast<std::size_t>(2u) * 1024u * 1024u);
}

} // namespace hyle::mem
