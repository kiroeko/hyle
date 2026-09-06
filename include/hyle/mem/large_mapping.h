#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/mem/visibility.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

// Large anonymous regions with explicit THP/prefault/NUMA policy.
//
//   * TLB: large-page align + THP hint (access walk ~ N/2MiB).
//   * NUMA: preferred / bind / interleave on large spans.
//   * Process-global allocator arena: reserve via
//     reserve_large_page_arena() (opt-in, may need privileges; GiB request
//     chunks do not prove the resulting OS page granularity).
// Fiber stacks / small objects must NOT use LargeMapping.

namespace hyle::mem {

// ---- LargeMapping ----------------------------------------------------------

enum class LargeMappingFlags : std::uint8_t {
  kNone = 0,
  kThpHint = 1u << 0,
  kPrefault = 1u << 1,
  // After place: hard-bind to node (Linux MPOL_BIND). Default is preferred.
  kNumaBind = 1u << 2,
  // Stripe across all nodes (ignores numa_node). Shared multi-MB arenas.
  kNumaInterleave = 1u << 3,
  kDefault = kThpHint,
};

HYLE_FORCE_INLINE LargeMappingFlags operator|(LargeMappingFlags a,
                                               LargeMappingFlags b) noexcept {
  return static_cast<LargeMappingFlags>(static_cast<std::uint8_t>(a) |
                                        static_cast<std::uint8_t>(b));
}
HYLE_FORCE_INLINE LargeMappingFlags operator&(LargeMappingFlags a,
                                               LargeMappingFlags b) noexcept {
  return static_cast<LargeMappingFlags>(static_cast<std::uint8_t>(a) &
                                        static_cast<std::uint8_t>(b));
}
HYLE_FORCE_INLINE bool any(LargeMappingFlags f) noexcept {
  return static_cast<std::uint8_t>(f) != 0;
}

HYLE_FORCE_INLINE constexpr bool valid_large_mapping_flags(
    LargeMappingFlags flags) noexcept {
  constexpr auto known =
      static_cast<std::uint8_t>(LargeMappingFlags::kThpHint) |
      static_cast<std::uint8_t>(LargeMappingFlags::kPrefault) |
      static_cast<std::uint8_t>(LargeMappingFlags::kNumaBind) |
      static_cast<std::uint8_t>(LargeMappingFlags::kNumaInterleave);
  const auto bits = static_cast<std::uint8_t>(flags);
  const auto placement =
      static_cast<std::uint8_t>(LargeMappingFlags::kNumaBind) |
      static_cast<std::uint8_t>(LargeMappingFlags::kNumaInterleave);
  return (bits & ~known) == 0 && (bits & placement) != placement;
}

class LargeMapping {
public:
  LargeMapping() noexcept = default;

  // Reserve+commit `bytes` (large-page aligned). NUMA:
  //   kNumaInterleave → stripe all nodes
  //   else reserve_numa(node) + preferred or kNumaBind
  static Result<LargeMapping> allocate(
      std::size_t bytes, LargeMappingFlags flags = LargeMappingFlags::kDefault,
      int numa_node = platform::kNumaAny) noexcept {
    LargeMapping r;
    r.requested_flags_ = flags;
    r.requested_numa_node_ = numa_node;
    if (!valid_large_mapping_flags(flags)) {
      return std::unexpected(Status::InvalidArgument(
          "LargeMapping allocate: unknown or conflicting flags"));
    }
    if (bytes == 0) {
      return std::unexpected(
          Status::InvalidArgument("LargeMapping allocate: size is zero"));
    }
    const std::size_t aligned = platform::large_page_align_up(bytes);
    if (aligned == 0) {
      return std::unexpected(
          Status::OutOfRange("LargeMapping allocate: size overflow"));
    }
    const bool interleave = any(flags & LargeMappingFlags::kNumaInterleave);
    if (any(flags & LargeMappingFlags::kNumaBind) &&
        numa_node == platform::kNumaAny) {
      return std::unexpected(Status::InvalidArgument(
          "LargeMapping allocate: hard bind requires an explicit NUMA node"));
    }
    if (interleave && numa_node != platform::kNumaAny) {
      return std::unexpected(Status::InvalidArgument(
          "LargeMapping allocate: interleave requires NUMA-any"));
    }
    if (!interleave && numa_node != platform::kNumaAny &&
        (numa_node < 0 ||
         !platform::numa_node_available(static_cast<unsigned>(numa_node)))) {
      return std::unexpected(Status::OutOfRange(
          "LargeMapping allocate: NUMA node ID is unavailable"));
    }
    if (interleave) {
      HYLE_ASSIGN_OR_RETURN(
          r.region_,
          platform::VmRegion::allocate_numa_interleaved(aligned));
      r.interleave_chunk_size_ = platform::page_size();
    } else {
      HYLE_ASSIGN_OR_RETURN(r.region_,
                            platform::VmRegion::reserve_numa(aligned, numa_node));
      Status status = r.region_.commit(0, r.region_.size());
      if (!status.ok()) {
        r.region_.release().IgnoreError();
        return std::unexpected(
            std::move(status).WithContext("LargeMapping allocate commit"));
      }
    }
    platform::NumaPlacement placement = r.region_.numa_placement();
    if (interleave) {
      r.accepted_flags_ = r.accepted_flags_ | LargeMappingFlags::kNumaInterleave;
    } else if (numa_node != platform::kNumaAny) {
      const auto pol = any(flags & LargeMappingFlags::kNumaBind)
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
          r.region_.release().IgnoreError();
          return std::unexpected(
              std::move(error).WithContext("LargeMapping allocate hard bind"));
        }
        placement = *placed;
      }
      if (pol == platform::NumaPolicy::kBind) {
        r.accepted_flags_ = r.accepted_flags_ | LargeMappingFlags::kNumaBind;
      }
    }
    r.numa_placement_applied_ =
        placement == platform::NumaPlacement::kApplied;
    if (any(flags & LargeMappingFlags::kThpHint)) {
      Status status =
          platform::vm_advise_hugepage(r.region_.data(), r.region_.size());
      if (status.ok())
        r.accepted_flags_ = r.accepted_flags_ | LargeMappingFlags::kThpHint;
    }
    if (any(flags & LargeMappingFlags::kPrefault)) {
      auto *p = static_cast<volatile char *>(r.region_.data());
      const std::size_t step = platform::page_size();
      const std::size_t n = r.region_.size();
      for (std::size_t off = 0; off < n; off += step)
        p[off] = p[off];
      r.accepted_flags_ = r.accepted_flags_ | LargeMappingFlags::kPrefault;
    }
    return r;
  }

  static Result<LargeMapping> allocate_local(
      std::size_t bytes,
      LargeMappingFlags flags = LargeMappingFlags::kDefault) noexcept {
    if (any(flags & LargeMappingFlags::kNumaInterleave)) {
      return std::unexpected(Status::InvalidArgument(
          "LargeMapping allocate_local: interleave is not local placement"));
    }
    auto current = platform::numa_current_node();
    if (!current) {
      return std::unexpected(std::move(current).error().WithContext(
          "LargeMapping allocate_local current NUMA node"));
    }
    if (*current > static_cast<unsigned>((std::numeric_limits<int>::max)())) {
      return std::unexpected(Status::OutOfRange(
          "LargeMapping allocate_local: NUMA node does not fit int"));
    }
    return allocate(bytes, flags, static_cast<int>(*current));
  }

  // Shared multi-MB arena: interleave + THP.
  static Result<LargeMapping> allocate_interleaved(
      std::size_t bytes,
      LargeMappingFlags flags = LargeMappingFlags::kThpHint) noexcept {
    return allocate(bytes, flags | LargeMappingFlags::kNumaInterleave,
                    platform::kNumaAny);
  }

  // Interleave over an explicit node set. Stripes follow `nodes` order and
  // repeated IDs express weighted placement; an empty span means every
  // available node. bytes_per_chunk==0 means one OS page. The returned region
  // is already committed and writable.
  static Result<LargeMapping> allocate_interleaved(
      std::size_t bytes, std::span<const unsigned> nodes,
      std::size_t bytes_per_chunk = 0,
      LargeMappingFlags flags = LargeMappingFlags::kNone) noexcept {
    flags = flags | LargeMappingFlags::kNumaInterleave;
    if (!valid_large_mapping_flags(flags)) {
      return std::unexpected(Status::InvalidArgument(
          "LargeMapping allocate interleaved: unknown or conflicting flags"));
    }
    if (bytes == 0) {
      return std::unexpected(Status::InvalidArgument(
          "LargeMapping allocate interleaved: size is zero"));
    }
    const std::size_t aligned = platform::large_page_align_up(bytes);
    if (aligned == 0) {
      return std::unexpected(
          Status::OutOfRange("LargeMapping allocate interleaved: size overflow"));
    }

    LargeMapping r;
    r.requested_flags_ = flags;
    r.requested_numa_node_ = platform::kNumaAny;
    HYLE_ASSIGN_OR_RETURN(
        r.region_, platform::VmRegion::allocate_numa_interleaved(
                       aligned, nodes, bytes_per_chunk));
    r.interleave_chunk_size_ =
        bytes_per_chunk == 0 ? platform::page_size() : bytes_per_chunk;
    r.accepted_flags_ = r.accepted_flags_ | LargeMappingFlags::kNumaInterleave;
    r.numa_placement_applied_ =
        r.region_.numa_placement() == platform::NumaPlacement::kApplied;

    if (any(flags & LargeMappingFlags::kThpHint)) {
      Status status =
          platform::vm_advise_hugepage(r.region_.data(), r.region_.size());
      if (status.ok()) {
        r.accepted_flags_ = r.accepted_flags_ | LargeMappingFlags::kThpHint;
      }
    }
    if (any(flags & LargeMappingFlags::kPrefault)) {
      auto* p = static_cast<volatile char*>(r.region_.data());
      const std::size_t step = platform::page_size();
      for (std::size_t off = 0; off < r.region_.size(); off += step) {
        p[off] = p[off];
      }
      r.accepted_flags_ = r.accepted_flags_ | LargeMappingFlags::kPrefault;
    }
    return r;
  }

  LargeMapping(LargeMapping&& o) noexcept
      : region_(std::move(o.region_)), requested_flags_(o.requested_flags_),
        accepted_flags_(o.accepted_flags_),
        requested_numa_node_(o.requested_numa_node_),
        interleave_chunk_size_(o.interleave_chunk_size_),
        numa_placement_applied_(o.numa_placement_applied_) {
    o.accepted_flags_ = LargeMappingFlags::kNone;
    o.requested_flags_ = LargeMappingFlags::kNone;
    o.requested_numa_node_ = platform::kNumaAny;
    o.interleave_chunk_size_ = 0;
    o.numa_placement_applied_ = false;
  }
  LargeMapping& operator=(LargeMapping&& o) noexcept {
    if (this != &o) {
      region_ = std::move(o.region_);
      requested_flags_ = o.requested_flags_;
      accepted_flags_ = o.accepted_flags_;
      requested_numa_node_ = o.requested_numa_node_;
      interleave_chunk_size_ = o.interleave_chunk_size_;
      numa_placement_applied_ = o.numa_placement_applied_;
      o.accepted_flags_ = LargeMappingFlags::kNone;
      o.requested_flags_ = LargeMappingFlags::kNone;
      o.requested_numa_node_ = platform::kNumaAny;
      o.interleave_chunk_size_ = 0;
      o.numa_placement_applied_ = false;
    }
    return *this;
  }
  LargeMapping(const LargeMapping&) = delete;
  LargeMapping& operator=(const LargeMapping&) = delete;

  bool valid() const noexcept { return region_.valid(); }
  explicit operator bool() const noexcept { return valid(); }

  void* data() const noexcept { return region_.data(); }
  std::size_t size() const noexcept { return region_.size(); }

  LargeMappingFlags requested_flags() const noexcept { return requested_flags_; }
  // Policies accepted by the OS-facing calls. For THP this means the advice
  // call was accepted, not that the kernel has already promoted every physical
  // page.
  LargeMappingFlags accepted_flags() const noexcept { return accepted_flags_; }
  int requested_numa_node() const noexcept { return requested_numa_node_; }
  // Effective stripe size for an interleaved region; 0 otherwise.
  std::size_t interleave_chunk_size() const noexcept {
    return interleave_chunk_size_;
  }
  bool numa_placement_applied() const noexcept {
    return numa_placement_applied_;
  }

  Status reset() noexcept {
    Status status = region_.release();
    if (!status.ok()) return status;
    accepted_flags_ = LargeMappingFlags::kNone;
    requested_flags_ = LargeMappingFlags::kNone;
    requested_numa_node_ = platform::kNumaAny;
    interleave_chunk_size_ = 0;
    numa_placement_applied_ = false;
    return Status::Ok();
  }

  Status advise_hugepage() noexcept {
    if (!valid()) {
      return Status::FailedPrecondition("LargeMapping advise: region is empty");
    }
    requested_flags_ = requested_flags_ | LargeMappingFlags::kThpHint;
    Status status =
        platform::vm_advise_hugepage(region_.data(), region_.size());
    if (status.ok())
      accepted_flags_ = accepted_flags_ | LargeMappingFlags::kThpHint;
    return status;
  }

 private:
  platform::VmRegion region_;
  LargeMappingFlags requested_flags_ = LargeMappingFlags::kNone;
  LargeMappingFlags accepted_flags_ = LargeMappingFlags::kNone;
  int requested_numa_node_ = platform::kNumaAny;
  std::size_t interleave_chunk_size_ = 0;
  bool numa_placement_applied_ = false;
};

}  // namespace hyle::mem
