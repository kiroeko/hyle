#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/platform/virtual_memory.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_MEM_API
#elif defined(HYLE_MEM_BUILD)
#  define HYLE_MEM_API HYLE_EXPORT
#else
#  define HYLE_MEM_API HYLE_IMPORT
#endif

// Large-region hugepage support (mem-06b).
//
// Goal: cut TLB pressure on multi-megabyte arenas / host tensors
// (access walk ~ N/page_size → N/large_page_size). Allocation stays O(1);
// the win is fewer page-table entries on sequential/random scans of big spans.
//
// Layers:
//   1) platform::vm_advise_hugepage — soft THP hint (Linux).
//   2) HugeRegion —  large-page-aligned reserve+commit + optional THP + prefault.
//   3) enable_allocator_hugepages — process policy for mimalloc (no mi_* leak).
//
// Fiber stacks and small objects must NOT use HugeRegion (size << 2 MiB).

namespace hyle::mem {

// ---- process-wide mimalloc policy (L3) --------------------------------------

struct AllocatorHugepagePolicy {
  // Allow transparent huge pages inside mimalloc's OS segments (Linux).
  // Default true matches mimalloc's own default (except Android).
  bool allow_thp = true;
  // Allow explicit large OS pages (2/4 MiB). Needs privileges on Windows
  // (Lock Pages in Memory) and often hugetlb setup on Linux. Default false:
  // failure must not break generic embeds.
  bool allow_large_os_pages = false;
};

// Apply policy to the process mimalloc instance. Safe to call once at startup
// before heavy allocation. Idempotent. Never throws.
HYLE_MEM_API void enable_allocator_hugepages(
    AllocatorHugepagePolicy policy = {}) noexcept;

// ---- HugeRegion (L2): explicit multi-MB span --------------------------------

enum class HugeFlags : std::uint8_t {
  kNone = 0,
  // Call vm_advise_hugepage after commit (Linux THP). Soft; ignored if unsupported.
  kThpHint = 1u << 0,
  // Touch every base page once so faults happen at allocate time (can block).
  // Folly does this so later work is not stalled on THP collapse. Optional.
  kPrefault = 1u << 1,
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

// Move-only RW region, size rounded up to large_page_size() (or page_size()).
// Backed by platform::VmRegion (no mimalloc). Suitable for big arenas / tensors.
class HugeRegion {
 public:
  HugeRegion() noexcept = default;

  // Reserve+commit `bytes` (aligned up). On failure: !valid().
  static HugeRegion allocate(std::size_t bytes,
                             HugeFlags flags = HugeFlags::kDefault) noexcept {
    HugeRegion r;
    if (bytes == 0) return r;
    const std::size_t aligned = platform::large_page_align_up(bytes);
    r.region_ = platform::VmRegion::reserve(aligned);
    if (!r.region_.valid()) return HugeRegion{};
    if (!r.region_.commit(0, r.region_.size())) {
      r.region_.release();
      return HugeRegion{};
    }
    if (any(flags & HugeFlags::kThpHint)) {
      // Soft: ignore failure (Windows/macOS, or THP=never).
      (void)platform::vm_advise_hugepage(r.region_.data(), r.region_.size());
    }
    if (any(flags & HugeFlags::kPrefault)) {
      // Touch one byte per base page — O(size/page_size) faults, intentional.
      auto* p = static_cast<volatile char*>(r.region_.data());
      const std::size_t step = platform::page_size();
      const std::size_t n = r.region_.size();
      for (std::size_t off = 0; off < n; off += step) p[off] = p[off];
    }
    return r;
  }

  HugeRegion(HugeRegion&& o) noexcept : region_(std::move(o.region_)) {}
  HugeRegion& operator=(HugeRegion&& o) noexcept {
    if (this != &o) region_ = std::move(o.region_);
    return *this;
  }
  HYLE_DISABLE_COPY(HugeRegion);

  bool valid() const noexcept { return region_.valid(); }
  explicit operator bool() const noexcept { return valid(); }

  void* data() const noexcept { return region_.data(); }
  std::size_t size() const noexcept { return region_.size(); }

  void reset() noexcept { region_.release(); }

  // Soft re-hint (e.g. after partial decommit/recommit). No-op if invalid.
  bool advise_hugepage() noexcept {
    if (!valid()) return false;
    return platform::vm_advise_hugepage(region_.data(), region_.size());
  }

 private:
  platform::VmRegion region_;
};

// Convenience: THP-hint an existing page-aligned range (e.g. mi slab). Soft.
HYLE_FORCE_INLINE bool advise_hugepage_range(void* ptr,
                                             std::size_t size) noexcept {
  return platform::vm_advise_hugepage(ptr, size);
}

// Threshold: regions at least one large page are candidates for HugeRegion.
HYLE_FORCE_INLINE std::size_t hugepage_region_threshold() noexcept {
  std::size_t lp = platform::large_page_size();
  return lp != 0 ? lp : (static_cast<std::size_t>(2u) * 1024u * 1024u);
}

}  // namespace hyle::mem
