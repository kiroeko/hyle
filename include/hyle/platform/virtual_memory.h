#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/platform/numa.h"

// Host virtual memory: reserve/commit/decommit/release + protect.
// Real consumers include guarded fiber stacks and large allocator backing.
// Not device/VRAM, not general malloc (mimalloc), not cross-process shmem
// (io-05).
//
// Windows: VirtualAlloc MEM_RESERVE / MEM_COMMIT / MEM_DECOMMIT / MEM_RELEASE +
// VirtualProtect. POSIX: mmap PROT_NONE + mprotect + munmap (+ madvise for
// decommit).

namespace hyle::platform {

// ---- page geometry ----------------------------------------------------------

// OS page size in bytes (typically 4096). All VM ranges are page-granular.
HYLE_PLATFORM_API std::size_t page_size() noexcept;

// Large (huge) page size in bytes when the platform exposes one (commonly 2 MiB
// on x86_64/aarch64). 0 if unknown / unsupported. This is the *size unit* for
// THP / large-page alignment -- not a guarantee that every allocation gets one.
HYLE_PLATFORM_API std::size_t large_page_size() noexcept;

// Smallest multiple of page_size() that is >= n. n==0 -> 0.
HYLE_FORCE_INLINE std::size_t page_align_up(std::size_t n) noexcept {
  std::size_t p = page_size();
  if (n > SIZE_MAX - (p - 1))
    return 0;  // round-up would overflow -> 0 (caller treats as fail)
  return (n + (p - 1)) & ~(p - 1);
}

// Align up to large_page_size() when known; otherwise to page_size(). n==0 ->
// 0.
HYLE_FORCE_INLINE std::size_t large_page_align_up(std::size_t n) noexcept {
  if (n == 0) return 0;
  std::size_t lp = large_page_size();
  std::size_t p = lp != 0 ? lp : page_size();
  if (n > SIZE_MAX - (p - 1)) return 0;  // round-up would overflow -> 0
  return (n + (p - 1)) & ~(p - 1);
}

// ---- protection -------------------------------------------------------------

// Access rights for a committed (or reserved) range. kNone is the permanent
// guard-page mode (PAGE_NOACCESS / PROT_NONE) -- NOT Windows PAGE_GUARD
// (one-shot).
enum class Protect : std::uint8_t {
  kNone = 0,       // no access (overflow guard)
  kRead = 1,       // read-only
  kReadWrite = 2,  // read + write
};

// Page-range residency controls shared by anonymous and file-backed storage
// owners. Advice is a performance/residency request except kDontNeed, whose
// exact refault/writeback semantics follow the backing mapping. Unsupported
// native hints return Unimplemented rather than pretending they were applied.
enum class VmAdvice : std::uint8_t {
  kNormal = 0,
  kRandom = 1,
  kSequential = 2,
  kWillNeed = 3,
  kDontNeed = 4,
  kHugePage = 5,
  kNoHugePage = 6,
};

enum class VmLockMode : std::uint8_t {
  kEager = 0,
  kOnFault = 1,
};

enum class VmLockLimitPolicy : std::uint8_t {
  kPreserve = 0,
  kAdjustIfNeeded = 1,
};

struct VmLockOptions {
  VmLockMode mode = VmLockMode::kEager;
  VmLockLimitPolicy limit_policy = VmLockLimitPolicy::kPreserve;
};

// ---- free-function primitives -----------------------------------------------
// Thin OS wrappers. Sizes are rounded UP to page_size(). Non-empty range
// operations validate that their pointer is page-aligned and return
// InvalidArgument when it is not; size/alignment overflow is OutOfRange.
// Recoverable failures return Status/Result. OK and successful Result states do
// not allocate; native diagnostics are built only on the failure path.

// Reserve `size` bytes of address space (rounded up). Not readable/writable
// until vm_commit. Returns the base or a structured failure.
HYLE_PLATFORM_API Result<void*> vm_reserve(std::size_t size) noexcept;

// Allocate one committed read-write mapping in a single OS operation
// (VirtualAlloc RESERVE|COMMIT / mmap PROT_READ|WRITE). Pair with vm_release,
// passing the same request size; use vm_reserve+vm_commit when staged commit is
// the desired contract.
HYLE_PLATFORM_API Result<void*> vm_allocate_rw(std::size_t size) noexcept;

// Make [ptr, ptr+size) resident and Protect::kReadWrite.
HYLE_PLATFORM_API Status vm_commit(void* ptr, std::size_t size) noexcept;

// Drop physical pages for [ptr, ptr+size). Range stays reserved; re-commit
// before touch. Access after decommit without re-commit is undefined (fault).
HYLE_PLATFORM_API Status vm_decommit(void* ptr, std::size_t size) noexcept;

// Free the WHOLE reservation that started at `base` with `size` (the values
// returned/used by vm_reserve). Windows requires releasing from the original
// base; sub-range release is not supported.
HYLE_PLATFORM_API Status vm_release(void* base, std::size_t size) noexcept;

// Change protection on [ptr, ptr+size). On POSIX, kNone on a never-committed
// reservation is already the reserve default; on Windows, reserved pages are
// PAGE_NOACCESS until commit.
HYLE_PLATFORM_API Status vm_protect(void* ptr, std::size_t size,
                                    Protect prot) noexcept;

// Apply an access/residency policy to a valid, page-aligned mapped range. The
// owner must ensure that the rounded range remains mapped and supports the
// selected advice. kNormal is a successful no-op on platforms that have no
// explicit reset hint; all other unsupported policies are observable.
HYLE_PLATFORM_API Status vm_advise(void* ptr, std::size_t size,
                                   VmAdvice advice) noexcept;

// Pin/unpin a valid mapped page-aligned range. kAdjustIfNeeded permits one
// process-global working-set/memlock-limit adjustment and retry. Callers own
// the policy implications; unlock does not roll a process limit back.
HYLE_PLATFORM_API Status vm_lock(void* ptr, std::size_t size,
                                 VmLockOptions options = {}) noexcept;
HYLE_PLATFORM_API Status vm_unlock(void* ptr, std::size_t size) noexcept;

// ---- hugepage / THP hints ---------------------------------------------------
// Soft advice only: never required for correctness. Linux uses madvise
// (MADV_HUGEPAGE / MADV_NOHUGEPAGE). Windows/macOS have no portable equivalent
// for anonymous ranges -- these return Unimplemented without changing the
// mapping. ptr must be page-aligned; size is rounded up. Empty range -> OK.
HYLE_PLATFORM_API Status vm_advise_hugepage(void* ptr,
                                            std::size_t size) noexcept;
HYLE_PLATFORM_API Status vm_advise_no_hugepage(void* ptr,
                                               std::size_t size) noexcept;

// ---- RAII region ------------------------------------------------------------
// Owns one reservation. Move-only; releases on destruction.

class VmRegion {
 public:
  // Reserve `size` bytes (rounded up).
  HYLE_PLATFORM_API static Result<VmRegion> reserve(std::size_t size) noexcept;

  // Allocate an immediately usable RW region with one OS mapping operation.
  HYLE_PLATFORM_API static Result<VmRegion> allocate_rw(
      std::size_t size) noexcept;

  // Like reserve(), but soft-prefer NUMA `node` (see numa.h; kNumaAny = plain).
  HYLE_PLATFORM_API static Result<VmRegion> reserve_numa(std::size_t size,
                                                         int node) noexcept;

  // Allocate a committed RW span striped over all or selected NUMA nodes.
  // See numa.h for node/chunk validation and platform capability semantics.
  HYLE_PLATFORM_API static Result<VmRegion> allocate_numa_interleaved(
      std::size_t size, std::span<const unsigned> nodes = {},
      std::size_t bytes_per_chunk = 0) noexcept;

  // Take ownership of an existing page-aligned reservation from vm_reserve*.
  // `size` must be the rounded size used with release. Invalid input is
  // returned as a structured error and ownership remains with the caller.
  HYLE_PLATFORM_API static Result<VmRegion> adopt(void* base,
                                                  std::size_t size) noexcept;

  // Reserve usable+1 guard page, commit the usable range as RW, leave the LOW
  // page as Protect::kNone. Stack-grows-down friendly: put the stack top at
  // usable() + usable_size() and grow toward usable() / the guard.
  // usable is rounded up to a page; total = page_size() + usable_aligned.
  HYLE_PLATFORM_API static Result<VmRegion> reserve_guarded(
      std::size_t usable) noexcept;

  VmRegion() noexcept = default;
  HYLE_PLATFORM_API ~VmRegion();
  HYLE_PLATFORM_API VmRegion(VmRegion&& o) noexcept;
  HYLE_PLATFORM_API VmRegion& operator=(VmRegion&& o) noexcept;
  VmRegion(const VmRegion&) = delete;
  VmRegion& operator=(const VmRegion&) = delete;

  bool valid() const noexcept { return base_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  void* data() const noexcept { return base_; }
  std::size_t size() const noexcept {
    return size_;
  }  // reserved bytes (page-aligned)

  // Guarded layout (only meaningful after reserve_guarded). usable() is the
  // first RW byte; the page at data() is the permanent no-access guard.
  void* usable() const noexcept {
    return guarded_ ? static_cast<std::byte*>(base_) + page_size() : base_;
  }
  std::size_t usable_size() const noexcept {
    return guarded_ ? size_ - page_size() : size_;
  }
  bool has_guard() const noexcept { return guarded_; }
  NumaPlacement numa_placement() const noexcept { return numa_placement_; }
  bool numa_interleaved() const noexcept { return numa_interleaved_; }

  // Commit / decommit / protect a subrange by byte offset from data(). A
  // selected-NUMA allocation is already committed; decommit is rejected because
  // recommitting could lose its placement. Release the whole region instead.
  // offset+len are rounded (offset down / end up) only via the free functions'
  // size round-up; offset must be page-aligned when len>0.
  HYLE_PLATFORM_API Status commit(std::size_t offset, std::size_t len) noexcept;
  HYLE_PLATFORM_API Status decommit(std::size_t offset,
                                    std::size_t len) noexcept;
  HYLE_PLATFORM_API Status protect(std::size_t offset, std::size_t len,
                                   Protect prot) noexcept;

  // Commit the entire reservation as RW (not the guard page if has_guard()).
  HYLE_PLATFORM_API Status commit_usable() noexcept;

  // Early release (also done by the destructor). Safe on invalid regions.
  HYLE_PLATFORM_API Status release() noexcept;

 private:
  void* base_ = nullptr;
  std::size_t size_ = 0;
  bool guarded_ = false;
  bool numa_interleaved_ = false;
  NumaPlacement numa_placement_ = NumaPlacement::kNotNeeded;
};

}  // namespace hyle::platform
