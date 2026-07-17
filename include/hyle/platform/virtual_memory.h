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

// Host virtual memory (platform-02b): reserve/commit/decommit/release + protect.
// Real consumer = fiber stack guard pages (mem-design fiber_stack / sched-design §1).
// Not device/VRAM, not general malloc (mimalloc), not cross-process shmem (io-05).
//
// Windows: VirtualAlloc MEM_RESERVE / MEM_COMMIT / MEM_DECOMMIT / MEM_RELEASE +
// VirtualProtect. POSIX: mmap PROT_NONE + mprotect + munmap (+ madvise for decommit).
// clay has no virtual_memory API -- this is net-new.

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
  return (n + (p - 1)) & ~(p - 1);
}

// Align up to large_page_size() when known; otherwise to page_size(). n==0 -> 0.
HYLE_FORCE_INLINE std::size_t large_page_align_up(std::size_t n) noexcept {
  if (n == 0) return 0;
  std::size_t lp = large_page_size();
  std::size_t p = lp != 0 ? lp : page_size();
  return (n + (p - 1)) & ~(p - 1);
}

// ---- protection -------------------------------------------------------------

// Access rights for a committed (or reserved) range. kNone is the permanent
// guard-page mode (PAGE_NOACCESS / PROT_NONE) -- NOT Windows PAGE_GUARD (one-shot).
enum class Protect : std::uint8_t {
  kNone = 0,       // no access (overflow guard)
  kRead = 1,       // read-only
  kReadWrite = 2,  // read + write
};

// ---- free-function primitives -----------------------------------------------
// Thin OS wrappers. Sizes are rounded UP to page_size(); pointers with size>0
// must already be page-aligned (caller contract: HYLE_FATAL_ASSERT).
// Failures return nullptr / false; detail in vm_last_error().

// Reserve `size` bytes of address space (rounded up). Not readable/writable
// until vm_commit. Returns the base, or nullptr on failure.
HYLE_PLATFORM_API void* vm_reserve(std::size_t size) noexcept;

// Make [ptr, ptr+size) resident and Protect::kReadWrite.
HYLE_PLATFORM_API bool vm_commit(void* ptr, std::size_t size) noexcept;

// Drop physical pages for [ptr, ptr+size). Range stays reserved; re-commit
// before touch. Access after decommit without re-commit is undefined (fault).
HYLE_PLATFORM_API bool vm_decommit(void* ptr, std::size_t size) noexcept;

// Free the WHOLE reservation that started at `base` with `size` (the values
// returned/used by vm_reserve). Windows requires releasing from the original
// base; sub-range release is not supported.
HYLE_PLATFORM_API bool vm_release(void* base, std::size_t size) noexcept;

// Change protection on [ptr, ptr+size). On POSIX, kNone on a never-committed
// reservation is already the reserve default; on Windows, reserved pages are
// PAGE_NOACCESS until commit.
HYLE_PLATFORM_API bool vm_protect(void* ptr, std::size_t size, Protect prot) noexcept;

// Last VM error on THIS thread, or "" if none. Valid until the next failing
// vm_* call on the same thread (dynlib-style, no allocator).
HYLE_PLATFORM_API const char* vm_last_error() noexcept;

// ---- hugepage / THP hints ---------------------------------------------------
// Soft advice only: never required for correctness. Linux uses madvise
// (MADV_HUGEPAGE / MADV_NOHUGEPAGE). Windows/macOS have no portable equivalent
// for anonymous ranges -- these return false without changing the mapping.
// ptr must be page-aligned; size is rounded up. Empty range -> true no-op.
// Failure does NOT set a fatal condition; check return + vm_last_error().
HYLE_PLATFORM_API bool vm_advise_hugepage(void* ptr, std::size_t size) noexcept;
HYLE_PLATFORM_API bool vm_advise_no_hugepage(void* ptr, std::size_t size) noexcept;

// ---- RAII region ------------------------------------------------------------
// Owns one reservation. Move-only; releases on destruction.

class VmRegion {
 public:
  // Reserve `size` bytes (rounded up). On failure: !valid(), see vm_last_error().
  HYLE_PLATFORM_API static VmRegion reserve(std::size_t size) noexcept;

  // Like reserve(), but soft-prefer NUMA `node` (see numa.h; kNumaAny = plain).
  HYLE_PLATFORM_API static VmRegion reserve_numa(std::size_t size, int node) noexcept;

  // Take ownership of an existing page-aligned reservation from vm_reserve*.
  // `size` must be the rounded size used with release. On nullptr: !valid().
  HYLE_PLATFORM_API static VmRegion adopt(void* base, std::size_t size) noexcept;

  // Reserve usable+1 guard page, commit the usable range as RW, leave the LOW
  // page as Protect::kNone. Stack-grows-down friendly: put the stack top at
  // usable() + usable_size() and grow toward usable() / the guard.
  // usable is rounded up to a page; total = page_size() + usable_aligned.
  HYLE_PLATFORM_API static VmRegion reserve_guarded(std::size_t usable) noexcept;

  VmRegion() noexcept = default;
  HYLE_PLATFORM_API ~VmRegion();
  HYLE_PLATFORM_API VmRegion(VmRegion&& o) noexcept;
  HYLE_PLATFORM_API VmRegion& operator=(VmRegion&& o) noexcept;
  HYLE_DISABLE_COPY(VmRegion);

  bool valid() const noexcept { return base_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  void* data() const noexcept { return base_; }
  std::size_t size() const noexcept { return size_; }  // reserved bytes (page-aligned)

  // Guarded layout (only meaningful after reserve_guarded). usable() is the first
  // RW byte; the page at data() is the permanent no-access guard.
  void* usable() const noexcept {
    return guarded_ ? static_cast<std::byte*>(base_) + page_size() : base_;
  }
  std::size_t usable_size() const noexcept {
    return guarded_ ? size_ - page_size() : size_;
  }
  bool has_guard() const noexcept { return guarded_; }

  // Commit / decommit / protect a subrange by byte offset from data().
  // offset+len are rounded (offset down / end up) only via the free functions'
  // size round-up; offset must be page-aligned when len>0.
  HYLE_PLATFORM_API bool commit(std::size_t offset, std::size_t len) noexcept;
  HYLE_PLATFORM_API bool decommit(std::size_t offset, std::size_t len) noexcept;
  HYLE_PLATFORM_API bool protect(std::size_t offset, std::size_t len,
                                 Protect prot) noexcept;

  // Commit the entire reservation as RW (not the guard page if has_guard()).
  HYLE_PLATFORM_API bool commit_usable() noexcept;

  // Early release (also done by the destructor). Safe on invalid regions.
  HYLE_PLATFORM_API void release() noexcept;

 private:
  void* base_ = nullptr;
  std::size_t size_ = 0;
  bool guarded_ = false;
};

}  // namespace hyle::platform
