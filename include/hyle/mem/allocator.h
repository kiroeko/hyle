#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_MEM_API
#elif defined(HYLE_MEM_BUILD)
#define HYLE_MEM_API HYLE_EXPORT
#else
#define HYLE_MEM_API HYLE_IMPORT
#endif

namespace hyle::mem {

// Default alignment when a caller does not ask for more (matches the strictest
// fundamental scalar; backends never return less than this).
inline constexpr std::size_t kDefaultAlignment = alignof(std::max_align_t);

// ---- C-ABI allocator vtable -------------------------------------------------
// THREE function pointers (alloc/free/realloc), each carrying alignment. clay
// used ONE 5-arg realloc_func for all three "to minimize symbols" -- but that
// is a PERFORMANCE bug, not a virtue: on the Win64 ABI only 4 integer args land
// in registers, so the 5th (alignment) SPILLS TO THE STACK on every call, and
// one-function-does-three forces a dispatch branch (ptr==null? new==0?) on the
// hot path. Splitting is still 100% C-ABI clean (three C function pointers,
// trivially copyable, cross-DLL stable -- exactly what jemalloc/pmr/cudaMalloc
// do) AND register-friendly: alloc is 3 args (all in registers), free is 4 (all
// in registers), no dispatch branch. Measured: removes the spill + branch tax.
// Returns nullptr on OOM (allocation is a hot path; failure is a value, not a
// Status).
//   AllocFn  (state, size, alignment)                 -> ptr (nullptr on
//   OOM/size==0) FreeFn   (state, ptr, size, alignment)            -> void
//   (ptr==nullptr is a no-op) ReallocFn(state, ptr, old_size, new_size, align)
//   -> ptr (the COLD path; 5 args ok). Every valid backing obeys one common
//   contract: null ptr allocates; new_size==0 applies the backing's deallocation
//   semantics and returns nullptr (bulk allocators may defer physical reclaim
//   until reset); a failed non-zero resize leaves the original allocation live
//   and unchanged.
//   Backings that cannot grow may still return the same pointer when the new
//   request fits their existing fixed-capacity slot.
using AllocFn = void *(*)(void *state, std::size_t size,
                          std::size_t alignment) noexcept;
using FreeFn = void (*)(void *state, void *ptr, std::size_t size,
                        std::size_t alignment) noexcept;
using ReallocFn = void *(*)(void *state, void *ptr, std::size_t old_size,
                            std::size_t new_size,
                            std::size_t alignment) noexcept;

// ---- injectable allocator handle --------------------------------------------
// {fn, state}: trivially copyable, ABI-clean (function pointer + void*). Passed
// by value through every allocating API (charter: explicit injection, no hidden
// global heap). Methods are inline pure-forwarders -- zero overhead, the
// wrapper never costs a DLL call; only the backing fn lives in a TU.
class Allocator {
public:
  Allocator() noexcept = default; // null/invalid allocator
  Allocator(AllocFn alloc, FreeFn free, ReallocFn realloc, void *state) noexcept
      : alloc_fn_(alloc), free_fn_(free), realloc_fn_(realloc), state_(state) {}

  // Complete vtable only (all three fn pointers). A partial/default-constructed
  // Allocator is invalid -- checking all three (not just alloc) avoids a null
  // deref in deallocate/reallocate on a hand-built partial vtable (followup
  // #5).
  bool valid() const noexcept {
    return alloc_fn_ != nullptr && free_fn_ != nullptr &&
           realloc_fn_ != nullptr;
  }

  // These are logically const: they dispatch through the fn pointers without
  // mutating the handle itself (so allocating from a `const Allocator&` is
  // valid). Each forwards to the matching register-friendly entry -- no spill,
  // no dispatch branch.
  void *allocate(std::size_t size,
                 std::size_t alignment = kDefaultAlignment) const noexcept {
    return alloc_fn_(state_, size, alignment);
  }
  void deallocate(void *ptr, std::size_t size,
                  std::size_t alignment = kDefaultAlignment) const noexcept {
    free_fn_(state_, ptr, size, alignment);
  }
  void *reallocate(void *ptr, std::size_t old_size, std::size_t new_size,
                   std::size_t alignment = kDefaultAlignment) const noexcept {
    return realloc_fn_(state_, ptr, old_size, new_size, alignment);
  }

  // Raw vtable accessors (for handing the seam across a C-ABI boundary).
  // alloc_fn() doubles as the identity for comparing two handles to the same
  // backing (pmr).
  AllocFn alloc_fn() const noexcept { return alloc_fn_; }
  FreeFn free_fn() const noexcept { return free_fn_; }
  ReallocFn realloc_fn() const noexcept { return realloc_fn_; }
  void *state() const noexcept { return state_; }

  // ---- typed convenience layer --------------------------------------------
  // Header-inline (hot path, no DLL hop). Names chosen for C++ clarity, NOT
  // clay parity: clay called these callocate/release, but `release` collides
  // with unique_ptr::release (which does the OPPOSITE -- gives up ownership
  // WITHOUT freeing), so it's a real confusion source. HYLE uses
  // allocate_zeroed / deallocate_n instead (paired with allocate_n; behavior
  // unchanged).

  // Raw storage for `numel` T's, aligned for T. No construction.
  template <class T>
  T *allocate_n(std::size_t numel = 1,
                std::size_t alignment = alignof(T)) const noexcept {
    if (numel > SIZE_MAX / sizeof(T))
      return nullptr;
    return static_cast<T *>(allocate(sizeof(T) * numel, alignment));
  }

  // Zero-initialized storage for trivially-constructible T.
  template <class T>
  T *allocate_zeroed(std::size_t numel = 1,
                     std::size_t alignment = alignof(T)) const noexcept {
    static_assert(std::is_trivially_constructible_v<T>,
                  "allocate_zeroed requires trivially-constructible T");
    if (numel > SIZE_MAX / sizeof(T))
      return nullptr;
    std::size_t sz = sizeof(T) * numel;
    void *p = allocate(sz, alignment);
    if (p != nullptr)
      std::memset(p, 0, sz);
    return static_cast<T *>(p);
  }

  // Resize typed raw storage. Overflow/failure leaves the original allocation
  // live; new_numel==0 follows the backing's deallocation semantics.
  template <class T>
  T *reallocate_n(T *ptr, std::size_t old_numel, std::size_t new_numel,
                  std::size_t alignment = alignof(T)) const noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "reallocate_n requires trivially-copyable T");
    if (old_numel > SIZE_MAX / sizeof(T) ||
        new_numel > SIZE_MAX / sizeof(T))
      return nullptr;
    return static_cast<T *>(reallocate(ptr, sizeof(T) * old_numel,
                                       sizeof(T) * new_numel, alignment));
  }

  // Free typed storage from allocate_n/allocate_zeroed (no destructor call).
  template <class T>
  void deallocate_n(T *ptr, std::size_t numel = 1,
                    std::size_t alignment = alignof(T)) const noexcept {
    HYLE_FATAL_ASSERT(numel <= SIZE_MAX / sizeof(T));
    deallocate(ptr, sizeof(T) * numel, alignment);
  }

  // Allocate + placement-new a T. nullptr if allocation failed.
  template <class T, class... Args> T *construct(Args &&...args) const {
    void *p = allocate(sizeof(T), alignof(T));
    if (p == nullptr)
      return nullptr;
    try {
      return ::new (p) T(std::forward<Args>(args)...);
    } catch (...) {
      deallocate(p, sizeof(T), alignof(T));
      throw;
    }
  }

  // ~T() then free through THIS allocator (never global delete).
  template <class T> void destroy(T *ptr) const noexcept {
    if (ptr != nullptr) {
      ptr->~T();
      deallocate_n(ptr);
    }
  }

  // unique_ptr whose deleter destroys through this allocator. The deleter holds
  // the raw free fn/state (NOT an Allocator by value): inside Allocator's own
  // definition the type is incomplete, so a by-value Allocator member is
  // ill-formed (GCC rejects it; MSVC let it slide).
  template <class T> struct Deleter {
    FreeFn free;
    void *state;
    void operator()(T *p) const noexcept {
      if (p != nullptr) {
        p->~T();
        free(state, p, sizeof(T), alignof(T));
      }
    }
  };
  template <class T, class... Args>
  std::unique_ptr<T, Deleter<T>> make_unique(Args &&...args) const {
    return std::unique_ptr<T, Deleter<T>>(
        construct<T>(std::forward<Args>(args)...),
        Deleter<T>{free_fn_, state_});
  }

private:
  AllocFn alloc_fn_ = nullptr;
  FreeFn free_fn_ = nullptr;
  ReallocFn realloc_fn_ = nullptr;
  void *state_ = nullptr;
};

// Process-wide SystemRAM allocator. mem-01: bare aligned malloc/free; mem-02
// swaps mimalloc in behind this same handle with zero caller changes.
HYLE_MEM_API Allocator system_allocator() noexcept;

// ---- usable-size queries (SystemAllocator only) -----------------------------
// HYLE's Allocator is a SIZED-FREE model (deallocate carries the size, like
// Rust GlobalAlloc / C++17 sized delete / jemalloc sdallocx) -- faster, no
// per-block metadata forced on every backing. So there is no vtable
// usable_size(): most backings (arena/pool/stack) don't store per-pointer size.
// These two queries are SPECIFIC to the mimalloc-backed system_allocator(),
// where the backing DOES know the true physical block size (mimalloc rounds a
// request up to a size class).

// Actual usable bytes of a block from system_allocator() (>= the requested
// size; the extra is free to use, e.g. grow-in-place). ptr must be a live
// system_ allocator block. mimalloc mi_usable_size.
HYLE_MEM_API std::size_t system_usable_size(const void *ptr) noexcept;

// Backing allocator's recommended request size for at least `size` bytes,
// WITHOUT allocating (mimalloc mi_good_size / jemalloc nallocx shape). The
// result is >= size and can avoid a near-immediate realloc. It is NOT
// guaranteed to equal system_usable_size(allocate(size)): instrumentation and
// platform allocator modes may expose only the requested payload as usable.
HYLE_MEM_API std::size_t system_good_size(std::size_t size) noexcept;

// Eagerly collect cached free blocks owned by the current thread's system heap.
// Normal collection performs allocator maintenance without demanding an
// immediate OS purge. `force=true` aggressively returns every possible page to
// the OS and can be expensive; reserve it for explicit idle/pressure points,
// never an allocation hot path. Long-lived worker pools should invoke this on
// each worker whose heap needs collection. This is system_allocator-specific
// and deliberately does not enlarge the hot three-function Allocator vtable.
HYLE_MEM_API void system_collect(bool force = false) noexcept;

} // namespace hyle::mem
