#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "hyle/core/compiler.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/obs/trace.h"

// system_inline.h -- the COMPILE-TIME-bound default allocator. SystemAllocator
// is stateless/EBO-friendly and never pays the runtime Allocator vtable's
// function-pointer dispatch. In a static/merged build its methods inline to a
// direct mi_* call. A shared-library consumer instead makes one direct import
// call to hyle_mem's backend seam: the vendor symbols remain private to the DLL
// and the public header is independently linkable.
//
// Static consumers do not include mimalloc.h: the five C functions below use
// only void*/size_t, and the package carries their archive as a link-only
// implementation dependency. Shared consumers do not name mi_* at all.
//
#if defined(HYLE_STATIC) && !defined(HYLE_MEM_BUILD)
#if defined(_MSC_VER) && !defined(__EDG__)
// Keep the redeclaration type-compatible with mimalloc v3's public prototype.
// Some users (and the allocator comparison benchmark) legitimately include
// mimalloc.h in the same translation unit; MSVC diagnoses a later addition of
// __declspec(restrict) as C4565 under /WX.
#define HYLE_MEM_MI_ALLOCATOR_RETURN __declspec(allocator) __declspec(restrict)
#elif defined(_MSC_VER)
#define HYLE_MEM_MI_ALLOCATOR_RETURN __declspec(restrict)
#else
#define HYLE_MEM_MI_ALLOCATOR_RETURN
#endif
extern "C" {
HYLE_MEM_MI_ALLOCATOR_RETURN void* mi_malloc(std::size_t size) noexcept;
HYLE_MEM_MI_ALLOCATOR_RETURN void* mi_malloc_aligned(
    std::size_t size, std::size_t alignment) noexcept;
void mi_free(void* p) noexcept;
void* mi_realloc(void* p, std::size_t newsize) noexcept;
void* mi_realloc_aligned(void* p, std::size_t newsize, std::size_t alignment) noexcept;
}
#undef HYLE_MEM_MI_ALLOCATOR_RETURN
#endif

namespace hyle::mem {

inline constexpr std::size_t kSystemDefaultAlignment = kDefaultAlignment;

namespace detail {

// Reviewed mimalloc v3 invariant (MI_MAX_ALIGN_SIZE). Keeping the value at the
// private implementation seam lets platforms with a wider max_align_t fall
// back to mi_*_aligned instead of silently weakening HYLE's contract.
inline constexpr std::size_t kMimallocV3NaturalAlignment = 16;

HYLE_FORCE_INLINE std::size_t system_backend_checked_alignment(
    std::size_t alignment) noexcept {
  // The overwhelming common case is the compile-time default. Keep it to one
  // predictable comparison; weaker/over-aligned requests take the full
  // contract check before selecting the backend entry.
  if (HYLE_UNLIKELY(alignment != kSystemDefaultAlignment)) {
    HYLE_FATAL_ASSERT(is_pow2(alignment));
    if (alignment < kSystemDefaultAlignment) {
      return kSystemDefaultAlignment;
    }
  }
  return alignment;
}

#if defined(HYLE_STATIC)
inline void* system_backend_allocate(std::size_t size,
                                     std::size_t alignment) noexcept {
  alignment = system_backend_checked_alignment(alignment);
  if (alignment <= kMimallocV3NaturalAlignment) {
    // mimalloc v3 gives blocks at least MI_MAX_ALIGN_SIZE alignment once the
    // requested size reaches that boundary. Padding only sub-default requests
    // preserves its plain small-allocation fast path.
    return mi_malloc(size < kMimallocV3NaturalAlignment
                         ? kMimallocV3NaturalAlignment
                         : size);
  }
  return mi_malloc_aligned(size, alignment);
}
inline void system_backend_deallocate(void* ptr) noexcept { mi_free(ptr); }
inline void* system_backend_reallocate(void* ptr, std::size_t new_size,
                                       std::size_t alignment) noexcept {
  alignment = system_backend_checked_alignment(alignment);
  if (alignment <= kMimallocV3NaturalAlignment) {
    const std::size_t backing_size =
        new_size < kMimallocV3NaturalAlignment
            ? kMimallocV3NaturalAlignment
            : new_size;
    return mi_realloc(ptr, backing_size);
  }
  return mi_realloc_aligned(ptr, new_size, alignment);
}
#else
// Binary implementation seams, not a second user-facing allocator. Inline
// container code keeps direct-call dispatch without leaking the vendor ABI
// into a shared SDK's consumer link contract.
HYLE_MEM_API void* system_backend_allocate(std::size_t size,
                                           std::size_t alignment) noexcept;
HYLE_MEM_API void system_backend_deallocate(void* ptr) noexcept;
HYLE_MEM_API void* system_backend_reallocate(
    void* ptr, std::size_t new_size, std::size_t alignment) noexcept;
#endif

}  // namespace detail

// Stateless, zero-size compile-time allocator over the system backend. This is
// the DEFAULT for HYLE containers (Container<T, Alloc = SystemAllocator>): the
// common host-RAM path pays zero indirect dispatch. Trivially copyable and
// empty (EBO-friendly).
struct SystemAllocator {
  // Every request is normalized to at least max_align_t. The backend pads
  // sub-default sizes to that boundary; over-alignment uses mimalloc's aligned
  // entry.
  static void* allocate(std::size_t size,
                        std::size_t alignment = kSystemDefaultAlignment) noexcept {
    if (HYLE_UNLIKELY(size == 0)) {
      (void)detail::system_backend_checked_alignment(alignment);
      return nullptr;
    }
    void* pointer = detail::system_backend_allocate(size, alignment);
    if (pointer != nullptr) {
      HYLE_TRACE_ALLOC_NAMED(pointer, size, "hyle.system");
    }
    return pointer;
  }
  static void deallocate(
      void* ptr, std::size_t /*size*/ = 0,
      std::size_t alignment = kSystemDefaultAlignment) noexcept {
    // Matching alignment is an allocator deallocation precondition. Preserve a
    // Debug diagnostic, but keep Release on the same direct-free shape as
    // std::allocator/PMR and mimalloc itself.
    HYLE_ASSUME(alignment != 0 && (alignment & (alignment - 1)) == 0);
#if defined(HYLE_TRACE)
    if (ptr != nullptr) HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
#endif
    detail::system_backend_deallocate(ptr);
  }
  static void* reallocate(void* ptr, std::size_t /*old_size*/, std::size_t new_size,
                          std::size_t alignment = kSystemDefaultAlignment) noexcept {
    if (HYLE_UNLIKELY(new_size == 0)) {
      (void)detail::system_backend_checked_alignment(alignment);
      if (ptr != nullptr) {
        HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
        detail::system_backend_deallocate(ptr);
      }
      return nullptr;
    }
    // The backend realloc accepts nullptr as allocate, avoiding a redundant
    // pointer branch on every grow step.
    void* result =
        detail::system_backend_reallocate(ptr, new_size, alignment);
    if (result != nullptr) {
      if (ptr != nullptr) HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
      HYLE_TRACE_ALLOC_NAMED(result, new_size, "hyle.system");
    }
    return result;
  }

  // ---- typed convenience layer (mirrors mem::Allocator's, so containers that call
  // allocate_n<T>/allocate_zeroed/deallocate_n work uniformly with either
  // allocator). All inline. ----
  template <class T>
  static T* allocate_n(std::size_t numel = 1, std::size_t alignment = alignof(T)) noexcept {
    if (numel > SIZE_MAX / sizeof(T)) return nullptr;
    return static_cast<T*>(allocate(sizeof(T) * numel, alignment));
  }
  template <class T>
  static T* allocate_zeroed(std::size_t numel = 1,
                            std::size_t alignment = alignof(T)) noexcept {
    static_assert(std::is_trivially_constructible_v<T>,
                  "allocate_zeroed requires trivially-constructible T");
    if (numel > SIZE_MAX / sizeof(T)) return nullptr;
    std::size_t sz = sizeof(T) * numel;
    void* p = allocate(sz, alignment);
    if (p != nullptr) std::memset(p, 0, sz);
    return static_cast<T*>(p);
  }
  template <class T>
  static T* reallocate_n(T* ptr, std::size_t old_numel,
                         std::size_t new_numel,
                         std::size_t alignment = alignof(T)) noexcept {
    static_assert(std::is_trivially_copyable_v<T>,
                  "reallocate_n requires trivially-copyable T");
    if (old_numel > SIZE_MAX / sizeof(T) ||
        new_numel > SIZE_MAX / sizeof(T))
      return nullptr;
    return static_cast<T*>(reallocate(ptr, sizeof(T) * old_numel,
                                      sizeof(T) * new_numel, alignment));
  }
  template <class T>
  static void deallocate_n(T* ptr, std::size_t numel = 1,
                           std::size_t alignment = alignof(T)) noexcept {
    HYLE_FATAL_ASSERT(numel <= SIZE_MAX / sizeof(T));
    deallocate(ptr, sizeof(T) * numel, alignment);
  }
};

}  // namespace hyle::mem
