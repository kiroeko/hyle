#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "hyle/macros.h"
#include "hyle/obs/trace.h"

// system_inline.h -- the COMPILE-TIME-bound default allocator. The whole point: a caller
// that uses SystemAllocator (the container default) gets allocate() INLINED to a direct
// `call mi_malloc`, with NO function-pointer indirection, NO double-hop, NO dispatch branch
// -- i.e. parity with raw mimalloc in a merged/static build. The runtime `Allocator` vtable
// (allocator.h) stays as the OPT-IN path for genuine runtime polymorphism (per-request
// arena, mock, cross-DLL injection).
//
// KEY (resolves the false "inline => must expose mimalloc.h" dilemma): we do NOT include
// mimalloc.h. We forward-declare ONLY the 5 mi_* C functions we call. Their signatures use
// only void*/size_t -- no mi_* type leaks. mimalloc's mi_decl_*/mi_attr_* are optimization
// hints, NOT part of the ABI, so omitting them in our decl is link-compatible. This keeps
// mimalloc.h out of every consumer TU while still letting allocate() inline to mi_malloc.
//
// The symbol resolves against mimalloc_static at link time. In a single binary / static
// link, mi_malloc is in-image -> truly inlined. Across a DLL boundary the call is one
// import hop (physics of any cross-DLL call), still better than the fn-ptr double-hop.

extern "C" {
void* mi_malloc(std::size_t size) noexcept;
void* mi_malloc_aligned(std::size_t size, std::size_t alignment) noexcept;
void mi_free(void* p) noexcept;
void* mi_realloc(void* p, std::size_t newsize) noexcept;
void* mi_realloc_aligned(void* p, std::size_t newsize, std::size_t alignment) noexcept;
}

namespace hyle::mem {

// Default alignment when a caller does not ask for more (matches the strictest fundamental
// scalar). Kept in sync with allocator.h's kDefaultAlignment.
inline constexpr std::size_t kSystemDefaultAlignment = alignof(std::max_align_t);

// Stateless, zero-size compile-time allocator over mimalloc. All methods inline to a direct
// mi_* call. This is the DEFAULT for HYLE containers (Container<T, Alloc = SystemAllocator>):
// the common host-RAM path pays zero indirection. Trivially copyable, empty (EBO-friendly).
struct SystemAllocator {
  // Default-aligned fast path inlines to `mi_malloc(size)`; over-aligned to mi_malloc_aligned.
  static void* allocate(std::size_t size,
                        std::size_t alignment = kSystemDefaultAlignment) noexcept {
    if (size == 0) return nullptr;
    void* pointer = alignment <= kSystemDefaultAlignment
                        ? mi_malloc(size)
                        : mi_malloc_aligned(size, alignment);
    if (pointer != nullptr) {
      HYLE_TRACE_ALLOC_NAMED(pointer, size, "hyle.system");
    }
    return pointer;
  }
  static void deallocate(void* ptr, std::size_t /*size*/ = 0,
                         std::size_t /*alignment*/ = kSystemDefaultAlignment) noexcept {
    if (ptr != nullptr) {
      HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
      mi_free(ptr);
    }
  }
  static void* reallocate(void* ptr, std::size_t /*old_size*/, std::size_t new_size,
                          std::size_t alignment = kSystemDefaultAlignment) noexcept {
    if (new_size == 0) {
      if (ptr != nullptr) {
        HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
        mi_free(ptr);
      }
      return nullptr;
    }
    // Both mimalloc realloc forms accept nullptr as allocate. Calling them
    // directly avoids a redundant pointer branch on every grow step.
    void* result = alignment <= kSystemDefaultAlignment
                       ? mi_realloc(ptr, new_size)
                       : mi_realloc_aligned(ptr, new_size, alignment);
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
