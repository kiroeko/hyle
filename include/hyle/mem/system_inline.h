#pragma once

#include <cstddef>

#include "hyle/macros.h"

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
    if (alignment <= kSystemDefaultAlignment) return mi_malloc(size);
    return mi_malloc_aligned(size, alignment);
  }
  static void deallocate(void* ptr, std::size_t /*size*/ = 0,
                         std::size_t /*alignment*/ = kSystemDefaultAlignment) noexcept {
    if (ptr != nullptr) mi_free(ptr);
  }
  static void* reallocate(void* ptr, std::size_t /*old_size*/, std::size_t new_size,
                          std::size_t alignment = kSystemDefaultAlignment) noexcept {
    if (new_size == 0) {
      if (ptr != nullptr) mi_free(ptr);
      return nullptr;
    }
    if (alignment <= kSystemDefaultAlignment) {
      return ptr == nullptr ? mi_malloc(new_size) : mi_realloc(ptr, new_size);
    }
    return ptr == nullptr ? mi_malloc_aligned(new_size, alignment)
                          : mi_realloc_aligned(ptr, new_size, alignment);
  }
};

}  // namespace hyle::mem
