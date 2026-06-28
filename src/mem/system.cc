#include "hyle/mem/allocator.h"

#include "hyle/macros.h"

#include <mimalloc.h>  // PRIVATE include -- mi_* types never leak to a header

namespace hyle::mem {
namespace {

// Round an alignment request up to something the backing accepts: a power of
// two, at least kDefaultAlignment. A non-power-of-two request is a caller bug.
std::size_t normalize_alignment(std::size_t a) noexcept {
  if (a < kDefaultAlignment) return kDefaultAlignment;
  HYLE_FATAL_ASSERT((a & (a - 1)) == 0);
  return a;
}

// mimalloc backing (mem-design decision: borrow a battle-tested general malloc;
// self-rolling one is wasted effort and necessarily worse). Wrapped behind the
// Allocator vtable so mi_* types never appear in any HYLE header -- swapping the
// backing later touches only this file (same discipline as the ggml bridge).
// mimalloc tracks each block's size, so realloc needs no manual copy/old_size.
//
// This is the ONLY backing: the former OS aligned-malloc fallback was removed on
// purpose (user directive). A disable/fallback path is a trap -- it lets a build
// silently sidestep a mimalloc-dependent problem and be mis-judged as fine. If
// something breaks with mimalloc, we face it, not hide behind glibc.
void* system_realloc(void* /*state*/, void* ptr, std::size_t /*old_size*/,
                     std::size_t new_size, std::size_t alignment) noexcept {
  if (new_size == 0) {  // free (also covers allocate(0) -> nullptr)
    if (ptr != nullptr) mi_free(ptr);
    return nullptr;
  }
  std::size_t align = normalize_alignment(alignment);
  // mi_malloc already guarantees >= 16-byte (alignof(max_align_t)) alignment, so
  // the common default-aligned case takes the plain (faster) path -- the aligned
  // path adds bookkeeping that, in tight alloc/free loops, loses to glibc's
  // tcache. Only over-aligned requests (>16, e.g. AVX512 64B) need mi_*_aligned.
  // (kDefaultAlignment == alignof(max_align_t) == mimalloc's MI_MAX_ALIGN_SIZE
  // default; that macro is internal, so use our public constant.)
  if (align <= kDefaultAlignment) {
    if (ptr == nullptr) return mi_malloc(new_size);
    return mi_realloc(ptr, new_size);
  }
  // mi_realloc_aligned preserves alignment and returns nullptr on OOM with the
  // original block left intact (C realloc contract) -- matches our vtable.
  if (ptr == nullptr) return mi_malloc_aligned(new_size, align);
  return mi_realloc_aligned(ptr, new_size, align);
}

}  // namespace

Allocator system_allocator() noexcept {
  return Allocator(&system_realloc, nullptr);
}

}  // namespace hyle::mem

