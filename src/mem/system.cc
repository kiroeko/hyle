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
//
// THREE split entries (alloc/free/realloc) instead of one 5-arg realloc_func: the hot
// alloc path is 3 args (no Win64 stack spill) with NO dispatch branch -- the seam costs
// only the unavoidable indirect call now. mi_malloc already guarantees >= 16-byte
// (alignof(max_align_t)) alignment, so the common default-aligned case takes the plain
// (faster) path; only over-aligned (>16, e.g. AVX512 64B) needs mi_*_aligned.
void* system_alloc(void* /*state*/, std::size_t size, std::size_t alignment) noexcept {
  if (size == 0) return nullptr;
  // Common-case fast path: default (or weaker) alignment needs no normalization at all --
  // mi_malloc already guarantees >= 16-byte. Skip normalize_alignment (its assert + bit
  // test) for the hot path; only over-aligned requests pay it.
  if (alignment <= kDefaultAlignment) return mi_malloc(size);
  return mi_malloc_aligned(size, normalize_alignment(alignment));
}

void system_free(void* /*state*/, void* ptr, std::size_t /*size*/,
                 std::size_t /*alignment*/) noexcept {
  if (ptr != nullptr) mi_free(ptr);  // mi_free recovers size/alignment from the block
}

void* system_realloc(void* /*state*/, void* ptr, std::size_t /*old_size*/,
                     std::size_t new_size, std::size_t alignment) noexcept {
  if (new_size == 0) {  // realloc-to-0 == free
    if (ptr != nullptr) mi_free(ptr);
    return nullptr;
  }
  std::size_t align = normalize_alignment(alignment);
  // mi_realloc_aligned preserves alignment and returns nullptr on OOM with the
  // original block left intact (C realloc contract) -- matches our vtable.
  if (align <= kDefaultAlignment) {
    if (ptr == nullptr) return mi_malloc(new_size);
    return mi_realloc(ptr, new_size);
  }
  if (ptr == nullptr) return mi_malloc_aligned(new_size, align);
  return mi_realloc_aligned(ptr, new_size, align);
}

}  // namespace

Allocator system_allocator() noexcept {
  return Allocator(&system_alloc, &system_free, &system_realloc, nullptr);
}

}  // namespace hyle::mem

