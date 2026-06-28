#include "hyle/mem/allocator.h"

#include <cstring>

#include "hyle/macros.h"
#include "hyle/platform.h"

#if defined(HYLE_USE_MIMALLOC)
#  include <mimalloc.h>  // PRIVATE include -- mi_* types never leak to a header
#elif defined(HYLE_OS_WINDOWS)
#  include <malloc.h>  // _aligned_malloc / _aligned_free
#else
#  include <cstdlib>  // posix_memalign / free
#endif

namespace hyle::mem {
namespace {

// Round an alignment request up to something the backing accepts: a power of
// two, at least kDefaultAlignment. A non-power-of-two request is a caller bug.
std::size_t normalize_alignment(std::size_t a) noexcept {
  if (a < kDefaultAlignment) return kDefaultAlignment;
  HYLE_FATAL_ASSERT((a & (a - 1)) == 0);
  return a;
}

#if defined(HYLE_USE_MIMALLOC)

// mimalloc backing (mem-design decision: borrow a battle-tested general malloc;
// self-rolling one is wasted effort and necessarily worse). Wrapped behind the
// Allocator vtable so mi_* types never appear in any HYLE header -- swapping the
// backing later touches only this file (same discipline as the ggml bridge).
// mimalloc tracks each block's size, so realloc needs no manual copy/old_size.
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

#else  // bare OS aligned malloc (mem-01 baseline; HYLE_USE_MIMALLOC=OFF fallback)

void* aligned_alloc_raw(std::size_t size, std::size_t align) noexcept {
#  if defined(HYLE_OS_WINDOWS)
  return ::_aligned_malloc(size, align);
#  else
  void* p = nullptr;
  if (::posix_memalign(&p, align, size) != 0) return nullptr;
  return p;
#  endif
}

void aligned_free_raw(void* p) noexcept {
#  if defined(HYLE_OS_WINDOWS)
  ::_aligned_free(p);  // blocks from _aligned_malloc MUST use _aligned_free
#  else
  ::free(p);
#  endif
}

void* system_realloc(void* /*state*/, void* ptr, std::size_t old_size,
                     std::size_t new_size, std::size_t alignment) noexcept {
  if (new_size == 0) {
    if (ptr != nullptr) aligned_free_raw(ptr);
    return nullptr;
  }
  std::size_t align = normalize_alignment(alignment);
  if (ptr == nullptr) return aligned_alloc_raw(new_size, align);
  // realloc contract: on OOM the ORIGINAL block must stay valid (freeing it
  // here would hand back a dangling pointer).
  void* np = aligned_alloc_raw(new_size, align);
  if (np == nullptr) return nullptr;
  if (old_size != 0) {
    std::memcpy(np, ptr, old_size < new_size ? old_size : new_size);
  }
  aligned_free_raw(ptr);
  return np;
}

#endif

}  // namespace

Allocator system_allocator() noexcept {
  return Allocator(&system_realloc, nullptr);
}

}  // namespace hyle::mem

