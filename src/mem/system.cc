#include "hyle/mem/allocator.h"

#include <cstring>

#include "hyle/macros.h"
#include "hyle/platform.h"

#if defined(HYLE_OS_WINDOWS)
#  include <malloc.h>  // _aligned_malloc / _aligned_free / _msize
#else
#  include <cstdlib>  // posix_memalign / free
#endif

namespace hyle::mem {
namespace {

// Round an alignment request up to something the OS primitive accepts: a power
// of two, at least kDefaultAlignment (posix_memalign also needs >= sizeof(void*),
// which kDefaultAlignment already satisfies).
std::size_t normalize_alignment(std::size_t a) noexcept {
  if (a < kDefaultAlignment) return kDefaultAlignment;
  // a must be a power of two; a non-power-of-two request is a caller bug.
  HYLE_FATAL_ASSERT((a & (a - 1)) == 0);
  return a;
}

void* aligned_alloc_raw(std::size_t size, std::size_t align) noexcept {
#if defined(HYLE_OS_WINDOWS)
  return ::_aligned_malloc(size, align);
#else
  void* p = nullptr;
  // posix_memalign returns 0 on success; leaves p untouched on failure.
  if (::posix_memalign(&p, align, size) != 0) return nullptr;
  return p;
#endif
}

void aligned_free_raw(void* p) noexcept {
#if defined(HYLE_OS_WINDOWS)
  ::_aligned_free(p);  // blocks from _aligned_malloc MUST use _aligned_free
#else
  ::free(p);
#endif
}

// The SystemRAM backing: bare aligned malloc/free, alignment honored end-to-end
// (the whole point of mem-01). Reallocation is alloc+copy+free -- correct and
// portable; mem-02's mimalloc backing replaces this with mi_realloc_aligned.
void* system_realloc(void* /*state*/, void* ptr, std::size_t old_size,
                     std::size_t new_size, std::size_t alignment) noexcept {
  if (new_size == 0) {  // free (also covers allocate(0) -> nullptr)
    if (ptr != nullptr) aligned_free_raw(ptr);
    return nullptr;
  }
  std::size_t align = normalize_alignment(alignment);
  if (ptr == nullptr) {  // allocate
    return aligned_alloc_raw(new_size, align);
  }
  // reallocate: fresh aligned block, copy the overlap, release the old one.
  void* np = aligned_alloc_raw(new_size, align);
  if (np != nullptr && old_size != 0) {
    std::memcpy(np, ptr, old_size < new_size ? old_size : new_size);
  }
  aligned_free_raw(ptr);
  return np;
}

}  // namespace

Allocator system_allocator() noexcept {
  return Allocator(&system_realloc, nullptr);
}

}  // namespace hyle::mem
