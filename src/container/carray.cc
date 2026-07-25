#include "hyle/container/carray.h"

#include <cstring>
#include <limits>

#include "hyle/macros.h"

namespace hyle::container::detail {
namespace {

// Block layout: [ leading pad ][ CArrayHeader ][ elements... ]
//                ^block base    ^data - sizeof(header)         ^data
// data is aligned to max(align, alignof(header)); header sits in the bytes
// right before data; header_offset = data - block base (covers the pad too).
std::size_t compute_header_offset(std::size_t align) noexcept {
  HYLE_FATAL_ASSERT(align != 0 && (align & (align - 1)) == 0);
  std::size_t a = align < alignof(CArrayHeader) ? alignof(CArrayHeader) : align;
  HYLE_FATAL_ASSERT(a <= std::numeric_limits<std::uint32_t>::max());
  HYLE_FATAL_ASSERT(sizeof(CArrayHeader) <=
                    std::numeric_limits<std::size_t>::max() - (a - 1));
  // Smallest multiple of `a` that is >= sizeof(CArrayHeader), so the header
  // fits between the block base and the (a-aligned) data pointer.
  std::size_t off = (sizeof(CArrayHeader) + a - 1) & ~(a - 1);
  return off;
}

}  // namespace

void* carray_set_capacity_internal(void* data, std::uint64_t min_capacity,
                                   std::size_t item_size, std::size_t align,
                                   mem::Allocator alloc) {
  HYLE_FATAL_ASSERT(item_size != 0);
  std::size_t a = align < alignof(CArrayHeader) ? alignof(CArrayHeader) : align;
  std::size_t header_offset = compute_header_offset(a);
  if (data == nullptr) {
    if (min_capacity >
        (std::numeric_limits<std::size_t>::max() - header_offset) / item_size)
      return nullptr;
    const std::size_t new_bytes =
        header_offset + static_cast<std::size_t>(min_capacity) * item_size;
    // Fresh allocation (no old block to grow).
    auto* base = static_cast<std::byte*>(alloc.allocate(new_bytes, a));
    if (base == nullptr) return nullptr;
    std::byte* new_data = base + header_offset;
    auto* h = reinterpret_cast<CArrayHeader*>(new_data - sizeof(CArrayHeader));
    h->capacity = min_capacity;
    h->size = 0;
    h->header_offset = static_cast<std::uint32_t>(header_offset);
    h->align = static_cast<std::uint32_t>(a);
    h->alloc = alloc;
    return new_data;
  }

  // Resize an EXISTING block via reallocate: the base (header+data as one block)
  // is handed to the allocator, which can grow it IN PLACE (mimalloc
  // mi_realloc), skipping the alloc-new + memcpy + free that a manual grow
  // pays. It may also shrink for PodVector::shrink_to_fit(). header_offset and
  // align stay unchanged; reallocate preserves the block's bytes whether it
  // resizes in place or relocates. Only capacity changes afterward.
  CArrayHeader* old_h = carray_header(data);
  HYLE_FATAL_ASSERT(min_capacity >= old_h->size);
  // Existing storage owns its allocator/alignment metadata. Never let a caller
  // accidentally reallocate it through a different allocator.
  alloc = old_h->alloc;
  a = old_h->align;
  header_offset = old_h->header_offset;
  if (min_capacity >
      (std::numeric_limits<std::size_t>::max() - header_offset) / item_size)
    return nullptr;
  const std::size_t new_bytes =
      header_offset + static_cast<std::size_t>(min_capacity) * item_size;
  std::uint64_t old_size = old_h->size;
  std::size_t old_bytes =
      header_offset + static_cast<std::size_t>(old_h->capacity) * item_size;
  std::byte* old_base = static_cast<std::byte*>(data) - header_offset;
  auto* base = static_cast<std::byte*>(
      alloc.reallocate(old_base, old_bytes, new_bytes, a));
  if (base == nullptr) return nullptr;  // OOM: caller keeps old data (realloc left it)

  std::byte* new_data = base + header_offset;
  auto* h = reinterpret_cast<CArrayHeader*>(new_data - sizeof(CArrayHeader));
  // header_offset/align/alloc are the same values; capacity grows; size preserved.
  h->capacity = min_capacity;
  h->size = old_size;
  h->header_offset = static_cast<std::uint32_t>(header_offset);
  h->align = static_cast<std::uint32_t>(a);
  h->alloc = alloc;
  return new_data;
}

void carray_free_internal(void* data, std::size_t item_size) noexcept {
  if (data == nullptr) return;
  CArrayHeader* h = carray_header(data);
  std::byte* base = static_cast<std::byte*>(data) - h->header_offset;
  std::size_t bytes =
      h->header_offset + static_cast<std::size_t>(h->capacity) * item_size;
  mem::Allocator alloc = h->alloc;
  alloc.deallocate(base, bytes, h->align);
}

}  // namespace hyle::container::detail
