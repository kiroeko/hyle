#include "hyle/container/carray.h"

#include <cstring>

#include "hyle/macros.h"

namespace hyle::container::detail {
namespace {

// Block layout: [ leading pad ][ CArrayHeader ][ elements... ]
//                ^block base    ^data - sizeof(header)         ^data
// data is aligned to max(align, alignof(header)); header sits in the bytes
// right before data; header_offset = data - block base (covers the pad too).
std::size_t compute_header_offset(std::size_t align) noexcept {
  std::size_t a = align < alignof(CArrayHeader) ? alignof(CArrayHeader) : align;
  // Smallest multiple of `a` that is >= sizeof(CArrayHeader), so the header
  // fits between the block base and the (a-aligned) data pointer.
  std::size_t off = (sizeof(CArrayHeader) + a - 1) & ~(a - 1);
  return off;
}

}  // namespace

void* carray_set_capacity_internal(void* data, std::uint64_t min_capacity,
                                   std::size_t item_size, std::size_t align,
                                   mem::Allocator alloc) {
  std::uint64_t old_size = data ? carray_header(data)->size : 0;
  std::size_t a = align < alignof(CArrayHeader) ? alignof(CArrayHeader) : align;
  std::size_t header_offset = compute_header_offset(a);

  std::size_t bytes = header_offset + static_cast<std::size_t>(min_capacity) * item_size;
  auto* base = static_cast<std::byte*>(alloc.allocate(bytes, a));
  if (base == nullptr) return nullptr;  // OOM: caller keeps old data

  std::byte* new_data = base + header_offset;
  auto* h = reinterpret_cast<CArrayHeader*>(new_data - sizeof(CArrayHeader));
  h->capacity = min_capacity;
  h->size = old_size;
  h->header_offset = static_cast<std::uint32_t>(header_offset);
  h->align = static_cast<std::uint32_t>(a);
  h->alloc = alloc;

  if (data != nullptr) {
    std::memcpy(new_data, data, static_cast<std::size_t>(old_size) * item_size);
    carray_free_internal(data, item_size);
  }
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
