#include "hyle/container/pod_vector.h"

#include <limits>

#include "hyle/core/compiler.h"

namespace hyle::container::detail {
namespace {

bool allocation_bytes(std::size_t capacity, std::size_t item_size,
                      std::size_t* bytes) noexcept {
  HYLE_FATAL_ASSERT(item_size != 0);
  if (capacity > (std::numeric_limits<std::size_t>::max)() / item_size)
    return false;
  *bytes = capacity * item_size;
  return true;
}

}  // namespace

void* pod_vector_reallocate_internal(
    void* data, std::size_t old_capacity, std::size_t new_capacity,
    std::size_t item_size, std::size_t alignment,
    mem::AllocatorRef alloc) noexcept {
  HYLE_FATAL_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
  HYLE_FATAL_ASSERT(data != nullptr || old_capacity == 0);
  HYLE_FATAL_ASSERT(new_capacity != 0);

  std::size_t old_bytes = 0;
  std::size_t new_bytes = 0;
  if (!allocation_bytes(old_capacity, item_size, &old_bytes) ||
      !allocation_bytes(new_capacity, item_size, &new_bytes))
    return nullptr;

  return data == nullptr ? alloc.allocate(new_bytes, alignment)
                         : alloc.reallocate(data, old_bytes, new_bytes,
                                            alignment);
}

void pod_vector_free_internal(void* data, std::size_t capacity,
                              std::size_t item_size, std::size_t alignment,
                              mem::AllocatorRef alloc) noexcept {
  if (data == nullptr) return;
  std::size_t bytes = 0;
  HYLE_FATAL_ASSERT(allocation_bytes(capacity, item_size, &bytes));
  alloc.deallocate(data, bytes, alignment);
}

}  // namespace hyle::container::detail
