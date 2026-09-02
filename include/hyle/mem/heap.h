#pragma once

#include <cstddef>

#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/visibility.h"

namespace hyle::mem {

// An isolated, move-only host heap with individual free and explicit bulk
// release. Allocation/reallocation are owner-thread operations; deallocation
// may occur on another thread. clear()/release()/destruction invalidate every
// outstanding allocation, so the owner must first quiesce all users.
//
// allocator() borrows this Heap object rather than its current native heap.
// Consequently a borrow remains usable after a successful clear(), but moving
// or destroying the Heap invalidates every borrow. The wrapper itself is one
// pointer and never allocates a HYLE control block.
class Heap {
 public:
  HYLE_MEM_API static Result<Heap> create() noexcept;

  constexpr Heap() noexcept = default;
  Heap(const Heap&) = delete;
  Heap& operator=(const Heap&) = delete;
  HYLE_MEM_API Heap(Heap&& other) noexcept;
  HYLE_MEM_API Heap& operator=(Heap&& other) noexcept;
  HYLE_MEM_API ~Heap();

  bool valid() const noexcept { return native_heap_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  // A four-word runtime allocator borrow. Invalid Heap -> invalid allocator.
  HYLE_MEM_API AllocatorRef allocator() noexcept;

  // Destroy every allocation and replace the backing with a fresh isolated
  // heap. On replacement allocation failure the old heap and its allocations
  // remain unchanged. Existing allocator() borrows remain bound to this owner.
  HYLE_MEM_API Status clear() noexcept;

  // Return unused pages owned by this heap to its backing/OS when possible.
  // This does not invalidate live allocations.
  HYLE_MEM_API void collect(bool force = false) noexcept;

  // Destroy every allocation and leave this owner invalid. Idempotent.
  HYLE_MEM_API void release() noexcept;

 private:
  explicit Heap(void* native_heap) noexcept : native_heap_(native_heap) {}

  static void* allocate_erased(void* state, std::size_t size,
                               std::size_t alignment) noexcept;
  static void deallocate_erased(void* state, void* pointer, std::size_t size,
                                std::size_t alignment) noexcept;
  static void* reallocate_erased(void* state, void* pointer,
                                 std::size_t old_size,
                                 std::size_t new_size,
                                 std::size_t alignment) noexcept;

  void* native_heap_ = nullptr;
};

static_assert(sizeof(Heap) == sizeof(void*));

}  // namespace hyle::mem
