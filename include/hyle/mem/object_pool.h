#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/memory_arena.h"

namespace hyle::mem {

// Typed object pool: fixed-capacity create/destroy of T with slot reuse, on top
// of the lock-free MemoryArena.
//
// Fixes clay bug #3 (object_pool.h:66 vs :79): clay took slots via its Allocator
// but returned them with global `delete`, bypassing the allocator (UB). Here
// create() and destroy() BOTH go through the same MemoryArena -- never global
// new/delete. Construction/destruction are explicit (placement-new / ~T).
//
// Thread-safe: the backing arena is lock-free, so create()/destroy() may run
// concurrently from many threads.
template <class T>
class ObjectPool {
 public:
  ObjectPool(Allocator alloc, std::uint32_t capacity)
      : arena_(alloc, sizeof(T), capacity, alignof(T)) {}

  HYLE_DISABLE_COPY_AND_MOVE(ObjectPool);

  // Construct a T in a pooled slot. Returns nullptr if the pool is exhausted.
  template <class... Args>
  T* create(Args&&... args) {
    void* slot = arena_.allocate();
    if (slot == nullptr) return nullptr;
    return ::new (slot) T(std::forward<Args>(args)...);
  }

  // Destroy a T obtained from create() and return its slot to the pool. Runs
  // ~T then frees through the SAME arena (never global delete -> fixes bug #3).
  void destroy(T* p) noexcept {
    if (p == nullptr) return;
    p->~T();
    arena_.deallocate(p);
  }

  std::uint32_t capacity() const noexcept { return arena_.capacity(); }

 private:
  MemoryArena arena_;
};

}  // namespace hyle::mem
