#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/shared_pool.h"

namespace hyle::mem {

// Typed object pool: fixed-capacity create/destroy of T with slot reuse, on top of a
// sharded shared block pool. create() placement-news a T in a pooled slot; destroy()
// runs ~T then returns the slot -- BOTH through the SAME pool, never global new/delete
// (fixes clay bug #3, where slots were taken via the allocator but freed with global
// delete -- an alloc/free mismatch = UB).
//
// Two flavors mirror the two pools (see shared_pool.h), differing only in the backing:
//
//   ObjectPool<T>        over SharedPool (HARD capacity). A destroyed object's slot is
//                        immediately reusable by any thread; never falsely exhausts.
//                        The safe default -- use it unless you have a measured same-
//                        thread-churn hot spot.
//
//   ObjectPoolCached<T>  over SharedPoolCached (SOFT capacity, per-thread TLS fast
//                        path). Max throughput when a thread reuses its own objects and
//                        the pool is provisioned above the concurrent working set;
//                        a slot freed on thread A is invisible to other threads until A
//                        publishes/exits. Wrong tool for a capacity-tight cross-thread
//                        quota (use ObjectPool<T>).
//
// Both are thread-safe (the backing pool is lock-free): create()/destroy() may run
// concurrently from many threads. Construction/destruction are explicit; T's destructor
// must be non-throwing so destroy() can always return the slot.
template <class T, class Pool>
class BasicObjectPool {
 public:
  static_assert(std::is_nothrow_destructible_v<T>,
                "ObjectPool requires a non-throwing destructor");

  BasicObjectPool(Allocator alloc, std::uint32_t capacity)
      : pool_(alloc, sizeof(T), capacity, alignof(T)) {}

  HYLE_DISABLE_COPY_AND_MOVE(BasicObjectPool);

  // Construct a T in a pooled slot. Returns nullptr if the pool is exhausted.
  template <class... Args>
  T* create(Args&&... args) {
    void* slot = pool_.allocate();
    if (slot == nullptr) return nullptr;
    try {
      return ::new (slot) T(std::forward<Args>(args)...);
    } catch (...) {
      pool_.deallocate(slot);
      throw;
    }
  }

  // Destroy a T from create() and return its slot. Runs ~T then frees through the SAME
  // pool (never global delete -> fixes bug #3).
  void destroy(T* p) noexcept {
    if (p == nullptr) return;
    p->~T();
    pool_.deallocate(p);
  }

  bool valid() const noexcept { return pool_.valid(); }
  std::uint32_t capacity() const noexcept { return pool_.capacity(); }

  // Present only for ObjectPoolCached<T>: publish this thread's destroyed slots so
  // other threads can acquire them without waiting for this thread to exit.
  void publish_thread_cache() noexcept
    requires requires(Pool& p) { p.publish_thread_cache(); }
  {
    pool_.publish_thread_cache();
  }

 private:
  Pool pool_;
};

// HARD-capacity typed pool (default, immediate cross-thread visibility).
template <class T>
using ObjectPool = BasicObjectPool<T, SharedPool>;

// SOFT-capacity typed pool (per-thread TLS fast path, max same-thread throughput).
template <class T>
using ObjectPoolCached = BasicObjectPool<T, SharedPoolCached>;

}  // namespace hyle::mem
