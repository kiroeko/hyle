#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/pool_loan.h"
#include "hyle/mem/shared_pool.h"

namespace hyle::mem {

// Typed object pool: fixed-capacity create/destroy of T with slot reuse, on top
// of a sharded shared block pool. create() placement-news a T in a pooled slot;
// destroy() runs ~T then returns the slot -- BOTH through the SAME pool, never
// global new/delete (fixes clay bug #3, where slots were taken via the
// allocator but freed with global delete -- an alloc/free mismatch = UB).
//
// Two flavors mirror the two pools (see shared_pool.h), differing only in the
// backing:
//
//   ObjectPool<T>        over SharedPool (HARD capacity). A destroyed object's
//   slot is
//                        immediately reusable by any thread; never falsely
//                        exhausts. The safe default -- use it unless you have a
//                        measured same- thread-churn hot spot.
//
//   ObjectPoolCached<T>  over SharedPoolCached (SOFT capacity, per-thread TLS
//   fast
//                        path). Max throughput when a thread reuses its own
//                        objects and the pool is provisioned above the
//                        concurrent working set; a slot freed on thread A is
//                        invisible to other threads until A publishes/exits.
//                        Wrong tool for a capacity-tight cross-thread quota
//                        (use ObjectPool<T>).
//
// Both are thread-safe (the backing pool is lock-free): create()/destroy() may
// run concurrently from many threads. Construction/destruction are explicit;
// T's destructor must be non-throwing so destroy() can always return the slot.
// The pool must outlive every returned T*, and every live object must be
// destroy()'d exactly once through the SAME pool before pool destruction.
// HYLE_TESTING validates per-slot liveness before invoking T's destructor, so
// double destroy and foreign pointers fail before user code runs. Shipping
// builds carry no per-slot diagnostic state or instructions.
template <class T, class Pool> class BasicObjectPool {
public:
  static_assert(std::is_nothrow_destructible_v<T>,
                "ObjectPool requires a non-throwing destructor");

  BasicObjectPool(Allocator alloc, std::uint32_t capacity)
      : pool_(alloc, sizeof(T), capacity, alignof(T)) {}

  HYLE_DISABLE_COPY_AND_MOVE(BasicObjectPool);

  // Construct a T in a pooled slot. Returns nullptr if the pool is exhausted.
  template <class... Args> T *create(Args &&...args) {
    void *slot = pool_.allocate();
    if (slot == nullptr)
      return nullptr;
    try {
      return ::new (slot) T(std::forward<Args>(args)...);
    } catch (...) {
      pool_.deallocate(slot);
      throw;
    }
  }

  // Destroy a T from create() and return its slot. Runs ~T then frees through
  // the SAME pool (never global delete -> fixes bug #3).
  void destroy(T *p) noexcept {
    if (p == nullptr)
      return;
#if defined(HYLE_TESTING)
    // TEST-ONLY: validate before invoking user code, so double-destroy and
    // foreign-pointer bugs trap without running a destructor on invalid storage.
    pool_.testing_assert_checked_out(p);
#endif
    p->~T();
    pool_.deallocate(p);
  }

  void return_object(T *p) noexcept { destroy(p); }

  template <class... Args>
  PoolLoan<T, BasicObjectPool> try_loan(Args &&...args) {
    return PoolLoan<T, BasicObjectPool>(this,
                                        create(std::forward<Args>(args)...));
  }

  bool valid() const noexcept { return pool_.valid(); }
  std::uint32_t capacity() const noexcept { return pool_.capacity(); }

  // Present only for ObjectPoolCached<T>: publish this thread's destroyed slots
  // so other threads can acquire them without waiting for this thread to exit.
  void publish_thread_cache() noexcept
    requires requires(Pool &p) { p.publish_thread_cache(); }
  {
    pool_.publish_thread_cache();
  }

private:
  Pool pool_;
};

// HARD-capacity typed pool (default, immediate cross-thread visibility).
template <class T> using ObjectPool = BasicObjectPool<T, SharedPool>;

// SOFT-capacity typed pool (per-thread TLS fast path, max same-thread
// throughput).
template <class T>
using ObjectPoolCached = BasicObjectPool<T, SharedPoolCached>;

// Fixed-capacity pool for expensive reusable objects. Every T is constructed
// exactly once during pool construction, acquire/release preserves its state,
// and every T is destroyed exactly once with the pool. This is intentionally
// separate from ObjectPoolCached: Preserve is an object-lifetime policy, while
// Cached is a free-slot visibility/concurrency policy.
//
// Constructor arguments are observed as const lvalues and reused for every T.
// Use an object with an explicit reset() method when per-slot runtime
// reinitialization is required.
template <class T, class Pool> class BasicPreservingObjectPool {
public:
  static_assert(std::is_nothrow_destructible_v<T>,
                "PreservingObjectPool requires a non-throwing destructor");

  template <class... Args>
  BasicPreservingObjectPool(Allocator alloc, std::uint32_t capacity,
                            const Args &...args)
      : alloc_(alloc), pool_(alloc, sizeof(T), capacity, alignof(T)),
        capacity_(capacity) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    if (!pool_.valid() || capacity_ == 0 ||
        capacity_ > std::numeric_limits<std::size_t>::max() / sizeof(T *))
      return;
    all_ = alloc_.template allocate_n<T *>(capacity_);
    if (all_ == nullptr)
      return;

    try {
      for (; constructed_ < capacity_; ++constructed_) {
        void *slot = pool_.allocate();
        HYLE_FATAL_ASSERT(slot != nullptr);
        try {
          all_[constructed_] = ::new (slot) T(args...);
        } catch (...) {
          pool_.deallocate(slot);
          throw;
        }
      }
    } catch (...) {
      cleanup_constructed();
      throw;
    }
    for (std::uint32_t i = 0; i < capacity_; ++i)
      pool_.deallocate(all_[i]);
    valid_ = true;
  }

  ~BasicPreservingObjectPool() {
    HYLE_FATAL_ASSERT(outstanding_.load(std::memory_order_relaxed) == 0);
    cleanup_constructed();
  }

  HYLE_DISABLE_COPY_AND_MOVE(BasicPreservingObjectPool);

  bool valid() const noexcept { return valid_; }
  std::uint32_t capacity() const noexcept { return capacity_; }
  std::uint32_t outstanding() const noexcept {
    return outstanding_.load(std::memory_order_relaxed);
  }

  T *acquire() noexcept {
    if (!valid_)
      return nullptr;
    auto *p = static_cast<T *>(pool_.allocate());
    if (p != nullptr)
      outstanding_.fetch_add(1, std::memory_order_relaxed);
    return p;
  }

  void return_object(T *p) noexcept {
    if (p == nullptr)
      return;
#if defined(HYLE_TESTING)
    // TEST-ONLY: keep diagnostic failure ahead of counter underflow and pool
    // mutation. The shipping pool carries no per-slot liveness state.
    pool_.testing_assert_checked_out(p);
#endif
    const std::uint32_t before =
        outstanding_.fetch_sub(1, std::memory_order_relaxed);
    HYLE_FATAL_ASSERT(before != 0);
    pool_.deallocate(p);
  }

  PoolLoan<T, BasicPreservingObjectPool> try_loan() noexcept {
    return PoolLoan<T, BasicPreservingObjectPool>(this, acquire());
  }

  void publish_thread_cache() noexcept
    requires requires(Pool &p) { p.publish_thread_cache(); }
  {
    pool_.publish_thread_cache();
  }

private:
  void cleanup_constructed() noexcept {
    if (all_ == nullptr)
      return;
    for (std::uint32_t i = 0; i < constructed_; ++i) {
      all_[i]->~T();
      // During successful construction the slots were already returned. During
      // an exception they are still checked out and must be returned before
      // pool teardown.
      if (!valid_)
        pool_.deallocate(all_[i]);
    }
    alloc_.deallocate_n(all_, capacity_);
    all_ = nullptr;
    constructed_ = 0;
    valid_ = false;
  }

  Allocator alloc_;
  Pool pool_;
  T **all_ = nullptr;
  std::uint32_t capacity_ = 0;
  std::uint32_t constructed_ = 0;
  std::atomic<std::uint32_t> outstanding_{0};
  bool valid_ = false;
};

template <class T>
using PreservingObjectPool = BasicPreservingObjectPool<T, SharedPool>;

template <class T>
using PreservingObjectPoolCached =
    BasicPreservingObjectPool<T, SharedPoolCached>;

} // namespace hyle::mem
