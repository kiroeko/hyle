#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/thread.h" // hardware_concurrency() -> shard count

// Shared fixed-size block pools (allocator-injected). Two public pools over one
// sharded-slab core, differing ONLY in the alloc/free hot path:
//
//   SharedPool        HARD capacity. In normal tagged mode, the successful
//   home-shard
//                     path and every deallocate use one CAS; allocate scans if
//                     needed. A freed block is never hidden in TLS, so
//                     completed frees are immediately visible. This is the safe
//                     default for a bounded quota that is drawn/returned across
//                     threads (work-stealing handoff, N fixed slots). Pick this
//                     unless you have measured a same- thread-churn hot spot
//                     that needs the cached pool below.
//
//   SharedPoolCached  SOFT capacity, MAX throughput. Adds a per-thread TLS
//   freelist
//                     (owner-only, ZERO atomics on a same-thread free->alloc)
//                     with magazine batch refill/flush to the shards. Wins big
//                     when a thread reuses its OWN blocks and the pool is
//                     provisioned well above the concurrent working set.
//                     TRADE-OFF: a block freed on thread A is parked in A's TLS
//                     and is NOT visible to allocate() on other threads until A
//                     publishes/exits (see "Soft capacity"). Wrong tool for a
//                     capacity-tight cross-thread quota.
//
// Both borrow slab + link array from the injected Allocator. Each shard uses a
// 32-bit tagged 64-bit-CAS fast path; before the finite tag could wrap, that
// shard permanently enters a serialized 64-bit-change-counter fallback, so
// stale CAS can never become valid again. The free lists are sharded per core:
// clay's single-head toy ran 278x slower than per-thread heaps under 8-way
// churn. The `next` links live in a PARALLEL index array, never intrusively in
// user bytes.
//
// PooledRegion (bottom of file) is the single-threaded bump allocator over ONE
// borrowed pool block -- a different tool (bulk release, no per-object free),
// kept here because it borrows from a pool.

namespace hyle::mem {

namespace detail {
// Process-global (cross-DLL) lock guarding a cached pool's TLS-cache registry.
// One definition lives in hyle_mem (shared_pool.cc) so every header-inlined
// consumer serializes on the SAME mutex -- a per-DLL header static would let
// two parties race on one node through different locks. COLD path only (attach
// / detach / reset / dtor / thread-exit); the alloc/free hot path never touches
// it.
HYLE_MEM_API void tls_registry_lock() noexcept;
HYLE_MEM_API void tls_registry_unlock() noexcept;

struct TlsRegistryGuard {
  TlsRegistryGuard() noexcept { tls_registry_lock(); }
  ~TlsRegistryGuard() noexcept { tls_registry_unlock(); }
  TlsRegistryGuard(const TlsRegistryGuard &) = delete;
  TlsRegistryGuard &operator=(const TlsRegistryGuard &) = delete;
};

// ---- Sharded-slab core (shared by SharedPool + SharedPoolCached)
// ---------------- Owns the slab + parallel next_[] link array + per-shard
// cache-line-isolated Treiber heads and their tag-saturation fallback. NOT a
// standalone type: the two public pools derive from it and add their own
// alloc/free semantics. No virtuals -- the derived hot paths call these inline,
// zero indirection.
class ShardedSlab {
public:
  std::size_t block_size() const noexcept { return block_size_; }
  std::uint32_t capacity() const noexcept { return count_; }
  // False iff a construction-time allocation (slab / link array / heads) hit
  // OOM. An invalid pool hands out nothing; the C ABI checks this and returns
  // NULL instead of exposing a dead handle.
  bool valid() const noexcept { return valid_; }

protected:
  static constexpr std::uint32_t kNull = 0xFFFFFFFFu;
  static constexpr std::uint32_t kSaturatedVersion = 0xFFFFFFFFu;

  struct EmptyObservation {
    std::uint64_t head;
    std::uint64_t saturated_changes;
  };

#if defined(HYLE_TESTING)
  using EmptyScanHook = void (*)(void *, std::uint32_t) noexcept;
#endif

  // `block_size` bytes per block, `block_count` blocks, each aligned to
  // `alignment` (power of two). Slab + link array + heads come from `alloc`. On
  // OOM: valid()==false (graceful, not abort; mem-audit-followup #2 arch B).
  // The dtor is null-safe.
  ShardedSlab(Allocator alloc, std::size_t block_size,
              std::uint32_t block_count, std::size_t alignment)
      : alloc_(alloc), block_size_(block_size),
        alignment_(alignment < alignof(std::max_align_t)
                       ? alignof(std::max_align_t)
                       : alignment),
        count_(block_count) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(block_count >= 1 && block_count < kNull);
    HYLE_FATAL_ASSERT(block_size_ >= 1);
    HYLE_FATAL_ASSERT((alignment_ & (alignment_ - 1)) == 0);
    // Overflow guards (contract violation -> fail-fast): the round-up must not
    // wrap, and stride_ * count_ must fit size_t. Real only on 32-bit; cheap
    // insurance.
    HYLE_FATAL_ASSERT(block_size_ <= SIZE_MAX - (alignment_ - 1));
    stride_ = (block_size_ + alignment_ - 1) & ~(alignment_ - 1); // round up
    HYLE_FATAL_ASSERT(static_cast<std::size_t>(count_) <= SIZE_MAX / stride_);
    const std::size_t slab_bytes = stride_ * static_cast<std::size_t>(count_);
    // The injected backing is authoritative. Hugepage/NUMA placement is an
    // explicit allocator/provider policy; allocator identity must never
    // silently change a pool's storage domain or accounting behavior.
    slab_ = static_cast<std::byte *>(alloc_.allocate(slab_bytes, alignment_));
    if (slab_ == nullptr)
      return; // slab OOM -> invalid (dtor null-safe)
    next_ = static_cast<std::atomic<std::uint32_t> *>(
        alloc_.allocate(sizeof(std::atomic<std::uint32_t>) * count_,
                        alignof(std::atomic<std::uint32_t>)));
    if (next_ == nullptr)
      return; // link-array OOM -> invalid (dtor frees slab_)
    // Shard count = core count rounded up to pow2 (capped), never more than
    // blocks. Scaling to cores keeps it O(1) up to real concurrency; a FIXED
    // count would re-introduce O(ceil(T/shards)) contention past that. Pow2
    // makes homing a mask.
    num_shards_ = default_shards();
    if (num_shards_ > count_) {
      std::uint32_t p = 1;
      while (p * 2u <= count_)
        p <<= 1; // largest pow2 <= count_ (keep mask exact)
      num_shards_ = p;
    }
    shard_mask_ = num_shards_ - 1; // pow2 -> exact modulo via AND
    per_ = count_ / num_shards_;   // >=1; the last shard absorbs the remainder
    heads_ = static_cast<ShardHead *>(
        alloc_.allocate(sizeof(ShardHead) * num_shards_, alignof(ShardHead)));
    if (heads_ == nullptr)
      return; // heads OOM -> invalid (dtor frees slab_/next_)
    for (std::uint32_t s = 0; s < num_shards_; ++s)
      new (&heads_[s]) ShardHead();
    // Seed each shard from a contiguous index range [lo,hi). This is only the
    // initial distribution; normal frees migrate blocks toward the freeing
    // thread's home shard.
    for (std::uint32_t s = 0; s < num_shards_; ++s) {
      std::uint32_t lo = s * per_;
      std::uint32_t hi = (s + 1 == num_shards_) ? count_ : lo + per_;
      for (std::uint32_t i = lo; i < hi; ++i)
        new (&next_[i]) std::atomic<std::uint32_t>(i + 1 < hi ? i + 1 : kNull);
      heads_[s].h.store(pack(lo, 0), std::memory_order_relaxed);
    }
#if defined(HYLE_TESTING)
    // TEST-ONLY: per-slot liveness is deliberately absent from shipping pools.
    // It makes contract violations deterministic under the certification build
    // without adding an atomic, byte, or branch to the released type.
    testing_live_ =
        new (std::nothrow) std::atomic<std::uint8_t>[count_];
    HYLE_FATAL_ASSERT(testing_live_ != nullptr);
    for (std::uint32_t i = 0; i < count_; ++i)
      testing_live_[i].store(0, std::memory_order_relaxed);
#endif
    valid_ = true;
  }

  ~ShardedSlab() {
    if (valid_) {
#if defined(HYLE_TESTING)
      testing_assert_no_outstanding();
#endif
      for (std::uint32_t i = 0; i < count_; ++i)
        next_[i].~atomic();
      for (std::uint32_t s = 0; s < num_shards_; ++s)
        heads_[s].~ShardHead();
    }
    if (heads_ != nullptr)
      alloc_.deallocate(heads_, sizeof(ShardHead) * num_shards_,
                        alignof(ShardHead));
    if (next_ != nullptr)
      alloc_.deallocate(next_, sizeof(std::atomic<std::uint32_t>) * count_,
                        alignof(std::atomic<std::uint32_t>));
    if (slab_ != nullptr) {
      alloc_.deallocate(slab_, stride_ * count_, alignment_);
    }
#if defined(HYLE_TESTING)
    delete[] testing_live_;
#endif
  }

  void *checkout(std::uint32_t idx) noexcept {
    if (idx == kNull)
      return nullptr;
#if defined(HYLE_TESTING)
    HYLE_FATAL_ASSERT(testing_live_ != nullptr && idx < count_);
    HYLE_FATAL_ASSERT(
        testing_live_[idx].exchange(1, std::memory_order_acq_rel) == 0);
#endif
    return block_at(idx);
  }

  std::uint32_t return_index(void *p) noexcept {
    const std::uint32_t idx = idx_of_ptr(p);
#if defined(HYLE_TESTING)
    HYLE_FATAL_ASSERT(testing_live_ != nullptr && idx < count_);
    HYLE_FATAL_ASSERT(
        testing_live_[idx].exchange(0, std::memory_order_acq_rel) == 1);
#endif
    return idx;
  }

  void assert_checked_out(void *p) const noexcept {
    const std::uint32_t idx = idx_of_ptr(p);
#if defined(HYLE_TESTING)
    HYLE_FATAL_ASSERT(testing_live_ != nullptr && idx < count_);
    HYLE_FATAL_ASSERT(testing_live_[idx].load(std::memory_order_acquire) == 1);
#else
    (void)idx;
#endif
  }

#if defined(HYLE_TESTING)
  void testing_assert_no_outstanding() const noexcept {
    for (std::uint32_t i = 0; i < count_; ++i)
      HYLE_FATAL_ASSERT(testing_live_[i].load(std::memory_order_acquire) == 0);
  }

  void testing_discard_all() noexcept {
    for (std::uint32_t i = 0; i < count_; ++i)
      testing_live_[i].store(0, std::memory_order_release);
  }
#endif

  HYLE_DISABLE_COPY_AND_MOVE(ShardedSlab);

  // idx <-> pointer.
  void *block_at(std::uint32_t idx) const noexcept {
    return slab_ + static_cast<std::size_t>(idx) * stride_;
  }
  // Pointer -> block index, with pool-membership + alignment fatal-asserts (a
  // contract violation -> fail-fast). The alignment check is `idx*stride_ ==
  // off` (one multiply) not a second `off % stride_` (one idiv, not two, on the
  // free hot path).
  std::uint32_t idx_of_ptr(void *p) const noexcept {
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    const auto begin = reinterpret_cast<std::uintptr_t>(slab_);
    const std::size_t slab_bytes = stride_ * static_cast<std::size_t>(count_);
    HYLE_FATAL_ASSERT(addr >= begin && addr - begin < slab_bytes);
    const auto off = static_cast<std::size_t>(addr - begin);
    std::uint32_t idx = static_cast<std::uint32_t>(off / stride_);
    HYLE_FATAL_ASSERT(idx < count_ &&
                      static_cast<std::size_t>(idx) * stride_ == off);
    return idx;
  }

  // Stable shared-list home for the calling thread. Correctness does not depend
  // on the choice; fallback scans can take any shard. num_shards_ is pow2 ->
  // one mask.
  std::uint32_t home_shard() const noexcept {
    return tls_thread_seq() & shard_mask_;
  }

  // Pop one free index from shard s (kNull if empty). When empty, return the
  // exact tagged head observed so a caller can validate a stable all-shard
  // snapshot.
  std::uint32_t pop_shard(std::uint32_t s,
                          EmptyObservation *empty = nullptr) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      if (ver_of(old) == kSaturatedVersion)
        return pop_shard_saturated(s, empty);
      std::uint32_t idx = idx_of(old);
      if (idx == kNull) {
        if (empty != nullptr)
          *empty = EmptyObservation{old, 0};
        return kNull;
      }
      std::uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
      std::uint64_t desired = pack(nxt, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire))
        return idx;
    }
  }

  // Push one index back onto shard s.
  void push_shard(std::uint32_t s, std::uint32_t idx) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      if (ver_of(old) == kSaturatedVersion) {
        push_shard_saturated(s, idx);
        return;
      }
      next_[idx].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(idx, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire))
        return;
    }
  }

  // Pop up to `want` indices from shard s in ONE CAS (magazine refill). Writes
  // taken indices into out[], returns the count (0 if empty). The walk reads
  // next_[] of nodes STILL on the shard, but nothing commits until the CAS
  // gates on the full packed `old` (idx AND version): any concurrent pop/push
  // on this shard bumps the version, so a CAS that still sees `old` proves the
  // top-K chain was untouched since the load -> the walked `new_top` is valid.
  // Otherwise retry. At tag saturation this operation switches to the
  // serialized no-wrap path below.
  std::uint32_t pop_shard_batch(std::uint32_t s, std::uint32_t want,
                                std::uint32_t *out,
                                EmptyObservation *empty = nullptr) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      if (ver_of(old) == kSaturatedVersion)
        return pop_shard_batch_saturated(s, want, out, empty);
      std::uint32_t first = idx_of(old);
      if (first == kNull) {
        if (empty != nullptr)
          *empty = EmptyObservation{old, 0};
        return 0;
      }
      std::uint32_t n = 1, last = first;
      std::uint32_t nxt = next_[last].load(std::memory_order_relaxed);
      while (n < want && nxt != kNull) {
        last = nxt;
        nxt = next_[last].load(std::memory_order_relaxed);
        ++n;
      }
      std::uint64_t desired =
          pack(nxt, ver_of(old) + 1); // nxt == new top (or kNull)
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        std::uint32_t cur = first;
        for (std::uint32_t i = 0; i < n; ++i) { // chain now privately owned
          out[i] = cur;
          cur = next_[cur].load(std::memory_order_relaxed);
        }
        return n;
      }
    }
  }

  void notify_empty_scan(std::uint32_t s) noexcept {
#if defined(HYLE_TESTING)
    if (testing_empty_scan_hook_ != nullptr)
      testing_empty_scan_hook_(testing_empty_scan_state_, s);
#else
    (void)s;
#endif
  }

  // Pop from the home shard or any fallback shard. A failed first collect is
  // not enough: independent shard snapshots can otherwise miss a free that
  // moves behind the scan. A second identical tagged-head collect proves every
  // shard remained empty over one common interval (barring the documented
  // finite tag wrap).
  std::uint32_t pop_any_shard(std::uint32_t home) noexcept {
    thread_local std::uint32_t last_remote = 0; // hint only; holds no block
    EmptyObservation observed[kShardCap];
    for (;;) {
      std::uint32_t idx = pop_shard(home, &observed[home]);
      if (idx != kNull)
        return idx;
      notify_empty_scan(home);
      const std::uint32_t hint = last_remote & shard_mask_;
      if (hint != home) {
        idx = pop_shard(hint, &observed[hint]);
        if (idx != kNull)
          return idx;
        notify_empty_scan(hint);
      }
      for (std::uint32_t k = 1; k < num_shards_; ++k) {
        const std::uint32_t s = (home + k) & shard_mask_;
        if (s == hint)
          continue;
        idx = pop_shard(s, &observed[s]);
        if (idx != kNull) {
          last_remote = s;
          return idx;
        }
        notify_empty_scan(s);
      }
      bool stable = true;
      for (std::uint32_t s = 0; s < num_shards_; ++s) {
        if (!empty_observation_unchanged(s, observed[s])) {
          stable = false;
          break;
        }
      }
      if (stable)
        return kNull;
    }
  }

  std::uint32_t pop_any_shard_batch(std::uint32_t home, std::uint32_t want,
                                    std::uint32_t *out) noexcept {
    thread_local std::uint32_t last_remote = 0; // hint only; holds no block
    EmptyObservation observed[kShardCap];
    for (;;) {
      std::uint32_t n = pop_shard_batch(home, want, out, &observed[home]);
      if (n != 0)
        return n;
      const std::uint32_t hint = last_remote & shard_mask_;
      if (hint != home) {
        n = pop_shard_batch(hint, want, out, &observed[hint]);
        if (n != 0)
          return n;
      }
      for (std::uint32_t k = 1; k < num_shards_; ++k) {
        const std::uint32_t s = (home + k) & shard_mask_;
        if (s == hint)
          continue;
        n = pop_shard_batch(s, want, out, &observed[s]);
        if (n != 0) {
          last_remote = s;
          return n;
        }
      }
      bool stable = true;
      for (std::uint32_t s = 0; s < num_shards_; ++s) {
        if (!empty_observation_unchanged(s, observed[s])) {
          stable = false;
          break;
        }
      }
      if (stable)
        return 0;
    }
  }

  // Push a private run of `n` indices in ONE CAS (magazine flush). Blocks may
  // migrate between shards: once popped, an index belongs to no shard until
  // this CAS publishes it. Relinking next_[] is therefore race-free; the tag
  // protects stale source-head operations even when the node later appears on
  // another shard.
  void push_shard_batch(std::uint32_t s, const std::uint32_t *idxs,
                        std::uint32_t n) noexcept {
    if (n == 0)
      return;
    for (std::uint32_t i = n - 1; i > 0; --i)
      next_[idxs[i]].store(idxs[i - 1], std::memory_order_relaxed);
    const std::uint32_t chain_head = idxs[n - 1];
    const std::uint32_t chain_tail = idxs[0];
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      if (ver_of(old) == kSaturatedVersion) {
        push_shard_batch_saturated(s, chain_head, chain_tail);
        return;
      }
      next_[chain_tail].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(chain_head, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire))
        return;
    }
  }

  // A 32-bit fast-path tag is finite. Instead of wrapping back to a stale
  // value, version 0xFFFFFFFF permanently switches that shard to a serialized
  // fallback. This preserves correctness for arbitrarily long-lived pools; only
  // a shard that has already committed 2^32-1 mutations pays the fallback lock.
  void lock_saturated(std::uint32_t s) noexcept {
    while (heads_[s].saturated_lock.test_and_set(std::memory_order_acquire))
      heads_[s].saturated_lock.wait(true, std::memory_order_relaxed);
  }

  void unlock_saturated(std::uint32_t s) noexcept {
    heads_[s].saturated_lock.clear(std::memory_order_release);
    heads_[s].saturated_lock.notify_one();
  }

  std::uint32_t pop_shard_saturated(std::uint32_t s,
                                    EmptyObservation *empty) noexcept {
    lock_saturated(s);
    const std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
    HYLE_FATAL_ASSERT(ver_of(old) == kSaturatedVersion);
    const std::uint32_t idx = idx_of(old);
    if (idx == kNull) {
      if (empty != nullptr) {
        *empty = EmptyObservation{
            old, heads_[s].saturated_changes.load(std::memory_order_relaxed)};
      }
      unlock_saturated(s);
      return kNull;
    }
    const std::uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
    heads_[s].h.store(pack(nxt, kSaturatedVersion), std::memory_order_relaxed);
    heads_[s].saturated_changes.fetch_add(1, std::memory_order_relaxed);
    unlock_saturated(s);
    return idx;
  }

  void push_shard_saturated(std::uint32_t s, std::uint32_t idx) noexcept {
    lock_saturated(s);
    const std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
    HYLE_FATAL_ASSERT(ver_of(old) == kSaturatedVersion);
    next_[idx].store(idx_of(old), std::memory_order_relaxed);
    heads_[s].h.store(pack(idx, kSaturatedVersion), std::memory_order_relaxed);
    heads_[s].saturated_changes.fetch_add(1, std::memory_order_relaxed);
    unlock_saturated(s);
  }

  std::uint32_t pop_shard_batch_saturated(std::uint32_t s, std::uint32_t want,
                                          std::uint32_t *out,
                                          EmptyObservation *empty) noexcept {
    lock_saturated(s);
    const std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
    HYLE_FATAL_ASSERT(ver_of(old) == kSaturatedVersion);
    std::uint32_t first = idx_of(old);
    if (first == kNull) {
      if (empty != nullptr) {
        *empty = EmptyObservation{
            old, heads_[s].saturated_changes.load(std::memory_order_relaxed)};
      }
      unlock_saturated(s);
      return 0;
    }
    std::uint32_t n = 1;
    std::uint32_t last = first;
    std::uint32_t nxt = next_[last].load(std::memory_order_relaxed);
    while (n < want && nxt != kNull) {
      last = nxt;
      nxt = next_[last].load(std::memory_order_relaxed);
      ++n;
    }
    heads_[s].h.store(pack(nxt, kSaturatedVersion), std::memory_order_relaxed);
    heads_[s].saturated_changes.fetch_add(1, std::memory_order_relaxed);
    std::uint32_t cur = first;
    for (std::uint32_t i = 0; i < n; ++i) {
      out[i] = cur;
      cur = next_[cur].load(std::memory_order_relaxed);
    }
    unlock_saturated(s);
    return n;
  }

  void push_shard_batch_saturated(std::uint32_t s, std::uint32_t chain_head,
                                  std::uint32_t chain_tail) noexcept {
    lock_saturated(s);
    const std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
    HYLE_FATAL_ASSERT(ver_of(old) == kSaturatedVersion);
    next_[chain_tail].store(idx_of(old), std::memory_order_relaxed);
    heads_[s].h.store(pack(chain_head, kSaturatedVersion),
                      std::memory_order_relaxed);
    heads_[s].saturated_changes.fetch_add(1, std::memory_order_relaxed);
    unlock_saturated(s);
  }

  bool empty_observation_unchanged(std::uint32_t s,
                                   EmptyObservation observed) noexcept {
    if (ver_of(observed.head) != kSaturatedVersion) {
      return heads_[s].h.load(std::memory_order_acquire) == observed.head;
    }
    lock_saturated(s);
    const bool same =
        heads_[s].h.load(std::memory_order_relaxed) == observed.head &&
        heads_[s].saturated_changes.load(std::memory_order_relaxed) ==
            observed.saturated_changes;
    unlock_saturated(s);
    return same;
  }

  // Rebuild every shard's free list to its full contiguous range (reset).
  // O(count). Contract: NO block may be live, and callers quiesce concurrent
  // alloc/free first.
  void reset_shards() noexcept {
    for (std::uint32_t s = 0; s < num_shards_; ++s) {
      std::uint32_t lo = s * per_;
      std::uint32_t hi = (s + 1 == num_shards_) ? count_ : lo + per_;
      for (std::uint32_t i = lo; i < hi; ++i)
        next_[i].store(i + 1 < hi ? i + 1 : kNull, std::memory_order_relaxed);
      // Bump without ever wrapping a saturated tag back to zero.
      const std::uint32_t old_ver =
          ver_of(heads_[s].h.load(std::memory_order_relaxed));
      const std::uint32_t ver =
          old_ver == kSaturatedVersion ? kSaturatedVersion : old_ver + 1;
      heads_[s].h.store(pack(lo, ver), std::memory_order_release);
      if (old_ver == kSaturatedVersion)
        heads_[s].saturated_changes.fetch_add(1, std::memory_order_relaxed);
    }
  }

#if defined(HYLE_TESTING)
  void testing_set_empty_scan_hook(EmptyScanHook hook, void *state) noexcept {
    testing_empty_scan_hook_ = hook;
    testing_empty_scan_state_ = state;
  }

  void *testing_pop_from_shard(std::uint32_t s) noexcept {
    const std::uint32_t idx = pop_shard(s);
    return checkout(idx);
  }

  void testing_push_to_shard(std::uint32_t s, void *p) noexcept {
    push_shard(s, return_index(p));
  }

  void *testing_allocate_from_home(std::uint32_t home) noexcept {
    const std::uint32_t idx = pop_any_shard(home);
    return checkout(idx);
  }

  // Quiescent-only hook: force the no-wrap path without executing billions of
  // ops.
  void testing_force_tag_saturation() noexcept {
    for (std::uint32_t s = 0; s < num_shards_; ++s) {
      const std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
      heads_[s].h.store(pack(idx_of(old), kSaturatedVersion),
                        std::memory_order_relaxed);
      heads_[s].saturated_changes.store(0, std::memory_order_relaxed);
    }
  }
#endif

  Allocator alloc_;
  std::size_t block_size_;
  std::size_t stride_;
  std::size_t alignment_;
  std::uint32_t count_;

private:
  static std::uint64_t pack(std::uint32_t idx, std::uint32_t ver) noexcept {
    return (static_cast<std::uint64_t>(ver) << 32) | idx;
  }
  static std::uint32_t idx_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h);
  }
  static std::uint32_t ver_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h >> 32);
  }

  // Process-wide rotating thread sequence -> a stable per-thread shared-list
  // home. Correctness scans all shards, so a per-DLL copy only changes
  // distribution.
  static std::uint32_t tls_thread_seq() noexcept {
    static std::atomic<std::uint32_t> g{0};
    thread_local std::uint32_t id = g.fetch_add(1, std::memory_order_relaxed);
    return id;
  }

  static constexpr std::uint32_t kShardCap = 128; // bounds heads_ (128 -> 8 KB)
  static std::uint32_t default_shards() noexcept {
    unsigned c = hyle::platform::hardware_concurrency();
    if (c == 0)
      c = 8; // unknown -> a sane default
    std::uint32_t p = 1;
    while (p < c && p < kShardCap)
      p <<= 1;
    return p;
  }

  // One Treiber-stack head per shard, each on its OWN cache line (kill false
  // sharing).
  struct alignas(HYLE_CACHE_LINE_SIZE) ShardHead {
    std::atomic<std::uint64_t>
        h; // index|version (lock-free until tag saturation)
    std::atomic_flag saturated_lock = ATOMIC_FLAG_INIT;
    std::atomic<std::uint64_t> saturated_changes{0};
  };

  std::byte *slab_ = nullptr;
  std::atomic<std::uint32_t> *next_ = nullptr; // parallel free-list links
  ShardHead *heads_ = nullptr; // num_shards_ cache-line-isolated heads
  std::uint32_t num_shards_ = 0;
  std::uint32_t shard_mask_ = 0;
  std::uint32_t per_ = 0;
#if defined(HYLE_TESTING)
  EmptyScanHook testing_empty_scan_hook_ = nullptr;
  void *testing_empty_scan_state_ = nullptr;
  std::atomic<std::uint8_t> *testing_live_ = nullptr;
#endif

protected:
  bool valid_ =
      false; // set true once the ctor fully built the pool (else OOM'd)
};
} // namespace detail

// ---- SharedPool: HARD capacity, no off-shard cache
// ------------------------------- The home-shard success path and deallocate
// each use one CAS; allocate scans the other shards when home is empty. A
// completed free is immediately visible because no block is parked in TLS (no
// cache-induced false exhaustion). Choose this for a bounded quota
// drawn/returned across threads (fixed slot count, work-stealing handoff) or
// whenever deterministic capacity matters more than squeezing out the
// same-thread fast path. Same-thread churn still pays one CAS/op; if that is a
// measured hot spot and the pool is over-provisioned, use SharedPoolCached
// instead.
//
// Lifetime contract: every live block is returned exactly once to the SAME pool
// before reset/destruction, with alloc/free externally quiesced.
// Foreign/misaligned pointers fail-fast in every build. HYLE_TESTING adds
// per-slot liveness, double-return and outstanding-allocation diagnostics;
// shipping builds carry no per-slot diagnostic state or instructions.
class SharedPool : public detail::ShardedSlab {
public:
  SharedPool(Allocator alloc, std::size_t block_size, std::uint32_t block_count,
             std::size_t alignment = alignof(std::max_align_t))
      : detail::ShardedSlab(alloc, block_size, block_count, alignment) {}

#if defined(HYLE_TESTING)
  using TestingEmptyScanHook = void (*)(void *, std::uint32_t) noexcept;

  void testing_force_tag_saturation() noexcept {
    detail::ShardedSlab::testing_force_tag_saturation();
  }
  void testing_set_empty_scan_hook(TestingEmptyScanHook hook,
                                   void *state) noexcept {
    detail::ShardedSlab::testing_set_empty_scan_hook(hook, state);
  }
  void *testing_pop_from_shard(std::uint32_t shard) noexcept {
    return detail::ShardedSlab::testing_pop_from_shard(shard);
  }
  void testing_push_to_shard(std::uint32_t shard, void *p) noexcept {
    detail::ShardedSlab::testing_push_to_shard(shard, p);
  }
  void *testing_allocate_from_home(std::uint32_t home) noexcept {
    return detail::ShardedSlab::testing_allocate_from_home(home);
  }
#endif

  // Pop a free block. nullptr after this call observes every shard empty; no
  // free block can be hidden in a thread-local cache.
  void *allocate() noexcept {
    if (!valid_)
      return nullptr;
    const std::uint32_t idx = pop_any_shard(home_shard());
    return checkout(idx);
  }

  // Return a block from allocate() to the freeing thread's shared home shard.
  // Blocks may migrate between shards (the mature local-list shape); they are
  // never hidden in TLS, so the completed free is immediately visible. null is
  // a no-op.
  void deallocate(void *p) noexcept {
    if (p == nullptr || !valid_)
      return;
    push_shard(home_shard(), return_index(p));
  }

  // Return EVERY block to the pool in one shot. All prior pointers are bulk-
  // invalidated and must never be used again; quiesce concurrent alloc/free.
  void reset() noexcept {
    if (valid_) {
#if defined(HYLE_TESTING)
      testing_discard_all();
#endif
      reset_shards();
    }
  }

#if defined(HYLE_TESTING)
  // TEST-ONLY: assert before a typed wrapper runs the destructor, so a double
  // destroy traps before invoking user code twice.
  void testing_assert_checked_out(void *p) const noexcept {
    assert_checked_out(p);
  }
#endif

  // Adapt to the complete Allocator contract. A live fixed block resizes in
  // place while the request still fits; an over-capacity grow fails without
  // releasing it. null/zero follow the universal allocate/free rules.
  Allocator as_allocator() noexcept {
    return Allocator(&pool_alloc, &pool_free, &pool_realloc, this);
  }

private:
  static void *pool_alloc(void *state, std::size_t size,
                          std::size_t alignment) noexcept {
    auto *self = static_cast<SharedPool *>(state);
    if (size == 0)
      return nullptr;
    HYLE_FATAL_ASSERT(is_pow2(alignment) && size <= self->block_size_ &&
                      alignment <= self->alignment_);
    return self->allocate();
  }
  static void pool_free(void *state, void *ptr, std::size_t,
                        std::size_t) noexcept {
    if (ptr != nullptr)
      static_cast<SharedPool *>(state)->deallocate(ptr);
  }
  static void *pool_realloc(void *state, void *ptr, std::size_t old_size,
                            std::size_t new_size,
                            std::size_t alignment) noexcept {
    auto *self = static_cast<SharedPool *>(state);
    if (ptr == nullptr)
      return pool_alloc(state, new_size, alignment);
    HYLE_FATAL_ASSERT(is_pow2(alignment) && old_size > 0 &&
                      old_size <= self->block_size_ &&
                      alignment <= self->alignment_);
    self->assert_checked_out(ptr);
    if (new_size == 0) {
      self->deallocate(ptr);
      return nullptr;
    }
    if (new_size > self->block_size_)
      return nullptr;
    return ptr;
  }
};

// ---- SharedPoolCached: SOFT capacity, per-thread TLS fast path + magazine
// ---------- Adds a per-(thread,pool) TLS freelist so a same-thread free->alloc
// is a plain index stack with ZERO atomics -- the steady-state hot path that
// beats a per-op CAS. TLS miss/overflow transfer to the shared shards in
// BATCHES (magazine, kRefillBatch): one CAS moves K indices, so a working set
// past the TLS cap pays ~1/K CAS per block.
//
// Soft capacity (intentional, same class as tbb/jemalloc tcache): a block freed
// on a LIVE thread is parked in that thread's TLS and is invisible to
// allocate() on other threads until it flushes (publish_thread_cache / overflow
// / thread-exit / dtor). Window <= kTlsCacheCap per live thread. This means the
// pool can report empty while free blocks exist off-shard -- fine when the pool
// is provisioned above the concurrent working set (the intended use), a footgun
// for a capacity-tight cross-thread quota (use SharedPool for that).
// publish_thread_cache() / join restore hard capacity.
//
// The TLS node is allocated from system_allocator(), NOT the injected backing:
// a node can outlive the pool (a live thread's cache is freed only when THAT
// thread exits), and the injected backing may be gone by then (a C-ABI wrapper
// frees its backing when the pool is destroyed). A node is internal bookkeeping
// (like the registry lock), so a process-stable allocator decouples node
// lifetime from the pool's injected allocator. Every live block must still be
// returned exactly once to the SAME pool before reset/ destruction.
// Foreign/misaligned pointers fail-fast in every build. HYLE_TESTING adds
// per-slot liveness, double-return and outstanding-allocation diagnostics;
// shipping builds carry no per-slot diagnostic state or instructions.
class SharedPoolCached : public detail::ShardedSlab {
public:
  SharedPoolCached(Allocator alloc, std::size_t block_size,
                   std::uint32_t block_count,
                   std::size_t alignment = alignof(std::max_align_t))
      : detail::ShardedSlab(alloc, block_size, block_count, alignment) {}

#if defined(HYLE_TESTING)
  void testing_force_tag_saturation() noexcept {
    detail::ShardedSlab::testing_force_tag_saturation();
  }
#endif

  ~SharedPoolCached() {
    // Contract: callers have externally quiesced alloc/free and established the
    // usual happens-before edge, though worker threads may remain alive. Flush
    // their free indices, then detach so thread_local dtors see pool==null and
    // only free the node (no UAF into a dead pool). Runs before ~ShardedSlab
    // frees the slab.
    if (valid_)
      detach_all_caches();
  }

  // Pop a free block. nullptr if local TLS empty AND every shard empty. Free
  // blocks still sitting only in OTHER live threads' TLS are invisible (soft
  // capacity; call their publish_thread_cache or join them to restore hard
  // capacity).
  void *allocate() noexcept {
    if (!valid_)
      return nullptr;
    const std::uint32_t home = home_shard();
    if (ThreadCache *tc = tls_cache(/*create=*/true)) {
      if (tc->count >
          0) { // TLS hit: owner-only pop, zero atomics (steady state)
        const std::uint32_t idx = tc->idxs[--tc->count];
        return checkout(idx);
      }
      // TLS miss: refill a BATCH from the shards in ONE CAS, keep the surplus
      // warm, hand back one. tc->count is 0 here, so the batch lands at
      // tc->idxs[0..got).
      const std::uint32_t got =
          pop_any_shard_batch(home, kRefillBatch, tc->idxs);
      if (got == 0)
        return nullptr; // stable all-shard empty snapshot
      tc->count = got;
      return checkout(tc->idxs[--tc->count]);
    }
    // No TLS cache (OOM'd node): direct stable-snapshot shard path.
    const std::uint32_t idx = pop_any_shard(home);
    return checkout(idx);
  }

  // Return a block. Same-thread steady state lands in the TLS freelist
  // (owner-only, no atomics). Overflow publishes a batch to this thread's
  // shared home. null no-op.
  void deallocate(void *p) noexcept {
    if (p == nullptr || !valid_)
      return;
    std::uint32_t idx = return_index(p);
    if (ThreadCache *tc = tls_cache(/*create=*/true)) {
      if (tc->count < kTlsCacheCap) {
        tc->idxs[tc->count++] = idx;
        return;
      }
      flush_cache_half(tc); // keep half warm, flush half to shards
      if (tc->count < kTlsCacheCap) {
        tc->idxs[tc->count++] = idx;
        return;
      }
    }
    push_shard(home_shard(), idx); // no TLS (OOM): direct shared-home push
  }

  // Flush THIS thread's TLS freelist onto its shared home shard -- the explicit
  // barrier that makes this thread's freed blocks visible to others without
  // waiting for it to exit (soft-capacity handoff). Owner-only; no cross-thread
  // use.
  void publish_thread_cache() noexcept {
    if (!valid_)
      return;
    if (ThreadCache *tc = tls_cache(/*create=*/false))
      flush_cache_to_shards(tc);
  }

  // Return EVERY block to the pool in one shot. All prior pointers are bulk-
  // invalidated and must never be used again; quiesce concurrent alloc/free.
  // Also drains every registered TLS cache.
  void reset() noexcept {
    if (!valid_)
      return;
#if defined(HYLE_TESTING)
    testing_discard_all();
#endif
    flush_all_caches_keep_registered();
    reset_shards();
  }

#if defined(HYLE_TESTING)
  // TEST-ONLY: see SharedPool::testing_assert_checked_out().
  void testing_assert_checked_out(void *p) const noexcept {
    assert_checked_out(p);
  }
#endif

  Allocator as_allocator() noexcept {
    return Allocator(&pool_alloc, &pool_free, &pool_realloc, this);
  }

private:
  static void *pool_alloc(void *state, std::size_t size,
                          std::size_t alignment) noexcept {
    auto *self = static_cast<SharedPoolCached *>(state);
    if (size == 0)
      return nullptr;
    HYLE_FATAL_ASSERT(is_pow2(alignment) && size <= self->block_size_ &&
                      alignment <= self->alignment_);
    return self->allocate();
  }
  static void pool_free(void *state, void *ptr, std::size_t,
                        std::size_t) noexcept {
    if (ptr != nullptr)
      static_cast<SharedPoolCached *>(state)->deallocate(ptr);
  }
  static void *pool_realloc(void *state, void *ptr, std::size_t old_size,
                            std::size_t new_size,
                            std::size_t alignment) noexcept {
    auto *self = static_cast<SharedPoolCached *>(state);
    if (ptr == nullptr)
      return pool_alloc(state, new_size, alignment);
    HYLE_FATAL_ASSERT(is_pow2(alignment) && old_size > 0 &&
                      old_size <= self->block_size_ &&
                      alignment <= self->alignment_);
    self->assert_checked_out(ptr);
    if (new_size == 0) {
      self->deallocate(ptr);
      return nullptr;
    }
    if (new_size > self->block_size_)
      return nullptr;
    return ptr;
  }

  // TLS freelist depth: bounds soft-capacity (blocks parked off-shard) per live
  // thread. Folly's mature fixed-pool shape uses local lists up to 200 entries;
  // 128 covers common request/context batches while bounding this soft pool's
  // per-thread window.
  static constexpr std::uint32_t kTlsCacheCap = 128;
  // One CAS transfers 32 blocks between a TLS cache and its shared home list.
  static constexpr std::uint32_t kRefillBatch = 32;

  // One node per (thread, pool). Thread owns the node lifetime (thread_local
  // list + dtor); the pool owns a registry (cache_list_) so free indices never
  // die with a destroyed pool and never UAF after it. `pool` is ATOMIC: the hot
  // path loads it lockless to match a node to its pool; cold teardown
  // (thread-exit detach, dtor detach_all) writes it under the process-global
  // registry lock -- closing the dtor/thread-exit race. count/idxs are
  // owner-only on the hot path (no atomics).
  struct ThreadCache {
    std::atomic<SharedPoolCached *> pool{nullptr};
    std::uint32_t count = 0;
    std::uint32_t idxs[kTlsCacheCap]{};
    ThreadCache *next_in_thread = nullptr; // owner-thread-only
    ThreadCache *next_in_pool = nullptr;   // registry-lock-only
  };

  struct TlsCacheList {
    ThreadCache *head = nullptr;
    ~TlsCacheList() {
      while (head != nullptr) {
        ThreadCache *c = head;
        head = c->next_in_thread;
        c->next_in_thread = nullptr;
        // Decide + act under the global lock so the pool can't be torn down
        // between reading c->pool and using it (detach_all_caches nulls it
        // under the same lock). If pool is already null, it detached us first.
        {
          detail::TlsRegistryGuard g;
          SharedPoolCached *a = c->pool.load(std::memory_order_relaxed);
          if (a != nullptr)
            a->detach_cache_locked(c);
        }
        free_thread_cache_node(c);
      }
    }
  };

  static TlsCacheList &tls_list() noexcept {
    thread_local TlsCacheList list;
    return list;
  }

  static void free_thread_cache_node(ThreadCache *c) noexcept {
    c->~ThreadCache();
    system_allocator().deallocate(c, sizeof(ThreadCache), alignof(ThreadCache));
  }

  ThreadCache *tls_cache(bool create) noexcept {
    TlsCacheList &list = tls_list();
    // Walk + reclaim zombies (pool destroyed, node left with pool==nullptr) so
    // a long-lived thread churning many pools doesn't leak a node per destroyed
    // pool. Reads `pool` acquire: observing null means detach finished flushing
    // -> safe to unlink (owner-only) and free.
    ThreadCache *prev = nullptr;
    ThreadCache *c = list.head;
    while (c != nullptr) {
      ThreadCache *next = c->next_in_thread;
      SharedPoolCached *a = c->pool.load(std::memory_order_acquire);
      if (a == this)
        return c;
      if (a == nullptr) {
        if (prev != nullptr)
          prev->next_in_thread = next;
        else
          list.head = next;
        free_thread_cache_node(c);
        c = next;
        continue;
      }
      prev = c;
      c = next;
    }
    if (!create || !valid_)
      return nullptr;
    void *mem =
        system_allocator().allocate(sizeof(ThreadCache), alignof(ThreadCache));
    if (mem == nullptr)
      return nullptr; // fall back to pure shard CAS path
    auto *fresh = new (mem) ThreadCache();
    fresh->pool.store(this, std::memory_order_relaxed);
    fresh->next_in_thread = list.head; // owner-only
    list.head = fresh;
    {
      detail::TlsRegistryGuard g;
      fresh->next_in_pool = cache_list_;
      cache_list_ = fresh;
    }
    return fresh;
  }

  // Publish tc->idxs[base .. base+n) to the freeing thread's shared home shard
  // in ONE CAS. This is the mature local-list/global-transfer shape: free
  // blocks migrate toward the thread using them instead of being scattered by
  // their slab address. The active cache range is dense, so no bucketing or
  // O(n^2) search is needed.
  void flush_range_batched(ThreadCache *tc, std::uint32_t base,
                           std::uint32_t n) noexcept {
    push_shard_batch(home_shard(), tc->idxs + base, n);
    tc->count = base;
  }

  void flush_cache_to_shards(ThreadCache *tc) noexcept {
    flush_range_batched(tc, 0, tc->count);
  }

  void flush_cache_half(ThreadCache *tc) noexcept {
    const std::uint32_t keep =
        tc->count - tc->count / 2; // keep the warm lower half
    flush_range_batched(tc, keep, tc->count - keep);
  }

  // Called from the thread_local dtor with the global registry lock ALREADY
  // held: flush this node's blocks to shards, unlink from the registry, null
  // its pool (release) so a concurrent tls_cache() walk that sees null knows
  // the flush is done. Caller frees.
  void detach_cache_locked(ThreadCache *tc) noexcept {
    flush_cache_to_shards(tc);
    ThreadCache **pp = &cache_list_;
    while (*pp != nullptr) {
      if (*pp == tc) {
        *pp = tc->next_in_pool;
        break;
      }
      pp = &(*pp)->next_in_pool;
    }
    tc->next_in_pool = nullptr;
    tc->pool.store(nullptr, std::memory_order_release);
  }

  void detach_all_caches() noexcept {
    detail::TlsRegistryGuard g;
    ThreadCache *c = cache_list_;
    cache_list_ = nullptr;
    while (c != nullptr) {
      ThreadCache *next = c->next_in_pool;
      flush_cache_to_shards(c);
      c->next_in_pool = nullptr;
      // Release pairs with the acquire in a concurrent tls_cache() zombie-walk:
      // the flush above happens-before that thread frees the node.
      c->pool.store(nullptr, std::memory_order_release);
      c = next;
    }
  }

  // reset(): drop every TLS freelist (reset_shards below reclaims the full
  // range; TLS-held free indices are not live). Keeps nodes registered. Callers
  // quiesce first.
  void flush_all_caches_keep_registered() noexcept {
    detail::TlsRegistryGuard g;
    for (ThreadCache *c = cache_list_; c != nullptr; c = c->next_in_pool)
      c->count = 0;
  }

  // Registry of this pool's live TLS caches; mutated only under the
  // process-global detail::tls_registry_lock. NOT read on the hot path
  // (alloc/free reach the node via the per-thread list and validate ownership
  // with the node's atomic `pool`).
  ThreadCache *cache_list_ = nullptr;
};

// Bump (region) allocator over ONE borrowed pool block. Borrows a block from a
// SharedPool, hands out aligned bump slices, returns the whole block on
// destruction. There is NO per-allocation free: release everything at once by
// destroying the region (or clear()). Ideal for a transient phase (one
// taskgraph pass, one request) backed by a reusable pool block. Borrows from
// the HARD SharedPool: a single-block borrow gains nothing from a per-thread
// cache, and the hard pool's immediate visibility keeps block lifetime simple.
//
// SINGLE-THREADED bump (no atomics -- a bump cursor is inherently one serial
// pointer). Concurrency = one PooledRegion PER THREAD/fiber (each over its own
// pooled block), the per-core scratch model, not a contended shared cursor.
// Explicitly named PooledRegion because it is neither owned nor growable: it
// borrows exactly one block from a hard SharedPool.
class PooledRegion {
public:
  explicit PooledRegion(SharedPool *pool) : pool_(pool) {
    HYLE_FATAL_ASSERT(pool != nullptr);
    block_ = static_cast<std::byte *>(pool_->allocate());
    block_size_ = pool_->block_size();
    // Pool exhausted -> invalid (block_ == nullptr); caller must check valid().
    // A resource failure returns gracefully (not a contract violation), and the
    // dtor's deallocate(nullptr) is a documented no-op, so invalid is safe.
    cur_ = block_;
    end_ = block_ != nullptr ? block_ + block_size_ : nullptr;
  }

  bool valid() const noexcept { return block_ != nullptr; }

  ~PooledRegion() { pool_->deallocate(block_); }

  HYLE_DISABLE_COPY_AND_MOVE(PooledRegion);

  // Bump `bytes` aligned to `alignment`. nullptr if the block is full.
  void *allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(alignment)); // 0/non-pow2 -> wild ptr
    if (bytes == 0)
      return nullptr;
    const auto cur = reinterpret_cast<std::uintptr_t>(cur_);
    std::uintptr_t aligned = 0;
    if (!detail::try_align_up_address(cur, alignment, &aligned))
      return nullptr;
    const auto e = reinterpret_cast<std::uintptr_t>(end_);
    if (aligned > e || bytes > e - aligned)
      return nullptr;
    cur_ = reinterpret_cast<std::byte *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
  }

  // Release everything (reuse the block from the start); only safe when no
  // allocations are live. clear() is the common Region-family spelling.
  void clear() noexcept { cur_ = block_; }

  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t used() const noexcept {
    return valid() ? static_cast<std::size_t>(cur_ - block_) : 0;
  }
  std::size_t remaining() const noexcept {
    return valid() ? static_cast<std::size_t>(end_ - cur_) : 0;
  }

  // Adapt to the Allocator vtable: allocate -> bump, deallocate -> no-op (bulk
  // release on destruction), realloc -> bump a fresh slice + copy (regions can't
  // grow in place).
  Allocator as_allocator() noexcept {
    return Allocator(&region_alloc, &region_free, &region_realloc, this);
  }

private:
  static void *region_alloc(void *state, std::size_t size,
                            std::size_t alignment) noexcept {
    return static_cast<PooledRegion *>(state)->allocate(size, alignment);
  }
  static void region_free(void *, void *, std::size_t, std::size_t) noexcept {
  } // bulk release
  static void *region_realloc(void *state, void *ptr, std::size_t old_size,
                              std::size_t new_size,
                              std::size_t alignment) noexcept {
    auto *self = static_cast<PooledRegion *>(state);
    if (new_size == 0)
      return nullptr;
    void *p = self->allocate(new_size, alignment);
    if (ptr != nullptr && p != nullptr && old_size != 0)
      std::memcpy(p, ptr, old_size < new_size ? old_size : new_size);
    return p;
  }

  SharedPool *pool_;
  std::byte *block_;
  std::size_t block_size_;
  std::byte *cur_ = nullptr;
  std::byte *end_ = nullptr;
};

} // namespace hyle::mem
