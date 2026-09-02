#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/cpu.h" // process-available parallelism -> shard count

// Shared fixed-size block pool plus an explicit owner-local cache layer.
//
//   ConcurrentBlockPool        HARD capacity. In normal tagged mode, the successful
//   home-shard
//                     path and every deallocate use one CAS; allocate scans if
//                     needed. A freed block is never hidden in TLS, so
//                     completed frees are immediately visible. This is the safe
//                     default for a bounded quota that is drawn/returned across
//                     threads (work-stealing handoff, N fixed slots). Pick this
//                     unless a measured same-thread hot spot justifies an
//                     explicit BlockPoolCache owned by that worker.
//
// Both borrow slab + link array from the injected AllocatorRef. Each shard uses a
// 32-bit tagged 64-bit-CAS fast path; before the finite tag could wrap, that
// shard permanently enters a serialized 64-bit-change-counter fallback, so
// stale CAS can never become valid again. The free lists are sharded per core:
// clay's single-head toy ran 278x slower than per-thread heaps under 8-way
// churn. The `next` links live in a PARALLEL index array, never intrusively in
// user bytes.
//
// PooledArena (bottom of file) is the single-threaded bump allocator over ONE
// borrowed pool block -- a different tool (bulk release, no per-object free),
// kept here because it borrows from a pool.

namespace hyle::mem {

class BlockPoolCache;

namespace detail {
// ---- Sharded-slab core -----------------------------------------------------
// Owns the slab + parallel next_[] link array + per-shard
// cache-line-isolated Treiber heads and their tag-saturation fallback. NOT a
// standalone type: ConcurrentBlockPool derives from it and exposes hard operations plus
// narrow batch hooks to BlockPoolCache. No virtuals or hot-path indirection.
class ShardedSlab {
public:
  std::size_t block_size() const noexcept { return block_size_; }
  std::uint32_t capacity() const noexcept { return count_; }
  // False iff a construction-time allocation (slab / link array / heads) hit
  // OOM. An invalid pool hands out nothing instead of exposing a dead handle.
  bool valid() const noexcept { return valid_; }

protected:
  static constexpr std::uint32_t kNull = 0xFFFFFFFFu;
  static constexpr std::uint32_t kSaturatedVersion = 0xFFFFFFFFu;

  struct EmptyObservation {
    std::uint64_t head;
    std::uint64_t saturated_changes;
  };

#if defined(HYLE_TESTING)
  // TEST-ONLY: deterministic interleaving hook used by concurrency tests.
  using EmptyScanHook = void (*)(void *, std::uint32_t) noexcept;
#endif

  // `block_size` bytes per block, `block_count` blocks, each aligned to
  // `alignment` (power of two). Slab + link array + heads come from `alloc`. On
  // OOM: valid()==false (graceful, not abort; mem-audit-followup #2 arch B).
  // The dtor is null-safe.
  ShardedSlab(AllocatorRef alloc, std::size_t block_size,
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
  // TEST-ONLY: require all diagnostic per-slot liveness bits to be clear.
  void testing_assert_no_outstanding() const noexcept {
    for (std::uint32_t i = 0; i < count_; ++i)
      HYLE_FATAL_ASSERT(testing_live_[i].load(std::memory_order_acquire) == 0);
  }

  // TEST-ONLY: clear diagnostic liveness before a quiescent bulk reset.
  void testing_discard_all() noexcept {
    for (std::uint32_t i = 0; i < count_; ++i)
      testing_live_[i].store(0, std::memory_order_release);
  }
#endif

  ShardedSlab(const ShardedSlab&) = delete;
  ShardedSlab& operator=(const ShardedSlab&) = delete;
  ShardedSlab(ShardedSlab&&) = delete;
  ShardedSlab& operator=(ShardedSlab&&) = delete;

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
      // Acquire consumes the releasing push that published `next_[idx]`.
      // A pop publishes no preceding writes, so acq_rel is unnecessarily
      // stronger on the successful RMW.
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_acquire,
                                            std::memory_order_acquire))
        return idx;
    }
  }

  // Push one index back onto shard s.
  void push_shard(std::uint32_t s, std::uint32_t idx) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
    for (;;) {
      if (ver_of(old) == kSaturatedVersion) {
        push_shard_saturated(s, idx);
        return;
      }
      next_[idx].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(idx, ver_of(old) + 1);
      // Release publishes the relaxed link store. A failed CAS only supplies
      // the next head value used to retry, so neither load needs acquire.
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_release,
                                            std::memory_order_relaxed))
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
      // As in pop_shard(), acquire consumes every link in the detached chain;
      // the successful pop itself has nothing to publish.
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_acquire,
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

  // The overwhelmingly common successful home-shard path stays in the caller
  // and has no O(shards) stack frame. Only a miss enters the stable-empty proof
  // below. A failed first collect is not enough: independent shard snapshots
  // can otherwise miss a free that moves behind the scan. A second identical
  // tagged-head collect proves every shard remained empty over one common
  // interval (barring the documented finite tag wrap).
  HYLE_FORCE_INLINE std::uint32_t
  pop_any_shard(std::uint32_t home) noexcept {
    const std::uint32_t idx = pop_shard(home);
    return idx != kNull ? idx : pop_any_shard_slow(home);
  }

  HYLE_NOINLINE std::uint32_t
  pop_any_shard_slow(std::uint32_t home) noexcept {
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

  HYLE_FORCE_INLINE std::uint32_t
  pop_any_shard_batch(std::uint32_t home, std::uint32_t want,
                      std::uint32_t *out) noexcept {
    const std::uint32_t n = pop_shard_batch(home, want, out);
    return n != 0 ? n : pop_any_shard_batch_slow(home, want, out);
  }

  HYLE_NOINLINE std::uint32_t
  pop_any_shard_batch_slow(std::uint32_t home, std::uint32_t want,
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
    std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
    for (;;) {
      if (ver_of(old) == kSaturatedVersion) {
        push_shard_batch_saturated(s, chain_head, chain_tail);
        return;
      }
      next_[chain_tail].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(chain_head, ver_of(old) + 1);
      // Release publishes the entire privately assembled chain. Retry loads
      // need only the latest packed head value.
      if (heads_[s].h.compare_exchange_weak(old, desired,
                                            std::memory_order_release,
                                            std::memory_order_relaxed))
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
  // TEST-ONLY: install a deterministic empty-shard observation hook.
  void testing_set_empty_scan_hook(EmptyScanHook hook, void *state) noexcept {
    testing_empty_scan_hook_ = hook;
    testing_empty_scan_state_ = state;
  }

  // TEST-ONLY: pop directly from one shard for structural concurrency tests.
  void *testing_pop_from_shard(std::uint32_t s) noexcept {
    const std::uint32_t idx = pop_shard(s);
    return checkout(idx);
  }

  // TEST-ONLY: push directly to one shard for structural concurrency tests.
  void testing_push_to_shard(std::uint32_t s, void *p) noexcept {
    push_shard(s, return_index(p));
  }
  // TEST-ONLY: expose the constructed shard count to structural tests.
  std::uint32_t testing_num_shards() const noexcept { return num_shards_; }

  // TEST-ONLY: exercise cross-shard fallback from a chosen home shard.
  void *testing_allocate_from_home(std::uint32_t home) noexcept {
    const std::uint32_t idx = pop_any_shard(home);
    return checkout(idx);
  }

  // TEST-ONLY: quiescent hook forcing the no-wrap path without billions of ops.
  void testing_force_tag_saturation() noexcept {
    for (std::uint32_t s = 0; s < num_shards_; ++s) {
      const std::uint64_t old = heads_[s].h.load(std::memory_order_relaxed);
      heads_[s].h.store(pack(idx_of(old), kSaturatedVersion),
                        std::memory_order_relaxed);
      heads_[s].saturated_changes.store(0, std::memory_order_relaxed);
    }
  }
#endif

  AllocatorRef alloc_;
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

  // Bounds heads_: target-family spacing keeps the maximum x86 array at 8 KiB
  // and the native Apple/generic conservative array at 16 KiB.
  static constexpr std::uint32_t kShardCap = 128;
  static std::uint32_t default_shards() noexcept {
    auto available = hyle::platform::available_cpu_count();
    std::size_t c = available && *available != 0 ? *available : 8;
    std::uint32_t p = 1;
    while (p < c && p < kShardCap)
      p <<= 1;
    return p;
  }

  // One Treiber-stack head per shard, each on its OWN cache line (kill false
  // sharing).
  struct alignas(core::kDestructiveInterferenceSize) ShardHead {
    std::atomic<std::uint64_t>
        h; // index|version (lock-free until tag saturation)
    std::atomic_flag saturated_lock = ATOMIC_FLAG_INIT;
    std::atomic<std::uint64_t> saturated_changes{0};
  };
  static_assert(alignof(ShardHead) == core::kDestructiveInterferenceSize);
  static_assert(sizeof(ShardHead) == core::kDestructiveInterferenceSize);

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

// ---- ConcurrentBlockPool: HARD capacity, no off-shard cache
// ------------------------------- The home-shard success path and deallocate
// each use one CAS; allocate scans the other shards when home is empty. A
// completed free is immediately visible because no block is parked in TLS (no
// cache-induced false exhaustion). Choose this for a bounded quota
// drawn/returned across threads (fixed slot count, work-stealing handoff) or
// whenever deterministic capacity matters more than squeezing out the
// same-thread fast path. Same-thread churn still pays one CAS/op; if that is a
// measured hot spot and the pool is over-provisioned, attach BlockPoolCache.
//
// Lifetime contract: every live block is returned exactly once to the SAME pool
// before destruction. reset() is a distinct quiescent bulk-invalidation
// boundary: no concurrent operation may run, and no old pointer may be touched
// or returned afterward. Every attached BlockPoolCache must first publish its
// locally free indices; otherwise reset would make the same slots visible twice.
// Foreign/misaligned pointers fail-fast in every build. HYLE_TESTING adds
// per-slot liveness, double-return and outstanding-allocation diagnostics;
// shipping builds carry no per-slot diagnostic state or instructions.
class ConcurrentBlockPool : public detail::ShardedSlab {
public:
  ConcurrentBlockPool(AllocatorRef alloc, std::size_t block_size, std::uint32_t block_count,
             std::size_t alignment = alignof(std::max_align_t))
      : detail::ShardedSlab(alloc, block_size, block_count, alignment) {}

#if defined(HYLE_TESTING)
  // TEST-ONLY: public test spelling for the internal empty-scan callback.
  using TestingEmptyScanHook = void (*)(void *, std::uint32_t) noexcept;

  // TEST-ONLY: force every shard tag to its saturation boundary.
  void testing_force_tag_saturation() noexcept {
    detail::ShardedSlab::testing_force_tag_saturation();
  }
  // TEST-ONLY: install a deterministic empty-shard observation hook.
  void testing_set_empty_scan_hook(TestingEmptyScanHook hook,
                                   void *state) noexcept {
    detail::ShardedSlab::testing_set_empty_scan_hook(hook, state);
  }
  // TEST-ONLY: pop directly from a selected shard.
  void *testing_pop_from_shard(std::uint32_t shard) noexcept {
    return detail::ShardedSlab::testing_pop_from_shard(shard);
  }
  // TEST-ONLY: push directly to a selected shard.
  void testing_push_to_shard(std::uint32_t shard, void *p) noexcept {
    detail::ShardedSlab::testing_push_to_shard(shard, p);
  }
  // TEST-ONLY: expose the constructed shard count.
  std::uint32_t testing_num_shards() const noexcept {
    return detail::ShardedSlab::testing_num_shards();
  }
  // TEST-ONLY: exercise fallback allocation from a chosen home shard.
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

  // Adapt to the complete AllocatorRef contract. A live fixed block resizes in
  // place while the request still fits; an over-capacity grow fails without
  // releasing it. null/zero follow the universal allocate/free rules.
  AllocatorRef as_allocator() noexcept {
    return AllocatorRef(&pool_alloc, &pool_free, &pool_realloc, this);
  }

private:
  friend class BlockPoolCache;

  // Explicit owner-local caches use these narrow batch hooks. Cached indices
  // are free (not checked out) while they live outside the shared shards; the
  // checkout/return hooks preserve the same diagnostics as allocate/free.
  std::uint32_t cache_refill(std::uint32_t want,
                             std::uint32_t *out) noexcept {
    return pop_any_shard_batch(home_shard(), want, out);
  }
  void cache_publish(const std::uint32_t *indices,
                     std::uint32_t count) noexcept {
    push_shard_batch(home_shard(), indices, count);
  }
  void *cache_checkout(std::uint32_t index) noexcept { return checkout(index); }
  std::uint32_t cache_return(void *p) noexcept { return return_index(p); }
  void cache_assert_checked_out(void *p) const noexcept {
    assert_checked_out(p);
  }
  bool cache_request_fits(std::size_t size,
                          std::size_t alignment) const noexcept {
    return size > 0 && size <= block_size_ && is_pow2(alignment) &&
           alignment <= alignment_;
  }

  static void *pool_alloc(void *state, std::size_t size,
                          std::size_t alignment) noexcept {
    auto *self = static_cast<ConcurrentBlockPool *>(state);
    if (size == 0)
      return nullptr;
    if (!is_pow2(alignment) || size > self->block_size_ ||
        alignment > self->alignment_)
      return nullptr;
    return self->allocate();
  }
  static void pool_free(void *state, void *ptr, std::size_t,
                        std::size_t) noexcept {
    if (ptr != nullptr)
      static_cast<ConcurrentBlockPool *>(state)->deallocate(ptr);
  }
  static void *pool_realloc(void *state, void *ptr, std::size_t old_size,
                            std::size_t new_size,
                            std::size_t alignment) noexcept {
    auto *self = static_cast<ConcurrentBlockPool *>(state);
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

// Bump (region) allocator over ONE borrowed pool block. Borrows a block from a
// ConcurrentBlockPool, hands out aligned bump slices, returns the whole block on
// destruction. There is NO per-allocation free: release everything at once by
// destroying the region (or clear()). Ideal for a transient phase (one
// taskgraph pass, one request) backed by a reusable pool block. Borrows from
// the HARD ConcurrentBlockPool: a single-block borrow gains nothing from a per-thread
// cache, and the hard pool's immediate visibility keeps block lifetime simple.
//
// SINGLE-THREADED bump (no atomics -- a bump cursor is inherently one serial
// pointer). Concurrency = one PooledArena PER THREAD/fiber (each over its own
// pooled block), the per-core scratch model, not a contended shared cursor.
// Explicitly named PooledArena because it is neither owned nor growable: it
// borrows exactly one block from a hard ConcurrentBlockPool.
class PooledArena {
public:
  explicit PooledArena(ConcurrentBlockPool *pool) : pool_(pool) {
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

  ~PooledArena() { pool_->deallocate(block_); }

  PooledArena(const PooledArena&) = delete;
  PooledArena& operator=(const PooledArena&) = delete;
  PooledArena(PooledArena&&) = delete;
  PooledArena& operator=(PooledArena&&) = delete;

  // Bump `bytes` aligned to `alignment`. nullptr if the block is full.
  void *allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(alignment)); // 0/non-pow2 -> wild ptr
    if (bytes == 0 || cur_ == nullptr)
      return nullptr;
    const auto cur = reinterpret_cast<std::uintptr_t>(cur_);
    const auto e = reinterpret_cast<std::uintptr_t>(end_);
    const auto mask = alignment - 1;
    if (HYLE_LIKELY((cur & mask) == 0)) {
      if (bytes > e - cur)
        return nullptr;
      cur_ = reinterpret_cast<std::byte *>(cur + bytes);
      return reinterpret_cast<void *>(cur);
    }
    const auto padding = (std::uintptr_t{0} - cur) & mask;
    const auto available = e - cur;
    const auto consumed = bytes + padding;
    if (consumed < bytes || consumed > available)
      return nullptr;
    const auto aligned = cur + padding;
    cur_ = reinterpret_cast<std::byte *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
  }

  // Release everything (reuse the block from the start); only safe when no
  // allocations are live. clear() is the common Arena-family spelling.
  void clear() noexcept { cur_ = block_; }

  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t used() const noexcept {
    return valid() ? static_cast<std::size_t>(cur_ - block_) : 0;
  }
  std::size_t remaining() const noexcept {
    return valid() ? static_cast<std::size_t>(end_ - cur_) : 0;
  }

  // Adapt to the AllocatorRef vtable: allocate -> bump, deallocate -> no-op (bulk
  // release on destruction), realloc -> bump a fresh slice + copy (regions can't
  // grow in place).
  AllocatorRef as_allocator() noexcept {
    return AllocatorRef(&region_alloc, &region_free, &region_realloc, this);
  }

private:
  static void *region_alloc(void *state, std::size_t size,
                            std::size_t alignment) noexcept {
    return static_cast<PooledArena *>(state)->allocate(size, alignment);
  }
  static void region_free(void *, void *, std::size_t, std::size_t) noexcept {
  } // bulk release
  static void *region_realloc(void *state, void *ptr, std::size_t old_size,
                              std::size_t new_size,
                              std::size_t alignment) noexcept {
    auto *self = static_cast<PooledArena *>(state);
    if (new_size == 0)
      return nullptr;
    void *p = self->allocate(new_size, alignment);
    if (ptr != nullptr && p != nullptr && old_size != 0)
      std::memcpy(p, ptr, old_size < new_size ? old_size : new_size);
    return p;
  }

  ConcurrentBlockPool *pool_;
  std::byte *block_;
  std::size_t block_size_;
  std::byte *cur_ = nullptr;
  std::byte *end_ = nullptr;
};

} // namespace hyle::mem
