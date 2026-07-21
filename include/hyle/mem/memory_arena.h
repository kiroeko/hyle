#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "hyle/macros.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/hugepage.h"
#include "hyle/platform/thread.h"  // hardware_concurrency() -> shard count

namespace hyle::mem {

namespace detail {
// Process-global (cross-DLL) lock guarding the per-arena TLS-cache registry. One
// definition lives in hyle_mem (memory_arena.cc) so every header-inlined consumer
// serializes on the SAME mutex -- a per-DLL header static would let two parties
// race on one node through different locks. COLD path only (attach / detach /
// reset / dtor / thread-exit); the alloc/free hot path never touches it.
HYLE_MEM_API void tls_registry_lock() noexcept;
HYLE_MEM_API void tls_registry_unlock() noexcept;

struct TlsRegistryGuard {
  TlsRegistryGuard() noexcept { tls_registry_lock(); }
  ~TlsRegistryGuard() noexcept { tls_registry_unlock(); }
  TlsRegistryGuard(const TlsRegistryGuard&) = delete;
  TlsRegistryGuard& operator=(const TlsRegistryGuard&) = delete;
};
}  // namespace detail

// Lock-free fixed-size block pool (clay's MemoryArena, modernized).
//
// A slab of `block_count` equal-size blocks; allocate()/deallocate() pop/push a
// Treiber stack of free blocks. Constant-time, lock-free, allocator-injected.
// Ideal for same-size churn (fiber stacks, pooled nodes) where a general malloc
// is overkill.
//
// Architecture (composition of two paths — different scenarios, one type):
//  1. Per-thread TLS freelist cache (tbb/mimalloc tcache shape): same-thread
//     free→alloc is a plain index stack with NO atomics (owner-only). This is
//     the steady-state hot path that closes the always-CAS tax vs oneTBB.
//  2. Sharded Treiber stacks (shared backend): TLS miss / overflow /
//     publish_thread_cache() / thread-exit / dtor flush land here. Free is
//     address-routed to the owning shard (ABA-sound). Transfers between TLS and
//     shards are BATCHED (magazine model, kRefillBatch): a miss refills up to K
//     indices in one CAS (bounded-walk pop), an overflow flushes a per-shard run
//     in one CAS (chain-prepend push) -- so a working set larger than the TLS cap
//     pays ~1/K CAS per block instead of one each (the tbb/mimalloc amortization).
//
// Soft capacity (red-team R1, intentional — same class as tbb/jemalloc tcache):
// free blocks parked in another live thread's TLS are not visible to allocate()
// until that thread flushes (publish_thread_cache / overflow / exit / dtor).
// Window ≤ kTlsCacheCap per live thread. After join/publish, hard capacity is
// restored. Making remote TLS steal-safe requires a lock on the hot path and
// measured ~2× regression (parity tbb instead of beating it) — rejected.
//
// Two deliberate changes from clay (see mem-design.md):
//  1. Free-list head is ONE 64-bit atomic packing (index<<32 | version) -- pure
//     64-bit CAS, lock-free everywhere. clay used a 16-byte atomic<{ptr,tag}>
//     needing cmpxchg16b/-mcx16 (else a silent lock fallback). The version tag
//     defeats ABA.
//  2. Block count is a ctor parameter (clay capped at 64 static slots -- bug #5).
//
// The free-list `next` links live in a PARALLEL index array, not intrusively in
// the block bytes: no aliasing of user storage, clean lifetimes, TSan-clean.
// TLS caches store block indices in a small per-thread array (do not borrow
// next_[] while a block sits in TLS — next_[] is only for shard stacks).
class MemoryArena {
 public:
  // `block_size` bytes per block, `block_count` blocks, each aligned to
  // `alignment` (power of two). Slab + link array come from `alloc`.
  MemoryArena(Allocator alloc, std::size_t block_size, std::uint32_t block_count,
              std::size_t alignment = alignof(std::max_align_t))
      : alloc_(alloc),
        block_size_(block_size),
        count_(block_count),
        alignment_(alignment < alignof(std::max_align_t) ? alignof(std::max_align_t)
                                                         : alignment) {
    HYLE_FATAL_ASSERT(alloc_.valid());
    HYLE_FATAL_ASSERT(block_count >= 1 && block_count < kNull);
    HYLE_FATAL_ASSERT(block_size_ >= 1);
    HYLE_FATAL_ASSERT((alignment_ & (alignment_ - 1)) == 0);
    // Overflow guards (a contract violation -> fail-fast, like block_size==0):
    // the round-up `block_size_ + (alignment_-1)` must not wrap, and the slab
    // size `stride_ * count_` must fit in size_t. Real only on 32-bit size_t;
    // unreachable on 64-bit, but cheap insurance against a silent under-alloc.
    HYLE_FATAL_ASSERT(block_size_ <= SIZE_MAX - (alignment_ - 1));
    stride_ = (block_size_ + alignment_ - 1) & ~(alignment_ - 1);  // round up
    HYLE_FATAL_ASSERT(static_cast<std::size_t>(count_) <= SIZE_MAX / stride_);
    // Large slabs: HugeRegion (2MiB align + THP + local NUMA). Completeness of
    // the mem standard surface — not gated on a particular consumer.
    const std::size_t slab_bytes = stride_ * static_cast<std::size_t>(count_);
    if (slab_bytes >= hugepage_region_threshold()) {
      huge_slab_ = HugeRegion::allocate_local(slab_bytes, HugeFlags::kThpHint);
      if (huge_slab_.valid()) {
        slab_ = static_cast<std::byte*>(huge_slab_.data());
        // Page/large-page base is aligned to >> typical object alignments.
        HYLE_FATAL_ASSERT((reinterpret_cast<std::uintptr_t>(slab_) % alignment_) ==
                          0);
      }
    }
    if (slab_ == nullptr) {
      slab_ = static_cast<std::byte*>(alloc_.allocate(slab_bytes, alignment_));
    }
    // slab-OOM is a resource failure -> invalid (valid()==false), not abort
    // (mem-audit-followup #2 arch B): the C ABI checks valid() and returns NULL +
    // last-error; a C++ caller can check valid() too. The dtor is null-safe.
    if (slab_ == nullptr) return;
    next_ = static_cast<std::atomic<std::uint32_t>*>(
        alloc_.allocate(sizeof(std::atomic<std::uint32_t>) * count_,
                        alignof(std::atomic<std::uint32_t>)));
    if (next_ == nullptr) return;  // link-array OOM -> invalid (dtor frees slab_)
    // Shard count = the host's core count rounded up to a power of two (capped), but never
    // more than blocks (each shard needs >=1). Scaling to cores keeps it O(1) up to the
    // real concurrency level: a FIXED count (e.g. 16) would re-introduce O(ceil(T/16))
    // contention once threads > shards (the 64-core box pays again). Pow2 makes the homing
    // modulo cheap and (with the rotating tls seq) collision-free while live threads <= cores.
    num_shards_ = default_shards();
    if (num_shards_ > count_) {
      // Keep num_shards_ a POWER OF TWO (default_shards() already is): round the
      // clamp DOWN to the largest pow2 <= count_, never up. This lets the home-shard
      // pick and the fallback scan both use `& shard_mask_` (exact, covers ALL shards)
      // instead of a per-alloc `% num_shards_` idiv. A non-pow2 count would break the
      // mask's full coverage, so we never let num_shards_ go non-pow2.
      std::uint32_t p = 1;
      while (p * 2u <= count_) p <<= 1;
      num_shards_ = p;
    }
    shard_mask_ = num_shards_ - 1;  // num_shards_ is pow2 -> exact modulo via AND
    per_ = count_ / num_shards_;  // >=1; the last shard absorbs the remainder
    heads_ = static_cast<ShardHead*>(
        alloc_.allocate(sizeof(ShardHead) * num_shards_, alignof(ShardHead)));
    if (heads_ == nullptr) return;  // heads OOM -> invalid (dtor frees slab_/next_)
    for (std::uint32_t s = 0; s < num_shards_; ++s) new (&heads_[s]) ShardHead();
    // Build each shard's free list over its own contiguous index range [lo,hi): link
    // lo -> lo+1 -> ... -> hi-1 -> null, head = lo. So a block's free links live only
    // in its owning shard, and free (address-routed) returns it to the same shard.
    for (std::uint32_t s = 0; s < num_shards_; ++s) {
      std::uint32_t lo = s * per_;
      std::uint32_t hi = (s + 1 == num_shards_) ? count_ : lo + per_;
      for (std::uint32_t i = lo; i < hi; ++i) {
        new (&next_[i]) std::atomic<std::uint32_t>(i + 1 < hi ? i + 1 : kNull);
      }
      heads_[s].h.store(pack(lo, 0), std::memory_order_relaxed);
    }
    valid_ = true;  // all three allocations succeeded + elements constructed
  }

  ~MemoryArena() {
    // Quiesce TLS caches first: flush free indices back onto shards (so we don't
    // leak blocks conceptually), then detach so thread_local dtors see arena==null
    // and only free the ThreadCache node (no UAF into a dead arena).
    if (valid_) detach_all_caches();
    // Elements are placement-new'd only after all three allocations succeed
    // (valid_); a partially-constructed (OOM'd) arena skips element teardown and
    // frees only what was allocated (null-safe). See mem-audit-followup #2 arch B.
    if (valid_) {
      for (std::uint32_t i = 0; i < count_; ++i) next_[i].~atomic();
      for (std::uint32_t s = 0; s < num_shards_; ++s) heads_[s].~ShardHead();
    }
    if (heads_ != nullptr)
      alloc_.deallocate(heads_, sizeof(ShardHead) * num_shards_, alignof(ShardHead));
    if (next_ != nullptr)
      alloc_.deallocate(next_, sizeof(std::atomic<std::uint32_t>) * count_,
                        alignof(std::atomic<std::uint32_t>));
    if (huge_slab_.valid()) {
      huge_slab_.reset();
      slab_ = nullptr;
    } else if (slab_ != nullptr) {
      alloc_.deallocate(slab_, stride_ * count_, alignment_);
    }
  }

  HYLE_DISABLE_COPY_AND_MOVE(MemoryArena);

  // Pop a free block. nullptr if local TLS empty AND every shard empty. Free blocks
  // still sitting only in OTHER live threads' TLS are invisible (soft capacity;
  // call their publish_thread_cache or join them to restore hard capacity).
  void* allocate() noexcept {
    if (!valid_) return nullptr;  // OOM'd arena has no slab/shards to hand out
    const std::uint32_t home = tls_thread_seq() & shard_mask_;  // num_shards_ is pow2
    if (ThreadCache* tc = tls_cache(/*create=*/true)) {
      if (tc->count > 0) {  // TLS hit: owner-only pop, zero atomics (steady state)
        const std::uint32_t idx = tc->idxs[--tc->count];
        return slab_ + static_cast<std::size_t>(idx) * stride_;
      }
      // TLS miss: refill a BATCH from the shared shards in ONE CAS (magazine), keep the
      // surplus warm in TLS, hand back one. This amortization closes the per-block-CAS
      // tax vs tbb once the working set exceeds kTlsCacheCap. tc->count is 0 here, so
      // the batch lands at tc->idxs[0..got).
      std::uint32_t got = pop_shard_batch(home, kRefillBatch, tc->idxs);
      if (got == 0) {
        for (std::uint32_t k = 1; k < num_shards_; ++k) {
          got = pop_shard_batch((home + k) & shard_mask_, kRefillBatch, tc->idxs);
          if (got != 0) break;
        }
        if (got == 0) return nullptr;  // every shard empty
      }
      tc->count = got;
      const std::uint32_t idx = tc->idxs[--tc->count];
      return slab_ + static_cast<std::size_t>(idx) * stride_;
    }
    // No TLS cache (OOM'd node): single-block pop, returned directly (no surplus to park).
    std::uint32_t idx = pop_shard(home);
    if (idx == kNull) {
      for (std::uint32_t k = 1; k < num_shards_; ++k) {
        idx = pop_shard((home + k) & shard_mask_);
        if (idx != kNull) break;
      }
      if (idx == kNull) return nullptr;
    }
    return slab_ + static_cast<std::size_t>(idx) * stride_;
  }

  // Return a block previously obtained from allocate(). Same-thread steady state
  // lands in the TLS freelist (owner-only, no atomics). Overflow flushes a batch
  // back to the address-routed owning shards. null is a no-op.
  void deallocate(void* p) noexcept {
    if (p == nullptr || !valid_) return;
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    const auto begin = reinterpret_cast<std::uintptr_t>(slab_);
    const std::size_t slab_bytes = stride_ * static_cast<std::size_t>(count_);
    HYLE_FATAL_ASSERT(addr >= begin && addr - begin < slab_bytes);
    const auto off = static_cast<std::size_t>(addr - begin);
    std::uint32_t idx = static_cast<std::uint32_t>(off / stride_);
    // Belongs to this pool: idx in range AND p is block-aligned. The alignment check
    // is `idx*stride_ == off` (one multiply) rather than a second `off % stride_`
    // division -- same guarantee, one idiv instead of two on the free hot path.
    HYLE_FATAL_ASSERT(idx < count_ &&
                      static_cast<std::size_t>(idx) * stride_ == off);
    if (ThreadCache* tc = tls_cache(/*create=*/true)) {
      if (tc->count < kTlsCacheCap) {
        tc->idxs[tc->count++] = idx;
        return;
      }
      // Cache full: flush half to shards, keep half warm, then push this idx.
      flush_cache_half(tc);
      if (tc->count < kTlsCacheCap) {
        tc->idxs[tc->count++] = idx;
        return;
      }
    }
    // No TLS (OOM creating cache) or still full: direct shared push.
    push_shard(shard_of(idx), idx);
  }

  // Flush THIS thread's TLS freelist onto address-routed shards. Explicit barrier
  // when another thread must observe the free blocks without waiting for this
  // thread to exit (soft-capacity handoff). Owner-only; no cross-thread use.
  void publish_thread_cache() noexcept {
    if (!valid_) return;
    if (ThreadCache* tc = tls_cache(/*create=*/false)) {
      flush_cache_to_shards(tc);
    }
  }

  // Return EVERY block to the free pool in one shot (rebuild each shard's free
  // list to its full contiguous range). O(count). Contract: NO block may be live
  // -- reset() while allocations are outstanding is a use-after-free waiting to
  // happen (same rule as Arena::reset). Single-threaded w.r.t. allocate/deallocate:
  // callers must quiesce concurrent access first. Also drains every registered TLS
  // cache (quiesced threads may still hold free indices in TLS).
  void reset() noexcept {
    if (!valid_) return;
    flush_all_caches_keep_registered();
    for (std::uint32_t s = 0; s < num_shards_; ++s) {
      std::uint32_t lo = s * per_;
      std::uint32_t hi = (s + 1 == num_shards_) ? count_ : lo + per_;
      for (std::uint32_t i = lo; i < hi; ++i) {
        next_[i].store(i + 1 < hi ? i + 1 : kNull, std::memory_order_relaxed);
      }
      // Bump the version so any stale packed head observed by a racing thread
      // (there must be none per the contract, but stay ABA-sound) can't succeed.
      std::uint32_t ver = ver_of(heads_[s].h.load(std::memory_order_relaxed)) + 1;
      heads_[s].h.store(pack(lo, ver), std::memory_order_release);
    }
  }

  std::size_t block_size() const noexcept { return block_size_; }
  std::uint32_t capacity() const noexcept { return count_; }
  // False iff a construction-time allocation (slab / link array / heads) hit OOM.
  // An invalid arena hands out nothing (allocate() -> nullptr); the C ABI checks
  // this and returns NULL + last-error instead of exposing a dead handle.
  bool valid() const noexcept { return valid_; }

  // Adapt this pool to the Allocator vtable (split alloc/free/realloc): allocate -> pop a
  // fixed block (the request must fit block_size_; the pool serves one size class),
  // deallocate -> push the block back. A pool block can't grow, so realloc is rejected
  // (nullptr). Ideal as a Fiber stack source: the 64 KB stack churns the lock-free free
  // list instead of malloc/mmap.
  Allocator as_allocator() noexcept {
    return Allocator(&pool_alloc, &pool_free, &pool_realloc, this);
  }

 private:
  static void* pool_alloc(void* state, std::size_t size, std::size_t alignment) noexcept {
    auto* self = static_cast<MemoryArena*>(state);
    if (size == 0) return nullptr;
    // One size class: the caller's request must fit a pool block and not need stronger
    // alignment than the slab was built with.
    HYLE_FATAL_ASSERT(size <= self->block_size_ && alignment <= self->alignment_);
    return self->allocate();
  }
  static void pool_free(void* state, void* ptr, std::size_t /*size*/,
                        std::size_t /*alignment*/) noexcept {
    if (ptr != nullptr) static_cast<MemoryArena*>(state)->deallocate(ptr);
  }
  static void* pool_realloc(void* /*state*/, void* /*ptr*/, std::size_t /*old_size*/,
                            std::size_t /*new_size*/, std::size_t /*alignment*/) noexcept {
    return nullptr;  // a fixed-size pool block cannot be reallocated in place
  }

  static constexpr std::uint32_t kNull = 0xFFFFFFFFu;
  // TLS freelist depth: bounds soft-capacity (free blocks parked off-shard) per
  // live thread. 32 is enough to absorb same-thread churn without thrashing the
  // shared CAS path; overflow flushes half so the cache stays warm.
  static constexpr std::uint32_t kTlsCacheCap = 32;
  // Batch size for the shard<->TLS transfer (magazine model). A TLS miss refills up
  // to this many indices in ONE CAS (bounded-walk pop), and an overflow flush returns
  // a run in ONE CAS per shard bucket (chain-prepend push) -- so a working set larger
  // than kTlsCacheCap pays ~1/kRefillBatch CAS per block instead of one CAS each. This
  // is the tbb/mimalloc magazine amortization; kept below kTlsCacheCap so a refill
  // still leaves headroom and a half-flush moves a meaningful run.
  static constexpr std::uint32_t kRefillBatch = 16;
  // Shard the free list into independent Treiber stacks, one per cache-line-isolated head.
  // clay inherited a SINGLE head -> every thread CAS'd the same cache line, serializing
  // (O(threads), measured 278x slower than mimalloc's per-thread heaps under 8-way churn).
  // Per-core sharding is the SOTA fix mem-design called a HARD constraint (the per-core
  // pool + xcpu_freelist note) but the toy single-stack never implemented. A thread homes
  // on one shard (tls round-robin) so same-thread alloc/free hits an UNCONTENDED head; free
  // routes by ADDRESS back to the block's owning shard (each shard's list holds only its own
  // index range -> per-shard ABA version stays valid, no cross-shard next_ aliasing).
  //
  // Count = core count rounded up to pow2, clamped to [1, kShardCap]. NOT a fixed 16: a
  // fixed count re-introduces O(ceil(T/16)) contention once live threads exceed it (the
  // many-core box would pay again). kShardCap bounds the heads_ array (128 -> 8 KB) for
  // absurd core counts; beyond it sharing resumes, but 128-way is far past current hosts.
  static constexpr std::uint32_t kShardCap = 128;
  static std::uint32_t default_shards() noexcept {
    unsigned c = hyle::platform::hardware_concurrency();
    if (c == 0) c = 8;  // unknown -> a sane default
    std::uint32_t p = 1;
    while (p < c && p < kShardCap) p <<= 1;  // round up to pow2, capped
    return p;
  }

  static std::uint64_t pack(std::uint32_t idx, std::uint32_t ver) noexcept {
    return (static_cast<std::uint64_t>(ver) << 32) | idx;
  }
  static std::uint32_t idx_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h);
  }
  static std::uint32_t ver_of(std::uint64_t h) noexcept {
    return static_cast<std::uint32_t>(h >> 32);
  }

  // Process-wide rotating thread sequence -> a stable per-thread home shard. Purely a
  // DISTRIBUTION hint: correctness never depends on it (free routes by address), so the
  // header-inline thread_local being per-DLL is fine.
  static std::uint32_t tls_thread_seq() noexcept {
    static std::atomic<std::uint32_t> g{0};
    thread_local std::uint32_t id = g.fetch_add(1, std::memory_order_relaxed);
    return id;
  }

  // ---- TLS freelist cache (tbb-style same-thread fast path) --------------------
  // One ThreadCache node per (thread, arena). Thread owns the node lifetime
  // (thread_local list + dtor). Arena owns a registry for reset()/~MemoryArena
  // so free indices never die with a destroyed pool and never UAF after it.
  //
  // count/idxs are owner-only on the hot path (no atomics). reset/detach may
  // touch them only after concurrent allocate/deallocate is quiesced (same
  // contract as reset() itself). Remote steal was measured and rejected: a
  // correct steal needs a lock on every TLS op and drops the tbb win to TIE.
  //
  // `arena` is ATOMIC: the hot path loads it lockless to match the node to its
  // pool, while the cold teardown paths (detach on thread-exit, detach_all on
  // arena dtor) write it under the process-global registry lock. This closes the
  // dtor/thread-exit race where one party read `arena` while the other nulled it.
  //
  // The node itself is allocated from system_allocator(), NOT the arena's injected
  // backing: a node can outlive the arena (a live thread's cache is freed only when
  // THAT thread exits), and the injected backing may be gone by then (a C-ABI
  // wrapper frees its backing when the arena is destroyed). A node is internal
  // bookkeeping (like the registry lock), so a process-stable allocator is correct
  // and decouples node lifetime from the pool's injected allocator.
  struct ThreadCache {
    std::atomic<MemoryArena*> arena{nullptr};
    std::uint32_t count = 0;
    std::uint32_t idxs[kTlsCacheCap]{};
    ThreadCache* next_in_thread = nullptr;  // owner-thread-only
    ThreadCache* next_in_arena = nullptr;   // registry-lock-only
  };

  struct TlsCacheList {
    ThreadCache* head = nullptr;
    ~TlsCacheList() {
      while (head != nullptr) {
        ThreadCache* c = head;
        head = c->next_in_thread;
        c->next_in_thread = nullptr;
        // Decide + act under the global registry lock so the arena cannot be torn
        // down between reading c->arena and using it (detach_all_caches nulls it
        // under the same lock). If arena is already null, it detached us first.
        {
          detail::TlsRegistryGuard g;
          MemoryArena* a = c->arena.load(std::memory_order_relaxed);
          if (a != nullptr) a->detach_cache_locked(c);  // flush + unregister + null
        }
        free_thread_cache_node(c);
      }
    }
  };

  static TlsCacheList& tls_list() noexcept {
    thread_local TlsCacheList list;
    return list;
  }

  static void free_thread_cache_node(ThreadCache* c) noexcept {
    c->~ThreadCache();
    system_allocator().deallocate(c, sizeof(ThreadCache), alignof(ThreadCache));
  }

  ThreadCache* tls_cache(bool create) noexcept {
    TlsCacheList& list = tls_list();
    // Walk + reclaim zombies (arena destroyed, node left with arena==nullptr).
    // Without this, long-lived threads that churn many MemoryArenas leak one
    // ThreadCache node per destroyed arena until thread exit (red-team R2). The
    // walk reads `arena` with acquire so that observing null means detach already
    // finished flushing the node -> safe to unlink (owner-only) and free.
    ThreadCache* prev = nullptr;
    ThreadCache* c = list.head;
    while (c != nullptr) {
      ThreadCache* next = c->next_in_thread;
      MemoryArena* a = c->arena.load(std::memory_order_acquire);
      if (a == this) return c;
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
    if (!create || !valid_) return nullptr;
    void* mem = system_allocator().allocate(sizeof(ThreadCache), alignof(ThreadCache));
    if (mem == nullptr) return nullptr;  // fall back to pure shard CAS path
    auto* fresh = new (mem) ThreadCache();
    fresh->arena.store(this, std::memory_order_relaxed);
    fresh->next_in_thread = list.head;  // owner-only
    list.head = fresh;
    {
      detail::TlsRegistryGuard g;
      fresh->next_in_arena = cache_list_;
      cache_list_ = fresh;
    }
    return fresh;
  }

  // Flush tc->idxs[base .. base+n) back to the shards, batched: bucket the run by
  // owning shard and push each bucket in ONE CAS (push_shard_batch). Same-thread churn
  // is single-shard (a thread frees its own home-shard blocks) -> one bucket, one CAS;
  // cross-thread handoff (foreign blocks) fans out to a few buckets. `scratch` holds
  // one bucket's indices (<= n <= kTlsCacheCap). Leaves tc->count = base.
  void flush_range_batched(ThreadCache* tc, std::uint32_t base,
                           std::uint32_t n) noexcept {
    std::uint32_t scratch[kTlsCacheCap];
    // Selection by shard: for each not-yet-flushed index, gather all indices sharing
    // its shard into scratch, batch-push them, and mark them consumed (kNull). O(n)
    // per distinct shard; distinct shards are few (1 for same-thread, a handful else).
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::uint32_t idx = tc->idxs[base + i];
      if (idx == kNull) continue;  // already flushed as part of an earlier bucket
      const std::uint32_t s = shard_of(idx);
      std::uint32_t m = 0;
      scratch[m++] = idx;
      tc->idxs[base + i] = kNull;
      for (std::uint32_t j = i + 1; j < n; ++j) {
        const std::uint32_t o = tc->idxs[base + j];
        if (o != kNull && shard_of(o) == s) {
          scratch[m++] = o;
          tc->idxs[base + j] = kNull;
        }
      }
      push_shard_batch(s, scratch, m);
    }
    tc->count = base;
  }

  void flush_cache_to_shards(ThreadCache* tc) noexcept {
    flush_range_batched(tc, 0, tc->count);
  }

  void flush_cache_half(ThreadCache* tc) noexcept {
    const std::uint32_t keep = tc->count - tc->count / 2;  // keep the warm lower half
    flush_range_batched(tc, keep, tc->count - keep);
  }

  // Called from the thread_local dtor with the process-global registry lock ALREADY
  // held: flush this node's free blocks back to shards, unlink it from the arena
  // registry, and null its arena (release) so a concurrent tls_cache() walk that
  // observes null knows the flush is done. Caller frees the node after this returns.
  void detach_cache_locked(ThreadCache* tc) noexcept {
    flush_cache_to_shards(tc);
    ThreadCache** pp = &cache_list_;
    while (*pp != nullptr) {
      if (*pp == tc) {
        *pp = tc->next_in_arena;
        break;
      }
      pp = &(*pp)->next_in_arena;
    }
    tc->next_in_arena = nullptr;
    tc->arena.store(nullptr, std::memory_order_release);
  }

  void detach_all_caches() noexcept {
    detail::TlsRegistryGuard g;
    ThreadCache* c = cache_list_;
    cache_list_ = nullptr;
    while (c != nullptr) {
      ThreadCache* next = c->next_in_arena;
      flush_cache_to_shards(c);
      c->next_in_arena = nullptr;
      // Release: a thread exiting concurrently reads arena under the SAME lock, but
      // a lock-free tls_cache() zombie-walk reads it with acquire -- pair them so the
      // flush above happens-before that thread frees the node.
      c->arena.store(nullptr, std::memory_order_release);
      c = next;
    }
  }

  // reset(): drop every TLS freelist. Do NOT push to shards — rebuild below
  // reclaims the full index range (contract: no live blocks; TLS-held free
  // indices are not live). Keeps cache nodes registered for subsequent use.
  // Contract: callers quiesce concurrent allocate/deallocate first.
  void flush_all_caches_keep_registered() noexcept {
    detail::TlsRegistryGuard g;
    for (ThreadCache* c = cache_list_; c != nullptr; c = c->next_in_arena) {
      c->count = 0;
    }
  }

  // Pop a free block index from shard s's Treiber stack. kNull if that shard is empty.
  std::uint32_t pop_shard(std::uint32_t s) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      std::uint32_t idx = idx_of(old);
      if (idx == kNull) return kNull;  // this shard empty
      std::uint32_t nxt = next_[idx].load(std::memory_order_relaxed);
      std::uint64_t desired = pack(nxt, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return idx;
      }
    }
  }

  // Push block index back onto shard s's Treiber stack.
  void push_shard(std::uint32_t s, std::uint32_t idx) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      next_[idx].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(idx, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return;
      }
    }
  }

  // Pop up to `want` indices from shard s in ONE CAS (magazine refill). Writes the
  // taken indices into out[] and returns the count (0 if the shard was empty). The
  // walk reads next_[] of nodes STILL on the shard, but nothing commits until the CAS
  // gates on the full packed `old` (idx AND version): any concurrent pop/push on this
  // shard bumps the version (see pop_shard/push_shard/push_shard_batch), so a CAS that
  // still sees `old` proves the top-K chain was untouched since the load -> the walked
  // `new_top` is valid. Otherwise retry. ABA is the same 32-bit version guard the
  // single-block pop already relies on. O(want) walk + 1 CAS.
  std::uint32_t pop_shard_batch(std::uint32_t s, std::uint32_t want,
                                std::uint32_t* out) noexcept {
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      std::uint32_t first = idx_of(old);
      if (first == kNull) return 0;  // shard empty
      // Walk up to `want` nodes; `last` is the deepest taken, `new_top` its successor.
      std::uint32_t n = 1;
      std::uint32_t last = first;
      std::uint32_t nxt = next_[last].load(std::memory_order_relaxed);
      while (n < want && nxt != kNull) {
        last = nxt;
        nxt = next_[last].load(std::memory_order_relaxed);
        ++n;
      }
      std::uint64_t desired = pack(nxt, ver_of(old) + 1);  // nxt == new top (or kNull)
      if (heads_[s].h.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        // Won the chain [first .. last]; materialize its indices for the caller.
        std::uint32_t cur = first;
        for (std::uint32_t i = 0; i < n; ++i) {
          out[i] = cur;
          cur = next_[cur].load(std::memory_order_relaxed);
        }
        return n;
      }
      // CAS failed -> `old` reloaded; retry from the new head.
    }
  }

  // Push a run of `n` indices (all owned by shard s) in ONE CAS (magazine flush).
  // Links them into a private chain in next_[] (idxs[0] becomes the deepest, idxs[n-1]
  // the new head), then CAS-prepends the whole chain. The blocks are private (parked in
  // TLS, on no shard) until the CAS lands, so linking is race-free; the version bump
  // keeps it ABA-sound exactly like the single-block push. O(n) private links + 1 CAS.
  void push_shard_batch(std::uint32_t s, const std::uint32_t* idxs,
                        std::uint32_t n) noexcept {
    if (n == 0) return;
    // Chain idxs[n-1] -> idxs[n-2] -> ... -> idxs[0]; idxs[0] is the tail whose next
    // must point at the existing shard head (set inside the CAS loop).
    for (std::uint32_t i = n - 1; i > 0; --i)
      next_[idxs[i]].store(idxs[i - 1], std::memory_order_relaxed);
    const std::uint32_t chain_head = idxs[n - 1];
    const std::uint32_t chain_tail = idxs[0];
    std::uint64_t old = heads_[s].h.load(std::memory_order_acquire);
    for (;;) {
      next_[chain_tail].store(idx_of(old), std::memory_order_relaxed);
      std::uint64_t desired = pack(chain_head, ver_of(old) + 1);
      if (heads_[s].h.compare_exchange_weak(old, desired, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return;
      }
    }
  }

  // Shard owning block idx (contiguous index ranges -> O(1) address routing). The last
  // shard absorbs the remainder when count_ isn't a multiple of num_shards_.
  std::uint32_t shard_of(std::uint32_t idx) const noexcept {
    std::uint32_t s = idx / per_;
    return s < num_shards_ ? s : num_shards_ - 1;
  }

  // One Treiber-stack head per shard, each on its OWN cache line (kill false sharing
  // between shards -- the whole point of sharding).
  struct alignas(HYLE_CACHE_LINE_SIZE) ShardHead {
    std::atomic<std::uint64_t> h;  // index|version
  };

  Allocator alloc_;
  std::byte* slab_ = nullptr;
  HugeRegion huge_slab_;  // owns slab_ when large; empty → slab_ from alloc_
  std::atomic<std::uint32_t>* next_ = nullptr;  // parallel free-list links
  ShardHead* heads_ = nullptr;                  // num_shards_ cache-line-isolated heads
  std::size_t block_size_;
  std::size_t stride_;
  std::uint32_t count_;
  std::size_t alignment_;
  std::uint32_t num_shards_ = 0;
  std::uint32_t shard_mask_ = 0;  // num_shards_ - 1 (pow2), so home = seq & shard_mask_
  std::uint32_t per_ = 0;  // blocks per shard (last shard takes the remainder)
  bool valid_ = false;  // set true once the ctor fully built the arena (else OOM'd)
  // Registry of this arena's live TLS caches; mutated only under the process-global
  // detail::tls_registry_lock (attach / detach / reset / dtor). Read on the hot path
  // is NOT via this list -- allocate/deallocate reach the node through the per-thread
  // list and validate ownership with the node's atomic `arena`.
  ThreadCache* cache_list_ = nullptr;
};

// Bump (region) allocator over ONE MemoryArena block (clay's Arena). Borrows a
// block from the pool, hands out aligned bump slices from it (lock-free atomic
// bump -- concurrent allocate() is safe), and returns the whole block to the
// pool on destruction. There is NO per-allocation free: you release everything
// at once by destroying the Arena (or reset()). Ideal for a transient phase
// (one taskgraph pass, one request) backed by a reusable pool block.
//
// Difference vs StackAllocator: StackAllocator owns its own slab and has
// mark/rewind; Arena borrows a pooled block and returns it on destruction. Both
// are SINGLE-THREADED bump (no atomics -- a bump cursor is inherently one serial
// pointer; making it atomic only adds a per-alloc CAS and, shared across threads,
// the same single-point contention MemoryArena's head had). Concurrency = one
// Arena PER THREAD/fiber (each over its own pooled block) -- the SOTA model
// (per-core scratch arenas), not a contended shared cursor.
class Arena {
 public:
  explicit Arena(MemoryArena* pool)
      : pool_(pool) {
    HYLE_FATAL_ASSERT(pool != nullptr);
    block_ = static_cast<std::byte*>(pool_->allocate());
    block_size_ = pool_->block_size();
    // Pool exhausted -> invalid (block_ == nullptr); caller must check valid().
    // Per c-abi-authoring-guide 铁律 3, a resource failure (no free block) returns
    // gracefully -- it is not a contract violation, so no fatal assert here. The
    // dtor's pool_->deallocate(nullptr) is a documented no-op, so invalid is safe.
    cur_ = block_;
    end_ = block_ != nullptr ? block_ + block_size_ : nullptr;
  }

  // False iff the pool was exhausted at construction. An invalid Arena must not
  // be used (allocate/reset are UB on it); the C ABI checks this and returns NULL.
  bool valid() const noexcept { return block_ != nullptr; }

  ~Arena() { pool_->deallocate(block_); }

  HYLE_DISABLE_COPY_AND_MOVE(Arena);

  // Bump `bytes` aligned to `alignment`. nullptr if the block is full. Single-
  // threaded: a plain pointer bump (no atomic). One Arena is owned by one thread.
  void* allocate(std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t)) noexcept {
    HYLE_FATAL_ASSERT(is_pow2(alignment));  // Arena had NO guard: 0/non-pow2 -> wild ptr
    // Pointer-style bump: align cur_ directly, no +base/-base offset round-trip.
    auto cur = reinterpret_cast<std::uintptr_t>(cur_);
    std::uintptr_t aligned = (cur + alignment - 1) & ~(alignment - 1);
    auto e = reinterpret_cast<std::uintptr_t>(end_);
    if (aligned > e || bytes > e - aligned) return nullptr;
    cur_ = reinterpret_cast<std::byte*>(aligned + bytes);
    return reinterpret_cast<void*>(aligned);
  }

  // Release everything (reuse the block from the start). HYLE addition over clay
  // (clay relied solely on dtor); only safe when no allocations are live.
  void reset() noexcept { cur_ = block_; }

  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t used() const noexcept {
    return valid() ? static_cast<std::size_t>(cur_ - block_) : 0;
  }
  // Bytes left before the block is full (a max-aligned request up to this fits).
  std::size_t remaining() const noexcept {
    return valid() ? static_cast<std::size_t>(end_ - cur_) : 0;
  }

  // Adapt this Arena to the Allocator vtable (split): allocate -> bump, deallocate ->
  // no-op (the arena frees in bulk on destruction), realloc -> bump a fresh slice + copy
  // (arenas can't grow in place).
  Allocator as_allocator() noexcept {
    return Allocator(&arena_alloc, &arena_free, &arena_realloc, this);
  }

 private:
  static void* arena_alloc(void* state, std::size_t size, std::size_t alignment) noexcept {
    return static_cast<Arena*>(state)->allocate(size, alignment);
  }
  static void arena_free(void* /*state*/, void* /*ptr*/, std::size_t /*size*/,
                         std::size_t /*alignment*/) noexcept {
    // no-op: an arena frees everything in bulk on destruction / reset().
  }
  static void* arena_realloc(void* state, void* ptr, std::size_t old_size,
                             std::size_t new_size, std::size_t alignment) noexcept {
    auto* self = static_cast<Arena*>(state);
    if (new_size == 0) return nullptr;
    void* p = self->allocate(new_size, alignment);
    if (ptr != nullptr && p != nullptr && old_size != 0) {
      std::memcpy(p, ptr, old_size < new_size ? old_size : new_size);
    }
    return p;
  }

  MemoryArena* pool_;
  std::byte* block_;
  std::size_t block_size_;
  std::byte* cur_ = nullptr;  // bump cursor (pointer-style; one Arena per thread)
  std::byte* end_ = nullptr;  // one past the block end (block_ + block_size_)
};

}  // namespace hyle::mem
