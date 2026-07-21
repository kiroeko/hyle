#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/platform/thread.h"

using hyle::mem::SharedPool;
using hyle::mem::SharedPoolCached;
using hyle::mem::system_allocator;

namespace {
bool aligned(const void* p, std::size_t a) {
  return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}
}  // namespace

TEST(SharedPool, AllocateFreeRoundTrip) {
  SharedPool pool(system_allocator(), 64, 4);
  EXPECT_EQ(pool.capacity(), 4u);
  void* a = pool.allocate();
  void* b = pool.allocate();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
  pool.deallocate(a);
  pool.deallocate(b);
}

TEST(SharedPool, ExhaustionReturnsNull) {
  SharedPool pool(system_allocator(), 32, 3);
  void* p0 = pool.allocate();
  void* p1 = pool.allocate();
  void* p2 = pool.allocate();
  ASSERT_NE(p0, nullptr);
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);
  EXPECT_EQ(pool.allocate(), nullptr);  // exhausted
  pool.deallocate(p1);
  EXPECT_EQ(pool.allocate(), p1);  // freed block is reused
}

TEST(SharedPool, BlocksAlignedAndNonOverlapping) {
  constexpr std::size_t kAlign = 64;
  constexpr std::uint32_t kN = 16;
  SharedPool pool(system_allocator(), 100, kN, kAlign);
  std::vector<void*> ps;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(aligned(p, kAlign));
    std::memset(p, static_cast<int>(i), 100);  // write whole block
    ps.push_back(p);
  }
  // distinct addresses (non-overlap)
  std::unordered_set<void*> uniq(ps.begin(), ps.end());
  EXPECT_EQ(uniq.size(), kN);
  for (void* p : ps) pool.deallocate(p);
}

TEST(SharedPool, AboveClay64SlotLimit) {
  // clay capped arena allocators at 64 static slots (bug #5). Ours takes a
  // block_count parameter -- prove well past 64 works.
  constexpr std::uint32_t kN = 1000;
  SharedPool pool(system_allocator(), 16, kN);
  std::vector<void*> ps;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << "at " << i;
    ps.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : ps) pool.deallocate(p);
}

// Lock-free correctness gate (WSL TSan). N threads each loop alloc -> write a
// thread-unique marker -> verify it's still ours -> free. If two threads ever
// hold the same block, the marker check fails. Also tracks that no block leaks.
TEST(SharedPool, ConcurrentAllocFreeNoOverlap) {
  constexpr int kThreads = 8;
  constexpr int kItersPerThread = 100000;
  constexpr std::uint32_t kBlocks = 256;
  SharedPool pool(system_allocator(), 64, kBlocks);

  struct Args {
    SharedPool* pool;
    int tid;
    std::atomic<bool>* corrupt;
  };
  std::vector<Args> args(kThreads);
  std::atomic<bool> corrupt{false};
  std::vector<hyle::platform::Thread> threads;

  for (int t = 0; t < kThreads; ++t) {
    args[t] = Args{&pool, t, &corrupt};
    auto r = hyle::platform::Thread::spawn(
        [](void* a) {
          auto* x = static_cast<Args*>(a);
          const unsigned char mark = static_cast<unsigned char>(0x40 + x->tid);
          for (int i = 0; i < kItersPerThread; ++i) {
            void* p = x->pool->allocate();
            if (p == nullptr) {  // transient exhaustion under contention: retry
              hyle::platform::this_thread::yield();
              continue;
            }
            auto* bytes = static_cast<unsigned char*>(p);
            std::memset(bytes, mark, 64);  // claim the block
            // No other thread may hold this block now: it must still read ours.
            for (int k = 0; k < 64; ++k) {
              if (bytes[k] != mark) {
                x->corrupt->store(true, std::memory_order_relaxed);
                break;
              }
            }
            x->pool->deallocate(p);
          }
        },
        &args[t]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }
  for (auto& th : threads) th.join();

  EXPECT_FALSE(corrupt.load()) << "two threads held the same block";
  // All blocks must be reclaimable -> the free list is intact (no lost block).
  std::vector<void*> all;
  for (std::uint32_t i = 0; i < kBlocks; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << "lost block at " << i;
    all.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : all) pool.deallocate(p);
}

// reset() returns every block at once: exhaust the pool, reset (no live blocks),
// then the full capacity must be allocatable again.
TEST(SharedPool, ResetReturnsAllBlocks) {
  SharedPool pool(system_allocator(), 64, 16);
  std::vector<void*> first;
  for (void* p = pool.allocate(); p != nullptr; p = pool.allocate())
    first.push_back(p);
  EXPECT_EQ(first.size(), 16u);      // drained the whole pool
  // DON'T deallocate -- reset() reclaims everything in one shot (contract: no
  // live block still in USE; the memory itself is reclaimed wholesale).
  pool.reset();
  std::size_t again = 0;
  for (void* p = pool.allocate(); p != nullptr; p = pool.allocate()) ++again;
  EXPECT_EQ(again, 16u);             // full capacity available after reset
}

// A block allocated on one thread and freed on another migrates to the freeing
// thread's shared home list and remains reclaimable. This is the producer/consumer
// and work-stealing handoff shape used by mature pool tests.
TEST(SharedPool, CrossThreadFreeMigratesAndReclaims) {
  SharedPool pool(system_allocator(), 128, 64);
  constexpr int kN = 64;
  std::vector<void*> ptrs(kN, nullptr);
  struct Ctx { SharedPool* pool; std::vector<void*>* ptrs; };
  Ctx c{&pool, &ptrs};
  // Thread A allocates all blocks.
  auto alloc_t = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Ctx*>(p);
        for (auto& s : *x->ptrs) s = x->pool->allocate();
      }, &c);
  ASSERT_TRUE(alloc_t.has_value());
  std::move(*alloc_t).join();
  for (void* q : ptrs) ASSERT_NE(q, nullptr);
  // Thread B frees them all into its shared home shard.
  auto free_t = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Ctx*>(p);
        for (void* q : *x->ptrs) x->pool->deallocate(q);
      }, &c);
  ASSERT_TRUE(free_t.has_value());
  std::move(*free_t).join();
  // Everything freed -> full capacity allocatable again (no leak, no double-count).
  std::size_t again = 0;
  for (void* p = pool.allocate(); p != nullptr; p = pool.allocate()) ++again;
  EXPECT_EQ(again, static_cast<std::size_t>(kN));
}

// Hard capacity: blocks freed by another still-live thread are immediately visible;
// no publish/join barrier is needed and no false exhaustion is allowed.
TEST(SharedPool, RemoteFreeImmediatelyVisibleWhileThreadLives) {
  constexpr std::uint32_t kN = 32;
  SharedPool pool(system_allocator(), 64, kN);
  std::vector<void*> ptrs(kN, nullptr);
  std::atomic<int> phase{0};
  struct Ctx {
    SharedPool* pool;
    std::vector<void*>* ptrs;
    std::atomic<int>* phase;
  } c{&pool, &ptrs, &phase};

  for (void*& p : ptrs) {
    p = pool.allocate();
    ASSERT_NE(p, nullptr);
  }
  auto free_t = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Ctx*>(p);
        for (void* block : *x->ptrs) x->pool->deallocate(block);
        x->phase->store(1, std::memory_order_release);
        while (x->phase->load(std::memory_order_acquire) < 2)
          hyle::platform::this_thread::yield();
      },
      &c);
  ASSERT_TRUE(free_t.has_value());
  while (phase.load(std::memory_order_acquire) < 1)
    hyle::platform::this_thread::yield();

  std::vector<void*> held;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << "hard-capacity false exhaustion at " << i;
    held.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : held) pool.deallocate(p);
  phase.store(2, std::memory_order_release);
  std::move(*free_t).join();
}

TEST(SharedPool, EmptyScanRetriesWhenFreeMovesBehindScanner) {
  SharedPool pool(system_allocator(), 64, 2);  // exactly two one-block shards
  void* held0 = pool.testing_pop_from_shard(0);
  ASSERT_NE(held0, nullptr);  // shard 0 empty, shard 1 still free

  struct ScanCtx {
    std::atomic<int> phase{0};
  } scan;
  pool.testing_set_empty_scan_hook(
      [](void* raw, std::uint32_t shard) noexcept {
        auto* s = static_cast<ScanCtx*>(raw);
        int expected = 0;
        if (shard == 0 && s->phase.compare_exchange_strong(
                              expected, 1, std::memory_order_acq_rel)) {
          while (s->phase.load(std::memory_order_acquire) < 2)
            hyle::platform::this_thread::yield();
        }
      },
      &scan);

  struct Worker {
    SharedPool* pool;
    void* result = nullptr;
  } worker{&pool};
  auto thread = hyle::platform::Thread::spawn(
      [](void* raw) {
        auto* w = static_cast<Worker*>(raw);
        w->result = w->pool->testing_allocate_from_home(0);
      },
      &worker);
  ASSERT_TRUE(thread.has_value());
  while (scan.phase.load(std::memory_order_acquire) < 1)
    hyle::platform::this_thread::yield();

  // The worker already observed shard 0 empty. Move the sole free block behind its
  // scan: publish held0 to shard 0, then remove shard 1's block.
  pool.testing_push_to_shard(0, held0);
  void* held1 = pool.testing_pop_from_shard(1);
  scan.phase.store(2, std::memory_order_release);
  std::move(*thread).join();
  ASSERT_NE(held1, nullptr);

  // A one-pass scan returned nullptr here. The stable second collect notices shard 0
  // changed and retries, reclaiming held0 instead of falsely reporting exhaustion.
  EXPECT_EQ(worker.result, held0);
  pool.testing_set_empty_scan_hook(nullptr, nullptr);
  pool.deallocate(worker.result);
  pool.deallocate(held1);
}

TEST(SharedPool, SaturatedTagFallbackPreservesCapacity) {
  constexpr std::uint32_t kN = 64;
  SharedPool pool(system_allocator(), 64, kN);
  pool.testing_force_tag_saturation();
  std::vector<void*> blocks;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << i;
    blocks.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : blocks) pool.deallocate(p);
  blocks.clear();
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << "reclaim " << i;
    blocks.push_back(p);
  }
  for (void* p : blocks) pool.deallocate(p);
}

TEST(SharedPool, SaturatedTagFallbackIsConcurrentSafe) {
  constexpr int kThreads = 8;
  constexpr int kIters = 20000;
  SharedPool pool(system_allocator(), 64, 256);
  pool.testing_force_tag_saturation();
  std::atomic<bool> corrupt{false};
  struct Args {
    SharedPool* pool;
    int tid;
    std::atomic<bool>* corrupt;
  };
  std::vector<Args> args(kThreads);
  std::vector<hyle::platform::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    args[t] = Args{&pool, t, &corrupt};
    auto thread = hyle::platform::Thread::spawn(
        [](void* raw) {
          auto* a = static_cast<Args*>(raw);
          const unsigned char mark = static_cast<unsigned char>(0x60 + a->tid);
          for (int i = 0; i < kIters; ++i) {
            void* p = a->pool->allocate();
            if (p == nullptr) continue;
            std::memset(p, mark, 64);
            auto* bytes = static_cast<unsigned char*>(p);
            for (int k = 0; k < 64; ++k) {
              if (bytes[k] != mark) {
                a->corrupt->store(true, std::memory_order_relaxed);
                break;
              }
            }
            a->pool->deallocate(p);
          }
        },
        &args[t]);
    ASSERT_TRUE(thread.has_value());
    threads.push_back(std::move(*thread));
  }
  for (auto& thread : threads) thread.join();
  EXPECT_FALSE(corrupt.load(std::memory_order_relaxed));
}

// Same-thread free→alloc must reuse via TLS freelist (non-atomic path). After a
// free the next allocate returns the same pointer — proves the cache is live and
// not losing blocks to a second CAS round-trip identity.
TEST(SharedPoolCached, SameThreadTlsCacheReusesPointer) {
  SharedPoolCached pool(system_allocator(), 64, 8);
  void* a = pool.allocate();
  ASSERT_NE(a, nullptr);
  pool.deallocate(a);
  void* b = pool.allocate();
  EXPECT_EQ(a, b);  // TLS LIFO reuse
  pool.deallocate(b);
}

// Fill past kTlsCacheCap (128): overflow must flush without leak or double-free;
// full capacity remains reclaimable.
TEST(SharedPoolCached, TlsCacheOverflowStillReclaimsAll) {
  constexpr std::uint32_t kN = 256;  // > 128 TLS cap
  SharedPoolCached pool(system_allocator(), 32, kN);
  std::vector<void*> ps;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    ps.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : ps) pool.deallocate(p);  // 128 stay in TLS, rest flush
  std::size_t again = 0;
  for (void* p = pool.allocate(); p != nullptr; p = pool.allocate()) ++again;
  EXPECT_EQ(again, static_cast<std::size_t>(kN));
}

// reset() must clear TLS-held free indices so the rebuilt shard freelist is the
// sole source of truth (no double-handout of a TLS-cached index).
TEST(SharedPoolCached, ResetClearsTlsCache) {
  SharedPoolCached pool(system_allocator(), 64, 8);
  void* a = pool.allocate();
  void* b = pool.allocate();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  pool.deallocate(a);  // lands in TLS
  pool.deallocate(b);
  pool.reset();  // must drop TLS, rebuild full freelist
  std::size_t n = 0;
  for (void* p = pool.allocate(); p != nullptr; p = pool.allocate()) ++n;
  EXPECT_EQ(n, 8u);
}

// Soft capacity (red-team R1): free-to-TLS on a live thread parks blocks off-shard.
// Without publish/join, another thread can see nullptr while free blocks exist.
// publish_thread_cache() is the explicit handoff barrier.
TEST(SharedPoolCached, RemoteTlsSoftThenPublishRestoresHardCapacity) {
  constexpr std::uint32_t kN = 32;
  SharedPoolCached pool(system_allocator(), 64, kN);
  std::vector<void*> ptrs(kN, nullptr);
  struct Ctx {
    SharedPoolCached* pool;
    std::vector<void*>* ptrs;
    std::atomic<int>* phase;
  };
  std::atomic<int> phase{0};
  Ctx c{&pool, &ptrs, &phase};
  auto free_t = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Ctx*>(p);
        while (x->phase->load(std::memory_order_acquire) < 1)
          hyle::platform::this_thread::yield();
        for (void* q : *x->ptrs) x->pool->deallocate(q);
        // Phase 2: frees parked in TLS, not yet published.
        x->phase->store(2, std::memory_order_release);
        while (x->phase->load(std::memory_order_acquire) < 3)
          hyle::platform::this_thread::yield();
        x->pool->publish_thread_cache();  // handoff barrier
        x->phase->store(4, std::memory_order_release);
        while (x->phase->load(std::memory_order_acquire) < 5)
          hyle::platform::this_thread::yield();
      },
      &c);
  ASSERT_TRUE(free_t.has_value());
  for (std::uint32_t i = 0; i < kN; ++i) {
    ptrs[i] = pool.allocate();
    ASSERT_NE(ptrs[i], nullptr);
  }
  phase.store(1, std::memory_order_release);
  while (phase.load(std::memory_order_acquire) < 2)
    hyle::platform::this_thread::yield();
  // Soft window: free thread still live, TLS not published → main may exhaust.
  EXPECT_EQ(pool.allocate(), nullptr);
  phase.store(3, std::memory_order_release);
  while (phase.load(std::memory_order_acquire) < 4)
    hyle::platform::this_thread::yield();
  // After publish: full capacity visible on shards.
  std::vector<void*> held;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << "post-publish at " << i;
    held.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : held) pool.deallocate(p);
  phase.store(5, std::memory_order_release);
  std::move(*free_t).join();
}

// Churn many cached-pool lifetimes on one thread: ThreadCache zombies
// (pool==nullptr) must be reclaimed instead of growing an unbounded TLS list.
TEST(SharedPoolCached, ManyPoolLifetimesDoNotBreakTls) {
  for (int i = 0; i < 64; ++i) {
    SharedPoolCached pool(system_allocator(), 32, 8);
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    pool.deallocate(p);
  }
  SharedPoolCached live(system_allocator(), 32, 4);
  std::size_t n = 0;
  for (void* p = live.allocate(); p != nullptr; p = live.allocate()) ++n;
  EXPECT_EQ(n, 4u);
}

// Batched (magazine) refill/flush correctness gate (WSL TSan). Each thread holds
// far past kTlsCacheCap (300 > 128), driving overflow publish and batch refill.
// Marker-verify catches any block handed to two threads (a batch pop/push that
// corrupts the next_[] chain); the final full drain catches any leaked or double-
// counted block (a batch transfer that lost or duplicated an index).
TEST(SharedPoolCached, ConcurrentBatchedRefillFlushNoLossNoOverlap) {
  constexpr int kThreads = 8;
  constexpr int kHold = 300;  // >> kTlsCacheCap (128)
  constexpr int kRounds = 2000;
  constexpr std::uint32_t kBlocks = kThreads * (kHold + 8);
  SharedPoolCached pool(system_allocator(), 64, kBlocks);
  std::atomic<bool> corrupt{false};

  struct Args { SharedPoolCached* pool; int tid; std::atomic<bool>* corrupt; };
  std::vector<Args> args(kThreads);
  std::vector<hyle::platform::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    args[t] = Args{&pool, t, &corrupt};
    auto r = hyle::platform::Thread::spawn(
        [](void* a) {
          auto* x = static_cast<Args*>(a);
          const unsigned char mark = static_cast<unsigned char>(0x40 + x->tid);
          std::vector<void*> held(kHold);
          for (int r2 = 0; r2 < kRounds; ++r2) {
            for (int i = 0; i < kHold; ++i) {
              void* p = x->pool->allocate();
              held[i] = p;
              if (p != nullptr) std::memset(p, mark, 64);
            }
            for (int i = 0; i < kHold; ++i) {
              if (held[i] == nullptr) continue;
              auto* b = static_cast<unsigned char*>(held[i]);
              for (int k = 0; k < 64; ++k)
                if (b[k] != mark) { x->corrupt->store(true, std::memory_order_relaxed); break; }
              x->pool->deallocate(held[i]);
            }
          }
        },
        &args[t]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }
  for (auto& th : threads) th.join();

  EXPECT_FALSE(corrupt.load()) << "two threads held the same block (batch chain corruption)";
  // Every block reclaimable exactly once -> batch pop/push preserved the free list.
  std::size_t drained = 0;
  for (void* p = pool.allocate(); p != nullptr; p = pool.allocate()) ++drained;
  EXPECT_EQ(drained, static_cast<std::size_t>(kBlocks)) << "leak or double-count in batch transfer";
}

TEST(SharedPoolCached, SaturatedTagFallbackSupportsMagazineTransfer) {
  constexpr std::uint32_t kN = 256;  // past TLS capacity -> shared batch paths
  SharedPoolCached pool(system_allocator(), 64, kN);
  pool.testing_force_tag_saturation();
  std::vector<void*> blocks;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << i;
    blocks.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : blocks) pool.deallocate(p);
  pool.publish_thread_cache();
  blocks.clear();
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << "reclaim " << i;
    blocks.push_back(p);
  }
  for (void* p : blocks) pool.deallocate(p);
}

// A wrapping Allocator whose backing state lives in a heap block the TEST frees
// on demand -- mirrors the C-ABI wrapper (hyle_shared_pool_cached_o stores the
// backing by value and frees it in destroy()). If a ThreadCache node ever freed
// itself through the pool's injected allocator, it would touch this freed state.
// A process-stable node allocator must make that impossible.
struct BackingBox {
  std::atomic<int> live_allocs{0};
  std::atomic<std::size_t> largest_request{0};
  bool freed = false;
};
namespace {
void* box_alloc(void* s, std::size_t n, std::size_t a) noexcept {
  auto* box = static_cast<BackingBox*>(s);
  box->live_allocs.fetch_add(1, std::memory_order_relaxed);
  std::size_t seen = box->largest_request.load(std::memory_order_relaxed);
  while (seen < n && !box->largest_request.compare_exchange_weak(
                         seen, n, std::memory_order_relaxed)) {}
  return system_allocator().allocate(n, a);
}
void box_free(void* s, void* p, std::size_t n, std::size_t a) noexcept {
  auto* box = static_cast<BackingBox*>(s);
  box->live_allocs.fetch_sub(1, std::memory_order_relaxed);
  system_allocator().deallocate(p, n, a);
}
void* box_realloc(void* s, void* p, std::size_t o, std::size_t n,
                  std::size_t a) noexcept {
  (void)s;
  return system_allocator().reallocate(p, o, n, a);
}
}  // namespace

TEST(SharedPool, LargeSlabHonorsInjectedBacking) {
  constexpr std::size_t kBlockSize = 4096;
  constexpr std::uint32_t kBlocks = 1024;
  constexpr std::size_t kSlabBytes = kBlockSize * kBlocks;
  BackingBox box;
  hyle::mem::Allocator backing(&box_alloc, &box_free, &box_realloc, &box);
  {
    SharedPool pool(backing, kBlockSize, kBlocks);
    ASSERT_TRUE(pool.valid());
    EXPECT_GE(box.largest_request.load(std::memory_order_relaxed), kSlabBytes);
  }
  EXPECT_EQ(box.live_allocs.load(std::memory_order_relaxed), 0);
}

// Regression (审计缺陷 1): a SharedPoolCached uses an INJECTED backing whose state
// is freed at pool destruction. A worker builds a TLS node bound to the pool; the
// pool and backing die while the worker remains alive, then thread exit runs the
// TLS-cache dtor. The node must not touch the dead injected backing.
TEST(SharedPoolCached, InjectedBackingFreedBeforeWorkerExitNoUaf) {
  struct Ctx {
    std::atomic<int> phase{0};
  };
  Ctx ctx;
  // Backing box + pool live on the heap so we can destroy them mid-flight.
  auto* box = new BackingBox();
  hyle::mem::Allocator backing(&box_alloc, &box_free, &box_realloc, box);
  auto* pool = new SharedPoolCached(backing, 64, 8);

  struct Args { SharedPoolCached* pool; Ctx* ctx; };
  Args a{pool, &ctx};
  auto worker = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Args*>(p);
        void* blk = x->pool->allocate();  // builds this thread's TLS node
        x->pool->deallocate(blk);         // parks the index in the TLS node
        x->ctx->phase.store(1, std::memory_order_release);  // "node built"
        while (x->ctx->phase.load(std::memory_order_acquire) < 2)
          hyle::platform::this_thread::yield();
        // return -> thread_local TLS-cache dtor runs AFTER the pool is gone.
      },
      &a);
  ASSERT_TRUE(worker.has_value());

  while (ctx.phase.load(std::memory_order_acquire) < 1)
    hyle::platform::this_thread::yield();
  // Tear down the pool AND its injected backing while the worker is still alive.
  delete pool;
  delete box;  // the injected backing state is now freed
  ctx.phase.store(2, std::memory_order_release);  // let the worker exit
  std::move(*worker).join();  // node's TLS dtor must not touch the freed backing
}

// ---- Arena (bump region over one pool block) --------------------------------
using hyle::mem::Arena;

TEST(Arena, BumpAlignmentAndFull) {
  SharedPool pool(system_allocator(), 256, 4);  // 256-byte blocks
  Arena a(&pool);
  EXPECT_EQ(a.block_size(), 256u);
  void* p = a.allocate(16, 16);
  void* q = a.allocate(16, 16);
  ASSERT_NE(p, nullptr);
  ASSERT_NE(q, nullptr);
  EXPECT_LT(p, q);  // bumps upward
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 16, 0u);
  EXPECT_EQ(a.allocate(1000, 1), nullptr);  // won't fit the 256 block
}

TEST(Arena, ResetReuses) {
  SharedPool pool(system_allocator(), 128, 2);
  Arena a(&pool);
  void* p = a.allocate(64);
  ASSERT_NE(p, nullptr);
  a.reset();
  void* q = a.allocate(64);
  EXPECT_EQ(p, q);  // same region after reset
}

TEST(Arena, ReturnsBlockToPoolOnDestruction) {
  SharedPool pool(system_allocator(), 64, 1);  // single block
  { Arena a(&pool); EXPECT_NE(a.allocate(8), nullptr); }
  // block returned -> a second Arena can take it
  { Arena b(&pool); EXPECT_NE(b.allocate(8), nullptr); }
}

TEST(Arena, AsAllocatorBumpsDeallocNoop) {
  SharedPool pool(system_allocator(), 256, 1);
  Arena a(&pool);
  auto alloc = a.as_allocator();
  void* p = alloc.allocate(32, 16);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 16, 0u);
  alloc.deallocate(p, 32);  // no-op for an arena; must not crash or reuse
  void* q = alloc.allocate(32, 16);
  EXPECT_NE(p, q);  // deallocate did NOT free -> q is a fresh bump
}

// Arena is single-threaded by design; concurrency = one Arena PER THREAD (the SOTA
// per-core scratch model). Each thread owns its own Arena over its own pool block and
// bumps it independently -- no shared cursor, no contention. Verifies the per-thread
// arenas hand out disjoint memory (own block each) with zero atomics.
TEST(Arena, PerThreadArenasNoOverlap) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 200;
  constexpr std::size_t kBytes = 16;
  // One pool block per thread, each big enough for kPerThread bumps.
  SharedPool pool(system_allocator(), kPerThread * 64, kThreads);
  struct Args { SharedPool* pool; std::vector<void*> got; };
  std::vector<Args> args(kThreads);
  std::vector<hyle::platform::Thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    args[t].pool = &pool;
    auto r = hyle::platform::Thread::spawn(
        [](void* p) {
          auto* x = static_cast<Args*>(p);
          Arena a(x->pool);  // this thread's OWN arena over its OWN pooled block
          for (int i = 0; i < kPerThread; ++i) {
            void* q = a.allocate(kBytes, 16);
            if (q) x->got.push_back(q);
          }
        },
        &args[t]);
    ASSERT_TRUE(r.has_value());
    ts.push_back(std::move(*r));
  }
  for (auto& th : ts) th.join();
  // All returned pointers across all per-thread arenas must be distinct.
  std::unordered_set<void*> uniq;
  std::size_t total = 0;
  for (auto& x : args) for (void* q : x.got) { uniq.insert(q); ++total; }
  EXPECT_EQ(uniq.size(), total) << "per-thread arenas overlapped";
}

// Arena::allocate had NO alignment guard before #1 (alignment 0 or a non-power-of-2
// -> wild pointer / silent misalignment). #1 added is_pow2(), which fail-fast traps
// alignment 0. Locks in #1 for Arena::allocate.
TEST(ArenaDeathTest, ZeroAlignmentTraps) {
  EXPECT_DEATH(
      {
        SharedPool pool(system_allocator(), 4096, 2);
        Arena a(&pool);
        (void)a.allocate(16, 0);
      },
      ".*");
}
