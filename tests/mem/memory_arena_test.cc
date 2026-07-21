#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/memory_arena.h"
#include "hyle/platform/thread.h"

using hyle::mem::MemoryArena;
using hyle::mem::system_allocator;

namespace {
bool aligned(const void* p, std::size_t a) {
  return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}
}  // namespace

TEST(MemoryArena, AllocateFreeRoundTrip) {
  MemoryArena pool(system_allocator(), 64, 4);
  EXPECT_EQ(pool.capacity(), 4u);
  void* a = pool.allocate();
  void* b = pool.allocate();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
  pool.deallocate(a);
  pool.deallocate(b);
}

TEST(MemoryArena, ExhaustionReturnsNull) {
  MemoryArena pool(system_allocator(), 32, 3);
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

TEST(MemoryArena, BlocksAlignedAndNonOverlapping) {
  constexpr std::size_t kAlign = 64;
  constexpr std::uint32_t kN = 16;
  MemoryArena pool(system_allocator(), 100, kN, kAlign);
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

TEST(MemoryArena, AboveClay64SlotLimit) {
  // clay capped arena allocators at 64 static slots (bug #5). Ours takes a
  // block_count parameter -- prove well past 64 works.
  constexpr std::uint32_t kN = 1000;
  MemoryArena pool(system_allocator(), 16, kN);
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
TEST(MemoryArena, ConcurrentAllocFreeNoOverlap) {
  constexpr int kThreads = 8;
  constexpr int kItersPerThread = 100000;
  constexpr std::uint32_t kBlocks = 256;
  MemoryArena pool(system_allocator(), 64, kBlocks);

  struct Args {
    MemoryArena* pool;
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
TEST(MemoryArena, ResetReturnsAllBlocks) {
  MemoryArena pool(system_allocator(), 64, 16);
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

// A block allocated on one thread and freed on ANOTHER must route back to its
// owning shard (free is address-routed, not home-shard) and stay reusable. This
// is the work-stealing pattern (fiber stack allocated on core A, freed on core B).
TEST(MemoryArena, CrossThreadFreeRoutesBack) {
  MemoryArena pool(system_allocator(), 128, 64);
  constexpr int kN = 64;
  std::vector<void*> ptrs(kN, nullptr);
  struct Ctx { MemoryArena* pool; std::vector<void*>* ptrs; };
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
  // Thread B frees them all (cross-thread, address-routed back to owning shards).
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

// Same-thread free→alloc must reuse via TLS freelist (non-atomic path). After a
// free the next allocate returns the same pointer — proves the cache is live and
// not losing blocks to a second CAS round-trip identity.
TEST(MemoryArena, SameThreadTlsCacheReusesPointer) {
  MemoryArena pool(system_allocator(), 64, 8);
  void* a = pool.allocate();
  ASSERT_NE(a, nullptr);
  pool.deallocate(a);
  void* b = pool.allocate();
  EXPECT_EQ(a, b);  // TLS LIFO reuse
  pool.deallocate(b);
}

// Fill past kTlsCacheCap (32): overflow must flush to shards without leak or
// double-free; full capacity still reclaimable afterwards.
TEST(MemoryArena, TlsCacheOverflowStillReclaimsAll) {
  constexpr std::uint32_t kN = 64;  // > 32 TLS cap
  MemoryArena pool(system_allocator(), 32, kN);
  std::vector<void*> ps;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    ps.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : ps) pool.deallocate(p);  // 32 stay in TLS, rest flush to shards
  std::size_t again = 0;
  for (void* p = pool.allocate(); p != nullptr; p = pool.allocate()) ++again;
  EXPECT_EQ(again, static_cast<std::size_t>(kN));
}

// reset() must clear TLS-held free indices so the rebuilt shard freelist is the
// sole source of truth (no double-handout of a TLS-cached index).
TEST(MemoryArena, ResetClearsTlsCache) {
  MemoryArena pool(system_allocator(), 64, 8);
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
// publish_thread_cache() is the explicit handoff barrier (steal-with-lock was
// measured: restores hard capacity but drops the tbb win to TIE — rejected).
TEST(MemoryArena, RemoteTlsSoftThenPublishRestoresHardCapacity) {
  constexpr std::uint32_t kN = 32;
  MemoryArena pool(system_allocator(), 64, kN);
  std::vector<void*> ptrs(kN, nullptr);
  struct Ctx {
    MemoryArena* pool;
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

// Churn many arena lifetimes on one thread: ThreadCache zombies (arena==nullptr)
// must be reclaimed so we do not grow an unbounded TLS list (red-team R2).
TEST(MemoryArena, ManyArenaLifetimesDoNotBreakTls) {
  for (int i = 0; i < 64; ++i) {
    MemoryArena pool(system_allocator(), 32, 8);
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr);
    pool.deallocate(p);
  }
  MemoryArena live(system_allocator(), 32, 4);
  std::size_t n = 0;
  for (void* p = live.allocate(); p != nullptr; p = live.allocate()) ++n;
  EXPECT_EQ(n, 4u);
}

// A wrapping Allocator whose backing state lives in a heap block the TEST frees
// on demand -- mirrors the C-ABI wrapper (hyle_memory_arena_o stores the backing
// by value and frees it in destroy()). If a ThreadCache node ever freed itself
// through the arena's injected allocator, it would touch this freed state. A
// process-stable node allocator must make that impossible.
struct BackingBox {
  std::atomic<int> live_allocs{0};
  bool freed = false;
};
namespace {
void* box_alloc(void* s, std::size_t n, std::size_t a) noexcept {
  auto* box = static_cast<BackingBox*>(s);
  box->live_allocs.fetch_add(1, std::memory_order_relaxed);
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

// Regression (审计缺陷 1): a MemoryArena built on an INJECTED backing whose state
// is freed at arena destruction. A worker builds a TLS node bound to the arena,
// the arena (and its backing box) are destroyed while the worker is still alive,
// then the worker exits and runs its thread_local TLS-cache dtor. The node must
// NOT free itself through the (now-dead) injected backing. Under ASan this is a
// deterministic UAF if the node lifetime is coupled to the injected allocator.
TEST(MemoryArena, InjectedBackingFreedBeforeWorkerExitNoUaf) {
  struct Ctx {
    std::atomic<int> phase{0};
  };
  Ctx ctx;
  // Backing box + arena live on the heap so we can destroy them mid-flight.
  auto* box = new BackingBox();
  hyle::mem::Allocator backing(&box_alloc, &box_free, &box_realloc, box);
  auto* pool = new MemoryArena(backing, 64, 8);

  struct Args { MemoryArena* pool; Ctx* ctx; };
  Args a{pool, &ctx};
  auto worker = hyle::platform::Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Args*>(p);
        void* blk = x->pool->allocate();  // builds this thread's TLS node
        x->pool->deallocate(blk);         // parks the index in the TLS node
        x->ctx->phase.store(1, std::memory_order_release);  // "node built"
        while (x->ctx->phase.load(std::memory_order_acquire) < 2)
          hyle::platform::this_thread::yield();
        // return -> thread_local TLS-cache dtor runs AFTER the arena is gone.
      },
      &a);
  ASSERT_TRUE(worker.has_value());

  while (ctx.phase.load(std::memory_order_acquire) < 1)
    hyle::platform::this_thread::yield();
  // Tear down the arena AND its injected backing while the worker is still alive.
  delete pool;
  delete box;  // the injected backing state is now freed
  ctx.phase.store(2, std::memory_order_release);  // let the worker exit
  std::move(*worker).join();  // node's TLS dtor must not touch the freed backing
}

// ---- Arena (bump region over one pool block) --------------------------------
using hyle::mem::Arena;

TEST(Arena, BumpAlignmentAndFull) {
  MemoryArena pool(system_allocator(), 256, 4);  // 256-byte blocks
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
  MemoryArena pool(system_allocator(), 128, 2);
  Arena a(&pool);
  void* p = a.allocate(64);
  ASSERT_NE(p, nullptr);
  a.reset();
  void* q = a.allocate(64);
  EXPECT_EQ(p, q);  // same region after reset
}

TEST(Arena, ReturnsBlockToPoolOnDestruction) {
  MemoryArena pool(system_allocator(), 64, 1);  // single block
  { Arena a(&pool); EXPECT_NE(a.allocate(8), nullptr); }
  // block returned -> a second Arena can take it
  { Arena b(&pool); EXPECT_NE(b.allocate(8), nullptr); }
}

TEST(Arena, AsAllocatorBumpsDeallocNoop) {
  MemoryArena pool(system_allocator(), 256, 1);
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
  MemoryArena pool(system_allocator(), kPerThread * 64, kThreads);
  struct Args { MemoryArena* pool; std::vector<void*> got; };
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
        MemoryArena pool(system_allocator(), 4096, 2);
        Arena a(&pool);
        (void)a.allocate(16, 0);
      },
      ".*");
}
