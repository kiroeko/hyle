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

TEST(Arena, ConcurrentBumpNoOverlap) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 200;
  constexpr std::size_t kBytes = 16;
  MemoryArena pool(system_allocator(), kThreads * kPerThread * 64, 1);
  Arena a(&pool);
  struct Args { Arena* a; std::vector<void*> got; };
  std::vector<Args> args(kThreads);
  std::vector<hyle::platform::Thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    args[t].a = &a;
    auto r = hyle::platform::Thread::spawn(
        [](void* p) {
          auto* x = static_cast<Args*>(p);
          for (int i = 0; i < kPerThread; ++i) {
            void* q = x->a->allocate(kBytes, 16);
            if (q) x->got.push_back(q);
          }
        },
        &args[t]);
    ASSERT_TRUE(r.has_value());
    ts.push_back(std::move(*r));
  }
  for (auto& th : ts) th.join();
  // All returned pointers must be distinct (no two bumps overlapped).
  std::unordered_set<void*> uniq;
  std::size_t total = 0;
  for (auto& x : args) for (void* q : x.got) { uniq.insert(q); ++total; }
  EXPECT_EQ(uniq.size(), total) << "overlapping bump allocations";
}
