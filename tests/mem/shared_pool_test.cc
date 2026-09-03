#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/pool_cache.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/shared_pool.h"
#include "test_thread.h"

using hyle::mem::test_support::Thread;

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
#endif

using hyle::mem::ConcurrentBlockPool;
using hyle::mem::BlockPoolCache;
using hyle::mem::AllocatorRef;
using hyle::mem::QuotaAllocator;
using hyle::mem::system_allocator;

namespace {
bool aligned(const void* p, std::size_t a) {
  return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}

#if defined(HYLE_TESTING)
struct RecordingBackingState {
  std::array<void *, 3> pointers{};
  std::array<std::size_t, 3> sizes{};
  std::array<std::size_t, 3> alignments{};
  std::array<void *, 3> freed_pointers{};
  std::array<std::size_t, 3> freed_sizes{};
  std::array<std::size_t, 3> freed_alignments{};
  std::size_t allocation_calls = 0;
  std::size_t deallocation_calls = 0;
  std::size_t outstanding = 0;
};

void *recording_allocate(void *opaque, std::size_t size,
                         std::size_t alignment) noexcept {
  auto &state = *static_cast<RecordingBackingState *>(opaque);
  void *pointer = system_allocator().allocate(size, alignment);
  if (state.allocation_calls < state.sizes.size()) {
    state.pointers[state.allocation_calls] = pointer;
    state.sizes[state.allocation_calls] = size;
    state.alignments[state.allocation_calls] = alignment;
  }
  ++state.allocation_calls;
  if (pointer != nullptr)
    ++state.outstanding;
  return pointer;
}

void recording_deallocate(void *opaque, void *pointer, std::size_t size,
                           std::size_t alignment) noexcept {
  if (pointer == nullptr)
    return;
  auto &state = *static_cast<RecordingBackingState *>(opaque);
  if (state.deallocation_calls < state.freed_sizes.size()) {
    state.freed_pointers[state.deallocation_calls] = pointer;
    state.freed_sizes[state.deallocation_calls] = size;
    state.freed_alignments[state.deallocation_calls] = alignment;
  }
  ++state.deallocation_calls;
  --state.outstanding;
  system_allocator().deallocate(pointer, size, alignment);
}

void *recording_reallocate(void *, void *, std::size_t, std::size_t,
                           std::size_t) noexcept {
  return nullptr;
}
#endif
}  // namespace

TEST(ConcurrentBlockPool, AllocateFreeRoundTrip) {
  ConcurrentBlockPool pool(system_allocator(), 64, 4);
  EXPECT_EQ(pool.capacity(), 4u);
  void* a = pool.allocate();
  void* b = pool.allocate();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
  pool.deallocate(a);
  pool.deallocate(b);
}

TEST(ConcurrentBlockPool, ExhaustionReturnsNull) {
  ConcurrentBlockPool pool(system_allocator(), 32, 3);
  void* p0 = pool.allocate();
  void* p1 = pool.allocate();
  void* p2 = pool.allocate();
  ASSERT_NE(p0, nullptr);
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);
  EXPECT_EQ(pool.allocate(), nullptr);  // exhausted
  pool.deallocate(p1);
  void* reused = pool.allocate();
  EXPECT_EQ(reused, p1);  // freed block is reused
  pool.deallocate(p0);
  pool.deallocate(p2);
  pool.deallocate(reused);
}

#if defined(HYLE_TESTING)
TEST(ConcurrentBlockPool, DenseShardHeadsUseTargetFamilySpacing) {
  RecordingBackingState state;
  AllocatorRef backing(&recording_allocate, &recording_deallocate,
                    &recording_reallocate, &state);
  {
    ConcurrentBlockPool pool(backing, 64, 64);
    ASSERT_TRUE(pool.valid());
    ASSERT_EQ(state.allocation_calls, 3u);
    EXPECT_EQ(state.alignments[2],
              hyle::core::kDestructiveInterferenceSize);
    EXPECT_EQ(state.sizes[2],
              static_cast<std::size_t>(pool.testing_num_shards()) *
                  hyle::core::kDestructiveInterferenceSize);
  }
  ASSERT_EQ(state.deallocation_calls, 3u);
  EXPECT_EQ(state.outstanding, 0u);
  std::array<bool, 3> matched{};
  for (std::size_t allocation = 0; allocation < state.sizes.size();
       ++allocation) {
    bool found = false;
    for (std::size_t deallocation = 0;
         deallocation < state.freed_sizes.size(); ++deallocation) {
      if (!matched[deallocation] &&
          state.pointers[allocation] == state.freed_pointers[deallocation] &&
          state.sizes[allocation] == state.freed_sizes[deallocation] &&
          state.alignments[allocation] ==
              state.freed_alignments[deallocation]) {
        matched[deallocation] = true;
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "allocation pair " << allocation
                       << " was not returned to the backing allocator";
  }
}

TEST(ConcurrentBlockPool, DefaultShardsFollowAvailableParallelism) {
#if defined(_WIN32)
  if (::GetActiveProcessorGroupCount() != 1) GTEST_SKIP();
  DWORD_PTR original = 0, system = 0;
  if (!::GetProcessAffinityMask(::GetCurrentProcess(), &original, &system) ||
      original == 0) {
    GTEST_SKIP();
  }
  const DWORD_PTR single = original & (~original + 1);
  if (!::SetProcessAffinityMask(::GetCurrentProcess(), single)) GTEST_SKIP();
  ConcurrentBlockPool pool(system_allocator(), 64, 64);
  const BOOL restored = ::SetProcessAffinityMask(::GetCurrentProcess(), original);
  ASSERT_NE(restored, 0);
  ASSERT_TRUE(pool.valid());
  EXPECT_EQ(pool.testing_num_shards(), 1u);
#elif defined(__linux__)
  cpu_set_t original;
  CPU_ZERO(&original);
  if (::sched_getaffinity(0, sizeof(original), &original) != 0) GTEST_SKIP();
  int first = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &original)) {
      first = cpu;
      break;
    }
  }
  if (first < 0) GTEST_SKIP();
  cpu_set_t single;
  CPU_ZERO(&single);
  CPU_SET(first, &single);
  if (::sched_setaffinity(0, sizeof(single), &single) != 0) GTEST_SKIP();
  ConcurrentBlockPool pool(system_allocator(), 64, 64);
  const int restored = ::sched_setaffinity(0, sizeof(original), &original);
  ASSERT_EQ(restored, 0);
  ASSERT_TRUE(pool.valid());
  EXPECT_EQ(pool.testing_num_shards(), 1u);
#else
  GTEST_SKIP() << "platform has no process affinity API";
#endif
}
#endif

TEST(ConcurrentBlockPool, BlocksAlignedAndNonOverlapping) {
  constexpr std::size_t kAlign = 64;
  constexpr std::uint32_t kN = 16;
  ConcurrentBlockPool pool(system_allocator(), 100, kN, kAlign);
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

TEST(ConcurrentBlockPool, AboveClay64SlotLimit) {
  // clay capped arena allocators at 64 static slots (bug #5). Ours takes a
  // block_count parameter -- prove well past 64 works.
  constexpr std::uint32_t kN = 1000;
  ConcurrentBlockPool pool(system_allocator(), 16, kN);
  std::vector<void*> ps;
  for (std::uint32_t i = 0; i < kN; ++i) {
    void* p = pool.allocate();
    ASSERT_NE(p, nullptr) << "at " << i;
    ps.push_back(p);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void* p : ps) pool.deallocate(p);
}

TEST(BlockPoolCache, CallerStorageReusesLocallyAndPublishes) {
  ConcurrentBlockPool pool(system_allocator(), 64, 8);
  std::array<std::uint32_t, 4> storage{};
  BlockPoolCache cache(pool, storage, 4);
  ASSERT_TRUE(cache.valid());
  EXPECT_EQ(cache.capacity(), 4u);
  EXPECT_EQ(cache.transfer_batch(), 4u);

  void *p = cache.allocate();
  ASSERT_NE(p, nullptr);
  cache.deallocate(p);
  // A refill is intentionally one batch from the caller's home shard.  Its
  // size depends on process-available parallelism and the resulting shard
  // width, so only the local-capacity contract is portable across CI runners.
  EXPECT_GE(cache.cached(), 1u);
  EXPECT_LE(cache.cached(), cache.capacity());
  EXPECT_EQ(cache.allocate(), p);
  cache.deallocate(p);

  cache.publish();
  EXPECT_EQ(cache.cached(), 0u);
  std::vector<void *> all;
  for (std::uint32_t i = 0; i < pool.capacity(); ++i) {
    void *block = pool.allocate();
    ASSERT_NE(block, nullptr);
    all.push_back(block);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void *block : all)
    pool.deallocate(block);
}

TEST(BlockPoolCache, LocalFreeIsSoftUntilExplicitPublish) {
  ConcurrentBlockPool pool(system_allocator(), 64, 4);
  std::array<std::uint32_t, 4> storage{};
  BlockPoolCache cache(pool, storage, 4);
  std::array<void *, 4> held{};

  for (void *&p : held) {
    p = cache.allocate();
    ASSERT_NE(p, nullptr);
  }
  for (void *p : held)
    cache.deallocate(p);

  EXPECT_EQ(cache.cached(), 4u);
  EXPECT_EQ(pool.allocate(), nullptr);
  cache.publish();

  for (void *&p : held) {
    p = pool.allocate();
    ASSERT_NE(p, nullptr);
  }
  for (void *p : held)
    pool.deallocate(p);
}

TEST(BlockPoolCache, DestructionPublishesCallerStorage) {
  ConcurrentBlockPool pool(system_allocator(), 64, 4);
  std::array<std::uint32_t, 4> storage{};
  {
    BlockPoolCache cache(pool, storage, 4);
    void *p = cache.allocate();
    ASSERT_NE(p, nullptr);
    cache.deallocate(p);

    const std::uint32_t locally_cached = cache.cached();
    ASSERT_GE(locally_cached, 1u);
    ASSERT_LE(locally_cached, pool.capacity());
    std::vector<void *> shared;
    for (std::uint32_t i = locally_cached; i < pool.capacity(); ++i) {
      void *block = pool.allocate();
      ASSERT_NE(block, nullptr);
      shared.push_back(block);
    }
    EXPECT_EQ(pool.allocate(), nullptr);
    for (void *block : shared)
      pool.deallocate(block);
  }

  std::array<void *, 4> held{};
  for (void *&p : held) {
    p = pool.allocate();
    ASSERT_NE(p, nullptr);
  }
  for (void *p : held)
    pool.deallocate(p);
}

TEST(BlockPoolCache, AllocatorViewKeepsFixedBlockContract) {
  ConcurrentBlockPool pool(system_allocator(), 128, 2, 64);
  std::array<std::uint32_t, 2> storage{};
  BlockPoolCache cache(pool, storage, 2);
  AllocatorRef alloc = cache.as_allocator();

  EXPECT_EQ(alloc.allocate(129, 64), nullptr);
  EXPECT_EQ(alloc.allocate(1, 128), nullptr);
  EXPECT_EQ(alloc.reallocate(nullptr, 0, 129, 64), nullptr);
  EXPECT_EQ(cache.cached(), 0u);
  void *p = alloc.allocate(96, 64);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(alloc.reallocate(p, 96, 128, 64), p);
  EXPECT_EQ(alloc.reallocate(p, 128, 256, 64), nullptr);
  alloc.deallocate(p, 128, 64);
  cache.publish();
}

TEST(BlockPoolCache, PublishMakesCoreResetSafe) {
  ConcurrentBlockPool pool(system_allocator(), 64, 8);
  std::array<std::uint32_t, 4> storage{};
  BlockPoolCache cache(pool, storage, 2);

  void *p = cache.allocate();
  ASSERT_NE(p, nullptr);
  cache.deallocate(p);
  ASSERT_GT(cache.cached(), 0u);

  // Every attached cache must be empty before the quiescent bulk reset.
  cache.publish();
  ASSERT_EQ(cache.cached(), 0u);
  pool.reset();

  std::vector<void *> all;
  for (std::uint32_t i = 0; i < pool.capacity(); ++i) {
    void *block = pool.allocate();
    ASSERT_NE(block, nullptr) << i;
    all.push_back(block);
  }
  EXPECT_EQ(pool.allocate(), nullptr);
  for (void *block : all)
    pool.deallocate(block);
}

TEST(BlockPoolCache, OwnedMetadataOomLeavesInvalidEmptyCache) {
  ConcurrentBlockPool pool(system_allocator(), 64, 8);
  QuotaAllocator metadata(system_allocator(), 3 * sizeof(std::uint32_t));
  BlockPoolCache cache(pool, metadata.get(), 4, 2);

  EXPECT_FALSE(cache.valid());
  EXPECT_EQ(cache.allocate(), nullptr);
  AllocatorRef alloc = cache.as_allocator();
  EXPECT_TRUE(alloc.valid());
  EXPECT_EQ(alloc.allocate(32, 16), nullptr);
  EXPECT_EQ(alloc.reallocate(nullptr, 0, 32, 16), nullptr);
  EXPECT_EQ(cache.cached(), 0u);
  EXPECT_EQ(metadata.used(), 0u);
}

TEST(BlockPoolCache, ZeroTransferBatchSelectsClampedDefault) {
  ConcurrentBlockPool pool(system_allocator(), 64, 64);
  std::array<std::uint32_t, 8> storage{};
  BlockPoolCache cache(pool, storage, 0);
  EXPECT_TRUE(cache.valid());
  EXPECT_EQ(cache.transfer_batch(), storage.size());
}

TEST(BlockPoolCache, ConcurrentOwnersPreserveEveryBlock) {
  constexpr int kThreads = 8;
  constexpr int kHold = 80;
  constexpr int kRounds = 500;
  constexpr std::uint32_t kBlocks = kThreads * (kHold + 8);
  ConcurrentBlockPool pool(system_allocator(), 64, kBlocks);
  std::atomic<bool> corrupt{false};

  struct Args {
    ConcurrentBlockPool *pool;
    int tid;
    std::atomic<bool> *corrupt;
    std::array<std::uint32_t, 32> storage{};
  };
  std::vector<Args> args(kThreads);
  std::vector<Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    args[t].pool = &pool;
    args[t].tid = t;
    args[t].corrupt = &corrupt;
    auto thread = Thread::spawn(
        [](void *raw) {
          auto *a = static_cast<Args *>(raw);
          BlockPoolCache cache(*a->pool, a->storage, 8);
          const unsigned char mark = static_cast<unsigned char>(0x40 + a->tid);
          std::array<void *, kHold> held{};
          for (int round = 0; round < kRounds; ++round) {
            for (void *&p : held) {
              p = cache.allocate();
              if (p != nullptr)
                std::memset(p, mark, 64);
            }
            for (void *p : held) {
              if (p == nullptr)
                continue;
              auto *bytes = static_cast<unsigned char *>(p);
              for (int i = 0; i < 64; ++i) {
                if (bytes[i] != mark) {
                  a->corrupt->store(true, std::memory_order_relaxed);
                  break;
                }
              }
              cache.deallocate(p);
            }
          }
          cache.publish();
        },
        &args[t]);
    ASSERT_TRUE(thread.has_value());
    threads.push_back(std::move(*thread));
  }
  for (auto &thread : threads)
    thread.join();

  EXPECT_FALSE(corrupt.load(std::memory_order_relaxed));
  std::vector<void *> drained;
  for (void *p = pool.allocate(); p != nullptr; p = pool.allocate())
    drained.push_back(p);
  EXPECT_EQ(drained.size(), static_cast<std::size_t>(kBlocks));
  for (void *p : drained)
    pool.deallocate(p);
}

#if defined(HYLE_TESTING)
TEST(BlockPoolCache, SaturatedTagFallbackSupportsBatchTransfer) {
  constexpr std::uint32_t kBlocks = 256;
  ConcurrentBlockPool pool(system_allocator(), 64, kBlocks);
  pool.testing_force_tag_saturation();
  std::array<std::uint32_t, 32> storage{};
  BlockPoolCache cache(pool, storage, 8);
  std::vector<void *> blocks;

  for (std::uint32_t i = 0; i < kBlocks; ++i) {
    void *p = cache.allocate();
    ASSERT_NE(p, nullptr) << i;
    blocks.push_back(p);
  }
  EXPECT_EQ(cache.allocate(), nullptr);
  for (void *p : blocks)
    cache.deallocate(p);
  cache.publish();

  blocks.clear();
  for (std::uint32_t i = 0; i < kBlocks; ++i) {
    void *p = pool.allocate();
    ASSERT_NE(p, nullptr) << "reclaim " << i;
    blocks.push_back(p);
  }
  for (void *p : blocks)
    pool.deallocate(p);
}
#endif

// Lock-free correctness gate (WSL TSan). N threads each loop alloc -> write a
// thread-unique marker -> verify it's still ours -> free. If two threads ever
// hold the same block, the marker check fails. Also tracks that no block leaks.
TEST(ConcurrentBlockPool, ConcurrentAllocFreeNoOverlap) {
  constexpr int kThreads = 8;
  constexpr int kItersPerThread = 100000;
  constexpr std::uint32_t kBlocks = 256;
  ConcurrentBlockPool pool(system_allocator(), 64, kBlocks);

  struct Args {
    ConcurrentBlockPool* pool;
    int tid;
    std::atomic<bool>* corrupt;
  };
  std::vector<Args> args(kThreads);
  std::atomic<bool> corrupt{false};
  std::vector<Thread> threads;

  for (int t = 0; t < kThreads; ++t) {
    args[t] = Args{&pool, t, &corrupt};
    auto r = Thread::spawn(
        [](void* a) {
          auto* x = static_cast<Args*>(a);
          const unsigned char mark = static_cast<unsigned char>(0x40 + x->tid);
          for (int i = 0; i < kItersPerThread; ++i) {
            void* p = x->pool->allocate();
            if (p == nullptr) {  // transient exhaustion under contention: retry
              std::this_thread::yield();
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
TEST(ConcurrentBlockPool, ResetReturnsAllBlocks) {
  ConcurrentBlockPool pool(system_allocator(), 64, 16);
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
  pool.reset();
}

// A block allocated on one thread and freed on another migrates to the freeing
// thread's shared home list and remains reclaimable. This is the producer/consumer
// and work-stealing handoff shape used by mature pool tests.
TEST(ConcurrentBlockPool, CrossThreadFreeMigratesAndReclaims) {
  ConcurrentBlockPool pool(system_allocator(), 128, 64);
  constexpr int kN = 64;
  std::vector<void*> ptrs(kN, nullptr);
  struct Ctx { ConcurrentBlockPool* pool; std::vector<void*>* ptrs; };
  Ctx c{&pool, &ptrs};
  // Thread A allocates all blocks.
  auto alloc_t = Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Ctx*>(p);
        for (auto& s : *x->ptrs) s = x->pool->allocate();
      }, &c);
  ASSERT_TRUE(alloc_t.has_value());
  std::move(*alloc_t).join();
  for (void* q : ptrs) ASSERT_NE(q, nullptr);
  // Thread B frees them all into its shared home shard.
  auto free_t = Thread::spawn(
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
  pool.reset();
}

// Hard capacity: blocks freed by another still-live thread are immediately visible;
// no publish/join barrier is needed and no false exhaustion is allowed.
TEST(ConcurrentBlockPool, RemoteFreeImmediatelyVisibleWhileThreadLives) {
  constexpr std::uint32_t kN = 32;
  ConcurrentBlockPool pool(system_allocator(), 64, kN);
  std::vector<void*> ptrs(kN, nullptr);
  std::atomic<int> phase{0};
  struct Ctx {
    ConcurrentBlockPool* pool;
    std::vector<void*>* ptrs;
    std::atomic<int>* phase;
  } c{&pool, &ptrs, &phase};

  for (void*& p : ptrs) {
    p = pool.allocate();
    ASSERT_NE(p, nullptr);
  }
  auto free_t = Thread::spawn(
      [](void* p) {
        auto* x = static_cast<Ctx*>(p);
        for (void* block : *x->ptrs) x->pool->deallocate(block);
        x->phase->store(1, std::memory_order_release);
        while (x->phase->load(std::memory_order_acquire) < 2)
          std::this_thread::yield();
      },
      &c);
  ASSERT_TRUE(free_t.has_value());
  while (phase.load(std::memory_order_acquire) < 1)
    std::this_thread::yield();

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

#if defined(HYLE_TESTING)
TEST(ConcurrentBlockPool, EmptyScanRetriesWhenFreeMovesBehindScanner) {
  ConcurrentBlockPool pool(system_allocator(), 64, 2);  // exactly two one-block shards
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
            std::this_thread::yield();
        }
      },
      &scan);

  struct Worker {
    ConcurrentBlockPool* pool;
    void* result = nullptr;
  } worker{&pool};
  auto thread = Thread::spawn(
      [](void* raw) {
        auto* w = static_cast<Worker*>(raw);
        w->result = w->pool->testing_allocate_from_home(0);
      },
      &worker);
  ASSERT_TRUE(thread.has_value());
  while (scan.phase.load(std::memory_order_acquire) < 1)
    std::this_thread::yield();

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

TEST(ConcurrentBlockPool, SaturatedTagFallbackPreservesCapacity) {
  constexpr std::uint32_t kN = 64;
  ConcurrentBlockPool pool(system_allocator(), 64, kN);
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

TEST(ConcurrentBlockPool, SaturatedTagFallbackIsConcurrentSafe) {
  constexpr int kThreads = 8;
  constexpr int kIters = 20000;
  ConcurrentBlockPool pool(system_allocator(), 64, 256);
  pool.testing_force_tag_saturation();
  std::atomic<bool> corrupt{false};
  struct Args {
    ConcurrentBlockPool* pool;
    int tid;
    std::atomic<bool>* corrupt;
  };
  std::vector<Args> args(kThreads);
  std::vector<Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    args[t] = Args{&pool, t, &corrupt};
    auto thread = Thread::spawn(
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
#endif

// Wrapping allocator used to prove large slabs honor their injected backing.
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

TEST(ConcurrentBlockPool, LargeSlabHonorsInjectedBacking) {
  constexpr std::size_t kBlockSize = 4096;
  constexpr std::uint32_t kBlocks = 1024;
  constexpr std::size_t kSlabBytes = kBlockSize * kBlocks;
  BackingBox box;
  hyle::mem::AllocatorRef backing(&box_alloc, &box_free, &box_realloc, &box);
  {
    ConcurrentBlockPool pool(backing, kBlockSize, kBlocks);
    ASSERT_TRUE(pool.valid());
    EXPECT_GE(box.largest_request.load(std::memory_order_relaxed), kSlabBytes);
  }
  EXPECT_EQ(box.live_allocs.load(std::memory_order_relaxed), 0);
}

TEST(ConcurrentBlockPool, AllocatorReallocateObeysCompleteContract) {
  ConcurrentBlockPool pool(system_allocator(), 128, 1, 64);
  auto alloc = pool.as_allocator();

  EXPECT_EQ(alloc.allocate(129, 64), nullptr);
  EXPECT_EQ(alloc.allocate(1, 128), nullptr);
  EXPECT_EQ(alloc.reallocate(nullptr, 0, 129, 64), nullptr);
  EXPECT_EQ(alloc.reallocate(nullptr, 0, 0, 16), nullptr);
  auto *p = static_cast<unsigned char *>(
      alloc.reallocate(nullptr, 0, 32, 16));
  ASSERT_NE(p, nullptr);
  p[0] = 0x5A;

  EXPECT_EQ(alloc.reallocate(p, 32, 64, 16), p);
  EXPECT_EQ(p[0], 0x5A);
  EXPECT_EQ(alloc.reallocate(p, 64, 256, 16), nullptr);
  EXPECT_EQ(p[0], 0x5A) << "failed grow must preserve the original block";

  EXPECT_EQ(alloc.reallocate(p, 64, 0, 16), nullptr);
  void *reused = alloc.allocate(128, 64);
  ASSERT_NE(reused, nullptr) << "realloc-to-zero must restore pool capacity";
  alloc.deallocate(reused, 128, 64);
}

#if defined(HYLE_TESTING)
TEST(ConcurrentBlockPoolDiagnosticsDeathTest, DoubleReturnTraps) {
  EXPECT_DEATH(
      {
        ConcurrentBlockPool pool(system_allocator(), 64, 1);
        void* p = pool.allocate();
        pool.deallocate(p);
        pool.deallocate(p);
      },
      ".*");
}

TEST(ConcurrentBlockPoolDiagnosticsDeathTest, ForeignPointerTraps) {
  EXPECT_DEATH(
      {
        ConcurrentBlockPool pool(system_allocator(), 64, 1);
        int foreign = 0;
        pool.deallocate(&foreign);
      },
      ".*");
}

TEST(ConcurrentBlockPoolDiagnosticsDeathTest,
     OutstandingBlockAtDestructionTraps) {
  EXPECT_DEATH(
      {
        ConcurrentBlockPool pool(system_allocator(), 64, 1);
        (void)pool.allocate();
      },
      ".*");
}
#endif

TEST(ConcurrentBlockPool, QuotaCompositionKeepsAccountingAndCapacityTogether) {
  ConcurrentBlockPool pool(system_allocator(), 128, 1, 64);
  hyle::mem::QuotaAllocator quota(pool.as_allocator(), 128);
  auto alloc = quota.get();

  auto *p = static_cast<unsigned char *>(alloc.allocate(32, 16));
  ASSERT_NE(p, nullptr);
  p[0] = 7;
  EXPECT_EQ(quota.used(), 32u);
  EXPECT_EQ(alloc.reallocate(p, 32, 96, 16), p);
  EXPECT_EQ(p[0], 7);
  EXPECT_EQ(quota.used(), 96u);

  EXPECT_EQ(alloc.reallocate(p, 96, 129, 16), nullptr);
  EXPECT_EQ(quota.used(), 96u);
  EXPECT_EQ(p[0], 7);

  EXPECT_EQ(alloc.reallocate(p, 96, 0, 16), nullptr);
  EXPECT_EQ(quota.used(), 0u);
  void *reused = alloc.allocate(64, 16);
  ASSERT_NE(reused, nullptr) << "balanced quota must not hide a leaked pool slot";
  alloc.deallocate(reused, 64, 16);
  EXPECT_EQ(quota.used(), 0u);
}

// ---- PooledArena (bump region over one pool block) -------------------------
using hyle::mem::PooledArena;

TEST(PooledArena, BumpAlignmentAndFull) {
  ConcurrentBlockPool pool(system_allocator(), 256, 4);  // 256-byte blocks
  PooledArena a(&pool);
  EXPECT_EQ(a.block_size(), 256u);
  void* p = a.allocate(16, 16);
  void* q = a.allocate(16, 16);
  ASSERT_NE(p, nullptr);
  ASSERT_NE(q, nullptr);
  EXPECT_LT(p, q);  // bumps upward
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 16, 0u);
  EXPECT_EQ(a.allocate(1000, 1), nullptr);  // won't fit the 256 block
}

TEST(PooledArena, UnrepresentableAlignmentReturnsNull) {
  ConcurrentBlockPool pool(system_allocator(), 256, 1);
  PooledArena region(&pool);
  constexpr std::size_t kHugeAlignment =
      std::size_t{1} << (sizeof(std::size_t) * 8 - 1);
  EXPECT_EQ(region.allocate(1, kHugeAlignment), nullptr);
  EXPECT_EQ(region.used(), 0u);
}

TEST(PooledArena, ZeroSizeReturnsNullWithoutAdvancing) {
  ConcurrentBlockPool pool(system_allocator(), 256, 1);
  PooledArena region(&pool);
  ASSERT_TRUE(region.valid());
  EXPECT_EQ(region.allocate(0), nullptr);
  EXPECT_EQ(region.used(), 0u);
}

TEST(PooledArena, ClearReuses) {
  ConcurrentBlockPool pool(system_allocator(), 128, 2);
  PooledArena a(&pool);
  void* p = a.allocate(64);
  ASSERT_NE(p, nullptr);
  a.clear();
  void* q = a.allocate(64);
  EXPECT_EQ(p, q);  // same region after reset
}

TEST(PooledArena, ReturnsBlockToPoolOnDestruction) {
  ConcurrentBlockPool pool(system_allocator(), 64, 1);  // single block
  { PooledArena a(&pool); EXPECT_NE(a.allocate(8), nullptr); }
  // block returned -> a second PooledArena can take it
  { PooledArena b(&pool); EXPECT_NE(b.allocate(8), nullptr); }
}

TEST(PooledArena, AsAllocatorBumpsDeallocNoop) {
  ConcurrentBlockPool pool(system_allocator(), 256, 1);
  PooledArena a(&pool);
  auto alloc = a.as_allocator();
  void* p = alloc.allocate(32, 16);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 16, 0u);
  alloc.deallocate(p, 32);  // no-op for a region; must not crash or reuse
  void* q = alloc.allocate(32, 16);
  EXPECT_NE(p, q);  // deallocate did NOT free -> q is a fresh bump
}

// PooledArena is single-threaded by design; concurrency = one region PER
// THREAD. Each thread owns a distinct borrowed pool block and bump cursor.
TEST(PooledArena, PerThreadRegionsNoOverlap) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 200;
  constexpr std::size_t kBytes = 16;
  // One pool block per thread, each big enough for kPerThread bumps.
  ConcurrentBlockPool pool(system_allocator(), kPerThread * 64, kThreads);
  std::atomic<int> ready{0};
  std::atomic<bool> release{false};
  struct Args {
    ConcurrentBlockPool* pool;
    std::atomic<int>* ready;
    std::atomic<bool>* release;
    std::vector<void*> got;
  };
  std::vector<Args> args(kThreads);
  std::vector<Thread> ts;
  bool spawn_failed = false;
  for (int t = 0; t < kThreads; ++t) {
    args[t].pool = &pool;
    args[t].ready = &ready;
    args[t].release = &release;
    auto r = Thread::spawn(
        [](void* p) {
          auto* x = static_cast<Args*>(p);
          PooledArena a(x->pool);
          for (int i = 0; i < kPerThread; ++i) {
            void* q = a.allocate(kBytes, 16);
            if (q) x->got.push_back(q);
          }
          x->ready->fetch_add(1, std::memory_order_release);
          x->ready->notify_one();
          x->release->wait(false, std::memory_order_acquire);
        },
        &args[t]);
    if (!r.has_value()) {
      spawn_failed = true;
      break;
    }
    ts.push_back(std::move(*r));
  }
  if (spawn_failed) {
    release.store(true, std::memory_order_release);
    release.notify_all();
    for (auto& th : ts) th.join();
    FAIL() << "could not create every worker thread";
  }

  int observed = ready.load(std::memory_order_acquire);
  while (observed != kThreads) {
    ready.wait(observed, std::memory_order_acquire);
    observed = ready.load(std::memory_order_acquire);
  }

  // Keep every region alive while comparing addresses. Without this gate a
  // low-core runner may finish one worker before starting another, where reuse
  // of the returned pool block is both legal and expected.
  std::unordered_set<void*> uniq;
  std::size_t total = 0;
  for (auto& x : args) for (void* q : x.got) { uniq.insert(q); ++total; }
  EXPECT_EQ(uniq.size(), total) << "per-thread regions overlapped";

  release.store(true, std::memory_order_release);
  release.notify_all();
  for (auto& th : ts) th.join();
}

// PooledArena::allocate rejects alignment 0 or a non-power-of-2
// -> wild pointer / silent misalignment). #1 added is_pow2(), which fail-fast traps
// alignment 0.
TEST(PooledArenaDeathTest, ZeroAlignmentTraps) {
  EXPECT_DEATH(
      {
        ConcurrentBlockPool pool(system_allocator(), 4096, 2);
        PooledArena a(&pool);
        (void)a.allocate(16, 0);
      },
      ".*");
}
