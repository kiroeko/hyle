#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::BlockingPool;
using hyle::sched::Fiber;
using hyle::sched::JoinHandle;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::spawn_blocking;

namespace {
struct AllocationCounts {
  std::atomic<std::size_t> allocations{0};
  std::atomic<std::size_t> frees{0};
};

void* counting_alloc(void* raw, std::size_t size,
                     std::size_t alignment) noexcept {
  void* pointer = system_allocator().allocate(size, alignment);
  if (pointer != nullptr) {
    static_cast<AllocationCounts*>(raw)->allocations.fetch_add(
        1, std::memory_order_relaxed);
  }
  return pointer;
}

void counting_free(void* raw, void* pointer, std::size_t size,
                   std::size_t alignment) noexcept {
  if (pointer != nullptr) {
    static_cast<AllocationCounts*>(raw)->frees.fetch_add(
        1, std::memory_order_relaxed);
  }
  system_allocator().deallocate(pointer, size, alignment);
}

void* counting_realloc(void*, void* pointer, std::size_t old_size,
                       std::size_t new_size,
                       std::size_t alignment) noexcept {
  return system_allocator().reallocate(pointer, old_size, new_size, alignment);
}
}  // namespace

TEST(Blocking, RecoverableConstructionRejectsInvalidOrMissingResources) {
  hyle::Status status;
  BlockingPool invalid_allocator(AllocatorRef{}, 1, std::chrono::seconds(1),
                                 &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(invalid_allocator.valid());

  BlockingPool no_threads(system_allocator(), 0, std::chrono::seconds(1),
                          &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(no_threads.valid());

  BlockingPool negative_keep_alive(system_allocator(), 1,
                                   -std::chrono::milliseconds(1), &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(negative_keep_alive.valid());

  AllocatorRef failing(
      [](void*, std::size_t, std::size_t) noexcept -> void* { return nullptr; },
      [](void*, void*, std::size_t, std::size_t) noexcept {},
      [](void*, void*, std::size_t, std::size_t,
         std::size_t) noexcept -> void* { return nullptr; },
      nullptr);
  BlockingPool no_metadata(failing, 1, std::chrono::seconds(1), &status);
  EXPECT_TRUE(status.IsResourceExhausted());
  EXPECT_FALSE(no_metadata.valid());
}

TEST(Blocking, InjectedAllocatorOwnsPoolBookkeepingAndJobs) {
  AllocationCounts counts;
  AllocatorRef allocator(&counting_alloc, &counting_free, &counting_realloc,
                      &counts);
  {
    BlockingPool pool(allocator, 2, std::chrono::milliseconds(10));
    auto handle = spawn_blocking(pool, allocator, [] { return 42; });
    ASSERT_EQ(*handle.join(), 42);
    pool.shutdown();
  }
  EXPECT_GT(counts.allocations.load(std::memory_order_relaxed), 4u);
  EXPECT_EQ(counts.allocations.load(std::memory_order_relaxed),
            counts.frees.load(std::memory_order_relaxed));
}

// A blocking closure returns its value; a bare thread joins it (thread-mode).
TEST(Blocking, ReturnsValueThreadJoin) {
  BlockingPool pool(system_allocator());
  auto h = spawn_blocking(pool, system_allocator(), [] { return 42; });
  auto r = h.join();
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 42);
}

// A void closure runs for its side effect; join() confirms completion.
TEST(Blocking, VoidClosureRuns) {
  BlockingPool pool(system_allocator());
  std::atomic<bool> ran{false};
  auto h = spawn_blocking(pool, system_allocator(),
                          [&] { ran.store(true, std::memory_order_release); });
  auto r = h.join();
  EXPECT_TRUE(r.has_value());
  EXPECT_TRUE(ran.load());
}

TEST(Blocking, UserExceptionBecomesInternalStatus) {
  BlockingPool pool(system_allocator(), 1);
  auto h = spawn_blocking(pool, system_allocator(), []() -> int {
    throw 42;
  });
  auto r = h.join();
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsInternal());
  pool.shutdown();
}

TEST(Blocking, AllocationFailureBecomesResourceExhausted) {
  BlockingPool pool(system_allocator(), 1);
  auto fail = hyle::mem::AllocatorRef(
      [](void*, std::size_t, std::size_t) noexcept -> void* { return nullptr; },
      [](void*, void*, std::size_t, std::size_t) noexcept {},
      [](void*, void*, std::size_t, std::size_t,
         std::size_t) noexcept -> void* { return nullptr; },
      nullptr);
  auto h = spawn_blocking(pool, fail, [] { return 7; });
  ASSERT_TRUE(h.valid());
  EXPECT_TRUE(h.is_finished());
  auto r = h.join();
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsResourceExhausted());
  pool.shutdown();
}

// THE core value: a fiber offloads a BLOCKING call and the worker stays free. One
// worker, two fibers: fiber A spawn_blocking a closure that sleeps 100ms then
// join()s it (fiber-aware -- yields the worker while blocked); fiber B just runs
// and sets a flag. If spawn_blocking wrongly ran the sleep ON the worker, B could
// not run until A's sleep finished. We assert B ran well before A's join returned.
struct OffloadArg {
  BlockingPool* pool;
  std::atomic<bool>* b_ran;
  std::atomic<bool>* a_joined;
};
void offload_a(void* p) noexcept {
  auto* a = static_cast<OffloadArg*>(p);
  auto h = spawn_blocking(*a->pool, system_allocator(), [] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 7;
  });
  auto r = h.join();  // fiber-aware: worker is free while the pool thread sleeps
  (void)r;
  a->a_joined->store(true, std::memory_order_release);
}
void offload_b(void* p) noexcept {
  auto* a = static_cast<OffloadArg*>(p);
  // If the worker were stuck in A's blocking sleep, this never runs until after.
  a->b_ran->store(true, std::memory_order_release);
}
TEST(Blocking, WorkerNotStalledByBlockingCall) {
  BlockingPool pool(system_allocator());
  std::atomic<bool> b_ran{false};
  std::atomic<bool> a_joined{false};
  OffloadArg arg{&pool, &b_ran, &a_joined};
  auto fa = std::make_unique<Fiber>(kDefaultStackSize,
                                    &offload_a, &arg);
  auto fb = std::make_unique<Fiber>(kDefaultStackSize,
                                    &offload_b, &arg);
  Scheduler sched(1, system_allocator(), 256);  // ONE worker
  sched.spawn(fa.get());
  sched.spawn(fb.get());
  // Give the scheduler a moment: B should run almost immediately (A is parked on
  // the pool), long before A's 100ms sleep completes.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_TRUE(b_ran.load());        // B ran while A's blocking call was offloaded
  EXPECT_FALSE(a_joined.load());    // A still waiting on the 100ms pool sleep
  sched.drain();
  EXPECT_TRUE(a_joined.load());     // A's join eventually returns
}

// Many concurrent blocking tasks all complete with correct results, and the pool
// spawns multiple threads to run them in parallel (not serialized on one thread).
TEST(Blocking, ManyConcurrentTasks) {
  constexpr int kN = 32;
  BlockingPool pool(system_allocator());
  std::vector<JoinHandle<int>> hs;
  hs.reserve(kN);
  for (int i = 0; i < kN; ++i)
    hs.push_back(spawn_blocking(pool, system_allocator(), [i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return i * i;
    }));
  long sum = 0;
  for (int i = 0; i < kN; ++i) {
    auto r = hs[i].join();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, i * i);
    sum += *r;
  }
  long expect = 0;
  for (int i = 0; i < kN; ++i) expect += static_cast<long>(i) * i;
  EXPECT_EQ(sum, expect);
  // 32 tasks each sleeping 10ms: if serialized on one thread that's 320ms; the
  // pool ran them across many threads, so >1 thread was spawned.
  EXPECT_GT(pool.live_threads(), 1u);
}

// max_threads cap is honored: with a cap of 2 and more tasks than threads, the
// extra tasks queue and still all complete (never lost).
TEST(Blocking, RespectsThreadCap) {
  constexpr int kN = 10;
  BlockingPool pool(system_allocator(), /*max_threads=*/2);
  std::vector<JoinHandle<int>> hs;
  hs.reserve(kN);
  for (int i = 0; i < kN; ++i)
    hs.push_back(spawn_blocking(pool, system_allocator(), [i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      return i;
    }));
  long got = 0;
  for (auto& h : hs) {
    auto r = h.join();
    ASSERT_TRUE(r.has_value());
    got += *r;
  }
  EXPECT_EQ(got, 45);  // 0+1+..+9
  EXPECT_LE(pool.live_threads(), 2u);  // never exceeded the cap
}

// shutdown() cancels queued-but-unstarted tasks: their join() returns Cancelled.
// Fill 1 thread with a slow task, queue more, shut down -> queued ones cancel.
TEST(Blocking, ShutdownCancelsQueued) {
  BlockingPool pool(system_allocator(), /*max_threads=*/1);
  // First task occupies the single thread for a while.
  auto busy = spawn_blocking(pool, system_allocator(), [] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 1;
  });
  // These queue behind it (cap=1, thread busy).
  std::vector<JoinHandle<int>> queued;
  for (int i = 0; i < 5; ++i)
    queued.push_back(spawn_blocking(pool, system_allocator(), [] { return 2; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(5));  // ensure they're queued
  pool.shutdown();  // busy finishes; queued (unstarted) get cancelled
  auto rb = busy.join();
  EXPECT_TRUE(rb.has_value());  // the running one completed
  int cancelled = 0;
  for (auto& h : queued) {
    auto r = h.join();
    if (!r.has_value() && r.error().IsCancelled()) ++cancelled;
  }
  EXPECT_GT(cancelled, 0);  // at least some queued tasks were cancelled
}

TEST(Blocking, ConcurrentShutdownCallsAreCompletionBarriers) {
  BlockingPool pool(system_allocator(), /*max_threads=*/1);
  std::atomic<bool> started{false};
  std::atomic<bool> release{false};
  std::atomic<int> returned{0};
  auto busy = spawn_blocking(pool, system_allocator(), [&] {
    started.store(true, std::memory_order_release);
    while (!release.load(std::memory_order_acquire))
      std::this_thread::yield();
  });
  while (!started.load(std::memory_order_acquire))
    std::this_thread::yield();

  std::thread first([&] {
    pool.shutdown();
    returned.fetch_add(1, std::memory_order_release);
  });
  std::thread second([&] {
    pool.shutdown();
    returned.fetch_add(1, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(returned.load(std::memory_order_acquire), 0);
  release.store(true, std::memory_order_release);
  first.join();
  second.join();
  EXPECT_EQ(returned.load(std::memory_order_acquire), 2);
  EXPECT_TRUE(busy.join().has_value());
}

TEST(Blocking, CancelRequestSkipsQueuedClosureBeforeItStarts) {
  BlockingPool pool(system_allocator(), /*max_threads=*/1);
  std::atomic<bool> busy_started{false};
  std::atomic<bool> release_busy{false};
  std::atomic<bool> queued_ran{false};
  auto busy = spawn_blocking(pool, system_allocator(), [&] {
    busy_started.store(true, std::memory_order_release);
    while (!release_busy.load(std::memory_order_acquire))
      std::this_thread::yield();
  });
  while (!busy_started.load(std::memory_order_acquire))
    std::this_thread::yield();
  auto queued = spawn_blocking(pool, system_allocator(), [&] {
    queued_ran.store(true, std::memory_order_release);
  });
  queued.request_cancel();
  release_busy.store(true, std::memory_order_release);
  EXPECT_TRUE(busy.join().has_value());
  auto result = queued.join();
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsCancelled());
  EXPECT_FALSE(queued_ran.load(std::memory_order_acquire));
}

TEST(Blocking, RetiredThreadHandlesStayBoundedAcrossBursts) {
  BlockingPool pool(system_allocator(), /*max_threads=*/1,
                    std::chrono::milliseconds(1));
  for (int i = 0; i < 20; ++i) {
    auto h = spawn_blocking(pool, system_allocator(), [i] { return i; });
    ASSERT_EQ(*h.join(), i);
    EXPECT_LE(pool.retained_thread_handles(), 1u);
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
  }
}

// Concurrency stress (TSan gate): many spawn_blocking + join from the main thread,
// Churn pool threads and JoinState allocation/free through the shipping
// allocator. mimalloc is built with MI_TSAN in sanitizer configurations, so
// this path remains visible to TSan without a test-only allocator shape.
TEST(Blocking, ChurnStress) {
  constexpr int kN = 200;
  AllocatorRef al = system_allocator();
  BlockingPool pool(al, /*max_threads=*/8);
  std::atomic<int> total{0};
  for (int i = 0; i < kN; ++i) {
    auto h = spawn_blocking(pool, al, [i] { return i + 1; });
    auto r = h.join();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, i + 1);
    total.fetch_add(1, std::memory_order_relaxed);
  }
  EXPECT_EQ(total.load(), kN);
}
