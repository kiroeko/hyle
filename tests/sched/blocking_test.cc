#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::BlockingPool;
using hyle::sched::Fiber;
using hyle::sched::JoinHandle;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::spawn_blocking;

namespace {

// TSan-visible (malloc-backed) allocator for the churn stress test -- same
// rationale as join_test.cc: mimalloc's own heap blinds TSan to free->reuse, so a
// high-churn test would false-positive; malloc is intercepted, keeping the gate
// meaningful. Non-stress tests use system_allocator() (mimalloc, the ship path).
#if defined(_MSC_VER)
void* tsan_ma(void*, std::size_t n, std::size_t al) noexcept {
  if (al < sizeof(void*)) al = sizeof(void*);
  return _aligned_malloc(n, al);
}
void tsan_mf(void*, void* p, std::size_t, std::size_t) noexcept { _aligned_free(p); }
void* tsan_mr(void*, void* p, std::size_t, std::size_t nn, std::size_t al) noexcept {
  if (al < sizeof(void*)) al = sizeof(void*);
  return _aligned_realloc(p, nn, al);
}
#else
void* tsan_ma(void*, std::size_t n, std::size_t al) noexcept {
  if (al < sizeof(void*)) al = sizeof(void*);
  return std::aligned_alloc(al, (n + al - 1) & ~(al - 1));
}
void tsan_mf(void*, void* p, std::size_t, std::size_t) noexcept { std::free(p); }
void* tsan_mr(void*, void* p, std::size_t, std::size_t nn, std::size_t) noexcept {
  return std::realloc(p, nn);
}
#endif
Allocator tsan_visible_allocator() noexcept {
  return Allocator(&tsan_ma, &tsan_mf, &tsan_mr, nullptr);
}

}  // namespace

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
void offload_a(void* p) {
  auto* a = static_cast<OffloadArg*>(p);
  auto h = spawn_blocking(*a->pool, system_allocator(), [] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 7;
  });
  auto r = h.join();  // fiber-aware: worker is free while the pool thread sleeps
  (void)r;
  a->a_joined->store(true, std::memory_order_release);
}
void offload_b(void* p) {
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

// Concurrency stress (TSan gate): many spawn_blocking + join from the main thread,
// churning pool threads + JoinState alloc/free. TSan-visible allocator so the
// free->reuse edge stays visible (see note atop file).
TEST(Blocking, ChurnStress) {
  constexpr int kN = 200;
  Allocator al = tsan_visible_allocator();
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
