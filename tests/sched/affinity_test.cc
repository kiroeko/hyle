#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h"

// shard-per-core (io-04): a fiber PINNED to a worker via spawn_on must run ONLY
// on that worker -- across cooperative yields AND across external wakes -- and is
// never stolen onto another core. This is the scheduler primitive that lets a
// connection's fiber stay on the core that owns its io ring (submit + reap +
// resume single-threaded). Pure sched here (no io): prove the pin holds.

using hyle::mem::system_allocator;
using hyle::sched::current_worker_index;
using hyle::sched::current_waker;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Waker;

namespace {

// Records which worker index it observed on each of many yields. If the pin holds,
// every observation is the same (== the requested owner).
struct PinProbe {
  int target;
  std::atomic<int> observations{0};
  std::atomic<int> mismatches{0};
  std::atomic<bool> done{false};
};

void pin_probe_entry(void* p) {
  auto* pr = static_cast<PinProbe*>(p);
  for (int i = 0; i < 200; ++i) {
    int idx = current_worker_index();
    pr->observations.fetch_add(1, std::memory_order_relaxed);
    if (idx != pr->target) pr->mismatches.fetch_add(1, std::memory_order_relaxed);
    Fiber::yield();  // cooperative yield -> must come back on the SAME worker
  }
  pr->done.store(true, std::memory_order_release);
}

}  // namespace

// A fiber pinned to a specific worker observes that worker's index on every one
// of 200 yields -- never stolen to another core.
TEST(Affinity, PinnedFiberStaysOnItsWorkerAcrossYields) {
  constexpr unsigned kWorkers = 4;
  for (unsigned target = 0; target < kWorkers; ++target) {
    Scheduler sched(kWorkers, system_allocator(), 256);
    auto pr = std::make_unique<PinProbe>();
    pr->target = static_cast<int>(target);
    Fiber f(system_allocator(), kDefaultStackSize, &pin_probe_entry, pr.get());
    sched.spawn_on(target, &f);
    sched.drain();
    sched.shutdown();
    EXPECT_TRUE(pr->done.load());
    EXPECT_EQ(pr->observations.load(), 200);
    EXPECT_EQ(pr->mismatches.load(), 0)
        << "pinned fiber ran on a worker other than " << target;
  }
}

// Many pinned fibers, spread across workers, each stays put even under contention
// (all workers busy so stealing WOULD happen if the pin didn't hold).
TEST(Affinity, ManyPinnedFibersEachStayHome) {
  constexpr unsigned kWorkers = 4;
  constexpr int kPerWorker = 8;
  Scheduler sched(kWorkers, system_allocator(), 1024);

  std::vector<std::unique_ptr<PinProbe>> probes;
  std::vector<std::unique_ptr<Fiber>> fibers;
  for (unsigned wk = 0; wk < kWorkers; ++wk) {
    for (int j = 0; j < kPerWorker; ++j) {
      auto pr = std::make_unique<PinProbe>();
      pr->target = static_cast<int>(wk);
      auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                       &pin_probe_entry, pr.get());
      sched.spawn_on(wk, f.get());
      probes.push_back(std::move(pr));
      fibers.push_back(std::move(f));
    }
  }
  sched.drain();
  sched.shutdown();

  for (auto& pr : probes) {
    EXPECT_TRUE(pr->done.load());
    EXPECT_EQ(pr->observations.load(), 200);
    EXPECT_EQ(pr->mismatches.load(), 0);
  }
}

// A pinned fiber woken by an EXTERNAL (foreign-thread) Waker resumes back on its
// owner worker, not wherever the wake happened -- the wake-back-to-origin path
// (reinject routes pinned fibers to their owner's private queue).
namespace {
struct WakePin {
  int target;
  std::atomic<bool> ready{false};
  Waker waker;
  std::atomic<int> before{-1};
  std::atomic<int> after{-1};
  std::atomic<bool> done{false};
};
void wake_pin_entry(void* p) {
  auto* w = static_cast<WakePin*>(p);
  w->before.store(current_worker_index(), std::memory_order_relaxed);
  w->waker = current_waker();
  w->ready.store(true, std::memory_order_release);
  hyle::sched::suspend();  // leave the run loop until the foreign wake
  w->after.store(current_worker_index(), std::memory_order_relaxed);
  w->done.store(true, std::memory_order_release);
}
}  // namespace

TEST(Affinity, PinnedFiberWakesBackOnOwnerWorker) {
  constexpr unsigned kWorkers = 4;
  constexpr unsigned kTarget = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  auto w = std::make_unique<WakePin>();
  w->target = static_cast<int>(kTarget);
  Fiber f(system_allocator(), kDefaultStackSize, &wake_pin_entry, w.get());
  sched.spawn_on(kTarget, &f);

  while (!w->ready.load(std::memory_order_acquire)) std::this_thread::yield();
  EXPECT_TRUE(w->waker.valid());
  w->waker.wake();  // foreign-thread wake -> must re-inject onto owner worker

  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(w->done.load());
  EXPECT_EQ(w->before.load(), static_cast<int>(kTarget));
  EXPECT_EQ(w->after.load(), static_cast<int>(kTarget))
      << "woken pinned fiber resumed on the wrong worker";
}
