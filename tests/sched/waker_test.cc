#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h"

using hyle::mem::system_allocator;
using hyle::sched::current_waker;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::suspend;
using hyle::sched::Waker;

namespace {

// One wait episode: the fiber captures its Waker, signals it's ready, suspends,
// and (once re-injected by a wake) records that it resumed to completion.
struct Episode {
  std::atomic<bool> waker_ready{false};
  Waker waker;
  std::atomic<bool> resumed{false};
};

void waiting_entry(void* p) {
  auto* e = static_cast<Episode*>(p);
  e->waker = current_waker();                       // {scheduler, this fiber}
  e->waker_ready.store(true, std::memory_order_release);
  suspend();                                        // leave the run loop
  e->resumed.store(true, std::memory_order_release);  // re-injected -> ran here
}

}  // namespace

// A fiber suspends; a FOREIGN (non-worker) thread wakes it; it resumes to done.
TEST(Waker, SuspendThenWakeFromForeignThread) {
  Episode e;
  Fiber f(system_allocator(), kDefaultStackSize, &waiting_entry, &e);
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(&f);
    // Wait until the fiber has captured its Waker (it is now suspending).
    while (!e.waker_ready.load(std::memory_order_acquire))
      std::this_thread::yield();
    EXPECT_TRUE(e.waker.valid());
    EXPECT_FALSE(e.resumed.load());  // still parked: nothing rescheduled it
    e.waker.wake();                  // wake from this (main) thread
    sched.drain();                   // blocks until the fiber finishes
  }
  EXPECT_TRUE(e.resumed.load());
}

// Stress the wake-before-suspend rendezvous: many fibers each suspend, and the
// main thread wakes each the instant its Waker is ready -- which can land BEFORE
// the fiber has finished suspending. A broken rendezvous loses a wake -> that
// fiber never resumes -> drain() hangs -> ctest times out (a real failure).
TEST(Waker, ManyWakeBeforeSuspendRace) {
  constexpr int kN = 500;
  std::vector<std::unique_ptr<Episode>> eps;
  std::vector<std::unique_ptr<Fiber>> fibers;
  eps.reserve(kN);
  fibers.reserve(kN);
  for (int i = 0; i < kN; ++i) {
    eps.push_back(std::make_unique<Episode>());
    fibers.push_back(std::make_unique<Fiber>(system_allocator(),
                                             kDefaultStackSize, &waiting_entry,
                                             eps[i].get()));
  }
  {
    Scheduler sched(4, system_allocator(), 1024);
    for (auto& f : fibers) sched.spawn(f.get());
    // Wake each as soon as it published its Waker (races the fiber's suspend).
    for (auto& e : eps) {
      while (!e->waker_ready.load(std::memory_order_acquire))
        std::this_thread::yield();
      e->waker.wake();
    }
    sched.drain();
  }
  for (auto& e : eps) EXPECT_TRUE(e->resumed.load());  // every fiber resumed
}

namespace {
// A fiber that wakes another episode's Waker from INSIDE the scheduler (i.e.
// wake() called on a worker thread, the common io-completion case).
struct Waker2 {
  Episode* target;
};
void waker_entry(void* p) {
  auto* w = static_cast<Waker2*>(p);
  while (!w->target->waker_ready.load(std::memory_order_acquire)) Fiber::yield();
  w->target->waker.wake();
}
}  // namespace

// wake() called from a worker fiber (not a foreign thread) also re-injects.
TEST(Waker, WakeFromWorkerFiber) {
  Episode e;
  Waker2 w2{&e};
  Fiber waiter(system_allocator(), kDefaultStackSize, &waiting_entry, &e);
  Fiber waker(system_allocator(), kDefaultStackSize, &waker_entry, &w2);
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(&waiter);
    sched.spawn(&waker);
    sched.drain();
  }
  EXPECT_TRUE(e.resumed.load());
}

// current_waker() off a worker thread yields an invalid Waker (no fiber/sched).
TEST(Waker, CurrentWakerInvalidOffWorker) {
  Waker w = current_waker();
  EXPECT_FALSE(w.valid());
}
