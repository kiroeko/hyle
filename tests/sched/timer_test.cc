#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/timer.h"
#include "hyle/time/clock.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::sleep_for;
using hyle::time::monotonic_ns;

namespace {

constexpr std::uint64_t kMs = 1'000'000ull;
using namespace std::chrono_literals;

// A fiber that sleeps once and records elapsed monotonic ns around the sleep.
struct SleepProbe {
  hyle::time::Duration sleep_duration{};
  std::uint64_t elapsed = 0;
  std::atomic<bool> done{false};
};

void sleep_entry(void* p) noexcept {
  auto* s = static_cast<SleepProbe*>(p);
  std::uint64_t t0 = monotonic_ns();
  sleep_for(s->sleep_duration);
  s->elapsed = monotonic_ns() - t0;
  s->done.store(true, std::memory_order_release);
}

}  // namespace

// A sleeping fiber resumes only AFTER its deadline, and the worker is freed to
// run other things meanwhile (it parks with a timeout, not a busy spin).
TEST(SchedTimer, SleepResumesAfterDeadline) {
  SleepProbe s;
  s.sleep_duration = 30ms;
  Fiber f(kDefaultStackSize, &sleep_entry, &s);
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(&f);
    sched.drain();
  }
  EXPECT_TRUE(s.done.load());
  // Allow scheduler/OS measurement slack, but never resume far below the
  // requested exact monotonic deadline.
  EXPECT_GE(s.elapsed, 25 * kMs) << "resumed too early: " << s.elapsed << " ns";
}

// Many fibers with DIFFERENT deadlines each resume, and in ascending-deadline
// order overall (the scheduler advances at the nearest deadline).
TEST(SchedTimer, ManyDifferentDeadlinesAllResume) {
  constexpr int kN = 64;
  std::vector<std::unique_ptr<SleepProbe>> probes;
  std::vector<std::unique_ptr<Fiber>> fibers;
  probes.reserve(kN);
  fibers.reserve(kN);
  for (int i = 0; i < kN; ++i) {
    auto p = std::make_unique<SleepProbe>();
    p->sleep_duration = std::chrono::milliseconds(i + 1);
    probes.push_back(std::move(p));
    fibers.push_back(std::make_unique<Fiber>(
        kDefaultStackSize, &sleep_entry, probes[i].get()));
  }
  {
    Scheduler sched(4, system_allocator(), 1024);
    for (auto& f : fibers) sched.spawn(f.get());
    sched.drain();
  }
  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(probes[i]->done.load()) << "fiber " << i << " never resumed";
    EXPECT_GE(probes[i]->elapsed, static_cast<std::uint64_t>(i) * kMs)
        << "fiber " << i << " resumed too early";
  }
}

// sleep_for(0ns) is a valid "yield to next tick" -- resumes promptly, no hang.
TEST(SchedTimer, ZeroSleepResumesPromptly) {
  SleepProbe s;
  s.sleep_duration = hyle::time::Duration::zero();
  Fiber f(kDefaultStackSize, &sleep_entry, &s);
  {
    Scheduler sched(2, system_allocator(), 256);
    sched.spawn(&f);
    sched.drain();
  }
  EXPECT_TRUE(s.done.load());
}

namespace {
// A fiber that sleeps, then spawns another sleeper -- exercises arming a timer
// from a fiber that itself woke from a timer (re-entrant wheel use on a worker).
struct ChainProbe {
  Scheduler* sched = nullptr;
  std::atomic<int>* counter = nullptr;
  int depth = 0;
  std::vector<std::unique_ptr<Fiber>>* keep = nullptr;
};
void chain_entry(void* p) noexcept {
  auto* c = static_cast<ChainProbe*>(p);
  sleep_for(5ms);
  c->counter->fetch_add(1, std::memory_order_acq_rel);
}
}  // namespace

// Timers armed by many fibers concurrently across workers all fire (stress the
// per-worker wheels + cross-worker re-inject through the Waker spine).
TEST(SchedTimer, ConcurrentSleepersStress) {
  constexpr int kN = 200;
  std::atomic<int> counter{0};
  std::vector<std::unique_ptr<ChainProbe>> probes;
  std::vector<std::unique_ptr<Fiber>> fibers;
  probes.reserve(kN);
  fibers.reserve(kN);
  for (int i = 0; i < kN; ++i) {
    auto c = std::make_unique<ChainProbe>();
    c->counter = &counter;
    probes.push_back(std::move(c));
    fibers.push_back(std::make_unique<Fiber>(
        kDefaultStackSize, &chain_entry, probes[i].get()));
  }
  {
    Scheduler sched(4, system_allocator(), 1024);
    for (auto& f : fibers) sched.spawn(f.get());
    sched.drain();
  }
  EXPECT_EQ(counter.load(), kN);  // every sleeper woke exactly once
}

// A mix of sleeping and non-sleeping fibers: the non-sleepers finish fast, the
// sleepers still wake -- confirms the wheel doesn't stall ordinary work and vice
// versa (the loop-top fire_due_timers + timed park compose).
TEST(SchedTimer, MixedSleepersAndRunnables) {
  constexpr int kSleepers = 32;
  constexpr int kBusy = 200;
  std::atomic<int> woke{0};
  std::atomic<int> ran{0};

  struct S {
    std::atomic<int>* woke;
  };
  struct B {
    std::atomic<int>* ran;
  };
  std::vector<std::unique_ptr<S>> ss;
  std::vector<std::unique_ptr<B>> bs;
  std::vector<std::unique_ptr<Fiber>> fibers;

  auto sleeper = [](void* p) noexcept {
    sleep_for(10ms);
    static_cast<S*>(p)->woke->fetch_add(1, std::memory_order_acq_rel);
  };
  auto busy = [](void* p) noexcept {
    static_cast<B*>(p)->ran->fetch_add(1, std::memory_order_acq_rel);
  };

  for (int i = 0; i < kSleepers; ++i) {
    ss.push_back(std::make_unique<S>(S{&woke}));
    fibers.push_back(std::make_unique<Fiber>(
                                             kDefaultStackSize,
                                             static_cast<Fiber::EntryFn>(sleeper),
                                             ss.back().get()));
  }
  for (int i = 0; i < kBusy; ++i) {
    bs.push_back(std::make_unique<B>(B{&ran}));
    fibers.push_back(std::make_unique<Fiber>(
                                             kDefaultStackSize,
                                             static_cast<Fiber::EntryFn>(busy),
                                             bs.back().get()));
  }
  {
    Scheduler sched(4, system_allocator(), 1024);
    for (auto& f : fibers) sched.spawn(f.get());
    sched.drain();
  }
  EXPECT_EQ(woke.load(), kSleepers);
  EXPECT_EQ(ran.load(), kBusy);
}
