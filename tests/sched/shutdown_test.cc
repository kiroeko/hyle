#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// sched-08 graceful shutdown: two-phase drain for long-lived (server) fibers that
// never finish on their own. begin_shutdown() flips a flat "not accepting" flag +
// broadcasts it; a long-lived fiber polls shutdown_requested() at its yield points
// and breaks its loop; drain_for() waits (bounded) for the in-flight count to hit
// zero. No cancellation tree -- a flat scheduler-wide signal, per seastar/tokio.

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::kSmallStackSize;
using hyle::sched::Scheduler;
using hyle::sched::shutdown_requested;
using hyle::sched::wait_for_shutdown;

namespace {

// A long-lived fiber: loops "forever" (like an accept/connection loop), yielding
// each iteration, until it observes shutdown_requested() -- then it breaks and
// finishes (decrementing the scheduler's in-flight count so drain can complete).
struct LongLived {
  std::atomic<bool>* exited;
  std::atomic<int>* iterations;
};
void longlived_entry(void* p) {
  auto* s = static_cast<LongLived*>(p);
  while (!shutdown_requested()) {
    s->iterations->fetch_add(1, std::memory_order_relaxed);
    Fiber::yield();  // a real server fiber would co_await accept/recv here
  }
  s->exited->store(true, std::memory_order_release);  // observed shutdown -> wind down
}

}  // namespace

// A long-lived fiber never finishes on its own; begin_shutdown() makes it observe
// the signal, break, and finish -- so drain_for() completes instead of hanging.
TEST(Shutdown, LongLivedFiberWindsDownOnSignal) {
  auto exited = std::make_unique<std::atomic<bool>>(false);
  auto iters = std::make_unique<std::atomic<int>>(0);
  LongLived body{exited.get(), iters.get()};
  Fiber f(kDefaultStackSize, &longlived_entry, &body);

  Scheduler sched(2, system_allocator(), 256);
  EXPECT_TRUE(sched.spawn(&f));

  // Let it spin a bit (prove it's genuinely long-lived: it would never finish).
  while (iters->load(std::memory_order_relaxed) < 5) {}
  EXPECT_FALSE(exited->load());  // still looping -- a plain drain() would hang here

  sched.begin_shutdown();                       // phase 1: broadcast the signal
  EXPECT_TRUE(sched.drain_for(2'000'000'000));  // phase 2: it winds down within 2s
  sched.shutdown();                             // stop + join workers

  EXPECT_TRUE(exited->load());  // the fiber observed shutdown and exited its loop
}

// After begin_shutdown(), new spawns are rejected (not admitted, not counted) so a
// producer can't keep feeding work into a draining scheduler.
TEST(Shutdown, SpawnRejectedAfterBeginShutdown) {
  Scheduler sched(2, system_allocator(), 256);
  EXPECT_TRUE(sched.accepting());

  sched.begin_shutdown();
  EXPECT_FALSE(sched.accepting());

  // This spawn must be a no-op: the fiber never runs. If it were admitted, ran
  // would flip and the count would be non-zero (drain_for would time out).
  std::atomic<bool> ran{false};
  struct B { std::atomic<bool>* ran; };
  B b{&ran};
  auto entry = [](void* p) { static_cast<B*>(p)->ran->store(true); };
  Fiber f(kDefaultStackSize, entry, &b);
  EXPECT_FALSE(sched.spawn(&f));

  EXPECT_TRUE(sched.drain_for(500'000'000));  // nothing admitted -> drains at once
  sched.shutdown();
  EXPECT_FALSE(ran.load());  // rejected spawn never executed
}

TEST(Shutdown, SuspendedWaiterIsBroadcastWithoutPolling) {
  struct Waiter {
    std::atomic<bool> entered{false};
    std::atomic<bool> exited{false};
  } state;
  auto entry = [](void* raw) {
    auto* waiter = static_cast<Waiter*>(raw);
    waiter->entered.store(true, std::memory_order_release);
    wait_for_shutdown();
    waiter->exited.store(true, std::memory_order_release);
  };
  Fiber fiber(kSmallStackSize, entry, &state);
  Scheduler sched(2, system_allocator(), 64);
  ASSERT_TRUE(sched.spawn(&fiber));
  while (!state.entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  sched.begin_shutdown();
  EXPECT_TRUE(sched.drain_for(2'000'000'000));
  sched.shutdown();
  EXPECT_TRUE(state.exited.load(std::memory_order_acquire));
}

namespace {
struct ChildSpawn {
  Scheduler* scheduler;
  Fiber* child;
  std::atomic<bool>* child_admitted;
  std::atomic<bool>* ready;
  std::atomic<bool>* go;
};
void child_entry(void* p) {
  static_cast<std::atomic<bool>*>(p)->store(true, std::memory_order_release);
}
void parent_entry(void* p) {
  auto* arg = static_cast<ChildSpawn*>(p);
  arg->ready->store(true, std::memory_order_release);
  while (!arg->go->load(std::memory_order_acquire)) Fiber::yield();
  arg->child_admitted->store(arg->scheduler->spawn(arg->child),
                             std::memory_order_release);
}
}  // namespace

TEST(Shutdown, OrdinaryShutdownDrainsDescendantSpawns) {
  std::atomic<bool> child_ran{false};
  std::atomic<bool> child_admitted{false};
  std::atomic<bool> parent_ready{false};
  std::atomic<bool> parent_go{false};
  Fiber child(kSmallStackSize, &child_entry, &child_ran);
  Scheduler sched(1, system_allocator(), 64);
  ChildSpawn arg{&sched, &child, &child_admitted, &parent_ready, &parent_go};
  Fiber parent(kSmallStackSize, &parent_entry, &arg);
  EXPECT_TRUE(sched.spawn(&parent));
  while (!parent_ready.load(std::memory_order_acquire))
    std::this_thread::yield();
  std::thread closer([&] { sched.shutdown(); });
  // shutdown is now blocked draining the live parent. Admission must remain open
  // for that parent's structured descendant.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  parent_go.store(true, std::memory_order_release);
  closer.join();
  EXPECT_TRUE(child_admitted.load(std::memory_order_acquire));
  EXPECT_TRUE(child_ran.load(std::memory_order_acquire));
  EXPECT_FALSE(sched.accepting());
}

TEST(Shutdown, ConcurrentCallsJoinWorkersExactlyOnce) {
  std::atomic<bool> running{true};
  std::atomic<bool> entered{false};
  std::atomic<int> returned{0};
  struct Context {
    std::atomic<bool>* running;
    std::atomic<bool>* entered;
  } context{&running, &entered};
  Fiber fiber(kSmallStackSize,
              +[](void* raw) {
                auto* context = static_cast<Context*>(raw);
                context->entered->store(true, std::memory_order_release);
                while (context->running->load(std::memory_order_acquire))
                  Fiber::yield();
              },
              &context);
  Scheduler sched(1, system_allocator(), 16);
  ASSERT_TRUE(sched.spawn(&fiber));
  while (!entered.load(std::memory_order_acquire))
    std::this_thread::yield();

  std::thread first([&] {
    sched.shutdown();
    returned.fetch_add(1, std::memory_order_release);
  });
  std::thread second([&] {
    sched.shutdown();
    returned.fetch_add(1, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(returned.load(std::memory_order_acquire), 0);
  running.store(false, std::memory_order_release);
  first.join();
  second.join();
  EXPECT_EQ(returned.load(std::memory_order_acquire), 2);
}

// Red-team regression for admission accounting on both task legs. Every
// stackful spawn and worker-stack callback racing close is linearized: accepted
// work is counted and completes; rejected work is never published. No accepted
// task may be stranded after drain reports zero.
TEST(Shutdown, ConcurrentAdmissionAndCloseAreLinearizable) {
  constexpr int kFibers = 128;
  constexpr int kInlineJobs = 256;
  constexpr int kProducers = 4;
  for (int round = 0; round < 20; ++round) {
    {
      Scheduler sched(4, system_allocator(), 256);
      std::atomic<int> ran{0};
      std::atomic<int> accepted{0};
      std::atomic<bool> go{false};
      struct Arg {
        std::atomic<int>* ran;
      };
      std::vector<Arg> args(kFibers, Arg{&ran});
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kFibers);
      auto entry = [](void* p) {
        static_cast<Arg*>(p)->ran->fetch_add(1, std::memory_order_relaxed);
      };
      for (int i = 0; i < kFibers; ++i)
        fibers.push_back(
            std::make_unique<Fiber>(kSmallStackSize, entry, &args[i]));

      std::vector<std::thread> producers;
      for (int producer = 0; producer < kProducers; ++producer) {
        producers.emplace_back([&, producer] {
          while (!go.load(std::memory_order_acquire))
            std::this_thread::yield();
          for (int i = producer; i < kFibers; i += kProducers) {
            if (sched.spawn(fibers[i].get()))
              accepted.fetch_add(1, std::memory_order_relaxed);
          }
        });
      }
      go.store(true, std::memory_order_release);
      sched.begin_shutdown();
      for (auto& producer : producers) producer.join();
      EXPECT_TRUE(sched.drain_for(2'000'000'000));
      EXPECT_EQ(ran.load(std::memory_order_relaxed),
                accepted.load(std::memory_order_relaxed));
      sched.shutdown();
    }

    {
      Scheduler sched(4, system_allocator(), 256);
      std::atomic<int> ran{0};
      std::atomic<int> accepted{0};
      std::atomic<bool> go{false};
      auto entry = +[](void* p) {
        static_cast<std::atomic<int>*>(p)->fetch_add(
            1, std::memory_order_relaxed);
      };

      std::vector<std::thread> producers;
      for (int producer = 0; producer < kProducers; ++producer) {
        producers.emplace_back([&] {
          while (!go.load(std::memory_order_acquire))
            std::this_thread::yield();
          for (int i = 0; i < kInlineJobs / kProducers; ++i) {
            const auto status = sched.schedule(entry, &ran);
            if (status.ok())
              accepted.fetch_add(1, std::memory_order_relaxed);
            else
              EXPECT_EQ(status.code(), hyle::StatusCode::Cancelled);
          }
        });
      }
      go.store(true, std::memory_order_release);
      sched.begin_shutdown();
      for (auto& producer : producers) producer.join();
      EXPECT_TRUE(sched.drain_for(2'000'000'000));
      EXPECT_EQ(ran.load(std::memory_order_relaxed),
                accepted.load(std::memory_order_relaxed));
      sched.shutdown();
    }
  }
}

// drain_for returns false when work does NOT finish within the timeout (a fiber
// that ignores the shutdown signal). Proves the bounded-wait semantics -- the
// caller then escalates (cooperative cancel / abandon), never force-kills a
// stackful fiber.
TEST(Shutdown, DrainForTimesOutOnStuckFiber) {
  // A fiber that ignores shutdown_requested() and loops forever (yielding).
  auto keep_going = std::make_unique<std::atomic<bool>>(true);
  struct Stuck { std::atomic<bool>* go; };
  Stuck s{keep_going.get()};
  auto entry = [](void* p) {
    auto* st = static_cast<Stuck*>(p);
    while (st->go->load(std::memory_order_acquire)) Fiber::yield();
  };
  Fiber f(kDefaultStackSize, entry, &s);

  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(&f);
  sched.begin_shutdown();
  // The fiber ignores the signal, so drain can't complete -> timeout returns false.
  EXPECT_FALSE(sched.drain_for(100'000'000));  // 100ms budget, fiber won't exit

  // Release it so teardown can finish (real code would abandon it at process exit).
  keep_going->store(false, std::memory_order_release);
  sched.shutdown();
}
