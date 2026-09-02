#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
#endif

#include "hyle/mem/allocator.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::SchedulerIdlePolicy;
using hyle::sched::SchedulerOptions;
using hyle::sched::Waker;
using hyle::platform::CpuOrder;

namespace {
void* fail_alloc(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void*, void*, std::size_t, std::size_t) noexcept {}
void* fail_realloc(void*, void*, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}

struct WorkerHookProbe {
  std::atomic<unsigned> starts[4]{};
  std::atomic<unsigned> stops[4]{};
};

void worker_start_hook(void* raw, unsigned worker) noexcept {
  auto* probe = static_cast<WorkerHookProbe*>(raw);
  if (worker < 4)
    probe->starts[worker].fetch_add(1, std::memory_order_relaxed);
}

void worker_stop_hook(void* raw, unsigned worker) noexcept {
  auto* probe = static_cast<WorkerHookProbe*>(raw);
  if (worker < 4)
    probe->stops[worker].fetch_add(1, std::memory_order_relaxed);
}

// A fiber body that yields `yields` times, bumping a shared atomic each segment.
struct Body {
  int yields;
  std::atomic<int>* ran;       // total resume segments observed (sanity)
  std::atomic<int>* completed;  // fibers that reached the end
};
void body_entry(void* p) {
  auto* b = static_cast<Body*>(p);
  for (int i = 0; i < b->yields; ++i) {
    b->ran->fetch_add(1, std::memory_order_relaxed);
    Fiber::yield();
  }
  b->completed->fetch_add(1, std::memory_order_acq_rel);
}

#if defined(_WIN32) || defined(__linux__)
void observe_current_cpu(void* raw) {
  unsigned cpu = ~0u;
#  if defined(_WIN32)
  PROCESSOR_NUMBER number{};
  ::GetCurrentProcessorNumberEx(&number);
  cpu = number.Number;
  for (WORD group = 0; group < number.Group; ++group) {
    cpu += ::GetActiveProcessorCount(group);
  }
#  else
  const int current = ::sched_getcpu();
  if (current >= 0) cpu = static_cast<unsigned>(current);
#  endif
  static_cast<std::atomic<unsigned>*>(raw)->store(cpu,
                                                  std::memory_order_release);
}
#endif
}  // namespace

TEST(Scheduler, RecoverableConstructionReportsQueueOom) {
  hyle::Status status;
  hyle::mem::AllocatorRef failing(&fail_alloc, &fail_free, &fail_realloc, nullptr);
  Scheduler sched(2, failing, 64, &status);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.IsResourceExhausted());
  EXPECT_FALSE(sched.valid());
}

TEST(Scheduler, RecoverableConstructionRejectsInvalidShape) {
  hyle::Status status;
  Scheduler no_allocator(2, hyle::mem::AllocatorRef{}, 64, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(no_allocator.valid());

  Scheduler no_workers(0, system_allocator(), 64, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(no_workers.valid());

  Scheduler non_power_of_two(2, system_allocator(), 63, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(non_power_of_two.valid());

  Scheduler too_small(2, system_allocator(), 1, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(too_small.valid());

  Scheduler too_many_workers(1025, system_allocator(), 64, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(too_many_workers.valid());

  Scheduler too_large_queue(
      2, system_allocator(), std::size_t{1} << 32, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(too_large_queue.valid());

  SchedulerOptions invalid_options;
  invalid_options.worker_cpu_order = static_cast<CpuOrder>(255);
  Scheduler invalid_order(2, system_allocator(), 64, invalid_options, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(invalid_order.valid());

  unsigned cpu = 0;
  invalid_options.worker_cpu_set = std::span<const unsigned>(&cpu, 1);
  Scheduler conflicting_placement(2, system_allocator(), 64,
                                  invalid_options, &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(conflicting_placement.valid());

  invalid_options = {};
  invalid_options.worker_priority =
      static_cast<hyle::platform::ThreadPriority>(127);
  Scheduler invalid_priority(1, system_allocator(), 64, invalid_options,
                             &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(invalid_priority.valid());

  invalid_options = {};
  invalid_options.worker_config_policy =
      static_cast<hyle::platform::ThreadConfigPolicy>(255);
  Scheduler invalid_config_policy(1, system_allocator(), 64, invalid_options,
                                  &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(invalid_config_policy.valid());

  invalid_options = {};
  invalid_options.idle_policy = static_cast<SchedulerIdlePolicy>(255);
  Scheduler invalid_idle_policy(1, system_allocator(), 64, invalid_options,
                                &status);
  EXPECT_TRUE(status.IsInvalidArgument());
  EXPECT_FALSE(invalid_idle_policy.valid());
}

TEST(Scheduler, WorkerCpuOrderIsAnExplicitConstructionPreference) {
  SchedulerOptions options;
  options.worker_cpu_order = CpuOrder::kPerformanceFirst;
  options.worker_priority = hyle::platform::ThreadPriority::kHigh;
  hyle::Status status;
  Scheduler scheduler(1, system_allocator(), 64, options, &status);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(scheduler.valid());
}

TEST(Scheduler, PerformanceFirstPinsFirstWorkerToPreferredCpu) {
#if !defined(_WIN32) && !defined(__linux__)
  GTEST_SKIP() << "platform has no hard thread affinity";
#else
  unsigned expected = ~0u;
  auto available = hyle::platform::available_cpu_ids(
      CpuOrder::kPerformanceFirst, &expected, 1);
  ASSERT_TRUE(available.has_value()) << available.error().message();
  ASSERT_NE(expected, ~0u);

  std::atomic<unsigned> observed{~0u};
  SchedulerOptions options;
  options.worker_cpu_order = CpuOrder::kPerformanceFirst;
  Scheduler scheduler(1, system_allocator(), 64, options);
  ASSERT_TRUE(scheduler.schedule(&observe_current_cpu, &observed).ok());
  scheduler.drain();
  EXPECT_EQ(observed.load(std::memory_order_acquire), expected);
#endif
}

TEST(Scheduler, RequiredInvalidCpuPlacementIsRecoverable) {
  const unsigned invalid_cpu = ~0u;
  SchedulerOptions options;
  options.worker_cpu_set = std::span<const unsigned>(&invalid_cpu, 1);
  options.worker_config_policy =
      hyle::platform::ThreadConfigPolicy::kRequired;
  hyle::Status status;
  Scheduler scheduler(1, system_allocator(), 64, options, &status);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(scheduler.valid());
}

TEST(Scheduler, IdlePoliciesAndWorkerLifecycleHooksAreOperational) {
  for (SchedulerIdlePolicy policy : {SchedulerIdlePolicy::kBalanced,
                                     SchedulerIdlePolicy::kLowLatency,
                                     SchedulerIdlePolicy::kPowerSaving}) {
    WorkerHookProbe probe;
    SchedulerOptions options;
    options.idle_policy = policy;
    options.on_worker_start = &worker_start_hook;
    options.on_worker_stop = &worker_stop_hook;
    options.worker_hook_context = &probe;
    std::atomic<unsigned> ran{0};
    {
      Scheduler scheduler(4, system_allocator(), 64, options);
      ASSERT_TRUE(scheduler.schedule(
          +[](void* raw) noexcept {
            static_cast<std::atomic<unsigned>*>(raw)->fetch_add(
                1, std::memory_order_relaxed);
          },
          &ran).ok());
      scheduler.shutdown();
    }
    EXPECT_EQ(ran.load(std::memory_order_relaxed), 1u);
    for (unsigned worker = 0; worker < 4; ++worker) {
      EXPECT_EQ(probe.starts[worker].load(std::memory_order_relaxed), 1u);
      EXPECT_EQ(probe.stops[worker].load(std::memory_order_relaxed), 1u);
    }
  }
}

TEST(Scheduler, DetachedCallbacksOwnTheirLifetimeAndReportAdmission) {
  std::atomic<int> ran{0};
  auto callback = +[](void* raw) {
    static_cast<std::atomic<int>*>(raw)->fetch_add(
        1, std::memory_order_relaxed);
  };

  Scheduler sched(2, system_allocator(), 64);
  EXPECT_TRUE(sched.spawn_detached(callback, &ran).ok());
  EXPECT_TRUE(
      sched.spawn_detached_on(1, callback, &ran, kDefaultStackSize).ok());
  for (int i = 0; i < 100; ++i)
    EXPECT_TRUE(sched.schedule(callback, &ran).ok());
  sched.drain();
  EXPECT_EQ(ran.load(std::memory_order_relaxed), 102);

  EXPECT_TRUE(sched.schedule(nullptr, nullptr).IsInvalidArgument());
  EXPECT_TRUE(
      sched.spawn_detached(nullptr, nullptr, kDefaultStackSize)
          .IsInvalidArgument());
  EXPECT_TRUE(
      sched.spawn_detached_on(2, callback, &ran, kDefaultStackSize)
          .IsInvalidArgument());
  sched.begin_shutdown();
  EXPECT_TRUE(
      sched.spawn_detached(callback, &ran, kDefaultStackSize).IsCancelled());
  EXPECT_TRUE(sched.schedule(callback, &ran).IsCancelled());
  sched.shutdown();
}

TEST(Scheduler, OwningCallableConveniencesSupportMoveOnlyCaptures) {
  Scheduler scheduler(2, system_allocator(), 64);
  std::atomic<int> result{0};

  auto inline_value = std::make_unique<int>(7);
  ASSERT_TRUE(scheduler.schedule(
      [value = std::move(inline_value), &result]() noexcept {
        result.fetch_add(*value, std::memory_order_relaxed);
      }).ok());
  EXPECT_EQ(inline_value, nullptr);

  auto detached_value = std::make_unique<int>(11);
  ASSERT_TRUE(scheduler.spawn_detached(
      [value = std::move(detached_value), &result]() noexcept {
        Fiber::yield();
        result.fetch_add(*value, std::memory_order_relaxed);
      }).ok());
  EXPECT_EQ(detached_value, nullptr);

  std::atomic<int> observed_worker{-1};
  ASSERT_TRUE(scheduler.spawn_detached_on(1, [&]() noexcept {
    observed_worker.store(hyle::sched::current_worker_index(),
                          std::memory_order_release);
    result.fetch_add(13, std::memory_order_relaxed);
  }).ok());

  scheduler.drain();
  EXPECT_EQ(result.load(std::memory_order_relaxed), 31);
  EXPECT_EQ(observed_worker.load(std::memory_order_acquire), 1);
}

TEST(Scheduler, InlineScheduleMultiProducerConservesEveryTask) {
  constexpr int kProducers = 8;
  constexpr int kPerProducer = 1000;
  std::atomic<int> ran{0};
  auto callback = +[](void* raw) {
    static_cast<std::atomic<int>*>(raw)->fetch_add(
        1, std::memory_order_relaxed);
  };

  Scheduler sched(4, system_allocator(), 64);
  std::vector<std::thread> producers;
  producers.reserve(kProducers);
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&] {
      for (int i = 0; i < kPerProducer; ++i)
        ASSERT_TRUE(sched.schedule(callback, &ran).ok());
    });
  }
  for (auto& producer : producers) producer.join();
  sched.drain();
  EXPECT_EQ(ran.load(std::memory_order_relaxed), kProducers * kPerProducer);
}

TEST(Scheduler, RecursiveInlineSaturationReturnsInsteadOfSelfDeadlocking) {
  struct Context {
    Scheduler* scheduler;
    std::atomic<int> ran{0};
    std::atomic<int> accepted{0};
    std::atomic<int> full{0};
  };
  auto parent = +[](void* raw) {
    auto* context = static_cast<Context*>(raw);
    context->ran.fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < 8; ++i) {
      const hyle::Status status = context->scheduler->schedule(
          +[](void* leaf_raw) {
            static_cast<Context*>(leaf_raw)->ran.fetch_add(
                1, std::memory_order_relaxed);
          },
          context);
      if (status.ok())
        context->accepted.fetch_add(1, std::memory_order_relaxed);
      else if (status.IsResourceExhausted())
        context->full.fetch_add(1, std::memory_order_relaxed);
    }
  };

  Scheduler sched(1, system_allocator(), 2);
  Context context{&sched};
  ASSERT_TRUE(sched.schedule(parent, &context).ok());
  sched.drain();
  EXPECT_EQ(context.ran.load(std::memory_order_relaxed),
            1 + context.accepted.load(std::memory_order_relaxed));
  EXPECT_EQ(context.accepted.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(context.full.load(std::memory_order_relaxed), 6);
}

TEST(Scheduler, RecursiveFiberSaturationReturnsInsteadOfSelfDeadlocking) {
  struct Context {
    Scheduler* scheduler;
    std::atomic<int> ran{0};
    std::atomic<int> accepted{0};
    std::atomic<int> full{0};
    bool pinned = false;
  };
  auto parent = +[](void* raw) {
    auto* context = static_cast<Context*>(raw);
    context->ran.fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < 8; ++i) {
      const hyle::Status status =
          context->pinned
              ? context->scheduler->spawn_detached_on(
                    0,
                    +[](void* leaf_raw) {
                      static_cast<Context*>(leaf_raw)->ran.fetch_add(
                          1, std::memory_order_relaxed);
                    },
                    context, kDefaultStackSize)
              : context->scheduler->spawn_detached(
                    +[](void* leaf_raw) {
                      static_cast<Context*>(leaf_raw)->ran.fetch_add(
                          1, std::memory_order_relaxed);
                    },
                    context, kDefaultStackSize);
      if (status.ok())
        context->accepted.fetch_add(1, std::memory_order_relaxed);
      else if (status.IsResourceExhausted())
        context->full.fetch_add(1, std::memory_order_relaxed);
    }
  };

  {
    Scheduler sched(1, system_allocator(), 2);
    Context context{&sched};
    ASSERT_TRUE(
        sched.spawn_detached(parent, &context, kDefaultStackSize).ok());
    sched.drain();
    EXPECT_EQ(context.ran.load(std::memory_order_relaxed),
              1 + context.accepted.load(std::memory_order_relaxed));
    EXPECT_EQ(context.accepted.load(std::memory_order_relaxed), 4);
    EXPECT_EQ(context.full.load(std::memory_order_relaxed), 4);
  }
  {
    Scheduler sched(1, system_allocator(), 2);
    Context context{&sched};
    context.pinned = true;
    ASSERT_TRUE(
        sched.spawn_detached_on(0, parent, &context, kDefaultStackSize).ok());
    sched.drain();
    EXPECT_EQ(context.ran.load(std::memory_order_relaxed),
              1 + context.accepted.load(std::memory_order_relaxed));
    EXPECT_EQ(context.accepted.load(std::memory_order_relaxed), 2);
    EXPECT_EQ(context.full.load(std::memory_order_relaxed), 6);
  }
}

TEST(Scheduler, WakeFloodUsesOverflowInsteadOfDeadlockingConsumer) {
  constexpr int kWaiters = 24;
  struct Context {
    std::vector<Waker> wakers;
    std::atomic<int> next_slot{0};
    std::atomic<int> ready{0};
    std::atomic<int> completed{0};
  };
  auto waiter = +[](void* raw) {
    auto* context = static_cast<Context*>(raw);
    const int slot =
        context->next_slot.fetch_add(1, std::memory_order_relaxed);
    context->wakers[slot] = hyle::sched::current_waker();
    context->ready.fetch_add(1, std::memory_order_release);
    hyle::sched::suspend();
    context->completed.fetch_add(1, std::memory_order_release);
  };
  auto notifier = +[](void* raw) {
    auto* context = static_cast<Context*>(raw);
    for (const Waker& waker : context->wakers) waker.wake();
  };

  for (bool pinned : {false, true}) {
    Scheduler sched(1, system_allocator(), 2);
    Context context;
    context.wakers.resize(kWaiters);
    std::vector<std::unique_ptr<Fiber>> fibers;
    fibers.reserve(kWaiters);
    for (int i = 0; i < kWaiters; ++i) {
      fibers.push_back(
          std::make_unique<Fiber>(kDefaultStackSize, waiter, &context));
      ASSERT_TRUE(pinned ? sched.spawn_on(0, fibers.back().get())
                         : sched.spawn(fibers.back().get()));
    }
    while (context.ready.load(std::memory_order_acquire) != kWaiters)
      std::this_thread::yield();
    ASSERT_TRUE((pinned ? sched.spawn_detached_on(
                              0, notifier, &context, kDefaultStackSize)
                        : sched.spawn_detached(
                              notifier, &context, kDefaultStackSize))
                    .ok());
    sched.drain();
    EXPECT_EQ(context.completed.load(std::memory_order_acquire), kWaiters);
  }
}

TEST(Scheduler, ManyFibersAllComplete) {
  constexpr int kFibers = 1000;
  constexpr int kYields = 3;
  std::atomic<int> ran{0};
  std::atomic<int> completed{0};

  std::vector<Body> bodies(kFibers, Body{kYields, &ran, &completed});
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kFibers);

  {
    Scheduler sched(8, system_allocator(), 1024);
    for (int i = 0; i < kFibers; ++i) {
      fibers.push_back(std::make_unique<Fiber>(
                                               kDefaultStackSize, &body_entry,
                                               &bodies[i]));
      sched.spawn(fibers[i].get());
    }
    sched.shutdown();  // drains (waits all done) then joins workers
  }

  EXPECT_EQ(completed.load(), kFibers);
  EXPECT_EQ(ran.load(), kFibers * kYields);  // every segment ran exactly once
}

TEST(Scheduler, ShutdownReclaimsFiberStackCaches) {
  Scheduler sched(1, system_allocator(), 64);
  auto* pool = sched.fiber_stack_pool(hyle::mem::kFiberStackDefault);
  ASSERT_NE(pool, nullptr);
  hyle::mem::FiberStack stack = pool->acquire().value();
  pool->release(std::move(stack));
  ASSERT_EQ(pool->cached(), 1u);

  sched.shutdown();
  EXPECT_EQ(pool->cached(), 0u);
}

// Force stealing: a single "seed" fiber, running on one worker, spawns many
// children onto THAT worker's local queue. The other workers have nothing of
// their own and must steal to make progress -> they will, and everything still
// completes exactly once.
namespace {
struct SeedArgs {
  Scheduler* sched;
  int children;
  std::vector<std::unique_ptr<Fiber>>* pool;
  std::vector<Body>* bodies;
  std::atomic<int>* ran;
  std::atomic<int>* completed;
};
void seed_entry(void* p) {
  auto* a = static_cast<SeedArgs*>(p);
  for (int i = 0; i < a->children; ++i) {
    (*a->bodies)[i] = Body{2, a->ran, a->completed};
    // child fibers are pre-created in the pool; spawn from inside the seed
    // fiber so they land on this worker's LOCAL queue (forcing others to steal).
    a->sched->spawn((*a->pool)[i].get());
  }
  a->completed->fetch_add(1, std::memory_order_acq_rel);  // seed itself
}
}  // namespace

TEST(Scheduler, StealingDistributesLocalWork) {
  constexpr int kChildren = 500;
  std::atomic<int> ran{0};
  std::atomic<int> completed{0};

  std::vector<Body> bodies(kChildren);
  std::vector<std::unique_ptr<Fiber>> pool;
  pool.reserve(kChildren);

  {
    Scheduler sched(4, system_allocator(), 1024);
    for (int i = 0; i < kChildren; ++i) {
      pool.push_back(std::make_unique<Fiber>(
          kDefaultStackSize, &body_entry, &bodies[i]));
    }
    SeedArgs sa{&sched, kChildren, &pool, &bodies, &ran, &completed};
    Fiber seed(kDefaultStackSize, &seed_entry, &sa);
    sched.spawn(&seed);
    // Keep admission open while the seed fans out its children. Direct
    // shutdown() closes admission first; drain() is the "finish this generation,
    // including descendants it admits" boundary.
    sched.drain();
    sched.shutdown();
  }

  EXPECT_EQ(completed.load(), kChildren + 1);  // children + seed
  EXPECT_EQ(ran.load(), kChildren * 2);
}

TEST(Scheduler, SingleWorkerStillWorks) {
  std::atomic<int> ran{0};
  std::atomic<int> completed{0};
  Body b{5, &ran, &completed};
  auto f = std::make_unique<Fiber>(kDefaultStackSize,
                                   &body_entry, &b);
  {
    Scheduler sched(1, system_allocator(), 64);
    sched.spawn(f.get());
    sched.shutdown();
  }
  EXPECT_EQ(completed.load(), 1);
  EXPECT_EQ(ran.load(), 5);
}

// No lost wakeup (exercises park): spawn fibers one at a time with a gap so the
// workers actually go to sleep between submissions. Every spawn must wake a
// worker to take it -- a lost wakeup would strand a fiber and hang drain()
// (ctest timeout = failure). Each fiber just runs to completion (no yield).
TEST(Scheduler, NoLostWakeupAcrossParks) {
  constexpr int kFibers = 40;
  std::atomic<int> ran{0};
  std::atomic<int> completed{0};
  std::vector<Body> bodies(kFibers, Body{0, &ran, &completed});
  std::vector<std::unique_ptr<Fiber>> fibers;
  fibers.reserve(kFibers);

  {
    Scheduler sched(8, system_allocator(), 64);
    for (int i = 0; i < kFibers; ++i) {
      fibers.push_back(std::make_unique<Fiber>(
          kDefaultStackSize, &body_entry, &bodies[i]));
      sched.spawn(fibers[i].get());
      // Let the pool settle into parked state before the next spawn.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    sched.shutdown();  // drains; hangs here if any wakeup was lost
  }
  EXPECT_EQ(completed.load(), kFibers);
}

// Global-queue interval / anti-starvation (complexity opportunity ③). A fiber that
// yields in a loop keeps its worker's LOCAL ring perpetually non-empty; find_task
// checks local before global, so without a periodic global-queue check a single
// busy worker starves any task injected to the GLOBAL queue forever. Probe-
// confirmed before the fix (a 1-worker yield-hog did 17M yields while a globally-
// injected victim never ran in 500ms); this guards the regression. 1 worker so no
// other worker can steal the victim -- it MUST come through the interval check.
namespace {
struct YieldHog {
  std::atomic<bool>* stop;
};
void yield_hog(void* p) {
  auto* h = static_cast<YieldHog*>(p);
  while (!h->stop->load(std::memory_order_acquire)) Fiber::yield();
}
}  // namespace
TEST(Scheduler, GlobalQueueNotStarvedByLocalFirehose) {
  std::atomic<bool> stop{false};
  YieldHog hog{&stop};
  auto fh = std::make_unique<Fiber>(kDefaultStackSize,
                                    &yield_hog, &hog);
  std::atomic<int> vran{0};
  std::atomic<int> vdone{0};
  Body victim{0, &vran, &vdone};  // 0 yields -> completes in one run
  auto fv = std::make_unique<Fiber>(kDefaultStackSize,
                                    &body_entry, &victim);
  bool ran_while_hogging = false;
  {
    Scheduler sched(1, system_allocator(), 1024);
    sched.spawn(fh.get());  // once running the hog yields onto its LOCAL ring and
                            // firehoses it, so find_task keeps serving local.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let hog take over
    sched.spawn(fv.get());  // foreign-thread spawn -> GLOBAL inject shard
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {
      if (vdone.load(std::memory_order_acquire) > 0) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // CAPTURE the verdict BEFORE releasing the hog. Critical: if we only checked
    // after shutdown(), drain() would run the (starved) victim during teardown once
    // the hog stops, flipping the flag to 1 and MASKING the starvation -- the exact
    // "test setup hides the bug" trap. The honest signal is "did it run WHILE the
    // hog was still firehosing the local ring?".
    ran_while_hogging = vdone.load(std::memory_order_acquire) > 0;
    stop.store(true, std::memory_order_release);  // let the hog exit so drain finishes
    sched.shutdown();
  }
  EXPECT_TRUE(ran_while_hogging)
      << "victim in global queue was starved by local firehose (never ran while the "
         "hog kept the local ring full)";
}
