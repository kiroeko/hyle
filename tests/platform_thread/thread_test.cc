#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE 1  // cpu_set_t / CPU_ISSET / sched_getaffinity
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#if !defined(_WIN32)
#  include <pthread.h>  // pthread_getname_np (Linux + macOS)
#  include <sched.h>    // sched_getaffinity + cpu_set_t (Linux)
#else
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include "hyle/platform/cpu.h"
#include "hyle/platform/thread.h"

using hyle::platform::Thread;
using hyle::platform::ThreadConfigPolicy;
using hyle::platform::ThreadOpts;
using hyle::platform::ThreadPriority;
using hyle::platform::CpuAffinity;
namespace this_thread = hyle::platform::this_thread;

TEST(Thread, SpawnRunsAndJoins) {
  std::atomic<int> ran{0};
  auto r = Thread::spawn([](void* a) { static_cast<std::atomic<int>*>(a)->store(
                             42, std::memory_order_release); },
                         &ran);
  ASSERT_TRUE(r.has_value());
  Thread t = std::move(*r);
  EXPECT_TRUE(t.joinable());
  t.join();
  EXPECT_FALSE(t.joinable());
  EXPECT_EQ(ran.load(std::memory_order_acquire), 42);
}

TEST(Thread, SpawnWithNameAndAffinity) {
  // name + affinity are best-effort; the contract here is "applies without
  // crashing and the thread still runs". (Value verification needs OS probing.)
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.name = "hyle-worker";
  unsigned cpu = 0;
  auto available = hyle::platform::available_cpu_ids(&cpu, 1);
  if (available && *available != 0) opts.affinity = cpu;
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
      opts);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_TRUE(ran.load());
}

#if defined(_WIN32)
namespace {
struct PriorityProbe {
  BOOL queried = FALSE;
  DWORD control = 0;
  DWORD state = 0;
  int native_priority = THREAD_PRIORITY_ERROR_RETURN;
};

void priority_probe_entry(void* arg) {
  auto* probe = static_cast<PriorityProbe*>(arg);
  THREAD_POWER_THROTTLING_STATE qos{};
  qos.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
  probe->queried = ::GetThreadInformation(
      ::GetCurrentThread(), ThreadPowerThrottling, &qos, sizeof(qos));
  probe->control = qos.ControlMask;
  probe->state = qos.StateMask;
  probe->native_priority = ::GetThreadPriority(::GetCurrentThread());
}
}  // namespace

TEST(Thread, ExplicitPriorityConfiguresWindowsExecutionQos) {
  struct Case {
    ThreadPriority priority;
    int native_priority;
    bool throttled;
    bool system_managed;
  };
  constexpr Case cases[] = {
      {ThreadPriority::kLow, THREAD_PRIORITY_BELOW_NORMAL, true, false},
      {ThreadPriority::kNormal, THREAD_PRIORITY_NORMAL, false, true},
      {ThreadPriority::kHigh, THREAD_PRIORITY_ABOVE_NORMAL, false, false},
  };
  for (const Case& test : cases) {
    PriorityProbe probe;
    ThreadOpts opts;
    opts.priority = test.priority;
    opts.config_policy = ThreadConfigPolicy::kRequired;
    auto thread = Thread::spawn(&priority_probe_entry, &probe, opts);
    ASSERT_TRUE(thread.has_value()) << thread.error().message();
    std::move(*thread).join();
    ASSERT_TRUE(probe.queried);
    EXPECT_EQ(probe.native_priority, test.native_priority);
    const bool controls_speed =
        (probe.control & THREAD_POWER_THROTTLING_EXECUTION_SPEED) != 0;
    const bool throttles_speed =
        (probe.state & THREAD_POWER_THROTTLING_EXECUTION_SPEED) != 0;
    EXPECT_EQ(controls_speed, !test.system_managed);
    EXPECT_EQ(throttles_speed, test.throttled);
  }
}
#endif

#if !defined(_WIN32)
// Verify name/affinity actually TAKE EFFECT (not just "spawn didn't crash"), by
// reading them back from INSIDE the spawned thread. POSIX only: Linux + macOS
// both have pthread_getname_np; affinity read-back is Linux-only (sched_getaffinity).
namespace {
struct ProbeResult {
  char name[64] = {0};
  int name_rc = -1;
#if defined(__linux__)
  unsigned cpu = 0;
  int affinity_ok = -1;  // 1 if the pinned CPU is set in the running mask
#endif
};
void probe_entry(void* arg) {
  auto* r = static_cast<ProbeResult*>(arg);
  r->name_rc = ::pthread_getname_np(::pthread_self(), r->name, sizeof(r->name));
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (r->cpu < CPU_SETSIZE &&
      ::sched_getaffinity(0, sizeof(set), &set) == 0) {
    r->affinity_ok = CPU_ISSET(r->cpu, &set) ? 1 : 0;
  }
#endif
}
}  // namespace

TEST(Thread, NameTakesEffect) {
  ProbeResult res;
  ThreadOpts opts;
  opts.name = "hyle-probe";
  unsigned cpu = 0;
  auto available = hyle::platform::available_cpu_ids(&cpu, 1);
  if (available && *available != 0) {
    opts.affinity = cpu;
#if defined(__linux__)
    res.cpu = cpu;
#endif
  }
  auto r = Thread::spawn(&probe_entry, &res, opts);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  ASSERT_EQ(res.name_rc, 0);
  // Linux truncates thread names to 15 chars; "hyle-probe" (10) fits exactly.
  EXPECT_STREQ(res.name, "hyle-probe");
#if defined(__linux__)
  // The actual process-available id (which may be sparse) must be in the mask.
  if (res.affinity_ok != -1) { EXPECT_EQ(res.affinity_ok, 1); }
#endif
}
#endif  // !_WIN32

TEST(Thread, MoveTransfersOwnership) {
  std::atomic<int> ran{0};
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<int>*>(a)->fetch_add(1); }, &ran);
  ASSERT_TRUE(r.has_value());
  Thread a = std::move(*r);
  Thread b = std::move(a);  // a is now empty, b owns the thread
  EXPECT_FALSE(a.joinable());
  EXPECT_TRUE(b.joinable());
  b.join();
  EXPECT_EQ(ran.load(), 1);
}

TEST(Thread, HardwareConcurrencyIsPositive) {
  EXPECT_GE(hyle::platform::hardware_concurrency(), 1u);
}

TEST(ThisThread, IdAndHelpers) {
  EXPECT_NE(this_thread::get_id(), 0u);
  this_thread::yield();             // must not crash
  EXPECT_TRUE(this_thread::set_name("hyle-main").ok());
  EXPECT_FALSE(this_thread::set_affinity(
                   std::numeric_limits<unsigned>::max())
                   .ok());
  EXPECT_TRUE(this_thread::set_priority(
                  static_cast<ThreadPriority>(
                      std::numeric_limits<std::int8_t>::max()))
                  .IsInvalidArgument());
  SUCCEED();
}

TEST(ThisThread, ReportsCurrentAffinityWithoutChangingIt) {
  auto count = this_thread::affinity(nullptr, 0);
#if defined(__APPLE__)
  ASSERT_FALSE(count.has_value());
  EXPECT_TRUE(count.error().IsUnimplemented());
#else
  ASSERT_TRUE(count.has_value()) << count.error().message();
  ASSERT_GT(*count, 0u);
  std::vector<unsigned> cpus(*count);
  auto filled = this_thread::affinity(cpus.data(), cpus.size());
  ASSERT_TRUE(filled.has_value()) << filled.error().message();
  EXPECT_EQ(*filled, cpus.size());
  for (std::size_t i = 1; i < cpus.size(); ++i) {
    EXPECT_LT(cpus[i - 1], cpus[i]);
  }
#endif
}

TEST(ThisThread, ReportsDynamicIsaState) {
  auto state = this_thread::isa_state();
  ASSERT_TRUE(state.has_value()) << state.error().message();
  if (state->scalable_vector_bytes != 0) {
    EXPECT_EQ(state->scalable_vector_bytes % 16u, 0u);
  }
  if (state->streaming_vector_bytes != 0) {
    EXPECT_EQ(state->streaming_vector_bytes % 16u, 0u);
  }
}

TEST(ThisThread, DistinctThreadsHaveDistinctIds) {
  std::atomic<std::uint64_t> child_id{0};
  auto r = Thread::spawn(
      [](void* a) {
        static_cast<std::atomic<std::uint64_t>*>(a)->store(
            this_thread::get_id());
      },
      &child_id);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_NE(child_id.load(), 0u);
  EXPECT_NE(child_id.load(), this_thread::get_id());
}

TEST(Thread, DefaultConstructedNotJoinable) {
  Thread t;
  EXPECT_FALSE(t.joinable());
}

TEST(Thread, CallableOverloadOwnsCapturingLambda) {
  std::atomic<int> value{0};
  auto result = Thread::spawn([&value] {
    value.store(42, std::memory_order_release);
  });
  ASSERT_TRUE(result.has_value()) << result.error().message();
  std::move(*result).join();
  EXPECT_EQ(value.load(std::memory_order_acquire), 42);
}

TEST(Thread, CallableOverloadAcceptsMoveOnlyState) {
  std::atomic<int> value{0};
  auto owned = std::make_unique<int>(73);
  auto result = Thread::spawn(
      [state = std::move(owned), &value] {
        value.store(*state, std::memory_order_release);
      });
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_EQ(owned, nullptr);
  std::move(*result).join();
  EXPECT_EQ(value.load(std::memory_order_acquire), 73);
}

TEST(Thread, CallableIsDestroyedWithoutRunningWhenRequiredSetupFails) {
  struct MoveOnlyProbe {
    std::atomic<int>* runs;
    std::atomic<int>* destroys;
    bool owns = true;

    MoveOnlyProbe(std::atomic<int>* run_count,
                  std::atomic<int>* destroy_count) noexcept
        : runs(run_count), destroys(destroy_count) {}
    MoveOnlyProbe(MoveOnlyProbe&& other) noexcept
        : runs(other.runs), destroys(other.destroys), owns(other.owns) {
      other.owns = false;
    }
    MoveOnlyProbe(const MoveOnlyProbe&) = delete;
    ~MoveOnlyProbe() {
      if (owns) destroys->fetch_add(1, std::memory_order_relaxed);
    }
    void operator()() {
      runs->fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::atomic<int> runs{0};
  std::atomic<int> destroys{0};
  ThreadOpts opts;
  opts.affinity = std::numeric_limits<unsigned>::max();
  opts.config_policy = ThreadConfigPolicy::kRequired;
  MoveOnlyProbe probe(&runs, &destroys);
  auto result = Thread::spawn(std::move(probe), opts);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(runs.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(destroys.load(std::memory_order_relaxed), 1);
}

TEST(Thread, NullEntryIsRejected) {
  auto r = Thread::spawn(nullptr, nullptr);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsInvalidArgument());
}

TEST(Thread, SpawnWithStackSizeRuns) {
  // Contract: stack_size is honored by the OS attrs path and does not prevent
  // the thread from running. We do not probe the actual stack limit.
  std::atomic<int> ran{0};
  ThreadOpts opts;
  opts.stack_size = 2u * 1024u * 1024u;  // 2 MiB
  auto r = Thread::spawn(
      [](void* a) {
        static_cast<std::atomic<int>*>(a)->store(7, std::memory_order_release);
      },
      &ran, opts);
  ASSERT_TRUE(r.has_value()) << "spawn with stack_size failed";
  std::move(*r).join();
  EXPECT_EQ(ran.load(std::memory_order_acquire), 7);
}

TEST(Thread, SpawnWithNameOnly) {
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.name = "hyle-named";
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
      opts);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_TRUE(ran.load());
}

TEST(Thread, AffinityOutOfRangeDoesNotCrash) {
  // Best-effort: absurd CPU index must not abort spawn/join.
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.affinity = 4096u;
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
      opts);
  ASSERT_TRUE(r.has_value());
  std::move(*r).join();
  EXPECT_TRUE(ran.load());
}

TEST(Thread, RequiredSetupRejectsInvalidAffinityWithoutRunningEntry) {
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.affinity = std::numeric_limits<unsigned>::max();
  opts.config_policy = ThreadConfigPolicy::kRequired;
  auto result = Thread::spawn(
      [](void* arg) { static_cast<std::atomic<bool>*>(arg)->store(true); },
      &ran, opts);
  ASSERT_FALSE(result.has_value());
  EXPECT_FALSE(ran.load());
}

TEST(Thread, CpuAffinityRepresentsSingleAndSetWithoutOwningTheView) {
  CpuAffinity none;
  EXPECT_TRUE(none.empty());

  CpuAffinity one{7u};
  ASSERT_EQ(one.size(), 1u);
  EXPECT_EQ(one[0], 7u);

  const unsigned cpus[] = {2u, 6u, 9u};
  CpuAffinity set{std::span<const unsigned>(cpus)};
  ASSERT_EQ(set.size(), 3u);
  EXPECT_EQ(set[0], 2u);
  EXPECT_EQ(set[1], 6u);
  EXPECT_EQ(set[2], 9u);
}

TEST(Thread, SpawnCopiesCpuAffinitySetBeforeReturning) {
  auto count = hyle::platform::available_cpu_ids(nullptr, 0);
  ASSERT_TRUE(count.has_value());
  if (*count == 0) GTEST_SKIP() << "no available CPU ids";

  std::vector<unsigned> available(*count);
  auto filled = hyle::platform::available_cpu_ids(available.data(),
                                                   available.size());
  ASSERT_TRUE(filled.has_value());
  const std::size_t selected = std::min<std::size_t>(2, available.size());
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.affinity = CpuAffinity(
      std::span<const unsigned>(available.data(), selected));
#if !defined(__APPLE__)
  opts.config_policy = ThreadConfigPolicy::kRequired;
#endif
  auto result = Thread::spawn(
      [](void* arg) { static_cast<std::atomic<bool>*>(arg)->store(true); },
      &ran, opts);
  ASSERT_TRUE(result.has_value());
  // The view is no longer needed once spawn has returned.
  std::vector<unsigned>().swap(available);
  std::move(*result).join();
  EXPECT_TRUE(ran.load());
}

#if defined(__linux__)
TEST(Thread, CpuAffinitySetTakesEffectExactly) {
  auto count = hyle::platform::available_cpu_ids(nullptr, 0);
  ASSERT_TRUE(count.has_value());
  if (*count < 2) GTEST_SKIP() << "need two available CPUs";
  std::vector<unsigned> available(*count);
  ASSERT_TRUE(hyle::platform::available_cpu_ids(available.data(),
                                                available.size()));
  const unsigned selected[] = {available[0], available[1]};

  struct Probe {
    unsigned cpus[2];
    int exact = 0;
  } probe{{selected[0], selected[1]}, 0};
  ThreadOpts opts;
  opts.affinity = CpuAffinity(std::span<const unsigned>(selected));
  opts.config_policy = ThreadConfigPolicy::kRequired;
  auto result = Thread::spawn(
      [](void* arg) {
        auto* probe = static_cast<Probe*>(arg);
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (::sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
        probe->exact = CPU_COUNT(&mask) == 2 &&
                               CPU_ISSET(probe->cpus[0], &mask) &&
                               CPU_ISSET(probe->cpus[1], &mask)
                           ? 1
                           : 0;
      },
      &probe, opts);
  ASSERT_TRUE(result.has_value());
  std::move(*result).join();
  EXPECT_EQ(probe.exact, 1);
}
#endif

#if defined(_WIN32)
TEST(Thread, RequiredAffinityRejectsCrossProcessorGroupSet) {
  if (::GetActiveProcessorGroupCount() < 2) {
    GTEST_SKIP() << "need two Windows processor groups";
  }
  const unsigned first_group_size = ::GetActiveProcessorCount(0);
  const unsigned cpus[] = {0u, first_group_size};
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.affinity = CpuAffinity(std::span<const unsigned>(cpus));
  opts.config_policy = ThreadConfigPolicy::kRequired;
  auto result = Thread::spawn(
      [](void* arg) { static_cast<std::atomic<bool>*>(arg)->store(true); },
      &ran, opts);
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsInvalidArgument());
  EXPECT_FALSE(ran.load());
}
#endif

TEST(Thread, RequiredSetupCompletesBeforeSpawnReturns) {
  std::atomic<bool> ran{false};
  ThreadOpts opts;
  opts.name = "hyle-required-name-that-is-longer-than-linux-limit";
  opts.config_policy = ThreadConfigPolicy::kRequired;
  auto result = Thread::spawn(
      [](void* arg) { static_cast<std::atomic<bool>*>(arg)->store(true); },
      &ran, opts);
  ASSERT_TRUE(result.has_value());
  std::move(*result).join();
  EXPECT_TRUE(ran.load());
}

TEST(Thread, PriorityBestEffortRuns) {
  // Priority is best-effort (raising needs privilege on POSIX). Every level must
  // spawn + run + join cleanly regardless of whether the OS actually grants it.
  for (ThreadPriority p : {ThreadPriority::kLowest, ThreadPriority::kLow,
                           ThreadPriority::kNormal, ThreadPriority::kHigh,
                           ThreadPriority::kHighest}) {
    std::atomic<bool> ran{false};
    ThreadOpts opts;
    opts.priority = p;
    auto r = Thread::spawn(
        [](void* a) { static_cast<std::atomic<bool>*>(a)->store(true); }, &ran,
        opts);
    ASSERT_TRUE(r.has_value());
    std::move(*r).join();
    EXPECT_TRUE(ran.load());
  }
}

TEST(Thread, ManyThreadsJoinClean) {
  constexpr int kN = 32;
  std::atomic<int> count{0};
  Thread threads[kN];
  for (int i = 0; i < kN; ++i) {
    auto r = Thread::spawn(
        [](void* a) {
          static_cast<std::atomic<int>*>(a)->fetch_add(1,
                                                       std::memory_order_acq_rel);
        },
        &count);
    ASSERT_TRUE(r.has_value());
    threads[i] = std::move(*r);
  }
  for (int i = 0; i < kN; ++i) threads[i].join();
  EXPECT_EQ(count.load(), kN);
}

TEST(Thread, MoveAssignIntoEmpty) {
  std::atomic<int> ran{0};
  auto r = Thread::spawn(
      [](void* a) { static_cast<std::atomic<int>*>(a)->store(1); }, &ran);
  ASSERT_TRUE(r.has_value());
  Thread empty;
  empty = std::move(*r);  // empty is not joinable -> allowed
  EXPECT_TRUE(empty.joinable());
  empty.join();
  EXPECT_EQ(ran.load(), 1);
}

// join() on a non-joinable Thread is a contract bug -> fail-fast. Locks in the
// joinable() guard added to Thread::join() (platform-audit #3).
TEST(ThreadDeathTest, JoinNonJoinableAborts) {
  EXPECT_DEATH(
      {
        Thread t;   // default-constructed: not joinable (handle_ == 0)
        t.join();   // HYLE_FATAL_ASSERT(joinable()) fires
      },
      ".*");
}
