#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <utility>

#include "hyle/io/io_context.h"
#include "hyle/io/signal.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if !defined(_WIN32)
#include <csignal>
#endif

using hyle::io::IoContext;
using hyle::io::ProcessSignal;
using hyle::io::SignalInterest;
using hyle::io::SignalSet;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;

#if !defined(_WIN32)

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

struct SignalResult {
  std::atomic<int> first{-1};
  std::atomic<int> second{-1};
  std::atomic<bool> done{false};
};

Task<int> wait_once(AllocatorRef, SignalResult *result, SignalSet *signals) {
  auto received = co_await signals->wait();
  result->first.store(received.has_value() ? static_cast<int>(*received) : -1,
                      std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> wait_twice(AllocatorRef, SignalResult *result, SignalSet *signals) {
  auto first = co_await signals->wait();
  auto second = co_await signals->wait();
  result->first.store(first.has_value() ? static_cast<int>(*first) : -1,
                      std::memory_order_release);
  result->second.store(second.has_value() ? static_cast<int>(*second) : -1,
                       std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> raise_terminate(AllocatorRef) {
  ::raise(SIGTERM);
  co_return 0;
}

Task<int> raise_both(AllocatorRef) {
  ::raise(SIGINT);
  ::raise(SIGTERM);
  co_return 0;
}

void prior_interrupt_handler(int) {}

#endif

} // namespace

#if !defined(_WIN32)

TEST(IoSignal, WaitReturnsTypedSignal) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler, 1);
  ASSERT_TRUE(io.valid());
  auto created = SignalSet::create(io, kOwner, SignalInterest::kTerminate);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  SignalSet signals = std::move(*created);
  SignalResult result;

  Drive waiter{wait_once(system_allocator(), &result, &signals)};
  Drive raiser{raise_terminate(system_allocator())};
  Fiber wait_fiber(kDefaultStackSize, &drive_entry, &waiter);
  Fiber raise_fiber(kDefaultStackSize, &drive_entry, &raiser);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &wait_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &raise_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result.done.load());
  EXPECT_EQ(result.first.load(), static_cast<int>(ProcessSignal::kTerminate));
}

TEST(IoSignal, DistinctBurstSignalsAreNotOverwritten) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler, 1);
  ASSERT_TRUE(io.valid());
  auto created = SignalSet::create(
      io, kOwner, SignalInterest::kInterrupt | SignalInterest::kTerminate);
  ASSERT_TRUE(created.has_value());
  SignalSet signals = std::move(*created);
  SignalResult result;

  Drive waiter{wait_twice(system_allocator(), &result, &signals)};
  Drive raiser{raise_both(system_allocator())};
  Fiber wait_fiber(kDefaultStackSize, &drive_entry, &waiter);
  Fiber raise_fiber(kDefaultStackSize, &drive_entry, &raiser);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &wait_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &raise_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result.done.load());
  EXPECT_NE(result.first.load(), result.second.load());
  EXPECT_TRUE(
      (result.first.load() == static_cast<int>(ProcessSignal::kInterrupt) &&
       result.second.load() == static_cast<int>(ProcessSignal::kTerminate)) ||
      (result.first.load() == static_cast<int>(ProcessSignal::kTerminate) &&
       result.second.load() == static_cast<int>(ProcessSignal::kInterrupt)));
}

TEST(IoSignal, FactoryRejectsDuplicateAndInvalidOwnership) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler, 1);
  ASSERT_TRUE(io.valid());
  auto first = SignalSet::create(
      io, kOwner, SignalInterest::kInterrupt | SignalInterest::kTerminate);
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(SignalSet::create(io, kOwner, SignalInterest::kInterrupt)
                  .error()
                  .IsAlreadyExists());

  SignalSet moved = std::move(*first);
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.interest(),
            SignalInterest::kInterrupt | SignalInterest::kTerminate);
}

TEST(IoSignal, InvalidInputIsTyped) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler, 1);
  ASSERT_TRUE(io.valid());
  EXPECT_TRUE(SignalSet::create(io, kOwner, SignalInterest::kNone)
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(
      SignalSet::create(io, io.num_workers(), SignalInterest::kInterrupt)
          .error()
          .IsOutOfRange());
}

TEST(IoSignal, RestoresPriorHandler) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler, 1);
  ASSERT_TRUE(io.valid());

  struct sigaction original{};
  struct sigaction custom{};
  struct sigaction observed{};
  ASSERT_EQ(::sigaction(SIGINT, nullptr, &original), 0);
  custom.sa_handler = &prior_interrupt_handler;
  sigemptyset(&custom.sa_mask);
  ASSERT_EQ(::sigaction(SIGINT, &custom, nullptr), 0);
  {
    auto created = SignalSet::create(io, kOwner, SignalInterest::kInterrupt);
    ASSERT_TRUE(created.has_value());
  }
  ASSERT_EQ(::sigaction(SIGINT, nullptr, &observed), 0);
  EXPECT_EQ(observed.sa_handler, &prior_interrupt_handler);
  ASSERT_EQ(::sigaction(SIGINT, &original, nullptr), 0);
}

#else

TEST(IoSignal, WindowsFactoryOwnsOneProcessHandler) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler, 1);
  ASSERT_TRUE(io.valid());
  auto created = SignalSet::create(
      io, kOwner, SignalInterest::kInterrupt | SignalInterest::kTerminate);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  EXPECT_TRUE(SignalSet::create(io, kOwner, SignalInterest::kTerminate)
                  .error()
                  .IsAlreadyExists());
  SignalSet moved = std::move(*created);
  EXPECT_TRUE(moved.valid());
}

TEST(IoSignal, WindowsInvalidInputIsTyped) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler, 1);
  ASSERT_TRUE(io.valid());
  EXPECT_TRUE(SignalSet::create(io, kOwner, SignalInterest::kNone)
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(
      SignalSet::create(io, io.num_workers(), SignalInterest::kInterrupt)
          .error()
          .IsOutOfRange());
}

#endif
