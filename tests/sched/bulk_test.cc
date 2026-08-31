#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/bulk.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

namespace {

using hyle::Status;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::bulk_execute;
using hyle::sched::BulkExecuteOptions;
using hyle::sched::CancelToken;
using hyle::sched::Event;
using hyle::sched::is_cancel_requested;
using hyle::sched::Scheduler;
using hyle::sched::spawn_task;
using hyle::sched::Task;

Status count_raw(void* context, std::size_t begin,
                 std::size_t end) noexcept {
  auto* hits = static_cast<std::atomic<unsigned>*>(context);
  for (std::size_t i = begin; i < end; ++i) {
    hits[i].fetch_add(1, std::memory_order_relaxed);
  }
  return Status();
}

TEST(BulkExecute, RawAndVoidLambdaCoverEachIndexExactlyOnce) {
  Scheduler scheduler(4, system_allocator(), 64);
  constexpr std::size_t kCount = 257;
  std::vector<std::atomic<unsigned>> raw_hits(kCount);
  std::vector<std::atomic<unsigned>> lambda_hits(kCount);
  std::atomic<unsigned> lambda_calls{0};
  std::atomic<unsigned> invalid_lambda_bounds{0};

  BulkExecuteOptions options;
  options.grain_size = 7;
  Status raw = bulk_execute(scheduler, 0, kCount, &count_raw,
                            raw_hits.data(), options);
  ASSERT_TRUE(raw.ok()) << raw.message();

  Status adapted = bulk_execute(
      scheduler, 0, kCount,
      [&](std::size_t begin, std::size_t end) noexcept {
        lambda_calls.fetch_add(1, std::memory_order_relaxed);
        const std::size_t expected_end =
            begin + 7 < kCount ? begin + 7 : kCount;
        if (begin % 7 != 0 || end != expected_end) {
          invalid_lambda_bounds.fetch_add(1, std::memory_order_relaxed);
        }
        for (std::size_t i = begin; i < end; ++i) {
          lambda_hits[i].fetch_add(1, std::memory_order_relaxed);
        }
      },
      options);
  ASSERT_TRUE(adapted.ok()) << adapted.message();
  EXPECT_EQ(lambda_calls.load(std::memory_order_relaxed), 37u);
  EXPECT_EQ(invalid_lambda_bounds.load(std::memory_order_relaxed), 0u);

  for (std::size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(raw_hits[i].load(std::memory_order_relaxed), 1u) << i;
    EXPECT_EQ(lambda_hits[i].load(std::memory_order_relaxed), 1u) << i;
  }
  scheduler.shutdown();
}

TEST(BulkExecute, EmptyAndInvalidInputsAreTyped) {
  Scheduler scheduler(2, system_allocator(), 16);
  std::atomic<unsigned> calls{0};
  auto body = [&](std::size_t, std::size_t) noexcept {
    calls.fetch_add(1, std::memory_order_relaxed);
  };

  EXPECT_TRUE(bulk_execute(scheduler, 9, 9, body).ok());
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 0u);

  EXPECT_TRUE(bulk_execute(scheduler, 9, 8, body).IsInvalidArgument());
  BulkExecuteOptions zero_grain;
  zero_grain.grain_size = 0;
  EXPECT_TRUE(
      bulk_execute(scheduler, 0, 1, body, zero_grain).IsInvalidArgument());
  EXPECT_TRUE(bulk_execute(scheduler, 0, 1, nullptr, nullptr)
                  .IsInvalidArgument());

  Status construction;
  Scheduler invalid(1, hyle::mem::Allocator{}, 16, &construction);
  ASSERT_FALSE(invalid.valid());
  EXPECT_TRUE(bulk_execute(invalid, 0, 1, body).IsFailedPrecondition());

  scheduler.shutdown();
  EXPECT_TRUE(bulk_execute(scheduler, 0, 1, body).IsCancelled());
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 0u);
}

TEST(BulkExecute, FirstBodyErrorStopsTheBatchAndReturnsStatus) {
  Scheduler scheduler(4, system_allocator(), 64);
  std::atomic<unsigned> calls{0};
  BulkExecuteOptions options;
  options.grain_size = 1;
  Status result = bulk_execute(
      scheduler, 0, 1024,
      [&](std::size_t, std::size_t) noexcept -> Status {
        calls.fetch_add(1, std::memory_order_relaxed);
        return Status::Internal("tile failed");
      },
      options);
  EXPECT_TRUE(result.IsInternal()) << result.message();
  EXPECT_GT(calls.load(std::memory_order_relaxed), 0u);
  EXPECT_LE(calls.load(std::memory_order_relaxed), 4u);
  scheduler.shutdown();
}

TEST(BulkExecute, CancellationIsSharedAndVisibleInsideBody) {
  Scheduler scheduler(4, system_allocator(), 64);
  CancelToken token = CancelToken::root(system_allocator());
  BulkExecuteOptions options;
  options.grain_size = 1;
  options.cancel_token = &token;

  token.cancel();
  std::atomic<unsigned> calls{0};
  Status before = bulk_execute(
      scheduler, 0, 64,
      [&](std::size_t, std::size_t) noexcept {
        calls.fetch_add(1, std::memory_order_relaxed);
      },
      options);
  EXPECT_TRUE(before.IsCancelled());
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 0u);

  token = CancelToken::root(system_allocator());
  std::atomic<bool> scope_seen{false};
  Status during = bulk_execute(
      scheduler, 0, 4096,
      [&](std::size_t, std::size_t) noexcept {
        if (calls.fetch_add(1, std::memory_order_relaxed) == 0) {
          token.cancel();
          scope_seen.store(is_cancel_requested(), std::memory_order_release);
        }
      },
      options);
  EXPECT_TRUE(during.IsCancelled()) << during.message();
  EXPECT_TRUE(scope_seen.load(std::memory_order_acquire));
  EXPECT_LT(calls.load(std::memory_order_relaxed), 4096u);
  scheduler.shutdown();
}

Task<int> run_on_fiber(hyle::mem::Allocator, Scheduler* scheduler,
                       std::atomic<unsigned>* hits,
                       std::size_t count) noexcept {
  BulkExecuteOptions options;
  options.grain_size = 3;
  Status status = bulk_execute(
      *scheduler, 0, count,
      [hits](std::size_t begin, std::size_t end) noexcept {
        for (std::size_t i = begin; i < end; ++i) {
          hits[i].fetch_add(1, std::memory_order_relaxed);
        }
      },
      options);
  co_return status.ok() ? 0 : -1;
}

TEST(BulkExecute, SchedulerFiberWaitsWithoutDeadlockingOneWorker) {
  Scheduler scheduler(1, system_allocator(), 32);
  constexpr std::size_t kCount = 129;
  std::vector<std::atomic<unsigned>> hits(kCount);
  auto task = run_on_fiber(system_allocator(), &scheduler, hits.data(), kCount);
  auto handle = spawn_task(scheduler, system_allocator(), std::move(task));
  auto result = handle.join();
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_EQ(*result, 0);
  for (std::size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(hits[i].load(std::memory_order_relaxed), 1u) << i;
  }
  scheduler.shutdown();
}

struct NestedContext {
  Scheduler* scheduler = nullptr;
  std::atomic<unsigned>* hits = nullptr;
  std::size_t count = 0;
  Status result;
  Event completed;
};

void run_nested(void* raw) noexcept {
  auto& context = *static_cast<NestedContext*>(raw);
  BulkExecuteOptions options;
  options.grain_size = 5;
  context.result = bulk_execute(
      *context.scheduler, 0, context.count,
      [&](std::size_t begin, std::size_t end) noexcept {
        for (std::size_t i = begin; i < end; ++i) {
          context.hits[i].fetch_add(1, std::memory_order_relaxed);
        }
      },
      options);
  context.completed.signal();
}

TEST(BulkExecute, WorkerStackNestedCallUsesProgressSafeSerialFallback) {
  Scheduler scheduler(1, system_allocator(), 16);
  constexpr std::size_t kCount = 65;
  std::vector<std::atomic<unsigned>> hits(kCount);
  NestedContext context;
  context.scheduler = &scheduler;
  context.hits = hits.data();
  context.count = kCount;

  Status admission = scheduler.schedule(&run_nested, &context);
  ASSERT_TRUE(admission.ok()) << admission.message();
  context.completed.wait();
  ASSERT_TRUE(context.result.ok()) << context.result.message();
  for (std::size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(hits[i].load(std::memory_order_relaxed), 1u) << i;
  }
  scheduler.shutdown();
}

TEST(BulkExecute, CallerContextArraySupportsHeterogeneousWorkRecords) {
  struct Work {
    void (*fn)(void*) noexcept;
    void* context;
  };
  std::atomic<unsigned> a{0};
  std::atomic<unsigned> b{0};
  auto increment_one = [](void* raw) noexcept {
    static_cast<std::atomic<unsigned>*>(raw)->fetch_add(
        1, std::memory_order_relaxed);
  };
  auto increment_two = [](void* raw) noexcept {
    static_cast<std::atomic<unsigned>*>(raw)->fetch_add(
        2, std::memory_order_relaxed);
  };
  Work work[] = {{increment_one, &a}, {increment_two, &b},
                 {increment_two, &a}, {increment_one, &b}};

  Scheduler scheduler(2, system_allocator(), 16);
  Status status = bulk_execute(
      scheduler, 0, std::size(work),
      [&](std::size_t begin, std::size_t end) noexcept {
        for (std::size_t i = begin; i < end; ++i) {
          work[i].fn(work[i].context);
        }
      });
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(a.load(std::memory_order_relaxed), 3u);
  EXPECT_EQ(b.load(std::memory_order_relaxed), 3u);
  scheduler.shutdown();
}

struct AllocationCounter {
  std::atomic<std::size_t> calls{0};
};

void* counting_allocate(void* raw, std::size_t size,
                        std::size_t alignment) noexcept {
  static_cast<AllocationCounter*>(raw)->calls.fetch_add(
      1, std::memory_order_relaxed);
  return system_allocator().allocate(size, alignment);
}

void counting_deallocate(void*, void* pointer, std::size_t size,
                         std::size_t alignment) noexcept {
  system_allocator().deallocate(pointer, size, alignment);
}

void* counting_reallocate(void* raw, void* pointer, std::size_t old_size,
                          std::size_t new_size,
                          std::size_t alignment) noexcept {
  static_cast<AllocationCounter*>(raw)->calls.fetch_add(
      1, std::memory_order_relaxed);
  return system_allocator().reallocate(pointer, old_size, new_size, alignment);
}

TEST(BulkExecute, RunnerPathDoesNotAllocateFromSchedulerAllocator) {
  AllocationCounter counter;
  Allocator allocator(&counting_allocate, &counting_deallocate,
                      &counting_reallocate, &counter);
  Scheduler scheduler(4, allocator, 64);
  std::atomic<std::uint64_t> total{0};
  auto body = [&](std::size_t begin, std::size_t end) noexcept {
    total.fetch_add(end - begin, std::memory_order_relaxed);
  };
  ASSERT_TRUE(bulk_execute(scheduler, 0, 32, body).ok());  // warm wait path
  scheduler.drain();  // exclude the warm runner's Scheduler epilogue
  const std::size_t before = counter.calls.load(std::memory_order_relaxed);

  BulkExecuteOptions options;
  options.grain_size = 3;
  ASSERT_TRUE(bulk_execute(scheduler, 0, 4096, body, options).ok());
  scheduler.drain();
  EXPECT_EQ(counter.calls.load(std::memory_order_relaxed), before);
  EXPECT_EQ(total.load(std::memory_order_relaxed), 4128u);
  scheduler.shutdown();
}

TEST(BulkExecute, RepeatedBatchLifetimeStress) {
  Scheduler scheduler(4, system_allocator(), 64);
  for (std::size_t iteration = 0; iteration < 500; ++iteration) {
    std::atomic<std::size_t> covered{0};
    BulkExecuteOptions options;
    options.grain_size = iteration % 11 + 1;
    options.max_concurrency = iteration % 4 + 1;
    const std::size_t count = iteration % 97 + 1;
    Status status = bulk_execute(
        scheduler, 0, count,
        [&](std::size_t begin, std::size_t end) noexcept {
          covered.fetch_add(end - begin, std::memory_order_relaxed);
        },
        options);
    ASSERT_TRUE(status.ok()) << "iteration " << iteration << ": "
                             << status.message();
    ASSERT_EQ(covered.load(std::memory_order_relaxed), count)
        << "iteration " << iteration;
  }
  scheduler.shutdown();
}

}  // namespace
