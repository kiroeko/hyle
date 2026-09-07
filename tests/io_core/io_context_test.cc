#include <atomic>
#include <cstddef>

#include <gtest/gtest.h>

#include "hyle/io/io_context.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/scheduler.h"

namespace {

struct CountingAllocator {
  static inline std::atomic<std::size_t> live_allocations{0};

  static void *allocate(void *, std::size_t size,
                        std::size_t alignment) noexcept {
    void *pointer = hyle::mem::system_allocator().allocate(size, alignment);
    if (pointer != nullptr)
      live_allocations.fetch_add(1, std::memory_order_relaxed);
    return pointer;
  }

  static void deallocate(void *, void *pointer, std::size_t size,
                         std::size_t alignment) noexcept {
    if (pointer != nullptr)
      live_allocations.fetch_sub(1, std::memory_order_relaxed);
    hyle::mem::system_allocator().deallocate(pointer, size, alignment);
  }

  static void *reallocate(void *, void *pointer, std::size_t old_size,
                          std::size_t new_size,
                          std::size_t alignment) noexcept {
    void *replacement = hyle::mem::system_allocator().reallocate(
        pointer, old_size, new_size, alignment);
    if (pointer == nullptr && replacement != nullptr)
      live_allocations.fetch_add(1, std::memory_order_relaxed);
    else if (pointer != nullptr && new_size == 0)
      live_allocations.fetch_sub(1, std::memory_order_relaxed);
    return replacement;
  }

  static hyle::mem::AllocatorRef get() noexcept {
    return {&allocate, &deallocate, &reallocate, nullptr};
  }
};

TEST(IoContext, DerivesWorkerCountAndBalancesInjectedMetadata) {
  CountingAllocator::live_allocations.store(0, std::memory_order_relaxed);
  hyle::sched::Scheduler scheduler(2, hyle::mem::system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());

  {
    hyle::io::IoContextOptions options;
    options.allocator = CountingAllocator::get();
    options.max_blocking_threads = 4;
    hyle::io::IoContext context(scheduler, options);
    ASSERT_TRUE(context.valid()) << context.status().message();
    EXPECT_EQ(context.num_workers(), scheduler.num_workers());
    EXPECT_GE(
        CountingAllocator::live_allocations.load(std::memory_order_relaxed),
        2u);
    scheduler.shutdown();
  }

  EXPECT_EQ(CountingAllocator::live_allocations.load(std::memory_order_relaxed),
            0u);
}

TEST(IoContext, InvalidOptionsReturnTypedStatus) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());

  hyle::io::IoContextOptions zero_depth;
  zero_depth.queue_depth = 0;
  hyle::io::IoContext invalid_depth(scheduler, zero_depth);
  EXPECT_FALSE(invalid_depth.valid());
  EXPECT_TRUE(invalid_depth.status().IsInvalidArgument());

  hyle::io::IoContextOptions invalid_allocator;
  invalid_allocator.allocator = {};
  hyle::io::IoContext invalid_alloc(scheduler, invalid_allocator);
  EXPECT_FALSE(invalid_alloc.valid());
  EXPECT_TRUE(invalid_alloc.status().IsInvalidArgument());

  scheduler.shutdown();
}

} // namespace
