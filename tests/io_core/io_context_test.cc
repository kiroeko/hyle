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

struct FailAfterAllocator {
  std::size_t successful = 0;
  std::size_t fail_after = 0;
  std::size_t live = 0;

  static void *allocate(void *context, std::size_t size,
                        std::size_t alignment) noexcept {
    auto &self = *static_cast<FailAfterAllocator *>(context);
    if (self.successful >= self.fail_after) return nullptr;
    void *pointer = hyle::mem::system_allocator().allocate(size, alignment);
    if (pointer != nullptr) {
      ++self.successful;
      ++self.live;
    }
    return pointer;
  }

  static void deallocate(void *context, void *pointer, std::size_t size,
                         std::size_t alignment) noexcept {
    auto &self = *static_cast<FailAfterAllocator *>(context);
    if (pointer != nullptr) --self.live;
    hyle::mem::system_allocator().deallocate(pointer, size, alignment);
  }

  static void *reallocate(void *context, void *pointer, std::size_t old_size,
                          std::size_t new_size,
                          std::size_t alignment) noexcept {
    auto &self = *static_cast<FailAfterAllocator *>(context);
    if (pointer == nullptr) return allocate(context, new_size, alignment);
    if (new_size == 0) {
      deallocate(context, pointer, old_size, alignment);
      return nullptr;
    }
    if (self.successful >= self.fail_after) return nullptr;
    void *replacement = hyle::mem::system_allocator().reallocate(
        pointer, old_size, new_size, alignment);
    if (replacement != nullptr) ++self.successful;
    return replacement;
  }

  hyle::mem::AllocatorRef get() noexcept {
    return {&allocate, &deallocate, &reallocate, this};
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

TEST(IoContext, ExposesTypedBackendSnapshotWithoutRawDriverAccess) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());
  hyle::io::IoContext context(scheduler);
  ASSERT_TRUE(context.valid()) << context.status().message();

  const auto capabilities = context.capabilities();
  EXPECT_NE(capabilities.backend, hyle::io::IoBackend::kNone);
  EXPECT_NE(capabilities.backend_name(), "none");
  EXPECT_EQ(context.capabilities(context.num_workers()).backend,
            hyle::io::IoBackend::kNone);
#if HYLE_OS_LINUX
  EXPECT_EQ(context.io_uring().valid(),
            capabilities.backend == hyle::io::IoBackend::kIoUring);
#endif
  scheduler.shutdown();
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

TEST(IoContext, RejectsASecondDriverForTheSameScheduler) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());
  hyle::io::IoContext first(scheduler);
  ASSERT_TRUE(first.valid()) << first.status().message();
  hyle::io::IoContext duplicate(scheduler);
  EXPECT_FALSE(duplicate.valid());
  EXPECT_TRUE(duplicate.status().IsAlreadyExists());
  scheduler.shutdown();
}

TEST(IoContext, RingTableAllocationFailureRollsBackInjectedState) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());
  FailAfterAllocator allocator;
  // BlockingPool owns the first allocation. Reject the following ring-owner
  // table allocation and verify both partial owners unwind.
  allocator.fail_after = 1;
  {
    hyle::io::IoContextOptions options;
    options.allocator = allocator.get();
    options.max_blocking_threads = 1;
    hyle::io::IoContext context(scheduler, options);
    EXPECT_FALSE(context.valid());
    EXPECT_TRUE(context.status().IsResourceExhausted());
  }
  EXPECT_EQ(allocator.live, 0u);
  scheduler.shutdown();
}

} // namespace
