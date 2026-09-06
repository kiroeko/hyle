#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <string_view>
#include <thread>
#include <vector>

#include "hyle/mem/buffer_pool.h"

namespace {

using hyle::Result;
using hyle::Status;
using namespace hyle::mem;

class LimitedHostResource final : public MemoryResource {
 public:
  explicit LimitedHostResource(std::size_t limit) noexcept : limit_(limit) {}

  std::string_view name() const noexcept override { return "test.limited"; }
  MemoryProperties properties() const noexcept override {
    return MemoryProperties{
        .flags = MemoryProperty::kHostAccessible |
                 MemoryProperty::kHostDeviceCoherent,
        .minimum_alignment = alignof(std::max_align_t),
        .maximum_allocation = limit_,
        .domain = MemoryDomainId{this, 0},
    };
  }
  Result<ResourceAllocation> allocate(std::size_t size,
                                      std::size_t alignment) noexcept override {
    ++attempts_;
    if (size > limit_ - live_bytes_) {
      return std::unexpected(Status::ResourceExhausted());
    }
    auto allocation = system_resource().resource().allocate(size, alignment);
    if (allocation) live_bytes_ += size;
    return allocation;
  }
  void deallocate(ResourceAllocation allocation) noexcept override {
    system_resource().resource().deallocate(allocation);
    live_bytes_ -= allocation.size;
  }

  std::size_t attempts() const noexcept { return attempts_; }
  std::size_t live_bytes() const noexcept { return live_bytes_; }

 private:
  std::size_t limit_;
  std::size_t live_bytes_ = 0;
  std::size_t attempts_ = 0;
};

TEST(BufferPool, ReusesBoundedBestFitAndPreservesLogicalView) {
  BufferPoolOptions options;
  options.growth_percent = 0;
  options.allocation_granularity = 64;
  BufferPool pool = BufferPool::create(system_resource(), options).value();

  void* first_handle = nullptr;
  {
    BufferLease first = pool.acquire(100, 64).value();
    EXPECT_EQ(first.size(), 100u);
    EXPECT_EQ(first.capacity(), 128u);
    first_handle = first.view().native_handle();
    EXPECT_TRUE(first.view().fill(std::byte{0x31}).ok());
  }
  BufferPoolStats cached = pool.stats();
  EXPECT_EQ(cached.cached_buffers, 1u);
  EXPECT_EQ(cached.cached_bytes, 128u);
  EXPECT_EQ(cached.live_buffers, 0u);

  {
    BufferLease second = pool.acquire(96, 32).value();
    EXPECT_EQ(second.view().native_handle(), first_handle);
    EXPECT_EQ(second.size(), 96u);
    EXPECT_EQ(second.capacity(), 128u);
  }
  BufferPoolStats reused = pool.stats();
  EXPECT_EQ(reused.provider_allocations, 1u);
  EXPECT_EQ(reused.cache_reuses, 1u);
}

TEST(BufferPool, RetentionCanBeDisabledAndTakeTransfersOwnership) {
  BufferPoolOptions options;
  options.max_cached_buffers = 0;
  BufferPool pool = BufferPool::create(system_resource(), options).value();
  {
    BufferLease lease = pool.acquire(64).value();
    Buffer owned = lease.take();
    EXPECT_FALSE(lease.valid());
    EXPECT_TRUE(owned.valid());
  }
  BufferPoolStats after_take = pool.stats();
  EXPECT_EQ(after_take.live_buffers, 0u);
  EXPECT_EQ(after_take.cached_buffers, 0u);

  pool.acquire(64).value().reset();
  EXPECT_EQ(pool.stats().cached_buffers, 0u);
  EXPECT_EQ(pool.stats().discarded_on_return, 1u);
}

TEST(BufferPool, TrimsCachedStorageOnceAndRetriesProviderOom) {
  LimitedHostResource resource(64);
  BufferPoolOptions options;
  options.growth_percent = 0;
  options.allocation_granularity = 1;
  options.max_reuse_slack = 0;
  BufferPool pool =
      BufferPool::create(MemoryResourceRef(resource), options).value();

  pool.acquire(64).value().reset();
  ASSERT_EQ(resource.live_bytes(), 64u);
  BufferLease smaller = pool.acquire(32).value();
  EXPECT_EQ(smaller.size(), 32u);
  EXPECT_EQ(pool.stats().oom_trim_retries, 1u);
  EXPECT_EQ(pool.stats().cached_buffers, 0u);
  EXPECT_EQ(resource.attempts(), 3u);
}

TEST(BufferPool, ConcurrentAcquireReturnMaintainsExactAccounting) {
  BufferPool pool = BufferPool::create(system_resource()).value();
  std::atomic<bool> failed{false};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&pool, &failed, worker] {
      for (int iteration = 0; iteration < 200; ++iteration) {
        const std::size_t size = 256u +
                                 static_cast<std::size_t>((worker + iteration) % 4) *
                                     128u;
        auto acquired = pool.acquire(size, 64);
        if (!acquired || !acquired->view().fill(std::byte{0x17}).ok()) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  EXPECT_FALSE(failed.load(std::memory_order_relaxed));
  const BufferPoolStats stats = pool.stats();
  EXPECT_EQ(stats.live_buffers, 0u);
  EXPECT_EQ(stats.live_bytes, 0u);
  EXPECT_LE(stats.cached_buffers, 256u);
  EXPECT_LE(stats.cached_bytes, std::size_t{1} << 30);
}

TEST(BufferPool, RejectsInvalidResourceAndPolicy) {
  EXPECT_TRUE(BufferPool::create({}).error().IsInvalidArgument());
  BufferPoolOptions options;
  options.allocation_granularity = 3;
  EXPECT_TRUE(BufferPool::create(system_resource(), options)
                  .error()
                  .IsInvalidArgument());
}

}  // namespace
