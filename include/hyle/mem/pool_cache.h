#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/shared_pool.h"

namespace hyle::mem {

// Explicit single-owner front cache over a hard ConcurrentBlockPool.
//
// The ConcurrentBlockPool remains the thread-safe shared core. A worker owns one cache
// object and performs allocate/deallocate through it; free indices can remain
// local until publish() or cache destruction, so visibility is intentionally
// soft. The cache lifetime, storage, capacity and transfer batch are visible
// product choices rather than hidden TLS policy.
//
// `pool` must outlive the cache and every block acquired through it. One cache
// is owner-thread-only; different caches over the same ConcurrentBlockPool may be used
// concurrently. Destroying a cache publishes its locally free blocks but does
// not reclaim blocks that are still checked out by the user. Before resetting
// the core pool, quiesce users and call publish() on every attached cache;
// cached indices surviving a core reset would duplicate newly reset slots.
class BlockPoolCache {
public:
  static constexpr std::uint32_t kDefaultCapacity = 128;
  static constexpr std::uint32_t kDefaultTransferBatch = 32;

  // Strict no-hidden-allocation construction. `storage` is caller-owned and
  // must outlive the cache. Its element count is the local soft-capacity
  // window; contents are opaque and must not be modified while attached.
  BlockPoolCache(
      ConcurrentBlockPool &pool, std::span<std::uint32_t> storage,
      std::uint32_t transfer_batch = kDefaultTransferBatch) noexcept
      : pool_(&pool), indices_(storage.data()) {
    if (storage.empty() ||
        storage.size() > (std::numeric_limits<std::uint32_t>::max)())
      return;
    capacity_ = static_cast<std::uint32_t>(storage.size());
    transfer_batch_ = normalize_batch(transfer_batch, capacity_);
    valid_ = pool_->valid();
  }

  // Owned metadata construction. Only the cache's uint32 index array uses
  // `metadata`; the slab and shared links continue to belong to `pool`.
  BlockPoolCache(
      ConcurrentBlockPool &pool, AllocatorRef metadata,
      std::uint32_t capacity = kDefaultCapacity,
      std::uint32_t transfer_batch = kDefaultTransferBatch) noexcept
      : pool_(&pool), metadata_(metadata), capacity_(capacity), owns_(true) {
    if (!pool_->valid() || !metadata_.valid() || capacity_ == 0 ||
        static_cast<std::size_t>(capacity_) >
            (std::numeric_limits<std::size_t>::max)() / sizeof(std::uint32_t))
      return;
    indices_ = metadata_.allocate_n<std::uint32_t>(capacity_);
    if (indices_ == nullptr)
      return;
    transfer_batch_ = normalize_batch(transfer_batch, capacity_);
    valid_ = true;
  }

  // Common convenience; callers needing allocator-domain control use the
  // explicit metadata overload or caller-owned span above.
  explicit BlockPoolCache(
      ConcurrentBlockPool &pool, std::uint32_t capacity = kDefaultCapacity,
      std::uint32_t transfer_batch = kDefaultTransferBatch) noexcept
      : BlockPoolCache(pool, system_allocator(), capacity, transfer_batch) {}

  ~BlockPoolCache() {
    publish();
    if (owns_ && indices_ != nullptr)
      metadata_.deallocate_n(indices_, capacity_);
  }

  BlockPoolCache(const BlockPoolCache&) = delete;
  BlockPoolCache& operator=(const BlockPoolCache&) = delete;
  BlockPoolCache(BlockPoolCache&&) = delete;
  BlockPoolCache& operator=(BlockPoolCache&&) = delete;

  bool valid() const noexcept { return valid_; }
  std::uint32_t capacity() const noexcept { return capacity_; }
  std::uint32_t cached() const noexcept { return count_; }
  std::uint32_t transfer_batch() const noexcept { return transfer_batch_; }
  std::size_t block_size() const noexcept {
    return pool_ != nullptr ? pool_->block_size() : 0;
  }

  void *allocate() noexcept {
    if (!valid_)
      return nullptr;
    if (count_ == 0) {
      count_ = pool_->cache_refill(transfer_batch_, indices_);
      if (count_ == 0)
        return nullptr;
    }
    return pool_->cache_checkout(indices_[--count_]);
  }

  void deallocate(void *p) noexcept {
    if (p == nullptr)
      return;
    HYLE_FATAL_ASSERT(valid_);
    const std::uint32_t index = pool_->cache_return(p);
    if (count_ == capacity_)
      publish_some();
    HYLE_FATAL_ASSERT(count_ < capacity_);
    indices_[count_++] = index;
  }

  // Make every locally free block visible through the hard shared core.
  // Owner-thread-only, like allocate/deallocate on this cache object.
  void publish() noexcept {
    if (!valid_ || count_ == 0)
      return;
    pool_->cache_publish(indices_, count_);
    count_ = 0;
  }

  AllocatorRef as_allocator() noexcept {
    return AllocatorRef(&cache_alloc, &cache_free, &cache_realloc, this);
  }

#if defined(HYLE_TESTING)
  // TEST-ONLY: validate a cached slot before a typed wrapper runs user code.
  void testing_assert_checked_out(void *p) const noexcept {
    pool_->cache_assert_checked_out(p);
  }
#endif

private:
  static std::uint32_t normalize_batch(std::uint32_t requested,
                                       std::uint32_t capacity) noexcept {
    if (requested == 0)
      requested = kDefaultTransferBatch;
    return requested < capacity ? requested : capacity;
  }

  void publish_some() noexcept {
    const std::uint32_t n =
        transfer_batch_ < count_ ? transfer_batch_ : count_;
    HYLE_FATAL_ASSERT(n != 0);
    count_ -= n;
    pool_->cache_publish(indices_ + count_, n);
  }

  static void *cache_alloc(void *state, std::size_t size,
                           std::size_t alignment) noexcept {
    auto *self = static_cast<BlockPoolCache *>(state);
    if (size == 0)
      return nullptr;
    // Construction OOM is a runtime allocation failure, not a caller bug.
    // Keep the allocator handle complete and consistently always-failing.
    if (!self->valid_)
      return nullptr;
    if (!self->pool_->cache_request_fits(size, alignment))
      return nullptr;
    return self->allocate();
  }

  static void cache_free(void *state, void *ptr, std::size_t,
                         std::size_t) noexcept {
    if (ptr != nullptr)
      static_cast<BlockPoolCache *>(state)->deallocate(ptr);
  }

  static void *cache_realloc(void *state, void *ptr, std::size_t old_size,
                             std::size_t new_size,
                             std::size_t alignment) noexcept {
    auto *self = static_cast<BlockPoolCache *>(state);
    if (ptr == nullptr)
      return cache_alloc(state, new_size, alignment);
    HYLE_FATAL_ASSERT(self->valid_ &&
                      self->pool_->cache_request_fits(old_size, alignment));
    self->pool_->cache_assert_checked_out(ptr);
    if (new_size == 0) {
      self->deallocate(ptr);
      return nullptr;
    }
    if (!self->pool_->cache_request_fits(new_size, alignment))
      return nullptr;
    return ptr;
  }

  ConcurrentBlockPool *pool_ = nullptr;
  AllocatorRef metadata_;
  std::uint32_t *indices_ = nullptr;
  std::uint32_t capacity_ = 0;
  std::uint32_t count_ = 0;
  std::uint32_t transfer_batch_ = 0;
  bool owns_ = false;
  bool valid_ = false;
};

} // namespace hyle::mem
