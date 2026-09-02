#include "hyle/mem/buffer_pool.h"

#include <array>
#include <bit>
#include <limits>
#include <mutex>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {
namespace {

constexpr std::size_t kBucketCount = sizeof(std::size_t) * 8 + 1;

struct Entry {
  Buffer buffer;
  Entry* next = nullptr;
};

struct PoolImpl {
  PoolImpl(MemoryResourceRef selected_resource,
           BufferPoolOptions selected_options) noexcept
      : resource(selected_resource), options(selected_options) {}

  MemoryResourceRef resource;
  BufferPoolOptions options;
  mutable std::mutex mutex;
  std::array<Entry*, kBucketCount> buckets{};
  BufferPoolStats stats{};
};

PoolImpl* as_pool(void* pointer) noexcept {
  return static_cast<PoolImpl*>(pointer);
}

Entry* as_entry(void* pointer) noexcept { return static_cast<Entry*>(pointer); }

std::size_t bucket_for(std::size_t size) noexcept {
  return size == 0 ? 0 : static_cast<std::size_t>(std::bit_width(size));
}

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

Result<std::size_t> allocation_size(std::size_t request,
                                    const BufferPoolOptions& options,
                                    std::size_t maximum) noexcept {
  if (request == 0) return 0;
  if (request > maximum) {
    return std::unexpected(
        Status::OutOfRange("buffer pool request exceeds resource maximum"));
  }
  const std::size_t growth = request / 100 * options.growth_percent +
      (request % 100) * options.growth_percent / 100;
  std::size_t grown = request;
  if (growth <= maximum - request) grown += growth;
  const std::size_t mask = options.allocation_granularity - 1;
  if (grown > maximum - mask) grown = request;
  std::size_t rounded = (grown + mask) & ~mask;
  if (rounded < request || rounded > maximum) {
    if (request > maximum - mask) return request;
    rounded = (request + mask) & ~mask;
  }
  return rounded;
}

Entry* make_entry(Buffer buffer) noexcept {
  AllocatorRef allocator = system_allocator();
  void* storage = allocator.allocate(sizeof(Entry), alignof(Entry));
  if (storage == nullptr) return nullptr;
  return ::new (storage) Entry{std::move(buffer), nullptr};
}

void destroy_entry(Entry* entry) noexcept {
  if (entry == nullptr) return;
  entry->~Entry();
  system_allocator().deallocate(entry, sizeof(Entry), alignof(Entry));
}

void destroy_list(Entry* entries) noexcept {
  while (entries != nullptr) {
    Entry* next = entries->next;
    destroy_entry(entries);
    entries = next;
  }
}

Entry* detach_all(PoolImpl& pool) noexcept {
  Entry* output = nullptr;
  for (Entry*& bucket : pool.buckets) {
    while (bucket != nullptr) {
      Entry* entry = bucket;
      bucket = entry->next;
      entry->next = output;
      output = entry;
    }
  }
  pool.stats.cached_bytes = 0;
  pool.stats.cached_buffers = 0;
  return output;
}

Entry* find_reusable(PoolImpl& pool, std::size_t size,
                     std::size_t alignment) noexcept {
  const std::size_t first_bucket = bucket_for(size);
  for (std::size_t bucket_index = first_bucket;
       bucket_index < kBucketCount; ++bucket_index) {
    Entry** best_link = nullptr;
    std::size_t best_slack = (std::numeric_limits<std::size_t>::max)();
    Entry** link = &pool.buckets[bucket_index];
    std::uint8_t scanned = 0;
    while (*link != nullptr &&
           scanned < pool.options.max_entries_scanned_per_bucket) {
      Entry* entry = *link;
      const std::size_t capacity = entry->buffer.size();
      if (capacity >= size && entry->buffer.alignment() >= alignment) {
        const std::size_t slack = capacity - size;
        if (slack <= pool.options.max_reuse_slack && slack < best_slack) {
          best_link = link;
          best_slack = slack;
          if (slack == 0) break;
        }
      }
      link = &entry->next;
      ++scanned;
    }
    if (best_link != nullptr) {
      Entry* result = *best_link;
      *best_link = result->next;
      result->next = nullptr;
      pool.stats.cached_bytes -= result->buffer.size();
      --pool.stats.cached_buffers;
      ++pool.stats.cache_reuses;
      return result;
    }

    // Every capacity in a later bucket is at least 2^(bucket_index). Once that
    // lower bound already exceeds permitted slack, no later bucket can match.
    if (bucket_index < sizeof(std::size_t) * 8) {
      const std::size_t next_minimum = std::size_t{1} << bucket_index;
      if (next_minimum > size &&
          next_minimum - size > pool.options.max_reuse_slack) {
        break;
      }
    }
  }
  return nullptr;
}

void account_acquire(PoolImpl& pool, const Entry& entry) noexcept {
  pool.stats.live_bytes += entry.buffer.size();
  ++pool.stats.live_buffers;
  if (pool.stats.live_bytes > pool.stats.peak_live_bytes) {
    pool.stats.peak_live_bytes = pool.stats.live_bytes;
  }
}

void return_entry(PoolImpl& pool, Entry* entry, bool allow_cache) noexcept {
  bool cache = false;
  {
    std::lock_guard lock(pool.mutex);
    HYLE_FATAL_ASSERT(pool.stats.live_buffers != 0);
    HYLE_FATAL_ASSERT(pool.stats.live_bytes >= entry->buffer.size());
    pool.stats.live_bytes -= entry->buffer.size();
    --pool.stats.live_buffers;

    const std::size_t capacity = entry->buffer.size();
    const bool byte_room =
        pool.stats.cached_bytes <= pool.options.max_cached_bytes &&
        capacity <=
            pool.options.max_cached_bytes - pool.stats.cached_bytes;
    cache = allow_cache && capacity != 0 &&
            pool.stats.cached_buffers < pool.options.max_cached_buffers &&
            byte_room;
    if (cache) {
      const std::size_t bucket = bucket_for(capacity);
      entry->next = pool.buckets[bucket];
      pool.buckets[bucket] = entry;
      pool.stats.cached_bytes += capacity;
      ++pool.stats.cached_buffers;
    } else {
      ++pool.stats.discarded_on_return;
    }
  }
  if (!cache) destroy_entry(entry);
}

Buffer take_entry(PoolImpl& pool, Entry* entry) noexcept {
  {
    std::lock_guard lock(pool.mutex);
    HYLE_FATAL_ASSERT(pool.stats.live_buffers != 0);
    HYLE_FATAL_ASSERT(pool.stats.live_bytes >= entry->buffer.size());
    pool.stats.live_bytes -= entry->buffer.size();
    --pool.stats.live_buffers;
  }
  Buffer output = std::move(entry->buffer);
  destroy_entry(entry);
  return output;
}

}  // namespace

Result<BufferPool> BufferPool::create(MemoryResourceRef resource,
                                      BufferPoolOptions options) noexcept {
  if (!resource.valid()) {
    return std::unexpected(
        Status::InvalidArgument("buffer pool resource is invalid"));
  }
  if (!is_power_of_two(options.allocation_granularity)) {
    return std::unexpected(Status::InvalidArgument(
        "buffer pool allocation granularity must be a power of two"));
  }
  if (options.growth_percent > 100) {
    return std::unexpected(
        Status::InvalidArgument("buffer pool growth percent exceeds 100"));
  }
  if (options.max_entries_scanned_per_bucket == 0) {
    return std::unexpected(Status::InvalidArgument(
        "buffer pool bounded scan must inspect at least one entry"));
  }
  AllocatorRef allocator = system_allocator();
  void* storage = allocator.allocate(sizeof(PoolImpl), alignof(PoolImpl));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("buffer pool metadata allocation failed"));
  }
  auto* impl = ::new (storage) PoolImpl(resource, options);
  return BufferPool(impl);
}

BufferPool::BufferPool(BufferPool&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

BufferPool& BufferPool::operator=(BufferPool&& other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      PoolImpl* pool = as_pool(impl_);
      HYLE_FATAL_ASSERT(pool->stats.live_buffers == 0);
      trim();
      pool->~PoolImpl();
      system_allocator().deallocate(pool, sizeof(PoolImpl), alignof(PoolImpl));
    }
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

BufferPool::~BufferPool() {
  if (impl_ == nullptr) return;
  PoolImpl* pool = as_pool(impl_);
  HYLE_FATAL_ASSERT(pool->stats.live_buffers == 0);
  trim();
  pool->~PoolImpl();
  system_allocator().deallocate(pool, sizeof(PoolImpl), alignof(PoolImpl));
}

Result<BufferLease> BufferPool::acquire(std::size_t size,
                                        std::size_t alignment) noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("buffer pool is invalid"));
  }
  if (!is_power_of_two(alignment)) {
    return std::unexpected(Status::InvalidArgument(
        "buffer pool alignment must be a non-zero power of two"));
  }
  PoolImpl& pool = *as_pool(impl_);
  Entry* entry = nullptr;
  if (size != 0) {
    std::lock_guard lock(pool.mutex);
    entry = find_reusable(pool, size, alignment);
    if (entry != nullptr) account_acquire(pool, *entry);
  }

  if (entry == nullptr) {
    const MemoryProperties properties = pool.resource.properties();
    HYLE_ASSIGN_OR_RETURN(
        std::size_t capacity,
        allocation_size(size, pool.options, properties.maximum_allocation));
    auto created = Buffer::create(pool.resource, capacity, alignment);
    if (!created && created.error().IsResourceExhausted()) {
      trim();
      {
        std::lock_guard lock(pool.mutex);
        ++pool.stats.oom_trim_retries;
      }
      created = Buffer::create(pool.resource, capacity, alignment);
    }
    if (!created) return std::unexpected(std::move(created).error());
    entry = make_entry(std::move(*created));
    if (entry == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "buffer pool entry metadata allocation failed"));
    }
    {
      std::lock_guard lock(pool.mutex);
      ++pool.stats.provider_allocations;
      account_acquire(pool, *entry);
    }
  }

  auto logical = entry->buffer.view().subview(0, size);
  HYLE_FATAL_ASSERT(logical.has_value());
  return BufferLease(&pool, entry, *logical);
}

BufferPoolStats BufferPool::stats() const noexcept {
  if (!valid()) return {};
  PoolImpl& pool = *as_pool(impl_);
  std::lock_guard lock(pool.mutex);
  return pool.stats;
}

MemoryResourceRef BufferPool::resource() const noexcept {
  return valid() ? as_pool(impl_)->resource : MemoryResourceRef{};
}

void BufferPool::trim(std::size_t target_cached_bytes) noexcept {
  if (!valid()) return;
  PoolImpl& pool = *as_pool(impl_);
  Entry* discarded = nullptr;
  {
    std::lock_guard lock(pool.mutex);
    if (target_cached_bytes == 0) {
      discarded = detach_all(pool);
    } else {
      for (std::size_t bucket = kBucketCount;
           bucket != 0 && pool.stats.cached_bytes > target_cached_bytes;) {
        --bucket;
        while (pool.buckets[bucket] != nullptr &&
               pool.stats.cached_bytes > target_cached_bytes) {
          Entry* entry = pool.buckets[bucket];
          pool.buckets[bucket] = entry->next;
          pool.stats.cached_bytes -= entry->buffer.size();
          --pool.stats.cached_buffers;
          entry->next = discarded;
          discarded = entry;
        }
      }
    }
  }
  destroy_list(discarded);
}

BufferLease::BufferLease(BufferLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      entry_(std::exchange(other.entry_, nullptr)),
      view_(std::exchange(other.view_, BufferView{})) {}

BufferLease& BufferLease::operator=(BufferLease&& other) noexcept {
  if (this != &other) {
    reset();
    pool_ = std::exchange(other.pool_, nullptr);
    entry_ = std::exchange(other.entry_, nullptr);
    view_ = std::exchange(other.view_, BufferView{});
  }
  return *this;
}

BufferLease::~BufferLease() { reset(); }

std::size_t BufferLease::capacity() const noexcept {
  return valid() ? as_entry(entry_)->buffer.size() : 0;
}

void BufferLease::reset() noexcept {
  if (!valid()) return;
  return_entry(*as_pool(pool_), as_entry(entry_), true);
  pool_ = nullptr;
  entry_ = nullptr;
  view_ = {};
}

Buffer BufferLease::take() noexcept {
  if (!valid()) return {};
  Buffer output = take_entry(*as_pool(pool_), as_entry(entry_));
  pool_ = nullptr;
  entry_ = nullptr;
  view_ = {};
  return output;
}

}  // namespace hyle::mem
