#include "hyle/compute/resource_leases.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/core/assert.h"
#include "hyle/mem/range_allocator.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace hyle::compute {
namespace detail {

struct LeaseDomain;
struct LeaseBlock final : util::RefCounted<LeaseBlock> {
  LeaseBlock(LeaseDomain *owner, mem::RangeAllocation allocation,
             std::uint64_t identity) noexcept
      : domain(owner), range(std::move(allocation)), generation(identity) {}
  void destroy() noexcept;
  LeaseDomain *domain;
  mem::RangeAllocation range;
  std::uint64_t generation;
  Status failure{};
  std::uint32_t writes = 0;
  bool writer = true;
  bool writable = true;
};

struct LeaseDomain final : util::RefCounted<LeaseDomain> {
  LeaseDomain(mem::AllocatorRef selected, std::uint32_t count) noexcept
      : allocator(selected), blocks(selected, sizeof(LeaseBlock), count, alignof(LeaseBlock)) {}
  void destroy() noexcept {
    const auto selected = allocator;
    const auto retained_owner = resource_owner;
    const auto release_owner = release_resource_owner;
    this->~LeaseDomain();
    selected.deallocate(this, sizeof(LeaseDomain), alignof(LeaseDomain));
    if (release_owner) release_owner(retained_owner);
  }
  mem::AllocatorRef allocator;
  void *resource_owner = nullptr;
  LeaseAccess::ReleaseOwnerFn release_resource_owner = nullptr;
  ResourceBudget::Reservation charge{};
  mem::Buffer backing{};
  mem::RangeAllocator ranges{};
  mem::ConcurrentBlockPool blocks;
  util::SpinLock lock{};
  std::atomic<std::uint64_t> uses{0};
  std::uint64_t next_generation = 1, reclaimed = 0;
  bool closed = false;
};

void release_domain(LeaseDomain *owner) noexcept {
  bool last;
  {
    std::lock_guard guard(owner->lock);
    last = owner->is_unique();
    if (!last) owner->decref();
  }
  if (last) owner->decref();
}

void LeaseBlock::destroy() noexcept {
  auto *owner = domain;
  {
    std::lock_guard guard(owner->lock);
    this->~LeaseBlock();
    owner->blocks.deallocate(this);
    ++owner->reclaimed;
  }
  release_domain(owner);
}

struct LeaseUseEntry final {
  LeaseBlock *block = nullptr;
  bool write = false;
};
struct LeaseUsePoolState;
struct LeaseUseEpisode final {
  LeaseUsePoolState *pool = nullptr;
  CompletionRetention retention{};
  std::uint32_t count = 0;
};
struct LeaseUsePoolState final : util::RefCounted<LeaseUsePoolState> {
  LeaseUsePoolState(mem::AllocatorRef selected, std::size_t bytes,
                    std::uint32_t batches, std::uint32_t resources) noexcept
      : allocator(selected), episodes(selected, bytes, batches, alignof(LeaseUseEpisode)),
        capacity(resources) {}
  void destroy() noexcept {
    const auto selected = allocator;
    this->~LeaseUsePoolState();
    selected.deallocate(this, sizeof(LeaseUsePoolState), alignof(LeaseUsePoolState));
  }
  mem::AllocatorRef allocator;
  mem::ConcurrentBlockPool episodes;
  std::uint32_t capacity;
  util::SpinLock lock{};
  std::atomic<std::uint32_t> active{0};
  bool closed = false;
};

void release_use_pool(LeaseUsePoolState *owner) noexcept {
  bool last;
  {
    std::lock_guard guard(owner->lock);
    last = owner->is_unique();
    if (!last) owner->decref();
  }
  if (last) owner->decref();
}

constexpr std::size_t kEntryOffset =
    (sizeof(LeaseUseEpisode) + alignof(LeaseUseEntry) - 1) & ~(alignof(LeaseUseEntry) - 1);

LeaseUseEntry *entries(LeaseUseEpisode *episode) noexcept {
  return reinterpret_cast<LeaseUseEntry *>(reinterpret_cast<std::byte *>(episode) + kEntryOffset);
}

void episode_terminal(void *opaque, const Status &terminal) noexcept {
  if (terminal.ok()) return;
  auto *episode = static_cast<LeaseUseEpisode *>(opaque);
  for (std::uint32_t index = 0; index < episode->count; ++index) {
    auto &entry = entries(episode)[index];
    if (entry.write) {
      std::lock_guard guard(entry.block->domain->lock);
      entry.block->failure.Update(terminal);
    }
  }
}

void release_episode(void *opaque) noexcept {
  auto *episode = static_cast<LeaseUseEpisode *>(opaque);
  auto *pool = episode->pool;
  for (std::uint32_t index = 0; index < episode->count; ++index) {
    auto &entry = entries(episode)[index];
    auto *block = entry.block;
    auto *domain = block->domain;
    {
      std::lock_guard guard(domain->lock);
      domain->incref();
      if (entry.write) {
        HYLE_FATAL_ASSERT(block->writes == 1);
        --block->writes;
      }
    }
    block->decref();
    bool last_domain;
    {
      std::lock_guard guard(domain->lock);
      last_domain = domain->is_unique();
      if (!last_domain) domain->decref();
      domain->uses.fetch_sub(1, std::memory_order_release);
      domain->uses.notify_all();
    }
    if (last_domain) domain->decref();
    std::destroy_at(&entry);
  }
  std::destroy_at(episode);
  pool->episodes.deallocate(episode);
  bool last_pool;
  {
    std::lock_guard guard(pool->lock);
    last_pool = pool->is_unique();
    if (!last_pool) pool->decref();
    pool->active.fetch_sub(1, std::memory_order_release);
    pool->active.notify_all();
  }
  if (last_pool) pool->decref();
}

Status add_use(LeaseUseEpisode *episode, LeaseBlock *block, bool write) noexcept {
  if (episode == nullptr || block == nullptr)
    return Status::FailedPrecondition("resource use or lease is empty");
  for (std::uint32_t index = 0; index < episode->count; ++index) {
    const auto &entry = entries(episode)[index];
    if (entry.block == block)
      return entry.write == write ? Status::Ok()
                                 : Status::InvalidArgument("one use cannot mix lease permissions");
  }
  if (episode->count == episode->pool->capacity)
    return Status::ResourceExhausted("resource use metadata capacity exhausted");
  auto *domain = block->domain;
  {
    std::lock_guard guard(domain->lock);
    if (domain->closed) return Status::FailedPrecondition("lease domain is closed");
    if (block->writer != write)
      return Status::FailedPrecondition("resource use does not match lease permission");
    if (!block->failure.ok()) return block->failure;
    if (write && block->writes != 0)
      return Status::Unavailable("resource already has an admitted writer");
    if (write) ++block->writes;
    block->incref();
    domain->uses.fetch_add(1, std::memory_order_relaxed);
  }
  std::construct_at(entries(episode) + episode->count, block, write);
  ++episode->count;
  return Status::Ok();
}

template<class Counter> void drain_counter(Counter &counter) noexcept {
  for (auto active = counter.load(std::memory_order_acquire); active != 0;
       active = counter.load(std::memory_order_acquire))
    counter.wait(active, std::memory_order_acquire);
}

Result<mem::BufferView> lease_view(LeaseBlock *block, std::size_t offset,
                                 std::size_t size) noexcept {
  if (block == nullptr) return std::unexpected(Status::FailedPrecondition("lease is empty"));
  return block->domain->backing.view().subview(
      static_cast<std::size_t>(block->range.offset()) + offset, size);
}

Result<LeaseDomain *> prepare_lease_domain(mem::MemoryResourceRef resource,
    std::size_t capacity, std::size_t alignment, LeasePoolOptions options) noexcept {
  if (!resource.valid() || capacity == 0 || options.max_blocks == 0 ||
      options.max_blocks == UINT32_MAX || !options.allocator.valid() ||
      alignment == 0 || (alignment & (alignment - 1)) != 0)
    return std::unexpected(Status::InvalidArgument("invalid lease pool capacity/resource/alignment"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto ranges, mem::RangeAllocator::create(
      capacity, {.max_allocations = options.max_blocks, .metadata_allocator = allocator}));
  auto *storage = allocator.allocate(sizeof(LeaseDomain), alignof(LeaseDomain));
  if (!storage) return std::unexpected(Status::ResourceExhausted("lease domain metadata"));
  auto *state = std::construct_at(static_cast<LeaseDomain *>(storage), allocator, options.max_blocks);
  if (!state->blocks.valid()) {
    state->decref();
    return std::unexpected(Status::ResourceExhausted("lease block metadata"));
  }
  state->ranges = std::move(ranges);
  if (options.resource_budget.valid()) {
    auto charge = options.resource_budget.reserve(resource, capacity, ResourceBudgetCategory::kExternal);
    if (!charge) {
      state->decref();
      return std::unexpected(std::move(charge).error());
    }
    state->charge = std::move(*charge);
  }
  return state;
}
} // namespace detail

void detail::LeaseAccess::retain_resource_owner(ReadLease &lease, void *owner,
                                               ReleaseOwnerFn release) noexcept {
  HYLE_FATAL_ASSERT(lease.block_ && owner && release);
  auto *domain = lease.block_->domain;
  HYLE_FATAL_ASSERT(!domain->resource_owner && !domain->release_resource_owner);
  domain->resource_owner = owner;
  domain->release_resource_owner = release;
}
void detail::LeaseAccess::retain_resource_owner(LeasePool &pool, void *owner,
                                               ReleaseOwnerFn release) noexcept {
  HYLE_FATAL_ASSERT(pool.state_ && owner && release);
  HYLE_FATAL_ASSERT(!pool.state_->resource_owner && !pool.state_->release_resource_owner);
  pool.state_->resource_owner = owner;
  pool.state_->release_resource_owner = release;
}

Result<ReadLease> ReadLease::adopt(mem::Buffer &&backing, LeaseImportOptions options) noexcept {
  if (!backing.valid()) return std::unexpected(Status::InvalidArgument("lease backing is empty"));
  const auto bytes = backing.size();
  HYLE_ASSIGN_OR_RETURN(auto *state, detail::prepare_lease_domain(backing.resource(), bytes,
      backing.alignment(), {.max_blocks = 1, .allocator = options.allocator,
                            .resource_budget = options.resource_budget}));
  LeasePool candidate(state);
  auto *storage = state->blocks.allocate();
  if (!storage) return std::unexpected(Status::ResourceExhausted("readonly lease block metadata"));
  auto range = state->ranges.allocate(bytes, backing.alignment());
  if (!range) {
    state->blocks.deallocate(storage);
    return std::unexpected(std::move(range).error());
  }
  state->incref();
  auto *block = std::construct_at(static_cast<detail::LeaseBlock *>(storage), state,
      std::move(*range), state->next_generation++);
  block->writer = false;
  block->writable = false;
  // All fallible preparation is over. Drop only the cold root reference,
  // without closing admission for the returned immutable native-use lease.
  state->backing = std::move(backing);
  candidate.state_ = nullptr;
  detail::release_domain(state);
  return ReadLease(block, 0, bytes);
}

ReadLease::ReadLease(const ReadLease &other) noexcept
    : block_(other.block_), offset_(other.offset_), size_(other.size_) {
  if (block_) block_->incref();
}
ReadLease &ReadLease::operator=(const ReadLease &other) noexcept {
  if (this == &other) return *this;
  if (other.block_) other.block_->incref();
  reset();
  block_ = other.block_; offset_ = other.offset_; size_ = other.size_;
  return *this;
}
ReadLease::ReadLease(ReadLease &&other) noexcept
    : block_(std::exchange(other.block_, nullptr)), offset_(other.offset_), size_(other.size_) {}
ReadLease &ReadLease::operator=(ReadLease &&other) noexcept {
  if (this != &other) {
    reset(); block_ = std::exchange(other.block_, nullptr);
    offset_ = other.offset_; size_ = other.size_;
  }
  return *this;
}
ReadLease::~ReadLease() { reset(); }
void ReadLease::reset() noexcept {
  if (auto *block = std::exchange(block_, nullptr)) block->decref();
  offset_ = size_ = 0;
}
std::uint64_t ReadLease::generation() const noexcept { return block_ ? block_->generation : 0; }
Result<mem::ConstBufferView> ReadLease::view() const noexcept {
  HYLE_ASSIGN_OR_RETURN(auto storage, detail::lease_view(block_, offset_, size_));
  return mem::ConstBufferView(storage);
}
Result<ConstTensorView> ReadLease::tensor(TensorDesc description) const noexcept {
  HYLE_ASSIGN_OR_RETURN(auto storage, view());
  return ConstTensorView::create(description, storage);
}
Result<ReadLease> ReadLease::subview(std::size_t offset, std::size_t size) const noexcept {
  if (!block_) return std::unexpected(Status::FailedPrecondition("lease is empty"));
  if (offset > size_ || size > size_ - offset)
    return std::unexpected(Status::OutOfRange("lease subview exceeds range"));
  block_->incref();
  return ReadLease(block_, offset_ + offset, size);
}
Result<WriteLease> ReadLease::try_write() noexcept {
  if (!block_) return std::unexpected(Status::FailedPrecondition("lease is empty"));
  std::lock_guard guard(block_->domain->lock);
  if (block_->domain->closed || !block_->writable || !block_->is_unique())
    return std::unexpected(Status::Unavailable("lease is not exclusively writable"));
  block_->writer = true;
  return WriteLease(std::exchange(block_, nullptr), offset_, size_);
}

WriteLease::WriteLease(WriteLease &&other) noexcept
    : block_(std::exchange(other.block_, nullptr)), offset_(other.offset_), size_(other.size_) {}
WriteLease &WriteLease::operator=(WriteLease &&other) noexcept {
  if (this != &other) {
    reset(); block_ = std::exchange(other.block_, nullptr);
    offset_ = other.offset_; size_ = other.size_;
  }
  return *this;
}
WriteLease::~WriteLease() { reset(); }
void WriteLease::reset() noexcept {
  if (auto *block = std::exchange(block_, nullptr)) block->decref();
  offset_ = size_ = 0;
}
std::uint64_t WriteLease::generation() const noexcept { return block_ ? block_->generation : 0; }
Result<mem::BufferView> WriteLease::view() const noexcept {
  if (!block_) return std::unexpected(Status::FailedPrecondition("lease is empty"));
  std::lock_guard guard(block_->domain->lock);
  if (block_->writes != 0)
    return std::unexpected(Status::Unavailable("lease has an admitted writer"));
  if (!block_->failure.ok()) return std::unexpected(block_->failure);
  return detail::lease_view(block_, offset_, size_);
}
Result<TensorView> WriteLease::tensor(TensorDesc description) const noexcept {
  HYLE_ASSIGN_OR_RETURN(auto storage, view());
  return TensorView::create(description, storage);
}
Result<ReadLease> WriteLease::publish() noexcept {
  if (!block_) return std::unexpected(Status::FailedPrecondition("lease is empty"));
  std::lock_guard guard(block_->domain->lock);
  if (!block_->is_unique())
    return std::unexpected(Status::Unavailable("lease still has admitted uses"));
  if (!block_->failure.ok()) return std::unexpected(block_->failure);
  block_->writer = false;
  return ReadLease(std::exchange(block_, nullptr), offset_, size_);
}

Result<LeasePool> LeasePool::create(mem::MemoryResourceRef resource, std::size_t capacity,
                                  LeasePoolOptions options) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto *state, detail::prepare_lease_domain(resource, capacity,
      options.backing_alignment, options));
  LeasePool result(state);
  HYLE_ASSIGN_OR_RETURN(state->backing, mem::Buffer::create(resource, capacity, options.backing_alignment));
  return result;
}
Result<LeasePool> LeasePool::adopt(mem::Buffer &&backing, LeasePoolOptions options) noexcept {
  if (!backing.valid()) return std::unexpected(Status::InvalidArgument("lease backing is empty"));
  HYLE_ASSIGN_OR_RETURN(auto *state, detail::prepare_lease_domain(backing.resource(), backing.size(),
      backing.alignment(), options));
  state->backing = std::move(backing);
  return LeasePool(state);
}
LeasePool::LeasePool(LeasePool &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
LeasePool &LeasePool::operator=(LeasePool &&other) noexcept {
  if (this != &other) {
    if (state_) { close().IgnoreError(); detail::release_domain(state_); }
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
LeasePool::~LeasePool() { if (state_) { close().IgnoreError(); detail::release_domain(state_); } }
Result<WriteLease> LeasePool::allocate(std::size_t bytes, std::size_t alignment) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("lease pool is empty"));
  if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      alignment > state_->backing.alignment())
    return std::unexpected(Status::InvalidArgument("invalid lease size/alignment"));
  std::lock_guard guard(state_->lock);
  if (state_->closed) return std::unexpected(Status::FailedPrecondition("lease pool is closed"));
  if (state_->next_generation == UINT64_MAX)
    return std::unexpected(Status::OutOfRange("lease generation exhausted"));
  auto *storage = state_->blocks.allocate();
  if (!storage) return std::unexpected(Status::ResourceExhausted("lease block capacity"));
  auto range = state_->ranges.allocate(bytes, alignment);
  if (!range) {
    state_->blocks.deallocate(storage);
    return std::unexpected(std::move(range).error());
  }
  state_->incref();
  auto *block = std::construct_at(static_cast<detail::LeaseBlock *>(storage), state_,
                                 std::move(*range), state_->next_generation++);
  return WriteLease(block, 0, bytes);
}
LeasePoolStats LeasePool::stats() const noexcept {
  if (!state_) return {};
  std::lock_guard guard(state_->lock);
  const auto ranges = state_->ranges.stats();
  return {static_cast<std::size_t>(ranges.capacity_bytes), static_cast<std::size_t>(ranges.allocated_bytes),
          static_cast<std::size_t>(ranges.peak_allocated_bytes), ranges.live_allocations,
          ranges.max_allocations, state_->uses.load(std::memory_order_acquire), state_->reclaimed, state_->closed};
}
Status LeasePool::close() noexcept {
  if (!state_) return Status::Ok();
  std::lock_guard guard(state_->lock);
  state_->closed = true;
  return state_->uses.load(std::memory_order_acquire) != 0
      ? Status::Unavailable("lease pool still has admitted native uses") : Status::Ok();
}
Status LeasePool::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError();
  detail::drain_counter(state_->uses);
  // Include the retiring callback's final domain access before a caller may
  // drop this pool and then its borrowed resource owner.
  std::lock_guard guard(state_->lock);
  return Status::Ok();
}

Result<LeaseUsePool> LeaseUsePool::create(LeaseUsePoolOptions options) noexcept {
  if (!options.allocator.valid() || options.max_batches == 0 || options.max_batches == UINT32_MAX ||
      options.max_resources_per_batch == 0 || options.max_resources_per_batch >
          (SIZE_MAX - detail::kEntryOffset) / sizeof(detail::LeaseUseEntry))
    return std::unexpected(Status::InvalidArgument("invalid lease use capacities"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *storage = allocator.allocate(sizeof(detail::LeaseUsePoolState), alignof(detail::LeaseUsePoolState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("lease use pool metadata"));
  const auto bytes = detail::kEntryOffset + options.max_resources_per_batch * sizeof(detail::LeaseUseEntry);
  auto *state = std::construct_at(static_cast<detail::LeaseUsePoolState *>(storage), allocator,
      bytes, options.max_batches, options.max_resources_per_batch);
  if (!state->episodes.valid()) {
    state->decref();
    return std::unexpected(Status::ResourceExhausted("lease use episode metadata"));
  }
  return LeaseUsePool(state);
}
LeaseUsePool::LeaseUsePool(LeaseUsePool &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
LeaseUsePool &LeaseUsePool::operator=(LeaseUsePool &&other) noexcept {
  if (this != &other) {
    if (state_) { close().IgnoreError(); detail::release_use_pool(state_); }
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
LeaseUsePool::~LeaseUsePool() { if (state_) { close().IgnoreError(); detail::release_use_pool(state_); } }
Result<LeaseUseBatch> LeaseUsePool::begin() noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("lease use pool is empty"));
  std::lock_guard guard(state_->lock);
  if (state_->closed) return std::unexpected(Status::FailedPrecondition("lease use pool is closed"));
  auto *storage = state_->episodes.allocate();
  if (!storage) return std::unexpected(Status::ResourceExhausted("lease use episode capacity"));
  auto *episode = std::construct_at(static_cast<detail::LeaseUseEpisode *>(storage));
  episode->pool = state_;
  episode->retention.own = {episode, &detail::episode_terminal, &detail::release_episode};
  state_->incref();
  state_->active.fetch_add(1, std::memory_order_relaxed);
  return LeaseUseBatch(episode);
}
std::uint32_t LeaseUsePool::active_batches() const noexcept {
  return state_ ? state_->active.load(std::memory_order_acquire) : 0;
}
Status LeaseUsePool::close() noexcept {
  if (!state_) return Status::Ok();
  std::lock_guard guard(state_->lock);
  state_->closed = true;
  return state_->active.load(std::memory_order_acquire) != 0
      ? Status::Unavailable("lease use pool still has live episodes") : Status::Ok();
}
Status LeaseUsePool::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError(); detail::drain_counter(state_->active);
  std::lock_guard guard(state_->lock);
  return Status::Ok();
}
LeaseUseBatch::LeaseUseBatch(LeaseUseBatch &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
LeaseUseBatch &LeaseUseBatch::operator=(LeaseUseBatch &&other) noexcept {
  if (this != &other) {
    if (state_) detail::release_episode(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
LeaseUseBatch::~LeaseUseBatch() { if (state_) detail::release_episode(state_); }
Status LeaseUseBatch::read(const ReadLease &lease) noexcept { return detail::add_use(state_, lease.block_, false); }
Status LeaseUseBatch::write(const WriteLease &lease) noexcept { return detail::add_use(state_, lease.block_, true); }
Status LeaseUseBatch::attach(Completion &completion) noexcept {
  if (!state_) return Status::FailedPrecondition("lease use is empty");
  const auto result = detail::CompletionAccess::attach_retention(completion, state_->retention);
  if (result.ok()) state_ = nullptr;
  return result;
}
} // namespace hyle::compute
