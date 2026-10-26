#include "hyle/compute/resource_leases.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/resource_budget_access.h"
#include "hyle/core/assert.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/range_allocator.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace hyle::compute {
namespace detail {

struct LeaseDomain;
const unsigned retained_domain_authority = 0;
const unsigned retained_host_address_authority = 0;
struct RetainedBacking final {
  mem::MemoryResourceRef resource;
  SharedBackingReservation charge;
  RetiringBackingReservation retiring;
};
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
  std::uint32_t native_uses = 0;
  bool writer = true;
  bool writable = true;
  // One opt-in tail, not a general range permission map. Allocation generation
  // and committed readers survive an independently failed append epoch.
  bool append_mode = false;
  bool append_owner = false;
  std::size_t append_committed = 0;
  std::size_t append_reserved = 0;
  std::uint64_t append_epoch = 0;
  Status append_failure{};
};

struct LeaseDomain final : util::RefCounted<LeaseDomain> {
  LeaseDomain(mem::AllocatorRef selected, std::uint32_t count) noexcept
      : allocator(selected), retained(selected),
        blocks(selected, sizeof(LeaseBlock), count, alignof(LeaseBlock)) {}
  void destroy() noexcept {
    const auto selected = allocator;
    const auto retained_owner = resource_owner;
    const auto release_owner = release_resource_owner;
    // Keep only the actual allocator alive past payload-owner release. This
    // is independent of that owner, which can own the real native payload.
    auto allocator_guard = std::move(allocator_owner);
    auto retiring = std::move(retained);
    auto direct_charge = std::move(charge);
    for (auto &entry : retiring)
      entry.retiring = ResourceBudgetAccess::retire_backing(entry.charge);
    this->~LeaseDomain();
    selected.deallocate(this, sizeof(LeaseDomain), alignof(LeaseDomain));
    if (release_owner) release_owner(retained_owner);
    // Direct/deferred quotas and spill metadata are released last, outside
    // the lookup/domain locks, while allocator_guard still owns their heap.
  }
  mem::AllocatorRef allocator;
  void *resource_owner = nullptr;
  LeaseAccess::ReleaseOwnerFn release_resource_owner = nullptr;
  std::shared_ptr<void> allocator_owner;
  // Teardown first withdraws lookup identity, then frees actual backing and
  // provider ownership, and only then releases its original quota charges.
  mem::Buffer backing{};
  ResourceBudget::Reservation charge{};
  container::SmallVector<RetainedBacking, 1, mem::AllocatorRef> retained;
  BackingIdentity retained_identity{};
  std::size_t retained_bytes = 0, retained_offset = 0;
  mem::RangeAllocator ranges{};
  mem::ConcurrentBlockPool blocks;
  util::SpinLock lock{};
  std::atomic<std::uint64_t> uses{0};
  std::uint64_t next_generation = 1, reclaimed = 0;
  bool closed = false;
};

Result<LeaseBackingInfo> LeaseAccess::backing(const ReadLease &lease) noexcept {
  if (!lease.block_) return std::unexpected(Status::FailedPrecondition("lease is empty"));
  const auto *domain = lease.block_->domain;
  return LeaseBackingInfo{domain, domain->backing.resource(), domain->backing.size()};
}
Result<LeaseBackingInfo> LeaseAccess::backing(const WriteLease &lease) noexcept {
  if (!lease.block_) return std::unexpected(Status::FailedPrecondition("lease is empty"));
  const auto *domain = lease.block_->domain;
  return LeaseBackingInfo{domain, domain->backing.resource(), domain->backing.size()};
}
Status LeaseAccess::set_retained_backing(ReadLease &lease,
    mem::MemoryResourceRef nominated, const void *authority,
    const void *canonical_owner, std::size_t full_bytes,
    std::size_t backing_offset) noexcept {
  if (!lease.valid() || !nominated.valid() || !authority || !canonical_owner ||
      !full_bytes || full_bytes == SIZE_MAX)
    return Status::InvalidArgument("retained backing identity/resource/capacity is empty");
  auto &domain = *lease.block_->domain;
  std::lock_guard guard(domain.lock);
  if (domain.backing.resource() != nominated || backing_offset > full_bytes ||
      domain.backing.size() > full_bytes - backing_offset)
    return Status::InvalidArgument("retained backing wrapper is outside its original allocation");
  if (domain.retained_identity.value) {
    if (domain.retained_identity.authority == authority &&
        domain.retained_identity.value == canonical_owner &&
        domain.retained_bytes == full_bytes && domain.retained_offset == backing_offset)
      return Status::Ok();
    return Status::FailedPrecondition("retained backing identity is already established");
  }
  if (!domain.retained.empty())
    return Status::FailedPrecondition("cannot change an already-admitted backing identity");
  domain.retained_identity = {authority, canonical_owner};
  domain.retained_bytes = full_bytes;
  domain.retained_offset = backing_offset;
  return Status::Ok();
}
static Result<BackingIdentity> original_backing_identity(LeaseDomain &domain) noexcept {
  // An opaque provider token is not a cross-wrapper physical identity. Only
  // the explicit stable-host-address contract permits merging independent
  // domains here; mapping grants no new public permission to the full buffer.
  BackingIdentity identity{&retained_domain_authority, &domain};
  const auto actual = domain.backing.resource();
  if (domain.backing.size() && actual.properties().has(mem::MemoryProperty::kHostAddressIdentity)) {
    HYLE_ASSIGN_OR_RETURN(auto mapped, actual.resource().map_host(
        domain.backing.native_handle(), 0, domain.backing.size(), mem::HostAccess::kRead));
    identity = {&retained_host_address_authority, mapped.data};
    HYLE_RETURN_IF_ERROR(actual.resource().unmap_host(domain.backing.native_handle(), mapped.token,
        0, domain.backing.size(), mem::HostAccess::kRead));
    if (!identity.value) return std::unexpected(Status::Internal("retained backing has no promised canonical address"));
  }
  return identity;
}
Status LeaseAccess::retain_budget(ReadLease &lease, mem::MemoryResourceRef nominated,
                                 const ResourceBudget &budget) noexcept {
  if (!budget.valid()) return Status::Ok();
  if (!lease.valid() || !nominated.valid())
    return Status::InvalidArgument("retained result budget requires an owned lease and admitted resource");
  auto &domain = *lease.block_->domain;
  {
    std::lock_guard guard(domain.lock);
    for (const auto &retained : domain.retained)
      if (retained.resource == nominated && retained.charge.same_scope(budget)) return Status::Ok();
  }
  HYLE_ASSIGN_OR_RETURN(auto identity, original_backing_identity(domain));
  std::lock_guard guard(domain.lock);
  for (const auto &retained : domain.retained)
    if (retained.resource == nominated && retained.charge.same_scope(budget)) return Status::Ok();
  if (!domain.retained.try_reserve(domain.retained.size() + 1))
    return Status::ResourceExhausted("retained result budget metadata");
  // Read the annotation under the same lock as admission: a concurrent first
  // export cannot install a different key between proof and token publication.
  const auto bytes = domain.retained_identity.value ? domain.retained_bytes : domain.backing.size();
  if (domain.retained_identity.value) identity = domain.retained_identity;
  SharedBackingReservation retained;
  if (bytes == domain.backing.size() && domain.backing.resource() == nominated &&
      ResourceBudgetAccess::same_scope(domain.charge, budget)) {
    HYLE_ASSIGN_OR_RETURN(retained, ResourceBudgetAccess::adopt_backing(
        budget, nominated, identity, bytes, domain.charge));
  } else {
    HYLE_ASSIGN_OR_RETURN(retained, ResourceBudgetAccess::reserve_backing(
        budget, nominated, identity, bytes));
  }
  HYLE_FATAL_ASSERT(domain.retained.try_push_back(RetainedBacking{nominated, std::move(retained), {}}));
  return Status::Ok();
}
mem::AllocatorRef LeaseAccess::metadata_backing_allocator(
    const ReadLease &lease) noexcept {
  if (!lease.block_) return {};
  const auto *domain = lease.block_->domain;
  // The allocator is stable and its actual Domain allocation retains every
  // known budget scope through this cold ownership query. Do not depend on
  // the primary payload charge, which can already have been adopted.
  return ResourceBudgetAccess::original_allocator(domain->allocator);
}
Result<mem::AllocatorRef> LeaseAccess::freeze_owner_allocator(
    const LeasePool &pool, const WriteLease &writer) noexcept {
  if (!pool.state_ || !writer.block_ || writer.block_->domain != pool.state_)
    return std::unexpected(Status::InvalidArgument(
        "storage owner freeze requires the original pool writer"));
  auto &domain = *pool.state_;
  std::lock_guard guard(domain.lock);
  if (domain.resource_owner || domain.release_resource_owner || domain.allocator_owner)
    return std::unexpected(Status::FailedPrecondition(
        "storage owner freeze cannot replace an existing lifetime owner"));
  return ResourceBudgetAccess::original_allocator(domain.allocator);
}
bool LeaseAccess::unique_read_owner(const ReadLease &lease) noexcept {
  if (!lease.block_) return false;
  std::lock_guard guard(lease.block_->domain->lock);
  return lease.block_->is_unique();
}

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
  std::uint64_t append_epoch = 0;
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
    if (entry.append_epoch != 0) {
      std::lock_guard guard(entry.block->domain->lock);
      HYLE_FATAL_ASSERT(entry.block->append_epoch == entry.append_epoch);
      entry.block->append_failure.Update(terminal);
    } else if (entry.write) {
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
      HYLE_FATAL_ASSERT(block->native_uses != 0);
      --block->native_uses;
      if (entry.write) {
        HYLE_FATAL_ASSERT(block->writes != 0);
        --block->writes;
      }
      if (block->append_mode && !block->append_owner && block->native_uses == 0) {
        // Dropping the tail authority cannot retire accepted work. Only this
        // original native-use release may finish its abandoned reservation.
        block->append_reserved = 0;
        block->append_failure = Status::Ok();
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

template<bool Unique = false, bool Ordered = false, bool ExclusiveRead = false>
Status add_use(LeaseUseEpisode *episode, LeaseBlock *block, bool write) noexcept {
  if (episode == nullptr || block == nullptr)
    return Status::FailedPrecondition("resource use or lease is empty");
  if constexpr (!Unique) {
    for (std::uint32_t index = 0; index < episode->count; ++index) {
      const auto &entry = entries(episode)[index];
      if (entry.block == block)
        return entry.append_epoch == 0 && entry.write == write ? Status::Ok()
                                   : Status::InvalidArgument("one use cannot mix lease permissions");
    }
  }
  if (episode->count == episode->pool->capacity)
    return Status::ResourceExhausted("resource use metadata capacity exhausted");
  auto *domain = block->domain;
  {
    std::lock_guard guard(domain->lock);
    if (domain->closed) return Status::FailedPrecondition("lease domain is closed");
    if (block->append_reserved != 0)
      return Status::Unavailable("append suffix is reserved for its ordered use");
    if (block->writer != (write || ExclusiveRead))
      return Status::FailedPrecondition("resource use does not match lease permission");
    if (!block->failure.ok()) return block->failure;
    if (block->native_uses == UINT32_MAX)
      return Status::ResourceExhausted("resource native use count exhausted");
    if constexpr (!Ordered) {
      if (write && block->writes != 0)
        return Status::Unavailable("resource already has an admitted writer");
    }
    if constexpr (Ordered) {
      if (write && block->writes == UINT32_MAX)
        return Status::ResourceExhausted("resource writer use count exhausted");
    }
    if (write) ++block->writes;
    ++block->native_uses;
    block->incref();
    domain->uses.fetch_add(1, std::memory_order_relaxed);
  }
  std::construct_at(entries(episode) + episode->count, block, write, std::uint64_t{0});
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

Result<AppendLeasePublication> LeaseAccess::publish_appendable(
    WriteLease &lease, std::size_t prefix_bytes) noexcept {
  if (!lease.block_)
    return std::unexpected(Status::FailedPrecondition("appendable writer is empty"));
  auto *block = lease.block_;
  std::lock_guard guard(block->domain->lock);
  if (block->domain->closed || !block->writer || block->append_mode)
    return std::unexpected(Status::FailedPrecondition("appendable writer is not fresh"));
  if (lease.offset_ != 0 || lease.size_ != block->range.size() || prefix_bytes > lease.size_)
    return std::unexpected(Status::InvalidArgument("appendable publication requires the original full block"));
  if (!block->is_unique() || block->native_uses != 0)
    return std::unexpected(Status::Unavailable("appendable writer still has admitted uses"));
  if (!block->failure.ok()) return std::unexpected(block->failure);
  const bool has_tail = prefix_bytes != lease.size_;
  block->append_mode = true;
  block->append_owner = has_tail;
  block->append_committed = prefix_bytes;
  block->writer = false;
  block->writable = false;
  if (has_tail) block->incref();
  ReadLease prefix(std::exchange(lease.block_, nullptr), 0, prefix_bytes);
  return AppendLeasePublication{std::move(prefix), has_tail ? AppendLease(block) : AppendLease{}};
}

bool LeaseAccess::is_appendable_prefix(const ReadLease &lease) noexcept {
  if (!lease.block_) return false;
  const auto *block = lease.block_;
  std::lock_guard guard(block->domain->lock);
  return block->append_mode && lease.offset_ <= block->append_committed &&
      lease.size_ <= block->append_committed - lease.offset_;
}

Status LeaseAccess::validate_prefix(const AppendLease &appender,
                                   const ReadLease &prefix) noexcept {
  if (!appender.block_ || prefix.block_ != appender.block_)
    return Status::FailedPrecondition("append prefix has a different allocation");
  const auto *block = appender.block_;
  std::lock_guard guard(block->domain->lock);
  if (!block->append_mode || !block->append_owner ||
      prefix.offset_ > block->append_committed ||
      prefix.size_ > block->append_committed - prefix.offset_)
    return Status::FailedPrecondition("read lease exceeds the committed append prefix");
  return Status::Ok();
}

Status LeaseAccess::append_unique(LeaseUseBatch &uses, const AppendLease &lease) noexcept {
  auto *episode = uses.state_;
  auto *block = lease.block_;
  if (!episode || !block)
    return Status::FailedPrecondition("resource use or append lease is empty");
  if (episode->count == episode->pool->capacity)
    return Status::ResourceExhausted("resource use metadata capacity exhausted");
  // Like read_unique/write_unique this entry is already root-deduplicated by
  // the Graph. native_uses below also rejects any second admission of this root.
  auto *domain = block->domain;
  std::uint64_t epoch;
  {
    std::lock_guard guard(domain->lock);
    if (domain->closed || !block->append_owner || block->append_reserved == 0)
      return Status::FailedPrecondition("append lease has no open reservation");
    if (!block->append_failure.ok()) return block->append_failure;
    if (block->native_uses != 0)
      return Status::Unavailable("append lease still has admitted native uses");
    epoch = block->append_epoch;
    ++block->native_uses;
    block->incref();
    domain->uses.fetch_add(1, std::memory_order_relaxed);
  }
  std::construct_at(entries(episode) + episode->count, block, false, epoch);
  ++episode->count;
  return Status::Ok();
}

AppendLease::AppendLease(AppendLease &&other) noexcept
    : block_(std::exchange(other.block_, nullptr)) {}
AppendLease &AppendLease::operator=(AppendLease &&other) noexcept {
  if (this != &other) {
    reset();
    block_ = std::exchange(other.block_, nullptr);
  }
  return *this;
}
AppendLease::~AppendLease() { reset(); }
void AppendLease::reset() noexcept {
  auto *block = std::exchange(block_, nullptr);
  if (!block) return;
  {
    std::lock_guard guard(block->domain->lock);
    block->append_owner = false;
    if (block->native_uses == 0) {
      block->append_reserved = 0;
      block->append_failure = Status::Ok();
    }
  }
  block->decref();
}
std::uint64_t AppendLease::generation() const noexcept {
  return block_ ? block_->generation : 0;
}
std::uint64_t AppendLease::epoch() const noexcept {
  if (!block_) return 0;
  std::lock_guard guard(block_->domain->lock);
  return block_->append_epoch;
}
std::size_t AppendLease::committed_bytes() const noexcept {
  if (!block_) return 0;
  std::lock_guard guard(block_->domain->lock);
  return block_->append_committed;
}
std::size_t AppendLease::reserved_bytes() const noexcept {
  if (!block_) return 0;
  std::lock_guard guard(block_->domain->lock);
  return block_->append_reserved;
}
std::size_t AppendLease::capacity_bytes() const noexcept {
  if (!block_) return 0;
  return static_cast<std::size_t>(block_->range.size());
}
Status AppendLease::reserve(std::size_t bytes) noexcept {
  if (!block_) return Status::FailedPrecondition("append lease is empty");
  auto *block = block_;
  std::lock_guard guard(block->domain->lock);
  if (block->domain->closed || !block->append_owner)
    return Status::FailedPrecondition("append lease domain is closed");
  if (block->append_reserved != 0 || block->native_uses != 0)
    return Status::Unavailable("append lease has a reservation or admitted native uses");
  const auto capacity = block->range.size();
  if (bytes == 0 || bytes > capacity - block->append_committed)
    return Status::OutOfRange("append reservation exceeds the unused tail");
  if (block->append_epoch == UINT64_MAX)
    return Status::OutOfRange("append epoch exhausted");
  ++block->append_epoch;
  block->append_reserved = bytes;
  block->append_failure = Status::Ok();
  return Status::Ok();
}
Result<mem::BufferView> AppendLease::view() const noexcept {
  if (!block_) return std::unexpected(Status::FailedPrecondition("append lease is empty"));
  const auto *block = block_;
  std::lock_guard guard(block->domain->lock);
  if (block->domain->closed || !block->append_owner || block->append_reserved == 0)
    return std::unexpected(Status::FailedPrecondition("append lease has no open reservation"));
  if (block->native_uses != 0)
    return std::unexpected(Status::Unavailable("append lease has admitted native uses"));
  if (!block->append_failure.ok()) return std::unexpected(block->append_failure);
  return lease_view(block_, block->append_committed, block->append_reserved);
}
Result<ReadLease> LeaseAccess::publish_append(AppendLease &lease) noexcept {
  if (!lease.block_)
    return std::unexpected(Status::FailedPrecondition("append lease is empty"));
  auto *block = lease.block_;
  std::lock_guard guard(block->domain->lock);
  if (block->domain->closed || !block->append_owner || block->append_reserved == 0)
    return std::unexpected(Status::FailedPrecondition("append lease has no open reservation"));
  if (block->native_uses != 0)
    return std::unexpected(Status::Unavailable("append lease still has admitted native uses"));
  if (!block->append_failure.ok()) return std::unexpected(block->append_failure);
  block->append_committed += block->append_reserved;
  block->append_reserved = 0;
  block->incref();
  return ReadLease(block, 0, block->append_committed);
}
Result<ReadLease> AppendLease::publish() noexcept { return LeaseAccess::publish_append(*this); }
Status AppendLease::rollback() noexcept {
  if (!block_) return Status::FailedPrecondition("append lease is empty");
  std::lock_guard guard(block_->domain->lock);
  if (block_->native_uses != 0)
    return Status::Unavailable("append rollback requires actual native retirement");
  block_->append_reserved = 0;
  block_->append_failure = Status::Ok();
  return Status::Ok();
}

Result<mem::BufferView> LeaseAccess::ordered_view(const WriteLease &lease) noexcept {
  if (!lease.block_) return std::unexpected(Status::FailedPrecondition("copy mutable lease is empty"));
  std::lock_guard guard(lease.block_->domain->lock);
  if (lease.block_->domain->closed || !lease.block_->writer)
    return std::unexpected(Status::FailedPrecondition("copy mutable lease domain is closed"));
  if (!lease.block_->failure.ok()) return std::unexpected(lease.block_->failure);
  return lease_view(lease.block_, lease.offset_, lease.size_);
}

Result<LeaseDomain *> prepare_lease_domain(mem::MemoryResourceRef resource,
    std::size_t capacity, std::size_t alignment, LeasePoolOptions options,
    ResourceBudgetCategory category = ResourceBudgetCategory::kExternal) noexcept {
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
    auto charge = options.resource_budget.reserve(resource, capacity, category);
    if (!charge) {
      state->decref();
      return std::unexpected(std::move(charge).error());
    }
    state->charge = std::move(*charge);
  }
  return state;
}
} // namespace detail

void detail::LeaseAccess::retain_allocator_owner(ReadLease &lease,
                                               std::shared_ptr<void> owner) noexcept {
  HYLE_FATAL_ASSERT(lease.block_);
  auto *domain = lease.block_->domain;
  HYLE_FATAL_ASSERT(!domain->allocator_owner);
  domain->allocator_owner = std::move(owner);
}
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

void detail::LeaseAccess::retire_native(LeaseUseBatch &uses, Status terminal) noexcept {
  if (!uses.valid()) return;
  auto retired = Completion::completed(std::move(terminal));
  HYLE_FATAL_ASSERT(uses.attach(retired).ok());
}

void detail::LeaseAccess::detach_pool_root(LeasePool &pool) noexcept {
  HYLE_FATAL_ASSERT(pool.state_ && pool.state_->resource_owner && pool.state_->release_resource_owner);
  detail::release_domain(std::exchange(pool.state_,nullptr));
}
Result<mem::ConstBufferView> detail::LeaseAccess::backing_span(const LeasePool&pool) noexcept {
  if(!pool.state_)return std::unexpected(Status::FailedPrecondition("lease pool is empty"));
  return mem::ConstBufferView(pool.state_->backing.view());
}
void detail::LeaseAccess::retire_host_snapshot_pool(LeasePool &pool, const ReadLease &ready) noexcept {
  if (!pool.state_) return;
  auto *domain = pool.state_;
  HYLE_FATAL_ASSERT(ready.block_ && ready.block_->domain == domain);
  {
    std::lock_guard guard(domain->lock);
    const auto ranges = domain->ranges.stats();
    HYLE_FATAL_ASSERT(!domain->closed && !domain->resource_owner &&
        domain->backing.resource() == mem::system_resource() &&
        ranges.max_allocations == 1 && ranges.live_allocations == 1 &&
        ready.offset_ == 0 && ready.size_ == domain->backing.size() && !ready.block_->writer);
    ready.block_->writable = false;
  }
  detail::release_domain(std::exchange(pool.state_, nullptr));
}

Status detail::LeaseAccess::read_unique(LeaseUseBatch &uses, const ReadLease &lease) noexcept {
  return detail::add_use<true>(uses.state_, lease.block_, false);
}
Status detail::LeaseAccess::ordered_read_unique(LeaseUseBatch &uses, const WriteLease &lease) noexcept {
  return detail::add_use<true, true, true>(uses.state_, lease.block_, false);
}
Status detail::LeaseAccess::write_unique(LeaseUseBatch &uses, const WriteLease &lease,
                                       bool ordered_after_prior) noexcept {
  return ordered_after_prior ? detail::add_use<true, true>(uses.state_, lease.block_, true)
                             : detail::add_use<true>(uses.state_, lease.block_, true);
}
Result<ReadLease> detail::LeaseAccess::publish_after_writes(WriteLease &lease) noexcept {
  if (!lease.block_)
    return std::unexpected(Status::FailedPrecondition("copy output lease is empty"));
  auto *block = lease.block_;
  std::lock_guard guard(block->domain->lock);
  if (block->writes != 0)
    return std::unexpected(Status::Unavailable("copy output still has admitted writers"));
  if (!block->failure.ok()) return std::unexpected(block->failure);
  HYLE_FATAL_ASSERT(block->writer);
  block->writer = false;
  return ReadLease(std::exchange(lease.block_, nullptr), lease.offset_, lease.size_);
}

Result<ReadLease> ReadLease::adopt(mem::Buffer &&backing, LeaseImportOptions options) noexcept {
  return detail::LeaseAccess::adopt_readonly(std::move(backing), std::move(options),
                                           ResourceBudgetCategory::kExternal);
}

Result<ReadLease> detail::LeaseAccess::adopt_metadata(mem::Buffer &&backing,
    LeaseImportOptions options) noexcept {
  return adopt_readonly(std::move(backing), std::move(options), ResourceBudgetCategory::kMetadata);
}

Result<ReadLease> detail::LeaseAccess::adopt_readonly(mem::Buffer &&backing,
    LeaseImportOptions options, ResourceBudgetCategory category) noexcept {
  if (!backing.valid()) return std::unexpected(Status::InvalidArgument("lease backing is empty"));
  const auto bytes = backing.size();
  HYLE_ASSIGN_OR_RETURN(auto *state, detail::prepare_lease_domain(backing.resource(), bytes,
      backing.alignment(), {.max_blocks = 1, .allocator = options.allocator,
                            .resource_budget = options.resource_budget}, category));
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
  if (!block_->is_unique())
    return std::unexpected(Status::Unavailable("lease has admitted native uses"));
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
Result<LeasePool> LeasePool::create(mem::MemoryResourceRef resource, std::size_t capacity,
    LeasePoolOptions options, const ResourceBudget &additional) noexcept {
  if (!additional.valid()) return create(resource, capacity, std::move(options));
  // This pending charge must outlive root on every failure after allocation.
  // Successful adoption moves it into root's original retirement chain.
  ResourceBudget::Reservation secondary;
  HYLE_ASSIGN_OR_RETURN(auto *state, detail::prepare_lease_domain(resource, capacity,
      options.backing_alignment, options));
  LeasePool root(state);
  if (detail::ResourceBudgetAccess::same_scope(state->charge, additional)) {
    HYLE_ASSIGN_OR_RETURN(state->backing, mem::Buffer::create(resource, capacity, options.backing_alignment));
    return root;
  }
  HYLE_ASSIGN_OR_RETURN(secondary, additional.reserve(resource, capacity));
  const auto entries = options.resource_budget.valid() ? 2u : 1u;
  if (!state->retained.try_reserve(entries))
    return std::unexpected(Status::ResourceExhausted("additional backing constraint metadata"));
  HYLE_ASSIGN_OR_RETURN(state->backing, mem::Buffer::create(resource, capacity, options.backing_alignment));
  // A real host allocation must use the same address identity as independent
  // readonly wrappers. Do not publish a provisional Domain key then change it.
  HYLE_ASSIGN_OR_RETURN(auto identity, detail::original_backing_identity(*state));
  const auto attach = [&](const ResourceBudget &budget,
                          ResourceBudget::Reservation &pending) noexcept -> Status {
    HYLE_ASSIGN_OR_RETURN(auto token, detail::ResourceBudgetAccess::adopt_backing(
        budget, resource, identity, capacity, pending));
    // Already-reserved storage makes this immediate publication nonfallible;
    // no local successful token can retire before root's actual Buffer.
    HYLE_FATAL_ASSERT(state->retained.try_push_back(
        detail::RetainedBacking{resource, std::move(token), {}}));
    return Status::Ok();
  };
  if (options.resource_budget.valid()) HYLE_RETURN_IF_ERROR(attach(options.resource_budget, state->charge));
  HYLE_RETURN_IF_ERROR(attach(additional, secondary));
  return root;
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
Result<ReadLease> LeasePool::freeze(WriteLease &&writer) noexcept {
  if (!state_ || !writer.block_ || writer.block_->domain != state_)
    return std::unexpected(Status::InvalidArgument("readonly pool freeze requires its original writer"));
  auto *domain = state_;
  auto *block = writer.block_;
  std::unique_lock guard(domain->lock);
  const auto ranges = domain->ranges.stats();
  if (domain->closed || ranges.live_allocations != 1 || writer.offset_ != 0 ||
      writer.size_ != domain->backing.size() || block->append_mode || !block->writer)
    return std::unexpected(Status::FailedPrecondition("readonly pool freeze requires one full ordinary block"));
  if (!block->is_unique())
    return std::unexpected(Status::Unavailable("readonly pool freeze has admitted uses"));
  if (!block->failure.ok()) return std::unexpected(block->failure);
  block->writer = false;
  block->writable = false;
  ReadLease result(std::exchange(writer.block_, nullptr), writer.offset_, writer.size_);
  state_ = nullptr;
  guard.unlock();
  detail::release_domain(domain);
  return result;
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
      options.max_resources_per_batch == 0 || std::cmp_greater(options.max_resources_per_batch,
          (SIZE_MAX - detail::kEntryOffset) / sizeof(detail::LeaseUseEntry)))
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
