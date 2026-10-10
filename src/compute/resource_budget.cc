#include "hyle/compute/resource_budget.h"

#include <atomic>
#include <memory>
#include <utility>

#include "hyle/util/ref_counted.h"

namespace hyle::compute {
namespace detail {
struct ResourceBudgetRow final {
  ResourceBudgetRow(ResourceBudgetState *parent, ResourceBudgetLimit limit,
                    mem::AllocatorRef allocator) noexcept
      : owner(parent), resource(limit.resource), properties(limit.resource.properties()),
        quota(allocator, limit.max_bytes) {}
  ResourceBudgetState *owner;
  mem::MemoryResourceRef resource;
  mem::MemoryProperties properties;
  mem::QuotaAllocator quota;
  std::array<std::atomic<std::size_t>, kResourceBudgetCategoryCount> categories{};
};

struct ResourceBudgetState final : util::RefCounted<ResourceBudgetState> {
  explicit ResourceBudgetState(ResourceBudgetOptions options) noexcept
      : allocator(options.allocator), opaque(options.allocator),
        allow_unattributed(options.allow_unattributed) {}
  mem::AllocatorRef allocator;
  mem::QuotaAllocator opaque;
  mem::QuotaAllocator::Reservation control_charge;
  ResourceBudgetRow *rows = nullptr;
  ResourceBudgetRow *metadata = nullptr;
  std::size_t count = 0, allocation_bytes = 0, unassigned_control = 0;
  std::atomic<std::size_t> unknown{0};
  bool allow_unattributed;
  ResourceBudgetRow *find(mem::MemoryResourceRef resource) const noexcept {
    for (std::size_t i = 0; i < count; ++i)
      if (rows[i].resource == resource) return rows + i;
    return nullptr;
  }
  void destroy() noexcept {
    const auto backing = allocator;
    const auto bytes = allocation_bytes;
    if (metadata != nullptr)
      metadata->categories[0].fetch_sub(control_charge.size(), std::memory_order_relaxed);
    control_charge.reset();
    for (std::size_t i = 0; i < count; ++i) std::destroy_at(rows + i);
    std::destroy_at(this);
    backing.deallocate(this, bytes, alignof(ResourceBudgetState));
  }
};
static_assert(alignof(ResourceBudgetState) >= alignof(ResourceBudgetRow));
static_assert(sizeof(ResourceBudgetState) % alignof(ResourceBudgetRow) == 0);
} // namespace detail
namespace {
void *allocate(void *opaque, std::size_t bytes, std::size_t alignment) noexcept {
  auto &row = *static_cast<detail::ResourceBudgetRow *>(opaque);
  auto *p = row.quota.get().allocate(bytes, alignment);
  if (p != nullptr) {
    row.categories[0].fetch_add(bytes, std::memory_order_relaxed);
    row.owner->incref();
  }
  return p;
}
void deallocate(void *opaque, void *p, std::size_t bytes,
                std::size_t alignment) noexcept {
  if (p == nullptr) return;
  auto &row = *static_cast<detail::ResourceBudgetRow *>(opaque);
  auto *owner = row.owner;
  row.quota.get().deallocate(p, bytes, alignment);
  row.categories[0].fetch_sub(bytes, std::memory_order_relaxed);
  owner->decref();
}
void *reallocate(void *opaque, void *p, std::size_t old_bytes,
                 std::size_t bytes, std::size_t alignment) noexcept {
  if (p == nullptr) return allocate(opaque, bytes, alignment);
  auto &row = *static_cast<detail::ResourceBudgetRow *>(opaque);
  auto *owner = row.owner;
  auto *result = row.quota.get().reallocate(p, old_bytes, bytes, alignment);
  if (result != nullptr || bytes == 0) {
    if (bytes >= old_bytes)
      row.categories[0].fetch_add(bytes - old_bytes, std::memory_order_relaxed);
    else
      row.categories[0].fetch_sub(old_bytes - bytes, std::memory_order_relaxed);
  }
  if (bytes == 0) owner->decref();
  return result;
}
} // namespace

ResourceBudget::Reservation::Reservation(detail::ResourceBudgetState *state,
    mem::QuotaAllocator::Reservation charge, bool unknown, std::size_t row,
    ResourceBudgetCategory category) noexcept
    : state_(state), charge_(std::move(charge)), unknown_(unknown), row_(row),
      category_(category) {
  state_->incref();
  if (unknown_) state_->unknown.fetch_add(1, std::memory_order_relaxed);
  if (row_ != SIZE_MAX)
    state_->rows[row_].categories[static_cast<std::size_t>(category_)].fetch_add(
        charge_.size(), std::memory_order_relaxed);
}
ResourceBudget::Reservation::Reservation(Reservation &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)), charge_(std::move(other.charge_)),
      unknown_(other.unknown_), row_(other.row_), category_(other.category_) {}
ResourceBudget::Reservation &ResourceBudget::Reservation::operator=(Reservation &&other) noexcept {
  if (this != &other) {
    reset(); state_ = std::exchange(other.state_, nullptr);
    charge_ = std::move(other.charge_); unknown_ = other.unknown_;
    row_ = other.row_; category_ = other.category_;
  }
  return *this;
}
ResourceBudget::Reservation::~Reservation() { reset(); }
void ResourceBudget::Reservation::reset() noexcept {
  auto *owner = std::exchange(state_, nullptr);
  if (owner == nullptr) return;
  if (row_ != SIZE_MAX)
    owner->rows[row_].categories[static_cast<std::size_t>(category_)].fetch_sub(
        charge_.size(), std::memory_order_relaxed);
  charge_.reset();
  if (unknown_) owner->unknown.fetch_sub(1, std::memory_order_relaxed);
  owner->decref();
}
ResourceBudget::ResourceBudget(const ResourceBudget &other) noexcept : state_(other.state_) {
  if (state_ != nullptr) state_->incref();
}
ResourceBudget &ResourceBudget::operator=(const ResourceBudget &other) noexcept {
  if (this != &other) {
    if (other.state_ != nullptr) other.state_->incref();
    if (state_ != nullptr) state_->decref();
    state_ = other.state_;
  }
  return *this;
}
ResourceBudget::ResourceBudget(ResourceBudget &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
ResourceBudget &ResourceBudget::operator=(ResourceBudget &&other) noexcept {
  if (this != &other) {
    if (state_ != nullptr) state_->decref();
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
ResourceBudget::~ResourceBudget() { if (state_ != nullptr) state_->decref(); }
Result<ResourceBudget> ResourceBudget::create(ResourceBudgetOptions options) noexcept {
  if (!options.allocator.valid() || options.limits.empty() ||
      !options.metadata_resource.valid() ||
      !options.metadata_resource.properties().has(mem::MemoryProperty::kHostAccessible))
    return std::unexpected(Status::InvalidArgument("invalid resource budget"));
  for (std::size_t i = 0; i < options.limits.size(); ++i) {
    if (!options.limits[i].resource.valid())
      return std::unexpected(Status::InvalidArgument("invalid budget resource"));
    for (std::size_t j = 0; j < i; ++j)
      if (options.limits[i].resource == options.limits[j].resource)
        return std::unexpected(Status::InvalidArgument("duplicate budget resource"));
  }
  if (options.limits.size() > (SIZE_MAX - sizeof(detail::ResourceBudgetState)) /
                                 sizeof(detail::ResourceBudgetRow))
    return std::unexpected(Status::ResourceExhausted("resource budget size overflows"));
  const auto bytes = sizeof(detail::ResourceBudgetState) +
                     options.limits.size() * sizeof(detail::ResourceBudgetRow);
  auto *memory = options.allocator.allocate(bytes, alignof(detail::ResourceBudgetState));
  if (memory == nullptr)
    return std::unexpected(Status::ResourceExhausted("resource budget metadata"));
  auto *state = std::construct_at(static_cast<detail::ResourceBudgetState *>(memory), options);
  state->allocation_bytes = bytes;
  state->rows = reinterpret_cast<detail::ResourceBudgetRow *>(state + 1);
  for (auto limit : options.limits) {
    std::construct_at(state->rows + state->count, state, limit, options.allocator);
    ++state->count;
  }
  state->metadata = state->find(options.metadata_resource);
  if (state->metadata == nullptr) {
    state->decref();
    return std::unexpected(Status::InvalidArgument("metadata resource is absent from budget"));
  }
  auto control = state->metadata->quota.try_reserve_bytes(bytes);
  if (!control) {
    state->decref();
    return std::unexpected(Status::ResourceExhausted("resource budget cannot hold control state"));
  }
  state->control_charge = std::move(*control);
  state->metadata->categories[0].fetch_add(bytes, std::memory_order_relaxed);
  return ResourceBudget(state);
}
std::size_t ResourceBudget::resource_count() const noexcept { return state_ ? state_->count : 0; }
Result<mem::MemoryResourceRef> ResourceBudget::resource(std::size_t index) const noexcept {
  if (!state_ || index >= state_->count)
    return std::unexpected(Status::OutOfRange("resource budget index"));
  return state_->rows[index].resource;
}
Result<ResourceBudgetUsage> ResourceBudget::usage(mem::MemoryResourceRef resource) const noexcept {
  auto *row = state_ ? state_->find(resource) : nullptr;
  if (row == nullptr)
    return std::unexpected(Status::InvalidArgument("resource is absent from budget"));
  ResourceBudgetUsage result{row->properties, row->quota.used(), row->quota.peak(), row->quota.limit()};
  for (std::size_t i = 0; i < result.live_by_category.size(); ++i)
    result.live_by_category[i] = row->categories[i].load(std::memory_order_relaxed);
  return result;
}
UnattributedStorageUsage ResourceBudget::unattributed_usage() const noexcept {
  return state_ ? UnattributedStorageUsage{state_->opaque.used(),
      state_->unknown.load(std::memory_order_relaxed), state_->unassigned_control}
      : UnattributedStorageUsage{};
}
bool ResourceBudget::allows_unattributed() const noexcept {
  return state_ && state_->allow_unattributed;
}
Result<ResourceBudget::Reservation> ResourceBudget::reserve(
    mem::MemoryResourceRef resource, std::size_t bytes,
    ResourceBudgetCategory category) const noexcept {
  auto *row = state_ ? state_->find(resource) : nullptr;
  if (row == nullptr || static_cast<std::size_t>(category) >= kResourceBudgetCategoryCount)
    return std::unexpected(Status::InvalidArgument("resource is absent from budget"));
  auto charge = row->quota.try_reserve_bytes(bytes);
  if (!charge)
    return std::unexpected(Status::ResourceExhausted("shared resource budget is exhausted"));
  return Reservation(state_, std::move(*charge), false,
      static_cast<std::size_t>(row - state_->rows), category);
}
Result<ResourceBudget::Reservation> ResourceBudget::reserve_unattributed(
    std::optional<std::size_t> bytes) const noexcept {
  if (bytes && *bytes == 0) return Reservation{};
  if (!state_ || !state_->allow_unattributed)
    return std::unexpected(Status::ResourceExhausted("private storage has no budget domain"));
  auto charge = state_->opaque.try_reserve_bytes(bytes.value_or(0));
  if (!charge)
    return std::unexpected(Status::ResourceExhausted("private storage sum overflows"));
  return Reservation(state_, std::move(*charge), !bytes, SIZE_MAX,
                     ResourceBudgetCategory::kExternal);
}
Result<mem::AllocatorRef> ResourceBudget::metadata_allocator() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("invalid resource budget"));
  return mem::AllocatorRef{allocate, deallocate, reallocate, state_->metadata};
}
} // namespace hyle::compute
