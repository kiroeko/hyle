#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <new>
#include <memory>
#include <optional>
#include <utility>

#include "hyle/graph/executable.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::detail {

inline constexpr std::array kCompiledStorageFields{
    &CompiledStorageReport::materialized_constant_bytes,
    &CompiledStorageReport::provider_persistent_bytes,
    &CompiledStorageReport::packed_constant_bytes,
    &CompiledStorageReport::unknown_provider_count};

inline Status subtract_compiled_storage(CompiledStorageReport &total,
                                        CompiledStorageReport shared) noexcept {
  for (const auto field : kCompiledStorageFields)
    if (shared.*field > total.*field)
      return Status::Internal("graph shared storage exceeds route storage");
  for (const auto field : kCompiledStorageFields)
    total.*field -= shared.*field;
  return Status::Ok();
}

inline Status add_compiled_storage(CompiledStorageReport &total,
                                   CompiledStorageReport part,
                                   bool shared_constants = false,
                                   CompiledStorageReport shared = {}) noexcept {
  if (shared_constants)
    part.materialized_constant_bytes = 0;
  HYLE_RETURN_IF_ERROR(subtract_compiled_storage(part, shared));
  for (const auto field : kCompiledStorageFields) {
    if (part.*field > SIZE_MAX - total.*field)
      return Status::ResourceExhausted("graph compiled storage sum overflows");
    total.*field += part.*field;
  }
  if (total.provider_persistent_bytes >
          SIZE_MAX - total.materialized_constant_bytes ||
      total.packed_constant_bytes > SIZE_MAX -
                                        total.materialized_constant_bytes -
                                        total.provider_persistent_bytes)
    return Status::ResourceExhausted("graph compiled storage sum overflows");
  return Status::Ok();
}

// Graph owns only family/lifetime composition. Mem owns the strict byte
// admission algorithm. Each successful metadata allocation retains this
// allocator owner, so independently retained signatures need no Plan cycle.
struct StorageChargeState final {
  mem::AllocatorRef allocator;
  mem::QuotaAllocator::Reservation aggregate;
  compute::ResourceBudget::Reservation resource;
};

// Only budgeted compilation allocates a charge record. Ordinary packed
// artifacts keep one empty pointer rather than paying for all quota tokens.
class StorageCharge final {
public:
  StorageCharge() noexcept = default;
  explicit StorageCharge(StorageChargeState *state) noexcept : state_(state) {}
  StorageCharge(const StorageCharge &) = delete;
  StorageCharge &operator=(const StorageCharge &) = delete;
  StorageCharge(StorageCharge &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
  StorageCharge &operator=(StorageCharge &&other) noexcept {
    if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); }
    return *this;
  }
  ~StorageCharge() { reset(); }
private:
  void reset() noexcept {
    auto *state = std::exchange(state_, nullptr);
    if (state == nullptr) return;
    const auto owner = state->allocator;
    std::destroy_at(state);
    owner.deallocate(state, sizeof(StorageChargeState), alignof(StorageChargeState));
  }
  StorageChargeState *state_ = nullptr;
};

class StorageBudgetState final : public util::RefCounted<StorageBudgetState> {
public:
  // Returns the constructing +1, immediately adopted by compile_plan. Do not
  // instantiate constrained Ref<Self> in an incomplete class declaration.
  static Result<StorageBudgetState *>
  create(mem::AllocatorRef backing, InvocationStorageLimits limits,
         std::optional<mem::AllocatorRef> control_owner = {},
         bool account_control = false,
         compute::ResourceBudget resource_budget = {}) noexcept {
    if (account_control && limits.max_bytes < sizeof(StorageBudgetState))
      return std::unexpected(Status::ResourceExhausted(
          "graph storage budget cannot hold its control state"));
    const auto owner = control_owner.value_or(backing);
    void *storage =
        owner.allocate(sizeof(StorageBudgetState), alignof(StorageBudgetState));
    if (storage == nullptr)
      return std::unexpected(
          Status::ResourceExhausted("graph storage budget allocation failed"));
    auto *result = ::new (storage) StorageBudgetState(
        backing, owner, limits, std::move(resource_budget));
    if (account_control)
      result->control_charge_ =
          std::move(*result->reserve(sizeof(StorageBudgetState)));
    return result;
  }

  void destroy() noexcept {
    const auto owner = control_owner_;
    this->~StorageBudgetState();
    owner.deallocate(this, sizeof(StorageBudgetState),
                     alignof(StorageBudgetState));
  }

  mem::AllocatorRef backing() const noexcept { return backing_; }
  mem::AllocatorRef allocator() noexcept {
    return {&allocate, &deallocate, &reallocate, this};
  }
  std::optional<mem::QuotaAllocator::Reservation>
  reserve(std::size_t bytes) noexcept {
    return quota_.try_reserve_bytes(bytes);
  }
  const compute::ResourceBudget &resource_budget() const noexcept {
    return resource_budget_;
  }
  Result<StorageCharge> reserve_resource(std::size_t bytes,
                                         mem::MemoryResourceRef resource) noexcept {
    if (bytes == 0) return StorageCharge{};
    auto aggregate = reserve(bytes);
    if (!aggregate)
      return std::unexpected(Status::ResourceExhausted("graph storage budget is exhausted"));
    compute::ResourceBudget::Reservation charge;
    if (resource_budget_) {
      HYLE_ASSIGN_OR_RETURN(charge, resource_budget_.reserve(
          resource, bytes, compute::ResourceBudgetCategory::kConstant));
    }
    return make_charge(std::move(*aggregate), std::move(charge));
  }
  Result<StorageCharge> reserve_private(std::optional<std::size_t> bytes) noexcept {
    if (bytes && *bytes == 0) return StorageCharge{};
    auto aggregate = reserve(bytes.value_or(0));
    if (!aggregate)
      return std::unexpected(Status::ResourceExhausted("graph storage budget is exhausted"));
    compute::ResourceBudget::Reservation charge;
    if (resource_budget_) {
      HYLE_ASSIGN_OR_RETURN(charge, resource_budget_.reserve_unattributed(bytes));
    }
    return make_charge(std::move(*aggregate), std::move(charge));
  }
  bool rejects_unknown() const noexcept {
    return (limits_.max_bytes != SIZE_MAX &&
            limits_.unknown == UnknownStoragePolicy::kReject) ||
           (resource_budget_ && !resource_budget_.allows_unattributed());
  }
  void note_unknown() noexcept {
    unknown_.store(true, std::memory_order_relaxed);
  }
  InvocationStorageUsage usage() const noexcept {
    return {quota_.used(), quota_.peak(), quota_.limit(),
            unknown_.load(std::memory_order_relaxed)};
  }

private:
  Result<StorageCharge> make_charge(mem::QuotaAllocator::Reservation aggregate,
      compute::ResourceBudget::Reservation resource) noexcept {
    const auto owner = allocator();
    auto *memory = owner.allocate(sizeof(StorageChargeState), alignof(StorageChargeState));
    if (memory == nullptr)
      return std::unexpected(Status::ResourceExhausted("graph quota retention metadata"));
    auto *state = ::new (memory) StorageChargeState{
        owner, std::move(aggregate), std::move(resource)};
    return StorageCharge(state);
  }
  StorageBudgetState(mem::AllocatorRef backing, mem::AllocatorRef control_owner,
                     InvocationStorageLimits limits,
                     compute::ResourceBudget resource_budget) noexcept
      : backing_(backing), control_owner_(control_owner), limits_(limits),
        quota_(backing, limits.max_bytes), resource_budget_(std::move(resource_budget)) {}

  static void *allocate(void *opaque, std::size_t bytes,
                        std::size_t alignment) noexcept {
    auto &self = *static_cast<StorageBudgetState *>(opaque);
    void *result = self.quota_.get().allocate(bytes, alignment);
    if (result != nullptr)
      self.incref();
    return result;
  }
  static void deallocate(void *opaque, void *pointer, std::size_t bytes,
                         std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return;
    auto &self = *static_cast<StorageBudgetState *>(opaque);
    self.quota_.get().deallocate(pointer, bytes, alignment);
    self.decref();
  }
  static void *reallocate(void *opaque, void *pointer, std::size_t old_bytes,
                          std::size_t new_bytes,
                          std::size_t alignment) noexcept {
    if (pointer == nullptr)
      return allocate(opaque, new_bytes, alignment);
    auto &self = *static_cast<StorageBudgetState *>(opaque);
    void *result =
        self.quota_.get().reallocate(pointer, old_bytes, new_bytes, alignment);
    if (new_bytes == 0)
      self.decref();
    return result;
  }

  mem::AllocatorRef backing_;
  mem::AllocatorRef control_owner_;
  InvocationStorageLimits limits_;
  mem::QuotaAllocator quota_;
  std::atomic<bool> unknown_{false};
  mem::QuotaAllocator::Reservation control_charge_;
  compute::ResourceBudget resource_budget_;
};

} // namespace hyle::graph::detail
