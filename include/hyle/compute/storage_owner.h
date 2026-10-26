#pragma once

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

#include "hyle/mem/buffer.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/resource_budget_access.h"
#include "hyle/compute/resource_leases.h"

namespace hyle::compute {

// Cold ownership of one actual typed resource and its original Context.
// Copies retain the same resource object; ref() never owns storage itself.
// This scope submits no work and does not reopen a closed Context.
class StorageOwner final {
public:
  StorageOwner() noexcept = default;

  // Validation precedes transfer. Without an explicit allocator owner, only
  // the complete process-lifetime system_allocator identity is accepted.
  template <class Resource>
    requires(!std::is_lvalue_reference_v<Resource> &&
             !std::is_const_v<std::remove_reference_t<Resource>> &&
             std::movable<std::remove_cvref_t<Resource>> &&
             std::is_nothrow_move_constructible_v<
                 std::remove_cvref_t<Resource>> &&
             requires(std::remove_cvref_t<Resource> &resource) {
               { resource.ref() } noexcept -> std::same_as<mem::MemoryResourceRef>;
             })
  static Result<StorageOwner> adopt(const Context &context,
                                    Resource &&resource) noexcept {
    try {
      const auto allocator = context.metadata_allocator();
      if (!same_allocator(allocator, mem::system_allocator()))
        return std::unexpected(Status::InvalidArgument(
            "storage owner requires the original metadata allocator owner"));
      return adopt_resource(context, std::forward<Resource>(resource), allocator,
                            {});
    } catch (...) {
      return std::unexpected(
          Status::ResourceExhausted("storage owner construction failed"));
    }
  }

  // The shared pointer owns this real allocator object. The private erasure
  // below changes only its stored type, never its ownership or deleter.
  template <class Resource, class Allocator>
    requires(!std::is_lvalue_reference_v<Resource> &&
             !std::is_const_v<std::remove_reference_t<Resource>> &&
             std::movable<std::remove_cvref_t<Resource>> &&
             std::is_nothrow_move_constructible_v<
                 std::remove_cvref_t<Resource>> &&
             !std::is_const_v<Allocator> &&
             requires(std::remove_cvref_t<Resource> &resource,
                      Allocator &allocator) {
               { resource.ref() } noexcept -> std::same_as<mem::MemoryResourceRef>;
               { allocator.ref() } -> std::same_as<mem::AllocatorRef>;
             })
  static Result<StorageOwner>
  adopt(const Context &context, Resource &&resource,
        std::shared_ptr<Allocator> allocator_owner) noexcept {
    try {
      if (!allocator_owner)
        return std::unexpected(
            Status::InvalidArgument("storage owner allocator is empty"));
      const auto allocator = allocator_owner->ref();
      if (!same_allocator(allocator, context.metadata_allocator()))
        return std::unexpected(Status::InvalidArgument(
            "storage owner metadata allocator differs from Context"));
      return adopt_resource(context, std::forward<Resource>(resource), allocator,
                            std::move(allocator_owner));
    } catch (...) {
      return std::unexpected(
          Status::ResourceExhausted("storage owner construction failed"));
    }
  }

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  mem::MemoryResourceRef ref() const noexcept {
    return state_ ? state_->resource : mem::MemoryResourceRef{};
  }
  mem::AllocatorRef allocator() const noexcept {
    return state_ ? state_->allocator : mem::AllocatorRef{};
  }
  bool owns_allocator(mem::AllocatorRef value) const noexcept {
    return state_ && same_allocator(state_->allocator, value);
  }
  const void *context_identity() const noexcept {
    return state_ ? state_->context.identity_key() : nullptr;
  }

  // Takes only ready, uniquely owned bytes from this exact resource. The
  // original ReadLease owns the allocation; its existing domain attachment
  // retains this scope through all readers/native uses. Failure keeps backing.
  Result<ReadLease> adopt_buffer(mem::Buffer &&backing,
                                 ResourceBudget budget = {}) const noexcept {
    if (!state_ || !backing.valid() || backing.resource() != ref())
      return std::unexpected(
          Status::InvalidArgument("storage owner buffer resource differs"));
    if (budget) {
      const auto actual = detail::ResourceBudgetAccess::backing_allocator(budget);
      if (!owns_allocator(actual) &&
          !same_allocator(actual, mem::system_allocator()))
        return std::unexpected(Status::InvalidArgument(
            "storage owner cannot retain the budget metadata allocator"));
    }
    try {
      // Allocate before ReadLease::adopt can consume the Buffer. Its release
      // callback runs after the lease backing and metadata are destroyed.
      auto retained = std::make_unique<StorageOwner>(*this);
      auto lease = ReadLease::adopt(std::move(backing),
          {.allocator = state_->allocator, .resource_budget = std::move(budget)});
      if (!lease)
        return std::unexpected(std::move(lease).error());
      detail::LeaseAccess::retain_allocator_owner(*lease, state_->allocator_owner);
      detail::LeaseAccess::retain_resource_owner(*lease, retained.release(),
          +[](void *owner) noexcept {
            delete static_cast<StorageOwner *>(owner);
          });
      return std::move(*lease);
    } catch (...) {
      return std::unexpected(
          Status::ResourceExhausted("storage owner lease allocation failed"));
    }
  }

  // Zero-copy freeze of the original full pool writer from this exact
  // resource. Retains this resource/Context and the real metadata allocator
  // through all escaped readers/native uses. Rejects a foreign allocator or
  // an already attached owner before transfer; every failure keeps both handles.
  Result<ReadLease> freeze(LeasePool &pool, WriteLease &&writer) const noexcept {
    if (!state_)
      return std::unexpected(Status::InvalidArgument("storage owner is empty"));
    HYLE_ASSIGN_OR_RETURN(const auto backing, detail::LeaseAccess::backing(writer));
    if (backing.resource != ref())
      return std::unexpected(Status::InvalidArgument(
          "storage owner pool resource differs"));
    HYLE_ASSIGN_OR_RETURN(const auto actual,
        detail::LeaseAccess::freeze_owner_allocator(pool, writer));
    if (!owns_allocator(actual) &&
        !same_allocator(actual, mem::system_allocator()))
      return std::unexpected(Status::InvalidArgument(
          "storage owner cannot retain the pool metadata allocator"));
    try {
      // This is the only new allocation. It precedes consumption, so neither
      // allocation failure nor the original freeze's rejection changes bytes,
      // writer authority or pool ownership.
      auto retained = std::make_unique<StorageOwner>(*this);
      auto lease = pool.freeze(std::move(writer));
      if (!lease) return std::unexpected(std::move(lease).error());
      detail::LeaseAccess::retain_allocator_owner(*lease, state_->allocator_owner);
      detail::LeaseAccess::retain_resource_owner(*lease, retained.release(),
          +[](void *owner) noexcept { delete static_cast<StorageOwner *>(owner); });
      return std::move(*lease);
    } catch (...) {
      return std::unexpected(
          Status::ResourceExhausted("storage owner pool freeze allocation failed"));
    }
  }

private:
  struct State {
    State(detail::ContextRef retained_context, mem::AllocatorRef selected,
          std::shared_ptr<void> retained_allocator) noexcept
        : allocator_owner(std::move(retained_allocator)),
          context(std::move(retained_context)), allocator(selected) {}
    virtual ~State() = default;

    // Derived resource dies first, then Context, then the real allocator.
    // The shared control block itself uses the process-lifetime system heap.
    std::shared_ptr<void> allocator_owner;
    detail::ContextRef context;
    mem::AllocatorRef allocator;
    mem::MemoryResourceRef resource;
  };

  template <class Resource> struct ResourceState final : State {
    ResourceState(detail::ContextRef context, mem::AllocatorRef allocator,
                  std::shared_ptr<void> allocator_owner, Resource &&resource) noexcept
        : State(std::move(context), allocator, std::move(allocator_owner)),
          owned_resource(std::move(resource)) {
      this->resource = owned_resource.ref();
    }
    Resource owned_resource;
  };

  static bool same_allocator(mem::AllocatorRef left,
                             mem::AllocatorRef right) noexcept {
    return left.valid() && right.valid() &&
           left.alloc_fn() == right.alloc_fn() &&
           left.free_fn() == right.free_fn() &&
           left.realloc_fn() == right.realloc_fn() &&
           left.state() == right.state();
  }

  template <class Resource>
  static Result<StorageOwner>
  adopt_resource(const Context &context, Resource &&resource,
                 mem::AllocatorRef allocator,
                 std::shared_ptr<void> allocator_owner) {
    HYLE_ASSIGN_OR_RETURN(auto retained, detail::ContextRef::acquire(context));
    const auto source = resource.ref();
    if (!source.valid())
      return std::unexpected(
          Status::InvalidArgument("storage owner resource is empty"));
    HYLE_RETURN_IF_ERROR(retained.validate_resource(source));
    StorageOwner result;
    result.state_ = std::make_shared<ResourceState<std::remove_cvref_t<Resource>>>(
        std::move(retained), allocator, std::move(allocator_owner),
        std::forward<Resource>(resource));
    return result;
  }

  std::shared_ptr<State> state_;
};

} // namespace hyle::compute
