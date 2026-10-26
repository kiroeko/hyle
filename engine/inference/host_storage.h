#pragma once

#include <concepts>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/compute/storage_owner.h"

namespace hyle::inference {

// Independent cold storage for parked bytes. Common Compute keeps its original
// StorageOwner; typed routes keep only their actual host resource and a stable
// Backend identity token. No Model, execution carrier or Backend State is owned.
class HostStorage final {
  template<class Resource> struct NativeOwner final {
    NativeOwner(std::shared_ptr<const void> selected, Resource &&memory) noexcept
        : identity(std::move(selected)), resource(std::move(memory)) {}
    std::shared_ptr<const void> identity;
    Resource resource;
  };
public:
  HostStorage() noexcept = default;

  static Result<HostStorage> retain(compute::StorageOwner owner) noexcept {
    if (!owner)
      return std::unexpected(Status::InvalidArgument("host storage requires its original StorageOwner"));
    HostStorage result;
    result.resource_ = owner.ref();
    result.identity_ = owner.context_identity();
    result.common_ = std::move(owner);
    return result;
  }

  template<class Resource>
    requires(!std::is_lvalue_reference_v<Resource> &&
             !std::is_const_v<std::remove_reference_t<Resource>> &&
             std::is_nothrow_move_constructible_v<std::remove_cvref_t<Resource>> &&
             requires(std::remove_cvref_t<Resource> &resource) {
               { resource.ref() } noexcept -> std::same_as<mem::MemoryResourceRef>;
             })
  static Result<HostStorage> adopt(Resource &&resource,
      std::shared_ptr<const void> backend_identity) noexcept {
    if (!backend_identity || !resource.ref().valid() ||
        !resource.ref().properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("typed host storage requires its original host resource and Backend identity"));
    try {
      auto owner = std::make_shared<NativeOwner<std::remove_cvref_t<Resource>>>(
          std::move(backend_identity), std::forward<Resource>(resource));
      HostStorage result;
      result.resource_ = owner->resource.ref();
      result.identity_ = owner->identity.get();
      result.native_ = std::move(owner);
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("typed host storage owner metadata"));
    }
  }

  bool valid() const noexcept { return bool(common_) || bool(native_); }
  explicit operator bool() const noexcept { return valid(); }
  mem::MemoryResourceRef ref() const noexcept { return valid() ? resource_ : mem::MemoryResourceRef{}; }
  const void *context_identity() const noexcept { return valid() ? identity_ : nullptr; }

private:
  compute::StorageOwner common_;
  std::shared_ptr<const void> native_;
  mem::MemoryResourceRef resource_;
  const void *identity_ = nullptr;
};

} // namespace hyle::inference
