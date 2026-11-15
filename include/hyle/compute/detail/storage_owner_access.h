#pragma once

#include "hyle/compute/storage_owner.h"

namespace hyle::compute::detail {

// Native route only: the caller has proved that this exact resource belongs
// to the retained original runtime. Transfer follows all fallible checks.
struct StorageOwnerAccess final {
  template<class Runtime, class Resource>
  static Result<StorageOwner> adopt_native(std::shared_ptr<Runtime> runtime,
      Resource &&resource, mem::AllocatorRef allocator = mem::system_allocator()) noexcept {
    if (!runtime || !resource.ref().valid() ||
        !StorageOwner::same_allocator(allocator, mem::system_allocator()))
      return std::unexpected(Status::InvalidArgument("native storage owner requires its actual endpoint, resource and allocator"));
    try {
      StorageOwner result;
      result.state_ = std::make_shared<StorageOwner::NativeResourceState<std::remove_cvref_t<Resource>>>(
          std::move(runtime), std::forward<Resource>(resource));
      return result;
    } catch (...) {
      return std::unexpected(Status::ResourceExhausted("native storage owner metadata"));
    }
  }
};

} // namespace hyle::compute::detail
