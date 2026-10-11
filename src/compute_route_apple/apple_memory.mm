#include "hyle/compute/apple_memory.h"

#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mlx/allocator.h>
#include "hyle/mem/std_allocator.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/core/assert.h"

#include "apple_internal.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace detail {
namespace { const char memory_authority = 0; }

struct MemoryState : mem::MemoryResource {
  std::string_view name() const noexcept override { return "MLX shared Metal"; }
  mem::MemoryProperties properties() const noexcept override {
    return {.flags = mem::MemoryProperty::kHostAccessible |
                     mem::MemoryProperty::kDeviceAccessible |
                     mem::MemoryProperty::kHostDeviceCoherent,
            .minimum_alignment = alignof(std::max_align_t),
            .maximum_allocation = SIZE_MAX,
            .domain = {&memory_authority, 0}};
  }
  Result<mem::ResourceAllocation> allocate(std::size_t bytes, std::size_t alignment) noexcept override {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
      return std::unexpected(Status::InvalidArgument("Apple allocation alignment must be a power of two"));
    if (bytes == 0) return mem::ResourceAllocation{nullptr, 0, alignment};
    try {
      auto buffer = mx::allocator::malloc(bytes);
      if (buffer.ptr() == nullptr)
        return std::unexpected(Status::ResourceExhausted("MLX buffer allocation failed"));
      id<MTLBuffer> metal = (__bridge id<MTLBuffer>)buffer.ptr();
      if (metal.storageMode != MTLStorageModeShared ||
          (reinterpret_cast<std::uintptr_t>(metal.contents) & (alignment - 1)) != 0) {
        mx::allocator::free(buffer);
        return std::unexpected(Status::InvalidArgument("MLX backing cannot satisfy shared host alignment"));
      }
      return mem::ResourceAllocation{buffer.ptr(), bytes, alignment};
    } catch (...) { return std::unexpected(native_exception()); }
  }
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    if (allocation.native_handle != nullptr)
      mx::allocator::free(mx::allocator::Buffer(allocation.native_handle));
  }
  Result<mem::ResourceHostMapping> map_host(void *handle, std::size_t offset,
                                           std::size_t bytes, mem::HostAccess) noexcept override {
    if (bytes == 0) return mem::ResourceHostMapping{};
    id<MTLBuffer> metal = (__bridge id<MTLBuffer>)handle;
    if (metal.storageMode != MTLStorageModeShared || offset > metal.length || bytes > metal.length - offset)
      return std::unexpected(Status::InvalidArgument("Apple host map exceeds native shared backing"));
    return mem::ResourceHostMapping{static_cast<std::byte *>(metal.contents) + offset, nullptr};
  }
  Status unmap_host(void *, void *, std::size_t, std::size_t, mem::HostAccess) noexcept override {
    return Status::Ok();
  }
  Status flush(void *, std::size_t, std::size_t) noexcept override { return Status::Ok(); }
  Status invalidate(void *, std::size_t, std::size_t) noexcept override { return Status::Ok(); }
};

struct ArrayStorage final {
  ResourceBudget::Reservation charge;
  ResourceBudget::Reservation private_metadata;
  mem::Buffer buffer;
};
struct LeasedArrayStorage final {
  ResourceBudget::Reservation private_metadata;
  ReadLease source;
};
struct ImportedArrayStorage final : MemoryState {
  ImportedArrayStorage(mx::array source, mem::AllocatorRef selected) noexcept
      : owner(std::move(source)), allocator(selected) {}
  mx::array owner;
  mem::AllocatorRef allocator;
  ResourceBudget::Reservation private_metadata;
  Result<mem::ResourceAllocation> allocate(std::size_t, std::size_t) noexcept override {
    return std::unexpected(Status::Unimplemented("imported MLX array cannot allocate storage"));
  }
  // The actual MLX array owns/free its buffer. This token is a readonly
  // retained view, not a second MLX allocator ownership claim.
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    HYLE_FATAL_ASSERT(allocation.native_handle == owner.buffer().ptr());
  }
};
} // namespace detail

MemoryResource::MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept : state_(std::move(state)) {}
MemoryResource::MemoryResource(MemoryResource &&) noexcept = default;
MemoryResource &MemoryResource::operator=(MemoryResource &&) noexcept = default;
MemoryResource::~MemoryResource() = default;

Result<MemoryResource> MemoryResource::create(const Context &context) noexcept {
  if (!context.valid()) return std::unexpected(Status::InvalidArgument("Apple memory requires a valid Context"));
  try { return MemoryResource(std::make_unique<detail::MemoryState>()); }
  catch (...) { return std::unexpected(detail::native_exception()); }
}
mem::MemoryResourceRef MemoryResource::ref() const noexcept {
  return state_ ? mem::MemoryResourceRef(*state_) : mem::MemoryResourceRef{};
}
Result<mx::array> MemoryResource::allocate(const TensorDesc &description, ResourceBudget budget,
                                         ResourceBudgetCategory category) const noexcept {
  if (!valid()) return std::unexpected(Status::FailedPrecondition("Apple memory resource is empty"));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  if (description.is_plain() && !description.is_dense())
    return std::unexpected(Status::InvalidArgument("Apple owned arrays require a dense plain carrier"));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  try {
    auto metadata = mem::system_allocator();
    if (budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(metadata, budget.metadata_allocator());
    }
    auto storage = std::allocate_shared<detail::ArrayStorage>(
        mem::StdAllocator<detail::ArrayStorage>(metadata));
    if (budget.valid()) {
      // MLX owns its array descriptors. Do not pretend those opaque native
      // allocations are included in our requested payload/control bytes.
      HYLE_ASSIGN_OR_RETURN(storage->private_metadata, budget.reserve_unattributed(std::nullopt));
      HYLE_ASSIGN_OR_RETURN(storage->charge, budget.reserve(ref(), bytes, category));
    }
    HYLE_ASSIGN_OR_RETURN(storage->buffer, mem::Buffer::create(ref(), bytes, alignof(std::max_align_t)));
    mx::array result(spec.shape, spec.dtype, nullptr, {});
    // Evaluate the handle before moving its owner into the deleter: function
    // argument evaluation order must not decide whether storage is null.
    const auto handle = storage->buffer.native_handle();
    result.set_data(mx::allocator::Buffer(handle),
                    [storage = std::move(storage)](mx::allocator::Buffer) {});
    result.set_status(mx::array::available);
    return result;
  } catch (...) { return std::unexpected(detail::native_exception()); }
}
Result<mx::array> MemoryResource::binding(ReadLease source, const TensorDesc &description,
                                        LeaseImportOptions options) const noexcept {
  if (!valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple lease binding resource/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, source.tensor(description));
  const auto storage = tensor.storage();
  if (dynamic_cast<detail::MemoryState *>(&storage.resource().resource()) == nullptr ||
      (description.is_plain() && !description.is_dense()))
    return std::unexpected(Status::InvalidArgument("Apple lease requires original Metal resource and dense carrier"));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if (bytes == 0) return std::unexpected(Status::InvalidArgument("empty Apple lease binding"));
  try {
    auto allocator = options.allocator;
    if (options.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
    }
    auto owner = std::allocate_shared<detail::LeasedArrayStorage>(
        mem::StdAllocator<detail::LeasedArrayStorage>(allocator));
    if (options.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(owner->private_metadata,
                           options.resource_budget.reserve_unattributed(std::nullopt));
    }
    mx::array result(spec.shape, spec.dtype, nullptr, {});
    if (storage.offset() % result.itemsize() != 0 || bytes != result.nbytes() ||
        storage.offset() / result.itemsize() > static_cast<std::size_t>(INT64_MAX))
      return std::unexpected(Status::InvalidArgument("Apple lease carrier origin/size mismatch"));
    id<MTLBuffer> metal = (__bridge id<MTLBuffer>)storage.native_handle();
    if (metal == nil || metal.storageMode != MTLStorageModeShared ||
        storage.offset() > metal.length || bytes > metal.length - storage.offset())
      return std::unexpected(Status::InvalidArgument("Apple lease exceeds shared Metal backing"));
    owner->source = std::move(source);
    mx::array base(mx::Shape{1}, mx::uint8, nullptr, {});
    base.set_data(mx::allocator::Buffer(storage.native_handle()),
                  [retained = std::move(owner)](mx::allocator::Buffer) {});
    result.copy_shared_buffer(base, result.strides(), result.flags(), result.size(),
                             static_cast<std::int64_t>(storage.offset() / result.itemsize()));
    result.set_status(mx::array::available);
    return result;
  } catch (...) { return std::unexpected(detail::native_exception()); }
}

Result<ReadLease> MemoryResource::import_readonly(const mx::array &source, const TensorDesc &description,
                                                LeaseImportOptions options) const noexcept {
  if (!valid() || !options.allocator.valid() || source.status() != mx::array::available ||
      !source.flags().row_contiguous)
    return std::unexpected(Status::InvalidArgument("Apple import requires a ready dense native array"));
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if ((description.is_plain() && !description.is_dense()) || source.shape() != spec.shape ||
      source.dtype() != spec.dtype || source.nbytes() != bytes || bytes == 0)
    return std::unexpected(Status::InvalidArgument("Apple import description/carrier mismatch"));
  HYLE_ASSIGN_OR_RETURN(auto info, storage_info(source));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *storage = allocator.allocate(sizeof(detail::ImportedArrayStorage), alignof(detail::ImportedArrayStorage));
  if (!storage) return std::unexpected(Status::ResourceExhausted("Apple import owner metadata"));
  auto *owner = std::construct_at(static_cast<detail::ImportedArrayStorage *>(storage), source, allocator);
  const auto destroy_owner = [](void *opaque) noexcept {
    auto *state = static_cast<detail::ImportedArrayStorage *>(opaque);
    const auto selected = state->allocator;
    std::destroy_at(state);
    selected.deallocate(state, sizeof(detail::ImportedArrayStorage), alignof(detail::ImportedArrayStorage));
  };
  if (options.resource_budget.valid()) {
    auto opaque = options.resource_budget.reserve_unattributed(std::nullopt);
    if (!opaque) { destroy_owner(owner); return std::unexpected(std::move(opaque).error()); }
    owner->private_metadata = std::move(*opaque);
  }
  auto backing = mem::Buffer::adopt(mem::MemoryResourceRef(*owner),
      {const_cast<void *>(info.buffer), info.backing_bytes, 1});
  if (!backing) { destroy_owner(owner); return std::unexpected(std::move(backing).error()); }
  auto imported = ReadLease::adopt(std::move(*backing), {.allocator = allocator});
  if (!imported) {
    backing = {};
    destroy_owner(owner);
    return std::unexpected(std::move(imported).error());
  }
  compute::detail::LeaseAccess::retain_resource_owner(*imported, owner, destroy_owner);
  return imported->subview(info.offset_bytes, bytes);
}

Result<NativeStorage> MemoryResource::storage_info(const mx::array &array) const noexcept {
  if (!valid() || array.status() != mx::array::available)
    return std::unexpected(Status::FailedPrecondition("Apple storage requires a ready native array"));
  if (array.offset() < 0)
    return std::unexpected(Status::OutOfRange("Apple native array offset is negative"));
  const auto handle = array.buffer().ptr();
  if (handle == nullptr) {
    if (array.nbytes() == 0) return NativeStorage{};
    return std::unexpected(Status::InvalidArgument("Apple array has no native backing"));
  }
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
  const auto offset = static_cast<std::size_t>(array.offset());
  if (buffer.storageMode != MTLStorageModeShared || offset > buffer.length ||
      array.nbytes() > buffer.length - offset)
    return std::unexpected(Status::InvalidArgument("Apple array exceeds shared native backing"));
  return NativeStorage{handle, offset, array.nbytes(), buffer.length};
}

namespace {
Status match_dense(const mx::array &array, const TensorDesc &description) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(description));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  if ((description.is_plain() && !description.is_dense()) ||
      array.shape() != spec.shape || array.dtype() != spec.dtype ||
      !array.flags().row_contiguous || array.nbytes() != bytes)
    return Status::InvalidArgument("Apple boundary requires the exact dense native carrier");
  return Status::Ok();
}
} // namespace

Status MemoryResource::copy_from_host(const mx::array &destination, ConstTensorView source) const noexcept {
  if (!source.valid()) return Status::InvalidArgument("invalid Apple boundary source");
  HYLE_RETURN_IF_ERROR(match_dense(destination, source.description()));
  HYLE_ASSIGN_OR_RETURN(auto storage, storage_info(destination));
  if (storage.view_bytes == 0) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto mapping, source.storage().map_host());
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage.buffer;
  std::memcpy(static_cast<std::byte *>(buffer.contents) + storage.offset_bytes,
              mapping.bytes().data(), storage.view_bytes);
  return mapping.close();
}
Status MemoryResource::copy_to_host(const mx::array &source, TensorView destination) const noexcept {
  if (!destination.valid()) return Status::InvalidArgument("invalid Apple boundary destination");
  HYLE_RETURN_IF_ERROR(match_dense(source, destination.description()));
  HYLE_ASSIGN_OR_RETURN(auto storage, storage_info(source));
  if (storage.view_bytes == 0) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto mapping, destination.storage().map_host(mem::HostAccess::kWrite));
  HYLE_ASSIGN_OR_RETURN(auto bytes, mapping.writable_bytes());
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage.buffer;
  std::memcpy(bytes.data(), static_cast<const std::byte *>(buffer.contents) + storage.offset_bytes,
              storage.view_bytes);
  return mapping.close();
}
} // namespace hyle::compute::apple
