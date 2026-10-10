#include "hyle/compute/apple_memory.h"

#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>
#include <mlx/allocator.h>
#include "hyle/mem/std_allocator.h"

#include "apple_internal.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace detail {
namespace { const char memory_authority = 0; }

struct MemoryState final : mem::MemoryResource {
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
