#include "vulkan_memory_internal.h"

#include <algorithm>
#include <cstdint>
#include <new>

namespace hyle::compute::vulkan {
namespace detail {
struct AllocationToken final {
  std::shared_ptr<Buffer> buffer;
};

struct MemoryState final : mem::MemoryResource {
  std::shared_ptr<Runtime> runtime;
  MemoryKind kind;
  mem::AllocatorRef allocator;

  std::string_view name() const noexcept override {
    return kind == MemoryKind::kDevice ? "Vulkan device buffer" : "Vulkan mapped staging";
  }
  mem::MemoryProperties properties() const noexcept override {
    return {.flags = kind == MemoryKind::kDevice ? mem::MemoryProperty::kDeviceAccessible
                : mem::MemoryProperty::kDeviceAccessible | mem::MemoryProperty::kHostAccessible,
            .minimum_alignment = 1,
            .maximum_allocation = static_cast<std::size_t>(runtime->device->properties.limits.maxStorageBufferRange),
            .domain = {runtime->device.get(), static_cast<std::uintptr_t>(kind)}};
  }
  Result<mem::ResourceAllocation> allocate(std::size_t bytes, std::size_t alignment) noexcept override {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
      return std::unexpected(Status::InvalidArgument("invalid Vulkan memory alignment"));
    if (bytes == 0) return mem::ResourceAllocation{nullptr, 0, alignment};
    std::lock_guard lock(runtime->mutex);
    if (runtime->closed) return std::unexpected(Status::FailedPrecondition("Vulkan context is closed"));
    if (!runtime->failure.ok()) return std::unexpected(runtime->failure);
    HYLE_ASSIGN_OR_RETURN(auto buffer, allocate_buffer(runtime->device, bytes,
        kind == MemoryKind::kStaging, false, {.allocator = allocator}));
    if (buffer->mapped != nullptr && reinterpret_cast<std::uintptr_t>(buffer->mapped) % alignment != 0)
      return std::unexpected(Status::InvalidArgument("Vulkan staging cannot satisfy host alignment"));
    auto *storage = allocator.allocate(sizeof(AllocationToken), alignof(AllocationToken));
    if (storage == nullptr) return std::unexpected(Status::ResourceExhausted("Vulkan resource token metadata"));
    auto *token = std::construct_at(static_cast<AllocationToken *>(storage), std::move(buffer));
    return mem::ResourceAllocation{token, bytes, alignment};
  }
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    if (allocation.native_handle == nullptr) return;
    auto *token = static_cast<AllocationToken *>(allocation.native_handle);
    std::destroy_at(token);
    allocator.deallocate(token, sizeof(AllocationToken), alignof(AllocationToken));
  }
  Result<mem::ResourceHostMapping> map_host(void *handle, std::size_t offset,
                                           std::size_t bytes, mem::HostAccess) noexcept override {
    if (bytes == 0) return mem::ResourceHostMapping{};
    auto *token = static_cast<AllocationToken *>(handle);
    if (kind != MemoryKind::kStaging || token == nullptr || token->buffer->mapped == nullptr ||
        offset > token->buffer->bytes || bytes > token->buffer->bytes - offset)
      return std::unexpected(Status::InvalidArgument("Vulkan map requires an in-range staging allocation"));
    return mem::ResourceHostMapping{static_cast<std::byte *>(token->buffer->mapped) + offset, nullptr};
  }
  Status unmap_host(void *, void *, std::size_t, std::size_t, mem::HostAccess) noexcept override { return Status::Ok(); }
  Status mapped_visibility(void *handle, std::size_t offset, std::size_t bytes, bool flush) noexcept {
    if (bytes == 0) return Status::Ok();
    auto *token = static_cast<AllocationToken *>(handle);
    if (kind != MemoryKind::kStaging || token == nullptr || offset > token->buffer->bytes ||
        bytes > token->buffer->bytes - offset)
      return Status::InvalidArgument("Vulkan visibility range requires staging");
    const auto &buffer = *token->buffer;
    if (buffer.memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return Status::Ok();
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buffer.memory;
    range.size = VK_WHOLE_SIZE; // Original whole mapping, atom alignment preserved.
    auto &device = *runtime->device;
    return native_status(flush ? device.fn.FlushMappedMemoryRanges(device.native.device, 1, &range)
                              : device.fn.InvalidateMappedMemoryRanges(device.native.device, 1, &range),
                         flush ? "vkFlushMappedMemoryRanges" : "vkInvalidateMappedMemoryRanges");
  }
  Status flush(void *handle, std::size_t offset, std::size_t bytes) noexcept override {
    return mapped_visibility(handle, offset, bytes, true);
  }
  Status invalidate(void *handle, std::size_t offset, std::size_t bytes) noexcept override {
    return mapped_visibility(handle, offset, bytes, false);
  }
};

Result<std::shared_ptr<Buffer>> MemoryAccess::buffer(
    const MemoryResource &resource, const mem::Buffer &allocation) noexcept {
  if (!resource.valid() || allocation.resource() != resource.ref() || allocation.native_handle() == nullptr)
    return std::unexpected(Status::InvalidArgument("Vulkan allocation resource differs"));
  return static_cast<const AllocationToken *>(allocation.native_handle())->buffer;
}
const std::shared_ptr<Runtime> &MemoryAccess::runtime(const MemoryResource &resource) noexcept {
  return resource.state_->runtime;
}
} // namespace detail

MemoryResource::MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept : state_(std::move(state)) {}
MemoryResource::MemoryResource(MemoryResource &&) noexcept = default;
MemoryResource &MemoryResource::operator=(MemoryResource &&) noexcept = default;
MemoryResource::~MemoryResource() = default;
Result<MemoryResource> MemoryResource::create(const Context &context, MemoryKind kind,
                                             mem::AllocatorRef allocator) noexcept {
  if (!context.valid() || !allocator.valid() || (kind != MemoryKind::kDevice && kind != MemoryKind::kStaging))
    return std::unexpected(Status::InvalidArgument("invalid Vulkan memory resource creation"));
  try {
    auto state = std::make_unique<detail::MemoryState>();
    state->runtime = detail::Access::runtime(context);
    state->kind = kind;
    state->allocator = allocator;
    return MemoryResource(std::move(state));
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan memory resource metadata"));
  }
}
mem::MemoryResourceRef MemoryResource::ref() const noexcept {
  return state_ ? mem::MemoryResourceRef(*state_) : mem::MemoryResourceRef{};
}
Result<Tensor> MemoryResource::allocate(const TensorDesc &description, AllocationOptions options) const noexcept {
  if (!valid() || !description.valid()) return std::unexpected(Status::InvalidArgument("invalid Vulkan tensor allocation"));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  auto &runtime = *state_->runtime;
  std::lock_guard lock(runtime.mutex);
  if (runtime.closed) return std::unexpected(Status::FailedPrecondition("Vulkan context is closed"));
  if (!runtime.failure.ok()) return std::unexpected(runtime.failure);
  HYLE_ASSIGN_OR_RETURN(auto buffer, detail::allocate_buffer(runtime.device, bytes,
      state_->kind == MemoryKind::kStaging, true,
      {.budget = options.resource_budget, .resource = ref(), .category = options.category, .allocator = state_->allocator}));
  return detail::Access::tensor(description, std::move(buffer));
}
Result<NativeStorage> MemoryResource::storage_info(const Tensor &tensor) const noexcept {
  if (!valid() || !tensor.valid() || detail::Access::buffer(tensor)->device != state_->runtime->device)
    return std::unexpected(Status::InvalidArgument("Vulkan tensor belongs to another native Device"));
  const auto &buffer = *detail::Access::buffer(tensor);
  return NativeStorage{buffer.handle, buffer.offset, buffer.bytes,
      buffer.allocation_bytes != 0 ? std::optional<std::size_t>(buffer.allocation_bytes) : std::nullopt};
}
} // namespace hyle::compute::vulkan
