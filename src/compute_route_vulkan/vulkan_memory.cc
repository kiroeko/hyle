#include "vulkan_memory_internal.h"
#include "vulkan_copy_internal.h"
#include "vulkan_transfer_internal.h"
#include "hyle/compute/detail/vulkan_batch_access.h"
#include "hyle/compute/detail/paged_arena_access.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <optional>
#include <vector>

namespace hyle::compute::vulkan {
namespace detail {
struct MemoryState;
struct AllocationToken final {
  std::shared_ptr<Buffer> buffer;
  mem::AllocatorRef allocator;
  std::size_t logical_bytes = 0;
  MemoryState *controller = nullptr;
};
namespace { const unsigned retained_buffer_authority = 0; }
struct LeaseViewOwner final {
  std::shared_ptr<Buffer> backing;
  ReadLease source;
};
template<class Frame>struct PagedViewOwner final {
  std::shared_ptr<Buffer> backing;
  Frame frame;
};

struct DeviceIO;
enum class DeviceIOKind : std::uint8_t { kRead, kWrite, kFill };
struct MemoryState final : mem::MemoryResource {
  ~MemoryState() noexcept override;
  // A public wrapper owns one reference. Real allocation/import tokens keep
  // this exact controller alive through late lease/Run destruction as well.
  // Internal synchronous DeviceIO scratch stays owned by its unique scope.
  mutable std::atomic<std::size_t> references{1};
  bool retain_tokens = false;
  MemoryState *retain_controller() const noexcept {
    if (!retain_tokens) return nullptr;
    references.fetch_add(1, std::memory_order_relaxed);
    return const_cast<MemoryState *>(this);
  }
  void release_controller() noexcept {
    if (references.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
  }
  std::shared_ptr<Runtime> runtime;
  MemoryKind kind;
  mem::AllocatorRef allocator;
  std::mutex io_mutex;
  std::unique_ptr<DeviceIO> io;

  Status read(void *, std::size_t, std::span<std::byte>) noexcept override;
  Status write(void *, std::size_t, std::span<const std::byte>) noexcept override;
  Status fill(void *, std::size_t, std::size_t, std::byte) noexcept override;
  Status device_io(void *, std::size_t, std::size_t, DeviceIOKind,
      std::byte *destination, const std::byte *source, std::byte value) noexcept;

  std::string_view name() const noexcept override {
    return kind == MemoryKind::kDevice ? "Vulkan device buffer" : "Vulkan mapped staging";
  }
  mem::MemoryProperties properties() const noexcept override {
    return {.flags = kind == MemoryKind::kDevice ? mem::MemoryProperty::kDeviceAccessible
                : mem::MemoryProperty::kDeviceAccessible | mem::MemoryProperty::kHostAccessible |
                  mem::MemoryProperty::kHostAddressIdentity,
            .minimum_alignment = 1,
            .maximum_allocation = static_cast<std::size_t>(runtime->device->properties.limits.maxStorageBufferRange),
            .domain = {runtime->device.get(), static_cast<std::uintptr_t>(kind)}};
  }
  Result<mem::ResourceAllocation> allocate(std::size_t bytes, std::size_t alignment) noexcept override {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
      return std::unexpected(Status::InvalidArgument("invalid Vulkan memory alignment"));
    if (bytes == 0) {
      (void)retain_controller();
      return mem::ResourceAllocation{nullptr, 0, alignment};
    }
    std::lock_guard lock(runtime->mutex);
    if (runtime->closed) return std::unexpected(Status::FailedPrecondition("Vulkan context is closed"));
    if (!runtime->failure.ok()) return std::unexpected(runtime->failure);
    HYLE_ASSIGN_OR_RETURN(auto buffer, allocate_buffer(runtime->device, bytes,
        kind == MemoryKind::kStaging, false, {.allocator = allocator, .transfer_access = true}));
    if (buffer->mapped != nullptr && reinterpret_cast<std::uintptr_t>(buffer->mapped) % alignment != 0)
      return std::unexpected(Status::InvalidArgument("Vulkan staging cannot satisfy host alignment"));
    auto *storage = allocator.allocate(sizeof(AllocationToken), alignof(AllocationToken));
    if (storage == nullptr) return std::unexpected(Status::ResourceExhausted("Vulkan resource token metadata"));
    auto *token = std::construct_at(static_cast<AllocationToken *>(storage), std::move(buffer), allocator, bytes, retain_controller());
    return mem::ResourceAllocation{token, bytes, alignment};
  }
  void deallocate(mem::ResourceAllocation allocation) noexcept override {
    if (allocation.native_handle == nullptr) {
      if (retain_tokens) release_controller();
      return;
    }
    auto *token = static_cast<AllocationToken *>(allocation.native_handle);
    const auto selected = token->allocator;
    auto *controller = token->controller;
    std::destroy_at(token);
    selected.deallocate(token, sizeof(AllocationToken), alignof(AllocationToken));
    if (controller) controller->release_controller(); // May destroy this; no later access.
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

// One bounded synchronous controller per device MemoryState. The staging
// resource outlives its Buffer; recording releases native owners first. No
// Context/Queue or executable is constructed on a read/write/fill call.
struct DeviceIO final {
  static constexpr std::size_t capacity = 1024u * 1024u;
  std::unique_ptr<MemoryState> staging_resource;
  mem::Buffer staging;
  std::shared_ptr<CopyRecording> recording;
  Status failure;
};
MemoryState::~MemoryState() noexcept = default;

Status MemoryState::read(void *handle, std::size_t offset,
    std::span<std::byte> destination) noexcept {
  if (kind == MemoryKind::kStaging)
    return mem::MemoryResource::read(handle, offset, destination);
  return device_io(handle, offset, destination.size(), DeviceIOKind::kRead,
      destination.data(), nullptr, {});
}
Status MemoryState::write(void *handle, std::size_t offset,
    std::span<const std::byte> source) noexcept {
  if (kind == MemoryKind::kStaging)
    return mem::MemoryResource::write(handle, offset, source);
  return device_io(handle, offset, source.size(), DeviceIOKind::kWrite,
      nullptr, source.data(), {});
}
Status MemoryState::fill(void *handle, std::size_t offset,
    std::size_t bytes, std::byte value) noexcept {
  if (kind == MemoryKind::kStaging)
    return mem::MemoryResource::fill(handle, offset, bytes, value);
  return device_io(handle, offset, bytes, DeviceIOKind::kFill,
      nullptr, nullptr, value);
}
Status MemoryState::device_io(void *handle, std::size_t offset, std::size_t bytes,
    DeviceIOKind operation, std::byte *destination, const std::byte *source,
    std::byte value) noexcept {
  if (!bytes) return Status::Ok();
  // This is the resource's own typed handle, not a consumer interpretation of
  // opaque native storage. Respect both its authorized logical range and the
  // original native allocation/transfer extent, including an alias's origin.
  const auto *token = static_cast<const AllocationToken *>(handle);
  if (!token || !token->buffer || token->buffer->device != runtime->device ||
      token->buffer->handle == VK_NULL_HANDLE || !token->buffer->allocation_bytes ||
      offset > token->logical_bytes || bytes > token->logical_bytes - offset ||
      offset > token->buffer->bytes || bytes > token->buffer->bytes - offset ||
      token->buffer->offset > SIZE_MAX || offset > SIZE_MAX - token->buffer->offset ||
      token->buffer->offset > token->buffer->allocation_bytes ||
      offset > token->buffer->allocation_bytes - token->buffer->offset ||
      bytes > token->buffer->allocation_bytes - token->buffer->offset - offset ||
      !std::in_range<std::int64_t>(token->logical_bytes))
    return Status::OutOfRange("Vulkan synchronous IO exceeds its typed logical/native allocation");
  if ((static_cast<std::size_t>(token->buffer->offset) + offset) & 3u)
    return Status::Unimplemented("Vulkan synchronous device IO requires exact four-byte aligned ranges");
  auto transfer_bytes = bytes;
  if (bytes & 3u) {
    // Only a complete original allocation owns its private transfer padding.
    // Never widen a subrange, alias, or imported owner into adjacent bytes.
    HYLE_ASSIGN_OR_RETURN(auto padded, buffer_capacity(token->logical_bytes));
    if (offset || bytes != token->logical_bytes || token->buffer->is_view ||
        token->buffer->external_owner || token->buffer->offset ||
        token->buffer->memory == VK_NULL_HANDLE ||
        token->buffer->bytes != padded || padded > token->buffer->allocation_bytes ||
        !std::in_range<std::size_t>(padded))
      return Status::Unimplemented("Vulkan synchronous device IO cannot pad an unaligned partial range");
    transfer_bytes = static_cast<std::size_t>(padded);
  }
  if (!std::in_range<std::int64_t>(transfer_bytes))
    return Status::OutOfRange("Vulkan synchronous IO transfer extent overflows");
  std::lock_guard guard(io_mutex);
  try {
    if (!io) {
      auto candidate = std::make_unique<DeviceIO>();
      candidate->staging_resource = std::make_unique<MemoryState>();
      candidate->staging_resource->runtime = runtime;
      candidate->staging_resource->kind = MemoryKind::kStaging;
      candidate->staging_resource->allocator = allocator;
      HYLE_ASSIGN_OR_RETURN(candidate->staging, mem::Buffer::create(
          mem::MemoryResourceRef(*candidate->staging_resource), DeviceIO::capacity, 4));
      HYLE_ASSIGN_OR_RETURN(candidate->recording, prepare_copy_recording_runtime(
          runtime, 1, allocator, DeviceIO::capacity, CopyRecordingMode::kReplayable));
      io = std::move(candidate);
    }
    HYLE_RETURN_IF_ERROR(io->failure);
    // Re-export one short-lived control token, not the payload. Exact Mem
    // subviews then use the same typed range validator as ordinary native
    // CopyRecording; no new staging/Run is allocated per call or per chunk.
    // The private whole-allocation proof above may authorize its padding for
    // this control token only; the caller's original logical extent is intact.
    const auto exported_bytes = transfer_bytes != bytes ? transfer_bytes : token->logical_bytes;
    HYLE_ASSIGN_OR_RETURN(auto description, TensorDesc::dense(DataType::kUInt8,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(exported_bytes)}));
    HYLE_ASSIGN_OR_RETURN(auto target, MemoryAccess::export_storage(runtime,
        mem::MemoryResourceRef(*this), Access::tensor(description, token->buffer), allocator));
    for (std::size_t done = 0; done < bytes;) {
      const auto count = (std::min)(DeviceIO::capacity, bytes - done);
      const auto copied = (std::min)(DeviceIO::capacity, transfer_bytes - done);
      HYLE_ASSIGN_OR_RETURN(auto stage, io->staging.view().subview(0, copied));
      HYLE_ASSIGN_OR_RETURN(auto payload, stage.subview(0, count));
      HYLE_ASSIGN_OR_RETURN(auto device, target.view().subview(offset + done, copied));
      Status status;
      if (operation == DeviceIOKind::kWrite)
        status = payload.write(std::span(source + done, count));
      else if (operation == DeviceIOKind::kFill)
        status = payload.fill(value);
      if (status.ok() && operation != DeviceIOKind::kRead && copied != count) {
        HYLE_ASSIGN_OR_RETURN(auto padding, stage.subview(count, copied - count));
        status = padding.fill(std::byte{0});
      }
      if (!status.ok()) { io->failure = status; return status; }
      const compute::detail::CopyBackendRange range = operation == DeviceIOKind::kRead
          ? compute::detail::CopyBackendRange{stage, mem::ConstBufferView(device)}
          : compute::detail::CopyBackendRange{device, mem::ConstBufferView(stage)};
      // Host staging writes have already flushed through the original base
      // resource path. Keep this synchronous IO on the original compute queue.
      status = bind_copy_recording(*io->recording, std::span(&range, 1));
      if (status.ok()) {
        auto completed = submit_copy_recording(*io->recording);
        if (!completed) status = std::move(completed).error();
        else {
          status = completed->wait();
        }
      }
      // clear has its own actual-retirement gate, including accepted failures.
      // A failed controller stays retained/failed; it is never silently rebuilt
      // to erase the original native diagnosis or overwrite pending staging.
      const auto cleared = clear_copy_recording(*io->recording);
      if (!cleared.ok()) {
        // Raw resource IO has no F1 episode preventing another resource/Graph
        // from reusing a pool page. If retirement is not proved, block every
        // new native admission on this Runtime; original owners/drain still
        // retain and retire the accepted work. A local cache flag is not enough.
        std::lock_guard lock(runtime->mutex);
        runtime->failure.Update(cleared);
      }
      status.Update(cleared);
      if (!status.ok()) {
        io->failure = status;
        // The failed cache will never replay. Relinquish its recording root:
        // the original pending Run already owns both actual native buffers.
        // Keeping recording->Buffer->controller here would create a permanent
        // controller->DeviceIO cycle even after a later real native drain.
        if (!cleared.ok()) io->recording.reset();
        return status;
      }
      if (operation == DeviceIOKind::kRead) {
        status = payload.read(std::span(destination + done, count));
        if (!status.ok()) { io->failure = status; return status; }
      }
      done += count;
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan synchronous IO cold metadata");
  }
}

Status MemoryAccess::validate_resource(const std::shared_ptr<Runtime> &runtime,
                                      mem::MemoryResourceRef resource) noexcept {
  if (!runtime || !resource.valid())
    return Status::InvalidArgument("Vulkan execution source requires a valid resource");
  const auto *native = dynamic_cast<const MemoryState *>(&resource.resource());
  if (!native || native->runtime != runtime)
    return Status::InvalidArgument("Vulkan resource belongs to another actual Runtime");
  return Status::Ok();
}

namespace {
enum class ConstantDirection { kUpload, kDownload, kDevice };
std::optional<ConstantDirection> constant_direction(
    const std::shared_ptr<Runtime> &runtime, const ConstantCopy &copy) noexcept {
  const auto device = [&](mem::MemoryResourceRef resource) {
    const auto *native = dynamic_cast<const MemoryState *>(&resource.resource());
    return native && native->runtime == runtime && native->kind == MemoryKind::kDevice;
  };
  const bool source = device(copy.source.storage().resource());
  const bool destination = device(copy.destination.storage().resource());
  if (source && destination) return ConstantDirection::kDevice;
  if (destination && copy.source.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
    return ConstantDirection::kUpload;
  if (source && copy.destination.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
    return ConstantDirection::kDownload;
  return std::nullopt;
}

// A complete owned root includes its private transfer padding. An arbitrary
// subview may only transfer the aligned range explicitly inside its span.
Result<bool> constant_device_range(const std::shared_ptr<Runtime> &runtime,
                                  mem::ConstBufferView view, std::size_t bytes) noexcept {
  if (!bytes) return true;
  if (!view.native_handle())
    return std::unexpected(Status::InvalidArgument("Vulkan constant copy has no device backing"));
  const auto &token = *static_cast<const AllocationToken *>(view.native_handle());
  const auto &backing = token.buffer;
  if (!backing || backing->device != runtime->device || view.offset() > backing->bytes ||
      view.size() > backing->bytes - view.offset() || bytes > view.size() ||
      view.offset() > token.logical_bytes || bytes > token.logical_bytes - view.offset())
    return std::unexpected(Status::OutOfRange("Vulkan constant copy exceeds device backing"));
  // Same ownership/TRANSFER usage proof as copy_span; unknown native imports
  // cannot acquire transfer authority merely through a MemoryResourceRef.
  if (!backing->allocation_bytes) return false;
  HYLE_ASSIGN_OR_RETURN(auto padded, buffer_capacity(bytes));
  HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(view.size()));
  const auto available = view.offset() == 0 && bytes == token.logical_bytes &&
          view.size() == token.logical_bytes &&
          capacity == backing->bytes ? backing->bytes : view.size();
  return (backing->offset % 4 + view.offset() % 4) % 4 == 0 && padded <= available;
}
} // namespace

Result<bool> MemoryAccess::can_copy_constant(const std::shared_ptr<Runtime> &runtime,
                                            const ConstantCopy &copy) noexcept {
  if (!runtime || !copy.destination.valid() || !copy.source.valid() ||
      copy.destination.description() != copy.source.description())
    return std::unexpected(Status::InvalidArgument("invalid Vulkan constant copy descriptions"));
  const auto direction = constant_direction(runtime, copy);
  if (!direction) return false;
  HYLE_ASSIGN_OR_RETURN(auto bytes, copy.destination.description().required_bytes());
  if (*direction != ConstantDirection::kDownload) {
    HYLE_ASSIGN_OR_RETURN(auto target, constant_device_range(runtime, copy.destination.storage(), bytes));
    if (!target) return false;
  }
  if (*direction != ConstantDirection::kUpload) {
    HYLE_ASSIGN_OR_RETURN(auto source, constant_device_range(runtime, copy.source.storage(), bytes));
    if (!source) return false;
  }
  if (*direction == ConstantDirection::kDevice) {
    // CopyRecording remains exact and four-byte aligned. Odd owned carriers
    // use the existing padded TransferSlot, after both endpoint proofs above.
    if (bytes % 4) {
      const auto &source = *static_cast<const AllocationToken *>(copy.source.storage().native_handle());
      const auto &target = *static_cast<const AllocationToken *>(copy.destination.storage().native_handle());
      if (source.buffer->handle == target.buffer->handle) return false;
    } else if (bytes) {
      HYLE_ASSIGN_OR_RETURN(auto source, copy_span(runtime, copy.source.storage()));
      HYLE_ASSIGN_OR_RETURN(auto target, copy_span(runtime, copy.destination.storage()));
      if (source.backing->handle == target.backing->handle && source.offset != target.offset &&
          (source.offset < target.offset ? target.offset - source.offset < bytes
                                         : source.offset - target.offset < bytes))
        return false;
    }
  }
  std::lock_guard lock(runtime->mutex);
  if (runtime->closed)
    return std::unexpected(Status::FailedPrecondition("Vulkan constant source is closed"));
  HYLE_RETURN_IF_ERROR(runtime->failure);
  return true;
}

Result<std::size_t> MemoryAccess::constant_copy_staging_bytes(
    const std::shared_ptr<Runtime> &runtime, const ConstantCopy &copy) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto supported, can_copy_constant(runtime, copy));
  if (!supported)
    return std::unexpected(Status::Unimplemented(
        "Vulkan constant copy has unsupported resources or transfer ranges"));
  HYLE_ASSIGN_OR_RETURN(auto bytes, copy.destination.description().required_bytes());
  if (*constant_direction(runtime, copy) == ConstantDirection::kDevice && bytes % 4 == 0)
    return std::size_t{0};
  if (!bytes) return std::size_t{0};
  HYLE_ASSIGN_OR_RETURN(auto padded, buffer_capacity(bytes));
  return static_cast<std::size_t>(padded);
}

Status MemoryAccess::copy_constants(const std::shared_ptr<Runtime> &runtime,
    std::span<const ConstantCopy> copies, mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid())
    return Status::InvalidArgument("invalid Vulkan constant copy allocator");
  try {
    std::size_t host_maximum = 0, device_maximum = 0;
    // Validate the complete batch, including both device endpoints and padding,
    // before retaining metadata or accepting any native copy/host write.
    for (const auto &copy : copies) {
      HYLE_ASSIGN_OR_RETURN(auto staging, constant_copy_staging_bytes(runtime, copy));
      host_maximum = std::max(host_maximum, staging);
      if (*constant_direction(runtime, copy) == ConstantDirection::kDevice && !staging) {
        HYLE_ASSIGN_OR_RETURN(auto bytes, copy.destination.description().required_bytes());
        device_maximum = std::max(device_maximum, bytes);
      }
    }
    if (!host_maximum && !device_maximum) return Status::Ok();
    if (host_maximum > static_cast<std::size_t>(INT64_MAX))
      return Status::OutOfRange("Vulkan constant staging exceeds tensor range");
    struct PreparedCopy final {
      Tensor source, destination;
      ConstantDirection direction = ConstantDirection::kUpload;
    };
    std::vector<PreparedCopy, mem::StdAllocator<PreparedCopy>> prepared{
        mem::StdAllocator<PreparedCopy>(allocator)};
    prepared.reserve(copies.size());
    // Retain every actual source/destination backing before any accepted work.
    for (const auto &copy : copies) {
      auto &entry = prepared.emplace_back();
      entry.direction = *constant_direction(runtime, copy);
      HYLE_ASSIGN_OR_RETURN(auto bytes, copy.destination.description().required_bytes());
      if (!bytes) continue;
      if (entry.direction != ConstantDirection::kDownload) {
        HYLE_ASSIGN_OR_RETURN(entry.destination, tensor_view(runtime, copy.destination.as_const(), allocator));
      }
      if (entry.direction != ConstantDirection::kUpload) {
        HYLE_ASSIGN_OR_RETURN(entry.source, tensor_view(runtime, copy.source, allocator));
      }
    }
    TransferSlot slot;
    if (host_maximum) {
      HYLE_ASSIGN_OR_RETURN(auto description, TensorDesc::dense(DataType::kUInt8,
          std::array<std::int64_t, 1>{static_cast<std::int64_t>(host_maximum)}));
      HYLE_ASSIGN_OR_RETURN(auto buffer, allocate_buffer(runtime->device, host_maximum, true, false,
          {.allocator = allocator, .transfer_access = true}));
      HYLE_ASSIGN_OR_RETURN(slot, TransferAccess::create(runtime,
          Access::tensor(description, std::move(buffer)), host_maximum, allocator));
    }
    std::shared_ptr<CopyRecording> recording;
    if (device_maximum) {
      HYLE_ASSIGN_OR_RETURN(recording, prepare_copy_recording_runtime(runtime, 1, allocator, device_maximum));
    }
    for (std::size_t index = 0; index < copies.size(); ++index) {
      const auto &entry = prepared[index];
      if (!entry.source.valid() && !entry.destination.valid()) continue;
      // Completion and its observation retire before this shared cold slot is
      // reused. A native wait or final host write error is fatal, never fallback.
      if (entry.direction == ConstantDirection::kUpload) {
        HYLE_ASSIGN_OR_RETURN(auto completed, slot.upload(entry.destination, copies[index].source));
        HYLE_RETURN_IF_ERROR(completed.wait());
      } else if (entry.direction == ConstantDirection::kDownload) {
        HYLE_ASSIGN_OR_RETURN(auto completed, slot.download(entry.source));
        HYLE_RETURN_IF_ERROR(completed.wait());
        HYLE_RETURN_IF_ERROR(slot.read(copies[index].destination));
      } else {
        HYLE_ASSIGN_OR_RETURN(auto bytes, copies[index].destination.description().required_bytes());
        if (bytes % 4) {
          HYLE_ASSIGN_OR_RETURN(auto completed, TransferAccess::copy(slot, entry.source, entry.destination));
          HYLE_RETURN_IF_ERROR(completed.wait());
          continue;
        }
        HYLE_ASSIGN_OR_RETURN(auto source, copies[index].source.storage().subview(0, bytes));
        HYLE_ASSIGN_OR_RETURN(auto target, copies[index].destination.storage().subview(0, bytes));
        const std::array<compute::detail::CopyBackendRange, 1> ranges{{{target, source}}};
        HYLE_RETURN_IF_ERROR(bind_copy_recording(*recording, ranges));
        HYLE_ASSIGN_OR_RETURN(auto completed, submit_copy_recording(*recording));
        HYLE_RETURN_IF_ERROR(completed.wait());
      }
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan constant transfer metadata");
  }
}

Result<mem::Buffer> MemoryAccess::export_storage(
    const std::shared_ptr<Runtime> &runtime, mem::MemoryResourceRef resource,
    const Tensor &source, mem::AllocatorRef allocator) noexcept {
  HYLE_RETURN_IF_ERROR(validate_resource(runtime, resource));
  if (!source.valid() || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid Vulkan backing export"));
  const auto &state = static_cast<const MemoryState &>(resource.resource());
  const auto &native = Access::buffer(source);
  const bool unbound_view = native->is_view && native->handle == VK_NULL_HANDLE && native->bytes == 0;
  if (native->device != runtime->device || (!unbound_view && native->bytes == 0) ||
      (state.kind == MemoryKind::kStaging) != (native->mapped != nullptr) ||
      !std::in_range<std::size_t>(native->bytes))
    return std::unexpected(Status::InvalidArgument("Vulkan backing export device/kind differs"));
  auto bytes = native->bytes;
  if (unbound_view) {
    // The actual native view object is stable, but its external input root
    // is bound later. This metadata is not exposed until cold bind succeeds.
    HYLE_ASSIGN_OR_RETURN(auto logical, source.description().required_bytes());
    HYLE_ASSIGN_OR_RETURN(bytes, buffer_capacity(logical));
  }
  HYLE_ASSIGN_OR_RETURN(auto authorized_bytes, source.description().required_bytes());
  auto *memory = allocator.allocate(sizeof(AllocationToken), alignof(AllocationToken));
  if (!memory)
    return std::unexpected(Status::ResourceExhausted("Vulkan backing export metadata"));
  auto *token = std::construct_at(static_cast<AllocationToken *>(memory), native, allocator, authorized_bytes, state.retain_controller());
  auto adopted = mem::Buffer::adopt(resource, {token, static_cast<std::size_t>(bytes), 1});
  if (!adopted) {
    auto *controller = token->controller;
    std::destroy_at(token);
    allocator.deallocate(token, sizeof(AllocationToken), alignof(AllocationToken));
    if (controller) controller->release_controller();
  }
  return adopted;
}

Result<RetainedBackingFacts> MemoryAccess::retained_backing(
    const std::shared_ptr<Runtime> &runtime, mem::MemoryResourceRef resource,
    const Tensor &source, const std::shared_ptr<Buffer> &root,
    VkDeviceSize relative_offset) noexcept {
  HYLE_RETURN_IF_ERROR(validate_resource(runtime, resource));
  if (!source.valid() || !root)
    return std::unexpected(Status::InvalidArgument("Vulkan retained backing has no typed source/root"));
  const auto &state = static_cast<const MemoryState &>(resource.resource());
  const auto &native = Access::buffer(source);
  // This proof is deliberately limited to an original owning device block.
  // External import owners are opaque and cannot be recovered by a cast.
  if (state.kind != MemoryKind::kDevice || root->device != runtime->device ||
      root->is_view || root->external_owner || root->memory == VK_NULL_HANDLE ||
      root->handle == VK_NULL_HANDLE || root->mapped || root->offset ||
      root->bytes == 0 || root->bytes == SIZE_MAX ||
      root->allocation_bytes < root->bytes || !std::in_range<std::size_t>(root->bytes) ||
      relative_offset > root->bytes || native->device != root->device ||
      native->handle != root->handle || native->mapped ||
      native->offset != relative_offset || native->bytes != root->bytes - relative_offset ||
      native->allocation_bytes != root->allocation_bytes)
    return std::unexpected(Status::InvalidArgument("Vulkan retained backing is not its exact owned block"));
  if (native.get() == root.get()) {
    if (relative_offset || native.owner_before(root) || root.owner_before(native))
      return std::unexpected(Status::InvalidArgument("Vulkan retained root ownership differs"));
  } else if (!native->is_view || native->memory != VK_NULL_HANDLE ||
      native->external_owner.get() != root.get() ||
      native->external_owner.owner_before(root) || root.owner_before(native->external_owner)) {
    return std::unexpected(Status::InvalidArgument("Vulkan retained alias does not own the nominated root"));
  }
  HYLE_ASSIGN_OR_RETURN(auto bytes, source.description().required_bytes());
  if (bytes > native->bytes)
    return std::unexpected(Status::InvalidArgument("Vulkan retained tensor exceeds its original backing"));
  return RetainedBackingFacts{&retained_buffer_authority, root.get(),
      static_cast<std::size_t>(root->bytes), static_cast<std::size_t>(relative_offset)};
}

static Result<std::shared_ptr<Buffer>> empty_carrier(const std::shared_ptr<Runtime> &runtime,
    mem::MemoryResourceRef resource, mem::AllocatorRef allocator, ResourceBudget budget) noexcept {
  std::lock_guard lock(runtime->mutex);
  if (runtime->closed) return std::unexpected(Status::FailedPrecondition("Vulkan context is closed"));
  HYLE_RETURN_IF_ERROR(runtime->failure);
  return allocate_buffer(runtime->device, 0, false, true,
      {.budget = std::move(budget), .resource = resource, .allocator = allocator});
}
Result<Tensor> MemoryAccess::tensor_view(const std::shared_ptr<Runtime> &runtime,
    const ConstTensorView &source, mem::AllocatorRef allocator, ResourceBudget budget) noexcept {
  if (!source.valid() || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid Vulkan tensor view import"));
  const auto storage = source.storage();
  HYLE_RETURN_IF_ERROR(validate_resource(runtime, storage.resource()));
  const auto &resource = static_cast<const MemoryState &>(storage.resource().resource());
  if (resource.kind != MemoryKind::kDevice)
    return std::unexpected(Status::InvalidArgument("Vulkan tensor inputs require device storage"));
  HYLE_ASSIGN_OR_RETURN(auto bytes, source.description().required_bytes());
  if (!storage.native_handle()) {
    if (bytes != 0 || storage.size() != 0)
      return std::unexpected(Status::InvalidArgument("Vulkan tensor input has no native backing"));
    // Original native descriptors retain a four-byte dummy for empty tensors.
    // This cold allocation has no readable logical payload.
    HYLE_ASSIGN_OR_RETURN(auto empty, empty_carrier(runtime, storage.resource(), allocator, std::move(budget)));
    return Access::tensor(source.description(), std::move(empty));
  }
  const auto &backing = static_cast<const AllocationToken *>(storage.native_handle())->buffer;
  if (!backing || backing->device != runtime->device ||
      storage.offset() > backing->bytes || storage.size() > backing->bytes - storage.offset() ||
      bytes > storage.size())
    return std::unexpected(Status::InvalidArgument("Vulkan tensor input exceeds original backing"));
  if (!bytes && (storage.size() < 4 || ((backing->offset + storage.offset()) & 3u) != 0)) {
    // An empty subspan, including one at a nonzero origin, grants no extra
    // bytes from its root. Supply an isolated native descriptor carrier when
    // the actual authorized span cannot supply an aligned four-byte range.
    HYLE_ASSIGN_OR_RETURN(auto empty, empty_carrier(runtime, storage.resource(), allocator, std::move(budget)));
    return Access::tensor(source.description(), std::move(empty));
  }
  HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(storage.size()));
  if (storage.offset() == 0 && capacity == backing->bytes)
    return Access::tensor(source.description(), backing);
  try {
    auto view = std::allocate_shared<Buffer>(mem::StdAllocator<Buffer>(allocator));
    view->device = backing->device;
    bind_view_buffer(*view, backing, storage.offset());
    view->bytes = storage.size();
    return Access::tensor(source.description(), std::move(view));
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan tensor input view metadata"));
  }
}

Result<std::shared_ptr<Buffer>> MemoryAccess::buffer(
    const MemoryResource &resource, const mem::Buffer &allocation) noexcept {
  if (!resource.valid() || allocation.resource() != resource.ref() || allocation.native_handle() == nullptr)
    return std::unexpected(Status::InvalidArgument("Vulkan allocation resource differs"));
  return static_cast<const AllocationToken *>(allocation.native_handle())->buffer;
}
const std::shared_ptr<Runtime> &MemoryAccess::runtime(const MemoryResource &resource) noexcept {
  return resource.state_->runtime;
}
Result<CopySpan> MemoryAccess::copy_span(const std::shared_ptr<Runtime> &runtime,
                                       mem::ConstBufferView view) noexcept {
  if (!runtime || !view.valid())
    return std::unexpected(Status::InvalidArgument("Vulkan copy view is invalid"));
  const auto *resource = dynamic_cast<const MemoryState *>(&view.resource().resource());
  if (!resource || resource->runtime != runtime)
    return std::unexpected(Status::InvalidArgument("Vulkan copy resource has another Device/family"));
  if (view.size() == 0 && view.native_handle() == nullptr)
    return CopySpan{{}, {reinterpret_cast<std::uintptr_t>(runtime->device.get()), 0, 0, 0, 4}, 0};
  if (!view.native_handle()) return std::unexpected(Status::InvalidArgument("Vulkan copy token is empty"));
  const auto &buffer = static_cast<const AllocationToken *>(view.native_handle())->buffer;
  if (!buffer || buffer->device != runtime->device || buffer->allocation_bytes == 0)
    return std::unexpected(Status::Unimplemented("Vulkan imported copy backing/usage is not proven"));
  // Every original owned root has independent DeviceMemory and TRANSFER_SRC/
  // DST usage. Its views preserve the same VkBuffer, byte offset and actual
  // allocation_bytes, while clearing memory to prevent duplicate frees.
  // Unknown external imports have no such allocation/usage proof.
  if (view.offset() > buffer->bytes || view.size() > buffer->bytes - view.offset() ||
      buffer->offset > SIZE_MAX || view.offset() > SIZE_MAX - static_cast<std::size_t>(buffer->offset))
    return std::unexpected(Status::OutOfRange("Vulkan copy exceeds native backing"));
  const auto origin = static_cast<std::size_t>(buffer->offset) + view.offset();
  std::uint64_t backing_key = 0;
  static_assert(sizeof(buffer->handle) <= sizeof(backing_key));
  std::memcpy(&backing_key, &buffer->handle, sizeof(buffer->handle));
  return CopySpan{buffer, {reinterpret_cast<std::uintptr_t>(runtime->device.get()), backing_key,
                         origin, view.size(), 4}, static_cast<VkDeviceSize>(origin)};
}
} // namespace detail

MemoryResource::MemoryResource() noexcept = default;
MemoryResource::MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept : state_(std::move(state)) {}
MemoryResource::MemoryResource(MemoryResource &&) noexcept = default;
MemoryResource &MemoryResource::operator=(MemoryResource &&other) noexcept {
  if (this != &other) {
    if (auto *previous = state_.release()) previous->release_controller();
    state_ = std::move(other.state_);
  }
  return *this;
}
MemoryResource::~MemoryResource() {
  if (auto *state = state_.release()) state->release_controller();
}
Result<MemoryResource> MemoryResource::create(const Context &context, MemoryKind kind,
                                             mem::AllocatorRef allocator) noexcept {
  if (!context.valid() || !allocator.valid() || (kind != MemoryKind::kDevice && kind != MemoryKind::kStaging))
    return std::unexpected(Status::InvalidArgument("invalid Vulkan memory resource creation"));
  try {
    auto state = std::make_unique<detail::MemoryState>();
    state->retain_tokens = true;
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
      {.budget = options.resource_budget, .resource = ref(), .category = options.category, .allocator = state_->allocator,
       .transfer_access = true}));
  auto *controller = state_->retain_controller();
  const auto retained = detail::retain_buffer_controller(*buffer, controller,
      +[](void *owner) noexcept { static_cast<detail::MemoryState *>(owner)->release_controller(); },
      state_->allocator);
  if (!retained.ok()) {
    controller->release_controller();
    return std::unexpected(retained);
  }
  return detail::Access::tensor(description, std::move(buffer));
}
Result<Tensor> MemoryResource::binding(ReadLease source, const TensorDesc &description,
                                      LeaseImportOptions options) const noexcept {
  if(!valid())return std::unexpected(Status::InvalidArgument("Vulkan lease resource is empty"));
  return detail::MemoryAccess::lease_tensor(state_->runtime,ref(),std::move(source),description,std::move(options));
}
Result<Tensor> detail::MemoryAccess::lease_tensor(const std::shared_ptr<Runtime>&runtime,
    mem::MemoryResourceRef resource,ReadLease source,const TensorDesc&description,
    LeaseImportOptions options) noexcept {
  HYLE_RETURN_IF_ERROR(validate_resource(runtime,resource));
  if (!options.allocator.valid() || static_cast<const MemoryState&>(resource.resource()).kind != MemoryKind::kDevice)
    return std::unexpected(Status::InvalidArgument("Vulkan lease binding requires a device resource/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, source.tensor(description));
  const auto storage = tensor.storage();
  if (storage.resource() != resource)
    return std::unexpected(Status::InvalidArgument("Vulkan lease belongs to another resource"));
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto capacity, detail::buffer_capacity(bytes));
  std::shared_ptr<detail::Buffer> backing;
  if (storage.native_handle()) {
    backing = static_cast<const detail::AllocationToken *>(storage.native_handle())->buffer;
    if (!backing || backing->device != runtime->device || storage.offset() > backing->bytes ||
        storage.size() > backing->bytes - storage.offset())
      return std::unexpected(Status::InvalidArgument("Vulkan lease origin/tail capacity exceeds owned range"));
  } else if (bytes || storage.size() || storage.offset()) {
    return std::unexpected(Status::InvalidArgument("Vulkan lease has no native backing"));
  }
  const bool carrier = !backing || ((backing->offset + storage.offset()) & 3u) != 0 ||
      capacity > storage.size();
  if (bytes && carrier)
    return std::unexpected(Status::InvalidArgument("Vulkan lease origin/tail capacity exceeds owned range"));
  if (carrier) {
    // A zero logical lease has no readable payload. Retain that exact lease,
    // using a private native descriptor carrier only when its real span has
    // none. Existing aligned four-byte backing is shared without allocation.
    HYLE_ASSIGN_OR_RETURN(backing, detail::empty_carrier(runtime, resource,
        options.allocator, options.resource_budget));
  }
  try {
    auto allocator = options.allocator;
    if (options.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
    }
    auto retained = std::allocate_shared<detail::LeaseViewOwner>(
        mem::StdAllocator<detail::LeaseViewOwner>(allocator));
    retained->backing = backing;
    retained->source = std::move(source);
    auto view = std::allocate_shared<detail::Buffer>(mem::StdAllocator<detail::Buffer>(allocator));
    view->device = backing->device;
    detail::bind_view_buffer(*view, backing, carrier ? 0 : storage.offset());
    view->bytes = carrier ? backing->bytes : storage.size();
    // A view exposes its Buffer parent while retaining the original lease.
    view->external_owner = std::shared_ptr<const void>(retained, backing.get());
    return detail::Access::tensor(description, std::move(view));
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan lease binding metadata"));
  }
}

template<class Frame>static Result<Tensor> paged_binding(const std::shared_ptr<detail::Runtime>&runtime,
    mem::MemoryResourceRef resource,Frame frame,LeaseImportOptions options)noexcept{
  HYLE_RETURN_IF_ERROR(detail::MemoryAccess::validate_resource(runtime,resource));
  if(!options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("Vulkan paged binding requires a resource/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto tensor,compute::detail::PagedArenaAccess::tensor(frame));
  const auto storage=tensor.storage();auto description=tensor.description();
  if constexpr(std::is_same_v<Frame,PagedRowCacheRead>){const auto cap=frame.layout().schema().capacity();
    HYLE_ASSIGN_OR_RETURN(description,row_cache_carrier_description(cap.encoding,std::int64_t(std::uint64_t(cap.max_pages)*cap.page_size)));}
  if(storage.resource()!=resource || storage.native_handle()==nullptr ||
      static_cast<const detail::MemoryState&>(resource.resource()).kind!=MemoryKind::kDevice)
    return std::unexpected(Status::InvalidArgument("Vulkan page arena belongs to another device resource"));
  HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());HYLE_ASSIGN_OR_RETURN(auto capacity,detail::buffer_capacity(bytes));
  const auto&backing=static_cast<const detail::AllocationToken*>(storage.native_handle())->buffer;
  if(!bytes || ((backing->offset+storage.offset())&3u)!=0 || capacity>storage.size() ||
      storage.offset()>backing->bytes || storage.size()>backing->bytes-storage.offset())
    return std::unexpected(Status::InvalidArgument("Vulkan paged origin/tail exceeds original capacity"));
  try{
    auto allocator=options.allocator;if(options.resource_budget.valid()){
      HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());
    }
    auto retained=std::allocate_shared<detail::PagedViewOwner<Frame>>(mem::StdAllocator<detail::PagedViewOwner<Frame>>(allocator));
    retained->backing=backing;retained->frame=std::move(frame);
    auto view=std::allocate_shared<detail::Buffer>(mem::StdAllocator<detail::Buffer>(allocator));
    view->device=backing->device;detail::bind_view_buffer(*view,backing,storage.offset());
    view->bytes=storage.size();
    // The alias pointer is the Buffer parent; its control block keeps pages.
    view->external_owner=std::shared_ptr<const void>(retained,backing.get());
    return detail::Access::tensor(description,std::move(view));
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Vulkan paged binding metadata"));}
}

Result<Tensor> detail::PagedBatchAccess::binding(const MemoryResource&memory,
    PagedArenaRead frame,LeaseImportOptions options)noexcept{
  if(!memory.valid())return std::unexpected(Status::InvalidArgument("Vulkan page resource is empty"));
  return paged_binding(MemoryAccess::runtime(memory),memory.ref(),std::move(frame),options);
}
Result<Tensor> detail::PagedBatchAccess::binding(const MemoryResource&memory,
    PagedRowCacheRead frame,LeaseImportOptions options)noexcept{
  if(!memory.valid())return std::unexpected(Status::InvalidArgument("Vulkan page resource is empty"));
  return paged_binding(MemoryAccess::runtime(memory),memory.ref(),std::move(frame),options);
}
Result<Tensor> detail::PagedBatchAccess::binding(const MemoryResource&memory,
    PagedArenaPendingRead frame,LeaseImportOptions options)noexcept{
  if(!memory.valid())return std::unexpected(Status::InvalidArgument("Vulkan page resource is empty"));
  return paged_binding(MemoryAccess::runtime(memory),memory.ref(),std::move(frame),options);
}
Result<Tensor> detail::MemoryAccess::paged_tensor(const std::shared_ptr<Runtime>&runtime,
    mem::MemoryResourceRef resource,PagedArenaRead frame,LeaseImportOptions options)noexcept{
  return paged_binding(runtime,resource,std::move(frame),options);
}
Result<Tensor> detail::MemoryAccess::paged_tensor(const std::shared_ptr<Runtime>&runtime,
    mem::MemoryResourceRef resource,PagedArenaPendingRead frame,LeaseImportOptions options)noexcept{
  return paged_binding(runtime,resource,std::move(frame),options);
}

Result<ReadLease> MemoryResource::import_readonly(const Tensor &source, LeaseImportOptions options) const noexcept {
  if (!valid() || !source.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid Vulkan readonly import"));
  const auto &native = detail::Access::buffer(source);
  if (native->device != state_->runtime->device || native->bytes == 0 ||
      (state_->kind == MemoryKind::kStaging) != (native->mapped != nullptr))
    return std::unexpected(Status::InvalidArgument("Vulkan import device/memory kind mismatch"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto authorized_bytes, source.description().required_bytes());
  auto *storage = allocator.allocate(sizeof(detail::AllocationToken), alignof(detail::AllocationToken));
  if (!storage) return std::unexpected(Status::ResourceExhausted("Vulkan import token metadata"));
  auto *token = std::construct_at(static_cast<detail::AllocationToken *>(storage), native, allocator, authorized_bytes, state_->retain_controller());
  auto adopted = mem::Buffer::adopt(ref(), {token, static_cast<std::size_t>(native->bytes), 1});
  if (!adopted) {
    auto *controller = token->controller;
    std::destroy_at(token);
    allocator.deallocate(token, sizeof(detail::AllocationToken), alignof(detail::AllocationToken));
    if (controller) controller->release_controller();
    return std::unexpected(std::move(adopted).error());
  }
  return ReadLease::adopt(std::move(*adopted), {.allocator = allocator});
}

Result<NativeStorage> MemoryResource::storage_info(const Tensor &tensor) const noexcept {
  if (!valid() || !tensor.valid() || detail::Access::buffer(tensor)->device != state_->runtime->device)
    return std::unexpected(Status::InvalidArgument("Vulkan tensor belongs to another native Device"));
  const auto &buffer = *detail::Access::buffer(tensor);
  return NativeStorage{buffer.handle, buffer.offset, buffer.bytes,
      buffer.allocation_bytes != 0 ? std::optional<std::size_t>(buffer.allocation_bytes) : std::nullopt,
      buffer.transfer_eligible};
}
Result<detail::CopySpan> detail::result_copy_span(const std::shared_ptr<Runtime>&runtime,
    const Results&result,std::size_t index,const TensorDesc&description)noexcept{
  const auto tensors=result.tensors();
  if(index>=tensors.size())return std::unexpected(Status::OutOfRange("Vulkan result copy output index"));
  const auto&tensor=tensors[index];
  if(!tensor.valid() || tensor.description()!=description || !description.is_plain() || !description.is_dense())
    return std::unexpected(Status::InvalidArgument("Vulkan result copy dense carrier differs"));
  const auto&buffer=Access::buffer(tensor);
  HYLE_ASSIGN_OR_RETURN(auto bytes,description.required_bytes());
  if(!runtime || !buffer || buffer->device!=runtime->device || !buffer->allocation_bytes ||
      buffer->offset>SIZE_MAX || bytes>buffer->bytes)
    return std::unexpected(Status::InvalidArgument("Vulkan result copy backing/Device is not qualified"));
  std::uint64_t key=0;static_assert(sizeof(buffer->handle)<=sizeof(key));
  std::memcpy(&key,&buffer->handle,sizeof(buffer->handle));
  return CopySpan{buffer,{reinterpret_cast<std::uintptr_t>(runtime->device.get()),key,
      static_cast<std::size_t>(buffer->offset),bytes,4},buffer->offset};
}
} // namespace hyle::compute::vulkan
