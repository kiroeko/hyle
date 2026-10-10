#include "vulkan_transfer_internal.h"

#include <algorithm>
#include <cstring>
#include <new>

namespace hyle::compute::vulkan {
namespace detail {
struct TransferState final {
  std::shared_ptr<Runtime> runtime;
  std::shared_ptr<Run> run;
  Tensor staging;
  TensorDesc description;
  std::size_t capacity = 0;
  bool download = false, submitted = false;
};
} // namespace detail
namespace {
Status validate_slot(const detail::TransferState &slot, const Tensor &tensor) noexcept {
  if (!tensor.valid() || detail::Access::buffer(tensor)->device != slot.runtime->device)
    return Status::InvalidArgument("Vulkan transfer requires the exact native Device/family");
  HYLE_ASSIGN_OR_RETURN(auto bytes, tensor.description().required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto padded, detail::buffer_capacity(bytes));
  const auto &buffer = *detail::Access::buffer(tensor);
  if (bytes > slot.capacity || padded > buffer.bytes || buffer.offset % 4 != 0)
    return Status::InvalidArgument("Vulkan transfer exceeds capacity or aligned native range");
  if (slot.runtime->closed) return Status::FailedPrecondition("Vulkan Context is closed");
  if (!slot.runtime->failure.ok()) return slot.runtime->failure;
  if (slot.run->pending || slot.run->completion_owner || slot.run->observers || slot.run->retiring)
    return Status::FailedPrecondition("Vulkan transfer prefix is still retained");
  if (!slot.run->terminal.ok()) return slot.run->terminal;
  return Status::Ok();
}
Status visibility(const detail::Buffer &buffer, bool flush) noexcept {
  if (buffer.memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return Status::Ok();
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = buffer.memory;
  range.size = VK_WHOLE_SIZE;
  auto &device = *buffer.device;
  return detail::native_status(flush ? device.fn.FlushMappedMemoryRanges(device.native.device, 1, &range)
                                    : device.fn.InvalidateMappedMemoryRanges(device.native.device, 1, &range),
                               flush ? "vkFlushMappedMemoryRanges" : "vkInvalidateMappedMemoryRanges");
}
Status record(detail::TransferState &slot, const Tensor &tensor, bool upload) noexcept {
  auto &device = *slot.runtime->device;
  const auto &staging = detail::Access::buffer(slot.staging);
  const auto &native = detail::Access::buffer(tensor);
  HYLE_ASSIGN_OR_RETURN(auto bytes, tensor.description().required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto padded, detail::buffer_capacity(bytes));
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  HYLE_RETURN_IF_ERROR(detail::native_status(device.fn.BeginCommandBuffer(slot.run->command, &begin), "vkBeginCommandBuffer"));
  // Scope the hazard to these exact buffers. Independent old compute/storage
  // does not acquire a fabricated global read/write dependency from this copy.
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = upload ? staging->handle : native->handle;
  barrier.offset = upload ? 0 : native->offset;
  barrier.size = padded;
  barrier.srcAccessMask = upload ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  device.fn.CmdPipelineBarrier(slot.run->command,
      upload ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
  if (bytes != 0) {
    VkBufferCopy copy{upload ? 0 : native->offset, upload ? native->offset : 0, padded};
    device.fn.CmdCopyBuffer(slot.run->command, upload ? staging->handle : native->handle,
                            upload ? native->handle : staging->handle, 1, &copy);
  }
  barrier.buffer = upload ? native->handle : staging->handle;
  barrier.offset = upload ? native->offset : 0;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = upload ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_HOST_READ_BIT;
  device.fn.CmdPipelineBarrier(slot.run->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
      upload ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_HOST_BIT,
      0, 0, nullptr, 1, &barrier, 0, nullptr);
  HYLE_RETURN_IF_ERROR(detail::native_status(device.fn.EndCommandBuffer(slot.run->command), "vkEndCommandBuffer"));
  slot.run->retained_buffers[0] = staging;
  slot.run->retained_buffers[1] = native;
  slot.description = tensor.description();
  slot.download = !upload;
  slot.submitted = false;
  return Status::Ok();
}
} // namespace

TransferSlot::TransferSlot(TransferSlot &&) noexcept = default;
TransferSlot &TransferSlot::operator=(TransferSlot &&) noexcept = default;
TransferSlot::~TransferSlot() = default;
Result<TransferSlot> TransferSlot::create(const Context &context, const MemoryResource &staging,
                                         std::size_t capacity, AllocationOptions options) noexcept {
  if (!context.valid() || !staging.valid() || capacity == 0)
    return std::unexpected(Status::InvalidArgument("invalid Vulkan transfer slot capacity/resources"));
  const auto &runtime = detail::Access::runtime(context);
  if (runtime != detail::MemoryAccess::runtime(staging) ||
      !staging.ref().properties().has(mem::MemoryProperty::kHostAccessible))
    return std::unexpected(Status::InvalidArgument("Vulkan staging resource has another Device/family"));
  if (capacity > static_cast<std::size_t>(INT64_MAX))
    return std::unexpected(Status::OutOfRange("Vulkan transfer capacity exceeds tensor range"));
  // Use an exact raw physical carrier; no math dtype or conversion is implied.
  HYLE_ASSIGN_OR_RETURN(auto shape, TensorDesc::dense(DataType::kUInt8,
      std::span<const std::int64_t>(std::array<std::int64_t, 1>{static_cast<std::int64_t>(capacity)})));
  auto allocator = mem::system_allocator();
  if (options.resource_budget) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  try {
    TransferSlot slot;
    slot.state_ = std::allocate_shared<detail::TransferState>(mem::StdAllocator<detail::TransferState>(allocator));
    slot.state_->runtime = runtime;
    slot.state_->capacity = capacity;
    HYLE_ASSIGN_OR_RETURN(slot.state_->staging, staging.allocate(shape, options));
    HYLE_ASSIGN_OR_RETURN(slot.state_->run, detail::create_run(runtime, allocator));
    slot.state_->run->retained_buffers.resize(2);
    return slot;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan transfer slot metadata"));
  }
}
Result<Completion> TransferSlot::upload(const Tensor &destination, ConstTensorView source) noexcept {
  if (!state_ || !source.valid() || source.description() != destination.description() ||
      !source.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
    return std::unexpected(Status::InvalidArgument("Vulkan upload description/host source differs"));
  std::lock_guard lock(state_->runtime->mutex);
  HYLE_RETURN_IF_ERROR(validate_slot(*state_, destination));
  HYLE_ASSIGN_OR_RETURN(auto bytes, destination.description().required_bytes());
  const auto &buffer = *detail::Access::buffer(state_->staging);
  auto *data = static_cast<std::byte *>(buffer.mapped);
  HYLE_RETURN_IF_ERROR(source.storage().read({data, bytes}));
  HYLE_ASSIGN_OR_RETURN(auto padded, detail::buffer_capacity(bytes));
  std::fill(data + bytes, data + padded, std::byte{});
  HYLE_RETURN_IF_ERROR(visibility(buffer, true));
  HYLE_RETURN_IF_ERROR(record(*state_, destination, true));
  auto completion = detail::submit_locked(state_->run);
  if (completion) state_->submitted = true;
  return completion;
}

Result<Completion> detail::TransferAccess::upload(
    TransferSlot &slot, const Tensor &destination, std::span<const UploadPart> parts,
    std::size_t used) noexcept {
  if (!slot.state_) return std::unexpected(Status::FailedPrecondition("empty Vulkan transfer slot"));
  auto &state = *slot.state_;
  std::lock_guard lock(state.runtime->mutex);
  HYLE_RETURN_IF_ERROR(validate_slot(state, destination));
  if (used > state.capacity) return std::unexpected(Status::OutOfRange("Vulkan selected upload exceeds capacity"));
  const auto &staging = *Access::buffer(state.staging);
  auto *pointer = static_cast<std::byte *>(staging.mapped);
  for (const auto &part : parts) {
    if (!part.source.valid() ||
        !part.source.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("invalid Vulkan host upload fragment"));
    HYLE_ASSIGN_OR_RETURN(auto bytes, part.source.description().required_bytes());
    if (part.offset > used || bytes > used - part.offset)
      return std::unexpected(Status::OutOfRange("Vulkan fragment range exceeds selected upload"));
  }
  HYLE_ASSIGN_OR_RETURN(auto padded, buffer_capacity(used));
  std::fill(pointer, pointer + padded, std::byte{});
  for (const auto &part : parts) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, part.source.description().required_bytes());
    HYLE_RETURN_IF_ERROR(part.source.storage().read({pointer + part.offset, bytes}));
  }
  HYLE_RETURN_IF_ERROR(visibility(staging, true));
  // The catalog destination is a fixed raw carrier. Only selected bytes plus
  // native alignment padding are transferred, not the whole reserved capacity.
  auto &device = *state.runtime->device;
  const auto &native = Access::buffer(destination);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  HYLE_RETURN_IF_ERROR(native_status(device.fn.BeginCommandBuffer(state.run->command, &begin), "vkBeginCommandBuffer"));
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = staging.handle; barrier.size = padded;
  barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  device.fn.CmdPipelineBarrier(state.run->command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 0, nullptr, 1, &barrier, 0, nullptr);
  if (used != 0) {
    VkBufferCopy copy{0, native->offset, padded};
    device.fn.CmdCopyBuffer(state.run->command, staging.handle, native->handle, 1, &copy);
  }
  barrier.buffer = native->handle; barrier.offset = native->offset;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  device.fn.CmdPipelineBarrier(state.run->command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, 0, nullptr, 1, &barrier, 0, nullptr);
  HYLE_RETURN_IF_ERROR(native_status(device.fn.EndCommandBuffer(state.run->command), "vkEndCommandBuffer"));
  state.run->retained_buffers[0] = Access::buffer(state.staging);
  state.run->retained_buffers[1] = native;
  state.download = false; state.submitted = false;
  auto completion = submit_locked(state.run);
  if (completion) state.submitted = true;
  return completion;
}
Result<Completion> TransferSlot::download(const Tensor &source) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("empty Vulkan transfer slot"));
  std::lock_guard lock(state_->runtime->mutex);
  HYLE_RETURN_IF_ERROR(validate_slot(*state_, source));
  HYLE_RETURN_IF_ERROR(record(*state_, source, false));
  auto completion = detail::submit_locked(state_->run);
  if (completion) state_->submitted = true;
  return completion;
}
Status TransferSlot::read(TensorView destination) const noexcept {
  if (!state_ || !destination.valid()) return Status::InvalidArgument("invalid Vulkan download destination");
  std::lock_guard lock(state_->runtime->mutex);
  if (!state_->submitted || !state_->download || state_->run->pending || state_->run->observers ||
      !state_->run->terminal.ok() || destination.description() != state_->description)
    return Status::FailedPrecondition("Vulkan download bytes are not successfully ready");
  const auto &buffer = *detail::Access::buffer(state_->staging);
  HYLE_RETURN_IF_ERROR(visibility(buffer, false));
  HYLE_ASSIGN_OR_RETURN(auto bytes, destination.description().required_bytes());
  return destination.storage().write({static_cast<const std::byte *>(buffer.mapped), bytes});
}
} // namespace hyle::compute::vulkan
