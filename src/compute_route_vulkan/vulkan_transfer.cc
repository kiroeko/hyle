#include "vulkan_internal.h"

#include <cstring>
#include <new>

namespace hyle::compute::vulkan {
namespace {

Status transfer(const std::shared_ptr<detail::Runtime> &runtime,
                const Tensor &tensor, const std::byte *source,
                std::byte *destination, std::size_t bytes,
                bool upload) noexcept {
  if (!runtime || !tensor ||
      (bytes != 0 && (upload ? source == nullptr : destination == nullptr)))
    return Status::InvalidArgument("invalid Vulkan transfer");
  const auto &native = detail::Access::buffer(tensor);
  const auto required = tensor.description().required_bytes();
  if (!required || native->device != runtime->device || bytes != *required ||
      bytes > native->bytes)
    return Status::InvalidArgument(
        "Vulkan transfer device or byte range mismatch");
  // The explicit transfer is synchronous, but still uses the same retained
  // submission machinery: a failed observation cannot free active staging.
  try {
    {
      std::lock_guard lock(runtime->mutex);
      if (upload && runtime->closed)
        return Status::FailedPrecondition("Vulkan context is closed");
      if (!runtime->failure.ok())
        return runtime->failure;
    }
    if (bytes == 0)
      return Status::Ok();
    auto staging = detail::allocate_buffer(runtime->device, bytes, true);
    if (!staging)
      return std::move(staging).error();
    auto run = detail::create_run(runtime);
    if (!run)
      return std::move(run).error();
    (*run)->retained_buffers = {native, *staging};
    auto &device = *runtime->device;
    VkMappedMemoryRange mapped{};
    mapped.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mapped.memory = (*staging)->memory;
    mapped.size = VK_WHOLE_SIZE;
    if (upload) {
      std::memcpy((*staging)->mapped, source, bytes);
      if (((*staging)->memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ==
          0) {
        auto status = detail::native_status(
            device.fn.FlushMappedMemoryRanges(device.native.device, 1, &mapped),
            "vkFlushMappedMemoryRanges");
        if (!status.ok())
          return status;
      }
    }
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    auto status = detail::native_status(
        device.fn.BeginCommandBuffer((*run)->command, &begin),
        "vkBeginCommandBuffer");
    if (!status.ok())
      return status;
    if (upload)
      detail::memory_barrier(
          device, (*run)->command, VK_PIPELINE_STAGE_HOST_BIT,
          VK_ACCESS_HOST_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);
    else
      detail::memory_barrier(
          device, (*run)->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy copy{};
    copy.srcOffset = upload ? 0 : native->offset;
    copy.dstOffset = upload ? native->offset : 0;
    copy.size = bytes;
    device.fn.CmdCopyBuffer(
        (*run)->command, upload ? (*staging)->handle : native->handle,
        upload ? native->handle : (*staging)->handle, 1, &copy);
    detail::memory_barrier(
        device, (*run)->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        upload ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
               : VK_PIPELINE_STAGE_HOST_BIT,
        upload ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_HOST_READ_BIT);
    status = detail::native_status(device.fn.EndCommandBuffer((*run)->command),
                                   "vkEndCommandBuffer");
    if (!status.ok())
      return status;
    Result<Completion> completion;
    {
      std::lock_guard lock(runtime->mutex);
      completion = detail::submit_locked(*run, !upload);
    }
    if (!completion)
      return std::move(completion).error();
    status = completion->wait();
    if (!status.ok())
      return status;
    if (!upload) {
      if (((*staging)->memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ==
          0) {
        status = detail::native_status(device.fn.InvalidateMappedMemoryRanges(
                                           device.native.device, 1, &mapped),
                                       "vkInvalidateMappedMemoryRanges");
        if (!status.ok())
          return status;
      }
      std::memcpy(destination, (*staging)->mapped, bytes);
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan transfer metadata");
  }
}

} // namespace

Result<Tensor>
Context::upload(const TensorDesc &description,
                std::span<const std::byte> bytes) const noexcept {
  auto tensor = allocate(description);
  if (!tensor)
    return std::unexpected(std::move(tensor).error());
  auto status = upload(*tensor, bytes);
  if (!status.ok())
    return std::unexpected(status);
  return tensor;
}
Status Context::upload(const Tensor &destination,
                       std::span<const std::byte> bytes) const noexcept {
  return transfer(state_, destination, bytes.data(), nullptr, bytes.size(),
                  true);
}
Status Context::download(const Tensor &source,
                         std::span<std::byte> bytes) const noexcept {
  return transfer(state_, source, nullptr, bytes.data(), bytes.size(), false);
}

} // namespace hyle::compute::vulkan
