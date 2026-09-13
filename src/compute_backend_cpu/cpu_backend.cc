#include "hyle/compute/backends/cpu.h"

#include <cstdint>

#include "../compute/backend_internal.h"

namespace hyle::compute::cpu {
namespace {

struct BackendPayload final {};
struct DevicePayload final {
  std::uint32_t index = 0;
};

Result<std::size_t> device_count_impl(const Backend &) noexcept {
  return std::size_t{1};
}

Result<Device> device_impl(const Backend &, std::size_t index) noexcept;

bool same_device_impl(const Device &left, const Device &right) noexcept {
  return detail::DeviceAccess::payload<DevicePayload>(left).index ==
         detail::DeviceAccess::payload<DevicePayload>(right).index;
}

Result<DeviceProperties> properties_impl(const Device &value) noexcept {
  const DevicePayload payload =
      detail::DeviceAccess::payload<DevicePayload>(value);
  if (payload.index != 0)
    return std::unexpected(Status::NotFound("CPU device is not available"));
  return DeviceProperties{
      .backend = kBackendId,
      .device_class = DeviceClass::kCpu,
      .backend_name = "cpu",
      .name = "HYLE host CPU",
      .unified_memory = true,
      .asynchronous_queues = false,
  };
}

Status validate_resource_impl(const Device &value,
                              mem::MemoryResourceRef resource) noexcept {
  if (value != device())
    return Status::InvalidArgument("CPU device identity is invalid");
  if (!resource.valid())
    return Status::InvalidArgument("tensor memory resource is invalid");
  if (!resource.properties().has(mem::MemoryProperty::kHostAccessible)) {
    return Status::FailedPrecondition(
        "CPU backend requires host-accessible tensor storage");
  }
  return Status::Ok();
}

Status create_context_impl(const Device &value, mem::AllocatorRef,
                           void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CPU context output is null");
  *output = nullptr;
  if (value != device())
    return Status::NotFound("CPU device is not available");
  return Status::Ok();
}

void destroy_context_impl(void *, mem::AllocatorRef) noexcept {}

Status create_queue_impl(void *, QueueOptions, mem::AllocatorRef,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CPU queue output is null");
  *output = nullptr;
  return Status::Ok();
}

void destroy_queue_impl(void *, mem::AllocatorRef) noexcept {}
Status begin_sequence_impl(void *, void **sequence) noexcept {
  if (sequence == nullptr)
    return Status::InvalidArgument("CPU sequence output is null");
  *sequence = nullptr;
  return Status::Ok();
}
Completion finish_sequence_impl(void *, void *, bool,
                                Status terminal) noexcept {
  return Completion::completed(std::move(terminal));
}
Status reserve_completions_impl(void *, std::size_t) noexcept {
  return Status::Ok();
}
Result<TransferProperties>
transfer_properties_impl(void *, mem::BufferView,
                         mem::ConstBufferView) noexcept {
  return TransferProperties{.mode = TransferMode::kImmediate, .peer = false};
}
detail::EnqueueResult copy_async_impl(void *, mem::BufferView destination,
                                      mem::ConstBufferView source) noexcept {
  return detail::EnqueueResult::accepted_command(destination.copy_from(source));
}
detail::EnqueueResult fill_async_impl(void *, mem::BufferView destination,
                                      std::byte value) noexcept {
  return detail::EnqueueResult::accepted_command(destination.fill(value));
}
detail::EnqueueResult wait_completion_impl(void *,
                                           const detail::CompletionVTable *,
                                           void *) noexcept {
  return detail::EnqueueResult::rejected(Status::Unimplemented(
      "CPU queue cannot consume an asynchronous backend completion"));
}
bool can_wait_queue_impl(void *, BackendId, void *) noexcept { return false; }
const void *query_extension_impl(void *, BackendExtensionId) noexcept {
  return nullptr;
}
Status drain_context_impl(void *) noexcept { return Status::Ok(); }
Status drain_queue_impl(void *) noexcept { return Status::Ok(); }

const detail::BackendVTable kBackendVTable{
    .id = kBackendId,
    .name = "cpu",
    .device_class = DeviceClass::kCpu,
    .device_count = &device_count_impl,
    .device = &device_impl,
    .same_device = &same_device_impl,
    .properties = &properties_impl,
    .validate_resource = &validate_resource_impl,
    .create_context = &create_context_impl,
    .destroy_context = &destroy_context_impl,
    .create_queue = &create_queue_impl,
    .destroy_queue = &destroy_queue_impl,
    .begin_sequence = &begin_sequence_impl,
    .finish_sequence = &finish_sequence_impl,
    .reserve_completions = &reserve_completions_impl,
    .transfer_properties = &transfer_properties_impl,
    .copy_async = &copy_async_impl,
    .fill_async = &fill_async_impl,
    .wait_completion = &wait_completion_impl,
    .can_wait_queue = &can_wait_queue_impl,
    .query_extension = &query_extension_impl,
    .drain_context = &drain_context_impl,
    .drain_queue = &drain_queue_impl,
};

Result<Device> device_impl(const Backend &, std::size_t index) noexcept {
  if (index != 0)
    return std::unexpected(
        Status::OutOfRange("CPU device index is out of range"));
  return detail::DeviceAccess::make(&kBackendVTable, DevicePayload{.index = 0});
}

} // namespace

Backend backend() noexcept {
  return detail::BackendAccess::make(&kBackendVTable, BackendPayload{});
}

Device device() noexcept {
  return detail::DeviceAccess::make(&kBackendVTable, DevicePayload{.index = 0});
}

Result<mem::MemoryResourceRef> memory_resource(Device value) noexcept {
  if (value != device())
    return std::unexpected(
        Status::InvalidArgument("device is not the HYLE CPU device"));
  return mem::system_resource();
}

} // namespace hyle::compute::cpu
