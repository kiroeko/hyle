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
