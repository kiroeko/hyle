// Included once, after the locked upstream implementation. This seam borrows
// upstream types/algorithms while giving each HYLE owner isolated native state.
#include "ggml-alloc.h"
#include <atomic>
namespace hyle::compute::ggml_vulkan::bridge {
struct Runtime {
  vk_instance_t instance;
  std::atomic<bool> device_lost{false};
};

void *enter(Runtime *runtime) noexcept {
  auto *previous = hyle_vk_active_instance;
  hyle_vk_active_instance = runtime ? &runtime->instance : nullptr;
  return previous;
}
void leave(void *previous) noexcept {
  hyle_vk_active_instance = static_cast<vk_instance_t *>(previous);
}
namespace {
struct Active {
  void *previous;
  explicit Active(Runtime *runtime) : previous(enter(runtime)) {}
  ~Active() { leave(previous); }
};
Failure current_failure() noexcept {
  try {
    throw;
  } catch (const std::bad_alloc &) {
    return {Error::Exhausted};
  } catch (const std::out_of_range &) {
    return {Error::OutOfRange};
  } catch (const vk::SystemError &error) {
    return {Error::Native, static_cast<VkResult>(error.code().value())};
  } catch (const std::runtime_error &) {
    return {Error::Unavailable};
  } catch (...) {
    return {Error::Internal};
  }
}
} // namespace

void destroy(Runtime *runtime) noexcept {
  if (!runtime)
    return;
  {
    Active active(runtime);
    // Callers retain Runtime until backend/buffer owners are retired. These
    // two caches otherwise form device -> buffer -> device ownership cycles.
    for (auto &device : runtime->instance.devices) {
      if (!device)
        continue;
      ggml_vk_destroy_buffer(device->sync_staging);
      while (!device->pinned_memory.empty())
        ggml_vk_host_free(device, std::get<0>(device->pinned_memory.back()));
      device.reset();
    }
    for (auto *device : runtime->instance.registry_devices) {
      delete static_cast<ggml_backend_vk_device_context *>(device->context);
      delete device;
    }
    if (runtime->instance.instance)
      runtime->instance.instance.destroy();
  }
  delete runtime;
}

Failure create(const Options &options, Runtime **result,
               Native *native) noexcept {
  *result = nullptr;
  *native = {};
  if (!options.get_instance_proc_addr)
    return {Error::Unavailable};
  Runtime *runtime = nullptr;
  try {
    runtime = new Runtime();
    runtime->instance.options = options;
    Active active(runtime);
    runtime->instance.dispatcher.init(options.get_instance_proc_addr);
    const auto &dispatcher = runtime->instance.dispatcher;
    if (!dispatcher.vkEnumerateInstanceVersion ||
        !dispatcher.vkEnumerateInstanceExtensionProperties ||
        !dispatcher.vkEnumerateInstanceLayerProperties ||
        !dispatcher.vkCreateInstance)
      throw vk::InitializationFailedError("incomplete Vulkan loader");
    ggml_vk_instance_init();
    auto device = ggml_vk_get_device(0);
    *native = {
        runtime->instance.instance, device->physical_device, device->device,
        device->compute_queue->handle->queue,
        device->compute_queue->queue_family_index,
        runtime->instance.loader_version, device->buffer_device_address,
        device->hyle_storage_update_after_bind, options.get_instance_proc_addr,
        device->properties.limits.minStorageBufferOffsetAlignment,
        device->hyle_sync2, device->hyle_timeline, device->hyle_scalar_layout,
        device->hyle_shader_f16, device->hyle_storage16, device->shader_int64,
        device->hyle_portability_subset,
        device->has_internally_synchronized_queues
            ? static_cast<std::uint32_t>(eInternallySynchronizedKHR)
            : 0u,
        // This pin forwards all supported core features unchanged to
        // DeviceCreateInfo. Capture the enabled Int16 fact cold.
        device->physical_device.getFeatures().shaderInt16 != VK_FALSE};
    *result = runtime;
    return {};
  } catch (...) {
    const auto error = current_failure();
    destroy(runtime);
    return error;
  }
}

void note_device_lost(Runtime *runtime) noexcept {
  runtime->device_lost.store(true);
}

Failure with_queue(Runtime *runtime, VkResult (*function)(void *, VkQueue),
                   void *state) noexcept {
  try {
    Active active(runtime);
    auto handle = runtime->instance.devices[0]->compute_queue->handle;
    // This is also the upstream submit lock order, including its NVIDIA
    // cross-queue workaround. Do not call back into this owner in the body.
    std::unique_lock<std::mutex> device_guard;
    if (handle->device_submit_mutex)
      device_guard = std::unique_lock<std::mutex>(*handle->device_submit_mutex);
    std::lock_guard<vk_queue_handle> queue_guard(*handle);
    const auto result = function(state, handle->queue);
    if (result == VK_ERROR_DEVICE_LOST)
      runtime->device_lost.store(true);
    return result == VK_SUCCESS ? Failure{} : Failure{Error::Native, result};
  } catch (...) {
    return current_failure();
  }
}

ggml_backend_buffer *borrow(VkBuffer handle, std::size_t bytes,
                            std::shared_ptr<const void> owner,
                            bool address_capable) {
  auto device = ggml_vk_get_device(0);
  if (!bytes || !owner)
    throw std::invalid_argument("ggml Vulkan borrowed buffer");
  auto buffer = std::make_shared<vk_buffer_struct>();
  buffer->device = device;
  buffer->external_owner = std::move(owner);
  buffer->buffer = handle;
  buffer->size = bytes;
  // Device support does not imply that an external buffer was created with
  // the address usage and memory allocation flags. Descriptor-only imports
  // need no address; a BDA operation must qualify its buffers explicitly.
  if (handle && address_capable && device->buffer_device_address)
    buffer->bda_addr =
        device->device.getBufferAddress(vk::BufferDeviceAddressInfo{handle});
  auto context = std::make_unique<ggml_backend_vk_buffer_context>(
      device, std::move(buffer), device->name);
  auto *result = ggml_backend_buffer_init(&device->buffer_type,
                                          ggml_backend_vk_buffer_interface,
                                          context.get(), bytes);
  context.release();
  return result;
}

Failure binding_buffer(Runtime *runtime, std::size_t bytes,
                       std::shared_ptr<const void> owner,
                       ggml_backend_buffer **result) noexcept {
  *result = nullptr;
  try {
    Active active(runtime);
    *result = borrow(VK_NULL_HANDLE, bytes, std::move(owner), false);
    return {};
  } catch (...) {
    return current_failure();
  }
}
void bind_buffer(ggml_backend_buffer *buffer, VkBuffer handle,
                 std::size_t bytes,
                 std::shared_ptr<const void> owner) noexcept {
  auto &native = static_cast<ggml_backend_vk_buffer_context *>(buffer->context)
                     ->dev_buffer;
  native->buffer = handle;
  native->size = bytes;
  native->external_owner = std::move(owner);
  buffer->size = bytes;
}

VkBuffer buffer_handle(ggml_backend_buffer *buffer) noexcept {
  if (!buffer || !ggml_backend_buffer_is_vk(buffer))
    return VK_NULL_HANDLE;
  return static_cast<ggml_backend_vk_buffer_context *>(buffer->context)
      ->dev_buffer->buffer;
}

Failure create_backend(Runtime *runtime, ggml_backend **result) noexcept {
  *result = nullptr;
  try {
    Active active(runtime);
    *result = ggml_backend_vk_init(0);
    return {};
  } catch (...) {
    return current_failure();
  }
}
Failure borrow_buffer(Runtime *runtime, VkBuffer handle, std::size_t bytes,
                      std::shared_ptr<const void> owner, bool address_capable,
                      ggml_backend_buffer **result) noexcept {
  *result = nullptr;
  if (!handle)
    return {Error::InvalidArgument};
  try {
    Active active(runtime);
    *result = borrow(handle, bytes, std::move(owner), address_capable);
    return {};
  } catch (...) {
    return current_failure();
  }
}
Failure allocate_buffer(Runtime *runtime, std::size_t bytes,
                        ggml_backend_buffer **result) noexcept {
  *result = nullptr;
  try {
    Active active(runtime);
    *result =
        ggml_backend_buft_alloc_buffer(ggml_backend_vk_buffer_type(0), bytes);
    return *result ? Failure{} : Failure{Error::Exhausted};
  } catch (...) {
    return current_failure();
  }
}
} // namespace hyle::compute::ggml_vulkan::bridge

#include "native_storage_impl.h"
#include "native_work_impl.h"
