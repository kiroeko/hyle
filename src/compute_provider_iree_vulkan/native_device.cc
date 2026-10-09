#include "runtime_internal.h"

#include <algorithm>
#include <limits>
#include <new>
#include <string_view>

#include "programs.h"

namespace hyle::compute::iree_vulkan {
namespace detail {
Status consume_status(iree_status_t status) noexcept {
  if (iree_status_is_ok(status))
    return Status::Ok();
  const auto code = iree_status_code(status);
  char message[512] = {};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(message), message, &length);
  // Both use the canonical gRPC status taxonomy. DEFERRED is a private IREE
  // progress marker, handled before translation by the invocation owner.
  const auto translated =
      code == IREE_STATUS_INCOMPATIBLE ? StatusCode::Unimplemented
      : code >= IREE_STATUS_CANCELLED && code <= IREE_STATUS_UNAUTHENTICATED
          ? static_cast<StatusCode>(code)
          : StatusCode::Unknown;
  Status result(translated,
                std::string_view(message, std::min<std::size_t>(
                                              length, sizeof(message) - 1)));
  iree_status_free(status);
  return result;
}

iree_status_t
Accounting::allocate(void *context, uint64_t bytes, VkDeviceMemory *memory,
                     iree_hal_vulkan_memory_allocate_action_t action,
                     void *data) noexcept {
  auto &self = *static_cast<Accounting *>(context);
  if (bytes > std::numeric_limits<std::size_t>::max())
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native allocation size");
  auto reservation =
      self.quota.try_reserve_bytes(static_cast<std::size_t>(bytes));
  if (!reservation)
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HYLE Vulkan memory quota");
  try {
    std::lock_guard lock(self.mutex);
    // Allocate metadata first: no fallible operation may lose an accepted
    // native allocation. The original allocation callback is synchronous.
    self.allocations.push_back({VK_NULL_HANDLE, std::move(*reservation)});
    auto status = action(data);
    if (iree_status_is_ok(status))
      self.allocations.back().memory = *memory;
    else
      self.allocations.pop_back();
    return status;
  } catch (const std::bad_alloc &) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native quota metadata");
  }
}

void Accounting::free(void *context, VkDeviceMemory memory,
                      iree_hal_vulkan_memory_free_action_t action,
                      void *data) noexcept {
  auto &self = *static_cast<Accounting *>(context);
  std::lock_guard lock(self.mutex);
  action(data);
  if (memory == VK_NULL_HANDLE)
    return;
  const auto found = std::find_if(
      self.allocations.begin(), self.allocations.end(),
      [memory](const auto &entry) { return entry.memory == memory; });
  HYLE_FATAL_ASSERT(found != self.allocations.end());
  self.allocations.erase(found);
}
} // namespace detail

Status NativeDevice::State::request_stop() noexcept {
  std::lock_guard lock(stop_mutex);
  if (!stopping.exchange(true, std::memory_order_acq_rel))
    stop_status = detail::consume_status(
        iree_hal_vulkan_device_request_stop(device.get(), retired.get()));
  return stop_status;
}

Result<NativeDevice>
NativeDevice::create(ggml_vulkan::detail::NativeDevice native,
                     mem::QuotaAllocator *quota) noexcept {
  if (!native)
    return std::unexpected(
        Status::InvalidArgument("IREE requires a native Vulkan owner"));
  const auto &vk = native.native();
  if (!vk.synchronization2 || !vk.timeline_semaphore || !vk.scalar_block_layout)
    return std::unexpected(
        Status::Unimplemented("IREE requires enabled synchronization2, "
                              "timeline semaphore and scalar layout"));
  try {
    auto state = std::make_shared<State>();
    state->native = std::move(native);
    if (quota)
      state->accounting = std::make_unique<detail::Accounting>(*quota);
    iree_async_proactor_pool_t *pool = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool)));
    state->pool.reset(pool);
    iree_hal_vulkan_syms_t *syms = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(iree_hal_vulkan_syms_create(
        reinterpret_cast<void *>(vk.get_instance_proc_addr),
        iree_allocator_system(), &syms)));
    state->syms.reset(syms);
    iree_hal_vulkan_device_options_t options;
    iree_hal_vulkan_device_options_initialize(&options);
    options.dispatch_abis = IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR;
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = pool;
    iree_hal_vulkan_external_device_params_t external = {};
    external.enabled_features = IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE;
    if (vk.shader_float16)
      external.enabled_features |=
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16;
    if (vk.storage_buffer16)
      external.enabled_features |=
          IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS;
    if (vk.shader_int64)
      external.enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64;
    if (vk.portability_subset)
      external.enabled_extensions =
          IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PORTABILITY_SUBSET;
    external.compute_queue_set = {vk.queue_family, 1};
    external.queue_create_flags = vk.queue_create_flags;
    external.external_queue_user_data = &state->native;
    external.external_queue_call =
        [](void *owner, VkQueue queue,
           iree_hal_vulkan_external_queue_action_fn_t action,
           void *data) -> iree_status_t {
      auto &device = *static_cast<ggml_vulkan::detail::NativeDevice *>(owner);
      if (device.native().queue != queue)
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "foreign IREE queue");
      struct Call {
        iree_hal_vulkan_external_queue_action_fn_t action;
        void *data;
        iree_status_t result = iree_ok_status();
        bool called = false;
      } call{action, data};
      const auto locked = device.with_queue(
          [](void *argument, VkQueue) -> VkResult {
            auto &value = *static_cast<Call *>(argument);
            value.called = true;
            value.result = value.action(value.data);
            return iree_status_code(value.result) == IREE_STATUS_DATA_LOSS
                       ? VK_ERROR_DEVICE_LOST
                       : VK_SUCCESS;
          },
          &call);
      if (call.called)
        return call.result;
      return iree_make_status(locked.code() == StatusCode::DataLoss
                                  ? IREE_STATUS_DATA_LOSS
                                  : IREE_STATUS_UNAVAILABLE,
                              "native queue owner rejected IREE submission");
    };
    if (state->accounting)
      external.memory_owner = {state->accounting.get(),
                               detail::Accounting::allocate,
                               detail::Accounting::free};
    iree_hal_device_t *device = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(iree_hal_vulkan_wrap_device(
        iree_make_cstring_view("vulkan"), &options, &params, syms, vk.instance,
        vk.physical, vk.device, &external, iree_allocator_system(), &device)));
    state->device.reset(device);
    iree_hal_semaphore_t *retired = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, &retired)));
    state->retired.reset(retired);
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_t *instance = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &instance)));
    state->instance.reset(instance);
    iree_async_frontier_tracker_t *tracker = nullptr;
    HYLE_RETURN_IF_ERROR(
        detail::consume_status(iree_async_frontier_tracker_create(
            iree_async_frontier_tracker_options_default(),
            iree_allocator_system(), &tracker)));
    state->tracker.reset(tracker);
    iree_hal_device_group_t *group = nullptr;
    HYLE_RETURN_IF_ERROR(
        detail::consume_status(iree_hal_device_group_create_from_device(
            device, tracker, iree_allocator_system(), &group)));
    state->group.reset(group);
    auto *vm = iree_runtime_instance_vm_instance(instance);
    iree_vm_module_t *hal = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(iree_hal_module_create(
        vm, iree_hal_module_device_policy_default(), group,
        IREE_HAL_MODULE_FLAG_NONE, iree_hal_module_debug_sink_stdio(stderr),
        iree_allocator_system(), &hal)));
    state->hal.reset(hal);
    const std::array programs{detail::checked_scan_program(),
                              detail::fp8_program(),
                              detail::precision_program(),
                              detail::artifacts_program(),
                              detail::mlx_program(),
                              detail::official_program(),
                              detail::iq1_program()};
    for (std::size_t i = 0; i < programs.size(); ++i) {
      iree_vm_module_t *module = nullptr;
      HYLE_RETURN_IF_ERROR(
          detail::consume_status(iree_vm_bytecode_module_create(
              vm, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
              iree_make_const_byte_span(programs[i].data(), programs[i].size()),
              iree_allocator_null(), iree_allocator_system(), &module)));
      state->programs[i].reset(module);
    }
    NativeDevice result;
    result.state_ = std::move(state);
    return result;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("IREE Vulkan owner"));
  }
}
} // namespace hyle::compute::iree_vulkan
