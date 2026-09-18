// Included after native_runtime_impl.h inside the locked Vulkan translation
// unit. Graph ownership/lowering lives above this native completion seam.
#include <chrono>
#include <thread>

namespace hyle::compute::ggml_vulkan::bridge {
struct Work {
  Runtime *runtime = nullptr;
  ggml_backend *backend = nullptr;
  vk::Fence fence;
  Failure failure;
  bool pending = false;
  bool needs_drain = false;
};

namespace {
WorkReport report(const Work &work) noexcept {
  return {work.failure, work.pending, work.runtime->device_lost.load()};
}
void remember(Work &work, Failure failure) noexcept {
  if (work.failure.error == Error::None)
    work.failure = failure;
  if (failure.native == VK_ERROR_DEVICE_LOST)
    work.runtime->device_lost.store(true);
}
VkResult drain_queue(const std::shared_ptr<vk_queue_handle> &handle) {
  std::unique_lock<std::mutex> device_guard;
  if (handle->device_submit_mutex)
    device_guard = std::unique_lock<std::mutex>(*handle->device_submit_mutex);
  std::lock_guard<vk_queue_handle> queue_guard(*handle);
  return VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueWaitIdle(handle->queue);
}
bool drain(Work &work) {
  auto device = work.runtime->instance.devices[0];
  const auto compute = device->compute_queue->handle;
  const auto transfer = device->transfer_queue &&
      device->transfer_queue->handle != compute
      ? device->transfer_queue->handle : nullptr;
  // Only the exceptional path drains queues. A failed observation never
  // licenses freeing the accepted prefix or resetting its command buffers.
  for (const auto &handle : {compute, transfer}) {
    if (!handle)
      continue;
    const auto result = drain_queue(handle);
    if (result == VK_ERROR_DEVICE_LOST) {
      remember(work, {Error::Native, result});
      return true;
    }
    if (result != VK_SUCCESS) {
      remember(work, {Error::Native, result});
      return false;
    }
  }
  return true;
}
void retire(Work &work) noexcept {
  auto *ctx = static_cast<ggml_backend_vk_context *>(work.backend->context);
  // This function is called only after the completion fence/all relevant
  // queues have retired, or device loss has terminated their work.
  work.pending = false;
  ctx->hyle_completion_fence = nullptr;
  ctx->compute_ctx.reset();
  ctx->transfer_ctx.reset();
  ctx->submit_pending = false;
  ctx->hyle_retired_failure = work.failure.error != Error::None ||
                             work.runtime->device_lost.load();
  try {
    if (!ctx->hyle_retired_failure && ctx->almost_ready_fence_pending)
      ctx->device->device.resetFences({ctx->almost_ready_fence});
    ctx->almost_ready_fence_pending = false;
    ggml_vk_graph_cleanup(ctx, !ctx->hyle_retired_failure);
  } catch (...) {
    remember(work, current_failure());
    ctx->hyle_retired_failure = true;
  }
  ctx->device->diag_cgraph = nullptr;
}
} // namespace

Failure create_work(Runtime *runtime, Work **result) noexcept {
  *result = nullptr;
  Work *work = nullptr;
  try {
    Active active(runtime);
    work = new Work();
    work->runtime = runtime;
    work->backend = ggml_backend_vk_init(0);
    work->fence = runtime->instance.devices[0]->device.createFence({});
    *result = work;
    return {};
  } catch (...) {
    const auto failure = current_failure();
    destroy_work(work);
    return failure;
  }
}

WorkReport submit_work(Work *work, ggml_cgraph *graph) noexcept {
  if (!work || !graph || graph->n_nodes <= 0)
    return {{Error::InvalidArgument}};
  if (work->pending)
    return {{Error::InvalidState}, true, work->runtime->device_lost.load()};
  if (work->failure.error != Error::None)
    return report(*work); // Failed native slots are never reused.
  if (work->runtime->device_lost.load()) {
    remember(*work, {Error::Native, VK_ERROR_DEVICE_LOST});
    return report(*work);
  }
  Active active(work->runtime);
  auto *ctx = static_cast<ggml_backend_vk_context *>(work->backend->context);
  try {
    ctx->device->device.resetFences({work->fence});
    // The caller binds resident tensors; no unfinished host upload/download
    // context may sneak into this pure region completion contract.
    if (!ctx->compute_ctx.expired() || !ctx->transfer_ctx.expired())
      return {{Error::InvalidState}};
    const bool attach_to_last = ctx->device->support_async &&
        !ctx->device->serialize_submissions && !vk_perf_logger_enabled;
    ctx->hyle_completion_fence = attach_to_last ? work->fence : vk::Fence{};
    ctx->hyle_completion_submitted = false;
    work->pending = true; // Conservatively retained before any fallible submit.
    if (ggml_backend_vk_graph_compute(work->backend, graph) != GGML_STATUS_SUCCESS) {
      remember(*work, {Error::Internal});
      work->needs_drain = true;
    } else if (!ctx->hyle_completion_submitted) {
      // Upstream's explicit serial/diagnostic mode, or a graph without an
      // emitted final batch. One region marker, never a per-kernel wrapper.
      ctx->device->compute_queue->handle->submit({}, work->fence);
    }
    ctx->hyle_completion_fence = nullptr;
  } catch (...) {
    ctx->hyle_completion_fence = nullptr;
    remember(*work, current_failure());
    work->needs_drain = work->pending;
    if (work->runtime->device_lost.load() && work->pending)
      retire(*work);
  }
  return report(*work);
}

WorkReport observe_work(Work *work, bool wait) noexcept {
  if (!work)
    return {{Error::InvalidArgument}};
  if (!work->pending)
    return report(*work);
  Active active(work->runtime);
  try {
    if (work->runtime->device_lost.load()) {
      remember(*work, {Error::Native, VK_ERROR_DEVICE_LOST});
      retire(*work);
    } else if (work->needs_drain) {
      // Nonblocking poll reports both the diagnosis and retained pending
      // ownership. Explicit wait/close performs exceptional queue-wide drain.
      if (wait && drain(*work))
        retire(*work);
    } else {
      const auto device = static_cast<VkDevice>(work->runtime->instance.devices[0]->device);
      const auto fence = static_cast<VkFence>(work->fence);
      const auto &api = work->runtime->instance.dispatcher;
      const auto result = wait
          ? api.vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX)
          : api.vkGetFenceStatus(device, fence);
      if (result == VK_SUCCESS) {
        retire(*work);
      } else if (result == VK_ERROR_DEVICE_LOST) {
        remember(*work, {Error::Native, result});
        retire(*work);
      } else if (result != VK_NOT_READY && result != VK_TIMEOUT) {
        remember(*work, {Error::Native, result});
        work->needs_drain = true;
        if (wait && drain(*work))
          retire(*work);
      }
    }
  } catch (...) {
    remember(*work, current_failure());
    work->needs_drain = true;
    if (work->runtime->device_lost.load())
      retire(*work);
  }
  return report(*work);
}

void destroy_work(Work *work) noexcept {
  if (!work)
    return;
  while (work->pending) {
    observe_work(work, true);
    if (work->pending)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  Active active(work->runtime);
  if (work->backend) {
    auto *ctx = static_cast<ggml_backend_vk_context *>(work->backend->context);
    // Final destruction needs no native command/event resets. In particular,
    // device-loss teardown must not throw while resetting a lost device.
    ctx->hyle_retired_failure = true;
    ggml_backend_free(work->backend);
  }
  work->runtime->instance.devices[0]->device.destroyFence(work->fence);
  delete work;
}
} // namespace hyle::compute::ggml_vulkan::bridge
