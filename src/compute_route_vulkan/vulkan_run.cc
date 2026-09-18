#include "vulkan_internal.h"

#include "hyle/obs/trace.h"

#include <limits>
#include <new>
#include <utility>

#include "completion_internal.h"
#include "hyle/core/compiler.h"
#include "hyle/time/clock.h"

namespace hyle::compute::vulkan::detail {

Run::~Run() {
  HYLE_FATAL_ASSERT(!pending && observers == 0);
  const auto &device = *runtime->device;
  if (command_pool != VK_NULL_HANDLE)
    device.fn.DestroyCommandPool(device.native.device, command_pool, nullptr);
  if (descriptor_pool != VK_NULL_HANDLE)
    device.fn.DestroyDescriptorPool(device.native.device, descriptor_pool,
                                    nullptr);
  if (descriptor_count != 0)
    runtime->device->descriptors_in_pools.fetch_sub(descriptor_count,
                                                    std::memory_order_relaxed);
  if (fence != VK_NULL_HANDLE)
    device.fn.DestroyFence(device.native.device, fence, nullptr);
}

Result<std::shared_ptr<Run>>
create_run(const std::shared_ptr<Runtime> &runtime) noexcept {
  try {
    auto run = std::make_shared<Run>();
    run->runtime = runtime;
    const auto &device = *runtime->device;
    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = device.native.queue_family;
    auto status =
        native_status(device.fn.CreateCommandPool(device.native.device, &pool,
                                                  nullptr, &run->command_pool),
                      "vkCreateCommandPool");
    if (!status.ok())
      return std::unexpected(status);
    VkCommandBufferAllocateInfo commands{};
    commands.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commands.commandPool = run->command_pool;
    commands.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commands.commandBufferCount = 1;
    status = native_status(device.fn.AllocateCommandBuffers(
                               device.native.device, &commands, &run->command),
                           "vkAllocateCommandBuffers");
    if (!status.ok())
      return std::unexpected(status);
    VkFenceCreateInfo fence{};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    status = native_status(device.fn.CreateFence(device.native.device, &fence,
                                                 nullptr, &run->fence),
                           "vkCreateFence");
    if (!status.ok())
      return std::unexpected(status);
    return run;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan submission metadata"));
  }
}

namespace {

void deliver(cd::CompletionLifetime lifetime, const Status &terminal,
             const Status &hint) noexcept {
  if (lifetime.valid()) {
    lifetime.terminal(lifetime.state, terminal.ok() ? hint : terminal);
    lifetime.release(lifetime.state);
  }
}

// Native success and device loss prove that referenced work is no longer in
// use. Other errors (including fence observation OOM) do not prove that fact.
bool is_terminal(VkResult result) noexcept {
  return result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST;
}

Result<CompletionPoll> observe(const std::shared_ptr<Run> &run,
                               bool wait) noexcept {
  auto &runtime = *run->runtime;
  auto &device = *runtime.device;
  std::unique_lock lock(runtime.mutex);
  if (!run->pending) {
    if (!run->terminal.ok())
      return std::unexpected(run->terminal);
    return CompletionPoll::kComplete;
  }
  ++run->observers;
  // Native fence waits do not hold the queue host mutex. A second producer
  // may submit its independent slot while this thread waits. observers prevents
  // fence reset/reuse until every concurrent observer has left the old run.
  VkResult result;
  if (wait) {
    lock.unlock();
    result =
        device.fn.WaitForFences(device.native.device, 1, &run->fence, VK_TRUE,
                                (std::numeric_limits<std::uint64_t>::max)());
    lock.lock();
  } else {
    // A nonblocking poll and its retirement bookkeeping form one host critical
    // section. An independent collector must not leave a transient observer
    // behind after the owning wait returns and the caller rebinds its slot.
    result = device.fn.GetFenceStatus(device.native.device, run->fence);
  }
  Status observation;
  if (!is_terminal(result) && result != VK_NOT_READY && result != VK_TIMEOUT) {
    observation =
        native_status(result, wait ? "vkWaitForFences" : "vkGetFenceStatus");
    if (wait) {
      // Only the exceptional path uses queue-wide drain. Preserve the original
      // diagnosis, even if an independent native observation proves completion.
      const auto drained = device.provider.with_queue(
          [](void *opaque, VkQueue queue) -> VkResult {
            return static_cast<Device *>(opaque)->fn.QueueWaitIdle(queue);
          }, &device);
      result = drained.ok() ? VK_SUCCESS
          : drained.native_error() == NativeError{NativeErrorDomain::Vulkan,
                                                 VK_ERROR_DEVICE_LOST}
              ? VK_ERROR_DEVICE_LOST : VK_NOT_READY;
    }
  }
  std::shared_ptr<Run> retired;
  cd::CompletionLifetime abandoned;
  Status terminal;
  Status hint;
  bool done = false;
  {
    --run->observers;
    if (!observation.ok() && run->observation_error.ok())
      run->observation_error = observation;
    if (run->pending && is_terminal(result)) {
      if (result == VK_ERROR_DEVICE_LOST) {
        runtime.failure = native_status(result, "Vulkan submitted work lost");
        run->terminal = runtime.failure;
      } else if (!run->observation_error.ok()) {
        run->terminal = run->observation_error;
      } else if (run->cancelled) {
        run->terminal = Status::Cancelled(
            "Vulkan result cancelled after native retirement");
      }
      run->pending = false;
      *run->previous = run->next;
      if (run->next != nullptr)
        run->next->previous = run->previous;
      run->next = nullptr;
      run->previous = nullptr;
      retired = std::move(run->pending_owner);
      abandoned = std::exchange(run->abandoned, {});
      hint = std::move(run->hint);
      run->retiring = abandoned.valid();
    }
    done = !run->pending;
    terminal = run->terminal;
  }
  lock.unlock();
  // Result destruction can release external owner tokens. Never invoke such
  // destructors under the native queue lock.
  if (retired && abandoned.valid()) {
    deliver(abandoned, terminal, hint);
    lock.lock();
    run->retiring = false;
    lock.unlock();
  }
  if (!done)
    return CompletionPoll::kPending;
  if (!terminal.ok())
    return std::unexpected(terminal);
  return CompletionPoll::kComplete;
}

const cd::CompletionVTable completion_vtable{
    .destroy =
        [](void *pointer, cd::CompletionLifetime lifetime,
           Status hint) noexcept {
          auto &run = *static_cast<Run *>(pointer);
          std::shared_ptr<Run> owner;
          Status terminal;
          bool pending;
          {
            std::lock_guard lock(run.runtime->mutex);
            pending = run.pending;
            if (pending) {
              run.abandoned = lifetime;
              run.hint = hint;
            } else {
              terminal = run.terminal;
              run.retiring = true;
            }
            owner = std::move(run.completion_owner);
          }
          if (!pending) {
            deliver(lifetime, terminal, hint);
            std::lock_guard lock(run.runtime->mutex);
            run.retiring = false;
          }
        },
    .poll = [](void *pointer) noexcept -> Result<CompletionPoll> {
      auto &run = *static_cast<Run *>(pointer);
      std::shared_ptr<Run> owner;
      {
        std::lock_guard lock(run.runtime->mutex);
        owner = run.completion_owner;
      }
      return observe(owner, false);
    },
    .wait = [](void *pointer) noexcept -> Status {
      auto &run = *static_cast<Run *>(pointer);
      std::shared_ptr<Run> owner;
      {
        std::lock_guard lock(run.runtime->mutex);
        owner = run.completion_owner;
      }
      for (;;) {
        auto observed = observe(owner, true);
        if (!observed)
          return std::move(observed).error();
        if (*observed == CompletionPoll::kComplete)
          return Status::Ok();
        // A native API failed without proving quiescence. wait keeps its
        // blocking contract; poll/collect offer nonblocking diagnosis instead.
        // Back off only on this exceptional path, not every native fence wait.
        time::sleep_for_ns(1000000);
      }
    },
    .request_cancel = [](void *pointer) noexcept -> Status {
      auto &run = *static_cast<Run *>(pointer);
      std::lock_guard lock(run.runtime->mutex);
      if (run.pending)
        run.cancelled = true;
      return Status::Ok();
    }};

} // namespace

Result<Completion> submit_locked(const std::shared_ptr<Run> &run,
                                 bool allow_closed) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.vulkan.submit");
  auto &runtime = *run->runtime;
  const auto &device = *runtime.device;
  if (runtime.closed && !allow_closed)
    return std::unexpected(
        Status::FailedPrecondition("Vulkan context is closed"));
  if (!runtime.failure.ok())
    return std::unexpected(runtime.failure);
  if (run->pending || run->completion_owner || run->observers != 0 ||
      run->retiring)
    return std::unexpected(
        Status::FailedPrecondition("Vulkan slot is still retained"));
  auto status =
      native_status(device.fn.ResetFences(device.native.device, 1, &run->fence),
                    "vkResetFences");
  if (!status.ok())
    return std::unexpected(status);
  VkSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  info.commandBufferCount = 1;
  info.pCommandBuffers = &run->command;
  struct Submission {
    PFN_vkQueueSubmit submit;
    const VkSubmitInfo *info;
    VkFence fence;
  } submission{device.fn.QueueSubmit, &info, run->fence};
  status = device.provider.with_queue(
      [](void *opaque, VkQueue queue) -> VkResult {
        const auto &call = *static_cast<const Submission *>(opaque);
        return call.submit(queue, 1, call.info, call.fence);
      }, &submission);
  if (!status.ok()) {
    // A failed native submit has not accepted our unsignaled fence. In
    // particular OOM must not manufacture pending work that will never signal.
    if (status.native_error() == NativeError{NativeErrorDomain::Vulkan,
                                            VK_ERROR_DEVICE_LOST})
      runtime.failure = status;
    return std::unexpected(status);
  }
  run->terminal = Status::Ok();
  run->observation_error = Status::Ok();
  run->hint = Status::Ok();
  run->cancelled = false;
  run->pending = true;
  run->pending_owner = run;
  run->completion_owner = run;
  run->next = runtime.pending;
  run->previous = &runtime.pending;
  if (run->next != nullptr)
    run->next->previous = &run->next;
  runtime.pending = run.get();
  return cd::CompletionAccess::make(&completion_vtable, run.get());
}

Status collect(const std::shared_ptr<Runtime> &runtime, bool drain) noexcept {
  if (drain) {
    Status first_error;
    for (;;) {
      std::shared_ptr<Run> run;
      {
        std::lock_guard lock(runtime->mutex);
        if (runtime->pending == nullptr)
          return first_error;
        run = runtime->pending->pending_owner;
      }
      // Fetch current membership again after every wait. A concurrent observer
      // may unlink the previous cursor and clear its next pointer; following a
      // stale snapshot must never make close report a completed drain early.
      auto state = observe(run, true);
      if (!state && first_error.ok())
        first_error = std::move(state).error();
      if (state && *state == CompletionPoll::kPending) {
        std::lock_guard lock(runtime->mutex);
        if (first_error.ok())
          first_error =
              run->observation_error.ok()
                  ? Status::Unavailable(
                        "Vulkan drain has not observed native completion")
                  : run->observation_error;
        return first_error;
      }
    }
  }
  std::shared_ptr<Run> cursor;
  std::size_t count = 0;
  {
    std::lock_guard lock(runtime->mutex);
    for (auto *run = runtime->pending; run != nullptr; run = run->next)
      ++count;
    if (runtime->pending != nullptr)
      cursor = runtime->pending->pending_owner;
  }
  Status first_error;
  // A bounded snapshot of membership: concurrent producers cannot keep a
  // nonblocking collect alive indefinitely by continually appending work.
  for (std::size_t index = 0; index < count && cursor; ++index) {
    std::shared_ptr<Run> next;
    {
      std::lock_guard lock(runtime->mutex);
      if (cursor->next != nullptr)
        next = cursor->next->pending_owner;
    }
    auto state = observe(cursor, false);
    if (!state && first_error.ok())
      first_error = std::move(state).error();
    if (state && *state == CompletionPoll::kPending) {
      std::lock_guard lock(runtime->mutex);
      if (!cursor->observation_error.ok() && first_error.ok())
        first_error = cursor->observation_error;
    }
    cursor = std::move(next);
  }
  return first_error;
}

} // namespace hyle::compute::vulkan::detail
