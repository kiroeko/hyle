#include "vulkan_internal.h"
#include "vulkan_iree.h"

#include "hyle/obs/trace.h"

#include <cstring>
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
  if (transfer_pool != VK_NULL_HANDLE)
    device.fn.DestroyCommandPool(device.native.device, transfer_pool, nullptr);
  if (descriptor_pool != VK_NULL_HANDLE)
    device.fn.DestroyDescriptorPool(device.native.device, descriptor_pool,
                                    nullptr);
  if (descriptor_count != 0)
    runtime->device->descriptors_in_pools.fetch_sub(descriptor_count,
                                                    std::memory_order_relaxed);
  if (fence != VK_NULL_HANDLE)
    device.fn.DestroyFence(device.native.device, fence, nullptr);
  if (transfer_fence != VK_NULL_HANDLE)
    device.fn.DestroyFence(device.native.device, transfer_fence, nullptr);
  if (transfer_ready != VK_NULL_HANDLE)
    device.fn.DestroySemaphore(device.native.device, transfer_ready, nullptr);
}

Result<std::shared_ptr<Run>>
create_run(const std::shared_ptr<Runtime> &runtime, mem::AllocatorRef allocator) noexcept {
  try {
    if (!allocator.valid()) return std::unexpected(Status::InvalidArgument("invalid Vulkan run metadata allocator"));
    auto run = std::allocate_shared<Run>(mem::StdAllocator<Run>(allocator), allocator);
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
    run->completion_fence = run->fence;
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

bool device_lost(const Status &status) noexcept {
  return status.native_error() ==
         NativeError{NativeErrorDomain::Vulkan, VK_ERROR_DEVICE_LOST};
}

Status drain_native(Device &device, bool transfer) noexcept {
  const auto compute = device.provider.with_queue(
      [](void *opaque, VkQueue queue) -> VkResult {
        return static_cast<Device *>(opaque)->fn.QueueWaitIdle(queue);
      },
      &device);
  if (!transfer) return compute;
  const auto copied = device.provider.with_transfer_queue(
      [](void *opaque, VkQueue queue) -> VkResult {
        return static_cast<Device *>(opaque)->fn.QueueWaitIdle(queue);
      }, &device);
  return !compute.ok() ? compute : copied;
}

VkResult drain_result(const Status &status) noexcept {
  return status.ok()           ? VK_SUCCESS
         : device_lost(status) ? VK_ERROR_DEVICE_LOST
                               : VK_NOT_READY;
}

// Only the owner of an idle slot resets its words. Readback is reached after
// all accepted native and managed work has retired, including abandonment.
Status checked_status(Run &run, bool reset) noexcept {
  if (!run.checked)
    return Status::Ok();
  const auto &device = *run.runtime->device;
  for (const auto &buffer : run.checked->buffers) {
    std::uint32_t value = 0;
    if (reset)
      std::memcpy(buffer->mapped, &value, sizeof(value));
    if ((buffer->memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
      VkMappedMemoryRange range{};
      range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      range.memory = buffer->memory;
      range.size = VK_WHOLE_SIZE;
      HYLE_RETURN_IF_ERROR(native_status(
          reset ? device.fn.FlushMappedMemoryRanges(device.native.device, 1,
                                                    &range)
                : device.fn.InvalidateMappedMemoryRanges(device.native.device,
                                                         1, &range),
          reset ? "vkFlushMappedMemoryRanges(checked status)"
                : "vkInvalidateMappedMemoryRanges(checked status)"));
    }
    if (!reset) {
      std::memcpy(&value, buffer->mapped, sizeof(value));
      if ((value & 2u) != 0)
        return Status::OutOfRange("Vulkan Scan prefix exceeds I32");
      if (value != 0)
        return Status::OutOfRange("Vulkan index is outside the selected axis");
    }
  }
  return Status::Ok();
}

// Called under Runtime::mutex after the parent native completion proof, or
// while explicitly draining a partially accepted sequence. No child joins
// the global pending list, and no per-kernel completion marker is submitted.
bool retire_regions(Run &run, bool drain) noexcept {
  bool retired = true;
  for (auto &stage : run.stages) {
    if (!stage.managed)
      continue;
    const auto state = drain ? stage.managed->wait() : stage.managed->poll();
    if (!state.status.ok() && run.observation_error.ok())
      run.observation_error = state.status;
    if (state.device_lost)
      run.runtime->failure =
          native_status(VK_ERROR_DEVICE_LOST, "Vulkan managed work lost");
    retired = retired && !state.pending;
  }
  return retired;
}

} // namespace

Status submit_native(const Device &device, VkCommandBuffer command,
                     VkFence fence) noexcept {
  VkSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  info.commandBufferCount = 1;
  info.pCommandBuffers = &command;
  struct Submission {
    PFN_vkQueueSubmit submit;
    const VkSubmitInfo *info;
    VkFence fence;
  } submission{device.fn.QueueSubmit, &info, fence};
  return device.provider.with_queue(
      [](void *opaque, VkQueue queue) -> VkResult {
        const auto &call = *static_cast<const Submission *>(opaque);
        return call.submit(queue, 1, call.info, call.fence);
      },
      &submission);
}

namespace {

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
  if (run->async) {
    // The current IREE notification advances all following native/managed
    // stages until the next IREE boundary. Never wait while holding its route
    // mutex: those notifications need the same mutex to publish/advance.
    while (!run->async->finished) {
      auto &program = *run->stages[run->async->next_stage].iree->program;
      lock.unlock();
      const auto state = wait ? program.wait() : program.poll();
      lock.lock();
      if (!state.status.ok() && run->observation_error.ok())
        run->observation_error = state.status;
      if ((!wait && !run->async->finished) ||
          (state.pending && !state.status.ok())) {
        --run->observers;
        return CompletionPoll::kPending;
      }
    }
    // The final fence can signal before the last notification returns.
    // Include every callback tail before dropping the parent ownership.
    lock.unlock();
    bool retired = true;
    Status failure;
    for (auto &stage : run->stages) {
      if (!stage.iree)
        continue;
      const auto state =
          wait ? stage.iree->program->wait() : stage.iree->program->poll();
      retired = retired && !state.pending;
      if (!state.status.ok() && failure.ok())
        failure = state.status;
    }
    lock.lock();
    if (!failure.ok() && run->observation_error.ok())
      run->observation_error = failure;
    if (!retired) {
      --run->observers;
      return CompletionPoll::kPending;
    }
  }
  // Native fence waits do not hold the queue host mutex. A second producer
  // may submit its independent slot while this thread waits. observers prevents
  // fence reset/reuse until every concurrent observer has left the old run.
  VkResult result = VK_NOT_READY;
  if (run->needs_drain) {
    if (device_lost(runtime.failure)) {
      result = VK_ERROR_DEVICE_LOST;
    } else if (wait) {
      const auto drained = drain_native(device, run->transfer_submitted);
      result = drain_result(drained);
      if (!drained.ok() && run->observation_error.ok())
        run->observation_error = drained;
    }
  } else if (wait) {
    lock.unlock();
    result = device.fn.WaitForFences(
        device.native.device, 1, &run->completion_fence, VK_TRUE,
        (std::numeric_limits<std::uint64_t>::max)());
    lock.lock();
  } else {
    // A nonblocking poll and its retirement bookkeeping form one host critical
    // section. An independent collector must not leave a transient observer
    // behind after the owning wait returns and the caller rebinds its slot.
    result =
        device.fn.GetFenceStatus(device.native.device, run->completion_fence);
  }
  Status observation;
  if (!is_terminal(result) && result != VK_NOT_READY && result != VK_TIMEOUT) {
    observation =
        native_status(result, wait ? "vkWaitForFences" : "vkGetFenceStatus");
    if (wait) {
      // Only the exceptional path uses queue-wide drain. Preserve the original
      // diagnosis, even if an independent native observation proves completion.
      result = drain_result(drain_native(device, run->transfer_submitted));
      run->needs_drain = !run->stages.empty();
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
    if (run->pending && result == VK_ERROR_DEVICE_LOST) {
      runtime.failure = native_status(result, "Vulkan submitted work lost");
      device.provider.note_device_lost();
    }
    if (run->pending && is_terminal(result) &&
        !retire_regions(*run, wait && run->needs_drain)) {
      // A provider observation/drain error still owns its accepted work.
      // The final parent fence alone does not authorize destroying that slot.
      run->needs_drain = true;
      result = VK_NOT_READY;
    }
    if (run->pending && is_terminal(result)) {
      if (!run->submission_failure.ok()) {
        run->terminal = run->submission_failure;
      } else if (device_lost(runtime.failure) && run->stages.empty()) {
        run->terminal = runtime.failure;
      } else if (!run->observation_error.ok()) {
        run->terminal = run->observation_error;
      } else if (device_lost(runtime.failure)) {
        run->terminal = runtime.failure;
      } else {
        const auto checked = checked_status(*run, false);
        if (!checked.ok()) {
          run->terminal = checked;
          run->submission_failure = checked;
          if (device_lost(checked)) {
            runtime.failure = checked;
            device.provider.note_device_lost();
          }
        } else if (run->cancelled) {
          run->terminal = Status::Cancelled(
              "Vulkan result cancelled after native retirement");
        }
      }
      if (!run->stages.empty() && !run->observation_error.ok() &&
          run->submission_failure.ok())
        run->submission_failure = run->observation_error;
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

Status submit_transfer(Run &run, bool &accepted) noexcept {
  auto &device = *run.runtime->device;
  HYLE_RETURN_IF_ERROR(native_status(device.fn.ResetFences(device.native.device, 1, &run.transfer_fence), "vkResetFences(transfer)"));
  struct Submission final {
    PFN_vkQueueSubmit submit;
    const VkSubmitInfo *info;
    VkFence fence;
  };
  const auto submit = [](void *opaque, VkQueue queue) -> VkResult {
    const auto &call = *static_cast<Submission *>(opaque);
    return call.submit(queue, 1, call.info, call.fence);
  };
  VkSubmitInfo copy{}; copy.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  copy.commandBufferCount = 1; copy.pCommandBuffers = &run.transfer_command;
  copy.signalSemaphoreCount = 1; copy.pSignalSemaphores = &run.transfer_ready;
  Submission first{device.fn.QueueSubmit, &copy, run.transfer_fence};
  const auto copied = device.provider.with_transfer_queue(submit, &first);
  if (!copied.ok()) return copied;
  accepted = true; run.transfer_submitted = true;
  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkSubmitInfo handoff{}; handoff.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  handoff.waitSemaphoreCount = 1; handoff.pWaitSemaphores = &run.transfer_ready;
  handoff.pWaitDstStageMask = &wait_stage;
  handoff.commandBufferCount = 1; handoff.pCommandBuffers = &run.command;
  Submission second{device.fn.QueueSubmit, &handoff, run.fence};
  const auto acquired = device.provider.with_queue(submit, &second);
  if (!acquired.ok()) {
    // A rejected handoff does not discard the already-accepted copy prefix.
    // Like the original lone-submit policy, rejection admits no second work.
    // The copy fence is the actual retirement proof; this slot cannot replay.
    run.submission_failure = acquired;
    run.completion_fence = run.transfer_fence;
  }
  return acquired;
}

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
  if (!run->submission_failure.ok())
    return std::unexpected(run->submission_failure);
  if (run->pending || run->completion_owner || run->observers != 0 ||
      run->retiring)
    return std::unexpected(
        Status::FailedPrecondition("Vulkan slot is still retained"));
  Status status;
  status = checked_status(*run, true);
  if (!status.ok()) {
    if (device_lost(status)) {
      runtime.failure = status;
      device.provider.note_device_lost();
    }
    return std::unexpected(status);
  }
  if (run->completion_fence == run->fence) {
    status = native_status(
        device.fn.ResetFences(device.native.device, 1, &run->fence),
        "vkResetFences");
    if (!status.ok()) {
      if (device_lost(status)) {
        runtime.failure = status;
        device.provider.note_device_lost();
      }
      return std::unexpected(status);
    }
  }
  run->needs_drain = false;
  run->transfer_submitted = false;
  run->cancelled = false;
  bool accepted = false;
  if (run->async) {
    *run->async = {};
    accepted = advance_async_locked(*run);
    status = run->submission_failure;
  } else if (run->stages.empty()) {
    if (run->transfer_submission) status = submit_transfer(*run, accepted);
    else {
      status = submit_native(device, run->command, run->fence);
      accepted = status.ok();
    }
  } else {
    for (std::size_t index = 0; index < run->stages.size(); ++index) {
      auto &stage = run->stages[index];
      if (stage.managed) {
        const auto submitted = stage.managed->submit();
        status = submitted.status;
        accepted = accepted || submitted.pending;
        if (submitted.device_lost)
          runtime.failure =
              native_status(VK_ERROR_DEVICE_LOST, "Vulkan managed submit lost");
      } else {
        status = submit_native(
            device, stage.native->command,
            index + 1 == run->stages.size() ? run->fence : VK_NULL_HANDLE);
        accepted = accepted || status.ok();
      }
      if (!status.ok()) {
        run->submission_failure = status;
        run->needs_drain = accepted;
        break;
      }
    }
  }
  if (device_lost(status))
    runtime.failure = status;
  if (!accepted) {
    // A lone rejected native submit owns no pending work and remains
    // retryable. A failed managed/composed slot keeps its failure instead.
    return std::unexpected(status);
  }
  run->terminal = Status::Ok();
  run->observation_error = run->submission_failure;
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
