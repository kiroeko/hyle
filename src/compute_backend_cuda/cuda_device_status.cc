#include "cuda_internal.h"

#include <mutex>
#include <utility>

namespace hyle::compute::cuda::detail {
void CapturedCommandsState::destroy() noexcept {
  auto *retained_context = context;
  const auto selected = allocator;
  {
    ScopedPrimaryContext scope(context->primary);
    if (scope) {
      if (executable != nullptr)
        runtime_status(cudaGraphExecDestroy(executable)).IgnoreError();
      if (device_status != nullptr)
        runtime_status(cudaFree(device_status)).IgnoreError();
    }
  }
  this->~CapturedCommandsState();
  selected.deallocate(this, sizeof(CapturedCommandsState),
                      alignof(CapturedCommandsState));
  retained_context->decref();
}

Result<StatusReference *> QueueState::acquire_status_reference() noexcept {
  std::lock_guard lock(slots_lock);
  auto *result = free_status_references;
  if (result == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("CUDA status reference capacity exhausted; "
                                  "reserve_completions before submission"));
  free_status_references = result->next;
  *result = {};
  return result;
}

void QueueState::release_status_reference(StatusReference *reference) noexcept {
  *reference = {};
  std::lock_guard lock(slots_lock);
  reference->next = free_status_references;
  free_status_references = reference;
}

void settle_status_references(EventSlot *slot) noexcept {
  // Only the owning Completion/reaper observes this fence. A dependent GPU
  // reads the immutable device word, not this CPU list or its lifetime hooks.
  auto *reference = std::exchange(slot->status_references, nullptr);
  while (reference != nullptr) {
    auto *next = reference->next;
    if (reference->capture != nullptr) {
      reference->capture->status_busy.store(false, std::memory_order_release);
      reference->capture->decref();
    }
    if (reference->source != nullptr)
      reference->source->owner->release(reference->source);
    slot->owner->release_status_reference(reference);
    reference = next;
  }
}

Status completed_device_status(const EventSlot *slot) noexcept {
  if (!slot->status_copied || !slot->report_device_status)
    return Status::Ok();
  switch (static_cast<DeviceStatusCode>(*slot->host_status)) {
  case DeviceStatusCode::kOk:
    return Status::Ok();
  case DeviceStatusCode::kIndexOutOfRange:
    return Status::OutOfRange("CUDA index is outside the selected axis");
  case DeviceStatusCode::kIntegerOverflow:
    return Status::OutOfRange("CUDA integer prefix sum overflows I32");
  }
  return Status::Internal("CUDA program returned an unknown device status");
}

compute::detail::EnqueueResult
acquire_device_status(QueueState *queue, unsigned **device_word) noexcept {
  using compute::detail::EnqueueResult;
  if (device_word == nullptr)
    return EnqueueResult::rejected(
        Status::InvalidArgument("CUDA status output is null"));
  *device_word = nullptr;
  if (queue == nullptr || queue->active_sequence == nullptr)
    return EnqueueResult::rejected(Status::FailedPrecondition(
        "CUDA device status requires an admitted sequence"));
  if (queue->context->status_merge_kernel == nullptr)
    return EnqueueResult::rejected(Status::Unimplemented(
        "CUDA device-status kernel is unavailable for this GPU/build"));
  bool &used = queue->capture_owned ? queue->capture_status_used
                                    : queue->active_sequence->status_used;
  auto *word = queue->capture_owned ? queue->capture_device_status
                                    : queue->active_sequence->device_status;
  if (word == nullptr)
    return EnqueueResult::rejected(
        Status::Internal("CUDA device status storage is absent"));
  *device_word = word;
  if (used)
    return EnqueueResult::no_op();
  used = true;
  return EnqueueResult::accepted_command(runtime_status(
      cudaMemsetAsync(word, 0, sizeof(unsigned), queue->stream)));
}

compute::detail::EnqueueResult
acquire_device_status(Queue &queue, unsigned **device_word) noexcept {
  if (queue.device().backend_id() != kBackendId) {
    if (device_word != nullptr)
      *device_word = nullptr;
    return compute::detail::EnqueueResult::rejected(
        Status::InvalidArgument("device status requires a CUDA queue"));
  }
  return acquire_device_status(
      static_cast<QueueState *>(
          compute::detail::ContextAccess::backend_queue(queue)),
      device_word);
}
} // namespace hyle::compute::cuda::detail
