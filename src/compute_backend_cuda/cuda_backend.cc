#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/command_capture.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

#include "../compute/backend_internal.h"
#include "cuda_internal.h"
#include "hyle/compute/detail/backend_extension.h"
#include "hyle/core/assert.h"

namespace hyle::compute::cuda {
namespace detail {

const char kMemoryDomainAuthority = 0;

namespace {

StatusCode runtime_code(cudaError_t error) noexcept {
  switch (error) {
  case cudaSuccess:
    return StatusCode::Ok;
  case cudaErrorMemoryAllocation:
    return StatusCode::ResourceExhausted;
  case cudaErrorInvalidValue:
  case cudaErrorInvalidDevice:
  case cudaErrorInvalidResourceHandle:
  case cudaErrorInvalidDeviceFunction:
  case cudaErrorInvalidConfiguration:
    return StatusCode::InvalidArgument;
  case cudaErrorNoDevice:
  case cudaErrorInsufficientDriver:
  case cudaErrorInitializationError:
  case cudaErrorSystemDriverMismatch:
    return StatusCode::Unavailable;
  case cudaErrorNotSupported:
    return StatusCode::Unimplemented;
  case cudaErrorNotReady:
    return StatusCode::Unavailable;
  case cudaErrorDeviceUninitialized:
    return StatusCode::FailedPrecondition;
  case cudaErrorOperatingSystem:
  case cudaErrorSystemNotReady:
    return StatusCode::Unavailable;
  default:
    return StatusCode::Internal;
  }
}

StatusCode driver_code(CUresult error) noexcept {
  switch (error) {
  case CUDA_SUCCESS:
    return StatusCode::Ok;
  case CUDA_ERROR_OUT_OF_MEMORY:
    return StatusCode::ResourceExhausted;
  case CUDA_ERROR_INVALID_VALUE:
  case CUDA_ERROR_INVALID_HANDLE:
  case CUDA_ERROR_INVALID_DEVICE:
  case CUDA_ERROR_INVALID_CONTEXT:
    return StatusCode::InvalidArgument;
  case CUDA_ERROR_NO_DEVICE:
  case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
  case CUDA_ERROR_SYSTEM_NOT_READY:
  case CUDA_ERROR_NOT_INITIALIZED:
  case CUDA_ERROR_DEINITIALIZED:
    return StatusCode::Unavailable;
  case CUDA_ERROR_NOT_SUPPORTED:
    return StatusCode::Unimplemented;
  case CUDA_ERROR_NOT_READY:
    return StatusCode::Unavailable;
  default:
    return StatusCode::Internal;
  }
}

} // namespace

Status runtime_status(cudaError_t error) noexcept {
  if (error == cudaSuccess)
    return Status::Ok();
  const char *message = cudaGetErrorString(error);
  return Status::FromNative(runtime_code(error),
                            NativeError{NativeErrorDomain::CudaRuntime,
                                        static_cast<std::int64_t>(error)},
                            message != nullptr ? std::string_view(message)
                                               : std::string_view{});
}

Status driver_status(CUresult error) noexcept {
  if (error == CUDA_SUCCESS)
    return Status::Ok();
  const char *message = nullptr;
  (void)cuGetErrorString(error, &message);
  return Status::FromNative(driver_code(error),
                            NativeError{NativeErrorDomain::CudaDriver,
                                        static_cast<std::int64_t>(error)},
                            message != nullptr ? std::string_view(message)
                                               : std::string_view{});
}

Status combine_terminal(Status observed, Status hint) noexcept {
  return observed.ok() && !hint.ok() ? std::move(hint) : std::move(observed);
}

ScopedPrimaryContext::ScopedPrimaryContext(CUcontext primary) noexcept
    : expected_(primary) {
  if (primary == nullptr) {
    status_ = Status::InvalidArgument("CUDA primary context is null");
    return;
  }
  status_ = driver_status(cuCtxPushCurrent(primary));
  pushed_ = status_.ok();
}

ScopedPrimaryContext::~ScopedPrimaryContext() {
  if (!pushed_)
    return;
  CUcontext popped = nullptr;
  const CUresult result = cuCtxPopCurrent(&popped);
  HYLE_FATAL_ASSERT(result == CUDA_SUCCESS);
  HYLE_FATAL_ASSERT(popped == expected_);
}

void ContextState::destroy() noexcept {
  HYLE_FATAL_ASSERT(queues == nullptr);
  if (primary != nullptr)
    (void)cuDevicePrimaryCtxRelease(ordinal);
  const mem::AllocatorRef selected = allocator;
  this->~ContextState();
  selected.deallocate(this, sizeof(ContextState), alignof(ContextState));
}

namespace {

Status destroy_event_block(EventBlock *block, ContextState *context) noexcept {
  Status result = Status::Ok();
  for (std::size_t index = 0; index < block->initialized; ++index) {
    if (block->slots[index].event != nullptr) {
      result.Update(
          runtime_status(cudaEventDestroy(block->slots[index].event)));
      block->slots[index].event = nullptr;
    }
  }
  const mem::AllocatorRef allocator = context->allocator;
  block->~EventBlock();
  allocator.deallocate(block, sizeof(EventBlock), alignof(EventBlock));
  return result;
}

} // namespace

void QueueState::destroy() noexcept {
  {
    std::lock_guard lock(context->queues_lock);
    QueueState **cursor = &context->queues;
    while (*cursor != nullptr && *cursor != this)
      cursor = &(*cursor)->context_next;
    HYLE_FATAL_ASSERT(*cursor == this);
    *cursor = context_next;
  }

  EventBlock *block = blocks;
  while (block != nullptr) {
    EventBlock *next = block->next;
    destroy_event_block(block, context).IgnoreError();
    block = next;
  }
  if (capture_owned && stream != nullptr) {
    cudaGraph_t abandoned = nullptr;
    (void)cudaStreamEndCapture(stream, &abandoned);
    capture_owned = false;
    if (abandoned != nullptr)
      runtime_status(cudaGraphDestroy(abandoned)).IgnoreError();
  }
  if (stream != nullptr && ownership != StreamOwnership::kBorrowed)
    runtime_status(cudaStreamDestroy(stream)).IgnoreError();

  ContextState *selected_context = context;
  const mem::AllocatorRef selected_allocator = allocator;
  this->~QueueState();
  selected_allocator.deallocate(this, sizeof(QueueState), alignof(QueueState));
  selected_context->decref();
}

Status QueueState::reserve(std::size_t minimum) noexcept {
  if (minimum >
      std::numeric_limits<std::size_t>::max() - (kEventSlotsPerBlock - 1)) {
    return Status::OutOfRange("CUDA completion reservation is too large");
  }
  for (;;) {
    {
      std::lock_guard lock(slots_lock);
      if (capacity >= minimum)
        return Status::Ok();
    }

    void *storage = allocator.allocate(sizeof(EventBlock), alignof(EventBlock));
    if (storage == nullptr) {
      return Status::ResourceExhausted(
          "CUDA completion event-block allocation failed");
    }
    auto *block = ::new (storage) EventBlock;
    ScopedPrimaryContext scope(context->primary);
    if (!scope) {
      block->~EventBlock();
      allocator.deallocate(block, sizeof(EventBlock), alignof(EventBlock));
      return scope.status();
    }
    for (; block->initialized < kEventSlotsPerBlock; ++block->initialized) {
      cudaEvent_t event = nullptr;
      const Status event_status = runtime_status(
          cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      if (!event_status.ok()) {
        destroy_event_block(block, context).IgnoreError();
        return event_status;
      }
      EventSlot &slot = block->slots[block->initialized];
      slot.owner = this;
      slot.event = event;
    }

    {
      std::lock_guard lock(slots_lock);
      block->next = blocks;
      blocks = block;
      for (EventSlot &slot : block->slots) {
        slot.next = free_slots;
        free_slots = &slot;
      }
      capacity += kEventSlotsPerBlock;
    }
  }
}

Result<EventSlot *> QueueState::acquire() noexcept {
  for (;;) {
    {
      std::lock_guard lock(slots_lock);
      if (free_slots != nullptr) {
        EventSlot *slot = free_slots;
        free_slots = slot->next;
        slot->next = nullptr;
        slot->fence = FenceKind::kReserved;
        slot->lifetime = {};
        slot->terminal_hint = Status::Ok();
        incref();
        return slot;
      }
    }
    std::size_t next_capacity = 1;
    {
      std::lock_guard lock(slots_lock);
      next_capacity = capacity + 1;
    }
    Status result = reserve(next_capacity);
    if (!result.ok())
      return std::unexpected(std::move(result));
  }
}

void QueueState::release(EventSlot *slot) noexcept {
  HYLE_FATAL_ASSERT(slot != nullptr && slot->owner == this);
  slot->fence = FenceKind::kReserved;
  slot->lifetime = {};
  slot->terminal_hint = Status::Ok();
  {
    std::lock_guard lock(slots_lock);
    slot->next = free_slots;
    free_slots = slot;
  }
  decref();
}

Result<QueueState *> create_queue_state(ContextState *context,
                                        cudaStream_t stream,
                                        StreamOwnership ownership,
                                        mem::AllocatorRef allocator) noexcept {
  if (context == nullptr || stream == nullptr || !allocator.valid()) {
    return std::unexpected(
        Status::InvalidArgument("CUDA queue construction is invalid"));
  }
  void *storage = allocator.allocate(sizeof(QueueState), alignof(QueueState));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("CUDA queue-state allocation failed"));
  }
  auto *result =
      ::new (storage) QueueState(context, stream, ownership, allocator);
  {
    std::lock_guard lock(context->queues_lock);
    result->context_next = context->queues;
    context->queues = result;
  }
  return result;
}

Result<CompletionPoll> poll_slot(EventSlot *slot) noexcept {
  if (slot == nullptr || slot->owner == nullptr) {
    return std::unexpected(Status::Internal("CUDA completion slot is invalid"));
  }
  cudaError_t result = cudaSuccess;
  if (slot->fence == FenceKind::kEvent)
    result = cudaEventQuery(slot->event);
  else if (slot->fence == FenceKind::kStreamFallback)
    result = cudaStreamQuery(slot->owner->stream);
  else
    return std::unexpected(
        Status::Internal("CUDA completion was not recorded"));

  if (result == cudaSuccess)
    return CompletionPoll::kComplete;
  if (result == cudaErrorNotReady)
    return CompletionPoll::kPending;
  return std::unexpected(runtime_status(result));
}

Status wait_slot(EventSlot *slot) noexcept {
  if (slot == nullptr || slot->owner == nullptr)
    return Status::Internal("CUDA completion slot is invalid");
  if (slot->fence == FenceKind::kEvent)
    return runtime_status(cudaEventSynchronize(slot->event));
  if (slot->fence == FenceKind::kStreamFallback)
    return runtime_status(cudaStreamSynchronize(slot->owner->stream));
  return Status::Internal("CUDA completion was not recorded");
}

void finish_abandoned(EventSlot *slot, Status observed) noexcept {
  const compute::detail::CompletionLifetime lifetime = slot->lifetime;
  Status terminal =
      combine_terminal(std::move(observed), std::move(slot->terminal_hint));
  slot->lifetime = {};
  if (lifetime.valid()) {
    lifetime.terminal(lifetime.state, terminal);
    lifetime.release(lifetime.state);
  }
  slot->owner->release(slot);
}

CompletionReaper::~CompletionReaper() {
  stopping_.store(true, std::memory_order_release);
  parker_.unpark();
  if (thread_.joinable())
    thread_.join();
}

CompletionReaper &CompletionReaper::instance() noexcept {
  static CompletionReaper reaper;
  return reaper;
}

Status CompletionReaper::start() noexcept {
  std::lock_guard lock(lock_);
  if (started_)
    return Status::Ok();
  auto created =
      platform::Thread::spawn(&CompletionReaper::run_entry, this,
                              platform::ThreadOpts{.name = "hyle-cuda-reaper"});
  if (!created)
    return std::move(created).error();
  thread_ = std::move(*created);
  started_ = true;
  return Status::Ok();
}

void CompletionReaper::retire(EventSlot *slot,
                              compute::detail::CompletionLifetime lifetime,
                              Status terminal_hint) noexcept {
  HYLE_FATAL_ASSERT(slot != nullptr);
  slot->lifetime = lifetime;
  slot->terminal_hint = std::move(terminal_hint);
  {
    std::lock_guard lock(lock_);
    HYLE_FATAL_ASSERT(started_);
    slot->next = pending_;
    pending_ = slot;
  }
  parker_.unpark();
}

void CompletionReaper::run_entry(void *context) noexcept {
  static_cast<CompletionReaper *>(context)->run();
}

EventSlot *CompletionReaper::take_all() noexcept {
  std::lock_guard lock(lock_);
  EventSlot *result = pending_;
  pending_ = nullptr;
  return result;
}

void CompletionReaper::append(EventSlot *head, EventSlot *tail) noexcept {
  if (head == nullptr)
    return;
  std::lock_guard lock(lock_);
  tail->next = pending_;
  pending_ = head;
}

void CompletionReaper::run() noexcept {
  while (!stopping_.load(std::memory_order_acquire)) {
    EventSlot *batch = take_all();
    if (batch == nullptr) {
      // A live CUDA Context reserves one failure-reportable reaper thread, but
      // the ordinary retained-Completion path must not turn that into a 1 kHz
      // background wakeup. unpark-before-park is remembered by Parker, so an
      // abandonment racing this empty observation cannot be lost.
      parker_.park();
      continue;
    }
    EventSlot *pending_head = nullptr;
    EventSlot *pending_tail = nullptr;
    while (batch != nullptr) {
      EventSlot *slot = batch;
      batch = batch->next;
      slot->next = nullptr;
      auto result = poll_slot(slot);
      if (result && *result == CompletionPoll::kPending) {
        if (pending_tail != nullptr)
          pending_tail->next = slot;
        else
          pending_head = slot;
        pending_tail = slot;
      } else if (result) {
        finish_abandoned(slot, Status::Ok());
      } else {
        finish_abandoned(slot, std::move(result).error());
      }
    }
    append(pending_head, pending_tail);
    // Only genuinely outstanding abandoned device work needs polling. A new
    // retirement supplies a Parker token and may bring the next check forward.
    if (pending_head != nullptr)
      parker_.park_for(std::chrono::milliseconds(1));
  }

  EventSlot *batch = take_all();
  while (batch != nullptr) {
    EventSlot *slot = batch;
    batch = batch->next;
    slot->next = nullptr;
    finish_abandoned(slot, wait_slot(slot));
  }
}

} // namespace detail

Status status(cudaError_t error) noexcept {
  return detail::runtime_status(error);
}

Status status(CUresult error) noexcept { return detail::driver_status(error); }

namespace {

struct BackendPayload final {};
struct DevicePayload final {
  int ordinal = -1;
};

struct DeviceNameRecord final {
  DeviceNameRecord *next = nullptr;
  int ordinal = -1;
  char text[256]{};
};

class DeviceNameCache final {
public:
  ~DeviceNameCache() {
    const mem::AllocatorRef allocator = mem::system_allocator();
    DeviceNameRecord *record = records_;
    while (record != nullptr) {
      DeviceNameRecord *next = record->next;
      record->~DeviceNameRecord();
      allocator.deallocate(record, sizeof(DeviceNameRecord),
                           alignof(DeviceNameRecord));
      record = next;
    }
  }

  std::string_view get(int ordinal, const char *name) noexcept {
    std::lock_guard lock(lock_);
    for (DeviceNameRecord *record = records_; record != nullptr;
         record = record->next) {
      if (record->ordinal == ordinal)
        return record->text;
    }
    const mem::AllocatorRef allocator = mem::system_allocator();
    void *storage =
        allocator.allocate(sizeof(DeviceNameRecord), alignof(DeviceNameRecord));
    if (storage == nullptr)
      return "NVIDIA CUDA GPU";
    auto *record = ::new (storage) DeviceNameRecord;
    record->ordinal = ordinal;
    if (name != nullptr) {
      const std::size_t length =
          std::min(std::strlen(name), sizeof(record->text) - 1);
      std::memcpy(record->text, name, length);
      record->text[length] = '\0';
    }
    if (record->text[0] == '\0') {
      constexpr char fallback[] = "NVIDIA CUDA GPU";
      std::memcpy(record->text, fallback, sizeof(fallback));
    }
    record->next = records_;
    records_ = record;
    return record->text;
  }

private:
  util::SpinLock lock_{};
  DeviceNameRecord *records_ = nullptr;
};

DeviceNameCache &device_names() noexcept {
  static DeviceNameCache cache;
  return cache;
}

Result<std::size_t> device_count_impl(const Backend &) noexcept {
  int count = 0;
  Status result = status(cudaGetDeviceCount(&count));
  if (!result.ok())
    return std::unexpected(std::move(result));
  if (count < 0)
    return std::unexpected(
        Status::Internal("CUDA returned a negative device count"));
  return static_cast<std::size_t>(count);
}

Result<Device> device_impl(const Backend &, std::size_t index) noexcept;

bool same_device_impl(const Device &left, const Device &right) noexcept {
  return compute::detail::DeviceAccess::payload<DevicePayload>(left).ordinal ==
         compute::detail::DeviceAccess::payload<DevicePayload>(right).ordinal;
}

Result<DeviceProperties> properties_impl(const Device &value) noexcept {
  const DevicePayload payload =
      compute::detail::DeviceAccess::payload<DevicePayload>(value);
  cudaDeviceProp properties{};
  Status result = status(cudaGetDeviceProperties(&properties, payload.ordinal));
  if (!result.ok())
    return std::unexpected(std::move(result));
  return DeviceProperties{
      .backend = kBackendId,
      .device_class = DeviceClass::kGpu,
      .backend_name = "cuda",
      .name = device_names().get(payload.ordinal, properties.name),
      .unified_memory = properties.managedMemory != 0,
      .asynchronous_queues = true,
  };
}

Status validate_resource_impl(const Device &value,
                              mem::MemoryResourceRef resource_value) noexcept {
  if (!resource_value.valid())
    return Status::InvalidArgument("CUDA tensor resource is invalid");
  const DevicePayload payload =
      compute::detail::DeviceAccess::payload<DevicePayload>(value);
  const cuda::detail::CudaResourceBase *cuda_resource =
      cuda::detail::resource_const(resource_value);
  if (cuda_resource == nullptr || cuda_resource->ordinal() != payload.ordinal) {
    return Status::FailedPrecondition(
        "CUDA tensor resource belongs to a different backend or device");
  }
  return Status::Ok();
}

Status create_context_impl(const Device &value, mem::AllocatorRef allocator,
                           void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA context output is null");
  *output = nullptr;
  if (!allocator.valid())
    return Status::InvalidArgument("CUDA context allocator is invalid");

  const DevicePayload payload =
      compute::detail::DeviceAccess::payload<DevicePayload>(value);
  HYLE_RETURN_IF_ERROR(cuda::detail::driver_status(cuInit(0)));
  CUdevice driver_device = 0;
  HYLE_RETURN_IF_ERROR(cuda::detail::driver_status(
      cuDeviceGet(&driver_device, payload.ordinal)));
  CUcontext primary = nullptr;
  HYLE_RETURN_IF_ERROR(cuda::detail::driver_status(
      cuDevicePrimaryCtxRetain(&primary, driver_device)));

  cudaExecutionContext_t execution = nullptr;
  Status result =
      status(cudaDeviceGetExecutionCtx(&execution, payload.ordinal));
  if (!result.ok()) {
    (void)cuDevicePrimaryCtxRelease(driver_device);
    return result;
  }
  int execution_device = -1;
  result = status(cudaExecutionCtxGetDevice(&execution_device, execution));
  if (!result.ok() || execution_device != payload.ordinal) {
    (void)cuDevicePrimaryCtxRelease(driver_device);
    return result.ok()
               ? Status::Internal("CUDA execution context device mismatch")
               : result;
  }
  result = cuda::detail::CompletionReaper::instance().start();
  if (!result.ok()) {
    (void)cuDevicePrimaryCtxRelease(driver_device);
    return result.WithContext("starting CUDA completion reaper");
  }

  void *storage = allocator.allocate(sizeof(cuda::detail::ContextState),
                                     alignof(cuda::detail::ContextState));
  if (storage == nullptr) {
    (void)cuDevicePrimaryCtxRelease(driver_device);
    return Status::ResourceExhausted("CUDA context-state allocation failed");
  }
  *output = ::new (storage) cuda::detail::ContextState(
      value, payload.ordinal, execution, primary, allocator);
  return Status::Ok();
}

void destroy_context_impl(void *context, mem::AllocatorRef) noexcept {
  if (context != nullptr)
    static_cast<cuda::detail::ContextState *>(context)->decref();
}

Status create_owned_queue(cuda::detail::ContextState *context,
                          mem::AllocatorRef allocator,
                          cuda::detail::StreamOwnership ownership,
                          cudaStream_t supplied, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA queue output is null");
  *output = nullptr;
  cudaStream_t native_stream = supplied;
  if (ownership == cuda::detail::StreamOwnership::kOwned) {
    HYLE_RETURN_IF_ERROR(status(cudaExecutionCtxStreamCreate(
        &native_stream, context->execution, cudaStreamNonBlocking, 0)));
  }
  auto state = cuda::detail::create_queue_state(context, native_stream,
                                                ownership, allocator);
  if (!state) {
    if (ownership == cuda::detail::StreamOwnership::kOwned)
      status(cudaStreamDestroy(native_stream)).IgnoreError();
    return std::move(state).error();
  }
  *output = *state;
  return Status::Ok();
}

Status create_queue_impl(void *context, QueueOptions,
                         mem::AllocatorRef allocator, void **output) noexcept {
  return create_owned_queue(static_cast<cuda::detail::ContextState *>(context),
                            allocator, cuda::detail::StreamOwnership::kOwned,
                            nullptr, output);
}

void destroy_queue_impl(void *queue, mem::AllocatorRef) noexcept {
  if (queue != nullptr)
    static_cast<cuda::detail::QueueState *>(queue)->decref();
}

void completion_destroy(void *state,
                        compute::detail::CompletionLifetime lifetime,
                        Status terminal_hint) noexcept;
Result<CompletionPoll> completion_poll(void *state) noexcept;
Status completion_wait(void *state) noexcept;
Status completion_cancel(void *) noexcept;

const compute::detail::CompletionVTable kCompletionVTable{
    .destroy = &completion_destroy,
    .poll = &completion_poll,
    .wait = &completion_wait,
    .request_cancel = &completion_cancel,
};

void completion_destroy(void *state,
                        compute::detail::CompletionLifetime lifetime,
                        Status terminal_hint) noexcept {
  auto *slot = static_cast<cuda::detail::EventSlot *>(state);
  auto result = cuda::detail::poll_slot(slot);
  if (result && *result == CompletionPoll::kPending) {
    cuda::detail::CompletionReaper::instance().retire(slot, lifetime,
                                                      std::move(terminal_hint));
    return;
  }
  slot->lifetime = lifetime;
  slot->terminal_hint = std::move(terminal_hint);
  Status observed = Status::Ok();
  if (!result)
    observed = std::move(result).error();
  cuda::detail::finish_abandoned(slot, std::move(observed));
}

Result<CompletionPoll> completion_poll(void *state) noexcept {
  return cuda::detail::poll_slot(static_cast<cuda::detail::EventSlot *>(state));
}

Status completion_wait(void *state) noexcept {
  return cuda::detail::wait_slot(static_cast<cuda::detail::EventSlot *>(state));
}

Status completion_cancel(void *) noexcept {
  return Status::Unimplemented("CUDA work cannot be preemptively cancelled");
}

Status begin_sequence_impl(void *queue, void **sequence) noexcept {
  if (sequence == nullptr)
    return Status::InvalidArgument("CUDA sequence output is null");
  *sequence = nullptr;
  auto slot = static_cast<cuda::detail::QueueState *>(queue)->acquire();
  if (!slot)
    return std::move(slot).error();
  *sequence = *slot;
  return Status::Ok();
}

Completion finish_sequence_impl(void *queue, void *sequence, bool accepted,
                                Status terminal) noexcept {
  auto *queue_state = static_cast<cuda::detail::QueueState *>(queue);
  auto *slot = static_cast<cuda::detail::EventSlot *>(sequence);
  if (slot == nullptr || slot->owner != queue_state) {
    return Completion::completed(
        Status::Internal("CUDA sequence slot does not belong to queue"));
  }
  if (!accepted) {
    queue_state->release(slot);
    return Completion::completed(std::move(terminal));
  }

  const cudaError_t recorded =
      cudaEventRecord(slot->event, queue_state->stream);
  slot->fence = recorded == cudaSuccess
                    ? cuda::detail::FenceKind::kEvent
                    : cuda::detail::FenceKind::kStreamFallback;
  Completion completion =
      compute::detail::CompletionAccess::make(&kCompletionVTable, slot);
  if (recorded != cudaSuccess)
    compute::detail::CompletionAccess::append_terminal(completion,
                                                       status(recorded));
  compute::detail::CompletionAccess::append_terminal(completion,
                                                     std::move(terminal));
  return completion;
}

Status reserve_completions_impl(void *queue, std::size_t minimum) noexcept {
  return static_cast<cuda::detail::QueueState *>(queue)->reserve(minimum);
}

bool queue_matches_resource(const cuda::detail::QueueState *queue,
                            const cuda::detail::CudaResourceBase *resource) {
  return resource != nullptr && resource->context() == queue->context;
}

Result<TransferProperties>
transfer_properties_impl(void *context, mem::BufferView destination,
                         mem::ConstBufferView source) noexcept {
  auto *selected = static_cast<cuda::detail::ContextState *>(context);
  const auto *destination_resource =
      cuda::detail::resource_const(destination.resource());
  const auto *source_resource = cuda::detail::resource_const(source.resource());
  if (destination_resource == nullptr || source_resource == nullptr ||
      destination_resource->context() != selected ||
      source_resource->context() != selected) {
    return std::unexpected(
        Status::Unimplemented("CUDA async transfer requires same-context "
                              "device, pinned, or managed storage"));
  }
  return TransferProperties{.mode = TransferMode::kQueuedDirect, .peer = false};
}

compute::detail::EnqueueResult
copy_async_impl(void *queue, mem::BufferView destination,
                mem::ConstBufferView source) noexcept {
  auto *selected = static_cast<cuda::detail::QueueState *>(queue);
  const auto *destination_resource =
      cuda::detail::resource_const(destination.resource());
  const auto *source_resource = cuda::detail::resource_const(source.resource());
  if (!queue_matches_resource(selected, destination_resource) ||
      !queue_matches_resource(selected, source_resource)) {
    return compute::detail::EnqueueResult::rejected(Status::Unimplemented(
        "CUDA async copy requires same-context CUDA storage"));
  }
  const cudaError_t result =
      cudaMemcpyAsync(cuda::detail::byte_pointer(destination),
                      cuda::detail::byte_pointer(source), destination.size(),
                      cudaMemcpyDefault, selected->stream);
  return compute::detail::EnqueueResult::accepted_command(status(result));
}

compute::detail::EnqueueResult fill_async_impl(void *queue,
                                               mem::BufferView destination,
                                               std::byte value) noexcept {
  auto *selected = static_cast<cuda::detail::QueueState *>(queue);
  const auto *destination_resource =
      cuda::detail::resource_const(destination.resource());
  if (!queue_matches_resource(selected, destination_resource) ||
      destination_resource->kind() == MemoryKind::kPinnedHost) {
    return compute::detail::EnqueueResult::rejected(Status::Unimplemented(
        "CUDA async fill requires same-context device or managed storage"));
  }
  const cudaError_t result =
      cudaMemsetAsync(cuda::detail::byte_pointer(destination),
                      std::to_integer<unsigned char>(value), destination.size(),
                      selected->stream);
  return compute::detail::EnqueueResult::accepted_command(status(result));
}

compute::detail::EnqueueResult
wait_completion_impl(void *queue,
                     const compute::detail::CompletionVTable *completion_vtable,
                     void *completion_state) noexcept {
  auto *destination = static_cast<cuda::detail::QueueState *>(queue);
  if (completion_vtable != &kCompletionVTable || completion_state == nullptr) {
    return compute::detail::EnqueueResult::rejected(Status::Unimplemented(
        "CUDA queue can wait only on a compatible CUDA completion"));
  }
  auto *source = static_cast<cuda::detail::EventSlot *>(completion_state);
  if (source->owner->context != destination->context ||
      source->fence != cuda::detail::FenceKind::kEvent) {
    return compute::detail::EnqueueResult::rejected(Status::Unimplemented(
        "CUDA completion is not an event from the same context"));
  }
  const cudaError_t result =
      cudaStreamWaitEvent(destination->stream, source->event, 0);
  return compute::detail::EnqueueResult::accepted_command(status(result));
}

compute::detail::CompletionEndpoint
completion_endpoint_impl(void *source_queue) noexcept {
  const auto *source =
      static_cast<const cuda::detail::QueueState *>(source_queue);
  return source == nullptr
             ? compute::detail::CompletionEndpoint{}
             : compute::detail::CompletionEndpoint{
                   .authority = &kCompletionVTable,
                   .identity = source->context,
               };
}

bool can_wait_endpoint_impl(
    void *destination_queue,
    compute::detail::CompletionEndpoint source) noexcept {
  const auto *destination =
      static_cast<const cuda::detail::QueueState *>(destination_queue);
  return destination != nullptr && source.authority == &kCompletionVTable &&
         source.identity == destination->context;
}

inline constexpr BackendExtensionId kCudaExtensionId{0x48594c4500000001ULL,
                                                     0x4355444100000001ULL, 1};

struct CudaExtensionV1 final {};
const CudaExtensionV1 kCudaExtension{};

struct CapturedCommandsState final {
  cuda::detail::ContextState *context = nullptr;
  cudaGraphExec_t executable = nullptr;
  mem::AllocatorRef allocator{};
};

Status begin_capture(void *backend_context, void *backend_queue) noexcept {
  auto *context = static_cast<cuda::detail::ContextState *>(backend_context);
  auto *queue = static_cast<cuda::detail::QueueState *>(backend_queue);
  if (context == nullptr || queue == nullptr || queue->context != context)
    return Status::InvalidArgument("CUDA capture queue is invalid");
  if (queue->capture_owned)
    return Status::FailedPrecondition("CUDA queue capture is already active");
  const Status begun = status(
      cudaStreamBeginCapture(queue->stream, cudaStreamCaptureModeThreadLocal));
  if (begun.ok())
    queue->capture_owned = true;
  return begun;
}

Status end_capture(void *backend_context, void *backend_queue,
                   mem::AllocatorRef allocator,
                   void **captured_commands) noexcept {
  auto *context = static_cast<cuda::detail::ContextState *>(backend_context);
  auto *queue = static_cast<cuda::detail::QueueState *>(backend_queue);
  if (captured_commands == nullptr)
    return Status::InvalidArgument("CUDA capture output is null");
  *captured_commands = nullptr;
  if (context == nullptr || queue == nullptr || queue->context != context ||
      !queue->capture_owned || !allocator.valid()) {
    return Status::FailedPrecondition("CUDA capture session is invalid");
  }
  cudaGraph_t graph = nullptr;
  const cudaError_t ended = cudaStreamEndCapture(queue->stream, &graph);
  queue->capture_owned = false;
  if (ended != cudaSuccess) {
    if (graph != nullptr)
      status(cudaGraphDestroy(graph)).IgnoreError();
    return status(ended);
  }
  if (graph == nullptr)
    return Status::Internal("CUDA capture returned no graph");
  cudaGraphExec_t executable = nullptr;
  const cudaError_t instantiated =
      cudaGraphInstantiateWithFlags(&executable, graph, 0);
  const Status graph_destroyed = status(cudaGraphDestroy(graph));
  if (instantiated != cudaSuccess)
    return status(instantiated);
  if (!graph_destroyed.ok()) {
    status(cudaGraphExecDestroy(executable)).IgnoreError();
    return graph_destroyed;
  }
  void *storage = allocator.allocate(sizeof(CapturedCommandsState),
                                     alignof(CapturedCommandsState));
  if (storage == nullptr) {
    status(cudaGraphExecDestroy(executable)).IgnoreError();
    return Status::ResourceExhausted("CUDA graph artifact allocation failed");
  }
  context->incref();
  *captured_commands =
      ::new (storage) CapturedCommandsState{context, executable, allocator};
  return Status::Ok();
}

Status abort_capture(void *backend_context, void *backend_queue) noexcept {
  auto *context = static_cast<cuda::detail::ContextState *>(backend_context);
  auto *queue = static_cast<cuda::detail::QueueState *>(backend_queue);
  if (context == nullptr || queue == nullptr || queue->context != context ||
      !queue->capture_owned) {
    return Status::FailedPrecondition("CUDA capture session is not active");
  }
  cudaGraph_t graph = nullptr;
  const cudaError_t ended = cudaStreamEndCapture(queue->stream, &graph);
  queue->capture_owned = false;
  Status result = status(ended);
  if (graph != nullptr)
    result.Update(status(cudaGraphDestroy(graph)));
  return result;
}

void destroy_capture(void *captured_commands,
                     mem::AllocatorRef allocator) noexcept {
  auto *captured = static_cast<CapturedCommandsState *>(captured_commands);
  if (captured == nullptr)
    return;
  cuda::detail::ContextState *context = captured->context;
  {
    cuda::detail::ScopedPrimaryContext scope(context->primary);
    if (scope)
      status(cudaGraphExecDestroy(captured->executable)).IgnoreError();
  }
  captured->~CapturedCommandsState();
  allocator.deallocate(captured, sizeof(CapturedCommandsState),
                       alignof(CapturedCommandsState));
  context->decref();
}

compute::detail::EnqueueResult
launch_capture(void *backend_context, void *backend_queue,
               void *captured_commands) noexcept {
  auto *context = static_cast<cuda::detail::ContextState *>(backend_context);
  auto *queue = static_cast<cuda::detail::QueueState *>(backend_queue);
  auto *captured = static_cast<CapturedCommandsState *>(captured_commands);
  if (context == nullptr || queue == nullptr || captured == nullptr ||
      queue->context != context || captured->context != context) {
    return compute::detail::EnqueueResult::rejected(
        Status::InvalidArgument("CUDA graph launch context is invalid"));
  }
  const cudaError_t launched =
      cudaGraphLaunch(captured->executable, queue->stream);
  if (launched != cudaSuccess) {
    return compute::detail::EnqueueResult::rejected(status(launched));
  }
  return compute::detail::EnqueueResult::accepted_command();
}

const compute::detail::CommandCaptureVTable kCommandCaptureVTable{
    .begin = &begin_capture,
    .end = &end_capture,
    .abort = &abort_capture,
    .destroy = &destroy_capture,
    .launch = &launch_capture,
};

const void *query_extension_impl(void *, BackendExtensionId id) noexcept {
  if (id == kCudaExtensionId)
    return &kCudaExtension;
  if (id == compute::detail::kCommandCaptureExtensionId)
    return &kCommandCaptureVTable;
  return nullptr;
}

Status drain_queue_impl(void *queue) noexcept {
  return status(cudaStreamSynchronize(
      static_cast<cuda::detail::QueueState *>(queue)->stream));
}

Status drain_context_impl(void *context) noexcept {
  auto *selected = static_cast<cuda::detail::ContextState *>(context);
  std::size_t count = 0;
  {
    std::lock_guard lock(selected->queues_lock);
    for (auto *queue = selected->queues; queue != nullptr;
         queue = queue->context_next)
      ++count;
  }
  if (count == 0)
    return Status::Ok();
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(void *))
    return Status::OutOfRange("CUDA queue snapshot is too large");

  auto **queues = static_cast<cuda::detail::QueueState **>(
      selected->allocator.allocate(count * sizeof(void *), alignof(void *)));
  if (queues == nullptr) {
    return Status::ResourceExhausted(
        "CUDA context drain snapshot allocation failed");
  }
  std::size_t captured = 0;
  {
    std::lock_guard lock(selected->queues_lock);
    for (auto *queue = selected->queues; queue != nullptr && captured < count;
         queue = queue->context_next) {
      queue->incref();
      queues[captured++] = queue;
    }
  }
  Status result = Status::Ok();
  for (std::size_t index = 0; index < captured; ++index) {
    result.Update(status(cudaStreamSynchronize(queues[index]->stream)));
    queues[index]->decref();
  }
  selected->allocator.deallocate(queues, count * sizeof(void *),
                                 alignof(void *));
  return result;
}

const compute::detail::BackendVTable kBackendVTable{
    .id = kBackendId,
    .name = "cuda",
    .device_class = DeviceClass::kGpu,
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
    .completion_endpoint = &completion_endpoint_impl,
    .can_wait_endpoint = &can_wait_endpoint_impl,
    .query_extension = &query_extension_impl,
    .drain_context = &drain_context_impl,
    .drain_queue = &drain_queue_impl,
};

Result<Device> device_impl(const Backend &, std::size_t index) noexcept {
  if (index > static_cast<std::size_t>(INT_MAX)) {
    return std::unexpected(
        Status::OutOfRange("CUDA device ordinal is out of range"));
  }
  auto count = device_count_impl(Backend{});
  if (!count)
    return std::unexpected(std::move(count).error());
  if (index >= *count)
    return std::unexpected(
        Status::OutOfRange("CUDA device ordinal is out of range"));
  return compute::detail::DeviceAccess::make(
      &kBackendVTable, DevicePayload{.ordinal = static_cast<int>(index)});
}

Result<compute::detail::BackendExtensionRef>
cuda_extension(const Context &context) noexcept {
  return compute::detail::BackendExtensionAccess::query(context,
                                                        kCudaExtensionId);
}

Result<compute::detail::BackendExtensionRef>
cuda_extension(const Queue &queue) noexcept {
  return compute::detail::BackendExtensionAccess::query(queue,
                                                        kCudaExtensionId);
}

} // namespace

Backend backend() noexcept {
  return compute::detail::BackendAccess::make(&kBackendVTable,
                                              BackendPayload{});
}

Result<Device> device(std::size_t ordinal) noexcept {
  return backend().device(ordinal);
}

Result<int> device_ordinal(Device value) noexcept {
  if (compute::detail::DeviceAccess::vtable(value) != &kBackendVTable) {
    return std::unexpected(
        Status::InvalidArgument("device is not a CUDA device"));
  }
  return compute::detail::DeviceAccess::payload<DevicePayload>(value).ordinal;
}

namespace {

Status execution_context_call(void *backend_context, const void *,
                              void *call_context) noexcept {
  *static_cast<cudaExecutionContext_t *>(call_context) =
      static_cast<cuda::detail::ContextState *>(backend_context)->execution;
  return Status::Ok();
}

Status primary_context_call(void *backend_context, const void *,
                            void *call_context) noexcept {
  *static_cast<CUcontext *>(call_context) =
      static_cast<cuda::detail::ContextState *>(backend_context)->primary;
  return Status::Ok();
}

compute::detail::EnqueueResult submit_call(void *, const void *,
                                           void *backend_queue,
                                           void *call_context) noexcept {
  struct SubmitContext final {
    StreamCommand command = nullptr;
    void *context = nullptr;
  };
  auto *selected = static_cast<cuda::detail::QueueState *>(backend_queue);
  auto *call = static_cast<SubmitContext *>(call_context);
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  const cudaError_t queried = cudaStreamIsCapturing(selected->stream, &capture);
  if (queried != cudaSuccess) {
    return compute::detail::EnqueueResult::rejected(status(queried));
  }
  if (capture != cudaStreamCaptureStatusNone && !selected->capture_owned) {
    return compute::detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "cuda::submit does not join an externally active stream capture"));
  }
  return compute::detail::EnqueueResult::accepted_command(
      call->command(selected->stream, call->context));
}

struct ImportedStream final {
  cudaStream_t stream = nullptr;
  cuda::detail::StreamOwnership ownership =
      cuda::detail::StreamOwnership::kBorrowed;
};

Status imported_queue_factory(void *backend_context,
                              mem::AllocatorRef allocator, void *call_context,
                              void **output) noexcept {
  auto *imported = static_cast<ImportedStream *>(call_context);
  return create_owned_queue(
      static_cast<cuda::detail::ContextState *>(backend_context), allocator,
      imported->ownership, imported->stream, output);
}

Result<Queue> import_stream(const Context &context, cudaStream_t native_stream,
                            cuda::detail::StreamOwnership ownership) noexcept {
  if (native_stream == nullptr || native_stream == cudaStreamLegacy ||
      native_stream == cudaStreamPerThread) {
    return std::unexpected(Status::InvalidArgument(
        "CUDA imported stream must be a non-default stream"));
  }
  auto ordinal = device_ordinal(context.device());
  if (!ordinal)
    return std::unexpected(std::move(ordinal).error());

  unsigned int flags = 0;
  HYLE_RETURN_IF_ERROR(status(cudaStreamGetFlags(native_stream, &flags)));
  if ((flags & cudaStreamNonBlocking) == 0) {
    return std::unexpected(Status::FailedPrecondition(
        "CUDA imported stream must have cudaStreamNonBlocking"));
  }
  int stream_device = -1;
  HYLE_RETURN_IF_ERROR(
      status(cudaStreamGetDevice(native_stream, &stream_device)));
  if (stream_device != *ordinal) {
    return std::unexpected(Status::InvalidArgument(
        "CUDA imported stream belongs to a different device"));
  }
  CUcontext stream_context = nullptr;
  HYLE_RETURN_IF_ERROR(cuda::detail::driver_status(cuStreamGetCtx(
      reinterpret_cast<CUstream>(native_stream), &stream_context)));
  auto expected_context = primary_context(context);
  if (!expected_context)
    return std::unexpected(std::move(expected_context).error());
  if (stream_context != *expected_context) {
    return std::unexpected(Status::InvalidArgument(
        "CUDA imported stream belongs to a different execution context"));
  }

  ImportedStream imported{native_stream, ownership};
  return compute::detail::ContextAccess::create_queue(
      context, &imported_queue_factory, &imported);
}

} // namespace

Result<cudaExecutionContext_t>
execution_context(const Context &context) noexcept {
  auto extension = cuda_extension(context);
  if (!extension)
    return std::unexpected(std::move(extension).error());
  cudaExecutionContext_t result = nullptr;
  Status call = compute::detail::BackendExtensionAccess::call(
      *extension, &execution_context_call, &result);
  if (!call.ok())
    return std::unexpected(std::move(call));
  return result;
}

Result<CUcontext> primary_context(const Context &context) noexcept {
  auto extension = cuda_extension(context);
  if (!extension)
    return std::unexpected(std::move(extension).error());
  CUcontext result = nullptr;
  Status call = compute::detail::BackendExtensionAccess::call(
      *extension, &primary_context_call, &result);
  if (!call.ok())
    return std::unexpected(std::move(call));
  return result;
}

Result<cudaStream_t> stream(const Queue &queue) noexcept {
  auto extension = cuda_extension(queue);
  if (!extension)
    return std::unexpected(std::move(extension).error());
  auto *state = static_cast<cuda::detail::QueueState *>(
      compute::detail::ContextAccess::backend_queue(queue));
  if (state == nullptr)
    return std::unexpected(Status::Internal("CUDA queue state is null"));
  return state->stream;
}

Result<Queue> borrow_stream(const Context &context,
                            cudaStream_t native_stream) noexcept {
  return import_stream(context, native_stream,
                       cuda::detail::StreamOwnership::kBorrowed);
}

Result<Queue> adopt_stream(const Context &context,
                           cudaStream_t native_stream) noexcept {
  return import_stream(context, native_stream,
                       cuda::detail::StreamOwnership::kAdopted);
}

Result<Completion> submit(Queue &queue, StreamCommand command,
                          void *context) noexcept {
  if (command == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("CUDA stream command is null"));
  }
  auto extension = cuda_extension(queue);
  if (!extension)
    return std::unexpected(std::move(extension).error());
  struct SubmitContext final {
    StreamCommand command = nullptr;
    void *context = nullptr;
  } call{command, context};
  return compute::detail::BackendExtensionAccess::submit(*extension, queue,
                                                         &submit_call, &call);
}

} // namespace hyle::compute::cuda
