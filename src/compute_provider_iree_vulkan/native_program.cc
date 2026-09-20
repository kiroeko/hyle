#include "native_program.h"

#include <array>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <span>

#include "runtime_internal.h"

namespace hyle::compute::iree_vulkan {
namespace {
using View =
    detail::Owner<iree_hal_buffer_view_t, iree_hal_buffer_view_release>;
using Fence = detail::Owner<iree_hal_fence_t, iree_hal_fence_release>;

Result<View> allocate_view(iree_hal_device_t *device,
                           std::span<const iree_hal_dim_t> shape,
                           iree_hal_element_type_t type) noexcept {
  iree_device_size_t bytes = 0;
  HYLE_RETURN_IF_ERROR(detail::consume_status(iree_hal_buffer_compute_view_size(
      shape.size(), shape.data(), type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      &bytes)));
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree_hal_buffer_t *buffer = nullptr;
  HYLE_RETURN_IF_ERROR(
      detail::consume_status(iree_hal_allocator_allocate_buffer(
          iree_hal_device_allocator(device), params, bytes, &buffer)));
  detail::Owner<iree_hal_buffer_t, iree_hal_buffer_release> owner(buffer);
  iree_hal_buffer_view_t *view = nullptr;
  HYLE_RETURN_IF_ERROR(detail::consume_status(iree_hal_buffer_view_create(
      buffer, shape.size(), shape.data(), type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(), &view)));
  return View(view);
}

Result<Fence> fence(iree_hal_device_t *device) noexcept {
  iree_hal_semaphore_t *semaphore = nullptr;
  HYLE_RETURN_IF_ERROR(detail::consume_status(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore)));
  detail::Owner<iree_hal_semaphore_t, iree_hal_semaphore_release> owner(
      semaphore);
  iree_hal_fence_t *result = nullptr;
  HYLE_RETURN_IF_ERROR(detail::consume_status(iree_hal_fence_create_at(
      semaphore, 1, iree_allocator_system(), &result)));
  return Fence(result);
}

Result<BufferSpan> span(iree_hal_buffer_view_t *view) noexcept {
  auto *buffer = iree_hal_buffer_view_buffer(view);
  iree_hal_buffer_t *backing = nullptr;
  HYLE_RETURN_IF_ERROR(detail::consume_status(
      iree_hal_vulkan_buffer_resolve_backing(buffer, &backing)));
  VkDeviceMemory memory = VK_NULL_HANDLE;
  BufferSpan result;
  HYLE_RETURN_IF_ERROR(detail::consume_status(
      iree_hal_vulkan_buffer_handle(backing, &memory, &result.buffer)));
  iree_device_size_t offset = 0;
  HYLE_RETURN_IF_ERROR(
      detail::consume_status(iree_hal_vulkan_buffer_resolve_backing_offset(
          buffer, backing, 0, &offset)));
  result.offset = offset;
  const auto bytes = iree_hal_buffer_view_byte_length(view);
  if (bytes > std::numeric_limits<std::size_t>::max())
    return std::unexpected(Status::OutOfRange("IREE native buffer size"));
  result.bytes = static_cast<std::size_t>(bytes);
  return result;
}
} // namespace

struct NativeProgram::State final {
  NativeDevice device;
  ProgramSpec spec;
  detail::Owner<iree_vm_context_t, iree_vm_context_release> context;
  iree_vm_function_t function = {};
  std::array<View, 4> views;
  std::size_t count = 0;
  detail::Owner<iree_vm_list_t, iree_vm_list_release> inputs, outputs;
  Fence ready, done, retired;
  bool retiring = false;
  std::mutex mutex;
  std::condition_variable changed;
  iree_async_proactor_t *proactor = nullptr; // Borrowed from device's pool.
  iree_vm_async_invoke_state_t invocation = {};
  std::size_t callbacks = 0;
  bool vm_finished = false;
  bool gpu_finished = false;
  CompletionNotify notify = nullptr;
  void *notify_data = nullptr;
  bool notified = false;
  bool notifying = false;
  bool pending = false;
  bool invoked = false;
  Status failure;

  void notify_ready(std::unique_lock<std::mutex> &lock) noexcept {
    // A notification may enqueue a native copy and extend this same slot.
    // If that copy finishes while the notification is returning, the last
    // caller of this helper delivers its next notification without polling.
    while (vm_finished && gpu_finished && callbacks == 0 && !notifying) {
      if (!notified) {
        notified = true;
        notifying = true;
        const auto callback = notify;
        const auto status = failure;
        lock.unlock();
        if (callback)
          callback(notify_data, status);
        lock.lock();
        notifying = false;
        continue;
      }
      pending = false;
      break;
    }
    changed.notify_all();
  }

  iree_hal_fence_t *completion_fence() const noexcept {
    return retiring ? retired.get() : done.get();
  }
  void observe_completion() noexcept {
    const auto completed = iree_hal_fence_semaphore_list(completion_fence());
    auto observing = iree_vm_loop_wait_one(
        loop(),
        iree_hal_semaphore_await(completed.semaphores[0],
                                 completed.payload_values[0]),
        iree_infinite_timeout(), gpu_completed, this);
    if (!iree_status_is_ok(observing) && failure.ok())
      failure = detail::consume_status(observing);
    else
      iree_status_free(observing);
    // If notification submission itself fails, a host wait may still obtain
    // native retirement. It must not release resources on the error alone.
  }
  void retire_owner() noexcept {
    retiring = true;
    auto stopped = device.state_->request_stop();
    if (!stopped.ok()) {
      if (failure.ok())
        failure = stopped;
      return;
    }
    observe_completion();
  }

  // The locked artifacts only yield on original HAL semaphore waits. This is
  // a private VM/proactor adapter, not a general event loop or kernel executor.
  struct Operation {
    iree_async_semaphore_wait_operation_t wait = {};
    State *state = nullptr;
    iree_vm_loop_callback_t callback = {};
    std::vector<iree_async_semaphore_t *> semaphores;
    std::vector<std::uint64_t> values;
  };
  iree_vm_loop_t loop() noexcept { return {this, control}; }
  static iree_status_t control(void *self, iree_vm_loop_command_t command,
                               const void *parameters, void **) noexcept {
    auto &state = *static_cast<State *>(self);
    try {
      auto operation = std::make_unique<Operation>();
      operation->state = &state;
      auto &wait = operation->wait;
      wait.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
      wait.base.completion_fn = complete;
      wait.base.user_data = operation.get();
      wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
      std::span<const iree_wait_source_t> sources;
      iree_time_t deadline = IREE_TIME_INFINITE_FUTURE;
      if (command == IREE_VM_LOOP_COMMAND_CALL) {
        operation->callback =
            static_cast<const iree_vm_loop_call_params_t *>(parameters)
                ->callback;
      } else if (command == IREE_VM_LOOP_COMMAND_WAIT_ONE) {
        const auto &params =
            *static_cast<const iree_vm_loop_wait_one_params_t *>(parameters);
        operation->callback = params.callback;
        deadline = params.deadline_ns;
        sources = {&params.wait_source, 1};
      } else if (command == IREE_VM_LOOP_COMMAND_WAIT_ALL ||
                 command == IREE_VM_LOOP_COMMAND_WAIT_ANY) {
        const auto &params =
            *static_cast<const iree_vm_loop_wait_multi_params_t *>(parameters);
        operation->callback = params.callback;
        deadline = params.deadline_ns;
        sources = {params.wait_sources, params.count};
        if (command == IREE_VM_LOOP_COMMAND_WAIT_ANY)
          wait.mode = IREE_ASYNC_WAIT_MODE_ANY;
      } else {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unsupported locked-program VM wait");
      }
      if (deadline != IREE_TIME_INFINITE_FUTURE)
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "unexpected locked-program VM deadline");
      const auto ready = iree_hal_fence_semaphore_list(state.ready.get());
      const auto hal_resolve =
          iree_hal_semaphore_await(ready.semaphores[0], 0).resolve;
      for (auto source : sources) {
        if (iree_wait_source_is_immediate(source)) {
          if (wait.mode == IREE_ASYNC_WAIT_MODE_ANY) {
            operation->semaphores.clear();
            operation->values.clear();
            break;
          }
          continue;
        }
        if (source.resolve != hal_resolve)
          return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "VM wait is not an original HAL semaphore");
        operation->semaphores.push_back(
            static_cast<iree_async_semaphore_t *>(source.self));
        operation->values.push_back(source.data);
      }
      if (!operation->semaphores.empty()) {
        wait.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
        wait.semaphores = operation->semaphores.data();
        wait.values = operation->values.data();
        wait.count = operation->semaphores.size();
      }
      // Every caller holds mutex. A completion cannot touch the operation until
      // submit has returned and ownership has transferred to the proactor.
      ++state.callbacks;
      auto status = iree_async_proactor_submit_one(state.proactor, &wait.base);
      if (iree_status_is_ok(status))
        operation.release();
      else
        --state.callbacks;
      return status;
    } catch (const std::bad_alloc &) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "IREE VM continuation resources");
    }
  }
  static void complete(void *data, iree_async_operation_t *,
                       iree_status_t status,
                       iree_async_completion_flags_t) noexcept {
    std::unique_ptr<Operation> operation(static_cast<Operation *>(data));
    auto &state = *operation->state;
    std::unique_lock lock(state.mutex);
    const auto callback = operation->callback;
    // These are single-shot operations. The original proactor explicitly
    // returns operation storage ownership to its final callback.
    operation.reset();
    auto called = callback.fn(callback.user_data, state.loop(), status);
    if (!iree_status_is_ok(called) && state.failure.ok())
      state.failure = detail::consume_status(called);
    else
      iree_status_free(called);
    --state.callbacks;
    // notifying keeps the slot alive through the route callback tail, even
    // when that callback extends this program with a native-copy observation.
    state.notify_ready(lock);
  }
  static iree_status_t gpu_completed(void *data, iree_vm_loop_t,
                                     iree_status_t status) noexcept {
    auto &state = *static_cast<State *>(data);
    const auto code = iree_status_code(status);
    if (state.failure.ok())
      state.failure = detail::consume_status(status);
    else
      iree_status_free(status);
    if (code == IREE_STATUS_DATA_LOSS)
      state.device.state_->native.note_device_lost();
    if (code != IREE_STATUS_OK && !state.retiring)
      state.retire_owner();
    else if (code == IREE_STATUS_OK)
      state.gpu_finished = true;
    return iree_ok_status();
  }
  static iree_status_t vm_completed(void *data, iree_vm_loop_t,
                                    iree_status_t status,
                                    iree_vm_list_t *outputs) noexcept {
    auto &state = *static_cast<State *>(data);
    iree_vm_list_release(outputs); // The asynchronous API transfers this ref.
    state.vm_finished = true;
    if (!iree_status_is_ok(status)) {
      state.failure = detail::consume_status(status);
      state.retire_owner();
    } else {
      state.observe_completion();
    }
    return iree_ok_status();
  }
};

NativeProgram::NativeProgram(std::unique_ptr<State> state) noexcept
    : state_(std::move(state)) {}
NativeProgram::~NativeProgram() {
  (void)wait();
  // A failed native observation may have started a separate retirement watch.
  // Never free its callback cookie, even on the last-resort destruction path.
  std::unique_lock lock(state_->mutex);
  state_->changed.wait(lock, [&] {
    return !state_->invoked || (state_->vm_finished && state_->callbacks == 0);
  });
}

Result<std::unique_ptr<NativeProgram>>
NativeProgram::create(NativeDevice device, ProgramSpec spec) noexcept {
  if (!device || spec.rows <= 0 || spec.columns <= 0)
    return std::unexpected(Status::InvalidArgument(
        "IREE program owner and positive shape required"));
  if (device.state_->stopping.load(std::memory_order_acquire))
    return std::unexpected(
        Status::FailedPrecondition("IREE owner has stopped"));
  const bool scan = spec.kind == ProgramKind::ScanInclusive ||
                    spec.kind == ProgramKind::ScanExclusive;
  const auto &features = device.state_->native.native();
  if (scan && !features.shader_int64)
    return std::unexpected(
        Status::Unimplemented("checked Scan requires enabled shaderInt64"));
  if (!scan && (!features.shader_float16 || !features.storage_buffer16))
    return std::unexpected(Status::Unimplemented(
        "FP8 program requires enabled Float16 and storage16"));
  const auto block = spec.kind == ProgramKind::Float8Block128 ? 128 : 32;
  if (!scan &&
      (spec.depth <= 0 || spec.depth % block ||
       (spec.kind == ProgramKind::Float8Block128 && spec.columns % block)))
    return std::unexpected(
        Status::InvalidArgument("IREE FP8 complete block dimensions required"));
  // Original SPIR-V descriptor programs use 32-bit indexing, including F32
  // workspaces and the wide checked Scan intermediate. Reject before
  // allocation.
  constexpr std::int64_t max_elements =
      std::numeric_limits<std::int32_t>::max() / 8;
  if (spec.rows > max_elements / spec.columns ||
      (!scan && (spec.rows > max_elements / spec.depth ||
                 spec.columns > max_elements / spec.depth)))
    return std::unexpected(
        Status::OutOfRange("IREE descriptor program index/workspace domain"));
  try {
    auto state = std::make_unique<State>();
    state->device = std::move(device);
    state->spec = spec;
    auto &owner = *state->device.state_;
    iree_hal_fence_t *retired = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(iree_hal_fence_create_at(
        owner.retired.get(), 1, iree_allocator_system(), &retired)));
    state->retired.reset(retired);
    HYLE_RETURN_IF_ERROR(detail::consume_status(
        iree_async_proactor_pool_get(owner.pool.get(), 0, &state->proactor)));
    std::array modules{owner.hal.get(), owner.programs[scan ? 0 : 1].get()};
    iree_vm_context_t *context = nullptr;
    HYLE_RETURN_IF_ERROR(
        detail::consume_status(iree_vm_context_create_with_modules(
            iree_runtime_instance_vm_instance(owner.instance.get()),
            IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
            iree_allocator_system(), &context)));
    state->context.reset(context);
    const char *name = nullptr;
    switch (spec.kind) {
    case ProgramKind::ScanInclusive:
      name = "module.scan_inclusive";
      break;
    case ProgramKind::ScanExclusive:
      name = "module.scan_exclusive";
      break;
    case ProgramKind::Float8Block128:
      name = "module.fp8_f32_scale";
      break;
    case ProgramKind::Float8Group32:
      name = "module.fp8_ue8_scale";
      break;
    default:
      return std::unexpected(Status::InvalidArgument("IREE program kind"));
    }
    HYLE_RETURN_IF_ERROR(
        detail::consume_status(iree_vm_context_resolve_function(
            context, iree_make_cstring_view(name), &state->function)));
    iree_vm_list_t *inputs = nullptr, *outputs = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), 6,
                            iree_allocator_system(), &inputs)));
    state->inputs.reset(inputs);
    HYLE_RETURN_IF_ERROR(detail::consume_status(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                            iree_allocator_system(), &outputs)));
    state->outputs.reset(outputs);
    auto *hal = owner.device.get();
    if (scan) {
      const std::array<iree_hal_dim_t, 2> shape{
          static_cast<iree_hal_dim_t>(spec.rows),
          static_cast<iree_hal_dim_t>(spec.columns)};
      HYLE_ASSIGN_OR_RETURN(
          state->views[0],
          allocate_view(hal, shape, IREE_HAL_ELEMENT_TYPE_SINT_32));
      state->count = 1;
    } else {
      const auto rows = static_cast<iree_hal_dim_t>(spec.rows);
      const auto columns = static_cast<iree_hal_dim_t>(spec.columns);
      const auto depth = static_cast<iree_hal_dim_t>(spec.depth);
      const auto blocks = static_cast<iree_hal_dim_t>(spec.depth / block);
      const std::array<iree_hal_dim_t, 2> lhs{rows, depth}, rhs{columns, depth};
      const std::array<iree_hal_dim_t, 2> lhs_scale{rows, blocks},
          rhs_scale{spec.kind == ProgramKind::Float8Block128
                        ? static_cast<iree_hal_dim_t>(spec.columns / block)
                        : columns,
                    blocks};
      HYLE_ASSIGN_OR_RETURN(
          state->views[0],
          allocate_view(hal, lhs, IREE_HAL_ELEMENT_TYPE_SINT_8));
      HYLE_ASSIGN_OR_RETURN(
          state->views[1],
          allocate_view(hal, rhs, IREE_HAL_ELEMENT_TYPE_SINT_8));
      const auto type = spec.kind == ProgramKind::Float8Block128
                            ? IREE_HAL_ELEMENT_TYPE_FLOAT_32
                            : IREE_HAL_ELEMENT_TYPE_SINT_8;
      HYLE_ASSIGN_OR_RETURN(state->views[2],
                            allocate_view(hal, lhs_scale, type));
      HYLE_ASSIGN_OR_RETURN(state->views[3],
                            allocate_view(hal, rhs_scale, type));
      state->count = 4;
    }
    return std::unique_ptr<NativeProgram>(new NativeProgram(std::move(state)));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("IREE invocation resources"));
  }
}

std::size_t NativeProgram::input_count() const noexcept {
  return state_->count;
}
Result<BufferSpan> NativeProgram::input(std::size_t index) const noexcept {
  if (index >= state_->count)
    return std::unexpected(Status::OutOfRange("IREE input port"));
  return span(state_->views[index].get());
}

WorkState NativeProgram::invoke(CompletionNotify notify,
                                void *user_data) noexcept {
  auto &state = *state_;
  std::lock_guard lock(state.mutex);
  if (state.pending)
    return {Status::FailedPrecondition("IREE invocation is busy"), true};
  if (!state.failure.ok())
    return {state.failure, false};
  if (state.device.state_->stopping.load(std::memory_order_acquire))
    return {Status::FailedPrecondition("IREE owner has stopped"), false};
  auto prepare = [&]() -> Status {
    HYLE_RETURN_IF_ERROR(
        detail::consume_status(iree_vm_list_resize(state.inputs.get(), 0)));
    HYLE_RETURN_IF_ERROR(
        detail::consume_status(iree_vm_list_resize(state.outputs.get(), 0)));
    HYLE_ASSIGN_OR_RETURN(state.ready,
                          fence(state.device.state_->device.get()));
    HYLE_ASSIGN_OR_RETURN(state.done, fence(state.device.state_->device.get()));
    for (std::size_t i = 0; i < state.count; ++i) {
      auto ref = iree_hal_buffer_view_retain_ref(state.views[i].get());
      auto pushed = iree_vm_list_push_ref_move(state.inputs.get(), &ref);
      iree_vm_ref_release(&ref);
      HYLE_RETURN_IF_ERROR(detail::consume_status(pushed));
    }
    for (auto *value : {state.ready.get(), state.done.get()}) {
      auto ref = iree_hal_fence_retain_ref(value);
      auto pushed = iree_vm_list_push_ref_move(state.inputs.get(), &ref);
      iree_vm_ref_release(&ref);
      HYLE_RETURN_IF_ERROR(detail::consume_status(pushed));
    }
    return Status::Ok();
  };
  auto status = prepare();
  if (!status.ok())
    return {status, false};
  // The original VM emits its own submissions. No native owner queue lock is
  // held here; its delayed QueueSubmit2 calls enter that lock via the overlay.
  auto signaled =
      detail::consume_status(iree_hal_fence_signal(state.ready.get()));
  if (!signaled.ok())
    return {signaled, false};
  state.vm_finished = false;
  state.gpu_finished = false;
  state.notify = notify;
  state.notify_data = user_data;
  state.notified = false;
  state.pending = true;
  state.invoked = true;
  auto invoked = iree_vm_async_invoke(
      state.loop(), &state.invocation, state.context.get(), state.function,
      IREE_VM_INVOCATION_FLAG_NONE, nullptr, state.inputs.get(),
      state.outputs.get(), iree_allocator_system(), State::vm_completed,
      &state);
  if (!iree_status_is_ok(invoked)) {
    // The initial proactor call was not accepted; no VM work has begun.
    state.failure = detail::consume_status(invoked);
    state.vm_finished = true;
    state.pending = false;
  }
  return {state.failure, state.pending};
}

WorkState NativeProgram::observe(bool blocking) noexcept {
  auto &state = *state_;
  std::unique_lock lock(state.mutex);
  if (!state.pending)
    return {state.failure, false};
  auto finished = [&] {
    return state.vm_finished && state.callbacks == 0 && !state.notifying;
  };
  if (blocking)
    state.changed.wait(lock, finished);
  else if (!finished())
    return {state.failure, true};
  if (!state.pending)
    return {state.failure, false};
  if (state.retiring && !state.device.state_->request_stop().ok())
    return {state.failure, true};
  auto status = blocking ? iree_hal_fence_wait(state.completion_fence(),
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE)
                         : iree_hal_fence_query(state.completion_fence());
  if (iree_status_is_deferred(status)) {
    iree_status_free(status);
    return {state.failure, true};
  }
  const auto code = iree_status_code(status);
  auto completion = detail::consume_status(status);
  if (state.failure.ok())
    state.failure = completion;
  if (code == IREE_STATUS_DATA_LOSS)
    state.device.state_->native.note_device_lost();
  if (code != IREE_STATUS_OK) {
    if (!state.retiring)
      state.retire_owner();
    return {state.failure, true};
  }
  state.gpu_finished = true;
  state.notify_ready(lock);
  return {state.failure, state.pending};
}

Status NativeProgram::after_native_submissions(CompletionNotify notify,
                                               void *user_data) noexcept {
  auto &state = *state_;
  std::lock_guard lock(state.mutex);
  if (!state.notifying || !state.pending || !state.gpu_finished ||
      !state.vm_finished || state.retiring || !state.failure.ok())
    return Status::FailedPrecondition(
        "IREE native continuation needs a successful notification");
  HYLE_ASSIGN_OR_RETURN(auto done, fence(state.device.state_->device.get()));
  const auto signals = iree_hal_fence_semaphore_list(done.get());
  auto status = detail::consume_status(iree_hal_device_queue_barrier(
      state.device.state_->device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signals, IREE_HAL_EXECUTE_FLAG_NONE));
  if (!status.ok()) {
    state.failure = status;
    state.gpu_finished = false;
    state.notify = nullptr;
    state.retire_owner();
    return status;
  }
  state.done = std::move(done);
  state.gpu_finished = false;
  state.notified = false;
  state.notify = notify;
  state.notify_data = user_data;
  state.observe_completion();
  return state.failure;
}
WorkState NativeProgram::poll() noexcept { return observe(false); }
WorkState NativeProgram::wait() noexcept { return observe(true); }
Result<BufferSpan> NativeProgram::output(std::size_t index) const noexcept {
  std::lock_guard lock(state_->mutex);
  if (!state_->invoked || !state_->gpu_finished || !state_->failure.ok())
    return std::unexpected(
        Status::FailedPrecondition("IREE private output is not ready"));
  if (index >= iree_vm_list_size(state_->outputs.get()))
    return std::unexpected(Status::OutOfRange("IREE output port"));
  View view(iree_vm_list_get_buffer_view_retain(state_->outputs.get(), index));
  if (!view)
    return std::unexpected(Status::Internal("IREE output is not a tensor"));
  return span(view.get());
}
} // namespace hyle::compute::iree_vulkan
