#include "native_program.h"

#include <array>
#include <cmath>
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

const char *artifact_entry(const ProgramSpec &spec) noexcept {
  constexpr std::array nf4{"module.nf4_f32", "module.nf4_f16",
                            "module.nf4_bf16"};
  constexpr std::array nested{"module.nf4_nested_f32", "module.nf4_nested_f16",
                              "module.nf4_nested_bf16"};
  constexpr std::array awq{"module.awq_f32_f32", "module.awq_f32_f16",
                           "module.awq_f32_bf16"};
  constexpr std::array ct{"module.ct_f32_f32", "module.ct_f32_f16",
                          "module.ct_f32_bf16"};
  const auto scale = static_cast<std::size_t>(spec.scale_type);
  const auto output = static_cast<std::size_t>(spec.output_type);
  if (scale >= 3 || output >= 3)
    return nullptr;
  if ((spec.kind == ProgramKind::AwqDecode ||
       spec.kind == ProgramKind::Symmetric4Decode) &&
      spec.scale_type != ArtifactScalar::Float32 && scale != output)
    return nullptr; // Typed source result; final Cast is a qualified route fragment.
  switch (spec.kind) {
  case ProgramKind::Nf4Decode:
    return nf4[output];
  case ProgramKind::Nf4NestedDecode:
    return nested[output];
  case ProgramKind::AwqDecode:
    return scale == 1 ? "module.awq_f16_f16"
           : scale == 2 ? "module.awq_bf16_bf16" : awq[output];
  case ProgramKind::Symmetric4Decode:
    return scale == 1 ? "module.ct_f16_f16"
           : scale == 2 ? "module.ct_bf16_bf16" : ct[output];
  default:
    return nullptr;
  }
}

iree_hal_element_type_t artifact_scalar(ArtifactScalar type) noexcept {
  switch (type) {
  case ArtifactScalar::Float16:
    return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  case ArtifactScalar::BFloat16:
    return IREE_HAL_ELEMENT_TYPE_SINT_16; // Standard bf16 bit carrier.
  default:
    return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  }
}

const char *mlx_entry(const ProgramSpec &spec) noexcept {
  const auto type = static_cast<std::size_t>(spec.output_type);
  if (type >= 3 || !mlx_program(spec.kind)) return nullptr;
  constexpr std::array<std::array<const char *, 3>, 7> names{{
      {"module.affine_linear_f32", "module.affine_linear_f16", "module.affine_linear_bf16"},
      {"module.affine_grouped_f32", "module.affine_grouped_f16", "module.affine_grouped_bf16"},
      {"module.mx4_linear_f32", "module.mx4_linear_f16", "module.mx4_linear_bf16"},
      {"module.mx4_grouped_f32", "module.mx4_grouped_f16", "module.mx4_grouped_bf16"},
      {"module.mx8_linear_f32", "module.mx8_linear_f16", "module.mx8_linear_bf16"},
      {"module.mx8_grouped_f32", "module.mx8_grouped_f16", "module.mx8_grouped_bf16"},
      {"module.affine_embedding_f32", "module.affine_embedding_f16", "module.affine_embedding_bf16"}}};
  return names[static_cast<std::size_t>(spec.kind) -
               static_cast<std::size_t>(ProgramKind::MlxAffineLinear)][type];
}

const char *official_entry(ProgramKind kind) noexcept {
  constexpr std::array names{
      "module.weight_decode_f32", "module.weight_decode_bf16",
      "module.fp8_roundtrip_64", "module.fp8_roundtrip_128", "module.fp4_roundtrip_32",
      "module.linear_normal_f32", "module.linear_power_f32",
      "module.grouped8_f32", "module.grouped4_f32",
      "module.encoded_block128"};
  return official_program(kind)
      ? names[static_cast<std::size_t>(kind) - static_cast<std::size_t>(ProgramKind::OfficialWeightDecodeF32)]
      : nullptr;
}
} // namespace

struct NativeProgram::State final {
  NativeDevice device;
  ProgramSpec spec;
  detail::Owner<iree_vm_context_t, iree_vm_context_release> context;
  iree_vm_function_t function = {};
  std::array<View, 6> views;
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
  const bool precision = spec.kind == ProgramKind::Float32ToBFloat16 ||
                         spec.kind == ProgramKind::Float16ToBFloat16;
  const bool artifact = artifact_program(spec.kind);
  const bool mlx = mlx_program(spec.kind);
  const bool official = official_program(spec.kind);
  const bool iq1 = iq1_program(spec.kind);
  const auto &features = device.state_->native.native();
  if (iq1 && (spec.depth <= 0 || spec.depth % 256 || spec.experts <= 0))
    return std::unexpected(Status::InvalidArgument("original IQ1 complete block/expert domain"));
  if (iq1 && (!features.shader_int16 || !features.shader_float16 ||
              !features.storage_buffer16 || !features.preserve_float32_specials))
    return std::unexpected(Status::Unimplemented("IQ1 requires original numeric device features"));
  if (scan && !features.shader_int64)
    return std::unexpected(
        Status::Unimplemented("checked Scan requires enabled shaderInt64"));
  if (precision && (!features.shader_int16 || !features.storage_buffer16 ||
                    !features.preserve_float32_specials ||
                    (spec.kind == ProgramKind::Float16ToBFloat16 &&
                     !features.shader_float16)))
    return std::unexpected(Status::Unimplemented(
        "BF16 conversion requires enabled numeric device features"));
  if (artifact && (!artifact_entry(spec) || spec.group_size == 0 ||
                   spec.group_size > INT32_MAX ||
                   spec.columns % spec.group_size != 0 ||
                   (spec.columns % (spec.kind == ProgramKind::Nf4Decode ||
                                            spec.kind == ProgramKind::Nf4NestedDecode
                                        ? 2
                                        : 8)) != 0 ||
                   (spec.kind == ProgramKind::AwqDecode && spec.rows % 8 != 0)))
    return std::unexpected(Status::InvalidArgument("IREE original artifact domain"));
  if (artifact && (!features.shader_int16 || !features.shader_float16 ||
                   !features.storage_buffer16 ||
                   !features.preserve_float32_specials))
    return std::unexpected(Status::Unimplemented(
        "artifact program requires enabled original numeric features"));
  const bool embedding = spec.kind == ProgramKind::MlxAffineEmbedding;
  const bool mx8 = spec.kind == ProgramKind::MlxMx8Linear ||
                   spec.kind == ProgramKind::MlxMx8Grouped;
  const bool affine = spec.kind == ProgramKind::MlxAffineLinear ||
                      spec.kind == ProgramKind::MlxAffineGrouped || embedding;
  if (mlx && (!mlx_entry(spec) || spec.experts <= 0 || spec.depth <= 0 ||
              spec.group_size == 0 || spec.depth % spec.group_size ||
              (!affine && spec.group_size != 32) ||
              (affine && spec.group_size != 32 && spec.group_size != 64 &&
               spec.group_size != 128)))
    return std::unexpected(Status::InvalidArgument("IREE original MLX domain"));
  if (mlx && (!features.shader_int16 || !features.shader_float16 ||
              !features.storage_buffer16 || !features.preserve_float32_specials))
    return std::unexpected(Status::Unimplemented("MLX requires enabled original numeric features"));
  const bool official_decode = spec.kind == ProgramKind::OfficialWeightDecodeF32 ||
                               spec.kind == ProgramKind::OfficialWeightDecodeBf16;
  if (official && (spec.depth <= 0 || spec.selection_slots <= 0 ||
                   spec.selection_slots > INT32_MAX || spec.activation_slots <= 0 ||
                   spec.activation_slots > INT32_MAX || spec.experts <= 0 ||
                   spec.depth % (official_roundtrip(spec.kind)
                       ? (spec.kind == ProgramKind::OfficialRoundtrip4Group32 ? 32
                          : spec.kind == ProgramKind::OfficialRoundtrip8Group64 ? 64 : 128)
                       : 128) ||
                   (official_grouped(spec.kind) &&
                    (spec.rows % spec.selection_slots ||
                     (spec.activation_slots != 1 && spec.activation_slots != spec.selection_slots))) ||
                   (official_checked(spec.kind) && !official_roundtrip(spec.kind) &&
                    (!std::isfinite(spec.absmax_floor) || spec.absmax_floor <= 0 ||
                     !std::isnormal(spec.absmax_floor / 448.f)))))
    return std::unexpected(Status::InvalidArgument("original official float shape/scalar domain"));
  if (official && (!features.shader_int16 || !features.shader_float16 ||
                   !features.storage_buffer16 || !features.preserve_float32_specials))
    return std::unexpected(Status::Unimplemented("official float requires enabled original numeric features"));
  if (!scan && !precision && !artifact && !mlx && !official && !iq1 &&
      (!features.shader_float16 || !features.storage_buffer16))
    return std::unexpected(Status::Unimplemented(
        "FP8 program requires enabled Float16 and storage16"));
  const auto block = spec.kind == ProgramKind::Float8Block128 ? 128 : 32;
  if (!scan && !precision && !artifact && !mlx && !official && !iq1 &&
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
      (!scan && !precision && !artifact && !mlx && !official && !iq1 &&
       (spec.rows > max_elements / spec.depth ||
        spec.columns > max_elements / spec.depth)))
    return std::unexpected(
        Status::OutOfRange("IREE descriptor program index/workspace domain"));
  if (iq1) {
    const auto row_bytes = spec.depth / 256 * iq1_block_bytes(spec.kind);
    if (spec.rows > INT32_MAX / 4 / spec.depth ||
        spec.experts > INT32_MAX / spec.columns ||
        spec.experts * spec.columns > INT32_MAX / row_bytes ||
        spec.rows * spec.columns > INT32_MAX / 4 / 32)
      return std::unexpected(Status::OutOfRange("IQ1 original packed index/workspace domain"));
  }
  if (mlx) {
    constexpr auto limit = std::int64_t{INT32_MAX} / 4;
    const auto fits = [](std::int64_t x, std::int64_t y) {
      return x > 0 && y > 0 && x <= limit / y;
    };
    if ((embedding ? !fits(spec.experts, spec.depth / 8)
                   : (!fits(spec.rows, spec.depth) ||
                      !fits(spec.experts, spec.columns) ||
                      !fits(spec.experts * spec.columns, spec.depth / (mx8 ? 4 : 8)) ||
                      !fits(spec.rows * spec.columns, spec.group_size))))
      return std::unexpected(Status::OutOfRange("MLX original tensor index/workspace domain"));
  }
  if (official) {
    constexpr auto limit = std::int64_t{INT32_MAX} / 4;
    const auto fits = [](std::int64_t x, std::int64_t y) {
      return x > 0 && y > 0 && x <= limit / y;
    };
    if (official_decode ? !fits(spec.rows, spec.columns)
        : official_roundtrip(spec.kind) ? !fits(spec.rows, spec.columns)
        : (!fits(spec.rows, spec.depth) || !fits(spec.experts, spec.columns) ||
           !fits(spec.experts * spec.columns, spec.depth) ||
           !fits(spec.rows * spec.columns, spec.kind == ProgramKind::OfficialGrouped4F32 ? 32 : 128)))
      return std::unexpected(Status::OutOfRange("official float original tensor index/workspace domain"));
  }
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
    std::array modules{owner.hal.get(),
                       owner.programs[scan         ? 0
                                      : precision  ? 2
                                      : artifact   ? 3
                                      : mlx        ? 4
                                      : official   ? 5
                                      : iq1        ? 6
                                                   : 1]
                           .get()};
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
    case ProgramKind::Float32ToBFloat16:
      name = "module.f32_to_bf16";
      break;
    case ProgramKind::Float16ToBFloat16:
      name = "module.f16_to_bf16";
      break;
    case ProgramKind::Nf4Decode:
    case ProgramKind::Nf4NestedDecode:
    case ProgramKind::AwqDecode:
    case ProgramKind::Symmetric4Decode:
      name = artifact_entry(spec);
      break;
    case ProgramKind::MlxAffineLinear:
    case ProgramKind::MlxAffineGrouped:
    case ProgramKind::MlxMx4Linear:
    case ProgramKind::MlxMx4Grouped:
    case ProgramKind::MlxMx8Linear:
    case ProgramKind::MlxMx8Grouped:
    case ProgramKind::MlxAffineEmbedding:
      name = mlx_entry(spec);
      break;
    case ProgramKind::OfficialWeightDecodeF32:
    case ProgramKind::OfficialWeightDecodeBf16:
    case ProgramKind::OfficialRoundtrip8Group64:
    case ProgramKind::OfficialRoundtrip8Group128:
    case ProgramKind::OfficialRoundtrip4Group32:
    case ProgramKind::OfficialLinearF32:
    case ProgramKind::OfficialLinearPowerF32:
    case ProgramKind::OfficialGrouped8F32:
    case ProgramKind::OfficialGrouped4F32:
    case ProgramKind::OfficialEncodedBlock128:
      name = official_entry(spec.kind);
      break;
    case ProgramKind::Iq1XsGrouped:
      name = "module.iq1_xs";
      break;
    case ProgramKind::Iq1XxsGrouped:
      name = "module.iq1_xxs";
      break;
    case ProgramKind::Iq1XxxsGrouped:
      name = "module.iq1_xxxs";
      break;
    default:
      return std::unexpected(Status::InvalidArgument("IREE program kind"));
    }
    HYLE_RETURN_IF_ERROR(
        detail::consume_status(iree_vm_context_resolve_function(
            context, iree_make_cstring_view(name), &state->function)));
    iree_vm_list_t *inputs = nullptr, *outputs = nullptr;
    HYLE_RETURN_IF_ERROR(detail::consume_status(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), 9,
                            iree_allocator_system(), &inputs)));
    state->inputs.reset(inputs);
    HYLE_RETURN_IF_ERROR(detail::consume_status(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                            iree_allocator_system(), &outputs)));
    state->outputs.reset(outputs);
    auto *hal = owner.device.get();
    if (iq1) {
      const auto r = static_cast<iree_hal_dim_t>(spec.rows);
      const auto k = static_cast<iree_hal_dim_t>(spec.depth);
      const std::array<iree_hal_dim_t, 2> a{r, k};
      const std::array<iree_hal_dim_t, 1> w{
          static_cast<iree_hal_dim_t>(spec.experts * spec.columns *
                                      (spec.depth / 256) * iq1_block_bytes(spec.kind))},
          ids{r};
      HYLE_ASSIGN_OR_RETURN(state->views[0], allocate_view(hal, a, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
      HYLE_ASSIGN_OR_RETURN(state->views[1], allocate_view(hal, w, IREE_HAL_ELEMENT_TYPE_SINT_8));
      HYLE_ASSIGN_OR_RETURN(state->views[2], allocate_view(hal, ids, IREE_HAL_ELEMENT_TYPE_SINT_32));
      state->count = 3;
    } else if (official) {
      const auto r = static_cast<iree_hal_dim_t>(spec.rows);
      const auto n = static_cast<iree_hal_dim_t>(spec.columns);
      const auto k = static_cast<iree_hal_dim_t>(spec.depth);
      const auto e = static_cast<iree_hal_dim_t>(spec.experts);
      const auto type = artifact_scalar(spec.output_type);
      if (official_roundtrip(spec.kind)) {
        const std::array<iree_hal_dim_t, 2> a{r, n};
        HYLE_ASSIGN_OR_RETURN(state->views[0], allocate_view(hal, a, IREE_HAL_ELEMENT_TYPE_SINT_16));
        state->count = 1;
      } else if (official_decode) {
        const std::array<iree_hal_dim_t, 2> codes{r, n}, scales{(r + 127) / 128, n / 128};
        HYLE_ASSIGN_OR_RETURN(state->views[0], allocate_view(hal, codes, IREE_HAL_ELEMENT_TYPE_SINT_8));
        HYLE_ASSIGN_OR_RETURN(state->views[1], allocate_view(hal, scales, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
        state->count = 2;
      } else {
        const bool encoded = spec.kind == ProgramKind::OfficialEncodedBlock128;
        const bool grouped = official_grouped(spec.kind);
        const bool four = spec.kind == ProgramKind::OfficialGrouped4F32;
        const auto a_rows = grouped ? r / spec.selection_slots * spec.activation_slots : r;
        const std::array<iree_hal_dim_t, 2> a{a_rows, k}, a_scale{r, k / 128};
        const std::array<iree_hal_dim_t, 3> codes{e, n, four ? k / 2 : k},
            w_scale{e, four ? n : (n + 127) / 128, k / (four ? 32 : 128)};
        HYLE_ASSIGN_OR_RETURN(state->views[0], allocate_view(hal, a,
            encoded ? IREE_HAL_ELEMENT_TYPE_SINT_8 : type));
        std::size_t port = 1;
        if (encoded) {
          HYLE_ASSIGN_OR_RETURN(state->views[port], allocate_view(hal, a_scale, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
          ++port;
        }
        HYLE_ASSIGN_OR_RETURN(state->views[port], allocate_view(hal, codes, IREE_HAL_ELEMENT_TYPE_SINT_8));
        ++port;
        HYLE_ASSIGN_OR_RETURN(state->views[port], allocate_view(hal, w_scale, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
        ++port;
        if (grouped) {
          const std::array<iree_hal_dim_t, 1> ids{r};
          HYLE_ASSIGN_OR_RETURN(state->views[port], allocate_view(hal, ids, IREE_HAL_ELEMENT_TYPE_SINT_32));
          ++port;
        }
        state->count = port;
      }
    } else if (mlx) {
      const auto rows = static_cast<iree_hal_dim_t>(spec.rows);
      const auto columns = static_cast<iree_hal_dim_t>(spec.columns);
      const auto depth = static_cast<iree_hal_dim_t>(spec.depth);
      const auto experts = static_cast<iree_hal_dim_t>(spec.experts);
      const auto type = artifact_scalar(spec.output_type);
      if (embedding) {
        const std::array<iree_hal_dim_t, 2> words{experts, depth / 8},
            scales{experts, depth / spec.group_size};
        const std::array<iree_hal_dim_t, 1> ids{rows};
        HYLE_ASSIGN_OR_RETURN(state->views[0], allocate_view(hal, words, IREE_HAL_ELEMENT_TYPE_SINT_32));
        HYLE_ASSIGN_OR_RETURN(state->views[1], allocate_view(hal, scales, type));
        HYLE_ASSIGN_OR_RETURN(state->views[2], allocate_view(hal, scales, type));
        HYLE_ASSIGN_OR_RETURN(state->views[3], allocate_view(hal, ids, IREE_HAL_ELEMENT_TYPE_SINT_32));
        state->count = 4;
      } else {
        const std::array<iree_hal_dim_t, 2> activation{rows, depth};
        const std::array<iree_hal_dim_t, 3> words{experts, columns, depth / (mx8 ? 4 : 8)},
            scales{experts, columns, depth / spec.group_size};
        HYLE_ASSIGN_OR_RETURN(state->views[0], allocate_view(hal, activation, type));
        HYLE_ASSIGN_OR_RETURN(state->views[1], allocate_view(hal, words, IREE_HAL_ELEMENT_TYPE_SINT_32));
        HYLE_ASSIGN_OR_RETURN(state->views[2], allocate_view(hal, scales,
            affine ? type : IREE_HAL_ELEMENT_TYPE_SINT_8));
        state->count = 3;
        if (affine) {
          HYLE_ASSIGN_OR_RETURN(state->views[3], allocate_view(hal, scales, type));
          state->count = 4;
        }
        if (mlx_grouped(spec.kind)) {
          const std::array<iree_hal_dim_t, 1> ids{rows};
          HYLE_ASSIGN_OR_RETURN(state->views[state->count],
              allocate_view(hal, ids, IREE_HAL_ELEMENT_TYPE_SINT_32));
          ++state->count;
        }
      }
    } else if (artifact) {
      const auto n = static_cast<iree_hal_dim_t>(spec.rows);
      const auto k = static_cast<iree_hal_dim_t>(spec.columns);
      const auto groups = static_cast<iree_hal_dim_t>(
          spec.rows * (spec.columns / spec.group_size));
      const bool nf4 = spec.kind == ProgramKind::Nf4Decode ||
                       spec.kind == ProgramKind::Nf4NestedDecode;
      const bool awq = spec.kind == ProgramKind::AwqDecode;
      const std::array<iree_hal_dim_t, 2> packed{
          awq ? k : n, awq ? n / 8 : k / (nf4 ? 2 : 8)};
      HYLE_ASSIGN_OR_RETURN(
          state->views[0], allocate_view(hal, packed, nf4
                                      ? IREE_HAL_ELEMENT_TYPE_SINT_8
                                      : IREE_HAL_ELEMENT_TYPE_SINT_32));
      if (nf4) {
        const std::array<iree_hal_dim_t, 1> group_shape{groups}, book{16};
        if (spec.kind == ProgramKind::Nf4NestedDecode) {
          const std::array<iree_hal_dim_t, 1> outer{
              groups / 256 + (groups % 256 != 0)}, scale_book{256}, scalar{1};
          HYLE_ASSIGN_OR_RETURN(state->views[1],
              allocate_view(hal, group_shape, IREE_HAL_ELEMENT_TYPE_SINT_8));
          HYLE_ASSIGN_OR_RETURN(state->views[2],
              allocate_view(hal, outer, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
          HYLE_ASSIGN_OR_RETURN(state->views[3],
              allocate_view(hal, scale_book, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
          HYLE_ASSIGN_OR_RETURN(state->views[4],
              allocate_view(hal, scalar, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
          HYLE_ASSIGN_OR_RETURN(state->views[5],
              allocate_view(hal, book, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
          state->count = 6;
        } else {
          HYLE_ASSIGN_OR_RETURN(state->views[1],
              allocate_view(hal, group_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
          HYLE_ASSIGN_OR_RETURN(state->views[2],
              allocate_view(hal, book, IREE_HAL_ELEMENT_TYPE_FLOAT_32));
          state->count = 3;
        }
      } else {
        const std::array<iree_hal_dim_t, 2> scales{
            awq ? k / spec.group_size : n, awq ? n : k / spec.group_size};
        if (awq) {
          const std::array<iree_hal_dim_t, 2> zeros{k / spec.group_size, n / 8};
          HYLE_ASSIGN_OR_RETURN(state->views[1],
              allocate_view(hal, zeros, IREE_HAL_ELEMENT_TYPE_SINT_32));
        }
        HYLE_ASSIGN_OR_RETURN(state->views[awq ? 2 : 1],
            allocate_view(hal, scales, artifact_scalar(spec.scale_type)));
        state->count = awq ? 3 : 2;
      }
    } else if (precision) {
      const std::array<iree_hal_dim_t, 1> shape{
          static_cast<iree_hal_dim_t>(spec.rows * spec.columns)};
      HYLE_ASSIGN_OR_RETURN(
          state->views[0],
          allocate_view(hal, shape,
                        spec.kind == ProgramKind::Float16ToBFloat16
                            ? IREE_HAL_ELEMENT_TYPE_FLOAT_16
                            : IREE_HAL_ELEMENT_TYPE_FLOAT_32));
      state->count = 1;
    } else if (scan) {
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
    if (artifact_program(state.spec.kind) || mlx_program(state.spec.kind)) {
      const auto group = iree_vm_value_make_i32(
          static_cast<std::int32_t>(state.spec.group_size));
      HYLE_RETURN_IF_ERROR(detail::consume_status(
          iree_vm_list_push_value(state.inputs.get(), &group)));
    }
    if (iq1_program(state.spec.kind)) {
      const auto n = iree_vm_value_make_i32(static_cast<std::int32_t>(state.spec.columns));
      HYLE_RETURN_IF_ERROR(detail::consume_status(iree_vm_list_push_value(state.inputs.get(), &n)));
    }
    if (official_grouped(state.spec.kind)) {
      for (const auto scalar : {state.spec.selection_slots, state.spec.activation_slots}) {
        const auto value = iree_vm_value_make_i32(static_cast<std::int32_t>(scalar));
        HYLE_RETURN_IF_ERROR(detail::consume_status(iree_vm_list_push_value(state.inputs.get(), &value)));
      }
    }
    if (official_checked(state.spec.kind) && !official_roundtrip(state.spec.kind)) {
      const auto floor = iree_vm_value_make_f32(state.spec.absmax_floor);
      HYLE_RETURN_IF_ERROR(detail::consume_status(iree_vm_list_push_value(state.inputs.get(), &floor)));
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
