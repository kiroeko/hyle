#include "hyle/compute/providers/onednn.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <utility>

#include "oneapi/dnnl/dnnl_ukernel.h"
#include "provider_internal.h"

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#error "HYLE's oneDNN provider requires the locked BRGEMM ukernel API"
#endif

namespace hyle::compute::onednn {
namespace {

static_assert(DNNL_VERSION_MAJOR == 3 && DNNL_VERSION_MINOR == 13 &&
                  DNNL_VERSION_PATCH == 2,
              "update the HYLE oneDNN provider identity with its source lock");

inline constexpr std::size_t kCacheLine = 64;
inline constexpr std::size_t kMinParallelTileColumns = 64;
inline constexpr std::size_t kNativeSerialRowCutoff = 16;

bool try_align_scratchpad(std::size_t size, std::size_t *aligned) noexcept {
  constexpr std::size_t kPadding = kCacheLine - 1;
  if (aligned == nullptr ||
      size > (std::numeric_limits<std::size_t>::max)() - kPadding) {
    return false;
  }
  *aligned = (size + kPadding) & ~kPadding;
  return true;
}

bool is_cache_aligned(const void *pointer) noexcept {
  return (reinterpret_cast<std::uintptr_t>(pointer) & (kCacheLine - 1)) == 0;
}

Status translate_status(dnnl_status_t status, const char *operation) noexcept {
  switch (status) {
  case dnnl_success:
    return Status::Ok();
  case dnnl_out_of_memory:
    return Status::ResourceExhausted(operation);
  case dnnl_invalid_arguments:
  case dnnl_invalid_graph:
  case dnnl_invalid_graph_op:
  case dnnl_invalid_shape:
  case dnnl_invalid_data_type:
    return Status::InvalidArgument(operation);
  case dnnl_unimplemented:
  case dnnl_last_impl_reached:
  case dnnl_not_required:
    return Status::Unimplemented(operation);
  case dnnl_runtime_error:
    return Status::Internal(operation);
  }
  return Status::Internal(operation);
}

struct OneDnnContext final {
  ParallelExecutorRef executor{};
};

struct OneDnnPlan final {
  OneDnnContext *context = nullptr;
  dnnl_brgemm_t base_kernel = nullptr;
  dnnl_brgemm_t wide_kernel = nullptr;
  std::size_t rows = 0;
  std::size_t inner = 0;
  std::size_t columns = 0;
  std::size_t task_count = 1;
  std::size_t base_columns = 0;
  std::size_t wide_task_count = 0;
  std::size_t scratch_stride = 0;
  std::size_t scratchpad_size = 0;
};

struct OneDnnInvocation final {
  mem::HostMapping scratchpad_mapping{};
  std::byte *scratchpad = nullptr;
};

void destroy_kernel(dnnl_brgemm_t &kernel) noexcept {
  if (kernel != nullptr) {
    dnnl_brgemm_destroy(kernel);
    kernel = nullptr;
  }
}

Status create_kernel(dnnl_brgemm_t *output, std::size_t rows, std::size_t inner,
                     std::size_t tile_columns, std::size_t leading_columns,
                     std::size_t *scratchpad_size) noexcept {
  if (output == nullptr || scratchpad_size == nullptr)
    return Status::InvalidArgument("oneDNN ukernel output is null");
  *output = nullptr;
  *scratchpad_size = 0;

  dnnl_brgemm_t kernel = nullptr;
  Status status = translate_status(
      dnnl_brgemm_create(&kernel, static_cast<dnnl_dim_t>(rows),
                         static_cast<dnnl_dim_t>(tile_columns),
                         static_cast<dnnl_dim_t>(inner), 1,
                         static_cast<dnnl_dim_t>(inner),
                         static_cast<dnnl_dim_t>(leading_columns),
                         static_cast<dnnl_dim_t>(leading_columns), dnnl_f32,
                         dnnl_f32, dnnl_f32),
      "oneDNN cannot create this BRGEMM ukernel");
  if (status.ok()) {
    status = translate_status(dnnl_brgemm_set_add_C(kernel, 0),
                              "oneDNN rejected overwrite output semantics");
  }
  if (status.ok()) {
    status = translate_status(dnnl_brgemm_finalize(kernel),
                              "oneDNN cannot finalize this BRGEMM ukernel");
  }
  if (status.ok()) {
    status = translate_status(dnnl_brgemm_generate(kernel),
                              "oneDNN cannot generate this BRGEMM ukernel");
  }
  if (status.ok()) {
    status = translate_status(
        dnnl_brgemm_get_scratchpad_size(kernel, scratchpad_size),
        "oneDNN did not expose its BRGEMM scratchpad requirement");
  }
  if (!status.ok()) {
    destroy_kernel(kernel);
    return status;
  }
  *output = kernel;
  return Status::Ok();
}

bool is_row_major_f32_matrix(const TensorDesc &description) noexcept {
  return description.valid() && description.data_type() == DataType::kFloat32 &&
         description.is_plain() && description.rank() == 2 &&
         description.is_dense();
}

Status create_context(const Provider &provider_value, Device device, void *,
                      mem::AllocatorRef allocator,
                      void **context_output) noexcept {
  if (context_output == nullptr)
    return Status::InvalidArgument("oneDNN context output is null");
  *context_output = nullptr;
  if (device.device_class() != DeviceClass::kCpu)
    return Status::Unimplemented("oneDNN provider requires a CPU device");

  void *storage =
      allocator.allocate(sizeof(OneDnnContext), alignof(OneDnnContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("oneDNN context allocation failed");
  const Options options =
      detail::ProviderAccess::payload<Options>(provider_value);
  *context_output = ::new (storage) OneDnnContext{.executor = options.executor};
  return Status::Ok();
}

void destroy_context(void *context_value,
                     mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<OneDnnContext *>(context_value);
  if (context == nullptr)
    return;
  context->~OneDnnContext();
  allocator.deallocate(context, sizeof(OneDnnContext), alignof(OneDnnContext));
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *plan = static_cast<OneDnnPlan *>(state);
  if (plan == nullptr)
    return;
  destroy_kernel(plan->wide_kernel);
  destroy_kernel(plan->base_kernel);
  plan->~OneDnnPlan();
  allocator.deallocate(plan, sizeof(OneDnnPlan), alignof(OneDnnPlan));
}

Status create_invocation(void *prepared_value, mem::AllocatorRef allocator,
                         mem::BufferView workspace,
                         void **invocation_output) noexcept {
  if (invocation_output == nullptr)
    return Status::InvalidArgument("oneDNN invocation output is null");
  *invocation_output = nullptr;
  auto *plan = static_cast<OneDnnPlan *>(prepared_value);
  if (plan == nullptr || plan->context == nullptr ||
      plan->base_kernel == nullptr || plan->scratchpad_size == 0) {
    return Status::Internal("oneDNN BRGEMM plan is invalid");
  }

  void *storage =
      allocator.allocate(sizeof(OneDnnInvocation), alignof(OneDnnInvocation));
  if (storage == nullptr)
    return Status::ResourceExhausted("oneDNN invocation allocation failed");
  auto *invocation = ::new (storage) OneDnnInvocation{};

  Status status = Status::Ok();
  auto mapping = workspace.map_host(mem::HostAccess::kReadWrite);
  if (!mapping) {
    status = std::move(mapping).error();
  } else {
    invocation->scratchpad_mapping = std::move(*mapping);
    auto bytes = invocation->scratchpad_mapping.writable_bytes();
    if (!bytes) {
      status = std::move(bytes).error();
    } else if (bytes->size() < plan->scratchpad_size ||
               !is_cache_aligned(bytes->data())) {
      status = Status::InvalidArgument(
          "oneDNN workspace must satisfy its size and 64-byte alignment");
    } else {
      invocation->scratchpad = bytes->data();
    }
  }

  if (!status.ok()) {
    invocation->scratchpad_mapping.close().IgnoreError();
    invocation->~OneDnnInvocation();
    allocator.deallocate(invocation, sizeof(OneDnnInvocation),
                         alignof(OneDnnInvocation));
    return status;
  }
  *invocation_output = invocation;
  return Status::Ok();
}

void destroy_invocation(void *invocation_value,
                        mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<OneDnnInvocation *>(invocation_value);
  if (invocation == nullptr)
    return;
  invocation->scratchpad_mapping.close().IgnoreError();
  invocation->~OneDnnInvocation();
  allocator.deallocate(invocation, sizeof(OneDnnInvocation),
                       alignof(OneDnnInvocation));
}

struct ExecutionEpisode final {
  OneDnnPlan *plan = nullptr;
  OneDnnInvocation *invocation = nullptr;
  const float *left = nullptr;
  const float *right = nullptr;
  float *output = nullptr;
  std::atomic<int> first_error{static_cast<int>(dnnl_success)};
};

void execute_tile(void *episode_value, std::size_t index,
                  std::size_t count) noexcept {
  auto &episode = *static_cast<ExecutionEpisode *>(episode_value);
  OneDnnPlan &plan = *episode.plan;
  if (count != plan.task_count || index >= count)
    return;

  const bool wide = index < plan.wide_task_count;
  const std::size_t first_column =
      wide ? index * (plan.base_columns + 1)
           : plan.wide_task_count * (plan.base_columns + 1) +
                 (index - plan.wide_task_count) * plan.base_columns;
  const dnnl_brgemm_t kernel = wide ? plan.wide_kernel : plan.base_kernel;
  std::byte *scratchpad =
      episode.invocation->scratchpad + index * plan.scratch_stride;
  const std::array<dnnl_dim_t, 2> offsets{0, 0};

  dnnl_status_t status = dnnl_brgemm_set_hw_context(kernel);
  if (status == dnnl_success) {
    status = dnnl_brgemm_execute(kernel, episode.left,
                                 episode.right + first_column, offsets.data(),
                                 episode.output + first_column, scratchpad);
    const dnnl_status_t release = dnnl_brgemm_release_hw_context();
    if (status == dnnl_success)
      status = release;
  }
  if (status != dnnl_success) {
    int expected = static_cast<int>(dnnl_success);
    episode.first_error.compare_exchange_strong(
        expected, static_cast<int>(status), std::memory_order_relaxed,
        std::memory_order_relaxed);
  }
}

Status execute_mapped(void *prepared_value, void *invocation_value,
                      std::span<const void *const> inputs,
                      std::span<void *const> outputs) noexcept {
  auto *plan = static_cast<OneDnnPlan *>(prepared_value);
  auto *invocation = static_cast<OneDnnInvocation *>(invocation_value);
  if (plan == nullptr || invocation == nullptr ||
      invocation->scratchpad == nullptr || inputs.size() != 2 ||
      outputs.size() != 1 || inputs[0] == nullptr || inputs[1] == nullptr ||
      outputs[0] == nullptr) {
    return Status::Internal("oneDNN BRGEMM execution state is invalid");
  }

  ExecutionEpisode episode{
      .plan = plan,
      .invocation = invocation,
      .left = static_cast<const float *>(inputs[0]),
      .right = static_cast<const float *>(inputs[1]),
      .output = static_cast<float *>(outputs[0]),
  };
  plan->context->executor.parallel_for(plan->task_count, &execute_tile,
                                       &episode);
  return translate_status(static_cast<dnnl_status_t>(episode.first_error.load(
                              std::memory_order_relaxed)),
                          "oneDNN BRGEMM execution failed");
}

void preserve_first_error(Status &terminal, Status candidate) noexcept {
  if (terminal.ok() && !candidate.ok())
    terminal = std::move(candidate);
}

Result<Completion> submit(void *prepared_value, void *invocation_value, Queue &,
                          std::span<const ConstTensorView> inputs,
                          std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 2 || outputs.size() != 1) {
    return Completion::completed(
        Status::Internal("oneDNN binding count is invalid"));
  }

  const mem::ConstBufferView left_storage = inputs[0].storage();
  const mem::ConstBufferView right_storage = inputs[1].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  const mem::MemoryProperties left_properties =
      left_storage.resource().properties();
  const mem::MemoryProperties right_properties =
      right_storage.resource().properties();
  const mem::MemoryProperties output_properties =
      output_storage.resource().properties();

  if (!left_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = left_storage.invalidate();
    if (!status.ok())
      return Completion::completed(std::move(status));
  }
  if (!right_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = right_storage.invalidate();
    if (!status.ok())
      return Completion::completed(std::move(status));
  }

  auto left_result = left_storage.map_host();
  if (!left_result)
    return Completion::completed(std::move(left_result).error());
  mem::HostMapping left = std::move(*left_result);
  auto right_result = right_storage.map_host();
  if (!right_result) {
    left.close().IgnoreError();
    return Completion::completed(std::move(right_result).error());
  }
  mem::HostMapping right = std::move(*right_result);
  auto output_result = output_storage.map_host(mem::HostAccess::kWrite);
  if (!output_result) {
    right.close().IgnoreError();
    left.close().IgnoreError();
    return Completion::completed(std::move(output_result).error());
  }
  mem::HostMapping output = std::move(*output_result);
  auto writable = output.writable_bytes();
  if (!writable) {
    Status terminal = std::move(writable).error();
    output.close().IgnoreError();
    right.close().IgnoreError();
    left.close().IgnoreError();
    return Completion::completed(std::move(terminal));
  }

  const std::array<const void *, 2> mapped_inputs{left.bytes().data(),
                                                  right.bytes().data()};
  const std::array<void *, 1> mapped_outputs{writable->data()};
  Status terminal = execute_mapped(prepared_value, invocation_value,
                                   mapped_inputs, mapped_outputs);
  if (terminal.ok() &&
      !output_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    preserve_first_error(terminal, output.flush());
  }
  preserve_first_error(terminal, output.close());
  preserve_first_error(terminal, right.close());
  preserve_first_error(terminal, left.close());
  return Completion::completed(std::move(terminal));
}

const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .submit = &submit,
};

Status prepare(void *context_value, Operation operation,
               std::span<const TensorDesc> inputs,
               std::span<const TensorDesc> outputs, mem::AllocatorRef allocator,
               detail::ProviderPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return Status::InvalidArgument("oneDNN prepared output is null");
  *prepared_output = {};
  auto *context = static_cast<OneDnnContext *>(context_value);
  if (context == nullptr)
    return Status::Internal("oneDNN provider context is invalid");
  if (operation.kind() != OperationKind::kMatMul || inputs.size() != 2 ||
      outputs.size() != 1) {
    return Status::Unimplemented(
        "oneDNN provider does not implement this operation");
  }
  if (!is_row_major_f32_matrix(inputs[0]) ||
      !is_row_major_f32_matrix(inputs[1]) ||
      !is_row_major_f32_matrix(outputs[0])) {
    return Status::Unimplemented(
        "oneDNN provider route requires dense row-major float32 matrices");
  }

  const auto left = inputs[0].dimensions();
  const auto right = inputs[1].dimensions();
  const auto output = outputs[0].dimensions();
  if (left[1] != right[0] || output[0] != left[0] || output[1] != right[1]) {
    return Status::InvalidArgument("oneDNN MatMul shapes do not agree");
  }
  if (left[0] == 0 || left[1] == 0 || right[1] == 0) {
    return Status::Unimplemented(
        "zero-size MatMul remains on the HYLE-native route");
  }

  const std::size_t rows = static_cast<std::size_t>(left[0]);
  const std::size_t inner = static_cast<std::size_t>(left[1]);
  const std::size_t columns = static_cast<std::size_t>(right[1]);
  const std::size_t max_tasks_by_width =
      (std::max)(std::size_t{1}, columns / kMinParallelTileColumns);
  const std::size_t task_count =
      (std::min)(context->executor.concurrency(), max_tasks_by_width);

  // The HYLE-native 1/4-row kernels have a structurally cheaper decode route.
  // A real external executor may still make independent N tiles worthwhile.
  if (task_count == 1 && rows < kNativeSerialRowCutoff) {
    return Status::Unimplemented(
        "small serial MatMul remains on the HYLE-native decode route");
  }

  const std::size_t base_columns = columns / task_count;
  const std::size_t wide_task_count = columns % task_count;
  std::size_t base_scratchpad = 0;
  dnnl_brgemm_t base_kernel = nullptr;
  Status status = create_kernel(&base_kernel, rows, inner, base_columns,
                                columns, &base_scratchpad);
  dnnl_brgemm_t wide_kernel = nullptr;
  std::size_t wide_scratchpad = 0;
  if (status.ok() && wide_task_count != 0) {
    status = create_kernel(&wide_kernel, rows, inner, base_columns + 1, columns,
                           &wide_scratchpad);
  }
  if (!status.ok()) {
    destroy_kernel(wide_kernel);
    destroy_kernel(base_kernel);
    return status;
  }

  const std::size_t per_task_scratchpad =
      (std::max)(base_scratchpad, wide_scratchpad);
  std::size_t scratch_stride = 0;
  if (!try_align_scratchpad((std::max)(per_task_scratchpad, std::size_t{1}),
                            &scratch_stride) ||
      task_count > (std::numeric_limits<std::size_t>::max)() / scratch_stride) {
    destroy_kernel(wide_kernel);
    destroy_kernel(base_kernel);
    return Status::OutOfRange("oneDNN MatMul workspace size overflows size_t");
  }
  const std::size_t scratchpad_size = task_count * scratch_stride;

  void *storage = allocator.allocate(sizeof(OneDnnPlan), alignof(OneDnnPlan));
  if (storage == nullptr) {
    destroy_kernel(wide_kernel);
    destroy_kernel(base_kernel);
    return Status::ResourceExhausted("oneDNN MatMul plan allocation failed");
  }
  auto *plan = ::new (storage) OneDnnPlan{
      .context = context,
      .base_kernel = base_kernel,
      .wide_kernel = wide_kernel,
      .rows = rows,
      .inner = inner,
      .columns = columns,
      .task_count = task_count,
      .base_columns = base_columns,
      .wide_task_count = wide_task_count,
      .scratch_stride = scratch_stride,
      .scratchpad_size = scratchpad_size,
  };
  prepared_output->vtable = &kPreparedVTable;
  prepared_output->state = plan;
  prepared_output->workspace = WorkspaceRequirement{
      .size = scratchpad_size,
      .alignment = kCacheLine,
  };
  return Status::Ok();
}

const detail::ProviderVTable kProviderVTable{
    .identity = ProviderIdentity{.id = kProviderId,
                                 .name = "onednn",
                                 .version = "3.13.2"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .prepare = &prepare,
};

} // namespace

Provider provider(Options options) noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, options);
}

std::string_view version() noexcept { return "3.13.2"; }

} // namespace hyle::compute::onednn
