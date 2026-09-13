#include "hyle/compute/providers/onednn.h"

#include <array>
#include <cstddef>
#include <new>
#include <optional>
#include <span>
#include <utility>

#include "brgemm_internal.h"
#include "provider_internal.h"

namespace hyle::compute::onednn {
namespace {

inline constexpr std::size_t kNativeSerialRowCutoff = 16;

struct OneDnnContext final {
  ParallelExecutorRef executor{};
};

struct OneDnnPrepared final {
  OneDnnContext *context = nullptr;
  internal::BrgemmPlan *brgemm = nullptr;
};

struct OneDnnInvocation final {
  mem::HostMapping scratchpad_mapping{};
  std::byte *scratchpad = nullptr;
};

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
  auto *prepared = static_cast<OneDnnPrepared *>(state);
  if (prepared == nullptr)
    return;
  internal::destroy_brgemm_plan(prepared->brgemm, allocator);
  prepared->~OneDnnPrepared();
  allocator.deallocate(prepared, sizeof(OneDnnPrepared),
                       alignof(OneDnnPrepared));
}

Status create_invocation(void *prepared_value, mem::AllocatorRef allocator,
                         mem::BufferView workspace,
                         void **invocation_output) noexcept {
  if (invocation_output == nullptr)
    return Status::InvalidArgument("oneDNN invocation output is null");
  *invocation_output = nullptr;
  auto *prepared = static_cast<OneDnnPrepared *>(prepared_value);
  if (prepared == nullptr || prepared->context == nullptr ||
      prepared->brgemm == nullptr) {
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
    } else {
      status = internal::validate_brgemm_scratchpad(
          prepared->brgemm, bytes->data(), bytes->size());
      if (status.ok())
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

Status execute_mapped(void *prepared_value, void *invocation_value,
                      std::span<const void *const> inputs,
                      std::span<void *const> outputs) noexcept {
  auto *prepared = static_cast<OneDnnPrepared *>(prepared_value);
  auto *invocation = static_cast<OneDnnInvocation *>(invocation_value);
  if (prepared == nullptr || prepared->context == nullptr ||
      prepared->brgemm == nullptr || invocation == nullptr ||
      invocation->scratchpad == nullptr || inputs.size() != 2 ||
      outputs.size() != 1 || inputs[0] == nullptr || inputs[1] == nullptr ||
      outputs[0] == nullptr) {
    return Status::Internal("oneDNN BRGEMM execution state is invalid");
  }
  return internal::execute_brgemm(prepared->brgemm, prepared->context->executor,
                                  static_cast<const float *>(inputs[0]),
                                  static_cast<const float *>(inputs[1]),
                                  static_cast<float *>(outputs[0]),
                                  invocation->scratchpad);
}

void preserve_first_error(Status &terminal, Status candidate) noexcept {
  if (terminal.ok() && !candidate.ok())
    terminal = std::move(candidate);
}

detail::EnqueueResult accepted(Status terminal) noexcept {
  return detail::EnqueueResult::accepted_command(std::move(terminal));
}

detail::EnqueueResult submit(void *prepared_value, void *invocation_value,
                             Queue &,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 2 || outputs.size() != 1) {
    return accepted(
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
      return accepted(std::move(status));
  }
  if (!right_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = right_storage.invalidate();
    if (!status.ok())
      return accepted(std::move(status));
  }

  auto left_result = left_storage.map_host();
  if (!left_result)
    return accepted(std::move(left_result).error());
  mem::HostMapping left = std::move(*left_result);
  auto right_result = right_storage.map_host();
  if (!right_result) {
    left.close().IgnoreError();
    return accepted(std::move(right_result).error());
  }
  mem::HostMapping right = std::move(*right_result);
  auto output_result = output_storage.map_host(mem::HostAccess::kWrite);
  if (!output_result) {
    right.close().IgnoreError();
    left.close().IgnoreError();
    return accepted(std::move(output_result).error());
  }
  mem::HostMapping output = std::move(*output_result);
  auto writable = output.writable_bytes();
  if (!writable) {
    Status terminal = std::move(writable).error();
    output.close().IgnoreError();
    right.close().IgnoreError();
    left.close().IgnoreError();
    return accepted(std::move(terminal));
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
  return accepted(std::move(terminal));
}

AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
    .delegated_submit = nullptr,
};

constexpr ProviderRuntimeProperties kRuntimeProperties{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

struct OneDnnRoute final {
  internal::BrgemmShape shape{};
  std::size_t task_count = 0;
};

Status select_route(void *context_value, Operation operation,
                    std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    OneDnnRoute &route) noexcept {
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

  route.shape = internal::BrgemmShape{
      .rows = static_cast<std::size_t>(left[0]),
      .inner = static_cast<std::size_t>(left[1]),
      .columns = static_cast<std::size_t>(right[1]),
  };
  route.task_count = internal::recommended_brgemm_task_count(
      route.shape.columns, context->executor.concurrency());

  // The HYLE-native 1/4-row kernels have a structurally cheaper decode route.
  // A real external executor may still make independent N tiles worthwhile.
  if (route.task_count == 1 && route.shape.rows < kNativeSerialRowCutoff) {
    return Status::Unimplemented(
        "small serial MatMul remains on the HYLE-native decode route");
  }
  return Status::Ok();
}

detail::ResourceBindingRequirement
host_requirement(const TensorDesc &description) noexcept {
  return detail::ResourceBindingRequirement{
      .required_properties = mem::MemoryProperty::kHostAccessible,
      .minimum_alignment = description.required_alignment(),
  };
}

Status propose(
    void *context_value, Operation operation,
    std::span<const TensorDesc> inputs,
    std::span<const TensorDesc> outputs,
    std::span<detail::ResourceBindingRequirement> input_requirements,
    std::span<detail::ResourceBindingRequirement> output_requirements,
    detail::ProviderOperationProposal *proposal) noexcept {
  if (proposal == nullptr)
    return Status::InvalidArgument("oneDNN proposal output is null");
  *proposal = {};
  if (input_requirements.size() != inputs.size() ||
      output_requirements.size() != outputs.size()) {
    return Status::InvalidArgument(
        "oneDNN proposal requirement counts do not match bindings");
  }
  OneDnnRoute route{};
  Status status =
      select_route(context_value, operation, inputs, outputs, route);
  if (!status.ok())
    return status;
  for (std::size_t index = 0; index < inputs.size(); ++index)
    input_requirements[index] = host_requirement(inputs[index]);
  for (std::size_t index = 0; index < outputs.size(); ++index)
    output_requirements[index] = host_requirement(outputs[index]);
  *proposal = detail::ProviderOperationProposal{
      .route = 1,
      .runtime = kRuntimeProperties,
      .workspace_resource = detail::ResourceBindingRequirement{
          .required_properties = mem::MemoryProperty::kHostAccessible,
          .minimum_alignment = internal::kBrgemmWorkspaceAlignment,
      },
  };
  return Status::Ok();
}

Status prepare(void *context_value,
               const detail::ProviderPrepareRequest &request,
               mem::AllocatorRef allocator,
               detail::ProviderPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return Status::InvalidArgument("oneDNN prepared output is null");
  *prepared_output = {};
  if (request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != request.outputs.size() ||
      request.route != 1) {
    return Status::InvalidArgument("oneDNN exact prepare request is invalid");
  }
  OneDnnRoute route{};
  Status status = select_route(context_value, request.operation,
                               request.inputs, request.outputs, route);
  if (!status.ok())
    return status;
  auto *context = static_cast<OneDnnContext *>(context_value);

  internal::BrgemmPlan *brgemm = nullptr;
  status = internal::create_brgemm_plan(route.shape, route.task_count,
                                        allocator, &brgemm);
  if (!status.ok())
    return status;

  void *storage =
      allocator.allocate(sizeof(OneDnnPrepared), alignof(OneDnnPrepared));
  if (storage == nullptr) {
    internal::destroy_brgemm_plan(brgemm, allocator);
    return Status::ResourceExhausted(
        "oneDNN prepared wrapper allocation failed");
  }
  auto *prepared = ::new (storage) OneDnnPrepared{
      .context = context,
      .brgemm = brgemm,
  };
  prepared_output->vtable = &kPreparedVTable;
  prepared_output->state = prepared;
  prepared_output->workspace = WorkspaceRequirement{
      .size = internal::brgemm_scratchpad_size(brgemm),
      .alignment = internal::kBrgemmWorkspaceAlignment,
  };
  prepared_output->runtime = kRuntimeProperties;
  // The wrapper allocation is known, but oneDNN owns opaque generated-kernel
  // storage behind BrgemmPlan.  Reporting only sizeof(OneDnnPrepared) would
  // turn a partial count into a false exact planning fact.
  prepared_output->persistent_bytes = std::nullopt;
  return Status::Ok();
}

Status drain_context(void *) noexcept { return Status::Ok(); }

const detail::ProviderVTable kProviderVTable{
    .identity = ProviderIdentity{.id = kProviderId,
                                 .name = "onednn",
                                 .version = "3.13.2"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
};

} // namespace

Provider provider(Options options) noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, options);
}

std::string_view version() noexcept { return "3.13.2"; }

} // namespace hyle::compute::onednn
