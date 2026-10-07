#include "hyle/compute/providers/native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <utility>

#include "../compute/native_route.h"
#include "../compute/provider_internal.h"
#include "default_rules.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/math/scalar.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"
#include "hyle/simd/ops.h"
#include "native_plan.h"
#if defined(HYLE_COMPUTE_TORCH_CPU)
#include "torch_cpu_program.h"
#endif

namespace hyle::compute::native {
namespace {

using Parameters = detail::NativeParameters;
using internal::aligned_for_storage;
using internal::completed;
using internal::NativePlan;
using internal::preserve_first_error;

struct Configuration final {
  Options options{};
  internal::DefaultRuleScope scope = internal::DefaultRuleScope::kExpert;
  bool single_native_thread = true;
};
struct NativeContext final {
  ParallelExecutorRef executor{};
  internal::DefaultRuleScope scope = internal::DefaultRuleScope::kExpert;
  bool qualified_environment = false;
};

struct NativeInvocation final {
  mem::BufferView workspace{};
};

inline constexpr std::size_t kMinimumParallelWorkItems = 16U * 1024U;

std::size_t saturating_product(std::size_t left, std::size_t right) noexcept {
  if (left == 0 || right == 0)
    return 0;
  if (left > (std::numeric_limits<std::size_t>::max)() / right)
    return (std::numeric_limits<std::size_t>::max)();
  return left * right;
}

std::size_t recommended_parallel_tasks(ParallelExecutorRef executor,
                                       std::size_t units,
                                       std::size_t work_per_unit) noexcept {
  const std::size_t concurrency = executor.concurrency();
  if (concurrency <= 1 || units <= 1 || work_per_unit == 0)
    return 1;
  const std::size_t total = saturating_product(units, work_per_unit);
  const std::size_t work_limited =
      total / kMinimumParallelWorkItems +
      static_cast<std::size_t>(total % kMinimumParallelWorkItems != 0);
  return (std::max)(std::size_t{1},
                    (std::min)({concurrency, units, work_limited}));
}

void run_rms_norm(const float *input, const float *weight, float *output,
                  const Parameters &parameters, float epsilon) noexcept {
  const std::size_t rows = parameters[0];
  const std::size_t hidden = parameters[1];
  if (hidden == 0)
    return;
  for (std::size_t row = 0; row < rows; ++row) {
    const float *row_input = input + row * hidden;
    float *row_output = output + row * hidden;
    if (weight == nullptr)
      simd::rms_norm_f32(row_input, row_output, hidden, epsilon);
    else
      simd::rms_norm_f32(row_input, weight, row_output, hidden, epsilon);
  }
}

struct FloatRouteEpisode final {
  const NativePlan *plan = nullptr;
  const float *first = nullptr;
  const float *second = nullptr;
  float *output = nullptr;
};

void run_float_range(void *context, std::size_t, std::size_t begin,
                     std::size_t end) noexcept {
  auto &episode = *static_cast<FloatRouteEpisode *>(context);
  const NativePlan &plan = *episode.plan;
  Parameters parameters = plan.parameters;
  parameters[0] = end - begin;
  const float *first = episode.first + begin * plan.row_first_stride;
  const float *second = episode.second == nullptr || plan.row_second_stride == 0
                            ? episode.second
                            : episode.second + begin * plan.row_second_stride;
  float *output = episode.output + begin * plan.row_output_stride;
  plan.route(first, second, output, parameters, plan.scalar);
}

Status validate_logical_f32(const TensorDesc &description) noexcept {
  if (!description.valid())
    return Status::InvalidArgument("operation tensor description is invalid");
  if (description.data_type() != DataType::kFloat32) {
    return Status::Unimplemented(
        "HYLE native route currently requires float32 tensors");
  }
  return Status::Ok();
}

Status validate_dense_f32(const TensorDesc &description) noexcept {
  Status status = validate_logical_f32(description);
  if (!status.ok())
    return status;
  if (!description.is_plain() || !description.is_dense()) {
    return Status::Unimplemented(
        "HYLE native route requires dense plain tensor storage");
  }
  return Status::Ok();
}

Result<std::size_t> outer_count(const TensorDesc &description) noexcept {
  std::size_t result = 1;
  const auto dimensions = description.dimensions();
  for (std::size_t axis = 0; axis + 1 < dimensions.size(); ++axis) {
    auto value = math::try_integer_cast<std::size_t>(dimensions[axis]);
    if (!value)
      return std::unexpected(
          Status::OutOfRange("tensor dimension exceeds size_t"));
    auto next = math::try_multiply(result, *value);
    if (!next)
      return std::unexpected(Status::OutOfRange("tensor row count overflows"));
    result = *next;
  }
  return result;
}

Result<std::size_t> dimension(const TensorDesc &description,
                              std::size_t axis) noexcept {
  auto result =
      math::try_integer_cast<std::size_t>(description.dimensions()[axis]);
  if (!result)
    return std::unexpected(
        Status::OutOfRange("tensor dimension exceeds size_t"));
  return *result;
}

Status create_context(const Provider &provider_value, Device device, void *,
                      mem::AllocatorRef allocator,
                      void **context_output) noexcept {
  if (context_output == nullptr)
    return Status::InvalidArgument("native context output is null");
  *context_output = nullptr;
  if (device.device_class() != DeviceClass::kCpu)
    return Status::Unimplemented("HYLE native provider requires a CPU device");
  void *storage =
      allocator.allocate(sizeof(NativeContext), alignof(NativeContext));
  if (storage == nullptr)
    return Status::ResourceExhausted(
        "native provider context allocation failed");
  const Configuration config =
      detail::ProviderAccess::payload<Configuration>(provider_value);
  *context_output = ::new (storage) NativeContext{
      .executor = config.options.executor,
      .scope = config.scope,
      .qualified_environment = config.single_native_thread &&
                               config.options.executor.concurrency() <= 1 &&
                               internal::qualified_default_environment()};
  return Status::Ok();
}

void destroy_context(void *context_value,
                     mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<NativeContext *>(context_value);
  if (context == nullptr)
    return;
  context->~NativeContext();
  allocator.deallocate(context, sizeof(NativeContext), alignof(NativeContext));
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *plan = static_cast<NativePlan *>(state);
  if (plan == nullptr)
    return;
  plan->~NativePlan();
  allocator.deallocate(plan, sizeof(NativePlan), alignof(NativePlan));
}

Status create_invocation(void *prepared_value, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("native invocation output is null");
  *output = nullptr;
  const auto *plan = static_cast<const NativePlan *>(prepared_value);
  if (plan == nullptr)
    return Status::Internal("native prepared plan is null");
  if (plan->workspace_submit_route == nullptr)
    return Status::Ok();
  void *storage =
      allocator.allocate(sizeof(NativeInvocation), alignof(NativeInvocation));
  if (storage == nullptr)
    return Status::ResourceExhausted(
        "native invocation workspace state allocation failed");
  *output = ::new (storage) NativeInvocation{.workspace = workspace};
  return Status::Ok();
}

void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<NativeInvocation *>(value);
  if (invocation == nullptr)
    return;
  invocation->~NativeInvocation();
  allocator.deallocate(invocation, sizeof(NativeInvocation),
                       alignof(NativeInvocation));
}

detail::EnqueueResult
submit_float(const NativePlan &plan, std::span<const ConstTensorView> inputs,
             std::span<const TensorView> outputs) noexcept {
  if (plan.route == nullptr || inputs.empty() || inputs.size() > 2 ||
      outputs.size() != 1) {
    return completed(Status::Internal("native float plan is invalid"));
  }

  auto output_elements = outputs[0].description().element_count();
  if (!output_elements)
    return completed(std::move(output_elements).error());
  if (*output_elements == 0)
    return completed(Status::Ok());

  const mem::ConstBufferView first_storage = inputs[0].storage();
  const mem::BufferView output_storage = outputs[0].storage();
  const bool in_place =
      first_storage.resource() == output_storage.resource() &&
      first_storage.native_handle() == output_storage.native_handle() &&
      first_storage.offset() == output_storage.offset() &&
      first_storage.size() == output_storage.size();
  mem::ConstBufferView second_storage;
  bool shared_inputs = false;
  if (inputs.size() == 2) {
    second_storage = inputs[1].storage();
    shared_inputs =
        first_storage.resource() == second_storage.resource() &&
        first_storage.native_handle() == second_storage.native_handle() &&
        first_storage.offset() == second_storage.offset() &&
        first_storage.size() == second_storage.size();
  }

  const mem::MemoryProperties first_properties =
      first_storage.resource().properties();
  const mem::MemoryProperties output_properties =
      output_storage.resource().properties();
  mem::MemoryProperties second_properties{};
  if (inputs.size() == 2)
    second_properties = second_storage.resource().properties();

  if (!first_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = first_storage.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }
  if (inputs.size() == 2 && !shared_inputs &&
      !second_properties.has(mem::MemoryProperty::kHostDeviceCoherent)) {
    Status status = second_storage.invalidate();
    if (!status.ok())
      return completed(std::move(status));
  }

  mem::HostMapping first_mapping;
  mem::HostMapping second_mapping;
  mem::HostMapping output_mapping;
  const float *first_data = nullptr;
  const float *second_data = nullptr;
  float *output_data = nullptr;

  if (in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kReadWrite);
    if (!mapping)
      return completed(std::move(mapping).error());
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable)
      return completed(std::move(writable).error());
    output_data = reinterpret_cast<float *>(writable->data());
    first_data = output_data;
  } else {
    auto mapping = first_storage.map_host();
    if (!mapping)
      return completed(std::move(mapping).error());
    first_mapping = std::move(*mapping);
    first_data = reinterpret_cast<const float *>(first_mapping.bytes().data());
  }

  if (inputs.size() == 2) {
    if (shared_inputs && !in_place) {
      second_data = first_data;
    } else {
      auto mapping = second_storage.map_host();
      if (!mapping) {
        first_mapping.close().IgnoreError();
        output_mapping.close().IgnoreError();
        return completed(std::move(mapping).error());
      }
      second_mapping = std::move(*mapping);
      second_data =
          reinterpret_cast<const float *>(second_mapping.bytes().data());
    }
  }

  if (!in_place) {
    auto mapping = output_storage.map_host(mem::HostAccess::kWrite);
    if (!mapping) {
      second_mapping.close().IgnoreError();
      first_mapping.close().IgnoreError();
      return completed(std::move(mapping).error());
    }
    output_mapping = std::move(*mapping);
    auto writable = output_mapping.writable_bytes();
    if (!writable) {
      output_mapping.close().IgnoreError();
      second_mapping.close().IgnoreError();
      first_mapping.close().IgnoreError();
      return completed(std::move(writable).error());
    }
    output_data = reinterpret_cast<float *>(writable->data());
  }

  const bool first_aligned = aligned_for_storage(
      first_data, inputs[0].description().required_alignment());
  const bool second_aligned =
      inputs.size() != 2 ||
      aligned_for_storage(second_data,
                          inputs[1].description().required_alignment());
  const bool output_aligned = aligned_for_storage(
      output_data, outputs[0].description().required_alignment());
  if (!first_aligned || !second_aligned || !output_aligned) {
    output_mapping.close().IgnoreError();
    second_mapping.close().IgnoreError();
    first_mapping.close().IgnoreError();
    return completed(Status::FailedPrecondition(
        "CPU operation tensor mapping does not satisfy its format alignment"));
  }

  FloatRouteEpisode episode{
      .plan = &plan,
      .first = first_data,
      .second = second_data,
      .output = output_data,
  };
  parallel_for_ranges(plan, plan.parameters[0], &run_float_range, &episode);
  Status terminal = Status::Ok();
  if (!output_properties.has(mem::MemoryProperty::kHostDeviceCoherent))
    preserve_first_error(terminal, output_mapping.flush());
  preserve_first_error(terminal, output_mapping.close());
  preserve_first_error(terminal, second_mapping.close());
  preserve_first_error(terminal, first_mapping.close());
  return completed(std::move(terminal));
}

detail::EnqueueResult submit(void *prepared_value, void *invocation_value,
                             Queue &, std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  auto *plan = static_cast<NativePlan *>(prepared_value);
  if (plan == nullptr || (plan->submit_route == nullptr &&
                          plan->workspace_submit_route == nullptr)) {
    return completed(Status::Internal("native execution state is invalid"));
  }
  if (plan->workspace_submit_route != nullptr) {
    auto *invocation = static_cast<NativeInvocation *>(invocation_value);
    if (invocation == nullptr)
      return completed(Status::Internal("native workspace state is invalid"));
    return plan->workspace_submit_route(*plan, inputs, outputs,
                                        invocation->workspace);
  }
  return plan->submit_route(*plan, inputs, outputs);
}

AliasSupport alias_support(void *prepared_value, std::size_t output_index,
                           std::size_t input_index) noexcept {
  const auto *plan = static_cast<const NativePlan *>(prepared_value);
  if (plan == nullptr || output_index != 0 || input_index >= 8 ||
      (plan->alias_input_mask &
       (std::uint8_t{1} << static_cast<std::uint8_t>(input_index))) == 0) {
    return {};
  }
  return AliasSupport{
      .supported = true,
      .byte_offset = input_index == 0 ? plan->alias_input0_offset : 0,
  };
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
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

Status build_plan(Operation operation, std::span<const TensorDesc> inputs,
                  std::span<const TensorDesc> outputs, NativePlan &plan,
                  ParallelExecutorRef executor = {},
                  bool domain_copy = false) noexcept {
  if (outputs.size() != operation_traits(operation).outputs)
    return Status::Internal("native route output arity is invalid");

  plan = {};
  switch (operation.kind()) {
  case OperationKind::kView:
  case OperationKind::kCopy:
  case OperationKind::kFill: {
    Status status =
        internal::prepare_transform(operation, inputs, outputs, plan,
                                     domain_copy);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kCast: {
    Status status = internal::prepare_cast(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kCustom: {
    Status status =
        internal::prepare_ds4_quantization(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kUnary:
  case OperationKind::kBinary: {
    Status status =
        internal::prepare_elementwise(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kReduce: {
    Status status =
        internal::prepare_reduction(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kRmsNorm: {
    if (operation.get_if<RmsNorm>()->accumulation_type != DataType::kFloat32)
      return Status::Unimplemented(
          "native RMSNorm supports float32 accumulation");
    if (inputs[0].data_type() == DataType::kFloat16 ||
        inputs[0].data_type() == DataType::kBFloat16) {
      Status status =
          internal::prepare_normalization(operation, inputs, outputs, plan);
      if (!status.ok())
        return status;
      break;
    }
    Status status = validate_dense_f32(inputs[0]);
    if (!status.ok())
      return status;
    if (inputs.size() == 2) {
      status = validate_dense_f32(inputs[1]);
      if (!status.ok())
        return status;
    }
    status = validate_dense_f32(outputs[0]);
    if (!status.ok())
      return status;
    auto rows = outer_count(inputs[0]);
    auto hidden = dimension(inputs[0], inputs[0].rank() - 1);
    if (!rows)
      return std::move(rows).error();
    if (!hidden)
      return std::move(hidden).error();
    plan.parameters = {*rows, *hidden, 0};
    plan.scalar = operation.get_if<RmsNorm>()->epsilon;
    plan.row_first_stride = *hidden;
    plan.row_output_stride = *hidden;
    plan.alias_input_mask = 1;
    plan.route = &run_rms_norm;
    plan.submit_route = &submit_float;
    break;
  }
  case OperationKind::kMatMul:
    return Status::Unimplemented(
        "native MatMul retired; use a mature CPU provider");
  case OperationKind::kGatedActivation: {
    HYLE_RETURN_IF_ERROR(
        internal::prepare_half_gated(operation, inputs, outputs, plan));
    break;
  }
  case OperationKind::kGather: {
    Status status =
        internal::prepare_indexing(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kSoftmax: {
    Status status =
        internal::prepare_transformer(operation, inputs, outputs, plan);
    if (!status.ok())
      return status;
    break;
  }
  case OperationKind::kScaledDotProductAttention:
    return Status::Unimplemented(
        "native SDPA retired; use a mature CPU provider");
  default:
    return Status::Unimplemented(
        "operation has no current HYLE native executable route");
  }

  plan.executor = executor;
  std::size_t parallel_units = plan.parameters[0];
  std::size_t work_per_unit = 0;
  switch (operation.kind()) {
  case OperationKind::kRmsNorm:
  case OperationKind::kGatedActivation:
    work_per_unit = plan.parameters[1];
    break;
  default:
    break;
  }
  plan.parallel_tasks =
      recommended_parallel_tasks(executor, parallel_units, work_per_unit);
  return Status::Ok();
}

detail::ResourceBindingRequirement
host_requirement(const TensorDesc &description) noexcept {
  return detail::ResourceBindingRequirement{
      .required_properties = mem::MemoryProperty::kHostAccessible,
      .minimum_alignment = description.required_alignment(),
  };
}

Status
validate_bounded_rms(const detail::ProviderOperationRequest &request) noexcept {
  for (const auto &map : request.shape.inputs)
    if (map.layout != detail::ProviderTensorLayout::kDerived)
      return Status::Unimplemented(
          "native bounded RMSNorm requires derived dense input layouts");
  for (const auto &map : request.shape.outputs)
    if (map.layout != detail::ProviderTensorLayout::kDerived)
      return Status::Unimplemented(
          "native bounded RMSNorm requires derived dense output layouts");
  if (request.operation.kind() != OperationKind::kRmsNorm ||
      request.inputs.empty() || request.inputs.size() > 2 ||
      request.outputs.size() != 1)
    return Status::Unimplemented(
        "native bounded route currently requires RMSNorm");
  const auto &input = request.inputs[0];
  const auto &output = request.outputs[0];
  if (input.rank() == 0 || input.rank() != output.rank())
    return Status::Unimplemented("native bounded RMSNorm rank differs");
  const auto fixed = [&](std::uint32_t value) noexcept {
    return value == detail::kFixedShapeAxis ||
           request.shape.dimensions[value].bounds.exact();
  };
  if (!fixed(request.shape.inputs[0].values[input.rank() - 1]))
    return Status::Unimplemented(
        "native bounded RMSNorm requires fixed hidden width");
  // Equal envelopes of independent symbols are NOT a whole-domain proof.
  // Every output extent must be the same expression or the same fixed value.
  for (std::size_t axis = 0; axis < input.rank(); ++axis) {
    const auto left = request.shape.inputs[0].values[axis];
    const auto right = request.shape.outputs[0].values[axis];
    if (input.dimensions()[axis] != output.dimensions()[axis] ||
        (left != right && !(fixed(left) && fixed(right))))
      return Status::Unimplemented(
          "native bounded RMSNorm needs identical input/output axes");
  }
  if (request.inputs.size() == 2) {
    for (std::size_t axis = 0; axis < request.inputs[1].rank(); ++axis) {
      if (!fixed(request.shape.inputs[1].values[axis]))
        return Status::Unimplemented(
            "native bounded RMSNorm requires fixed scale shape");
    }
  }
  // RMSNorm's schema is shape-preserving, with fixed dtype, epsilon, hidden
  // width and optional scale. The axis proof above makes these conditions
  // invariant over the domain; this is not sampling an envelope as evidence.
  HYLE_ASSIGN_OR_RETURN(const auto inferred,
                        infer_single_output(request.operation, request.inputs));
  if (inferred != output)
    return Status::InvalidArgument(
        "native bounded RMSNorm output does not match its schema");
  return Status::Ok();
}

Status instantiate_rms(void *root, std::span<const TensorDesc> inputs,
                       std::span<const TensorDesc> outputs,
                       mem::AllocatorRef allocator,
                       detail::ProviderPrepared *instance) noexcept {
  if (root == nullptr || instance == nullptr || inputs.empty() ||
      outputs.size() != 1)
    return Status::InvalidArgument(
        "native RMSNorm instance request is invalid");
  *instance = {};
  NativePlan concrete = *static_cast<const NativePlan *>(root);
  if (inputs[0].rank() == 0 ||
      inputs[0].dimensions().back() !=
          static_cast<std::int64_t>(concrete.parameters[1]))
    return Status::Internal(
        "native RMSNorm instance changed compiled hidden width");
  HYLE_ASSIGN_OR_RETURN(const auto rows, outer_count(inputs[0]));
  concrete.parameters[0] = rows;
  concrete.parallel_tasks = recommended_parallel_tasks(concrete.executor, rows,
                                                       concrete.parameters[1]);
  // Rebind scalar metadata only. Kernel, precision, alias and resource route
  // were compiled once; no build_plan(), proposal, prepare or SIMD dispatch.
  void *storage = allocator.allocate(sizeof(NativePlan), alignof(NativePlan));
  if (storage == nullptr)
    return Status::ResourceExhausted(
        "native RMSNorm instance allocation failed");
  *instance = {.vtable = &kPreparedVTable,
               .state = ::new (storage) NativePlan(concrete),
               .workspace = concrete.workspace,
               .runtime = kRuntimeProperties,
               .persistent_bytes = sizeof(NativePlan),
               .invocation_private_bytes = 0};
  return Status::Ok();
}

Status validate_bounded_copy(
    const detail::ProviderOperationRequest &request) noexcept {
  if (request.inputs.size() != 1 || request.outputs.size() != 1 ||
      !request.inputs[0].is_plain() || !request.outputs[0].is_plain() ||
      !request.outputs[0].is_dense() ||
      request.shape.outputs[0].layout !=
          detail::ProviderTensorLayout::kDerived ||
      request.inputs[0].data_type() != request.outputs[0].data_type() ||
      request.inputs[0].rank() != request.outputs[0].rank())
    return Status::Unimplemented(
        "native bounded Copy requires plain input and canonical dense output");
  const auto fixed = [&](std::uint32_t value) noexcept {
    return value == detail::kFixedShapeAxis ||
           request.shape.dimensions[value].bounds.exact();
  };
  for (std::size_t axis = 0; axis < request.inputs[0].rank(); ++axis) {
    const auto left = request.shape.inputs[0].values[axis];
    const auto right = request.shape.outputs[0].values[axis];
    if (request.inputs[0].dimensions()[axis] !=
            request.outputs[0].dimensions()[axis] ||
        (left != right && !(fixed(left) && fixed(right))))
      return Status::Unimplemented(
          "native bounded Copy needs identical input/output axes");
  }
  // Copy's input can have any checked non-negative strides, including zero.
  // Its output schema is dense, shape/dtype preserving and never aliases.
  return Status::Ok();
}

Status
validate_bounded(const detail::ProviderOperationRequest &request) noexcept {
  return request.operation.kind() == OperationKind::kCopy
             ? validate_bounded_copy(request)
             : validate_bounded_rms(request);
}

Status instantiate_copy(void *root, std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs,
                        mem::AllocatorRef allocator,
                        detail::ProviderPrepared *instance) noexcept {
  if (root == nullptr || instance == nullptr || inputs.size() != 1 ||
      outputs.size() != 1)
    return Status::InvalidArgument("native Copy instance request is invalid");
  *instance = {};
  void *storage = allocator.allocate(sizeof(NativePlan), alignof(NativePlan));
  if (storage == nullptr)
    return Status::ResourceExhausted("native Copy instance allocation failed");
  const auto &plan = *static_cast<const NativePlan *>(root);
  *instance = {.vtable = &kPreparedVTable,
               .state = ::new (storage) NativePlan(plan),
               .workspace = plan.workspace,
               .runtime = kRuntimeProperties,
               .persistent_bytes = sizeof(NativePlan),
               .invocation_private_bytes = 0};
  return Status::Ok();
}

Status
propose(void *context_value, const detail::ProviderOperationRequest &request,
        std::span<detail::ResourceBindingRequirement> input_requirements,
        std::span<detail::ResourceBindingRequirement> output_requirements,
        detail::ProviderOperationProposal *proposal) noexcept {
  if (proposal == nullptr)
    return Status::InvalidArgument("native proposal output is null");
  *proposal = {};
  if (!request.shape.exact()) {
    HYLE_RETURN_IF_ERROR(validate_bounded(request));
  }
  const Operation operation = request.operation;
  const auto inputs = request.inputs;
  const auto outputs = request.outputs;
  if (input_requirements.size() != inputs.size() ||
      output_requirements.size() != outputs.size()) {
    return Status::InvalidArgument(
        "native proposal requirement counts do not match bindings");
  }

  auto *context = static_cast<NativeContext *>(context_value);
  if (context == nullptr)
    return Status::Internal("native provider context is null");
#if defined(HYLE_COMPUTE_TORCH_CPU)
  if (internal::is_torch_cpu_operation(operation, inputs))
    return internal::propose_torch_cpu(request, input_requirements,
                                       output_requirements, proposal);
#endif
  NativePlan plan{};
  Status status =
      build_plan(operation, inputs, outputs, plan, context->executor,
                   !request.shape.exact());
  if (!status.ok())
    return status;
  for (std::size_t index = 0; index < inputs.size(); ++index)
    input_requirements[index] = host_requirement(inputs[index]);
  for (std::size_t index = 0; index < outputs.size(); ++index)
    output_requirements[index] = host_requirement(outputs[index]);
  *proposal = detail::ProviderOperationProposal{
      .route = static_cast<std::uint64_t>(operation.kind()),
      .runtime = kRuntimeProperties,
      .workspace_resource =
          detail::ResourceBindingRequirement{
              .required_properties = mem::MemoryProperty::kHostAccessible,
          },
      .cost =
          ProviderRouteCostFacts{
              .workspace = plan.workspace,
              .persistent_bytes = sizeof(NativePlan),
              .copy_bytes = 0,
              .repack_bytes = 0,
              .synchronization_count = 0,
              .dispatch_count = 1,
              .compilation_storage = {.resident_bytes = 0, .peak_bytes = 0},
          },
      .shape_coverage = request.shape.exact()
                            ? detail::ProviderShapeCoverage::kExactPoint
                            : detail::ProviderShapeCoverage::kEntireDomain,
      .implementation =
          ProviderImplementationId{
              static_cast<std::uint64_t>(operation.kind()) + 1 +
              (request.shape.exact() ? 0 : (std::uint64_t{1} << 32))},
  };
  if (request.shape.exact() && context->qualified_environment &&
      context->scope != internal::DefaultRuleScope::kExpert) {
    const auto policy =
        internal::qualified_default_policy(operation, inputs, outputs);
    if (policy.valid()) {
      proposal->policy = policy;
      proposal->preference =
          context->scope == internal::DefaultRuleScope::kDirect ? 50 : 175;
    }
  }
  return Status::Ok();
}

ProviderPrepareOutcome
prepare(void *context_value, const detail::ProviderPrepareRequest &request,
        mem::AllocatorRef allocator,
        detail::ProviderPrepared *prepared_output) noexcept {
  if (prepared_output == nullptr)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("native prepared output is null"));
  *prepared_output = {};
  if (request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != request.outputs.size() ||
      request.route != static_cast<std::uint64_t>(request.operation.kind())) {
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("native exact prepare request is invalid"));
  }
  auto *context = static_cast<NativeContext *>(context_value);
  if (context == nullptr)
    return ProviderPrepareOutcome::error(
        Status::Internal("native provider context is null"));
#if defined(HYLE_COMPUTE_TORCH_CPU)
  if (internal::is_torch_cpu_operation(request.operation, request.inputs))
    return internal::prepare_torch_cpu(request, allocator, prepared_output);
#endif
  if (!request.shape.exact()) {
    const auto status = validate_bounded(
        {request.operation, request.inputs, request.outputs, request.shape});
    if (!status.ok())
      return status.IsUnimplemented()
                 ? ProviderPrepareOutcome::unsupported(status)
                 : ProviderPrepareOutcome::error(status);
  }
  NativePlan plan{};
  Status status = build_plan(request.operation, request.inputs, request.outputs,
                             plan, context->executor, !request.shape.exact());
  if (!status.ok()) {
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  }

  void *storage = allocator.allocate(sizeof(NativePlan), alignof(NativePlan));
  if (storage == nullptr) {
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("native plan allocation failed"));
  }
  auto *prepared = ::new (storage) NativePlan(plan);
  prepared_output->vtable = &kPreparedVTable;
  prepared_output->state = prepared;
  prepared_output->workspace = plan.workspace;
  prepared_output->runtime = kRuntimeProperties;
  prepared_output->persistent_bytes = sizeof(NativePlan);
  prepared_output->invocation_private_bytes = 0;
  prepared_output->instantiate =
      request.shape.exact()                              ? nullptr
      : request.operation.kind() == OperationKind::kCopy ? &instantiate_copy
                                                         : &instantiate_rms;
  return ProviderPrepareOutcome::prepared();
}

Status drain_context(void *) noexcept { return Status::Ok(); }

const detail::ProviderVTable kProviderVTable{
    .identity =
        ProviderIdentity{.id = kProviderId,
                         .name = "hyle-native",
#if defined(HYLE_COMPUTE_TORCH_CPU)
                         .version =
                             "2+torchao-a701b6a60587+libtorch-2.8.0-cpu"},
#else
                         .version = "2"},
#endif
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
};

} // namespace

Provider provider(Options options) noexcept {
  return detail::ProviderAccess::make(&kProviderVTable,
                                      Configuration{.options = options});
}

namespace internal {
Provider default_provider(Options options, DefaultRuleScope scope,
                          bool single_native_thread) noexcept {
  return detail::ProviderAccess::make(
      &kProviderVTable, Configuration{options, scope, single_native_thread});
}
} // namespace internal
} // namespace hyle::compute::native
