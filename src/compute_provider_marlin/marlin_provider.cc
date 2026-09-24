#include "hyle/compute/providers/marlin.h"

#include <array>
#include <limits>
#include <new>
#include <optional>
#include <utility>

#include "marlin_internal.h"
#include "provider_internal.h"

namespace hyle::compute::marlin {
namespace {

inline constexpr std::string_view kProviderVersion = "vllm-51918e252af4";
inline constexpr std::array kQuantizedMatMulFacts{
    RouteFactPredicate::number(
        operation_route_fact::kQuantizedWeightMode,
        static_cast<std::uint64_t>(QuantizedWeightMode::kSignedSymmetric4)),
    RouteFactPredicate::number(operation_route_fact::kQuantizationGroupSize,
                               128),
    RouteFactPredicate::number(operation_route_fact::kAccumulationDataType,
                               static_cast<std::uint64_t>(DataType::kFloat32))};
inline constexpr std::array kQuantizedMatMulPorts{
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 0,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 2,
                         .maximum_rank = 2},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 1,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kOpaque,
                         .storage_format = kWeightFormat,
                         .minimum_rank = 2,
                         .maximum_rank = 2},
    RouteTensorPredicate{.direction = RoutePortDirection::kInput,
                         .index = 2,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kOpaque,
                         .storage_format = kScaleFormat,
                         .minimum_rank = 2,
                         .maximum_rank = 2},
    RouteTensorPredicate{.direction = RoutePortDirection::kOutput,
                         .index = 0,
                         .data_type = DataType::kFloat16,
                         .storage_kind = RouteStorageKind::kPlain,
                         .minimum_rank = 2,
                         .maximum_rank = 2}};
inline constexpr RoutePolicyEntry kQuantizedMatMulPolicy{
    .id = {0x48594c454d41524cULL, 1, 1},
    .preference = 100,
    .provider = kProviderId,
    .implementation = ProviderImplementationId{1},
    .selector = {.route = NativeRoute::kCuda,
                 .provider = kProviderId,
                 .provider_version = kProviderVersion,
                 .operation = OperationKind::kQuantizedMatMul,
                 .operation_facts = kQuantizedMatMulFacts,
                 .ports = kQuantizedMatMulPorts}};

struct Prepared final {
  CUcontext primary = nullptr;
  native::Program program{};
};

struct Invocation final {
  void *workspace = nullptr;
  bool launch_failed = false;
};

Status create_context(const Provider &, Device device, void *backend_state,
                      mem::AllocatorRef allocator, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("Marlin context output is null");
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("Marlin requires the CUDA backend");
  const auto *environment = cuda::detail::borrow_environment(backend_state);
  if (environment == nullptr)
    return Status::FailedPrecondition("Marlin CUDA context is invalid");
  auto hardware = internal::hardware(
      device, cuda::detail::borrow_primary_context(backend_state),
      *environment);
  if (!hardware)
    return std::move(hardware).error();
  void *storage = allocator.allocate(sizeof(internal::Hardware),
                                     alignof(internal::Hardware));
  if (storage == nullptr)
    return Status::ResourceExhausted("Marlin context allocation failed");
  *output = ::new (storage) internal::Hardware(*hardware);
  return Status::Ok();
}

void destroy_context(void *state, mem::AllocatorRef allocator) noexcept {
  auto *hardware = static_cast<internal::Hardware *>(state);
  if (hardware == nullptr)
    return;
  hardware->~Hardware();
  allocator.deallocate(hardware, sizeof(internal::Hardware),
                       alignof(internal::Hardware));
}

Status select_shape(Operation operation, std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) noexcept {
  if (inputs.size() != 3 || outputs.size() != 1)
    return Status::Unimplemented(
        "Marlin implements three-input Q4 MatMul only");
  const RouteMatchInput route_input{
      .platform = kBuildRoutePlatform,
      .route = NativeRoute::kCuda,
      .provider = {kProviderId, "marlin", kProviderVersion},
      .operation = operation,
      .inputs = inputs,
      .outputs = outputs};
  if (!match_route_policy(kQuantizedMatMulPolicy, route_input).matched())
    return Status::Unimplemented(
        "Marlin requires symmetric group128 FP16 Q4 storage");
  for (const auto &input : inputs) {
    if (!input.valid() || input.rank() != 2 ||
        input.data_type() != DataType::kFloat16)
      return Status::Unimplemented(
          "Marlin requires rank-2 FP16 logical inputs");
  }
  const auto &output = outputs[0];
  if (!inputs[0].is_plain() || !inputs[0].is_dense() || !output.valid() ||
      !output.is_plain() || !output.is_dense() || output.rank() != 2 ||
      output.data_type() != DataType::kFloat16)
    return Status::Unimplemented(
        "Marlin requires dense FP16 activations and output");
  const auto m = inputs[0].dimensions()[0];
  const auto k = inputs[0].dimensions()[1];
  const auto n = inputs[1].dimensions()[0];
  HYLE_RETURN_IF_ERROR(internal::weight_shape(n, k));
  if (m < 1 || m > 64)
    return Status::Unimplemented("Marlin supports M=1..64");
  if (inputs[1].dimensions()[1] != k || inputs[2].dimensions()[0] != n ||
      inputs[2].dimensions()[1] != k / 128 || output.dimensions()[0] != m ||
      output.dimensions()[1] != n)
    return Status::InvalidArgument("Marlin Q4 matrix shapes do not agree");
  const auto count =
      static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k);
  if (count / 2 > std::numeric_limits<std::size_t>::max())
    return Status::OutOfRange("Marlin packed matrix bytes overflow");
  HYLE_ASSIGN_OR_RETURN(auto weight_bytes, inputs[1].required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto scale_bytes, inputs[2].required_bytes());
  if (inputs[1].storage_format().kind() != StorageFormatKind::kOpaque ||
      inputs[2].storage_format().kind() != StorageFormatKind::kOpaque ||
      weight_bytes != count / 2 || scale_bytes != count / 128 * 2)
    return Status::Unimplemented(
        "Marlin requires its exact packed weights and scales");
  return Status::Ok();
}

constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true,
};

Status propose(void *, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (proposal == nullptr)
    return Status::InvalidArgument("Marlin proposal output is null");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("Marlin requires an exact prepared shape");
  if (inputs.size() != request.inputs.size() ||
      outputs.size() != request.outputs.size())
    return Status::InvalidArgument("Marlin proposal binding counts differ");
  HYLE_RETURN_IF_ERROR(
      select_shape(request.operation, request.inputs, request.outputs));
  const detail::ResourceBindingRequirement tensor{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = 16,
  };
  for (auto &requirement : inputs)
    requirement = tensor;
  for (auto &requirement : outputs)
    requirement = tensor;
  *proposal = detail::ProviderOperationProposal{
      .route = 1,
      .runtime = kRuntime,
      .workspace_resource = {.required_properties =
                                 mem::MemoryProperty::kDeviceAccessible,
                             .minimum_alignment = 256},
      .cost = {.workspace = std::nullopt,
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = 1},
      .implementation = ProviderImplementationId{1},
      .policy = kQuantizedMatMulPolicy.id,
      .preference = kQuantizedMatMulPolicy.preference,
  };
  return Status::Ok();
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared == nullptr)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}

Status create_invocation(void *state, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("Marlin invocation output is null");
  *output = nullptr;
  auto &prepared = *static_cast<Prepared *>(state);
  if (workspace.size() < prepared.program.workspace_bytes)
    return Status::InvalidArgument("Marlin workspace is too small");
  auto pointer = cuda::detail::mutable_pointer(workspace, 256, 1);
  if (!pointer)
    return std::move(pointer).error();
  // Cold initialization only. A successful native kernel resets its own locks.
  auto locks = workspace.subview(prepared.program.locks_offset,
                                 prepared.program.workspace_bytes -
                                     prepared.program.locks_offset);
  if (!locks)
    return std::move(locks).error();
  HYLE_RETURN_IF_ERROR(locks->fill(std::byte{0}));
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (storage == nullptr)
    return Status::ResourceExhausted("Marlin invocation allocation failed");
  *output = ::new (storage) Invocation{.workspace = *pointer};
  return Status::Ok();
}

void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(state);
  if (invocation == nullptr)
    return;
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}

detail::EnqueueResult submit(void *state, void *lane, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  auto &prepared = *static_cast<Prepared *>(state);
  auto &invocation = *static_cast<Invocation *>(lane);
  if (invocation.launch_failed)
    return detail::EnqueueResult::rejected(
        Status::FailedPrecondition("recreate the failed Marlin invocation"));
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const void *, 3> pointers{};
  for (std::size_t index = 0; index < pointers.size(); ++index) {
    auto pointer = cuda::detail::const_pointer(inputs[index].storage(), 16, 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    pointers[index] = *pointer;
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(), 16, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  const auto result =
      native::launch(prepared.program, pointers[0], pointers[1], pointers[2],
                     *output, invocation.workspace, *stream);
  invocation.launch_failed = result != cudaSuccess;
  return detail::EnqueueResult::accepted_command(cuda::status(result));
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
};

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  if (output == nullptr)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("Marlin prepared output is null"));
  *output = {};
  if (request.route != 1 || !request.shape.exact())
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("Marlin exact prepare request is invalid"));
  Status status =
      select_shape(request.operation, request.inputs, request.outputs);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  const auto &hardware = *static_cast<internal::Hardware *>(state);
  cuda::detail::ScopedPrimaryContext scope(hardware.primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  native::Program program;
  const auto configured = native::prepare(
      static_cast<int>(request.inputs[0].dimensions()[0]),
      static_cast<int>(request.inputs[1].dimensions()[0]),
      static_cast<int>(request.inputs[0].dimensions()[1]),
      hardware.multiprocessors, hardware.shared_bytes, &program);
  if (configured != cudaSuccess) {
    status = cuda::status(configured);
    return configured == cudaErrorNotSupported
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  }
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("Marlin prepared allocation failed"));
  auto *prepared = ::new (storage) Prepared{hardware.primary, program};
  *output = detail::ProviderPrepared{
      .vtable = &kPreparedVTable,
      .state = prepared,
      .workspace = {program.workspace_bytes, 256},
      .runtime = kRuntime,
      .persistent_bytes = sizeof(Prepared),
      .invocation_private_bytes = 0,
  };
  return ProviderPrepareOutcome::prepared();
}

Status drain_context(void *) noexcept { return Status::Ok(); }
const detail::ProviderVTable kProviderVTable{
    .identity = {.id = kProviderId,
                 .name = "marlin",
                 .version = kProviderVersion},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
};

} // namespace

Provider provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}

} // namespace hyle::compute::marlin
