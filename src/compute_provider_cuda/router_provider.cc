#include "router_provider.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

#include "hyle/compute/model_ops.h"
#include "native_context.h"
#include "sglang_router_kernel.h"

namespace hyle::compute::cuda_native::routing {
namespace {
constexpr std::uint64_t kRoute = 16;
constexpr std::uint64_t kHashRoute = 17;
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = true};
struct Prepared final {
  CUcontext primary;
  native::SglangRouterProgram program;
};
Status select(Operation operation, std::span<const TensorDesc> inputs,
               std::span<const TensorDesc> outputs) noexcept {
  if (!operation.custom_as<model_ops::SqrtSoftplusRouting>())
    return Status::Unimplemented("SGLang router requires the complete model region");
  std::array<TensorDesc, 2> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (!std::ranges::equal(inferred, outputs))
    return Status::InvalidArgument("SGLang router output descriptions differ");
  if (*inputs[0].element_count() > INT32_MAX)
    return Status::Unimplemented("SGLang router native indexing exceeds I32");
  return Status::Ok();
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (!prepared) return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (!output) return Status::InvalidArgument("SGLang router invocation is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  auto stream = cuda::stream(queue);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const void *, 2> in;
  std::array<void *, 2> out;
  for (std::size_t i = 0; i < 2; ++i) {
    auto input = cuda::detail::const_pointer(inputs[i].storage(), 4, 1);
    if (!input) return detail::EnqueueResult::rejected(std::move(input).error());
    auto output = cuda::detail::mutable_pointer(outputs[i].storage(), 4, 1);
    if (!output) return detail::EnqueueResult::rejected(std::move(output).error());
    in[i] = *input;
    out[i] = *output;
  }
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::sglang_router_launch(prepared.program, in[0], in[1], out[0], out[1], *stream)));
}
const detail::ProviderPreparedVTable kVTable{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};

struct HashPrepared final {
  CUcontext primary;
  native::SglangHashRouterProgram program;
};
Status select_hash(Operation operation, std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) noexcept {
  if (!operation.custom_as<model_ops::SqrtSoftplusHashRouting>())
    return Status::Unimplemented("SGLang hash router requires the complete table region");
  std::array<TensorDesc, 2> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (!std::ranges::equal(inferred, outputs))
    return Status::InvalidArgument("SGLang hash router output descriptions differ");
  const auto count = inputs[0].element_count();
  if (!count || *count > INT32_MAX ||
      static_cast<std::uint64_t>(inputs[2].dimensions()[0]) >
          std::numeric_limits<unsigned>::max())
    return Status::Unimplemented("SGLang hash router native index capacity exceeded");
  return Status::Ok();
}
void destroy_hash(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<HashPrepared *>(state);
  if (!prepared) return;
  prepared->~HashPrepared();
  allocator.deallocate(prepared, sizeof(HashPrepared), alignof(HashPrepared));
}
detail::EnqueueResult submit_hash(void *state, void *, Queue &queue,
                                  std::span<const ConstTensorView> inputs,
                                  std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const HashPrepared *>(state);
  auto stream = cuda::stream(queue);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const void *, 3> in;
  std::array<void *, 2> out;
  for (std::size_t i = 0; i < in.size(); ++i) {
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(), i == 1 ? 8 : 4, 1);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    in[i] = *pointer;
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    auto pointer = cuda::detail::mutable_pointer(outputs[i].storage(), 4, 1);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    out[i] = *pointer;
  }
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  unsigned *status_word = nullptr;
  auto initialized = cuda::detail::acquire_device_status(queue, &status_word);
  if (!initialized.status.ok()) return initialized;
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::sglang_hash_router_launch(prepared.program, in[0], in[1], in[2],
                                        out[0], out[1], status_word, *stream)));
}
const detail::ProviderPreparedVTable kHashVTable{
    .destroy_prepared = &destroy_hash, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit_hash};

Status propose_hash(const detail::ProviderOperationRequest &request,
                     std::span<detail::ResourceBindingRequirement> inputs,
                     std::span<detail::ResourceBindingRequirement> outputs,
                     detail::ProviderOperationProposal *proposal) noexcept {
  if (!proposal || inputs.size() != 3 || outputs.size() != 2)
    return Status::InvalidArgument("SGLang hash router binding counts differ");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("SGLang hash router requires an exact shape");
  HYLE_RETURN_IF_ERROR(select_hash(request.operation, request.inputs, request.outputs));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = 4};
  std::ranges::fill(inputs, binding);
  inputs[1].minimum_alignment = 8;
  std::ranges::fill(outputs, binding);
  *proposal = {.route = kHashRoute, .runtime = kRuntime,
               .cost = {.workspace = WorkspaceRequirement{},
                        .persistent_bytes = sizeof(HashPrepared), .copy_bytes = 0,
                        .repack_bytes = 0, .synchronization_count = 0,
                        .dispatch_count = 2}, // Existing status reset plus native kernel.
               .implementation = ProviderImplementationId{kHashRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare_hash(void *state,
                                    const detail::ProviderPrepareRequest &request,
                                    mem::AllocatorRef allocator,
                                    detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto status = select_hash(request.operation, request.inputs, request.outputs);
  if (!status.ok())
    return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(status)
                                     : ProviderPrepareOutcome::error(status);
  if (request.route != kHashRoute)
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented("SGLang hash router route mismatch"));
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  native::SglangHashRouterProgram program;
  const auto configured = native::sglang_hash_router_prepare(
      static_cast<unsigned>(request.inputs[0].dimensions()[0]),
      static_cast<unsigned>(request.inputs[0].dimensions()[1]),
      static_cast<unsigned>(request.inputs[2].dimensions()[0]), &program);
  if (configured == cudaErrorNotSupported)
    return ProviderPrepareOutcome::unsupported(cuda::status(configured));
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  auto *memory = allocator.allocate(sizeof(HashPrepared), alignof(HashPrepared));
  if (!memory)
    return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("SGLang hash router program allocation failed"));
  auto *prepared = ::new (memory) HashPrepared{primary, program};
  *output = {.vtable = &kHashVTable, .state = prepared, .runtime = kRuntime,
             .persistent_bytes = sizeof(HashPrepared), .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (request.operation.custom_as<model_ops::SqrtSoftplusHashRouting>())
    return propose_hash(request, inputs, outputs, proposal);
  if (!proposal || inputs.size() != 2 || outputs.size() != 2)
    return Status::InvalidArgument("SGLang router binding counts differ");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("SGLang router requires an exact shape");
  HYLE_RETURN_IF_ERROR(select(request.operation, request.inputs, request.outputs));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = 4};
  std::ranges::fill(inputs, binding);
  std::ranges::fill(outputs, binding);
  *proposal = {.route = kRoute, .runtime = kRuntime,
               .cost = {.workspace = WorkspaceRequirement{},
                        .persistent_bytes = sizeof(Prepared), .copy_bytes = 0,
                        .repack_bytes = 0, .synchronization_count = 0, .dispatch_count = 1},
               .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  if (request.operation.custom_as<model_ops::SqrtSoftplusHashRouting>())
    return prepare_hash(state, request, allocator, output);
  *output = {};
  auto status = select(request.operation, request.inputs, request.outputs);
  if (!status.ok())
    return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(status)
                                     : ProviderPrepareOutcome::error(status);
  if (request.route != kRoute)
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented("SGLang router route mismatch"));
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  native::SglangRouterProgram program;
  const auto desc = *request.operation.custom_as<model_ops::SqrtSoftplusRouting>();
  const auto configured = native::sglang_router_prepare(
      static_cast<unsigned>(request.inputs[0].dimensions()[0]),
      static_cast<unsigned>(request.inputs[0].dimensions()[1]), desc.scaling, &program);
  if (configured == cudaErrorNotSupported)
    return ProviderPrepareOutcome::unsupported(cuda::status(configured));
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  auto *memory = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!memory)
    return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("SGLang router program allocation failed"));
  auto *prepared = ::new (memory) Prepared{primary, program};
  *output = {.vtable = &kVTable, .state = prepared, .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared), .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::routing
