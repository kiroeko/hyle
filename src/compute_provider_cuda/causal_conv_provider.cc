#include "causal_conv_provider.h"

#include <array>
#include <climits>
#include <new>
#include <utility>

#include "causal_conv_kernel.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"

namespace hyle::compute::cuda_native::causal_conv {
namespace {
struct Selection final {
  native::CausalConvLayout layout{};
  std::array<std::size_t, 3> bytes{};
};
struct Prepared final {
  CUcontext primary = nullptr;
  native::CausalConvProgram program{};
  std::array<std::size_t, 3> bytes{};
};
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true,
};

Result<Selection> select(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) noexcept {
  const auto convolution = operation.custom_as<model_ops::CausalConv1d>();
  if (!convolution)
    return std::unexpected(Status::Unimplemented(
        "CUDA causal convolution operation differs"));
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(convolution->infer_output_descriptions(inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return std::unexpected(Status::InvalidArgument(
        "CUDA causal convolution output description differs"));
  // The public operation proves positive B/W/C/K, dense plain matching
  // inputs and exact [B,W-K+1,C] output. This implementation is F32 only.
  if (inputs[0].data_type() != DataType::kFloat32)
    return std::unexpected(Status::Unimplemented(
        "CUDA causal convolution requires F32 storage"));
  Selection selected;
  const std::array descriptions{inputs[0], inputs[1], outputs[0]};
  for (std::size_t index = 0; index < descriptions.size(); ++index) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, descriptions[index].required_bytes());
    if (bytes > std::size_t(PTRDIFF_MAX))
      return std::unexpected(Status::Unimplemented(
          "CUDA causal convolution exceeds native address capacity"));
    selected.bytes[index] = bytes;
  }
  HYLE_ASSIGN_OR_RETURN(const auto elements, outputs[0].element_count());
  selected.layout = {
      .elements = static_cast<std::uint64_t>(elements),
      .window = static_cast<std::uint64_t>(inputs[0].dimensions()[1]),
      .channels = static_cast<std::uint64_t>(inputs[0].dimensions()[2]),
      .taps = static_cast<std::uint64_t>(inputs[1].dimensions()[1]),
      .tokens = static_cast<std::uint64_t>(outputs[0].dimensions()[1])};
  return selected;
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
  if (!output)
    return Status::InvalidArgument("CUDA causal convolution invocation output is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}

detail::EnqueueResult submit(void *state, void *, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  if (inputs.size() != 2 || outputs.size() != 1)
    return detail::EnqueueResult::rejected(Status::InvalidArgument(
        "CUDA causal convolution binding counts differ"));
  const auto &prepared = *static_cast<const Prepared *>(state);
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  auto samples = cuda::detail::const_pointer(inputs[0].storage(), alignof(float), prepared.bytes[0]);
  if (!samples)
    return detail::EnqueueResult::rejected(std::move(samples).error());
  auto weights = cuda::detail::const_pointer(inputs[1].storage(), alignof(float), prepared.bytes[1]);
  if (!weights)
    return detail::EnqueueResult::rejected(std::move(weights).error());
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(), alignof(float), prepared.bytes[2]);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  // The existing QueueSequence/Completion retains prepared state and admitted
  // leases through real native retirement, including a failed launch prefix.
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::causal_conv_launch(prepared.program, *samples, *weights, *output, *stream)));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};
} // namespace

bool preserves_capture_bindings(const detail::ProviderPrepared &value) noexcept {
  if (value.vtable != &kPrepared || !value.state || value.workspace.size != 0 ||
      value.invocation_private_bytes != 0) return false;
  const auto &prepared = *static_cast<const Prepared *>(value.state);
  // This exact artifact owns immutable launch metadata only. Samples, weights
  // and destination are explicit ports; there is no Invocation allocation.
  return prepared.primary && prepared.program.kernel && prepared.program.blocks &&
      prepared.program.layout.elements && prepared.program.layout.channels &&
      prepared.program.layout.taps && prepared.program.layout.tokens;
}

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!proposal)
    return Status::InvalidArgument("CUDA causal convolution proposal output is null");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA causal convolution requires exact shapes");
  if (inputs.size() != 2 || outputs.size() != 1)
    return Status::InvalidArgument("CUDA causal convolution binding counts differ");
  const auto selected = select(request.operation, request.inputs, request.outputs);
  if (!selected) return selected.error();
  const detail::ResourceBindingRequirement tensor{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = alignof(float)};
  inputs[0] = inputs[1] = outputs[0] = tensor;
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = 1},
      .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  if (!output)
    return ProviderPrepareOutcome::error(Status::InvalidArgument(
        "CUDA causal convolution prepared output is null"));
  *output = {};
  if (request.route != kRoute || !request.shape.exact())
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
        "CUDA causal convolution exact route differs"));
  auto selected = select(request.operation, request.inputs, request.outputs);
  if (!selected)
    return selected.error().IsUnimplemented()
        ? ProviderPrepareOutcome::unsupported(std::move(selected).error())
        : ProviderPrepareOutcome::error(std::move(selected).error());
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  native::CausalConvProgram program;
  const auto configured = native::causal_conv_prepare(selected->layout, &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!storage)
    return ProviderPrepareOutcome::error(Status::ResourceExhausted(
        "CUDA causal convolution program allocation failed"));
  auto *prepared = ::new (storage) Prepared{
      .primary = primary, .program = program, .bytes = selected->bytes};
  *output = {.vtable = &kPrepared,
             .state = prepared,
             .workspace = {},
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::causal_conv
