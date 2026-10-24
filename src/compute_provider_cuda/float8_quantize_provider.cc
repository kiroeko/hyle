#include "float8_quantize_provider.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "float8_quantize_kernel.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"

namespace hyle::compute::cuda_native::float8_quantize {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true};
struct Prepared final {
  CUcontext primary;
  native::Program program;
};

Result<native::Program> select(Operation operation,
                               std::span<const TensorDesc> inputs,
                               std::span<const TensorDesc> outputs) noexcept {
  const auto op = operation.custom_as<model_ops::Float8BlockQuantize>();
  if (!op)
    return std::unexpected(Status::Unimplemented(
        "CUDA FP8 quantization operation differs"));
  std::array<TensorDesc, 2> inferred;
  HYLE_RETURN_IF_ERROR(op->infer_output_descriptions(inputs, inferred));
  if (outputs.size() != 2 || outputs[0] != inferred[0] ||
      outputs[1] != inferred[1])
    return std::unexpected(Status::InvalidArgument(
        "CUDA FP8 quantization output descriptions differ"));
  // The kernel uses 64-bit offsets and a bounded grid. Tensor descriptions
  // validate multiplication; additionally keep every pointer offset within
  // the platform's addressable signed difference domain before preparation.
  for (const auto &desc : inputs) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, desc.required_bytes());
    if (bytes > std::size_t(PTRDIFF_MAX))
      return std::unexpected(Status::Unimplemented(
          "CUDA FP8 quantization input address capacity exceeded"));
  }
  for (const auto &desc : outputs) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, desc.required_bytes());
    if (bytes > std::size_t(PTRDIFF_MAX))
      return std::unexpected(Status::Unimplemented(
          "CUDA FP8 quantization output address capacity exceeded"));
  }
  HYLE_ASSIGN_OR_RETURN(const auto elements, inputs[0].element_count());
  return native::Program{
      .groups = elements / 128,
      .floor = op->absmax_floor,
      .bfloat = inputs[0].data_type() == DataType::kBFloat16,
      .power2 = op->scale_policy == model_ops::Float8ScalePolicy::kPowerOfTwo,
      .multiply_reciprocal = op->scale_policy ==
                            model_ops::Float8ScalePolicy::kMultiplyReciprocal};
}

void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  allocator.destroy(static_cast<Prepared *>(state));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (!output)
    return Status::InvalidArgument("CUDA FP8 quantization invocation is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}

detail::EnqueueResult submit(void *state, void *, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  auto input = cuda::detail::const_pointer(inputs[0].storage(),
                                           prepared.program.bfloat ? 2 : 4, 1);
  if (!input)
    return detail::EnqueueResult::rejected(std::move(input).error());
  auto codes = cuda::detail::mutable_pointer(outputs[0].storage(), 1, 1);
  if (!codes)
    return detail::EnqueueResult::rejected(std::move(codes).error());
  auto scales = cuda::detail::mutable_pointer(outputs[1].storage(), 4, 1);
  if (!scales)
    return detail::EnqueueResult::rejected(std::move(scales).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  unsigned *status = nullptr;
  auto initialized = cuda::detail::acquire_device_status(sequence, &status);
  if (!initialized.status.ok())
    return initialized;
  // The status reset may already be accepted. Preserve that real prefix on
  // any later CUDA launch failure and let Completion report numeric errors.
  return detail::EnqueueResult::accepted_command(cuda::status(native::launch(
      prepared.program, *input, static_cast<unsigned char *>(*codes),
      static_cast<float *>(*scales), status, *stream)));
}

const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!proposal || inputs.size() != 1 || outputs.size() != 2)
    return Status::InvalidArgument("CUDA FP8 quantization binding counts differ");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA FP8 quantization requires exact shapes");
  HYLE_ASSIGN_OR_RETURN(const auto program,
                       select(request.operation, request.inputs, request.outputs));
  inputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
               .minimum_alignment = program.bfloat ? 2u : 4u};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment = 1};
  outputs[1] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment = 4};
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = *request.outputs[0].required_bytes() +
                               *request.outputs[1].required_bytes(),
               .synchronization_count = 0,
               // A new sequence also resets its common status word.
               .dispatch_count = 2},
      .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  if (!output)
    return ProviderPrepareOutcome::error(Status::InvalidArgument(
        "CUDA FP8 quantization prepared output is null"));
  *output = {};
  if (request.route != kRoute || !request.shape.exact())
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
        "CUDA FP8 quantization exact route differs"));
  auto program = select(request.operation, request.inputs, request.outputs);
  if (!program)
    return ProviderPrepareOutcome::error(std::move(program).error());
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  const auto configured = native::prepare(*program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  auto *prepared = allocator.construct<Prepared>(primary, *program);
  if (!prepared)
    return ProviderPrepareOutcome::error(Status::ResourceExhausted(
        "CUDA FP8 quantization metadata allocation failed"));
  *output = {.vtable = &kPrepared,
             .state = prepared,
             .workspace = {},
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::float8_quantize
