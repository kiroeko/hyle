#include "sampling_provider.h"
#include "sampling_kernel.h"

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <new>
#include <utility>

#include "hyle/compute/sampling_ops.h"
#include "native_context.h"

namespace hyle::compute::cuda_native::sampling {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = false};
struct Prepared final { CUcontext primary; Program program; };
struct Invocation final { void *workspace = nullptr; };

template<class Request>
Status describe(const Request &request,
                  Program &program) noexcept {
  const auto bits = request.operation.template custom_as<sampling_ops::RandomBits>();
  const auto categorical = request.operation.template custom_as<sampling_ops::Categorical>();
  if (!bits && !categorical)
    return Status::Unimplemented("CUDA operation is not a sampling numerical schema");
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA sampling requires fixed shapes");
  std::array<TensorDesc, 2> inferred;
  if (request.outputs.size() > inferred.size())
    return Status::InvalidArgument("CUDA sampling output count differs");
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(request.operation, request.inputs,
                         std::span(inferred).first(request.outputs.size())));
  if (!std::ranges::equal(request.outputs,
                           std::span(inferred).first(request.outputs.size())))
    return Status::InvalidArgument("CUDA sampling output descriptions differ");
  for (const auto &description : request.inputs) {
    if (*description.required_bytes() > std::size_t(PTRDIFF_MAX))
      return Status::Unimplemented("CUDA sampling input address capacity exceeded");
  }
  for (const auto &description : request.outputs) {
    if (*description.required_bytes() > std::size_t(PTRDIFF_MAX))
      return Status::Unimplemented("CUDA sampling output address capacity exceeded");
  }
  const auto rows = request.inputs[0].dimensions()[0];
  if (rows > INT32_MAX)
    return Status::Unimplemented("CUDA sampling row capacity exceeded");
  program = {};
  program.random_bits = bool(bits);
  program.log_probability = categorical && categorical->return_log_probability;
  program.rows = static_cast<std::uint32_t>(rows);
  program.width = bits ? bits->word_count
                       : static_cast<std::uint32_t>(request.inputs[0].dimensions()[1]);
  if (bits || rows == 0) return Status::Ok();
  program.type = request.inputs[0].data_type() == DataType::kFloat32 ? Type::kFloat32
               : request.inputs[0].data_type() == DataType::kFloat16 ? Type::kFloat16
                                                                  : Type::kBFloat16;
  program.chunks = (program.width + 4095U) / 4096U;
  const auto count = std::uint64_t(program.rows) * program.chunks;
  if (count > INT32_MAX)
    return Status::Unimplemented("CUDA sampling chunk grid capacity exceeded");
  const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
  program.row_max_offset = align(static_cast<std::size_t>(count) * sizeof(float));
  program.partial_offset = align(program.row_max_offset + std::size_t(rows) * sizeof(float));
  program.workspace_bytes = align(program.partial_offset + static_cast<std::size_t>(count) * 12);
  return Status::Ok();
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (!prepared) return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef allocator,
                           mem::BufferView workspace, void **output) noexcept {
  *output = nullptr;
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.program.workspace_bytes == 0) return Status::Ok();
  if (workspace.size() < prepared.program.workspace_bytes)
    return Status::InvalidArgument("CUDA sampling workspace is too small");
  HYLE_ASSIGN_OR_RETURN(auto pointer, cuda::detail::mutable_pointer(workspace, 256, 1));
  auto *memory = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!memory)
    return Status::ResourceExhausted("CUDA sampling invocation allocation failed");
  *output = ::new (memory) Invocation{pointer};
  return Status::Ok();
}
void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(state);
  if (!invocation) return;
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}
detail::EnqueueResult submit(void *state, void *slot, detail::QueueSequence &sequence,
    std::span<const ConstTensorView> inputs, std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  const auto &program = prepared.program;
  if (program.rows == 0 || program.width == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(sequence);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const void *, 2> input{};
  std::array<void *, 2> output{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(),
                        inputs[i].description().required_alignment(), 1);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    input[i] = *pointer;
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto pointer = cuda::detail::mutable_pointer(outputs[i].storage(), 4, 1);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    output[i] = *pointer;
  }
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  cudaStreamCaptureStatus capture;
  auto status = cuda::status(cudaStreamIsCapturing(*stream, &capture));
  if (!status.ok()) return detail::EnqueueResult::rejected(std::move(status));
  if (capture != cudaStreamCaptureStatusNone)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "CUDA sampling has no external capture qualification"));
  unsigned *word = nullptr;
  if (!program.random_bits) {
    auto initialized = cuda::detail::acquire_device_status(sequence, &word);
    if (!initialized.status.ok()) return initialized;
  }
  return detail::EnqueueResult::accepted_command(cuda::status(launch(program,
      program.random_bits ? nullptr : input[0],
      static_cast<const std::uint32_t *>(input[program.random_bits ? 0 : 1]),
      output[0], static_cast<float *>(output[1]),
      slot ? static_cast<Invocation *>(slot)->workspace : nullptr, word, *stream)));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace

Status propose(const detail::ProviderOperationRequest &request,
    std::span<detail::ResourceBindingRequirement> inputs,
    std::span<detail::ResourceBindingRequirement> outputs,
    detail::ProviderOperationProposal *proposal) noexcept {
  *proposal = {};
  Program program;
  HYLE_RETURN_IF_ERROR(describe(request, program));
  if (inputs.size() != request.inputs.size() || outputs.size() != request.outputs.size())
    return Status::InvalidArgument("CUDA sampling binding counts differ");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                  .minimum_alignment = request.inputs[i].required_alignment()};
  std::ranges::fill(outputs, detail::ResourceBindingRequirement{
      .required_properties = mem::MemoryProperty::kDeviceAccessible, .minimum_alignment = 4});
  *proposal = {.route = kRoute, .runtime = kRuntime,
      .workspace_resource = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                              .minimum_alignment = 256},
      .cost = {.workspace = WorkspaceRequirement{program.workspace_bytes, 256},
                .persistent_bytes = sizeof(Prepared), .copy_bytes = 0,
                .repack_bytes = 0, .synchronization_count = 0,
                .dispatch_count = program.rows == 0 || program.width == 0 ? 1U
                                   : program.random_bits ? 1U : 4U,
                .compilation_storage = {.resident_bytes = 0, .peak_bytes = 0}},
      .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
    mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  *output = {};
  Program program;
  auto status = describe(request, program);
  if (!status.ok())
    return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(std::move(status))
                                     : ProviderPrepareOutcome::error(std::move(status));
  if (request.route != kRoute)
    return ProviderPrepareOutcome::error(Status::InvalidArgument("CUDA sampling route differs"));
  auto *memory = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!memory)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("CUDA sampling prepared allocation failed"));
  auto *prepared = ::new (memory) Prepared{static_cast<CUcontext>(state), program};
  *output = {.vtable = &kPrepared, .state = prepared,
      .workspace = WorkspaceRequirement{program.workspace_bytes, 256}, .runtime = kRuntime,
      .persistent_bytes = sizeof(Prepared),
      .invocation_private_bytes = program.workspace_bytes ? sizeof(Invocation) : 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::sampling
