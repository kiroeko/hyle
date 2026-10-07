#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/core/environment.h"

#if HYLE_COMPILER_MSVC
// Pinned c10 irange emits C4702 in /O2 SYSTEM headers.
#pragma warning(push)
#pragma warning(disable : 4702)
#endif

#include <array>
#include <climits>
#include <exception>
#include <limits>
#include <new>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "awq_cuda.h"
#include "bitsandbytes_cuda.h"
#include "compressed_tensors_cuda.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"
#include "provider_internal.h"

namespace hyle::compute::torch_cuda {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kProvider,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = true,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false};
struct Context final { CUcontext primary = nullptr; int device = -1; };
struct Program final {
  Context context;
  model_ops::ArtifactWeightDecode operation;
  std::array<TensorDesc, 6> inputs;
  TensorDesc output;
  std::size_t count = 0;
  at::Tensor shifts, order;
};
struct Invocation final { at::Tensor result; };

Status failure() noexcept {
  try { throw; }
  catch (const c10::OutOfMemoryError &e) {
    return Status::ResourceExhausted(e.what_without_backtrace());
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("CUDA ATen private allocation failed");
  } catch (const c10::Error &e) {
    return Status::Internal(e.what_without_backtrace());
  } catch (const std::exception &e) { return Status::Internal(e.what()); }
  catch (...) { return Status::Internal("CUDA ATen unknown native failure"); }
}
at::ScalarType scalar(DataType type) noexcept {
  switch (type) {
  case DataType::kUInt8: return at::kByte;
  case DataType::kInt32: return at::kInt;
  case DataType::kFloat16: return at::kHalf;
  case DataType::kBFloat16: return at::kBFloat16;
  default: return at::kFloat;
  }
}
Status select(const Operation &operation, std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  if (!operation.custom_as<model_ops::ArtifactWeightDecode>())
    return Status::Unimplemented("CUDA ATen carrier requires artifact weight decode");
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return Status::InvalidArgument("CUDA ATen output differs from artifact inference");
  for (const auto &desc : inputs) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, desc.required_bytes());
    if (bytes > INT64_MAX / 32)
      return Status::OutOfRange("CUDA ATen temporary tensor capacity overflows");
  }
  HYLE_ASSIGN_OR_RETURN(const auto elements, outputs[0].element_count());
  if (elements > INT64_MAX / 32)
    return Status::OutOfRange("CUDA ATen decoded temporary capacity overflows");
  return Status::Ok();
}
detail::ResourceBindingRequirement device_binding(const TensorDesc &desc) noexcept {
  return {.required_properties = mem::MemoryProperty::kDeviceAccessible,
          .minimum_alignment = desc.required_alignment()};
}
Status create_context(const Provider &, Device device, void *backend,
                      mem::AllocatorRef allocator, void **output) noexcept {
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("CUDA ATen requires the CUDA backend");
  const auto primary = cuda::detail::borrow_primary_context(backend);
  const auto *environment = cuda::detail::borrow_environment(backend);
  if (!primary || !environment)
    return Status::FailedPrecondition("CUDA ATen backend context is invalid");
  auto *storage = allocator.allocate(sizeof(Context), alignof(Context));
  if (!storage)
    return Status::ResourceExhausted("CUDA ATen context allocation failed");
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) { allocator.deallocate(storage, sizeof(Context), alignof(Context)); return scope.status(); }
  CUdevice ordinal = 0;
  const auto status = cuCtxGetDevice(&ordinal);
  if (status != CUDA_SUCCESS) {
    allocator.deallocate(storage, sizeof(Context), alignof(Context));
    return cuda::status(status);
  }
  *output = ::new (storage) Context{primary, ordinal};
  return Status::Ok();
}
void destroy_context(void *state, mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<Context *>(state);
  context->~Context(); allocator.deallocate(context, sizeof(Context), alignof(Context));
}
Status propose(void *state, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  const auto device = static_cast<const Context *>(state)->device;
  if (device < 0 || device > (std::numeric_limits<c10::DeviceIndex>::max)())
    return Status::Unimplemented("CUDA ATen device ordinal exceeds the original SDK index capacity");
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA ATen artifact decode requires exact shapes");
  HYLE_RETURN_IF_ERROR(select(request.operation, request.inputs, request.outputs));
  for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = device_binding(request.inputs[i]);
  outputs[0] = device_binding(request.outputs[0]);
  const auto format = request.operation.custom_as<model_ops::ArtifactWeightDecode>()->format;
  *proposal = {.route = static_cast<std::uint64_t>(format) + 1,
      .runtime = kRuntime, .workspace_resource = device_binding(request.outputs[0]),
      .cost = {.workspace = {}, .persistent_bytes = std::nullopt,
               .copy_bytes = std::nullopt, .repack_bytes = std::nullopt,
               .synchronization_count = std::nullopt, .dispatch_count = std::nullopt},
      .implementation = ProviderImplementationId{0x5443415254000001ULL + static_cast<std::uint64_t>(format)}};
  return Status::Ok();
}
void destroy_program(void *state, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<Program *>(state);
  cuda::detail::ScopedPrimaryContext scope(program->context.primary);
  program->~Program(); allocator.deallocate(program, sizeof(Program), alignof(Program));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef allocator, mem::BufferView,
                         void **output) noexcept {
  *output = nullptr;
  auto *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage) return Status::ResourceExhausted("CUDA ATen invocation allocation failed");
  *output = ::new (storage) Invocation;
  return Status::Ok();
}
void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(state);
  invocation->~Invocation(); allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}
detail::EnqueueResult submit(void *state, void *slot, Queue &queue,
                            std::span<const ConstTensorView> inputs,
                            std::span<const TensorView> outputs) noexcept {
  const auto &program = *static_cast<const Program *>(state);
  auto &invocation = *static_cast<Invocation *>(slot);
  auto stream = cuda::stream(queue);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  cuda::detail::ScopedPrimaryContext scope(program.context.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  std::array<const void *, 6> pointers{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(),
        program.inputs[i].required_alignment(), data_type_size(program.inputs[i].data_type()));
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    pointers[i] = *pointer;
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
      program.output.required_alignment(), data_type_size(program.output.data_type()));
  if (!output) return detail::EnqueueResult::rejected(std::move(output).error());
  // Once ATen dispatch starts, any exception may follow an accepted prefix.
  // The real Queue Completion retains all borrowed payload until terminal.
  try {
    at::NoGradGuard no_grad;
    const auto index = static_cast<c10::DeviceIndex>(program.context.device);
    const auto external = c10::cuda::getStreamFromExternal(*stream, index);
    c10::cuda::CUDAStreamGuard stream_guard(external);
    std::array<at::Tensor, 6> tensors;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto &desc = program.inputs[i];
      tensors[i] = at::from_blob(const_cast<void *>(pointers[i]),
          at::IntArrayRef(desc.dimensions().data(), desc.rank()),
          at::TensorOptions().dtype(scalar(desc.data_type())).device(at::kCUDA, program.context.device));
    }
    const auto type = scalar(program.output.data_type());
    const auto rows = program.output.dimensions()[0], columns = program.output.dimensions()[1];
    switch (program.operation.format) {
    case model_ops::WeightArtifactFormat::kNf4:
      invocation.result = borrowed::bitsandbytes_cuda::decode_nf4(tensors[0], tensors[1], tensors[2],
          program.operation.group_size, rows, columns, type); break;
    case model_ops::WeightArtifactFormat::kNf4DoubleQuant: {
      auto scales = borrowed::bitsandbytes_cuda::nested_scales(tensors[1], tensors[2], tensors[3], tensors[4]);
      invocation.result = borrowed::bitsandbytes_cuda::decode_nf4(tensors[0], scales, tensors[5],
          program.operation.group_size, rows, columns, type); break;
    }
    case model_ops::WeightArtifactFormat::kAwqGemm4:
      invocation.result = borrowed::awq_cuda::decode_gemm4(tensors[0], tensors[1], tensors[2],
          program.operation.group_size, program.shifts, program.order).to(type); break;
    case model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4:
      invocation.result = borrowed::compressed_tensors_cuda::decode_symmetric4(tensors[0], tensors[1],
          program.operation.group_size, program.shifts).to(type); break;
    }
    auto destination = at::from_blob(*output,
        at::IntArrayRef(program.output.dimensions().data(), program.output.rank()),
        at::TensorOptions().dtype(type).device(at::kCUDA, program.context.device));
    destination.copy_(invocation.result, true);
    return detail::EnqueueResult::accepted_command(Status::Ok());
  } catch (...) { return detail::EnqueueResult::accepted_command(failure()); }
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy_program, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto status = select(request.operation, request.inputs, request.outputs);
  if (!status.ok()) return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(std::move(status))
                                                   : ProviderPrepareOutcome::error(std::move(status));
  const auto &context = *static_cast<const Context *>(state);
  if (context.device < 0 || context.device > (std::numeric_limits<c10::DeviceIndex>::max)())
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented("CUDA ATen device index capacity exceeded"));
  auto *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage) return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("CUDA ATen program allocation failed"));
  auto *program = ::new (storage) Program{.context = context,
      .operation = *request.operation.custom_as<model_ops::ArtifactWeightDecode>(),
      .inputs = {}, .output = request.outputs[0], .count = request.inputs.size()};
  for (std::size_t i = 0; i < program->count; ++i) program->inputs[i] = request.inputs[i];
  cuda::detail::ScopedPrimaryContext scope(context.primary);
  if (!scope) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(scope.status()); }
  try {
    const auto index = static_cast<c10::DeviceIndex>(context.device);
    c10::cuda::CUDAGuard device_guard(index);
    at::NoGradGuard no_grad;
    // Real torch_cuda import keeps the original dispatcher registration live;
    // CPU common DLLs remain the immutable locked host dispatch runtime.
    (void)at::cuda::getCurrentCUDABlasHandle();
    const auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, context.device);
    if (program->operation.format == model_ops::WeightArtifactFormat::kAwqGemm4 ||
        program->operation.format == model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4)
      program->shifts = at::arange(0, 32, 4, options);
    if (program->operation.format == model_ops::WeightArtifactFormat::kAwqGemm4)
      program->order = at::tensor({0, 4, 1, 5, 2, 6, 3, 7}, at::kLong).to(options, false, true);
    // Publish constants only after their original library stream has finished.
    status = cuda::status(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream(index).stream()));
    if (!status.ok()) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(std::move(status)); }
  } catch (...) {
    status = failure(); destroy_program(program, allocator);
    return status.IsResourceExhausted() ? ProviderPrepareOutcome::resource_declined(std::move(status))
                                       : ProviderPrepareOutcome::error(std::move(status));
  }
  *output = {.vtable = &kPrepared, .state = program, .workspace = {}, .runtime = kRuntime,
             .persistent_bytes = std::nullopt, .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
Status drain(void *) noexcept { return Status::Ok(); }
const detail::ProviderVTable kProvider{
    .identity = {.id = kProviderId, .name = "original-artifacts-cuda",
        .version = "libtorch-2.8.0+cu129+bnb-83364904+awq-88e4c76b+ct-ac8e2ba8"},
    .create_context = &create_context, .destroy_context = &destroy_context,
    .propose = &propose, .prepare = &prepare, .drain_context = &drain};
} // namespace
Provider provider() noexcept { return detail::ProviderAccess::make(&kProvider, std::uint8_t{0}); }
} // namespace hyle::compute::torch_cuda

#if HYLE_COMPILER_MSVC
#pragma warning(pop)
#endif
