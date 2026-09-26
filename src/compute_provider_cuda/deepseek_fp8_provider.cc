#include "hyle/compute/providers/cuda.h"

#include <algorithm>
#include <array>
#include <climits>
#include <new>

#include "deepseek_fp8_sm89.h"
#include "deepseek_quantization_sm89.h"
#include "fp8_provider.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"
#include "provider_internal.h"

namespace hyle::compute::cuda_native {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = true};
struct Shape final {
  int n, k;
  bool bfloat;
  bool roundtrip = false;
  std::size_t workspace() const noexcept { return roundtrip ? std::size_t(n) * sizeof(float) : 0; }
};
struct Prepared final {
  CUcontext primary;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  Shape shape;
};

Status create_context(const Provider &, Device device, void *backend,
                      mem::AllocatorRef, void **output) noexcept {
  if (!output) return Status::InvalidArgument("DeepSeek context output is null");
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("DeepSeek decode requires CUDA");
  auto primary = cuda::detail::borrow_primary_context(backend);
  if (!primary) return Status::FailedPrecondition("DeepSeek CUDA context is invalid");
  *output = primary;
  return Status::Ok();
}
void destroy_context(void *, mem::AllocatorRef) noexcept {}
Status drain_context(void *) noexcept { return Status::Ok(); }

Result<Shape> select(const detail::ProviderOperationRequest &request) noexcept {
  if (const auto roundtrip = request.operation.custom_as<model_ops::BlockQuantizationRoundtrip>();
      roundtrip && request.shape.exact()) {
    std::array<TensorDesc, 1> inferred;
    HYLE_RETURN_IF_ERROR(roundtrip->infer_output_descriptions(request.inputs, inferred));
    if (request.outputs.size() != 1 || request.outputs[0] != inferred[0])
      return std::unexpected(Status::InvalidArgument("DeepSeek roundtrip output differs from inference"));
    const int group = roundtrip->format == model_ops::BlockQuantizationFormat::kFloat4E2M1Block32 ? 32
        : (roundtrip->format == model_ops::BlockQuantizationFormat::kFloat8E4M3Block64 ? 64 : 128);
    const auto dims = inferred[0].dimensions();
    if (dims[0] > INT32_MAX / (dims[1] / group))
      return std::unexpected(Status::Unimplemented("DeepSeek roundtrip native group capacity exceeded"));
    return Shape{static_cast<int>(dims[0] * (dims[1] / group)), group, true, true};
  }
  const auto operation = request.operation.custom_as<model_ops::Float8WeightDecode>();
  if (!operation || !request.shape.exact())
    return std::unexpected(Status::Unimplemented("DeepSeek requires exact FP8 weight decode"));
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(operation->infer_output_descriptions(request.inputs, inferred));
  if (request.outputs.size() != 1 || request.outputs[0] != inferred[0])
    return std::unexpected(Status::InvalidArgument("DeepSeek decode output differs from inference"));
  const auto n = inferred[0].dimensions()[0], k = inferred[0].dimensions()[1];
  if (n > INT32_MAX / k || k / 128 > 65535)
    return std::unexpected(Status::Unimplemented("DeepSeek decode native index/grid capacity exceeded"));
  return Shape{static_cast<int>(n), static_cast<int>(k), operation->output_type == DataType::kBFloat16};
}
Status propose(void *state, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!proposal) return Status::InvalidArgument("DeepSeek proposal is null");
  *proposal = {};
  HYLE_ASSIGN_OR_RETURN(const auto shape, select(request));
  if (inputs.size() != 1 || outputs.size() != 1)
    return Status::InvalidArgument("DeepSeek binding counts differ");
  HYLE_RETURN_IF_ERROR(fp8::qualify(state));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = shape.roundtrip ? 16u : 4u};
  inputs[0] = outputs[0] = binding;
  outputs[0].minimum_alignment = shape.roundtrip ? 16u : (shape.bfloat ? 2u : 4u);
  *proposal = {.route = 1, .runtime = kRuntime, .workspace_resource = binding,
               .cost = {.workspace = WorkspaceRequirement{shape.workspace(), shape.roundtrip ? 4u : 1u}, .persistent_bytes = std::nullopt,
                        .copy_bytes = 0, .repack_bytes = 0, .synchronization_count = 0, .dispatch_count = 1},
               .implementation = ProviderImplementationId{shape.roundtrip ? (shape.k == 64 ? 2u : (shape.k == 32 ? 3u : 4u)) : 1u}};
  return Status::Ok();
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (!prepared) return;
  {
    cuda::detail::ScopedPrimaryContext scope(prepared->primary);
    if (scope && prepared->module) (void)cuModuleUnload(prepared->module);
  }
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef, mem::BufferView workspace, void **output) noexcept {
  if (!output) return Status::InvalidArgument("DeepSeek invocation output is null");
  *output = nullptr;
  const auto &shape = static_cast<const Prepared *>(state)->shape;
  if (shape.roundtrip) {
    if (workspace.size() < shape.workspace())
      return Status::InvalidArgument("DeepSeek roundtrip workspace is too small");
    HYLE_ASSIGN_OR_RETURN(auto *pointer, cuda::detail::mutable_pointer(workspace, 4, 1));
    *output = pointer;
  }
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *workspace, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  auto stream = cuda::stream(queue);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  auto source = cuda::detail::const_pointer(inputs[0].storage(), prepared.shape.roundtrip ? 16u : 4u, 1);
  if (!source) return detail::EnqueueResult::rejected(std::move(source).error());
  auto target = cuda::detail::mutable_pointer(outputs[0].storage(), prepared.shape.roundtrip ? 16u : (prepared.shape.bfloat ? 2u : 4u), 1);
  if (!target) return detail::EnqueueResult::rejected(std::move(target).error());
  auto codes = reinterpret_cast<CUdeviceptr>(*source);
  auto scales = codes + std::size_t(prepared.shape.n) * prepared.shape.k;
  auto result = reinterpret_cast<CUdeviceptr>(*target);
  int n = prepared.shape.n, k = prepared.shape.k;
  CUdeviceptr scratch = 0;
  std::array<void *, 7> arguments{&codes, &scales, &result, &n, &k, &scratch, &scratch};
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  if (prepared.shape.roundtrip) {
    auto native_workspace = reinterpret_cast<CUdeviceptr>(workspace);
    std::array<void *, 4> args{&native_workspace, &codes, &result, &n};
    return detail::EnqueueResult::accepted_command(cuda::status(cuLaunchKernel(
        prepared.function, unsigned((std::uint64_t(n) + 31) / 32), 1, 1,
        128, 1, 1, unsigned(32 * k * 2), *stream, args.data(), nullptr)));
  }
  return detail::EnqueueResult::accepted_command(cuda::status(cuLaunchKernel(
      prepared.function, unsigned((n + 127) / 128), unsigned(k / 128), 1,
      128, 1, 1, 0, *stream, arguments.data(), nullptr)));
}
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  if (!output) return ProviderPrepareOutcome::error(Status::InvalidArgument("DeepSeek prepared output is null"));
  *output = {};
  if (request.route != 1)
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented("DeepSeek route mismatch"));
  auto shape = select({.operation = request.operation, .inputs = request.inputs,
                       .outputs = request.outputs, .shape = request.shape});
  if (!shape) return ProviderPrepareOutcome::unsupported(std::move(shape).error());
  auto status = fp8::qualify(state);
  if (!status.ok()) return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(status)
                                                  : ProviderPrepareOutcome::error(status);
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  auto *memory = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!memory) return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("DeepSeek program allocation failed"));
  auto *prepared = ::new (memory) Prepared{primary, nullptr, nullptr, *shape};
  const auto *image = shape->roundtrip ? (shape->k == 64 ? kDeepSeekQuantization8
        : (shape->k == 128 ? kDeepSeekQuantization8Block128 : kDeepSeekQuantization4))
                                     : (shape->bfloat ? kDeepSeekDecodeBf16 : kDeepSeekDecodeF32);
  const auto *symbol = shape->roundtrip ? (shape->k != 32 ? "act_quant_kernel__kernel" : "fp4_quant_kernel__kernel")
                                      : "weight_dequant_kernel";
  auto result = cuModuleLoadData(&prepared->module, image);
  if (result == CUDA_SUCCESS) result = cuModuleGetFunction(&prepared->function, prepared->module, symbol);
  if (result == CUDA_SUCCESS) result = cuFuncLoad(prepared->function);
  if (result != CUDA_SUCCESS) {
    destroy(prepared, allocator);
    return result == CUDA_ERROR_OUT_OF_MEMORY ? ProviderPrepareOutcome::resource_declined(cuda::status(result))
                                             : ProviderPrepareOutcome::error(cuda::status(result));
  }
  *output = {.vtable = &kPreparedVTable, .state = prepared,
             .workspace = {shape->workspace(), shape->roundtrip ? 4u : 1u},
             .runtime = kRuntime, .persistent_bytes = std::nullopt, .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
const detail::ProviderVTable kProviderVTable{
    .identity = {.id = kDeepSeekProviderId, .name = "deepseek-cuda", .version = "9b4e9788-b5968e91-aot"},
    .create_context = &create_context, .destroy_context = &destroy_context,
    .propose = &propose, .prepare = &prepare, .drain_context = &drain_context};
} // namespace

Provider deepseek_provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}
} // namespace hyle::compute::cuda_native
