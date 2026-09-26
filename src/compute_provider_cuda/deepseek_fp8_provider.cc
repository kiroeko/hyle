#include "hyle/compute/providers/cuda.h"

#include <algorithm>
#include <array>
#include <climits>
#include <new>

#include "deepseek_fp8_sm89.h"
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
struct Shape final { int n, k; bool bfloat; };
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
      .required_properties = mem::MemoryProperty::kDeviceAccessible, .minimum_alignment = 4};
  inputs[0] = outputs[0] = binding;
  outputs[0].minimum_alignment = shape.bfloat ? 2u : 4u;
  *proposal = {.route = 1, .runtime = kRuntime, .workspace_resource = binding,
               .cost = {.workspace = WorkspaceRequirement{0, 1}, .persistent_bytes = std::nullopt,
                        .copy_bytes = 0, .repack_bytes = 0, .synchronization_count = 0, .dispatch_count = 1},
               .implementation = ProviderImplementationId{1}};
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
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView, void **output) noexcept {
  if (!output) return Status::InvalidArgument("DeepSeek invocation output is null");
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
  auto source = cuda::detail::const_pointer(inputs[0].storage(), 4, 1);
  if (!source) return detail::EnqueueResult::rejected(std::move(source).error());
  auto target = cuda::detail::mutable_pointer(outputs[0].storage(), prepared.shape.bfloat ? 2u : 4u, 1);
  if (!target) return detail::EnqueueResult::rejected(std::move(target).error());
  auto codes = reinterpret_cast<CUdeviceptr>(*source);
  auto scales = codes + std::size_t(prepared.shape.n) * prepared.shape.k;
  auto result = reinterpret_cast<CUdeviceptr>(*target);
  int n = prepared.shape.n, k = prepared.shape.k;
  CUdeviceptr scratch = 0;
  std::array<void *, 7> arguments{&codes, &scales, &result, &n, &k, &scratch, &scratch};
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
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
  auto result = cuModuleLoadData(&prepared->module, shape->bfloat ? kDeepSeekDecodeBf16 : kDeepSeekDecodeF32);
  if (result == CUDA_SUCCESS) result = cuModuleGetFunction(&prepared->function, prepared->module, "weight_dequant_kernel");
  if (result == CUDA_SUCCESS) result = cuFuncLoad(prepared->function);
  if (result != CUDA_SUCCESS) {
    destroy(prepared, allocator);
    return result == CUDA_ERROR_OUT_OF_MEMORY ? ProviderPrepareOutcome::resource_declined(cuda::status(result))
                                             : ProviderPrepareOutcome::error(cuda::status(result));
  }
  *output = {.vtable = &kPreparedVTable, .state = prepared, .workspace = {0, 1},
             .runtime = kRuntime, .persistent_bytes = std::nullopt, .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
const detail::ProviderVTable kProviderVTable{
    .identity = {.id = kDeepSeekProviderId, .name = "deepseek-cuda", .version = "9b4e9788-block128-aot"},
    .create_context = &create_context, .destroy_context = &destroy_context,
    .propose = &propose, .prepare = &prepare, .drain_context = &drain_context};
} // namespace

Provider deepseek_provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}
} // namespace hyle::compute::cuda_native
