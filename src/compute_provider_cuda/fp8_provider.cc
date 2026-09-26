#include "fp8_provider.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "hyle/compute/low_precision.h"
#include "native_context.h"
#include "fp8/sglang_fp8_sm89.h"

namespace hyle::compute::cuda_native::fp8 {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = true};

struct Shape final { int m, n, k; };
struct Prepared final {
  CUcontext primary = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  Shape shape{};
};

Result<Shape> select(Operation operation, std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) noexcept {
  const auto *matmul = operation.get_if<MatMul>();
  if (!matmul || matmul->accumulation_type != DataType::kFloat32 ||
      inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0].rank() != 2 || inputs[1].rank() != 2)
    return std::unexpected(Status::Unimplemented(
        "SGLang FP8 requires rank-two block128 MatMul with F32 accumulation"));
  const auto m = inputs[0].dimensions()[0];
  const auto k = inputs[0].dimensions()[1];
  const auto n = inputs[1].dimensions()[1];
  constexpr auto mode = Float8ScaleMode::kFloat32Block128;
  const auto a = float8_activation_description(m, k, mode);
  const auto b = float8_weight_description(k, n, mode);
  const auto c = TensorDesc::dense(DataType::kFloat32, std::array{m, n});
  if (!a || !b || !c || inputs[0] != *a || inputs[1] != *b || outputs[0] != *c)
    return std::unexpected(Status::Unimplemented(
        "SGLang FP8 requires exact E4M3FN/F32 block128 payload and scale layout"));
  // Original Triton pointer offsets and grid arithmetic use signed I32.
  if (m > INT32_MAX / k || n > INT32_MAX / k || m > INT32_MAX / n)
    return std::unexpected(Status::Unimplemented(
        "SGLang FP8 native indexing exceeds I32"));
  return Shape{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
}

Status qualify(CUcontext primary) noexcept {
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return scope.status();
  CUdevice device;
  int major = 0, minor = 0, driver = 0;
  HYLE_RETURN_IF_ERROR(cuda::status(cuCtxGetDevice(&device)));
  HYLE_RETURN_IF_ERROR(cuda::status(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)));
  HYLE_RETURN_IF_ERROR(cuda::status(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)));
  HYLE_RETURN_IF_ERROR(cuda::status(cuDriverGetVersion(&driver)));
  // This is a fixed native artifact qualification domain, checked only cold.
  // A new architecture/driver domain needs evidence before extending it.
  if (major != 8 || minor != 9 || driver != 13040)
    return Status::Unimplemented(
        "SGLang FP8 AOT is qualified for SM89 / CUDA Driver 13040");
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
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (!output) return Status::InvalidArgument("SGLang FP8 invocation is null");
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
  auto a = cuda::detail::const_pointer(inputs[0].storage(), 4, 1);
  if (!a) return detail::EnqueueResult::rejected(std::move(a).error());
  auto b = cuda::detail::const_pointer(inputs[1].storage(), 4, 1);
  if (!b) return detail::EnqueueResult::rejected(std::move(b).error());
  auto c = cuda::detail::mutable_pointer(outputs[0].storage(), 4, 1);
  if (!c) return detail::EnqueueResult::rejected(std::move(c).error());
  const auto [m, n, k] = prepared.shape;
  std::array<CUdeviceptr, 7> pointers{
      reinterpret_cast<CUdeviceptr>(*a), reinterpret_cast<CUdeviceptr>(*b),
      reinterpret_cast<CUdeviceptr>(*c),
      reinterpret_cast<CUdeviceptr>(*a) + std::size_t(m) * k,
      reinterpret_cast<CUdeviceptr>(*b) + std::size_t(n) * k, 0, 0};
  // Exact original Triton 3.8 ABI, including its two zero-sized scratch slots.
  std::array integers{m, n, k, k, 1, 1, k, n, 1, k / 128, 1, 1, k / 128};
  std::array<void *, 20> arguments;
  for (std::size_t i = 0; i < 5; ++i) arguments[i] = &pointers[i];
  for (std::size_t i = 0; i < integers.size(); ++i) arguments[i + 5] = &integers[i];
  arguments[18] = &pointers[5];
  arguments[19] = &pointers[6];
  const auto grid = unsigned((m / 64 + (m % 64 != 0)) *
                             (n / 128 + (n % 128 != 0)));
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  return detail::EnqueueResult::accepted_command(cuda::status(cuLaunchKernel(
      prepared.function, grid, 1, 1, 128, 1, 1, 24576, *stream,
      arguments.data(), nullptr)));
}
const detail::ProviderPreparedVTable kVTable{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace

Status propose(void *state, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!proposal || inputs.size() != 2 || outputs.size() != 1)
    return Status::InvalidArgument("SGLang FP8 binding counts differ");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("SGLang FP8 requires an exact shape");
  HYLE_ASSIGN_OR_RETURN(const auto shape,
                       select(request.operation, request.inputs, request.outputs));
  (void)shape;
  HYLE_RETURN_IF_ERROR(qualify(static_cast<CUcontext>(state)));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = 4};
  std::ranges::fill(inputs, binding);
  std::ranges::fill(outputs, binding);
  *proposal = {.route = kRoute, .runtime = kRuntime,
               .cost = {.workspace = WorkspaceRequirement{},
                        .persistent_bytes = std::nullopt, .copy_bytes = 0,
                        .repack_bytes = 0, .synchronization_count = 0, .dispatch_count = 1},
               .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (request.route != kRoute || !request.shape.exact())
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
        "SGLang FP8 exact route mismatch"));
  auto shape = select(request.operation, request.inputs, request.outputs);
  if (!shape) return ProviderPrepareOutcome::unsupported(std::move(shape).error());
  const auto primary = static_cast<CUcontext>(state);
  auto status = qualify(primary);
  if (!status.ok())
    return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(status)
                                    : ProviderPrepareOutcome::error(status);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  auto *memory = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!memory)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("SGLang FP8 program allocation failed"));
  auto *prepared = ::new (memory) Prepared{primary, nullptr, nullptr, *shape};
  auto result = cuModuleLoadData(&prepared->module, kSglangFp8Sm89);
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&prepared->function, prepared->module,
                                 "_w8a8_block_fp8_matmul");
  // Finish lazy native code allocation before capture/replay and concurrent use.
  if (result == CUDA_SUCCESS) result = cuFuncLoad(prepared->function);
  if (result != CUDA_SUCCESS) {
    destroy(prepared, allocator);
    status = cuda::status(result);
    return result == CUDA_ERROR_OUT_OF_MEMORY
               ? ProviderPrepareOutcome::resource_declined(status)
               : ProviderPrepareOutcome::error(status);
  }
  *output = {.vtable = &kVTable, .state = prepared, .runtime = kRuntime,
             .persistent_bytes = std::nullopt, .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::fp8
