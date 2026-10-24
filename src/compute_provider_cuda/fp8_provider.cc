#include "fp8_provider.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"
#include "fp8_shared_linear.h"
#include "float8_quantize_kernel.h"
#include "fp8/sglang_fp8_sm89.h"
#include "fp8/sglang_fp8_dynamic_sm89.h"
#include "fp8/sglang_fp8_quantize_cuda_sm89.h"

namespace hyle::compute::cuda_native::fp8 {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = true};

struct Shape final {
  int m, n, k;
  bool dynamic = false, bfloat = false, power2 = false, multiply_reciprocal = false;
  bool native_f32_power2 = false;
  float floor = 1.0e-10F;
  std::size_t workspace_bytes = 0;
  int quant_groups = 1;
  int second_n = 0;
};
struct Prepared final {
  CUcontext primary = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  Shape shape{};
  CUmodule quant_module = nullptr;
  CUfunction quant_function = nullptr;
};
struct Invocation final { void *workspace; };

float8_quantize::native::Program native_quantization(const Shape &shape) noexcept {
  return {.groups = static_cast<std::uint64_t>(shape.m) *
                        static_cast<std::uint64_t>(shape.k / 128),
          .floor = shape.floor, .bfloat = false, .power2 = true};
}

Result<Shape> select_one(Operation operation, std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) noexcept {
  const auto *matmul = operation.get_if<MatMul>();
  const auto linear = operation.custom_as<model_ops::Float8BlockLinear>();
  if ((!matmul && !linear) ||
      (matmul && matmul->accumulation_type != DataType::kFloat32) ||
      inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0].rank() != 2 || inputs[1].rank() != 2)
    return std::unexpected(Status::Unimplemented(
        "SGLang FP8 requires rank-two block128 MatMul with F32 accumulation"));
  const auto m = inputs[0].dimensions()[0];
  const auto k = inputs[0].dimensions()[1];
  const auto n = inputs[1].dimensions()[1];
  constexpr auto mode = Float8ScaleMode::kFloat32Block128;
  const auto encoded = float8_activation_description(m, k, mode);
  const auto a = linear ? TensorDesc::dense(inputs[0].data_type(), std::array{m, k})
                        : encoded;
  const auto b = float8_weight_description(k, n, mode);
  const auto c = TensorDesc::dense(linear ? inputs[0].data_type() : DataType::kFloat32,
                                  std::array{m, n});
  if (!encoded || !a || !b || !c || inputs[0] != *a || inputs[1] != *b || outputs[0] != *c)
    return std::unexpected(Status::Unimplemented(
        "SGLang FP8 requires exact E4M3FN/F32 block128 payload and scale layout"));
  // Original Triton pointer offsets and grid arithmetic use signed I32.
  if (m > INT32_MAX / k || n > INT32_MAX / k || m > INT32_MAX / n)
    return std::unexpected(Status::Unimplemented(
        "SGLang FP8 native indexing exceeds I32"));
  Shape shape{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
  if (linear) {
    std::array<TensorDesc, 1> inferred;
    HYLE_RETURN_IF_ERROR(linear->infer_output_descriptions(inputs, inferred));
    shape.dynamic = true;
    shape.bfloat = inputs[0].data_type() == DataType::kBFloat16;
    shape.power2 = linear->scale_policy == model_ops::Float8ScalePolicy::kPowerOfTwo;
    shape.multiply_reciprocal = linear->scale_policy == model_ops::Float8ScalePolicy::kMultiplyReciprocal;
    shape.floor = linear->absmax_floor;
    shape.workspace_bytes = *encoded->required_bytes();
    // Original CUDA wrapper chooses the largest whole number of groups from
    // 16/8/4/2/1. Bind it cold; the quantizer has no partial-block mask.
    if (!shape.power2) {
      const auto groups = m * (k / 128);
      shape.quant_groups = 16;
      while (groups % shape.quant_groups != 0) shape.quant_groups /= 2;
    }
    // The original pow2 Triton quantizer is BF16-only. F32 uses the existing
    // native F32 quantizer and its row-major scales, never that BF16 ABI.
    shape.native_f32_power2 = shape.power2 && !shape.bfloat;
  }
  return shape;
}

Result<Shape> select(Operation operation, std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) noexcept {
  const auto pair = operation.custom_as<internal::Float8BlockLinearPair>();
  if (!pair) return select_one(operation, inputs, outputs);
  if (inputs.size() != 3 || outputs.size() != 2)
    return std::unexpected(Status::InvalidArgument("SGLang FP8 pair port counts differ"));
  const auto linear = Operation::custom(model_ops::Float8BlockLinear{
      pair->scale_policy, pair->absmax_floor});
  const std::array first_inputs{inputs[0], inputs[1]};
  const std::array second_inputs{inputs[0], inputs[2]};
  HYLE_ASSIGN_OR_RETURN(auto shape, select_one(linear, first_inputs, outputs.first(1)));
  HYLE_ASSIGN_OR_RETURN(const auto second,
                       select_one(linear, second_inputs, outputs.subspan(1, 1)));
  // Both selections consume the same A and quantization descriptor. N is the
  // only independent GEMM dimension; the original quantizer never uses it.
  shape.second_n = second.n;
  return shape;
}

Status qualify_device(CUcontext primary) noexcept {
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return scope.status();
  CUdevice device;
  int major = 0, minor = 0;
  HYLE_RETURN_IF_ERROR(cuda::status(cuCtxGetDevice(&device)));
  HYLE_RETURN_IF_ERROR(cuda::status(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)));
  HYLE_RETURN_IF_ERROR(cuda::status(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)));
  // These embedded cubins target SM89. The native module/function loader
  // checks driver compatibility; a newer compatible driver is not a decline.
  if (major != 8 || minor != 9)
    return Status::Unimplemented(
        "CUDA block128 AOT requires SM89");
  return Status::Ok();
}

void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (!prepared) return;
  {
    cuda::detail::ScopedPrimaryContext scope(prepared->primary);
    if (scope && prepared->module) (void)cuModuleUnload(prepared->module);
    if (scope && prepared->quant_module) (void)cuModuleUnload(prepared->quant_module);
  }
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef allocator, mem::BufferView workspace,
                         void **output) noexcept {
  if (!output) return Status::InvalidArgument("SGLang FP8 invocation is null");
  *output = nullptr;
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.shape.dynamic) {
    if (workspace.size() < prepared.shape.workspace_bytes)
      return Status::InvalidArgument("SGLang FP8 quantization workspace is too small");
    HYLE_ASSIGN_OR_RETURN(auto *pointer, cuda::detail::mutable_pointer(workspace, 4, 1));
    auto *memory = allocator.allocate(sizeof(Invocation), alignof(Invocation));
    if (!memory) return Status::ResourceExhausted("SGLang FP8 invocation allocation failed");
    *output = ::new (memory) Invocation{pointer};
  }
  return Status::Ok();
}
void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  if (!state) return;
  auto *invocation = static_cast<Invocation *>(state);
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}

CUresult launch_matmul(const Prepared &prepared, const void *activation,
                       const void *weight, void *output, int n,
                       CUstream stream) noexcept {
  const auto m = prepared.shape.m, k = prepared.shape.k;
  std::array<CUdeviceptr, 7> pointers{
      reinterpret_cast<CUdeviceptr>(activation), reinterpret_cast<CUdeviceptr>(weight),
      reinterpret_cast<CUdeviceptr>(output),
      reinterpret_cast<CUdeviceptr>(activation) + std::size_t(m) * k,
      reinterpret_cast<CUdeviceptr>(weight) + std::size_t(n) * k, 0, 0};
  // Exact original Triton 3.8 ABI, including its two zero-sized scratch slots.
  const bool column_major_scales = prepared.shape.power2 && !prepared.shape.native_f32_power2;
  std::array integers{m, n, k, k, 1, 1, k, n, 1,
                     column_major_scales ? 1 : k / 128,
                     column_major_scales ? m : 1, 1, k / 128};
  std::array<void *, 20> arguments;
  for (std::size_t i = 0; i < 5; ++i) arguments[i] = &pointers[i];
  for (std::size_t i = 0; i < integers.size(); ++i) arguments[i + 5] = &integers[i];
  arguments[18] = &pointers[5];
  arguments[19] = &pointers[6];
  const auto grid = unsigned((m / 64 + (m % 64 != 0)) *
                             (n / 128 + (n % 128 != 0)));
  const auto shared_bytes = prepared.shape.bfloat
      ? kSglangFp8MatmulBf16SharedBytes : kSglangFp8Sm89SharedBytes;
  return cuLaunchKernel(prepared.function, grid, 1, 1, 128, 1, 1, shared_bytes,
                        stream, arguments.data(), nullptr);
}

CUresult launch_quantizer(const Prepared &prepared, const void *source,
                          const void *activation, CUstream stream) noexcept {
  // Keep the original Triton pow2 entry strictly BF16. The separately selected
  // native F32 path never calls this ABI.
  if (prepared.shape.power2 && !prepared.shape.bfloat)
    return CUDA_ERROR_INVALID_VALUE;
  const auto m = prepared.shape.m, k = prepared.shape.k;
  auto input = reinterpret_cast<CUdeviceptr>(source);
  auto encoded = reinterpret_cast<CUdeviceptr>(activation);
  auto scales = encoded + std::size_t(m) * k;
  std::array<CUdeviceptr, 2> scratch{};
  int group = 128, columns = k, scale_stride = m;
  int num_groups = m * (k / 128), groups_per_block = prepared.shape.quant_groups;
  int unused = 0;
  float floor = prepared.shape.floor, minimum = -448, maximum = 448;
  std::array<void *, 11> quant_args{&input, &encoded, &scales, &group,
      &num_groups, &groups_per_block, &floor, &minimum, &maximum, &unused, &unused};
  if (prepared.shape.power2) {
    // The unchanged Triton pow2 ABI ends with two null scratch pointers.
    quant_args[4] = &columns;
    quant_args[5] = &scale_stride;
    quant_args[9] = &scratch[0];
    quant_args[10] = &scratch[1];
  }
  return cuLaunchKernel(prepared.quant_function,
      unsigned(num_groups / groups_per_block), 1, 1,
      prepared.shape.power2 ? 32U : unsigned(groups_per_block * 16), 1, 1,
      0, stream, quant_args.data(), nullptr);
}

detail::EnqueueResult submit(void *state, void *lane, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  auto stream = cuda::stream(sequence);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  const auto alignment = prepared.shape.bfloat ? 2u : 4u;
  const auto input_alignment = prepared.shape.dynamic && !prepared.shape.power2 ? 16u : alignment;
  auto a = cuda::detail::const_pointer(inputs[0].storage(), input_alignment, 1);
  if (!a) return detail::EnqueueResult::rejected(std::move(a).error());
  const std::size_t branches = prepared.shape.second_n ? 2 : 1;
  std::array<const void *, 2> weights{};
  std::array<void *, 2> destinations{};
  // Validate every real port before the first launch. A failed second port
  // must not turn already accepted quantization/GEMM work into a rejection.
  for (std::size_t branch = 0; branch < branches; ++branch) {
    auto weight = cuda::detail::const_pointer(inputs[branch + 1].storage(), 4, 1);
    if (!weight) return detail::EnqueueResult::rejected(std::move(weight).error());
    auto output = cuda::detail::mutable_pointer(outputs[branch].storage(), alignment, 1);
    if (!output) return detail::EnqueueResult::rejected(std::move(output).error());
    weights[branch] = *weight;
    destinations[branch] = *output;
  }
  const auto activation = prepared.shape.dynamic
      ? static_cast<Invocation *>(lane)->workspace : *a;
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  if (prepared.shape.native_f32_power2) {
    unsigned *status = nullptr;
    auto initialized = cuda::detail::acquire_device_status(sequence, &status);
    if (!initialized.status.ok()) return initialized;
    auto *encoded = static_cast<unsigned char *>(static_cast<Invocation *>(lane)->workspace);
    auto *scales = reinterpret_cast<float *>(encoded +
        static_cast<std::size_t>(prepared.shape.m) * prepared.shape.k);
    const auto result = float8_quantize::native::launch(native_quantization(prepared.shape),
        *a, encoded, scales, status, *stream);
    // The shared status reset already owns a real accepted prefix. Numeric
    // failure is reported by that original Completion, including partial IO.
    if (result != cudaSuccess)
      return detail::EnqueueResult::accepted_command(cuda::status(result));
  } else if (prepared.shape.dynamic) {
    const auto result = launch_quantizer(prepared, *a, activation, *stream);
    if (result != CUDA_SUCCESS)
      return detail::EnqueueResult::accepted_command(cuda::status(result));
  }
  for (std::size_t branch = 0; branch < branches; ++branch) {
    const int n = branch == 0 ? prepared.shape.n : prepared.shape.second_n;
    const auto result = launch_matmul(prepared, activation, weights[branch],
                                      destinations[branch], n, *stream);
    if (result != CUDA_SUCCESS)
      return detail::EnqueueResult::accepted_command(cuda::status(result));
  }
  return detail::EnqueueResult::accepted_command();
}
const detail::ProviderPreparedVTable kVTable{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace

Status qualify(void *state) noexcept {
  return qualify_device(static_cast<CUcontext>(state));
}

Status propose(void *state, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  const bool pair = request.operation.custom_as<internal::Float8BlockLinearPair>().has_value();
  if (!proposal || inputs.size() != (pair ? 3u : 2u) || outputs.size() != (pair ? 2u : 1u))
    return Status::InvalidArgument("SGLang FP8 binding counts differ");
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("SGLang FP8 requires an exact shape");
  HYLE_ASSIGN_OR_RETURN(const auto shape,
                       select(request.operation, request.inputs, request.outputs));
  HYLE_RETURN_IF_ERROR(qualify(static_cast<CUcontext>(state)));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = 4};
  std::ranges::fill(inputs, binding);
  std::ranges::fill(outputs, binding);
  if (shape.bfloat) {
    inputs[0].minimum_alignment = 2;
    for (auto &output : outputs) output.minimum_alignment = 2;
  }
  if (shape.dynamic && !shape.power2) inputs[0].minimum_alignment = 16;
  const auto route = pair ? kPairRoute : shape.dynamic ? kLinearRoute : kRoute;
  *proposal = {.route = route, .runtime = kRuntime,
               .workspace_resource = binding,
               .cost = {.workspace = WorkspaceRequirement{shape.workspace_bytes,
                                                          shape.dynamic ? 4u : 1u},
                        .persistent_bytes = std::nullopt, .copy_bytes = 0,
                        .repack_bytes = 0, .synchronization_count = 0,
                        .dispatch_count = (pair ? 3u : shape.dynamic ? 2u : 1u) +
                            (shape.native_f32_power2 ? 1u : 0u)},
               .implementation = ProviderImplementationId{route}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if ((request.route != kRoute && request.route != kLinearRoute && request.route != kPairRoute) ||
      !request.shape.exact())
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented(
        "SGLang FP8 exact route mismatch"));
  auto shape = select(request.operation, request.inputs, request.outputs);
  if (!shape) return ProviderPrepareOutcome::unsupported(std::move(shape).error());
  if (request.route != (shape->second_n ? kPairRoute : shape->dynamic ? kLinearRoute : kRoute))
    return ProviderPrepareOutcome::error(Status::InvalidArgument("SGLang FP8 route/operation mismatch"));
  const auto primary = static_cast<CUcontext>(state);
  auto status = qualify(primary);
  if (!status.ok())
    return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(status)
                                    : ProviderPrepareOutcome::error(status);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  auto *memory = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!memory)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("SGLang FP8 program allocation failed"));
  auto *prepared = ::new (memory) Prepared{primary, nullptr, nullptr, *shape};
  auto result = cuModuleLoadData(&prepared->module,
      shape->bfloat ? kSglangFp8MatmulBf16 : kSglangFp8Sm89);
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&prepared->function, prepared->module,
                                 "_w8a8_block_fp8_matmul");
  // Finish lazy native code allocation before capture/replay and concurrent use.
  if (result == CUDA_SUCCESS) result = cuFuncLoad(prepared->function);
  if (result == CUDA_SUCCESS && shape->dynamic && !shape->native_f32_power2) {
    const auto *image = shape->power2 ? kSglangFp8QuantizeBf16Pow2 : kSglangFp8QuantizeCuda;
    result = cuModuleLoadData(&prepared->quant_module, image);
    if (result == CUDA_SUCCESS)
      result = cuModuleGetFunction(&prepared->quant_function, prepared->quant_module,
          shape->power2 ? "_per_token_group_quant_8bit_colmajor"
          : shape->multiply_reciprocal
              ? (shape->bfloat ? kSglangFp8QuantizeMultiplyReciprocalBf16Name
                               : kSglangFp8QuantizeMultiplyReciprocalF32Name)
              : (shape->bfloat ? kSglangFp8QuantizeBf16Name : kSglangFp8QuantizeF32Name));
    if (result == CUDA_SUCCESS) result = cuFuncLoad(prepared->quant_function);
  }
  if (result != CUDA_SUCCESS) {
    destroy(prepared, allocator);
    status = cuda::status(result);
    return result == CUDA_ERROR_OUT_OF_MEMORY
               ? ProviderPrepareOutcome::resource_declined(status)
               : ProviderPrepareOutcome::error(status);
  }
  if (shape->native_f32_power2) {
    // Finish native kernel loading before commands/capture, as with the
    // existing standalone Float8BlockQuantize public operation.
    status = cuda::status(float8_quantize::native::prepare(native_quantization(*shape)));
    if (!status.ok()) {
      destroy(prepared, allocator);
      return status.IsResourceExhausted() ? ProviderPrepareOutcome::resource_declined(status)
                                          : ProviderPrepareOutcome::error(status);
    }
  }
  *output = {.vtable = &kVTable, .state = prepared,
             .workspace = {shape->workspace_bytes, shape->dynamic ? 4u : 1u},
             .runtime = kRuntime, .persistent_bytes = std::nullopt,
             // Pair state uses only the supplied metadata allocator and caller
             // workspace; no additional untracked per-invocation storage.
             .invocation_private_bytes = shape->second_n ? 0u : shape->dynamic ? sizeof(Invocation) : 0u};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::fp8
