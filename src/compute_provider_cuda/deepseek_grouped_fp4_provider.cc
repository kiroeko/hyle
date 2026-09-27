#include "deepseek_grouped_fp4_provider.h"

#include <array>
#include <climits>
#include <new>

#include "deepseek_quantization_sm89.h"
#include "fp8_provider.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"

namespace hyle::compute::cuda_native::deepseek_fp4 {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = true};
struct Shape final {
  int e, n, k, slots, activation_slots, total_slots, rows;
  const unsigned char *image;
  std::size_t scale_offset() const noexcept { return std::size_t(rows) * k; }
  std::size_t workspace() const noexcept { return scale_offset() + std::size_t(rows) * (k / 128) * 4; }
};
struct Prepared final {
  CUcontext primary;
  Shape shape;
  CUmodule quant_module = nullptr, gemm_module = nullptr;
  CUfunction quant = nullptr, gemm = nullptr;
};
Result<Shape> select(const detail::ProviderOperationRequest &request) noexcept {
  const auto op = request.operation.custom_as<model_ops::GroupedFloat4Linear>();
  if (!op || !request.shape.exact())
    return std::unexpected(Status::Unimplemented("DeepSeek FP4 requires exact grouped linear"));
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(op->infer_output_descriptions(request.inputs, inferred));
  if (request.outputs.size() != 1 || request.outputs[0] != inferred[0])
    return std::unexpected(Status::InvalidArgument("DeepSeek FP4 output differs from inference"));
  const auto w = request.inputs[0].dimensions(), a = request.inputs[1].dimensions();
  const auto ids = request.inputs[2].dimensions();
  const unsigned char *image = nullptr;
  if (w[1] == 3072 && w[2] == 7168) image = kDeepSeekFp4_3072_7168;
  else if (w[1] == 7168 && w[2] == 3072) image = kDeepSeekFp4_7168_3072;
  else if (w[1] == 2048 && w[2] == 4096) image = kDeepSeekFp4_2048_4096;
  else if (w[1] == 4096 && w[2] == 2048) image = kDeepSeekFp4_4096_2048;
  if (!image || w[0] > INT32_MAX || ids[1] > 65535 || a[0] > 65535 / ids[1] ||
      a[0] * a[1] > INT32_MAX / w[2])
    return std::unexpected(Status::Unimplemented("DeepSeek FP4 native shape or grid capacity is unqualified"));
  return Shape{static_cast<int>(w[0]), static_cast<int>(w[1]), static_cast<int>(w[2]),
      static_cast<int>(ids[1]), static_cast<int>(a[1]), static_cast<int>(a[0] * ids[1]),
      static_cast<int>(a[0] * a[1]), image};
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *p = static_cast<Prepared *>(state);
  if (!p) return;
  {
    cuda::detail::ScopedPrimaryContext scope(p->primary);
    if (scope && p->quant_module) (void)cuModuleUnload(p->quant_module);
    if (scope && p->gemm_module) (void)cuModuleUnload(p->gemm_module);
  }
  p->~Prepared();
  allocator.deallocate(p, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef, mem::BufferView workspace,
                         void **output) noexcept {
  if (!output) return Status::InvalidArgument("DeepSeek FP4 invocation output is null");
  *output = nullptr;
  const auto &p = *static_cast<const Prepared *>(state);
  if (workspace.size() < p.shape.workspace())
    return Status::InvalidArgument("DeepSeek FP4 workspace is too small");
  HYLE_ASSIGN_OR_RETURN(auto *pointer, cuda::detail::mutable_pointer(workspace, 16, 1));
  *output = pointer;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *workspace, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &p = *static_cast<const Prepared *>(state);
  const auto &s = p.shape;
  auto stream = cuda::stream(queue);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<CUdeviceptr, 3> in;
  for (std::size_t i = 0; i < in.size(); ++i) {
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(), i == 2 ? 4 : 16, 1);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    in[i] = reinterpret_cast<CUdeviceptr>(*pointer);
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(), 16, 1);
  if (!output) return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  unsigned *status_word = nullptr;
  auto initialized = cuda::detail::acquire_device_status(queue, &status_word);
  if (!initialized.status.ok()) return initialized;
  auto codes = reinterpret_cast<CUdeviceptr>(workspace);
  auto scales_a = codes + s.scale_offset();
  auto scales_b = in[0] + std::size_t(s.e) * s.n * (s.k / 2);
  auto out = reinterpret_cast<CUdeviceptr>(*output);
  int groups = s.rows * (s.k / 128);
  std::array<void *, 4> quant{&scales_a, &in[1], &codes, &groups};
  auto result = cuLaunchKernel(p.quant, unsigned((std::uint64_t(groups) + 31) / 32), 1, 1,
      128, 1, 1, 8192, *stream, quant.data(), nullptr);
  if (result != CUDA_SUCCESS) return detail::EnqueueResult::accepted_command(cuda::status(result));
  int experts = s.e, slots = s.slots, activation_slots = s.activation_slots;
  std::array<void *, 10> args{&codes, &in[0], &out, &scales_a, &scales_b, &in[2],
                             &status_word, &experts, &slots, &activation_slots};
  return detail::EnqueueResult::accepted_command(cuda::status(cuLaunchKernel(
      p.gemm, unsigned(s.n / 128), 1, unsigned(s.total_slots),
      128, 1, 1, 10240, *stream, args.data(), nullptr)));
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
  if (!proposal) return Status::InvalidArgument("DeepSeek FP4 proposal is null");
  *proposal = {};
  HYLE_ASSIGN_OR_RETURN(const auto s, select(request));
  if (inputs.size() != 3 || outputs.size() != 1)
    return Status::InvalidArgument("DeepSeek FP4 binding count mismatch");
  HYLE_RETURN_IF_ERROR(fp8::qualify(state));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible, .minimum_alignment = 16};
  inputs[0] = inputs[1] = inputs[2] = outputs[0] = binding;
  inputs[2].minimum_alignment = 4;
  *proposal = {.route = kRoute, .runtime = kRuntime, .workspace_resource = binding,
      .cost = {.workspace = WorkspaceRequirement{s.workspace(), 16}, .persistent_bytes = std::nullopt,
               .copy_bytes = 0, .repack_bytes = 0, .synchronization_count = 0, .dispatch_count = 2},
      .implementation = ProviderImplementationId{5}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  if (!output) return ProviderPrepareOutcome::error(Status::InvalidArgument("DeepSeek FP4 output is null"));
  *output = {};
  if (request.route != kRoute)
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented("DeepSeek FP4 route mismatch"));
  auto s = select({.operation = request.operation, .inputs = request.inputs,
                   .outputs = request.outputs, .shape = request.shape});
  if (!s) return ProviderPrepareOutcome::unsupported(std::move(s).error());
  auto status = fp8::qualify(state);
  if (!status.ok()) return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(status)
                                                   : ProviderPrepareOutcome::error(status);
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  auto *memory = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!memory) return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("DeepSeek FP4 program allocation failed"));
  auto *p = ::new (memory) Prepared{primary, *s};
  const auto load = [](CUmodule &module, CUfunction &function, const unsigned char *image,
                        const char *name) noexcept {
    auto result = cuModuleLoadData(&module, image);
    if (result == CUDA_SUCCESS) result = cuModuleGetFunction(&function, module, name);
    if (result == CUDA_SUCCESS) result = cuFuncLoad(function);
    return result;
  };
  auto result = load(p->quant_module, p->quant, kDeepSeekFp8Encoded, "act_quant_kernel__kernel");
  if (result == CUDA_SUCCESS) result = load(p->gemm_module, p->gemm, s->image, "deepseek_indexed_fp4");
  if (result != CUDA_SUCCESS) {
    destroy(p, allocator);
    return result == CUDA_ERROR_OUT_OF_MEMORY ? ProviderPrepareOutcome::resource_declined(cuda::status(result))
                                             : ProviderPrepareOutcome::error(cuda::status(result));
  }
  *output = {.vtable = &kVTable, .state = p, .workspace = {s->workspace(), 16},
             .runtime = kRuntime, .persistent_bytes = std::nullopt, .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::deepseek_fp4
