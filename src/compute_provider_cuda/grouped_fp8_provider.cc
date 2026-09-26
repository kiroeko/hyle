#include "grouped_fp8_provider.h"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "hyle/compute/model_ops.h"
#include "fp8_provider.h"
#include "native_context.h"
#include "fp8/sglang_fp8_quantize_cuda_sm89.h"
#include "fp8/grouped/sglang_grouped_fp8_sm89.h"

namespace hyle::compute::cuda_native::grouped_fp8 {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = true};

struct Shape final {
  int e, n, k, rows, slots, capacity, quant_groups, buckets, scan;
  bool bfloat, shared_input, small;
  float floor;
  unsigned align_threads, align_shared, grid;
  std::size_t scale, sorted, experts, post, cumsum, workspace;
};
struct Prepared final {
  CUcontext primary;
  Shape shape;
  std::array<CUmodule, 3> modules{};
  CUfunction quant = nullptr, align = nullptr, sort = nullptr, gemm = nullptr;
};
struct Invocation final { CUdeviceptr workspace; };

Result<Shape> select(const detail::ProviderOperationRequest &request) noexcept {
  const auto op = request.operation.custom_as<model_ops::GroupedFloat8Linear>();
  if (!op || !request.shape.exact() || request.outputs.size() != 1)
    return std::unexpected(Status::Unimplemented("grouped FP8 requires exact model linear"));
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(op->infer_output_descriptions(request.inputs, inferred));
  if (inferred[0] != request.outputs[0])
    return std::unexpected(Status::InvalidArgument("grouped FP8 output description differs"));
  const auto &w = request.inputs[0];
  const auto &a = request.inputs[1];
  const auto e = w.dimensions()[0], n = w.dimensions()[1], k = w.dimensions()[2];
  const auto t = a.dimensions()[0], activations = a.dimensions()[1];
  // Fixed native top-8 GLM variants. The 1024-thread generic align scan needs
  // a thread for the final prefix, hence E+1 must be strictly below 1024.
  if (request.inputs[2].dimensions()[1] != 8 || e > 1022 ||
      n > INT32_MAX || k > INT32_MAX || t > INT32_MAX / 8 ||
      n > INT32_MAX / k || e > INT32_MAX / (n * k) ||
      t * 8 > INT32_MAX / k || t * 8 > INT32_MAX / n)
    return std::unexpected(Status::Unimplemented("grouped FP8 top-8 native indexing domain exceeded"));
  const auto slots = t * 8;
  const auto capacity = ((slots + (e + 1) * 63 + 63) / 64) * 64;
  const auto nblocks = n / 128 + (n % 128 != 0);
  if (capacity > INT32_MAX || capacity / 64 > INT32_MAX / nblocks)
    return std::unexpected(Status::Unimplemented("grouped FP8 padded native grid exceeds I32"));
  Shape s{};
  s.e = static_cast<int>(e); s.n = static_cast<int>(n); s.k = static_cast<int>(k);
  s.rows = static_cast<int>(t * activations); s.slots = static_cast<int>(slots);
  s.capacity = static_cast<int>(capacity); s.buckets = s.e + 1;
  s.scan = static_cast<int>(std::bit_ceil(static_cast<unsigned>(s.buckets)));
  s.quant_groups = 16;
  while ((s.rows * (s.k / 128)) % s.quant_groups) s.quant_groups /= 2;
  s.bfloat = a.data_type() == DataType::kBFloat16;
  s.shared_input = activations == 1;
  s.floor = op->absmax_floor;
  s.small = slots < 1024 && s.buckets <= 64;
  const unsigned threads = static_cast<unsigned>(std::max(s.buckets, 32));
  s.align_threads = s.small ? 256 + threads : 1024;
  s.align_shared = s.small ? ((threads + 1) * s.buckets + s.buckets + 1) * 4
                          : (s.buckets * 2 + 1 + s.scan + 32) * 4;
  s.grid = static_cast<unsigned>((capacity / 64) * nblocks);
  s.scale = std::size_t(s.rows) * s.k;
  // Original align uses vector stores; keep its sorted-token base 16 aligned.
  s.sorted = (s.scale + std::size_t(s.rows) * (s.k / 128) * 4 + 15) & ~std::size_t{15};
  s.experts = s.sorted + std::size_t(s.capacity) * 4;
  s.post = s.experts + std::size_t(s.capacity / 64) * 4;
  s.cumsum = s.post + 4;
  s.workspace = s.cumsum + (s.small ? 0 : std::size_t(s.buckets + 1) * 4);
  return s;
}

void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *p = static_cast<Prepared *>(state);
  if (!p) return;
  {
    cuda::detail::ScopedPrimaryContext scope(p->primary);
    if (scope) for (auto module : p->modules) if (module) (void)cuModuleUnload(module);
  }
  p->~Prepared();
  allocator.deallocate(p, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  if (!output) return Status::InvalidArgument("grouped FP8 invocation output is null");
  *output = nullptr;
  const auto &p = *static_cast<const Prepared *>(state);
  if (workspace.size() < p.shape.workspace)
    return Status::InvalidArgument("grouped FP8 workspace is too small");
  HYLE_ASSIGN_OR_RETURN(auto *pointer, cuda::detail::mutable_pointer(workspace, 16, 1));
  auto *memory = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!memory) return Status::ResourceExhausted("grouped FP8 invocation allocation failed");
  *output = ::new (memory) Invocation{reinterpret_cast<CUdeviceptr>(pointer)};
  return Status::Ok();
}
void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  if (!state) return;
  auto *invocation = static_cast<Invocation *>(state);
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}

detail::EnqueueResult submit(void *state, void *lane, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &p = *static_cast<const Prepared *>(state);
  const auto &s = p.shape;
  auto stream = cuda::stream(queue);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<CUdeviceptr, 3> in;
  for (std::size_t i = 0; i < in.size(); ++i) {
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(), i == 1 ? 16 : 4, 1);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    in[i] = reinterpret_cast<CUdeviceptr>(*pointer);
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(), s.bfloat ? 2 : 4, 1);
  if (!output) return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  unsigned *status_word = nullptr;
  auto initialized = cuda::detail::acquire_device_status(queue, &status_word);
  if (!initialized.status.ok()) return initialized;
  const auto base = static_cast<const Invocation *>(lane)->workspace;
  std::array<CUdeviceptr, 10> ptrs{base, in[0], reinterpret_cast<CUdeviceptr>(*output),
      base + s.scale, in[0] + std::size_t(s.e) * s.n * s.k,
      base + s.sorted, base + s.experts, base + s.post, 0, 0};
  const auto launch = [&](CUfunction f, unsigned grid, unsigned threads, unsigned shared,
                           auto &args) noexcept {
    return cuLaunchKernel(f, grid, 1, 1, threads, 1, 1, shared, *stream, args.data(), nullptr);
  };
  int block = 128, groups = s.rows * (s.k / 128), per_block = s.quant_groups, unused = 0;
  float floor = s.floor, minimum = -448, maximum = 448;
  std::array<void *, 11> quant{&in[1], &ptrs[0], &ptrs[3], &block, &groups, &per_block,
      &floor, &minimum, &maximum, &unused, &unused};
  auto result = launch(p.quant, unsigned(groups / per_block), unsigned(per_block * 16), 0, quant);
  if (result != CUDA_SUCCESS) return detail::EnqueueResult::accepted_command(cuda::status(result));

  int buckets = s.buckets, tile = 64, capacity = s.capacity, scan = s.scan;
  std::uint64_t count = static_cast<std::uint64_t>(s.slots);
  bool pad = true, ignore = false;
  auto cumsum = base + s.cumsum;
  if (s.small) {
    std::array<void *, 11> args{&in[2], &ptrs[5], &ptrs[6], &ptrs[7], &buckets,
        &tile, &count, &pad, &ignore, &capacity, &status_word};
    result = launch(p.align, 1, s.align_threads, s.align_shared, args);
  } else {
    std::array<void *, 13> args{&in[2], &ptrs[5], &ptrs[6], &ptrs[7], &buckets,
        &tile, &count, &cumsum, &pad, &ignore, &scan, &capacity, &status_word};
    result = launch(p.align, 2, s.align_threads, s.align_shared, args);
    if (result == CUDA_SUCCESS) {
      std::array<void *, 7> sort{&in[2], &ptrs[5], &cumsum, &count, &ignore, &buckets, &status_word};
      result = launch(p.sort, std::min(unsigned((count + 255) / 256), 65535u), 256, 0, sort);
    }
  }
  if (result != CUDA_SUCCESS) return detail::EnqueueResult::accepted_command(cuda::status(result));
  // Original grouped GEMM consumes only the bounds-checked sorted IDs. Invalid
  // slots never become expert addresses; Queue translates the status at Completion.
  std::array integers{s.n, s.k, s.capacity, s.slots, s.k, 1, s.n * s.k, 1, s.k,
      0, 0, s.n, 1, s.k / 128, 1, (s.n / 128 + (s.n % 128 != 0)) * (s.k / 128), 1, s.k / 128};
  std::array<void *, 28> gemm;
  for (std::size_t i = 0; i < 8; ++i) gemm[i] = &ptrs[i];
  for (std::size_t i = 0; i < integers.size(); ++i) gemm[i + 8] = &integers[i];
  gemm[26] = &ptrs[8]; gemm[27] = &ptrs[9]; // Triton 3.8 zero-sized scratch ABI.
  return detail::EnqueueResult::accepted_command(
      cuda::status(launch(p.gemm, s.grid, 128, 24576, gemm)));
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
  if (!proposal || inputs.size() != 3 || outputs.size() != 1)
    return Status::InvalidArgument("grouped FP8 proposal bindings differ");
  *proposal = {};
  HYLE_ASSIGN_OR_RETURN(const auto s, select(request));
  HYLE_RETURN_IF_ERROR(fp8::qualify(state));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible, .minimum_alignment = 4};
  std::ranges::fill(inputs, binding);
  inputs[1].minimum_alignment = 16;
  outputs[0] = binding;
  outputs[0].minimum_alignment = s.bfloat ? 2 : 4;
  auto workspace = binding;
  workspace.minimum_alignment = 16;
  *proposal = {.route = kRoute, .runtime = kRuntime, .workspace_resource = workspace,
      .cost = {.workspace = WorkspaceRequirement{s.workspace, 16}, .persistent_bytes = std::nullopt,
               .copy_bytes = 0, .repack_bytes = 0, .synchronization_count = 0,
               .dispatch_count = s.small ? 3u : 4u},
      .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  if (!output) return ProviderPrepareOutcome::error(Status::InvalidArgument("grouped FP8 prepared output is null"));
  *output = {};
  if (request.route != kRoute)
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented("grouped FP8 route mismatch"));
  auto shape = select({.operation = request.operation, .inputs = request.inputs,
                       .outputs = request.outputs, .shape = request.shape});
  if (!shape) return ProviderPrepareOutcome::unsupported(std::move(shape).error());
  auto status = fp8::qualify(state);
  if (!status.ok()) return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(status)
                                                   : ProviderPrepareOutcome::error(status);
  auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  auto *memory = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!memory) return ProviderPrepareOutcome::resource_declined(Status::ResourceExhausted("grouped FP8 program allocation failed"));
  auto *p = ::new (memory) Prepared{primary, *shape};
  const auto load = [](CUmodule &module, const unsigned char *image, CUfunction &function,
                        const char *name) noexcept {
    auto result = cuModuleLoadData(&module, image);
    if (result == CUDA_SUCCESS) result = cuModuleGetFunction(&function, module, name);
    if (result == CUDA_SUCCESS) result = cuFuncLoad(function);
    return result;
  };
  auto result = load(p->modules[0], fp8::kSglangFp8QuantizeCuda, p->quant,
      shape->bfloat ? fp8::kSglangFp8QuantizeBf16Name : fp8::kSglangFp8QuantizeF32Name);
  if (result == CUDA_SUCCESS)
    result = load(p->modules[1], kCheckedAlign, p->align, shape->small ? kAlignSmallName : kAlignName);
  if (result == CUDA_SUCCESS && !shape->small) {
    result = cuModuleGetFunction(&p->sort, p->modules[1], kAlignSortName);
    if (result == CUDA_SUCCESS) result = cuFuncLoad(p->sort);
  }
  const auto *gemm = shape->bfloat
      ? (shape->shared_input ? kGroupedBf16Top8 : kGroupedBf16Top1)
      : (shape->shared_input ? kGroupedF32Top8 : kGroupedF32Top1);
  if (result == CUDA_SUCCESS) result = load(p->modules[2], gemm, p->gemm, "fused_moe_kernel");
  if (result != CUDA_SUCCESS) {
    destroy(p, allocator);
    status = cuda::status(result);
    return result == CUDA_ERROR_OUT_OF_MEMORY ? ProviderPrepareOutcome::resource_declined(status)
                                             : ProviderPrepareOutcome::error(status);
  }
  *output = {.vtable = &kVTable, .state = p, .workspace = {shape->workspace, 16},
      .runtime = kRuntime, .persistent_bytes = std::nullopt,
      .invocation_private_bytes = sizeof(Invocation)};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::grouped_fp8
