#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/core/environment.h"

#if HYLE_COMPILER_MSVC
// Pinned c10 irange emits C4702 in /O2 SYSTEM headers.
#pragma warning(push)
#pragma warning(disable : 4702)
#endif

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/softplus.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "awq_cuda.h"
#include "bitsandbytes_cuda.h"
#include "compressed_tensors_cuda.h"
#include "mlx_packed_kernel.h"
#include "../compute_packed_shared/artifact_packed_linear.h"
#include "../compute_packed_shared/static_fp8_linear.h"
#include "hyle/compute/detail/capture_bindings.h"
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
constexpr auto kPackedRuntime = [] {
  auto runtime = kRuntime;
  runtime.blocks_host = false;
  return runtime;
}();
constexpr auto kSoftplusRuntime = [] {
  auto runtime = kRuntime;
  runtime.workspace_owner = ProviderWorkspaceOwner::kCaller;
  runtime.blocks_host = false;
  runtime.capture_compatible = true;
  return runtime;
}();
constexpr auto kArtifactPackedRuntime = [] {
  // MLX and artifact packed families submit native kernels on the Queue.
  auto runtime = kPackedRuntime;
  runtime.capture_compatible = true;
  return runtime;
}();
constexpr auto kStaticFloat8Runtime = [] {
  auto runtime = kArtifactPackedRuntime;
  runtime.workspace_owner = ProviderWorkspaceOwner::kCaller;
  runtime.replay_class = ProviderReplayClass::kPrepared;
  return runtime;
}();
struct Context final {
  CUcontext primary = nullptr;
  int device = -1;
  int sm = 0;
};
struct Program final {
  Context context;
  model_ops::ArtifactWeightDecode operation;
  model_ops::MlxQuantized mlx{};
  packed::internal::ArtifactPackedLinear packed_artifact{};
  bool is_static_fp8 = false;
  bool weight_scale_first = false;
  bool is_mlx = false;
  bool is_packed_artifact = false;
  bool is_gptq_decode = false;
  bool is_softplus = false;
  bool is_attention = false;
  ScaledDotProductAttention attention{};
  std::array<TensorDesc, 6> inputs;
  TensorDesc output;
  std::size_t count = 0;
  at::Tensor shifts, order;
};
struct Invocation final { at::Tensor result; void *encoded_activation = nullptr; };

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
  case DataType::kUInt32: return at::kUInt32;
  case DataType::kFloat16: return at::kHalf;
  case DataType::kBFloat16: return at::kBFloat16;
  case DataType::kFloat8E4M3FN: return at::kFloat8_e4m3fn;
  case DataType::kFloat8E5M2: return at::kFloat8_e5m2;
  default: return at::kFloat;
  }
}
Status select(const Context &context, const Operation &operation, std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  const auto mlx = operation.custom_as<model_ops::MlxQuantized>();
  const auto packed_artifact = operation.custom_as<packed::internal::ArtifactPackedLinear>();
  const auto artifact = operation.custom_as<model_ops::ArtifactWeightDecode>();
  const auto static_fp8 = operation.custom_as<packed::internal::StaticFloat8Linear>();
  const bool native_artifact = packed_artifact || static_fp8 ||
      (artifact && artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4);
  const auto *unary = operation.get_if<Unary>();
  const auto *attention = operation.get_if<ScaledDotProductAttention>();
  const bool softplus = unary && unary->kind == UnaryKind::kSoftplus;
  if (!operation.custom_as<model_ops::ArtifactWeightDecode>() && !mlx && !packed_artifact && !static_fp8 && !softplus && !attention)
    return Status::Unimplemented("CUDA ATen carrier does not support this operation");
  if (attention) {
    if (inputs.size() != 3 || outputs.size() != 1 ||
        attention->layout != AttentionLayout::kBatchSequenceHeadDepth ||
        attention->mask != MaskKind::kNone ||
        attention->accumulation_type != DataType::kFloat32 ||
        !std::isfinite(attention->scale) || context.sm < 80 || context.sm > 121)
      return Status::Unimplemented("CUDA ATen BF16 attention requires dense BSHD, FP32 accumulation and SM80-SM121");
    for (const auto &desc : inputs) {
      if (!desc.is_plain() || !desc.is_dense() || desc.rank() != 4 ||
          desc.data_type() != DataType::kBFloat16)
        return Status::Unimplemented("CUDA ATen fused attention requires dense BF16 input ports");
      HYLE_ASSIGN_OR_RETURN(const auto elements, desc.element_count());
      // The original kernel uses 32-bit dimensions/strides and CUDA's
      // 65535-sized batch/head grid axes. No Q*KV scores bound is needed.
      if (elements == 0 || elements > INT32_MAX)
        return Status::Unimplemented("CUDA ATen fused attention exceeds native indexing capacity");
    }
    const auto q = inputs[0].dimensions(), k = inputs[1].dimensions();
    const auto v = inputs[2].dimensions();
    if (q[0] > 65535 || q[2] > 65535 || q[3] < 8 || q[3] > 256 ||
        q[3] % 8 != 0 || v[3] != q[3] ||
        (attention->causal && k[1] < q[1]))
      return Status::Unimplemented("CUDA ATen fused attention shape exceeds its native BF16 domain");
  }
  if (softplus && (inputs.size() != 1 || outputs.size() != 1 ||
      !inputs[0].is_plain() || !inputs[0].is_dense() ||
      !outputs[0].is_plain() || !outputs[0].is_dense() ||
      (inputs[0].data_type() != DataType::kFloat32 &&
       inputs[0].data_type() != DataType::kFloat16 &&
       inputs[0].data_type() != DataType::kBFloat16)))
    return Status::Unimplemented("CUDA ATen stable Softplus requires dense floating storage");
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return Status::InvalidArgument("CUDA ATen output differs from operation inference");
  if (packed_artifact && (!inputs[0].is_plain() ||
      (packed_artifact->expert_count == 0 && !inputs[0].is_dense()) ||
      !outputs[0].is_plain() || !outputs[0].is_dense()))
    return Status::Unimplemented("CUDA packed linear requires admitted activation and dense output storage");
  for (const auto &desc : inputs) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, desc.required_bytes());
    if (bytes > (native_artifact ? std::uint64_t{INT64_MAX} : std::uint64_t{INT64_MAX / 32}))
      return Status::OutOfRange(native_artifact ? "CUDA packed artifact input exceeds native indexing capacity"
                                 : "CUDA ATen temporary tensor capacity overflows");
  }
  HYLE_ASSIGN_OR_RETURN(const auto elements, outputs[0].element_count());
  if (elements > (native_artifact ? std::uint64_t{INT64_MAX} : std::uint64_t{INT64_MAX / 32}))
    return Status::OutOfRange(native_artifact ? "CUDA packed artifact output exceeds native indexing capacity"
                               : "CUDA ATen decoded temporary capacity overflows");
  return Status::Ok();
}

at::Tensor attention_result(const Program &program,
                            const std::array<at::Tensor, 6> &inputs) {
  const auto &q = inputs[0], &k = inputs[1], &v = inputs[2];
  const auto mask = program.attention.causal
      ? sdp::CustomMaskType::CausalFromBottomRight
      : sdp::CustomMaskType::NoCustomMask;
  const auto fused = [&](const at::Tensor &query, const at::Tensor &key,
                         const at::Tensor &value) {
    // Call the original fused operator explicitly: generic SDPA may choose
    // a quadratic math fallback, and its causal flag is top-left aligned.
    // The locked BF16 CUTLASS kernel keeps accumulation/softmax statistics
    // in FP32 and never narrows K/V to F16 or materializes a score matrix.
    // Probability tiles remain BF16; this is not bitwise equivalence with
    // the fallback's fully FP32 intermediate tensors.
    return std::get<0>(at::_efficient_attention_forward(query, key, value,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        0.0, static_cast<std::int64_t>(mask), false,
        static_cast<double>(program.attention.scale)));
  };
  if (q.size(2) == k.size(2))
    return fused(q, k, v);
  const auto ratio = q.size(2) / k.size(2);
  std::vector<at::Tensor> groups;
  groups.reserve(static_cast<std::size_t>(k.size(2)));
  for (std::int64_t head = 0; head < k.size(2); ++head) {
    // The native operator needs equal logical head counts. Zero-head-stride
    // views share each KV head across its Q group without copying the cache.
    auto key = k.narrow(2, head, 1).expand({k.size(0), k.size(1), ratio, k.size(3)});
    auto value = v.narrow(2, head, 1).expand({v.size(0), v.size(1), ratio, v.size(3)});
    groups.push_back(fused(q.narrow(2, head * ratio, ratio), key, value));
  }
  return at::cat(groups, 2);
}

native::Scalar packed_scalar(DataType type) noexcept {
  if (type == DataType::kFloat16) return native::Scalar::kFloat16;
  if (type == DataType::kBFloat16) return native::Scalar::kBFloat16;
  return native::Scalar::kFloat32;
}
native::MlxFormat packed_format(model_ops::MlxWeightFormat format) noexcept {
  switch (format) {
  case model_ops::MlxWeightFormat::kAffine: return native::MlxFormat::kAffine;
  case model_ops::MlxWeightFormat::kMxfp4: return native::MlxFormat::kMxfp4;
  case model_ops::MlxWeightFormat::kMxfp8: return native::MlxFormat::kMxfp8;
  }
  return native::MlxFormat::kAffine; // Invalid metadata was rejected by inference.
}
native::MlxKind packed_kind(model_ops::MlxQuantizedKind kind) noexcept {
  switch (kind) {
  case model_ops::MlxQuantizedKind::kLinear: return native::MlxKind::kLinear;
  case model_ops::MlxQuantizedKind::kGrouped: return native::MlxKind::kGrouped;
  case model_ops::MlxQuantizedKind::kEmbedding: return native::MlxKind::kEmbedding;
  }
  return native::MlxKind::kLinear;
}
native::StaticFloat8 static_program(const Program &p) noexcept {
  return {.activation_type = packed_scalar(p.inputs[0].data_type()),
      .output_type = packed_scalar(p.output.data_type()),
      .rows = std::uint64_t(p.inputs[0].dimensions()[0]),
      .columns = std::uint64_t(p.inputs[1].dimensions()[0]),
      .depth = std::uint64_t(p.inputs[0].dimensions()[1]),
      .activation_scale_stride = *p.inputs[2].element_count() > 1 ? 1u : 0u,
      .weight_scale_stride = *p.inputs[3].element_count() > 1 ? 1u : 0u,
      .weight_scale_first = p.weight_scale_first ? 1u : 0u};
}
std::uint64_t packed_route(const model_ops::MlxQuantized &operation) noexcept {
  const std::uint64_t kind = operation.kind == model_ops::MlxQuantizedKind::kEmbedding ? 2 :
      operation.kind == model_ops::MlxQuantizedKind::kGrouped ? 1 : 0;
  const std::uint64_t family = operation.format == model_ops::MlxWeightFormat::kMxfp8 ? 2 :
      operation.format == model_ops::MlxWeightFormat::kMxfp4 ? 1 : 0;
  const std::uint64_t orientation = operation.orientation ==
      model_ops::MlxMatmulOrientation::kWeightAsStored ? 1 : 0;
  return 0x1000ULL | (orientation << 16) | (kind << 8) | (family << 4) | operation.bits;
}
native::MlxPacked packed_program(const Program &p) noexcept {
  const bool embedding = p.mlx.kind == model_ops::MlxQuantizedKind::kEmbedding;
  const auto w = embedding ? 0 : 1;
  const auto dims = p.inputs[w].dimensions();
  // Schema inference has checked the activation/scales/group/packed-row
  // agreement and arithmetic bounds. A U32 count cannot infer five/six-bit K
  // using an integral number of codes per word.
  const auto k = std::uint64_t(p.inputs[w + 1].dimensions().back()) * p.mlx.group_size;
  const auto columns = std::uint64_t(dims[dims.size() - 2]);
  const bool as_stored = p.mlx.orientation == model_ops::MlxMatmulOrientation::kWeightAsStored;
  const auto output_columns = embedding || as_stored ? k : columns;
  return {.format = packed_format(p.mlx.format), .kind = packed_kind(p.mlx.kind),
      .orientation = as_stored ? native::MlxOrientation::kWeightAsStored
                              : native::MlxOrientation::kTransposeWeight,
      .activation_type = packed_scalar(p.inputs[0].data_type()),
      .scale_type = packed_scalar(p.inputs[w + 1].data_type()),
      .output_type = packed_scalar(p.output.data_type()),
      .bits = p.mlx.bits, .group = p.mlx.group_size,
      .rows = *p.output.element_count() / output_columns,
      .columns = columns, .depth = k,
      .experts = p.mlx.kind == model_ops::MlxQuantizedKind::kGrouped ? std::uint64_t(dims[0]) : 1,
      .slots = p.mlx.kind == model_ops::MlxQuantizedKind::kGrouped
          ? std::uint64_t(p.inputs[p.count - 1].dimensions()[1]) : 1,
      .activation_token_stride = embedding ? 0 :
          std::uint64_t(p.inputs[0].byte_strides()[0]) / data_type_size(p.inputs[0].data_type()),
      .activation_slot_stride = p.mlx.kind == model_ops::MlxQuantizedKind::kGrouped &&
              p.inputs[0].dimensions()[1] != 1
          ? std::uint64_t(p.inputs[0].byte_strides()[1]) / data_type_size(p.inputs[0].data_type()) : 0};
}
std::uint64_t artifact_route(const packed::internal::ArtifactPackedLinear &operation) noexcept {
  const auto code = [](DataType type) -> std::uint64_t {
    return type == DataType::kFloat16 ? 0 : type == DataType::kBFloat16 ? 1 : 2;
  };
  const auto format = operation.format == model_ops::WeightArtifactFormat::kGptqV1Int4 ? 2ULL :
      operation.format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4 ? 1ULL : 0ULL;
  return 0x2000ULL | (operation.expert_count ? 0x1000ULL : 0) | (format << 8) |
      (code(operation.decode_type) << 4) | code(operation.operand_type);
}
native::ArtifactPacked artifact_program(const Program &p) noexcept {
  const auto format = p.is_gptq_decode ? p.operation.format : p.packed_artifact.format;
  const bool gptq = format == model_ops::WeightArtifactFormat::kGptqV1Int4;
  const auto first = p.is_gptq_decode ? 0U : 1U;
  const auto codes = p.inputs[first].dimensions();
  const auto depth = std::uint64_t(codes[gptq ? 0 : 1]) * 8;
  const auto experts = p.is_packed_artifact ? p.packed_artifact.expert_count : 0;
  const auto columns = std::uint64_t(codes[gptq ? 1 : 0]) / (experts ? experts : 1);
  const auto scale_port = first + (format == model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4 ? 1 : 2);
  return {.format = gptq ? native::ArtifactFormat::kGptqV1Int4 :
              format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4
                  ? native::ArtifactFormat::kCtAsymmetric : native::ArtifactFormat::kCtSymmetric,
      .scale_type = packed_scalar(p.inputs[scale_port].data_type()),
      .decode_type = packed_scalar(p.is_gptq_decode ? p.output.data_type() : p.packed_artifact.decode_type),
      .operand_type = packed_scalar(p.is_gptq_decode ? p.output.data_type() : p.packed_artifact.operand_type),
      .group = p.is_gptq_decode ? p.operation.group_size : p.packed_artifact.group_size,
      .rows = p.is_gptq_decode ? 1 : *p.output.element_count() / columns,
      .columns = columns, .depth = depth, .experts = experts,
      .slots = experts ? std::uint64_t(p.inputs[p.count - 1].dimensions()[1]) : 1,
      .activation_token_stride = experts ? std::uint64_t(p.inputs[0].byte_strides()[0]) : 0,
      .activation_slot_stride = experts && p.inputs[0].dimensions()[1] != 1
          ? std::uint64_t(p.inputs[0].byte_strides()[1]) : 0,
      .activation_depth_stride = experts ? std::uint64_t(p.inputs[0].byte_strides()[2]) : sizeof(float)};
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
  // The backend already owns and validates this cold fact. Do not create a
  // second SDK device query or driver dependency for the same ordinal.
  *output = ::new (storage) Context{primary, environment->device_ordinal,
                                  environment->sm_major * 10 + environment->sm_minor};
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
    return Status::Unimplemented("CUDA ATen requires exact shapes");
  HYLE_RETURN_IF_ERROR(select(*static_cast<const Context *>(state),
                             request.operation, request.inputs, request.outputs));
  for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = device_binding(request.inputs[i]);
  if (request.operation.get_if<ScaledDotProductAttention>())
    for (auto &input : inputs)
      input.minimum_alignment = (std::max)(input.minimum_alignment, std::size_t{16});
  outputs[0] = device_binding(request.outputs[0]);
  const auto mlx = request.operation.custom_as<model_ops::MlxQuantized>();
  const auto packed_artifact = request.operation.custom_as<packed::internal::ArtifactPackedLinear>();
  const auto artifact = request.operation.custom_as<model_ops::ArtifactWeightDecode>();
  const auto static_fp8 = request.operation.custom_as<packed::internal::StaticFloat8Linear>();
  const bool gptq_decode = artifact && artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4;
  const auto *unary = request.operation.get_if<Unary>();
  const auto route = static_fp8 ? 0x4000ULL : request.operation.get_if<ScaledDotProductAttention>() ? 0x300ULL
      : unary ? 0x200ULL : mlx ? packed_route(*mlx) : packed_artifact ? artifact_route(*packed_artifact)
      : static_cast<std::uint64_t>(request.operation.custom_as<model_ops::ArtifactWeightDecode>()->format) + 1;
  *proposal = {.route = route,
      .runtime = static_fp8 ? kStaticFloat8Runtime : unary ? kSoftplusRuntime : mlx || packed_artifact || gptq_decode ? kArtifactPackedRuntime : kRuntime,
      .workspace_resource = device_binding(request.outputs[0]),
      .cost = {.workspace = {}, .persistent_bytes = std::nullopt,
               .copy_bytes = std::nullopt, .repack_bytes = std::nullopt,
               .synchronization_count = std::nullopt, .dispatch_count = std::nullopt},
      .implementation = ProviderImplementationId{0x5443415254000000ULL + route}};
  if (static_fp8) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, request.inputs[0].element_count());
    proposal->cost.workspace = {.size = bytes, .alignment = 16};
    proposal->cost.persistent_bytes = sizeof(Program);
    proposal->cost.copy_bytes = 0;
    proposal->cost.repack_bytes = 0;
    proposal->cost.synchronization_count = 0;
    proposal->cost.dispatch_count = 2;
  } else if (mlx || packed_artifact || gptq_decode) {
    proposal->cost.persistent_bytes = sizeof(Program);
    proposal->cost.copy_bytes = 0;
    proposal->cost.repack_bytes = 0;
    proposal->cost.synchronization_count = 0;
    const bool checked_artifact = gptq_decode || (packed_artifact &&
        (packed_artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4 ||
         packed_artifact->expert_count != 0));
    proposal->cost.dispatch_count = checked_artifact ? 2 :
        packed_artifact || mlx->kind == model_ops::MlxQuantizedKind::kLinear ? 1 : 2;
  } else if (unary) {
    proposal->cost.copy_bytes = 0;
    proposal->cost.repack_bytes = 0;
    proposal->cost.synchronization_count = 0;
    proposal->cost.dispatch_count = 1;
  }
  return Status::Ok();
}
void destroy_program(void *state, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<Program *>(state);
  cuda::detail::ScopedPrimaryContext scope(program->context.primary);
  program->~Program(); allocator.deallocate(program, sizeof(Program), alignof(Program));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef allocator, mem::BufferView workspace,
                         void **output) noexcept {
  *output = nullptr;
  const auto &program = *static_cast<const Program *>(state);
  // Native packed paths use only immutable Program metadata and the Queue.
  // Do not allocate an unused ATen result holder or under-report that storage.
  if (program.is_mlx || program.is_packed_artifact || program.is_gptq_decode)
    return Status::Ok();
  auto *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage) return Status::ResourceExhausted("CUDA ATen invocation allocation failed");
  auto *invocation = ::new (storage) Invocation;
  *output = invocation;
  if (program.is_static_fp8) {
    if (workspace.size() < *program.inputs[0].element_count())
      return Status::InvalidArgument("CUDA static FP8 activation workspace is too small");
    HYLE_ASSIGN_OR_RETURN(invocation->encoded_activation,
        cuda::detail::mutable_pointer(workspace, 16, 1));
  }
  return Status::Ok();
}
void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(state);
  if (!invocation) return;
  invocation->~Invocation(); allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}
detail::EnqueueResult submit(void *state, void *slot, detail::QueueSequence &sequence,
                            std::span<const ConstTensorView> inputs,
                            std::span<const TensorView> outputs) noexcept {
  const auto &program = *static_cast<const Program *>(state);
  if (inputs.size() != program.count || outputs.size() != 1)
    return detail::EnqueueResult::rejected(Status::InvalidArgument("CUDA artifact binding count differs"));
  if ((program.is_mlx || program.is_packed_artifact) && *program.output.element_count() == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(sequence);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  cuda::detail::ScopedPrimaryContext scope(program.context.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  std::array<const void *, 6> pointers{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto alignment = program.is_attention
        ? (std::max)(program.inputs[i].required_alignment(), std::size_t{16})
        : program.inputs[i].required_alignment();
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(),
        alignment, data_type_size(program.inputs[i].data_type()));
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    pointers[i] = *pointer;
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
      program.output.required_alignment(), data_type_size(program.output.data_type()));
  if (!output) return detail::EnqueueResult::rejected(std::move(output).error());
  if (program.is_static_fp8) {
    const auto encoded = static_cast<Invocation *>(slot)->encoded_activation;
    if (!encoded) return detail::EnqueueResult::rejected(Status::FailedPrecondition("CUDA static FP8 workspace missing"));
    const auto output_bytes = *program.output.required_bytes();
    const auto out_address = reinterpret_cast<std::uintptr_t>(*output);
    const auto scratch_address = reinterpret_cast<std::uintptr_t>(encoded);
    const auto scratch_bytes = *program.inputs[0].element_count();
    if (output_bytes > UINTPTR_MAX - out_address || scratch_bytes > UINTPTR_MAX - scratch_address)
      return detail::EnqueueResult::rejected(Status::OutOfRange("CUDA static FP8 pointer range overflows"));
    const auto overlaps = [](std::uintptr_t a, std::size_t na, std::uintptr_t b, std::size_t nb) {
      return a < b + nb && b < a + na;
    };
    if (overlaps(out_address, output_bytes, scratch_address, scratch_bytes))
      return detail::EnqueueResult::rejected(Status::InvalidArgument("CUDA static FP8 output overlaps workspace"));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto address = reinterpret_cast<std::uintptr_t>(pointers[i]);
      const auto bytes = *program.inputs[i].required_bytes();
      if (inputs[i].storage().size() < bytes || bytes > UINTPTR_MAX - address)
        return detail::EnqueueResult::rejected(Status::OutOfRange("CUDA static FP8 input span differs"));
      if (overlaps(address, bytes, out_address, output_bytes) ||
          overlaps(address, bytes, scratch_address, scratch_bytes))
        return detail::EnqueueResult::rejected(Status::InvalidArgument("CUDA static FP8 ports overlap workspace or output"));
    }
    if (outputs[0].storage().size() < output_bytes)
      return detail::EnqueueResult::rejected(Status::OutOfRange("CUDA static FP8 output span is too small"));
    return detail::EnqueueResult::accepted_command(cuda::status(native::static_fp8_launch(
        static_program(program), pointers.data(), encoded, *output, *stream)));
  }
  if (program.is_mlx) {
    unsigned *word = nullptr;
    if (program.mlx.kind != model_ops::MlxQuantizedKind::kLinear) {
      auto initialized = cuda::detail::acquire_device_status(sequence, &word);
      if (!initialized.status.ok()) return initialized;
    }
    return detail::EnqueueResult::accepted_command(cuda::status(native::mlx_packed_launch(
        packed_program(program), pointers.data(), *output, word, *stream)));
  }
  if (program.is_packed_artifact || program.is_gptq_decode) {
    unsigned *word = nullptr;
    const auto spec = artifact_program(program);
    if (spec.format == native::ArtifactFormat::kGptqV1Int4 || spec.experts) {
      auto initialized = cuda::detail::acquire_device_status(sequence, &word);
      if (!initialized.status.ok()) return initialized;
    }
    return detail::EnqueueResult::accepted_command(cuda::status(program.is_gptq_decode
        ? native::gptq_decode_launch(spec, pointers.data(), *output, word, *stream)
        : native::artifact_packed_linear_launch(spec, pointers.data(), *output, word, *stream)));
  }
  // Once ATen dispatch starts, any exception may follow an accepted prefix.
  // The real Queue Completion retains all borrowed payload until terminal.
  try {
    auto &invocation = *static_cast<Invocation *>(slot);
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
    auto destination = at::from_blob(*output,
        at::IntArrayRef(program.output.dimensions().data(), program.output.rank()),
        at::TensorOptions().dtype(type).device(at::kCUDA, program.context.device));
    if (program.is_softplus) {
      // The original out kernel writes the real caller destination. Keeping
      // an allocated result followed by copy_ would add private storage and
      // another GPU command, and prevent ordinary command capture.
      at::softplus_out(destination, tensors[0]);
      return detail::EnqueueResult::accepted_command(Status::Ok());
    }
    if (program.is_attention) {
      invocation.result = attention_result(program, tensors);
    } else {
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
      case model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4:
        invocation.result = borrowed::compressed_tensors_cuda::decode_asymmetric4(
            tensors[0], tensors[1], tensors[2], program.operation.group_size,
            program.shifts).to(type); break;
      case model_ops::WeightArtifactFormat::kGptqV1Int4:
        return detail::EnqueueResult::accepted_command(
            Status::Internal("checked GPTQ decoder lost its prepared native route"));
      }
    }
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
  const auto &context = *static_cast<const Context *>(state);
  auto status = select(context, request.operation, request.inputs, request.outputs);
  if (!status.ok()) return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(std::move(status))
                                                   : ProviderPrepareOutcome::error(std::move(status));
  if (context.device < 0 || context.device > (std::numeric_limits<c10::DeviceIndex>::max)())
    return ProviderPrepareOutcome::unsupported(Status::Unimplemented("CUDA ATen device index capacity exceeded"));
  auto *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage) return ProviderPrepareOutcome::error(Status::ResourceExhausted("CUDA ATen program allocation failed"));
  const auto artifact = request.operation.custom_as<model_ops::ArtifactWeightDecode>();
  const auto mlx = request.operation.custom_as<model_ops::MlxQuantized>();
  const auto packed_artifact = request.operation.custom_as<packed::internal::ArtifactPackedLinear>();
  const auto static_fp8 = request.operation.custom_as<packed::internal::StaticFloat8Linear>();
  const auto *unary = request.operation.get_if<Unary>();
  const auto *attention = request.operation.get_if<ScaledDotProductAttention>();
  auto *program = ::new (storage) Program{.context = context,
      .operation = artifact ? *artifact : model_ops::ArtifactWeightDecode{},
      .mlx = mlx ? *mlx : model_ops::MlxQuantized{},
      .packed_artifact = packed_artifact ? *packed_artifact : packed::internal::ArtifactPackedLinear{},
      .is_static_fp8 = static_fp8.has_value(),
      .weight_scale_first = static_fp8 && static_fp8->scale_order ==
          packed::internal::StaticFloat8ScaleOrder::kWeightThenActivation,
      .is_mlx = mlx.has_value(), .is_packed_artifact = packed_artifact.has_value(),
      .is_gptq_decode = artifact && artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4,
      .is_softplus = unary != nullptr,
      .is_attention = attention != nullptr,
      .attention = attention ? *attention : ScaledDotProductAttention{},
      .inputs = {}, .output = request.outputs[0], .count = request.inputs.size(),
      .shifts = {}, .order = {}};
  for (std::size_t i = 0; i < program->count; ++i) program->inputs[i] = request.inputs[i];
  cuda::detail::ScopedPrimaryContext scope(context.primary);
  if (!scope) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(scope.status()); }
  if (program->is_mlx || program->is_packed_artifact || program->is_gptq_decode || program->is_static_fp8) {
    if (program->is_static_fp8) {
      status = cuda::status(native::static_fp8_prepare(static_program(*program)));
      if (!status.ok()) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(std::move(status)); }
    }
    if (program->is_packed_artifact || program->is_gptq_decode) {
      status = cuda::status(native::artifact_packed_prepare(artifact_program(*program), program->is_gptq_decode));
      if (!status.ok()) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(std::move(status)); }
    }
    *output = {.vtable = &kPrepared, .state = program,
        .workspace = {.size = program->is_static_fp8 ? *program->inputs[0].element_count() : 0, .alignment = 16},
        .runtime = program->is_static_fp8 ? kStaticFloat8Runtime : kArtifactPackedRuntime,
        .persistent_bytes = sizeof(Program),
        .invocation_private_bytes = program->is_static_fp8 ? sizeof(Invocation) : 0};
    return ProviderPrepareOutcome::prepared();
  }
  try {
    // The original CUDA hook initializes the device allocator before a BLAS
    // handle requests workspace. Merely loading its DLL does not do this.
    at::globalContext().lazyInitDevice(at::kCUDA);
    const auto index = static_cast<c10::DeviceIndex>(context.device);
    c10::cuda::CUDAGuard device_guard(index);
    at::NoGradGuard no_grad;
    // Real torch_cuda import keeps the original dispatcher registration live;
    // CPU common DLLs remain the immutable locked host dispatch runtime.
    if (!program->is_attention && !program->is_softplus)
      (void)at::cuda::getCurrentCUDABlasHandle();
    const auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, context.device);
    if (!program->is_attention && !program->is_softplus && (program->operation.format == model_ops::WeightArtifactFormat::kAwqGemm4 ||
        program->operation.format == model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4 ||
        program->operation.format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4))
      program->shifts = at::arange(0, 32, 4, options);
    if (!program->is_attention && !program->is_mlx && !program->is_softplus && program->operation.format == model_ops::WeightArtifactFormat::kAwqGemm4)
      program->order = at::tensor({0, 4, 1, 5, 2, 6, 3, 7}, at::kLong).to(options, false, true);
    // Publish constants only after their original library stream has finished.
    if (!program->is_attention && !program->is_softplus) {
      status = cuda::status(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream(index).stream()));
      if (!status.ok()) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(std::move(status)); }
    }
  } catch (...) {
    status = failure(); destroy_program(program, allocator);
    return status.IsResourceExhausted() ? ProviderPrepareOutcome::resource_declined(std::move(status))
                                       : ProviderPrepareOutcome::error(std::move(status));
  }
  *output = {.vtable = &kPrepared, .state = program, .workspace = {},
             .runtime = program->is_softplus ? kSoftplusRuntime : kRuntime,
             .persistent_bytes = std::nullopt, .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
Status drain(void *) noexcept { return Status::Ok(); }
Status inspect_capture_bindings(const detail::ProviderPrepared &value,
                                bool *preserves) noexcept {
  if (!preserves)
    return Status::InvalidArgument("CUDA artifact capture bindings output is null");
  *preserves = false;
  if (value.vtable != &kPrepared || !value.state)
    return Status::Unimplemented("foreign CUDA artifact prepared implementation");
  const auto &program = *static_cast<const Program *>(value.state);
  // This admitted Softplus out path writes the caller destination directly.
  // Do not include managed attention/decode paths that replace result owners,
  // or infer this proof from the shared table's other native packed routes.
  *preserves = program.is_softplus && value.runtime.capture_compatible &&
      value.runtime.mode == ProviderRuntimeMode::kSharedQueue;
  return Status::Ok();
}
const void *query_extension(void *, ProviderExtensionId id) noexcept {
  static constexpr detail::CaptureBindingsVTable capture{
      .inspect = &inspect_capture_bindings};
  return id == detail::kCaptureBindingsExtensionId ? &capture : nullptr;
}
const detail::ProviderVTable kProvider{
    .identity = {.id = kProviderId, .name = "original-artifacts-cuda",
        .version = "libtorch-2.8.0+cu129+bnb-83364904+awq-88e4c76b+packed_artifact-ac8e2ba8-native-packed+mlx-2b2bb497-native-packed+stable-softplus+bf16-efficient-sdpa"},
    .create_context = &create_context, .destroy_context = &destroy_context,
    .propose = &propose, .prepare = &prepare, .drain_context = &drain,
    .query_extension = &query_extension};
} // namespace
Provider provider() noexcept { return detail::ProviderAccess::make(&kProvider, std::uint8_t{0}); }
} // namespace hyle::compute::torch_cuda

#if HYLE_COMPILER_MSVC
#pragma warning(pop)
#endif
