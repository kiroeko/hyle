#include "artifact_provider.h"
#include "artifact_kernel.h"
#include "../compute_packed_shared/artifact_packed_linear.h"
#include "../compute_packed_shared/static_fp8_linear.h"
#include "hyle/compute/model_ops.h"
#include "native_context.h"

#include <array>
#include <climits>
#include <cstdint>
#include <new>
#include <optional>
#include <utility>

namespace hyle::compute::cuda_native::artifact {
bool handles(Operation operation) noexcept {
  const auto decode = operation.custom_as<model_ops::ArtifactWeightDecode>();
  return operation.custom_as<model_ops::MlxQuantized>() ||
      operation.custom_as<packed::internal::ArtifactPackedLinear>() ||
      operation.custom_as<packed::internal::StaticFloat8Linear>() ||
      (decode && decode->format == model_ops::WeightArtifactFormat::kGptqV1Int4);
}
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kProvider,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true};
constexpr auto kStaticFloat8Runtime = [] {
  auto value = kRuntime;
  value.workspace_owner = ProviderWorkspaceOwner::kCaller;
  value.replay_class = ProviderReplayClass::kPrepared;
  return value;
}();
struct Program final {
  CUcontext primary = nullptr;
  model_ops::ArtifactWeightDecode operation{};
  model_ops::MlxQuantized mlx{};
  packed::internal::ArtifactPackedLinear packed_artifact{};
  bool is_static_fp8 = false, weight_scale_first = false;
  bool is_mlx = false, is_packed_artifact = false, is_gptq_decode = false;
  std::array<TensorDesc, 7> inputs;
  TensorDesc output;
  std::size_t count = 0;
};
struct Invocation final { void *encoded_activation = nullptr; };

Status select(Operation operation, std::span<const TensorDesc> inputs,
              std::span<const TensorDesc> outputs) noexcept {
  if (!handles(operation)) return Status::Unimplemented("native CUDA packed operation is unavailable");
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return Status::InvalidArgument("native CUDA packed output differs from operation inference");
  const auto packed_artifact = operation.custom_as<packed::internal::ArtifactPackedLinear>();
  if (packed_artifact && (!inputs[0].is_plain() ||
      (packed_artifact->expert_count == 0 && !inputs[0].is_dense()) ||
      !outputs[0].is_plain() || !outputs[0].is_dense()))
    return Status::Unimplemented("CUDA packed linear requires admitted activation and dense output storage");
  const bool mlx = operation.custom_as<model_ops::MlxQuantized>().has_value();
  const auto limit = mlx ? std::uint64_t{INT64_MAX / 32} : std::uint64_t{INT64_MAX};
  for (const auto &desc : inputs) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, desc.required_bytes());
    if (bytes > limit) return Status::OutOfRange("CUDA packed artifact input exceeds native indexing capacity");
  }
  HYLE_ASSIGN_OR_RETURN(const auto elements, outputs[0].element_count());
  if (elements > limit) return Status::OutOfRange("CUDA packed artifact output exceeds native indexing capacity");
  return Status::Ok();
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
  const auto format = operation.format == model_ops::WeightArtifactFormat::kNf4DoubleQuant ? 3ULL :
      operation.format == model_ops::WeightArtifactFormat::kGptqV1Int4 ? 2ULL :
      operation.format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4 ? 1ULL : 0ULL;
  return 0x2000ULL | (operation.expert_count ? 0x1000ULL : 0) | (format << 8) |
      (code(operation.decode_type) << 4) | code(operation.operand_type);
}
native::ArtifactPacked artifact_program(const Program &p) noexcept {
  const auto format = p.is_gptq_decode ? p.operation.format : p.packed_artifact.format;
  const bool gptq = format == model_ops::WeightArtifactFormat::kGptqV1Int4;
  const bool nf4 = format == model_ops::WeightArtifactFormat::kNf4DoubleQuant;
  const auto first = p.is_gptq_decode ? 0U : 1U;
  const auto codes = p.inputs[first].dimensions();
  const auto depth = std::uint64_t(codes[gptq ? 0 : 1]) * (nf4 ? 2 : 8);
  const auto experts = p.is_packed_artifact ? p.packed_artifact.expert_count : 0;
  const auto columns = std::uint64_t(codes[gptq ? 1 : 0]) / (experts ? experts : 1);
  const auto scale_port = first + (format == model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4 ? 1 : 2);
  return {.format = nf4 ? native::ArtifactFormat::kNf4DoubleQuant :
              gptq ? native::ArtifactFormat::kGptqV1Int4 :
              format == model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4
                  ? native::ArtifactFormat::kCtAsymmetric : native::ArtifactFormat::kCtSymmetric,
      .scale_type = nf4 ? native::Scalar::kFloat32 : packed_scalar(p.inputs[scale_port].data_type()),
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
void destroy_program(void *state, mem::AllocatorRef allocator) noexcept {
  auto *program = static_cast<Program *>(state);
  program->~Program(); allocator.deallocate(program, sizeof(Program), alignof(Program));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *state, mem::AllocatorRef allocator, mem::BufferView workspace,
                         void **output) noexcept {
  *output = nullptr;
  const auto &program = *static_cast<const Program *>(state);
  if (!program.is_static_fp8) return Status::Ok();
  if (workspace.size() < *program.inputs[0].element_count())
    return Status::InvalidArgument("CUDA static FP8 activation workspace is too small");
  HYLE_ASSIGN_OR_RETURN(auto encoded, cuda::detail::mutable_pointer(workspace, 16, 1));
  auto *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage) return Status::ResourceExhausted("native CUDA packed invocation allocation failed");
  *output = ::new (storage) Invocation{encoded};
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
  cuda::detail::ScopedPrimaryContext scope(program.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  std::array<const void *, 7> pointers{};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto alignment = program.inputs[i].required_alignment();
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
  return detail::EnqueueResult::rejected(Status::Internal("native CUDA packed route is invalid"));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy_program, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!proposal || inputs.size() != request.inputs.size() || outputs.size() != request.outputs.size())
    return Status::InvalidArgument("native CUDA packed proposal binding count differs");
  if (!request.shape.exact()) return Status::Unimplemented("native CUDA packed requires exact shapes");
  HYLE_RETURN_IF_ERROR(select(request.operation, request.inputs, request.outputs));
  for (std::size_t i = 0; i < inputs.size(); ++i) inputs[i] = device_binding(request.inputs[i]);
  outputs[0] = device_binding(request.outputs[0]);
  const auto mlx = request.operation.custom_as<model_ops::MlxQuantized>();
  const auto packed_artifact = request.operation.custom_as<packed::internal::ArtifactPackedLinear>();
  const auto artifact = request.operation.custom_as<model_ops::ArtifactWeightDecode>();
  const bool static_fp8 = request.operation.custom_as<packed::internal::StaticFloat8Linear>().has_value();
  const bool gptq_decode = artifact && artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4;
  const auto route = static_fp8 ? 0x4000ULL : mlx ? packed_route(*mlx)
      : packed_artifact ? artifact_route(*packed_artifact) : static_cast<std::uint64_t>(artifact->format) + 1;
  *proposal = {.route = route, .runtime = static_fp8 ? kStaticFloat8Runtime : kRuntime,
      .workspace_resource = device_binding(request.outputs[0]),
      .cost = {.workspace = {}, .persistent_bytes = sizeof(Program),
          .copy_bytes = 0, .repack_bytes = 0, .synchronization_count = 0,
          .dispatch_count = std::nullopt},
      .implementation = ProviderImplementationId{0x5443415254000000ULL + route}};
  if (static_fp8) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, request.inputs[0].element_count());
    proposal->cost.workspace = {.size = bytes, .alignment = 16};
    proposal->cost.dispatch_count = 2;
  } else {
    const bool checked = gptq_decode || (packed_artifact &&
        (packed_artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4 || packed_artifact->expert_count != 0));
    proposal->cost.dispatch_count = checked ? 2 :
        (packed_artifact || mlx->kind == model_ops::MlxQuantizedKind::kLinear ? 1 : 2);
  }
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto status = select(request.operation, request.inputs, request.outputs);
  if (!status.ok()) return status.IsUnimplemented() ? ProviderPrepareOutcome::unsupported(std::move(status))
                                                   : ProviderPrepareOutcome::error(std::move(status));
  auto *storage = allocator.allocate(sizeof(Program), alignof(Program));
  if (!storage) return ProviderPrepareOutcome::error(Status::ResourceExhausted("native CUDA packed program allocation failed"));
  const auto artifact = request.operation.custom_as<model_ops::ArtifactWeightDecode>();
  const auto mlx = request.operation.custom_as<model_ops::MlxQuantized>();
  const auto packed_artifact = request.operation.custom_as<packed::internal::ArtifactPackedLinear>();
  const auto static_fp8 = request.operation.custom_as<packed::internal::StaticFloat8Linear>();
  auto *program = ::new (storage) Program{.primary = static_cast<CUcontext>(state),
      .operation = artifact ? *artifact : model_ops::ArtifactWeightDecode{},
      .mlx = mlx ? *mlx : model_ops::MlxQuantized{},
      .packed_artifact = packed_artifact ? *packed_artifact : packed::internal::ArtifactPackedLinear{},
      .is_static_fp8 = static_fp8.has_value(),
      .weight_scale_first = static_fp8 && static_fp8->scale_order == packed::internal::StaticFloat8ScaleOrder::kWeightThenActivation,
      .is_mlx = mlx.has_value(), .is_packed_artifact = packed_artifact.has_value(),
      .is_gptq_decode = artifact && artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4,
      .inputs = {}, .output = request.outputs[0], .count = request.inputs.size()};
  for (std::size_t i = 0; i < program->count; ++i) program->inputs[i] = request.inputs[i];
  cuda::detail::ScopedPrimaryContext scope(program->primary);
  if (!scope) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(scope.status()); }
  if (program->is_static_fp8) status = cuda::status(native::static_fp8_prepare(static_program(*program)));
  else if (program->is_packed_artifact || program->is_gptq_decode)
    status = cuda::status(native::artifact_packed_prepare(artifact_program(*program), program->is_gptq_decode));
  if (!status.ok()) { destroy_program(program, allocator); return ProviderPrepareOutcome::error(std::move(status)); }
  *output = {.vtable = &kPrepared, .state = program,
      .workspace = {.size = program->is_static_fp8 ? *program->inputs[0].element_count() : 0, .alignment = 16},
      .runtime = program->is_static_fp8 ? kStaticFloat8Runtime : kRuntime,
      .persistent_bytes = sizeof(Program), .invocation_private_bytes = program->is_static_fp8 ? sizeof(Invocation) : 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::artifact
