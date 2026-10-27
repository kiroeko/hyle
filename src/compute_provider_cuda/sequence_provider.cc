#include "sequence_provider.h"

#include <array>
#include <climits>
#include <cmath>
#include <new>
#include <utility>

#include "hyle/compute/sequence_ops.h"
#include "native_context.h"
#include "sequence_kernel.h"

namespace hyle::compute::cuda_native::sequence {
namespace {
struct Selection final {
  native::SequenceLayout layout{};
  native::ElementType type = native::ElementType::kFloat32;
  std::size_t input_count = 0;
  std::array<std::size_t, 7> input_bytes{}, input_alignment{};
  std::array<std::size_t, 2> output_bytes{}, output_alignment{};
};
struct Prepared final {
  CUcontext primary = nullptr;
  native::SequenceProgram program{};
  Selection selection{};
};
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false, .uses_private_threads = false,
    .supports_concurrent_invocations = true, .capture_compatible = true};

Result<Selection> select(Operation operation, std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs) noexcept {
  std::array<TensorDesc, 2> inferred;
  Selection selected;
  if (const auto gdn = operation.custom_as<sequence_ops::PackedGatedDeltaNet>()) {
    HYLE_RETURN_IF_ERROR(gdn->infer_output_descriptions(inputs, inferred));
    const auto q = inputs[0].dimensions(), v = inputs[2].dimensions();
    if (q[2] > 256) return std::unexpected(Status::Unimplemented(
        "CUDA packed GDN register tile supports key depth up to 256"));
    selected.layout = {.kind = native::SequenceKind::kGatedDeltaNet,
        .rows = std::uint64_t(q[0]), .batches = std::uint64_t(inputs[5].dimensions()[0]),
        .key_heads = std::uint64_t(q[1]), .key_depth = std::uint64_t(q[2]),
        .value_heads = std::uint64_t(v[1]), .value_depth = std::uint64_t(v[2]),
        .gate_width = std::uint64_t(inputs[3].dimensions()[2]), .snapshots = gdn->snapshot_count,
        .scale = gdn->output_scale.value_or(1.0F / std::sqrt(float(q[2])))};
  } else if (const auto conv = operation.custom_as<sequence_ops::PackedCausalConv1d>()) {
    HYLE_RETURN_IF_ERROR(conv->infer_output_descriptions(inputs, inferred));
    selected.layout = {.kind = native::SequenceKind::kCausalConv,
        .rows = std::uint64_t(inputs[0].dimensions()[0]),
        .batches = std::uint64_t(inputs[2].dimensions()[0]),
        .channels = std::uint64_t(inputs[0].dimensions()[1]),
        .taps = std::uint64_t(inputs[1].dimensions()[1])};
  } else return std::unexpected(Status::Unimplemented("CUDA packed sequence schema differs"));
  if (outputs.size() != 2 || outputs[0] != inferred[0] || outputs[1] != inferred[1])
    return std::unexpected(Status::InvalidArgument("CUDA packed sequence outputs differ"));
  selected.type = inputs[0].data_type() == DataType::kFloat32 ? native::ElementType::kFloat32
      : inputs[0].data_type() == DataType::kFloat16 ? native::ElementType::kFloat16 : native::ElementType::kBFloat16;
  selected.input_count = inputs.size();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, inputs[i].required_bytes());
    if (bytes > std::size_t(PTRDIFF_MAX)) return std::unexpected(Status::Unimplemented(
        "CUDA packed sequence input exceeds native address capacity"));
    selected.input_bytes[i] = bytes; selected.input_alignment[i] = data_type_size(inputs[i].data_type());
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(const auto bytes, outputs[i].required_bytes());
    if (bytes > std::size_t(PTRDIFF_MAX)) return std::unexpected(Status::Unimplemented(
        "CUDA packed sequence output exceeds native address capacity"));
    selected.output_bytes[i] = bytes; selected.output_alignment[i] = data_type_size(outputs[i].data_type());
  }
  return selected;
}

void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared) { prepared->~Prepared(); allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared)); }
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView, void **output) noexcept {
  if (!output) return Status::InvalidArgument("CUDA packed sequence invocation output is null");
  *output = nullptr; return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  const auto &selected = prepared.selection;
  if (inputs.size() != selected.input_count || outputs.size() != 2)
    return detail::EnqueueResult::rejected(Status::InvalidArgument("CUDA packed sequence binding counts differ"));
  auto stream = cuda::stream(sequence);
  if (!stream) return detail::EnqueueResult::rejected(std::move(stream).error());
  native::SequenceBindings bindings;
  for (std::size_t i = 0; i < inputs.size(); ++i) if (selected.input_bytes[i]) {
    auto pointer = cuda::detail::const_pointer(inputs[i].storage(), selected.input_alignment[i], selected.input_bytes[i]);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    bindings.inputs[i] = *pointer;
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) if (selected.output_bytes[i]) {
    auto pointer = cuda::detail::mutable_pointer(outputs[i].storage(), selected.output_alignment[i], selected.output_bytes[i]);
    if (!pointer) return detail::EnqueueResult::rejected(std::move(pointer).error());
    bindings.outputs[i] = *pointer;
  }
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope) return detail::EnqueueResult::rejected(scope.status());
  unsigned *word = nullptr;
  auto initialized = cuda::detail::acquire_device_status(sequence, &word);
  if (!initialized.status.ok()) return initialized;
  // Initialization may enqueue native work; all subsequent errors retain that
  // accepted prefix through the original Queue/Completion retirement.
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::sequence_launch(prepared.program, bindings, word, *stream)));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy, .alias_support = &alias,
    .create_invocation = &create_invocation, .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace

bool preserves_capture_bindings(const detail::ProviderPrepared &value) noexcept {
  if (value.vtable != &kPrepared || !value.state || value.workspace.size || value.invocation_private_bytes) return false;
  const auto &prepared = *static_cast<const Prepared *>(value.state);
  return prepared.primary && prepared.program.validate && prepared.program.payload &&
      prepared.program.validation_blocks && prepared.program.payload_blocks;
}
Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!proposal) return Status::InvalidArgument("CUDA packed sequence proposal output is null");
  *proposal = {};
  if (!request.shape.exact()) return Status::Unimplemented("CUDA packed sequence needs exact physical shapes");
  HYLE_ASSIGN_OR_RETURN(const auto selected, select(request.operation, request.inputs, request.outputs));
  if (inputs.size() != selected.input_count || outputs.size() != 2)
    return Status::InvalidArgument("CUDA packed sequence proposal binding counts differ");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                 .minimum_alignment = selected.input_alignment[i]};
  for (std::size_t i = 0; i < outputs.size(); ++i)
    outputs[i] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                  .minimum_alignment = selected.output_alignment[i]};
  *proposal = {.route = kRoute, .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{}, .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0, .repack_bytes = 0, .synchronization_count = 0, .dispatch_count = 2},
      .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state, const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator, detail::ProviderPrepared *output) noexcept {
  if (!output) return ProviderPrepareOutcome::error(Status::InvalidArgument("CUDA packed sequence prepared output is null"));
  *output = {};
  if (request.route != kRoute || !request.shape.exact()) return ProviderPrepareOutcome::unsupported(
      Status::Unimplemented("CUDA packed sequence route differs"));
  auto selected = select(request.operation, request.inputs, request.outputs);
  if (!selected) return selected.error().IsUnimplemented()
      ? ProviderPrepareOutcome::unsupported(std::move(selected).error())
      : ProviderPrepareOutcome::error(std::move(selected).error());
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope) return ProviderPrepareOutcome::error(scope.status());
  native::SequenceProgram program;
  const auto status = native::sequence_prepare(selected->type, selected->layout, &program);
  if (status != cudaSuccess) return ProviderPrepareOutcome::error(cuda::status(status));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!storage) return ProviderPrepareOutcome::error(Status::ResourceExhausted("CUDA packed sequence cold allocation failed"));
  auto *prepared = ::new (storage) Prepared{primary, program, *selected};
  *output = {.vtable = &kPrepared, .state = prepared, .workspace = {}, .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared), .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::sequence
