#include "reduction_provider.h"

#include <array>
#include <climits>
#include <new>
#include <utility>

#include "native_context.h"
#include "reduction_kernel.h"

namespace hyle::compute::cuda_native::reduction {
namespace {
struct Prepared final {
  CUcontext primary = nullptr;
  native::ReductionProgram program{};
  std::array<std::size_t, 2> alignment{};
  std::size_t output_alignment = 1;
  bool aliases_input = false;
};
struct Selection final {
  native::ReductionLayout layout{};
  native::ReductionKind kind = native::ReductionKind::kSum;
  native::ReductionMask mask = native::ReductionMask::kNone;
  bool dense = false;
  bool aliases_input = false;
};
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true,
};

Result<Selection> select(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) noexcept {
  const auto *reduce = operation.get_if<Reduce>();
  const auto *softmax = operation.get_if<Softmax>();
  if ((reduce == nullptr && softmax == nullptr) || inputs.empty() ||
      inputs.size() > 2 || outputs.size() != 1)
    return std::unexpected(
        Status::Unimplemented("CUDA reduction requires Reduce or Softmax"));
  const auto &input = inputs[0];
  const auto type = input.data_type();
  if (!input.is_plain() || !outputs[0].is_plain() || !outputs[0].is_dense() ||
      outputs[0].data_type() != type ||
      (type != DataType::kFloat32 && type != DataType::kFloat16 &&
       type != DataType::kBFloat16))
    return std::unexpected(Status::Unimplemented(
        "CUDA reduction requires plain FP32/FP16/BF16 and a dense output"));
  const auto accumulation = reduce != nullptr ? reduce->accumulation_type
                                              : softmax->accumulation_type;
  if ((accumulation != DataType::kFloat32 &&
       accumulation != DataType::kUnknown) ||
      (type != DataType::kFloat32 && accumulation != DataType::kFloat32))
    return std::unexpected(Status::Unimplemented(
        "CUDA reduction requires FP32 accumulation, explicitly for FP16/BF16"));
  HYLE_ASSIGN_OR_RETURN(const auto count, input.element_count());
  HYLE_ASSIGN_OR_RETURN(const auto output_count, outputs[0].element_count());
  if (count > INT32_MAX || output_count > INT32_MAX)
    return std::unexpected(Status::Unimplemented(
        "CUDA reduction requires at most INT32_MAX input/output elements"));
  Selection selected;
  auto &layout = selected.layout;
  layout.rank = static_cast<unsigned>(input.rank());
  if (reduce != nullptr) {
    switch (reduce->kind) {
    case ReductionKind::kSum:
      selected.kind = native::ReductionKind::kSum;
      break;
    case ReductionKind::kMean:
      selected.kind = native::ReductionKind::kMean;
      break;
    case ReductionKind::kMinimum:
      selected.kind = native::ReductionKind::kMinimum;
      break;
    case ReductionKind::kMaximum:
      selected.kind = native::ReductionKind::kMaximum;
      break;
    }
    if (reduce->axes.is_all())
      layout.axes = (1U << layout.rank) - 1;
    else
      for (const auto requested : reduce->axes.values()) {
        const auto axis = requested < 0
                              ? requested + static_cast<int>(layout.rank)
                              : requested;
        layout.axes |= 1U << axis;
      }
    layout.rows = static_cast<unsigned>(output_count);
    // With no output nothing is read. Empty reduced axes with nonempty output
    // have width zero and must still materialize the reduction identities.
    layout.width =
        output_count == 0 ? 0 : static_cast<unsigned>(count / output_count);
    selected.dense = input.is_dense();
    bool reduced = false;
    for (unsigned axis = 0; axis < layout.rank; ++axis) {
      if ((layout.axes & (1U << axis)) != 0)
        reduced = true;
      else if (reduced)
        selected.dense = false;
    }
  } else {
    selected.kind = softmax->kind == SoftmaxKind::kSoftmax
                        ? native::ReductionKind::kSoftmax
                        : native::ReductionKind::kLogSoftmax;
    switch (softmax->mask) {
    case MaskKind::kNone:
      break;
    case MaskKind::kBoolean:
      selected.mask = native::ReductionMask::kBoolean;
      break;
    case MaskKind::kAdditive:
      selected.mask = native::ReductionMask::kAdditive;
      break;
    }
    if (inputs.size() == 2 &&
        (!inputs[1].is_plain() ||
         inputs[1].data_type() !=
             (softmax->mask == MaskKind::kBoolean ? DataType::kBool : type)))
      return std::unexpected(Status::Unimplemented(
          "CUDA Softmax requires a plain Bool or same-type additive mask"));
    const auto axis = softmax->axis < 0
                          ? softmax->axis + static_cast<int>(layout.rank)
                          : softmax->axis;
    layout.axes = 1U << axis;
    layout.scale = softmax->scale;
    if (count != 0) {
      layout.width = static_cast<unsigned>(input.dimensions()[axis]);
      layout.rows = static_cast<unsigned>(count / layout.width);
      for (unsigned next = axis + 1; next < layout.rank; ++next)
        layout.inner *= static_cast<unsigned>(input.dimensions()[next]);
    }
    selected.dense =
        input.is_dense() && axis + 1 == static_cast<int>(layout.rank);
    selected.aliases_input = input == outputs[0];
  }
  // Do not narrow huge dimensions in an empty tensor. Its kernel either does
  // not run or only writes identities without consulting coordinates.
  if (count != 0) {
    for (unsigned axis = 0; axis < layout.rank; ++axis) {
      layout.dimensions[axis] = static_cast<unsigned>(input.dimensions()[axis]);
      layout.input_strides[axis] =
          static_cast<std::size_t>(input.byte_strides()[axis]);
    }
    if (inputs.size() == 2) {
      const auto &mask = inputs[1];
      const auto leading = input.rank() - mask.rank();
      for (std::size_t axis = 0; axis < mask.rank(); ++axis)
        layout.mask_strides[leading + axis] =
            mask.dimensions()[axis] == 1
                ? 0
                : static_cast<std::size_t>(mask.byte_strides()[axis]);
    }
  }
  return selected;
}

void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared == nullptr)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *state, std::size_t output,
                   std::size_t input) noexcept {
  return {output == 0 && input == 0 &&
              static_cast<Prepared *>(state)->aliases_input,
          0};
}
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA reduction invocation output is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.program.layout.rows == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const void *, 2> pointers{};
  if (prepared.program.layout.width != 0)
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto pointer = cuda::detail::const_pointer(inputs[index].storage(),
                                                 prepared.alignment[index], 1);
      if (!pointer)
        return detail::EnqueueResult::rejected(std::move(pointer).error());
      pointers[index] = *pointer;
    }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.output_alignment, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  return detail::EnqueueResult::accepted_command(
      cuda::status(native::reduction_launch(prepared.program, pointers[0],
                                            pointers[1], *output, *stream)));
}
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};
} // namespace

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA reduction requires an exact shape");
  HYLE_ASSIGN_OR_RETURN(
      const auto selected,
      select(request.operation, request.inputs, request.outputs));
  for (std::size_t index = 0; index < inputs.size(); ++index)
    inputs[index] = {
        .required_properties = mem::MemoryProperty::kDeviceAccessible,
        .minimum_alignment = request.inputs[index].required_alignment()};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment =
                    data_type_size(request.outputs[0].data_type())};
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = selected.layout.rows == 0
                                     ? std::optional<std::uint32_t>{}
                                     : std::optional<std::uint32_t>{1}},
      .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto selected = select(request.operation, request.inputs, request.outputs);
  if (!selected)
    return selected.error().IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(
                     std::move(selected).error())
               : ProviderPrepareOutcome::error(std::move(selected).error());
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  const auto type = request.inputs[0].data_type() == DataType::kFloat16
                        ? native::ElementType::kFloat16
                    : request.inputs[0].data_type() == DataType::kBFloat16
                        ? native::ElementType::kBFloat16
                        : native::ElementType::kFloat32;
  native::ReductionProgram program;
  const auto configured =
      native::reduction_prepare(type, selected->kind, selected->mask,
                                selected->dense, selected->layout, &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA reduction program allocation failed"));
  auto *prepared = ::new (storage) Prepared{
      .primary = primary,
      .program = program,
      .output_alignment = data_type_size(request.outputs[0].data_type()),
      .aliases_input = selected->aliases_input};
  for (std::size_t index = 0; index < request.inputs.size(); ++index)
    prepared->alignment[index] = request.inputs[index].required_alignment();
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::reduction
