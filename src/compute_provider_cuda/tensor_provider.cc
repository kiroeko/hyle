#include "tensor_provider.h"

#include <algorithm>
#include <array>
#include <climits>
#include <new>
#include <utility>

#include "../compute_ops/view_alias.h"
#include "native_context.h"
#include "tensor_kernel.h"

namespace hyle::compute::cuda_native::tensor {
namespace {
struct Prepared final {
  CUcontext primary = nullptr;
  native::TensorProgram program{};
  std::array<std::size_t, 3> input_alignment{};
  std::size_t output_alignment = 1;
  std::size_t alias_offset = 0;
  unsigned alias_mask = 0;
};

struct Selection final {
  native::TensorKind kind = native::TensorKind::kCopy;
  native::TensorLayout layout{};
  bool dense = true;
  bool view = false;
  std::size_t alias_offset = 0;
  unsigned alias_mask = 0;
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

bool numeric(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}
native::ElementType element_type(DataType type) noexcept {
  return type == DataType::kFloat16    ? native::ElementType::kFloat16
         : type == DataType::kBFloat16 ? native::ElementType::kBFloat16
                                       : native::ElementType::kFloat32;
}

Result<Selection> select(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs) noexcept {
  if (outputs.size() != 1 || inputs.size() > 3)
    return std::unexpected(Status::Unimplemented(
        "CUDA tensor primitives require one output and at most three inputs"));
  if (!outputs[0].is_plain() ||
      std::any_of(inputs.begin(), inputs.end(),
                  [](const TensorDesc &input) { return !input.is_plain(); }))
    return std::unexpected(
        Status::Unimplemented("CUDA tensor primitives require plain storage"));
  Selection selected;
  if (const auto *view = operation.get_if<View>()) {
    HYLE_ASSIGN_OR_RETURN(
        selected.alias_offset,
        detail::view_alias_offset(*view, inputs[0], outputs[0]));
    selected.view = true;
    selected.alias_mask = 1;
    return selected;
  }
  if (!outputs[0].is_dense())
    return std::unexpected(
        Status::Unimplemented("CUDA tensor primitives require a dense output"));
  HYLE_ASSIGN_OR_RETURN(const auto count, outputs[0].element_count());
  if (count > INT32_MAX)
    return std::unexpected(Status::Unimplemented(
        "CUDA tensor primitives require at most INT32_MAX output elements"));
  selected.layout.elements = static_cast<unsigned>(count);
  selected.layout.rank = static_cast<unsigned>(outputs[0].rank());
  bool arithmetic = false;
  switch (operation.kind()) {
  case OperationKind::kCopy:
    selected.kind = native::TensorKind::kCopy;
    break;
  case OperationKind::kFill:
    selected.kind = native::TensorKind::kFill;
    selected.layout.scalar_bits = operation.get_if<Fill>()->value.bits();
    break;
  case OperationKind::kCast:
    arithmetic = true;
    selected.kind = inputs[0].data_type() == outputs[0].data_type()
                        ? native::TensorKind::kCopy
                        : native::TensorKind::kCast;
    break;
  case OperationKind::kSelect:
    selected.kind = native::TensorKind::kSelect;
    break;
  case OperationKind::kUnary:
    arithmetic = true;
    switch (operation.get_if<Unary>()->kind) {
#define HYLE_CUDA_UNARY(name)                                                  \
  case UnaryKind::name:                                                        \
    selected.kind = native::TensorKind::name;                                  \
    break
      HYLE_CUDA_UNARY(kAbs);
      HYLE_CUDA_UNARY(kNeg);
      HYLE_CUDA_UNARY(kExp);
      HYLE_CUDA_UNARY(kLog);
      HYLE_CUDA_UNARY(kSqrt);
      HYLE_CUDA_UNARY(kRsqrt);
      HYLE_CUDA_UNARY(kTanh);
      HYLE_CUDA_UNARY(kSigmoid);
      HYLE_CUDA_UNARY(kRelu);
      HYLE_CUDA_UNARY(kGelu);
      HYLE_CUDA_UNARY(kGeluTanh);
      HYLE_CUDA_UNARY(kSilu);
      HYLE_CUDA_UNARY(kSoftplus);
#undef HYLE_CUDA_UNARY
    }
    break;
  case OperationKind::kBinary:
    arithmetic = true;
    switch (operation.get_if<Binary>()->kind) {
#define HYLE_CUDA_BINARY(name)                                                 \
  case BinaryKind::name:                                                       \
    selected.kind = native::TensorKind::name;                                  \
    break
      HYLE_CUDA_BINARY(kAdd);
      HYLE_CUDA_BINARY(kSubtract);
      HYLE_CUDA_BINARY(kMultiply);
      HYLE_CUDA_BINARY(kDivide);
      HYLE_CUDA_BINARY(kMinimum);
      HYLE_CUDA_BINARY(kMaximum);
#undef HYLE_CUDA_BINARY
    }
    break;
  default:
    return std::unexpected(Status::Unimplemented(
        "CUDA tensor provider does not implement this operation family"));
  }
  if (arithmetic &&
      (!numeric(outputs[0].data_type()) ||
       std::any_of(inputs.begin(), inputs.end(), [](const TensorDesc &input) {
         return !numeric(input.data_type());
       })))
    return std::unexpected(Status::Unimplemented(
        "CUDA tensor arithmetic and Cast require FP32/FP16/BF16"));
  const auto width = data_type_size(outputs[0].data_type());
  if (width != 1 && width != 2 && width != 4 && width != 8)
    return std::unexpected(Status::Unimplemented(
        "CUDA tensor storage requires a 1/2/4/8-byte scalar"));
  for (std::size_t axis = 0; axis < outputs[0].rank(); ++axis)
    selected.layout.dimensions[axis] =
        static_cast<unsigned>(outputs[0].dimensions()[axis]);
  for (std::size_t input = 0; input < inputs.size(); ++input) {
    const auto &desc = inputs[input];
    const bool same_shape =
        std::ranges::equal(desc.dimensions(), outputs[0].dimensions());
    selected.dense &= desc.is_dense() && same_shape;
    const auto leading = outputs[0].rank() - desc.rank();
    for (std::size_t axis = 0; axis < desc.rank(); ++axis)
      selected.layout.strides[input][axis + leading] =
          desc.dimensions()[axis] == 1
              ? 0
              : static_cast<std::size_t>(desc.byte_strides()[axis]);
    if ((operation.kind() == OperationKind::kUnary ||
         operation.kind() == OperationKind::kBinary ||
         (operation.kind() == OperationKind::kSelect && input != 0)) &&
        desc == outputs[0])
      selected.alias_mask |= 1U << input;
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
  const auto &prepared = *static_cast<const Prepared *>(state);
  return {output == 0 && input < 3 &&
              (prepared.alias_mask & (1U << input)) != 0,
          prepared.alias_offset};
}
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA tensor invocation output is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}

detail::EnqueueResult submit(void *state, void *, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.program.layout.elements == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const void *, 3> pointers{};
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    auto pointer = cuda::detail::const_pointer(
        inputs[index].storage(), prepared.input_alignment[index], 1);
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
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::tensor_launch(prepared.program, pointers[0], pointers[1],
                            pointers[2], *output, *stream)));
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
    return Status::Unimplemented("CUDA tensors require an exact shape");
  HYLE_ASSIGN_OR_RETURN(
      const auto selection,
      select(request.operation, request.inputs, request.outputs));
  for (std::size_t index = 0; index < inputs.size(); ++index)
    inputs[index] = {
        .required_properties = mem::MemoryProperty::kDeviceAccessible,
        .minimum_alignment =
            selection.view || !selection.dense
                ? 1
                : data_type_size(request.inputs[index].data_type())};
  outputs[0] = {
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment =
          selection.view ? 1 : data_type_size(request.outputs[0].data_type())};
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = selection.layout.elements == 0
                                     ? std::optional<std::uint32_t>{}
                                     : std::optional<std::uint32_t>{1}},
      .profile_key = ProviderRouteKey{kRoute}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto selection = select(request.operation, request.inputs, request.outputs);
  if (!selection)
    return selection.error().IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(
                     std::move(selection).error())
               : ProviderPrepareOutcome::error(std::move(selection).error());
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  native::TensorProgram program;
  if (!selection->view) {
    const auto type = element_type(request.outputs[0].data_type());
    const auto input_type = request.inputs.empty()
                                ? type
                                : element_type(request.inputs[0].data_type());
    const auto configured = native::tensor_prepare(
        selection->kind, type, input_type,
        static_cast<unsigned>(data_type_size(request.outputs[0].data_type())),
        selection->dense, selection->layout, &program);
    if (configured != cudaSuccess)
      return ProviderPrepareOutcome::error(cuda::status(configured));
  }
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA tensor program allocation failed"));
  auto *prepared = ::new (storage) Prepared{
      .primary = primary,
      .program = program,
      .output_alignment =
          selection->view ? 1 : data_type_size(request.outputs[0].data_type()),
      .alias_offset = selection->alias_offset,
      .alias_mask = selection->alias_mask};
  for (std::size_t index = 0; index < request.inputs.size(); ++index)
    prepared->input_alignment[index] =
        selection->view || !selection->dense
            ? 1
            : data_type_size(request.inputs[index].data_type());
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::tensor
