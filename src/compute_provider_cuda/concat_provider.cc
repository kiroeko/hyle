#include "concat_provider.h"

#include <algorithm>
#include <array>
#include <climits>
#include <new>
#include <utility>

#include "concat_kernel.h"
#include "native_context.h"

namespace hyle::compute::cuda_native::concatenation {
namespace {
struct Group final {
  native::ConcatProgram program{};
  std::array<std::size_t, native::kConcatInputs> alignment{};
};
struct Prepared final {
  CUcontext primary = nullptr;
  Group *groups = nullptr;
  std::size_t count = 0;
  std::size_t value_alignment = 1;
};
struct Selection final {
  std::size_t groups = 0;
  std::size_t persistent_bytes = sizeof(Prepared);
  unsigned elements = 0;
  unsigned inner = 1;
  unsigned axis = 0;
  std::uint32_t dispatches = 0;
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
  const auto *concat = operation.get_if<Concat>();
  if (concat == nullptr || inputs.empty() || outputs.size() != 1)
    return std::unexpected(Status::Unimplemented(
        "CUDA Concat requires input tensors and one output"));
  const auto &output = outputs[0];
  if (!output.is_plain() || !output.is_dense() || output.rank() == 0 ||
      std::any_of(inputs.begin(), inputs.end(),
                  [](const auto &input) { return !input.is_plain(); }))
    return std::unexpected(Status::Unimplemented(
        "CUDA Concat requires plain inputs and a dense output"));
  const auto bytes = data_type_size(output.data_type());
  if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8)
    return std::unexpected(Status::Unimplemented(
        "CUDA Concat requires 1/2/4/8-byte scalar storage"));
  HYLE_ASSIGN_OR_RETURN(const auto elements, output.element_count());
  if (elements > INT32_MAX || inputs.size() > UINT32_MAX)
    return std::unexpected(
        Status::Unimplemented("CUDA Concat requires output elements "
                              "<=INT32_MAX and input count <=UINT32_MAX"));
  Selection selection{.elements = static_cast<unsigned>(elements),
                      .axis = static_cast<unsigned>(
                          concat->axis < 0
                              ? concat->axis + static_cast<int>(output.rank())
                              : concat->axis)};
  if (elements == 0)
    return selection;
  selection.groups =
      (inputs.size() + native::kConcatInputs - 1) / native::kConcatInputs;
  if (selection.groups > (SIZE_MAX - sizeof(Prepared)) / sizeof(Group))
    return std::unexpected(
        Status::OutOfRange("CUDA Concat program metadata size overflows"));
  selection.persistent_bytes += selection.groups * sizeof(Group);
  for (std::size_t dim = selection.axis + 1; dim < output.rank(); ++dim)
    selection.inner *= static_cast<unsigned>(output.dimensions()[dim]);
  for (std::size_t base = 0; base < inputs.size();
       base += native::kConcatInputs) {
    const auto group =
        inputs.subspan(base, std::min<std::size_t>(native::kConcatInputs,
                                                   inputs.size() - base));
    if (std::any_of(group.begin(), group.end(), [&](const auto &input) {
          return input.dimensions()[selection.axis] != 0;
        }))
      ++selection.dispatches;
  }
  return selection;
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared == nullptr)
    return;
  if (prepared->groups != nullptr) {
    for (std::size_t i = 0; i < prepared->count; ++i)
      prepared->groups[i].~Group();
    allocator.deallocate(prepared->groups, prepared->count * sizeof(Group),
                         alignof(Group));
  }
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA Concat invocation output is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.count == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.value_alignment, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  bool accepted = false;
  for (std::size_t index = 0; index < prepared.count; ++index) {
    const auto &group = prepared.groups[index];
    const auto &layout = group.program.layout;
    if (layout.elements == 0)
      continue;
    native::ConcatPointers pointers;
    for (unsigned input = 0; input < layout.inputs; ++input) {
      if (layout.chunks[input] == (input == 0 ? 0 : layout.chunks[input - 1]))
        continue;
      auto pointer = cuda::detail::const_pointer(
          inputs[index * native::kConcatInputs + input].storage(),
          group.alignment[input], 1);
      if (!pointer)
        return {accepted, std::move(pointer).error()};
      pointers.input[input] = *pointer;
    }
    const auto status =
        native::concat_launch(group.program, pointers, *output, *stream);
    accepted = true;
    if (status != cudaSuccess)
      return detail::EnqueueResult::accepted_command(cuda::status(status));
  }
  return {accepted, Status::Ok()};
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
    return Status::Unimplemented("CUDA Concat requires an exact shape");
  HYLE_ASSIGN_OR_RETURN(
      const auto selection,
      select(request.operation, request.inputs, request.outputs));
  for (std::size_t input = 0; input < inputs.size(); ++input)
    inputs[input] = {
        .required_properties = mem::MemoryProperty::kDeviceAccessible,
        .minimum_alignment = request.inputs[input].required_alignment()};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment =
                    data_type_size(request.outputs[0].data_type())};
  *proposal = {.route = kRoute,
               .runtime = kRuntime,
               .cost = {.workspace = WorkspaceRequirement{},
                        .persistent_bytes = selection.persistent_bytes,
                        .copy_bytes = 0,
                        .repack_bytes = 0,
                        .synchronization_count = 0,
                        .dispatch_count = selection.dispatches == 0
                                              ? std::optional<std::uint32_t>{}
                                              : selection.dispatches},
               .implementation = ProviderImplementationId{kRoute}};
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
  auto *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA Concat program allocation failed"));
  auto *prepared = ::new (storage) Prepared{
      .primary = primary,
      .value_alignment = data_type_size(request.outputs[0].data_type())};
  if (selection->groups != 0) {
    prepared->groups = static_cast<Group *>(
        allocator.allocate(selection->groups * sizeof(Group), alignof(Group)));
    if (prepared->groups == nullptr) {
      destroy(prepared, allocator);
      return ProviderPrepareOutcome::resource_declined(
          Status::ResourceExhausted("CUDA Concat group allocation failed"));
    }
    prepared->count = selection->groups;
    for (std::size_t i = 0; i < prepared->count; ++i)
      ::new (prepared->groups + i) Group{};
  }
  const auto &description = request.outputs[0];
  unsigned offset = 0;
  for (std::size_t index = 0; index < prepared->count; ++index) {
    auto &group = prepared->groups[index];
    const auto sources = request.inputs.subspan(
        index * native::kConcatInputs,
        std::min<std::size_t>(native::kConcatInputs,
                              request.inputs.size() -
                                  index * native::kConcatInputs));
    native::ConcatLayout layout{
        .inputs = static_cast<unsigned>(sources.size()),
        .inner = selection->inner,
        .output_row =
            static_cast<unsigned>(description.dimensions()[selection->axis]) *
            selection->inner,
        .output_offset = offset,
        .rank = static_cast<unsigned>(description.rank()),
        .axis = selection->axis};
    for (unsigned axis = 0; axis < layout.rank; ++axis)
      layout.dimensions[axis] =
          static_cast<unsigned>(description.dimensions()[axis]);
    bool dense = true;
    unsigned chunk = 0;
    for (unsigned input = 0; input < layout.inputs; ++input) {
      chunk += static_cast<unsigned>(sources[input].dimensions()[layout.axis]) *
               layout.inner;
      layout.chunks[input] = chunk;
      dense &= sources[input].is_dense();
      group.alignment[input] = sources[input].required_alignment();
      for (unsigned axis = 0; axis < layout.rank; ++axis)
        layout.strides[input][axis] =
            static_cast<std::size_t>(sources[input].byte_strides()[axis]);
    }
    layout.elements = selection->elements / layout.output_row * chunk;
    offset += chunk;
    const auto configured =
        native::concat_prepare(static_cast<unsigned>(prepared->value_alignment),
                               dense, layout, &group.program);
    if (configured != cudaSuccess) {
      destroy(prepared, allocator);
      return ProviderPrepareOutcome::error(cuda::status(configured));
    }
  }
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = selection->persistent_bytes,
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::concatenation
