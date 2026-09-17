#include "scatter_provider.h"

#include <climits>
#include <new>
#include <utility>

#include "native_context.h"
#include "scatter_kernel.h"

namespace hyle::compute::cuda_native::scattering {
namespace {
constexpr std::size_t kWorkspaceAlignment = 256;
struct Prepared final {
  CUcontext primary = nullptr;
  native::ScatterProgram program{};
  std::size_t alignment[3]{};
  bool alias_source = false;
};
struct Invocation final {
  CUcontext primary = nullptr;
  native::ScatterSort sort{};
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
Result<native::ScatterLayout>
select(Operation operation, std::span<const TensorDesc> inputs,
       std::span<const TensorDesc> outputs) noexcept {
  const auto *scatter = operation.get_if<Scatter>();
  if (scatter == nullptr || inputs.size() != 3 || outputs.size() != 1)
    return std::unexpected(Status::Unimplemented(
        "CUDA Scatter requires source, indices, updates and one output"));
  const auto &source = inputs[0];
  const auto &indices = inputs[1];
  const auto &updates = inputs[2];
  const auto bytes = data_type_size(source.data_type());
  if (!source.is_plain() || source.rank() == 0 || !indices.is_plain() ||
      !updates.is_plain() ||
      (indices.data_type() != DataType::kInt32 &&
       indices.data_type() != DataType::kInt64) ||
      !outputs[0].is_plain() || !outputs[0].is_dense() ||
      outputs[0].data_type() != source.data_type() ||
      updates.data_type() != source.data_type() ||
      (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8))
    return std::unexpected(Status::Unimplemented(
        "CUDA Scatter requires plain 1/2/4/8-byte source/updates, "
        "I32/I64 indices and dense output"));
  if (scatter->kind == ScatterKind::kAdd &&
      source.data_type() != DataType::kFloat32 &&
      source.data_type() != DataType::kFloat16 &&
      source.data_type() != DataType::kBFloat16)
    return std::unexpected(Status::Unimplemented(
        "CUDA Scatter add requires F32/F16/BF16 storage"));
  const auto axis = scatter->axis < 0
                        ? scatter->axis + static_cast<int>(source.rank())
                        : scatter->axis;
  if (axis < 0 || axis >= static_cast<int>(source.rank()))
    return std::unexpected(
        Status::OutOfRange("CUDA Scatter axis is out of range"));
  HYLE_ASSIGN_OR_RETURN(const auto elements, outputs[0].element_count());
  if (elements > INT32_MAX)
    return std::unexpected(Status::Unimplemented(
        "CUDA Scatter requires output elements <=INT32_MAX"));
  native::ScatterLayout layout{
      .elements = static_cast<unsigned>(elements),
      .bytes = static_cast<unsigned>(bytes),
      .axis = static_cast<unsigned>(axis),
      .source_rank = static_cast<unsigned>(source.rank()),
      .index_rank = static_cast<unsigned>(indices.rank()),
      .update_rank = static_cast<unsigned>(updates.rank())};
  if (elements == 0)
    return layout;
  HYLE_ASSIGN_OR_RETURN(const auto count, indices.element_count());
  HYLE_ASSIGN_OR_RETURN(const auto update_count, updates.element_count());
  if (count > INT32_MAX || update_count > INT32_MAX)
    return std::unexpected(Status::Unimplemented(
        "CUDA Scatter requires indices and updates <=INT32_MAX"));
  layout.indices = static_cast<unsigned>(count);
  layout.updates = static_cast<unsigned>(update_count);
  layout.extent = static_cast<unsigned>(source.dimensions()[axis]);
  for (unsigned dim = 0; dim < layout.source_rank; ++dim) {
    layout.dimensions[dim] = static_cast<unsigned>(source.dimensions()[dim]);
    layout.source_strides[dim] =
        static_cast<std::size_t>(source.byte_strides()[dim]);
    if (dim > layout.axis)
      layout.inner *= layout.dimensions[dim];
  }
  // Empty indices are a source copy, so no empty-dimension division is needed.
  if (count != 0) {
    for (unsigned dim = 0; dim < layout.index_rank; ++dim) {
      layout.index_dimensions[dim] =
          static_cast<unsigned>(indices.dimensions()[dim]);
      layout.index_strides[dim] =
          static_cast<std::size_t>(indices.byte_strides()[dim]);
    }
    for (unsigned dim = 0; dim < layout.update_rank; ++dim) {
      layout.update_dimensions[dim] =
          static_cast<unsigned>(updates.dimensions()[dim]);
      layout.update_strides[dim] =
          static_cast<std::size_t>(updates.byte_strides()[dim]);
    }
  }
  return layout;
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
  return {.supported = output == 0 && input == 0 &&
                       static_cast<const Prepared *>(state)->alias_source};
}
Status create_invocation(void *state, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA Scatter invocation output is null");
  *output = nullptr;
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.program.workspace_bytes == 0)
    return Status::Ok();
  if (workspace.size() < prepared.program.workspace_bytes)
    return Status::InvalidArgument("CUDA Scatter workspace is too small");
  HYLE_ASSIGN_OR_RETURN(auto *scratch, cuda::detail::mutable_pointer(
                                           workspace, kWorkspaceAlignment, 1));
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (storage == nullptr)
    return Status::ResourceExhausted(
        "CUDA Scatter invocation allocation failed");
  auto *invocation = ::new (storage) Invocation{.primary = prepared.primary};
  *output = invocation; // Common owner destroys partial state on cold failure.
  return cuda::status(native::scatter_sort_create(prepared.program, scratch,
                                                  &invocation->sort));
}
void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(state);
  if (invocation == nullptr)
    return;
  cuda::detail::ScopedPrimaryContext scope(invocation->primary);
  native::scatter_sort_destroy(invocation->sort);
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}
detail::EnqueueResult submit(void *state, void *lane, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.program.layout.elements == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  const void *pointers[3]{};
  for (unsigned index = 0; index < 3; ++index) {
    if (index != 0 && prepared.program.layout.indices == 0)
      continue;
    auto pointer = cuda::detail::const_pointer(inputs[index].storage(),
                                               prepared.alignment[index], 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    pointers[index] = *pointer;
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.program.layout.bytes, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  if (prepared.program.layout.indices == 0 && pointers[0] == *output)
    return detail::EnqueueResult::no_op();
  const native::ScatterSort *sort = nullptr;
  if (prepared.program.workspace_bytes != 0) {
    if (lane == nullptr)
      return detail::EnqueueResult::rejected(
          Status::Internal("CUDA Scatter sort invocation is missing"));
    sort = &static_cast<const Invocation *>(lane)->sort;
  }
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  unsigned *word = nullptr;
  if (prepared.program.checked) {
    auto initialized = cuda::detail::acquire_device_status(queue, &word);
    if (!initialized.status.ok())
      return initialized;
  }
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::scatter_launch(prepared.program, sort, pointers[0], pointers[1],
                             pointers[2], *output, word, *stream)));
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
    return Status::Unimplemented("CUDA Scatter requires an exact shape");
  if (inputs.size() != 3 || outputs.size() != 1)
    return Status::InvalidArgument("CUDA Scatter binding counts differ");
  HYLE_ASSIGN_OR_RETURN(
      const auto layout,
      select(request.operation, request.inputs, request.outputs));
  for (unsigned index = 0; index < 3; ++index)
    inputs[index] = {
        .required_properties = mem::MemoryProperty::kDeviceAccessible,
        .minimum_alignment = request.inputs[index].required_alignment()};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment =
                    data_type_size(request.outputs[0].data_type())};
  const bool sorting = layout.elements != 0 && layout.indices > 1;
  const bool checked =
      layout.indices != 0 && request.operation.get_if<Scatter>()->bounds ==
                                 IndexBoundsPolicy::kChecked;
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .workspace_resource = {.required_properties =
                                 mem::MemoryProperty::kDeviceAccessible,
                             .minimum_alignment = kWorkspaceAlignment},
      .cost = {.workspace = sorting ? std::nullopt
                                    : std::optional{WorkspaceRequirement{}},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count =
                   sorting || layout.elements == 0
                       ? std::optional<std::uint32_t>{}
                       : std::optional<std::uint32_t>{checked ? 2U : 1U}},
      .profile_key = ProviderRouteKey{kRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto layout = select(request.operation, request.inputs, request.outputs);
  if (!layout)
    return layout.error().IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(layout).error())
               : ProviderPrepareOutcome::error(std::move(layout).error());
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  native::ScatterProgram program;
  const auto type = request.inputs[0].data_type();
  const auto *operation = request.operation.get_if<Scatter>();
  const auto configured = native::scatter_prepare(
      type == DataType::kFloat16    ? native::ElementType::kFloat16
      : type == DataType::kBFloat16 ? native::ElementType::kBFloat16
                                    : native::ElementType::kFloat32,
      operation->kind == ScatterKind::kAdd,
      request.inputs[1].data_type() == DataType::kInt64,
      request.inputs[0].is_dense(), request.inputs[1].is_dense(),
      request.inputs[2].is_dense(),
      operation->bounds == IndexBoundsPolicy::kChecked, *layout, &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA Scatter program allocation failed"));
  auto *prepared = ::new (storage)
      Prepared{.primary = primary,
               .program = program,
               .alignment = {request.inputs[0].required_alignment(),
                             request.inputs[1].required_alignment(),
                             request.inputs[2].required_alignment()},
               .alias_source = request.inputs[0] == request.outputs[0]};
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .workspace = {program.workspace_bytes, kWorkspaceAlignment},
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = program.workspace_bytes == 0
                                             ? std::optional<std::size_t>{0}
                                             : std::nullopt};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::scattering
