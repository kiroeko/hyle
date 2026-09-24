#include "gather_provider.h"

#include <climits>
#include <new>
#include <utility>

#include "gather_kernel.h"
#include "native_context.h"

namespace hyle::compute::cuda_native::gathering {
namespace {
struct Prepared final {
  CUcontext primary = nullptr;
  native::GatherProgram program{};
  std::size_t source_alignment = 1;
  std::size_t index_alignment = 1;
  std::size_t output_alignment = 1;
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
Result<native::GatherLayout>
select(Operation operation, std::span<const TensorDesc> inputs,
       std::span<const TensorDesc> outputs) noexcept {
  const auto *gather = operation.get_if<Gather>();
  if (gather == nullptr || inputs.size() != 2 || outputs.size() != 1)
    return std::unexpected(Status::Unimplemented(
        "CUDA Gather requires source, indices and one output"));
  const auto &source = inputs[0];
  const auto &indices = inputs[1];
  const auto bytes = data_type_size(source.data_type());
  if (!source.is_plain() || source.rank() == 0 || !indices.is_plain() ||
      (indices.data_type() != DataType::kInt32 &&
       indices.data_type() != DataType::kInt64) ||
      !outputs[0].is_plain() || !outputs[0].is_dense() ||
      outputs[0].data_type() != source.data_type() ||
      (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8))
    return std::unexpected(
        Status::Unimplemented("CUDA Gather requires plain 1/2/4/8-byte source, "
                              "I32/I64 indices and dense output"));
  const auto axis = gather->axis < 0
                        ? gather->axis + static_cast<int>(source.rank())
                        : gather->axis;
  if (axis < 0 || axis >= static_cast<int>(source.rank()))
    return std::unexpected(
        Status::OutOfRange("CUDA Gather axis is out of range"));
  HYLE_ASSIGN_OR_RETURN(const auto elements, outputs[0].element_count());
  if (elements > INT32_MAX)
    return std::unexpected(Status::Unimplemented(
        "CUDA Gather requires output elements <=INT32_MAX"));
  native::GatherLayout layout{
      .elements = static_cast<unsigned>(elements),
      .copy_bytes = static_cast<unsigned>(bytes),
      .axis = static_cast<unsigned>(axis),
      .source_rank = static_cast<unsigned>(source.rank()),
      .index_rank = static_cast<unsigned>(indices.rank())};
  if (elements == 0)
    return layout;
  layout.axis_extent = static_cast<std::size_t>(source.dimensions()[axis]);
  if (layout.axis_extent == 0)
    return std::unexpected(
        Status::OutOfRange("CUDA Gather cannot index an empty axis"));
  HYLE_ASSIGN_OR_RETURN(const auto count, indices.element_count());
  if (count > INT32_MAX)
    return std::unexpected(
        Status::Unimplemented("CUDA Gather requires indices <=INT32_MAX"));
  layout.indices = static_cast<unsigned>(count);
  for (unsigned dim = 0; dim < layout.source_rank; ++dim) {
    layout.dimensions[dim] = static_cast<std::size_t>(source.dimensions()[dim]);
    layout.strides[dim] = static_cast<std::size_t>(source.byte_strides()[dim]);
    if (dim > layout.axis)
      layout.inner *= static_cast<unsigned>(source.dimensions()[dim]);
  }
  for (unsigned dim = 0; dim < layout.index_rank; ++dim) {
    layout.index_dimensions[dim] =
        static_cast<unsigned>(indices.dimensions()[dim]);
    layout.index_strides[dim] =
        static_cast<std::size_t>(indices.byte_strides()[dim]);
  }
  // Vectorize complete contiguous suffixes without imposing 16-byte alignment
  // on ordinary tensor bindings. The bound kernel has a byte-safe address path.
  if (source.is_dense() &&
      (static_cast<std::size_t>(layout.inner) * bytes) % 16 == 0) {
    const auto scalars = static_cast<unsigned>(16 / bytes);
    layout.copy_bytes = 16;
    layout.elements /= scalars;
    layout.inner /= scalars;
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
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA Gather invocation output is null");
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
  auto source = cuda::detail::const_pointer(inputs[0].storage(),
                                            prepared.source_alignment, 1);
  if (!source)
    return detail::EnqueueResult::rejected(std::move(source).error());
  auto indices = cuda::detail::const_pointer(inputs[1].storage(),
                                             prepared.index_alignment, 1);
  if (!indices)
    return detail::EnqueueResult::rejected(std::move(indices).error());
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.output_alignment, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  unsigned *word = nullptr;
  if (prepared.program.validate != nullptr) {
    auto initialized = cuda::detail::acquire_device_status(queue, &word);
    if (!initialized.status.ok())
      return initialized;
  }
  return detail::EnqueueResult::accepted_command(
      cuda::status(native::gather_launch(prepared.program, *source, *indices,
                                         *output, word, *stream)));
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
    return Status::Unimplemented("CUDA Gather requires an exact shape");
  if (inputs.size() != 2 || outputs.size() != 1)
    return Status::InvalidArgument("CUDA Gather binding counts differ");
  HYLE_ASSIGN_OR_RETURN(
      const auto layout,
      select(request.operation, request.inputs, request.outputs));
  for (unsigned index = 0; index < 2; ++index)
    inputs[index] = {
        .required_properties = mem::MemoryProperty::kDeviceAccessible,
        .minimum_alignment = request.inputs[index].required_alignment()};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment =
                    data_type_size(request.outputs[0].data_type())};
  const bool checked =
      request.operation.get_if<Gather>()->bounds == IndexBoundsPolicy::kChecked;
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count =
                   layout.elements == 0
                       ? std::optional<std::uint32_t>{}
                       : std::optional<std::uint32_t>{checked ? 2U : 1U}},
      .implementation = ProviderImplementationId{kRoute}};
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
  native::GatherProgram program;
  const auto configured = native::gather_prepare(
      request.inputs[1].data_type() == DataType::kInt64,
      request.inputs[0].is_dense(), request.inputs[1].is_dense(),
      request.operation.get_if<Gather>()->bounds == IndexBoundsPolicy::kChecked,
      *layout, &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA Gather program allocation failed"));
  auto *prepared = ::new (storage)
      Prepared{primary, program, request.inputs[0].required_alignment(),
               request.inputs[1].required_alignment(),
               data_type_size(request.outputs[0].data_type())};
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::gathering
