#include "scan_provider.h"

#include <climits>
#include <new>
#include <utility>

#include "native_context.h"
#include "scan_kernel.h"

namespace hyle::compute::cuda_native::scanning {
namespace {
struct Prepared final {
  CUcontext primary = nullptr;
  native::ScanProgram program{};
  std::size_t alignment = 1;
  bool alias_source = false;
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
Result<native::ScanLayout>
select(Operation operation, std::span<const TensorDesc> inputs,
       std::span<const TensorDesc> outputs) noexcept {
  const auto *scan = operation.get_if<Scan>();
  if (scan == nullptr || inputs.size() != 1 || outputs.size() != 1)
    return std::unexpected(
        Status::Unimplemented("CUDA Scan requires one input and output"));
  const auto &input = inputs[0];
  if (!input.is_plain() || input.rank() == 0 ||
      (input.data_type() != DataType::kInt32 &&
       input.data_type() != DataType::kFloat32) ||
      !outputs[0].is_plain() || !outputs[0].is_dense() ||
      outputs[0].data_type() != input.data_type())
    return std::unexpected(Status::Unimplemented(
        "CUDA Scan requires plain I32/F32 and dense output"));
  const auto axis =
      scan->axis < 0 ? scan->axis + static_cast<int>(input.rank()) : scan->axis;
  if (axis < 0 || axis >= static_cast<int>(input.rank()))
    return std::unexpected(
        Status::OutOfRange("CUDA Scan axis is out of range"));
  HYLE_ASSIGN_OR_RETURN(const auto count, outputs[0].element_count());
  if (count > INT32_MAX)
    return std::unexpected(
        Status::Unimplemented("CUDA Scan requires elements <=INT32_MAX"));
  native::ScanLayout layout{.rank = static_cast<unsigned>(input.rank())};
  if (count == 0)
    return layout;
  layout.width = static_cast<unsigned>(input.dimensions()[axis]);
  layout.rows = static_cast<unsigned>(count / layout.width);
  for (unsigned dim = 0; dim < layout.rank; ++dim) {
    layout.dimensions[dim] = static_cast<unsigned>(input.dimensions()[dim]);
    layout.strides[dim] = static_cast<std::size_t>(input.byte_strides()[dim]);
    if (dim > static_cast<unsigned>(axis))
      layout.inner *= layout.dimensions[dim];
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
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA Scan invocation output is null");
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
  auto input =
      cuda::detail::const_pointer(inputs[0].storage(), prepared.alignment, 1);
  if (!input)
    return detail::EnqueueResult::rejected(std::move(input).error());
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(), 4, 1);
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
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::scan_launch(prepared.program, *input, *output, word, *stream)));
}
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
} // namespace
Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA Scan requires an exact shape");
  if (inputs.size() != 1 || outputs.size() != 1)
    return Status::InvalidArgument("CUDA Scan binding counts differ");
  HYLE_ASSIGN_OR_RETURN(
      const auto layout,
      select(request.operation, request.inputs, request.outputs));
  inputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
               .minimum_alignment = request.inputs[0].required_alignment()};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment = 4};
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count =
                   layout.rows == 0
                       ? std::optional<std::uint32_t>{}
                       : std::optional<
                             std::uint32_t>{request.inputs[0].data_type() ==
                                                    DataType::kInt32
                                                ? 2U
                                                : 1U}},
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
  native::ScanProgram program;
  const auto configured =
      native::scan_prepare(request.inputs[0].data_type() == DataType::kInt32,
                           request.operation.get_if<Scan>()->exclusive,
                           request.inputs[0].is_dense(), *layout, &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA Scan program allocation failed"));
  auto *prepared = ::new (storage)
      Prepared{primary, program, request.inputs[0].required_alignment(),
               request.inputs[0] == request.outputs[0]};
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::scanning
