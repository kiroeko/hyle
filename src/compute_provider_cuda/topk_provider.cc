#include "topk_provider.h"

#include <climits>
#include <new>
#include <utility>

#include "native_context.h"
#include "topk_kernel.h"

namespace hyle::compute::cuda_native::selection {
namespace {
struct Prepared final {
  CUcontext primary = nullptr;
  native::TopKProgram program{};
  std::size_t input_alignment = 1;
  std::size_t value_alignment = 1;
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
Result<native::TopKLayout>
select(Operation operation, std::span<const TensorDesc> inputs,
       std::span<const TensorDesc> outputs) noexcept {
  const auto *top = operation.get_if<TopK>();
  if (top == nullptr || inputs.size() != 1 || outputs.size() != 2)
    return std::unexpected(Status::Unimplemented(
        "CUDA selection requires TopK values and indices"));
  const auto &input = inputs[0];
  const auto type = input.data_type();
  if (!input.is_plain() || input.rank() == 0 ||
      (type != DataType::kFloat32 && type != DataType::kFloat16 &&
       type != DataType::kBFloat16) ||
      !outputs[0].is_plain() || !outputs[0].is_dense() ||
      outputs[0].data_type() != type || !outputs[1].is_plain() ||
      !outputs[1].is_dense() || outputs[1].data_type() != DataType::kInt32)
    return std::unexpected(
        Status::Unimplemented("CUDA TopK requires plain FP32/FP16/BF16 and "
                              "dense values/I32 indices"));
  const auto axis =
      top->axis < 0 ? top->axis + static_cast<int>(input.rank()) : top->axis;
  const auto width = input.dimensions()[axis];
  if (top->count > 1 && width > 4096)
    return std::unexpected(
        Status::Unimplemented("CUDA TopK k>1 requires a selected dimension "
                              "<=4096; k=1 supports longer rows"));
  HYLE_ASSIGN_OR_RETURN(const auto count, input.element_count());
  HYLE_ASSIGN_OR_RETURN(const auto output_count, outputs[0].element_count());
  if (count > INT32_MAX || output_count > INT32_MAX)
    return std::unexpected(Status::Unimplemented(
        "CUDA TopK requires at most INT32_MAX input/output elements"));
  native::TopKLayout layout{
      .rows = count == 0 ? 0U : static_cast<unsigned>(count / width),
      .width = static_cast<unsigned>(width),
      .count = static_cast<unsigned>(top->count),
      .rank = static_cast<unsigned>(input.rank()),
      .axis = static_cast<unsigned>(axis)};
  if (count != 0) {
    for (unsigned dim = 0; dim < layout.rank; ++dim) {
      layout.dimensions[dim] = static_cast<unsigned>(input.dimensions()[dim]);
      layout.strides[dim] = static_cast<std::size_t>(input.byte_strides()[dim]);
      if (dim > layout.axis)
        layout.inner *= layout.dimensions[dim];
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
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA TopK invocation output is null");
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
  auto input = cuda::detail::const_pointer(inputs[0].storage(),
                                           prepared.input_alignment, 1);
  if (!input)
    return detail::EnqueueResult::rejected(std::move(input).error());
  auto values = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.value_alignment, 1);
  if (!values)
    return detail::EnqueueResult::rejected(std::move(values).error());
  auto indices = cuda::detail::mutable_pointer(outputs[1].storage(),
                                               alignof(std::int32_t), 1);
  if (!indices)
    return detail::EnqueueResult::rejected(std::move(indices).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  return detail::EnqueueResult::accepted_command(
      cuda::status(native::topk_launch(prepared.program, *input, *values,
                                       *indices, *stream)));
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
    return Status::Unimplemented("CUDA TopK requires an exact shape");
  HYLE_ASSIGN_OR_RETURN(
      const auto layout,
      select(request.operation, request.inputs, request.outputs));
  inputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
               .minimum_alignment = request.inputs[0].required_alignment()};
  for (std::size_t index = 0; index < outputs.size(); ++index)
    outputs[index] = {.required_properties =
                          mem::MemoryProperty::kDeviceAccessible,
                      .minimum_alignment =
                          data_type_size(request.outputs[index].data_type())};
  *proposal = {.route = kRoute,
               .runtime = kRuntime,
               .cost = {.workspace = WorkspaceRequirement{},
                        .persistent_bytes = sizeof(Prepared),
                        .copy_bytes = 0,
                        .repack_bytes = 0,
                        .synchronization_count = 0,
                        .dispatch_count =
                            layout.rows == 0 ? std::optional<std::uint32_t>{}
                                             : std::optional<std::uint32_t>{1}},
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
  const auto type = request.inputs[0].data_type() == DataType::kFloat16
                        ? native::ElementType::kFloat16
                    : request.inputs[0].data_type() == DataType::kBFloat16
                        ? native::ElementType::kBFloat16
                        : native::ElementType::kFloat32;
  native::TopKProgram program;
  const auto configured = native::topk_prepare(
      type, request.operation.get_if<TopK>()->largest,
      request.inputs[0].is_dense() && layout->axis + 1 == layout->rank, *layout,
      &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA TopK program allocation failed"));
  auto *prepared = ::new (storage)
      Prepared{primary, program, request.inputs[0].required_alignment(),
               data_type_size(request.inputs[0].data_type())};
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::selection
