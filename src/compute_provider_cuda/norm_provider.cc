#include "norm_provider.h"

#include <array>
#include <climits>
#include <new>
#include <utility>

#include "native_context.h"
#include "norm_kernel.h"

namespace hyle::compute::cuda_native::normalization {
namespace {
struct Prepared final {
  CUcontext primary = nullptr;
  native::NormProgram program{};
  std::array<std::size_t, 3> alignment{};
};
struct Shape final {
  unsigned rows = 0;
  unsigned hidden = 0;
  float epsilon = 0;
  bool layer = false;
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

Result<Shape> select(Operation operation, std::span<const TensorDesc> inputs,
                     std::span<const TensorDesc> outputs) noexcept {
  const auto *rms = operation.get_if<RmsNorm>();
  const auto *layer = operation.get_if<LayerNorm>();
  if ((rms == nullptr && layer == nullptr) || inputs.empty() ||
      inputs.size() > 3 || outputs.size() != 1)
    return std::unexpected(Status::Unimplemented(
        "CUDA normalization requires RMSNorm or LayerNorm"));
  const auto accumulation =
      rms != nullptr ? rms->accumulation_type : layer->accumulation_type;
  if (accumulation != DataType::kFloat32)
    return std::unexpected(
        Status::Unimplemented("CUDA normalization requires FP32 accumulation"));
  for (const auto &input : inputs) {
    if (!input.is_plain() || !input.is_dense() ||
        (input.data_type() != DataType::kFloat32 &&
         input.data_type() != DataType::kFloat16 &&
         input.data_type() != DataType::kBFloat16))
      return std::unexpected(
          Status::Unimplemented("CUDA normalization requires dense "
                                "FP32/FP16/BF16 inputs and parameters"));
  }
  if (inputs[0].rank() == 0 || outputs[0] != inputs[0])
    return std::unexpected(Status::Unimplemented(
        "CUDA normalization requires a matching dense output and a last axis"));
  HYLE_ASSIGN_OR_RETURN(const auto count, outputs[0].element_count());
  if (count > INT32_MAX)
    return std::unexpected(Status::Unimplemented(
        "CUDA normalization requires at most INT32_MAX elements"));
  const auto hidden = inputs[0].dimensions().back();
  // An empty tensor has no native rows, even if another dimension is huge.
  return Shape{count == 0 ? 0U : static_cast<unsigned>(count / hidden),
               count == 0 ? 0U : static_cast<unsigned>(hidden),
               rms != nullptr ? rms->epsilon : layer->epsilon,
               layer != nullptr};
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared == nullptr)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t output, std::size_t input) noexcept {
  return {output == 0 && input == 0, 0};
}
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA norm invocation output is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}

detail::EnqueueResult submit(void *state, void *, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  if (prepared.program.rows == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const void *, 3> pointers{};
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    auto pointer = cuda::detail::const_pointer(inputs[index].storage(),
                                               prepared.alignment[index], 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    pointers[index] = *pointer;
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.alignment[0], 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  return detail::EnqueueResult::accepted_command(cuda::status(
      native::norm_launch(prepared.program, pointers[0], pointers[1],
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
    return Status::Unimplemented("CUDA norm requires an exact shape");
  HYLE_ASSIGN_OR_RETURN(
      const auto shape,
      select(request.operation, request.inputs, request.outputs));
  for (std::size_t index = 0; index < inputs.size(); ++index)
    inputs[index] = {
        .required_properties = mem::MemoryProperty::kDeviceAccessible,
        .minimum_alignment = data_type_size(request.inputs[index].data_type())};
  outputs[0] = inputs[0];
  *proposal = {.route = kRoute,
               .runtime = kRuntime,
               .cost = {.workspace = WorkspaceRequirement{},
                        .persistent_bytes = sizeof(Prepared),
                        .copy_bytes = 0,
                        .repack_bytes = 0,
                        .synchronization_count = 0,
                        .dispatch_count =
                            shape.rows == 0 ? std::optional<std::uint32_t>{}
                                            : std::optional<std::uint32_t>{1}},
               .profile_key = ProviderRouteKey{kRoute}};
  return Status::Ok();
}

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto shape = select(request.operation, request.inputs, request.outputs);
  if (!shape)
    return shape.error().IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(shape).error())
               : ProviderPrepareOutcome::error(std::move(shape).error());
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  const auto type = request.inputs[0].data_type() == DataType::kFloat16
                        ? native::ElementType::kFloat16
                    : request.inputs[0].data_type() == DataType::kBFloat16
                        ? native::ElementType::kBFloat16
                        : native::ElementType::kFloat32;
  native::NormProgram program;
  const auto configured = native::norm_prepare(
      type, shape->layer,
      request.inputs.size() > 1 &&
          request.inputs[1].data_type() == DataType::kFloat32,
      request.inputs.size() > 2 &&
          request.inputs[2].data_type() == DataType::kFloat32,
      shape->rows, shape->hidden, shape->epsilon, &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA norm program allocation failed"));
  auto *prepared =
      ::new (storage) Prepared{.primary = primary, .program = program};
  for (std::size_t index = 0; index < request.inputs.size(); ++index)
    prepared->alignment[index] =
        data_type_size(request.inputs[index].data_type());
  *output = {.vtable = &kPreparedVTable,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::normalization
