#include "norm_provider.h"

#include <array>
#include <algorithm>
#include <climits>
#include <new>
#include <type_traits>
#include <utility>

#include "native_context.h"
#include "norm_kernel.h"
#include "sglang_norm_kernel.h"

namespace hyle::compute::cuda_native::normalization {
namespace {
template <bool Sglang> struct Prepared final {
  CUcontext primary = nullptr;
  std::conditional_t<Sglang, native::SglangNormProgram, native::NormProgram>
      program{};
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

template <bool Sglang>
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
  if constexpr (Sglang) {
    if (rms == nullptr || inputs.size() != 2 ||
        inputs[0].data_type() == DataType::kFloat32 ||
        inputs[1].data_type() != inputs[0].data_type() || hidden > 16384 ||
        !native::sglang_norm_width(static_cast<unsigned>(hidden)))
      return std::unexpected(Status::Unimplemented(
          "SGLang RMSNorm requires matching FP16/BF16 weight and a supported "
          "warp/CTA width"));
  }
  // An empty tensor has no native rows, even if another dimension is huge.
  return Shape{count == 0 ? 0U : static_cast<unsigned>(count / hidden),
               count == 0 && !Sglang ? 0U : static_cast<unsigned>(hidden),
               rms != nullptr ? rms->epsilon : layer->epsilon,
               layer != nullptr};
}
template <bool Sglang>
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared<Sglang> *>(state);
  if (prepared == nullptr)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared<Sglang>),
                       alignof(Prepared<Sglang>));
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

template <bool Sglang>
detail::EnqueueResult submit(void *state, void *, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared<Sglang> *>(state);
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
  if constexpr (Sglang)
    return detail::EnqueueResult::accepted_command(
        cuda::status(native::sglang_norm_launch(
            prepared.program, pointers[0], pointers[1], *output, *stream)));
  else
    return detail::EnqueueResult::accepted_command(cuda::status(
        native::norm_launch(prepared.program, pointers[0], pointers[1],
                            pointers[2], *output, *stream)));
}
template <bool Sglang>
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy<Sglang>,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit<Sglang>,
};
} // namespace

template <bool Sglang>
Status propose_impl(const detail::ProviderOperationRequest &request,
                    std::span<detail::ResourceBindingRequirement> inputs,
                    std::span<detail::ResourceBindingRequirement> outputs,
                    detail::ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA norm requires an exact shape");
  HYLE_ASSIGN_OR_RETURN(
      const auto shape,
      select<Sglang>(request.operation, request.inputs, request.outputs));
  for (std::size_t index = 0; index < inputs.size(); ++index)
    inputs[index] = {
        .required_properties = mem::MemoryProperty::kDeviceAccessible,
        .minimum_alignment =
            Sglang ? 16 : data_type_size(request.inputs[index].data_type())};
  outputs[0] = inputs[0];
  *proposal = {.route = kRoute,
               .runtime = kRuntime,
               .cost = {.workspace = WorkspaceRequirement{},
                        .persistent_bytes = sizeof(Prepared<Sglang>),
                        .copy_bytes = 0,
                        .repack_bytes = 0,
                        .synchronization_count = 0,
                        .dispatch_count =
                            shape.rows == 0 ? std::optional<std::uint32_t>{}
                                            : std::optional<std::uint32_t>{1}},
               .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}

template <bool Sglang>
ProviderPrepareOutcome
prepare_impl(void *state, const detail::ProviderPrepareRequest &request,
             mem::AllocatorRef allocator,
             detail::ProviderPrepared *output) noexcept {
  *output = {};
  auto shape =
      select<Sglang>(request.operation, request.inputs, request.outputs);
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
  decltype(Prepared<Sglang>::program) program;
  const auto configured = [&] {
    if constexpr (Sglang)
      return native::sglang_norm_prepare(type, shape->rows, shape->hidden,
                                         shape->epsilon, &program);
    else
      return native::norm_prepare(
          type, shape->layer,
          request.inputs.size() > 1 &&
              request.inputs[1].data_type() == DataType::kFloat32,
          request.inputs.size() > 2 &&
              request.inputs[2].data_type() == DataType::kFloat32,
          shape->rows, shape->hidden, shape->epsilon, &program);
  }();
  if (configured == cudaErrorNotSupported)
    return ProviderPrepareOutcome::unsupported(cuda::status(configured));
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage =
      allocator.allocate(sizeof(Prepared<Sglang>), alignof(Prepared<Sglang>));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA norm program allocation failed"));
  auto *prepared =
      ::new (storage) Prepared<Sglang>{.primary = primary, .program = program};
  for (std::size_t index = 0; index < request.inputs.size(); ++index)
    prepared->alignment[index] =
        Sglang ? 16 : data_type_size(request.inputs[index].data_type());
  *output = {.vtable = &kPreparedVTable<Sglang>,
             .state = prepared,
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared<Sglang>),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}

Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  return propose_impl<false>(request, inputs, outputs, proposal);
}
Status sglang_propose(const detail::ProviderOperationRequest &request,
                      std::span<detail::ResourceBindingRequirement> inputs,
                      std::span<detail::ResourceBindingRequirement> outputs,
                      detail::ProviderOperationProposal *proposal) noexcept {
  return propose_impl<true>(request, inputs, outputs, proposal);
}
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  return prepare_impl<false>(state, request, allocator, output);
}
void require_alignment(detail::ProviderPrepared &artifact, std::size_t alignment) noexcept {
  auto &prepared = *static_cast<Prepared<false> *>(artifact.state);
  for (auto &value : prepared.alignment)
    value = (std::max)(value, alignment);
}
ProviderPrepareOutcome
sglang_prepare(void *state, const detail::ProviderPrepareRequest &request,
               mem::AllocatorRef allocator,
               detail::ProviderPrepared *output) noexcept {
  return prepare_impl<true>(state, request, allocator, output);
}
} // namespace hyle::compute::cuda_native::normalization
