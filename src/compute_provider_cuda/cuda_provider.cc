#include "hyle/compute/providers/cuda.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <new>
#include <utility>

#include "concat_provider.h"
#include "gather_provider.h"
#include "native_context.h"
#include "native_kernel.h"
#include "norm_provider.h"
#include "provider_internal.h"
#include "reduction_provider.h"
#include "scan_provider.h"
#include "scatter_provider.h"
#include "tensor_provider.h"
#include "topk_provider.h"

namespace hyle::compute::cuda_native {
namespace {

struct Prepared final {
  CUcontext primary = nullptr;
  native::Program program{};
  std::size_t alignment = 1;
};

Status create_context(const Provider &, Device device, void *backend_state,
                      mem::AllocatorRef, void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA kernel context output is null");
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("CUDA kernels require the CUDA backend");
  const auto primary = cuda::detail::borrow_primary_context(backend_state);
  if (primary == nullptr)
    return Status::FailedPrecondition("CUDA kernel context is invalid");
  // Compute retains the owning backend; this provider has no second context.
  *output = primary;
  return Status::Ok();
}

void destroy_context(void *, mem::AllocatorRef) noexcept {}

template <bool Sglang>
Status select_shape(Operation operation, std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) noexcept {
  const auto *gated = operation.get_if<GatedActivation>();
  if (gated == nullptr ||
      (gated->activation != GatedActivationKind::kSilu &&
       gated->activation != GatedActivationKind::kGelu &&
       gated->activation != GatedActivationKind::kGeluTanh) ||
      (gated->inputs != GatedInputKind::kPackedHalves &&
       gated->inputs != GatedInputKind::kSeparate) ||
      inputs.size() != (gated->inputs == GatedInputKind::kSeparate ? 2U : 1U) ||
      outputs.size() != 1)
    return Status::Unimplemented(
        "CUDA kernels require a supported gated activation");
  const auto &output = outputs[0];
  if (!output.valid() || !output.is_plain() || !output.is_dense() ||
      output.rank() == 0 ||
      (output.data_type() != DataType::kFloat16 &&
       output.data_type() != DataType::kFloat32 &&
       output.data_type() != DataType::kBFloat16))
    return Status::Unimplemented(
        "CUDA gated activation requires dense FP16/BF16/FP32");
  HYLE_ASSIGN_OR_RETURN(const auto count, output.element_count());
  if (count > INT32_MAX)
    return Status::Unimplemented(
        "CUDA gated activation requires at most INT32_MAX elements");
  for (const auto &input : inputs) {
    if (!input.valid() || !input.is_plain() || !input.is_dense() ||
        input.rank() != output.rank() ||
        input.data_type() != output.data_type())
      return Status::Unimplemented(
          "CUDA gated activation requires matching dense inputs");
    const auto in = input.dimensions();
    const auto out = output.dimensions();
    const auto multiplier =
        gated->inputs == GatedInputKind::kPackedHalves ? 2 : 1;
    if (!std::equal(in.begin(), in.end() - 1, out.begin()) ||
        in.back() != out.back() * multiplier)
      return Status::InvalidArgument(
          "CUDA gated activation shapes do not agree");
  }
  if constexpr (Sglang) {
    const auto width = output.data_type() == DataType::kFloat32 ? 4 : 8;
    if (gated->inputs != GatedInputKind::kPackedHalves ||
        output.dimensions().back() % width != 0)
      return Status::Unimplemented(
          "SGLang gated activation requires packed complete 16-byte vectors");
  }
  return Status::Ok();
}

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
Status propose(void *, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (proposal == nullptr)
    return Status::InvalidArgument("CUDA kernel proposal output is null");
  *proposal = {};
  if constexpr (!Sglang) {
    if (request.operation.kind() == OperationKind::kScan)
      return scanning::propose(request, inputs, outputs, proposal);
    if (request.operation.kind() == OperationKind::kScatter)
      return scattering::propose(request, inputs, outputs, proposal);
    if (request.operation.kind() == OperationKind::kGather)
      return gathering::propose(request, inputs, outputs, proposal);
    if (request.operation.kind() == OperationKind::kConcat)
      return concatenation::propose(request, inputs, outputs, proposal);
    if (request.operation.kind() == OperationKind::kTopK)
      return selection::propose(request, inputs, outputs, proposal);
    if (request.operation.kind() == OperationKind::kReduce ||
        request.operation.kind() == OperationKind::kSoftmax)
      return reduction::propose(request, inputs, outputs, proposal);
    if (request.operation.kind() == OperationKind::kRmsNorm ||
        request.operation.kind() == OperationKind::kLayerNorm)
      return normalization::propose(request, inputs, outputs, proposal);
    if (request.operation.kind() != OperationKind::kGatedActivation)
      return tensor::propose(request, inputs, outputs, proposal);
  }
  if (!request.shape.exact())
    return Status::Unimplemented(
        "CUDA gated activation requires an exact shape");
  if (inputs.size() != request.inputs.size() ||
      outputs.size() != request.outputs.size())
    return Status::InvalidArgument(
        "CUDA kernel proposal binding counts differ");
  HYLE_RETURN_IF_ERROR(
      select_shape<Sglang>(request.operation, request.inputs, request.outputs));
  const detail::ResourceBindingRequirement tensor{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment =
          Sglang ? 16 : data_type_size(request.outputs[0].data_type()),
  };
  std::fill(inputs.begin(), inputs.end(), tensor);
  outputs[0] = tensor;
  *proposal = detail::ProviderOperationProposal{
      .route = Sglang ? 2U : 1U,
      .runtime = kRuntime,
      .cost = {.workspace = WorkspaceRequirement{},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = *request.outputs[0].element_count() == 0
                                     ? std::optional<std::uint32_t>{}
                                     : std::optional<std::uint32_t>{1}},
      .profile_key = ProviderRouteKey{Sglang ? 2U : 1U},
  };
  return Status::Ok();
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared == nullptr)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}

Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (output == nullptr)
    return Status::InvalidArgument("CUDA kernel invocation output is null");
  *output =
      nullptr; // No mutable native state; common admission still owns a slot.
  return Status::Ok();
}

void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}

template <bool Sglang>
detail::EnqueueResult submit(void *state, void *, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<Prepared *>(state);
  if (prepared.program.elements == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  auto first =
      cuda::detail::const_pointer(inputs[0].storage(), prepared.alignment, 1);
  if (!first)
    return detail::EnqueueResult::rejected(std::move(first).error());
  const void *second = nullptr;
  if (inputs.size() == 2) {
    auto pointer =
        cuda::detail::const_pointer(inputs[1].storage(), prepared.alignment, 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    second = *pointer;
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.alignment, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  // Runtime errors may include a prior asynchronous failure. Preserve the
  // accepted Queue prefix and real backend terminal conservatively.
  if constexpr (Sglang)
    return detail::EnqueueResult::accepted_command(
        cuda::status(native::sglang_launch(prepared.program, *first, second,
                                           *output, *stream)));
  else
    return detail::EnqueueResult::accepted_command(cuda::status(
        native::launch(prepared.program, *first, second, *output, *stream)));
}

template <bool Sglang>
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit<Sglang>,
};

template <bool Sglang>
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  if (output == nullptr)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("CUDA kernel prepared output is null"));
  *output = {};
  if constexpr (!Sglang) {
    if (request.route == scanning::kRoute && request.shape.exact())
      return scanning::prepare(state, request, allocator, output);
    if (request.route == scattering::kRoute && request.shape.exact())
      return scattering::prepare(state, request, allocator, output);
    if (request.route == gathering::kRoute && request.shape.exact())
      return gathering::prepare(state, request, allocator, output);
    if (request.route == concatenation::kRoute && request.shape.exact())
      return concatenation::prepare(state, request, allocator, output);
    if (request.route == selection::kRoute && request.shape.exact())
      return selection::prepare(state, request, allocator, output);
    if (request.route == reduction::kRoute && request.shape.exact())
      return reduction::prepare(state, request, allocator, output);
    if (request.route == normalization::kRoute && request.shape.exact())
      return normalization::prepare(state, request, allocator, output);
    if (request.route == tensor::kRoute && request.shape.exact())
      return tensor::prepare(state, request, allocator, output);
  }
  if (request.route != (Sglang ? 2U : 1U) || !request.shape.exact())
    return ProviderPrepareOutcome::error(Status::InvalidArgument(
        "CUDA kernel exact prepare request is invalid"));
  Status status =
      select_shape<Sglang>(request.operation, request.inputs, request.outputs);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  const auto primary = static_cast<CUcontext>(state);
  cuda::detail::ScopedPrimaryContext scope(primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  const auto &description = request.outputs[0];
  native::Program program;
  const auto type = description.data_type() == DataType::kFloat32
                        ? native::ElementType::kFloat32
                    : description.data_type() == DataType::kFloat16
                        ? native::ElementType::kFloat16
                        : native::ElementType::kBFloat16;
  native::Activation activation;
  switch (request.operation.get_if<GatedActivation>()->activation) {
  case GatedActivationKind::kSilu:
    activation = native::Activation::kSilu;
    break;
  case GatedActivationKind::kGelu:
    activation = native::Activation::kGelu;
    break;
  case GatedActivationKind::kGeluTanh:
    activation = native::Activation::kGeluTanh;
    break;
  default:
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("CUDA gated activation kind is invalid"));
  }
  constexpr auto configure = Sglang ? native::sglang_prepare : native::prepare;
  const auto configured = configure(
      type, activation,
      request.operation.get_if<GatedActivation>()->inputs ==
          GatedInputKind::kPackedHalves,
      static_cast<unsigned>(*description.element_count()),
      static_cast<unsigned>(description.dimensions().back()), &program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CUDA kernel prepared allocation failed"));
  auto *prepared = ::new (storage) Prepared{
      primary, program, Sglang ? 16 : data_type_size(description.data_type())};
  *output = detail::ProviderPrepared{
      .vtable = &kPreparedVTable<Sglang>,
      .state = prepared,
      .runtime = kRuntime,
      .persistent_bytes = sizeof(Prepared),
      .invocation_private_bytes = 0,
  };
  return ProviderPrepareOutcome::prepared();
}

Status drain_context(void *) noexcept { return Status::Ok(); }
template <bool Sglang>
const detail::ProviderVTable kProviderVTable{
    .identity = {.id = Sglang ? kSglangProviderId : kProviderId,
                 .name = Sglang ? "sglang-cuda" : "hyle-cuda",
                 .version = Sglang ? "06992c92-precise-gated" : "v10-scan"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose<Sglang>,
    .prepare = &prepare<Sglang>,
    .drain_context = &drain_context,
};

} // namespace

Provider provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable<false>, std::uint8_t{0});
}

Provider sglang_provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable<true>, std::uint8_t{0});
}

} // namespace hyle::compute::cuda_native
