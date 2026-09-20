#include "hyle/compute/providers/flashinfer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>

#include "native_attention.h"
#include "native_context.h"
#include "provider_internal.h"

namespace hyle::compute::flashinfer {
namespace {

constexpr std::size_t kAlignment = 16;
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kManaged,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = false,
};

struct ProviderContext final {
  CUcontext primary = nullptr;
  std::uint64_t maximum_grid = 0;
};

struct Prepared final {
  CUcontext primary = nullptr;
  native::Attention attention{};
  std::size_t batches = 0;
  std::size_t query_batch_bytes = 0;
  std::size_t kv_batch_bytes = 0;
  std::size_t workspace_bytes = 0;
};

struct Invocation final {
  void *workspace = nullptr;
  bool failed = false;
};

Status create_context(const Provider &, Device device, void *backend_state,
                      mem::AllocatorRef allocator, void **output) noexcept {
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("FlashInfer requires CUDA");
  const auto primary = cuda::detail::borrow_primary_context(backend_state);
  const auto *environment = cuda::detail::borrow_environment(backend_state);
  if (primary == nullptr || environment == nullptr)
    return Status::FailedPrecondition("FlashInfer CUDA context is invalid");
  std::uint64_t maximum_grid = 0;
  if (environment->sm_major == 8 && environment->sm_minor == 9) {
    cuda::detail::ScopedPrimaryContext scope(primary);
    HYLE_RETURN_IF_ERROR(scope.status());
    int multiprocessors = 0;
    int threads_per_multiprocessor = 0;
    HYLE_RETURN_IF_ERROR(cuda::status(
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                               environment->device_ordinal)));
    HYLE_RETURN_IF_ERROR(cuda::status(cudaDeviceGetAttribute(
        &threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor,
        environment->device_ordinal)));
    if (multiprocessors <= 0 || threads_per_multiprocessor < 128)
      return Status::Internal("FlashInfer CUDA residency facts are invalid");
    // Both admitted upstream entry families use at least 128 threads/block.
    maximum_grid = static_cast<std::uint64_t>(multiprocessors) *
                   static_cast<std::uint64_t>(threads_per_multiprocessor / 128);
  }
  void *storage =
      allocator.allocate(sizeof(ProviderContext), alignof(ProviderContext));
  if (storage == nullptr)
    return Status::ResourceExhausted("FlashInfer context allocation failed");
  *output = ::new (storage) ProviderContext{primary, maximum_grid};
  return Status::Ok();
}

void destroy_context(void *state, mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<ProviderContext *>(state);
  if (context == nullptr)
    return;
  context->~ProviderContext();
  allocator.deallocate(context, sizeof(ProviderContext),
                       alignof(ProviderContext));
}

Status describe(const ProviderContext &context, Operation operation,
                std::span<const TensorDesc> inputs,
                std::span<const TensorDesc> outputs,
                Prepared &prepared) noexcept {
  const auto *attention = operation.get_if<ScaledDotProductAttention>();
  if (context.maximum_grid == 0 || attention == nullptr || inputs.size() != 3 ||
      outputs.size() != 1 || !attention->causal ||
      attention->layout != AttentionLayout::kBatchSequenceHeadDepth ||
      attention->mask != MaskKind::kNone ||
      attention->accumulation_type != DataType::kFloat32)
    return Status::Unimplemented(
        "FlashInfer requires the admitted causal SM89 domain");
  const auto type = inputs[0].data_type();
  if (type != DataType::kFloat16 && type != DataType::kBFloat16)
    return Status::Unimplemented("FlashInfer requires FP16/BF16");
  for (const auto &description : inputs) {
    if (!description.valid() || !description.is_plain() ||
        !description.is_dense() || description.rank() != 4 ||
        description.data_type() != type)
      return Status::Unimplemented(
          "FlashInfer requires dense matching BSHD tensors");
  }
  const auto q = inputs[0].dimensions();
  const auto k = inputs[1].dimensions();
  const auto v = inputs[2].dimensions();
  if (q[0] <= 0 || q[1] <= 0 || k[1] < q[1] || q[2] != 8 || k[2] != 2 ||
      q[3] != 256 || k[3] != 256 || v[3] != 256 ||
      q[1] > std::numeric_limits<std::int32_t>::max() / (8 * 256) ||
      k[1] > std::numeric_limits<std::int32_t>::max() / (2 * 256))
    return Status::Unimplemented(
        "FlashInfer head/length domain is not admitted");
  if (q[0] != k[0] || !std::equal(k.begin(), k.end(), v.begin()) ||
      outputs[0] != inputs[0] || !std::isfinite(attention->scale))
    return Status::InvalidArgument(
        "FlashInfer attention descriptions disagree");
  const auto query_length = static_cast<std::uint64_t>(q[1]);
  const auto kv_length = static_cast<std::uint64_t>(k[1]);
  const auto chunks_by_length = (kv_length + 255) / 256;
  // Upstream split-KV uses chunks of at least 256 tokens. For prefill,
  // chunks * q_length * q_heads <= maximum_grid * maximum_CTA_tile_q (128).
  // For decode the bound is maximum_grid * GQA_group_size (4). Take the
  // tighter shape/residency bound; batches reuse scratch in one ordered stream.
  const auto partial_rows =
      (std::min)(chunks_by_length * query_length * 8,
                 context.maximum_grid * (query_length == 1 ? 4 : 128));
  const auto scratch_bytes =
      partial_rows * (256 * sizeof(std::uint16_t) + sizeof(float));
  if (scratch_bytes > std::numeric_limits<std::size_t>::max())
    return Status::OutOfRange("FlashInfer workspace size overflows");
  prepared = {
      .primary = context.primary,
      .attention = {.query_length = static_cast<std::uint32_t>(q[1]),
                    .kv_length = static_cast<std::uint32_t>(k[1]),
                    .scale = attention->scale,
                    .bfloat16 = type == DataType::kBFloat16},
      .batches = static_cast<std::size_t>(q[0]),
      .query_batch_bytes = static_cast<std::size_t>(query_length * 8 * 256 * 2),
      .kv_batch_bytes = static_cast<std::size_t>(kv_length * 2 * 256 * 2),
      .workspace_bytes = static_cast<std::size_t>(scratch_bytes)};
  return Status::Ok();
}

Status propose(void *state, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  *proposal = {};
  if (!request.shape.exact())
    return Status::Unimplemented("FlashInfer requires an exact prepared shape");
  if (inputs.size() != request.inputs.size() ||
      outputs.size() != request.outputs.size())
    return Status::InvalidArgument("FlashInfer proposal binding counts differ");
  Prepared prepared;
  HYLE_RETURN_IF_ERROR(describe(*static_cast<ProviderContext *>(state),
                                request.operation, request.inputs,
                                request.outputs, prepared));
  const detail::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment = kAlignment};
  for (auto &input : inputs)
    input = binding;
  for (auto &output : outputs)
    output = binding;
  *proposal = {
      .route = 1,
      .runtime = kRuntime,
      .workspace_resource = binding,
      .cost = {.workspace =
                   WorkspaceRequirement{prepared.workspace_bytes, kAlignment},
               .persistent_bytes = std::nullopt,
               .copy_bytes = 0,
               .repack_bytes = 0,
               .synchronization_count = 0,
               .dispatch_count = std::nullopt},
      .profile_key = ProviderRouteKey{1}};
  return Status::Ok();
}

void destroy_prepared(void *state, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(state);
  if (prepared == nullptr)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}

Status create_invocation(void *state, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  *output = nullptr;
  const auto &prepared = *static_cast<Prepared *>(state);
  if (workspace.size() < prepared.workspace_bytes)
    return Status::InvalidArgument("FlashInfer workspace is too small");
  auto pointer = cuda::detail::mutable_pointer(workspace, kAlignment, 1);
  if (!pointer)
    return std::move(pointer).error();
  void *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (storage == nullptr)
    return Status::ResourceExhausted("FlashInfer invocation allocation failed");
  *output = ::new (storage) Invocation{*pointer};
  return Status::Ok();
}

void destroy_invocation(void *state, mem::AllocatorRef allocator) noexcept {
  auto *invocation = static_cast<Invocation *>(state);
  if (invocation == nullptr)
    return;
  invocation->~Invocation();
  allocator.deallocate(invocation, sizeof(Invocation), alignof(Invocation));
}

detail::EnqueueResult submit(void *state, void *lane, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<Prepared *>(state);
  auto &invocation = *static_cast<Invocation *>(lane);
  if (invocation.failed)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "FlashInfer invocation failed; create another"));
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  std::array<const std::byte *, 3> pointers{};
  for (std::size_t i = 0; i < pointers.size(); ++i) {
    auto pointer =
        cuda::detail::const_pointer(inputs[i].storage(), kAlignment, 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    pointers[i] = static_cast<const std::byte *>(*pointer);
  }
  auto output =
      cuda::detail::mutable_pointer(outputs[0].storage(), kAlignment, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  Status terminal;
  try {
    for (std::size_t batch = 0; batch < prepared.batches; ++batch) {
      terminal = cuda::status(native::execute(
          prepared.attention, pointers[0] + batch * prepared.query_batch_bytes,
          pointers[1] + batch * prepared.kv_batch_bytes,
          pointers[2] + batch * prepared.kv_batch_bytes,
          static_cast<std::byte *>(*output) +
              batch * prepared.query_batch_bytes,
          invocation.workspace, *stream));
      if (!terminal.ok())
        break;
    }
  } catch (const std::bad_alloc &) {
    terminal = Status::ResourceExhausted("FlashInfer host allocation failed");
  } catch (const std::exception &error) {
    terminal = Status::Internal(error.what());
  }
  invocation.failed = !terminal.ok();
  // A failure may follow a partial native launch. The existing Queue terminal
  // owns the accepted prefix, Invocation and workspace until quiescence.
  return detail::EnqueueResult::accepted_command(std::move(terminal));
}

AliasSupport alias_support(void *, std::size_t, std::size_t) noexcept {
  return {};
}
const detail::ProviderPreparedVTable kPreparedVTable{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias_support,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit,
};

ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  if (request.route != 1 || !request.shape.exact())
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("FlashInfer prepare route differs"));
  Prepared value;
  auto status =
      describe(*static_cast<ProviderContext *>(state), request.operation,
               request.inputs, request.outputs, value);
  if (!status.ok())
    return status.IsUnimplemented()
               ? ProviderPrepareOutcome::unsupported(std::move(status))
               : ProviderPrepareOutcome::error(std::move(status));
  void *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (storage == nullptr)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("FlashInfer artifact allocation failed"));
  *output = {.vtable = &kPreparedVTable,
             .state = ::new (storage) Prepared(value),
             .workspace = {value.workspace_bytes, kAlignment},
             .runtime = kRuntime,
             .persistent_bytes = std::nullopt,
             .invocation_private_bytes = std::nullopt};
  return ProviderPrepareOutcome::prepared();
}

Status drain_context(void *) noexcept { return Status::Ok(); }
const detail::ProviderVTable kProviderVTable{
    .identity = {.id = kProviderId,
                 .name = "flashinfer",
                 .version = "0.6.18+69ff11fc"},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context,
};

} // namespace

Provider provider() noexcept {
  return detail::ProviderAccess::make(&kProviderVTable, std::uint8_t{0});
}

} // namespace hyle::compute::flashinfer
