#include "hyle/compute/providers/cuda.h"

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <new>

#include "collectives.h"
#include "native_context.h"
#include "provider_internal.h"

namespace hyle::compute::cuda_native {
namespace {
struct Context final {
  CUcontext primary = nullptr;
  bool qualified = false;
};
struct Prepared final {
  CUcontext primary = nullptr;
  collectives::Program program;
  std::size_t output_alignment = 1;
};
struct Invocation final {
  void *workspace = nullptr;
};

ProviderRuntimeProperties
runtime(const collectives::Program &program) noexcept {
  return {.mode = ProviderRuntimeMode::kSharedQueue,
          .workspace_owner = ProviderWorkspaceOwner::kCaller,
          .replay_class = ProviderReplayClass::kManaged,
          .blocks_host =
              (program.kind == collectives::Kind::kScan &&
               program.type == collectives::Type::kInt32) ||
              (program.kind == collectives::Kind::kGather && program.checked),
          .uses_private_threads = false,
          .supports_concurrent_invocations = true,
          .capture_compatible = false};
}

Status describe(Operation operation, std::span<const TensorDesc> inputs,
                std::span<const TensorDesc> outputs,
                collectives::Program *program) noexcept {
  *program = {};
  const auto *scan = operation.get_if<Scan>();
  const auto *topk = operation.get_if<TopK>();
  const auto *gather = operation.get_if<Gather>();
  if ((!scan && !topk && !gather) || inputs.size() != (gather ? 2u : 1u) ||
      outputs.size() != (topk ? 2u : 1u))
    return Status::Unimplemented("CCCL supports Scan, TopK and Gather");
  const auto &source = inputs[0];
  if (!source.valid() || !source.is_plain() || source.rank() == 0)
    return Status::Unimplemented("CCCL requires plain rank-positive input");
  std::array<TensorDesc, 2> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(
      operation, inputs, std::span(inferred).first(outputs.size())));
  if (!std::ranges::equal(outputs, std::span(inferred).first(outputs.size())))
    return Status::InvalidArgument(
        "CCCL output descriptions differ from inference");
  if (gather) {
    const auto &indices = inputs[1];
    const auto count = outputs[0].element_count();
    const auto index_count = indices.element_count();
    const auto width = data_type_size(source.data_type());
    if (!indices.is_plain() || !count || !index_count || *count > INT32_MAX ||
        *index_count > INT32_MAX ||
        (width != 1 && width != 2 && width != 4 && width != 8))
      return Status::Unimplemented(
          "CCCL Gather type or index capacity exceeded");
    for (const auto &description : inputs) {
      const auto bytes = description.required_bytes();
      if (!bytes || *bytes > static_cast<std::size_t>(
                                 std::numeric_limits<std::ptrdiff_t>::max()))
        return Status::Unimplemented("CCCL Gather storage span exceeded");
    }
    program->kind = collectives::Kind::kGather;
    program->count = *count;
    program->element_bytes = width;
    program->checked = gather->bounds == IndexBoundsPolicy::kChecked;
    program->rank = static_cast<std::uint32_t>(source.rank());
    program->dense = source.is_dense();
    std::ranges::copy(source.dimensions(), program->dimensions);
    std::ranges::copy(source.byte_strides(), program->strides);
    const auto axis = static_cast<std::size_t>(
        gather->axis < 0
            ? gather->axis + static_cast<std::int32_t>(source.rank())
            : gather->axis);
    program->width = static_cast<std::size_t>(source.dimensions()[axis]);
    program->index_count = *index_count;
    program->index_i64 = indices.data_type() == DataType::kInt64;
    program->index_rank = static_cast<std::uint32_t>(indices.rank());
    std::ranges::copy(indices.dimensions(), program->index_dimensions);
    std::ranges::copy(indices.byte_strides(), program->index_strides);
    if (*count != 0)
      for (std::size_t i = axis + 1; i < source.rank(); ++i)
        program->inner *= static_cast<std::size_t>(source.dimensions()[i]);
    return Status::Ok();
  }
  switch (source.data_type()) {
  case DataType::kFloat32:
    program->type = collectives::Type::kFloat32;
    break;
  case DataType::kInt32:
    program->type = collectives::Type::kInt32;
    break;
  case DataType::kFloat16:
    program->type = collectives::Type::kFloat16;
    break;
  case DataType::kBFloat16:
    program->type = collectives::Type::kBFloat16;
    break;
  default:
    return Status::Unimplemented("CCCL scalar type is not qualified");
  }
  const auto count = source.element_count();
  const auto bytes = source.required_bytes();
  if (!count || !bytes || *count > INT32_MAX ||
      *bytes >
          static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    return Status::Unimplemented("CCCL index or storage capacity exceeded");
  program->count = *count;
  program->rank = static_cast<std::uint32_t>(source.rank());
  program->dense = source.is_dense();
  std::ranges::copy(source.dimensions(), program->dimensions);
  std::ranges::copy(source.byte_strides(), program->strides);
  const auto requested = scan ? scan->axis : topk->axis;
  const auto axis = static_cast<std::size_t>(
      requested < 0 ? requested + static_cast<std::int32_t>(source.rank())
                    : requested);
  program->width = static_cast<std::size_t>(source.dimensions()[axis]);
  if (scan) {
    program->kind = collectives::Kind::kScan;
    program->exclusive = scan->exclusive;
  } else {
    program->kind = collectives::Kind::kTopK;
    program->k = topk->count;
    program->largest = topk->largest;
  }
  // Zero outer dimensions can coexist with very large unused extents.
  if (*count != 0)
    for (std::size_t i = axis + 1; i < source.rank(); ++i)
      program->inner *= static_cast<std::size_t>(source.dimensions()[i]);
  return Status::Ok();
}

Status create_context(const Provider &, Device device, void *backend,
                      mem::AllocatorRef allocator, void **output) noexcept {
  *output = nullptr;
  if (device.backend_id() != cuda::kBackendId)
    return Status::Unimplemented("CCCL requires CUDA");
  const auto primary = cuda::detail::borrow_primary_context(backend);
  const auto *environment = cuda::detail::borrow_environment(backend);
  if (!primary || !environment)
    return Status::FailedPrecondition("CCCL CUDA context is invalid");
  auto *storage = allocator.allocate(sizeof(Context), alignof(Context));
  if (!storage)
    return Status::ResourceExhausted("CCCL context allocation failed");
  *output = ::new (storage) Context{
      primary, collectives::qualified_version() &&
                   environment->build_toolkit_version == 13030 &&
                   environment->runtime_version == 13030 &&
                   environment->sm_major == 8 && environment->sm_minor == 9};
  return Status::Ok();
}
void destroy_context(void *value, mem::AllocatorRef allocator) noexcept {
  auto *context = static_cast<Context *>(value);
  if (!context)
    return;
  context->~Context();
  allocator.deallocate(context, sizeof(Context), alignof(Context));
}

Status propose(void *value, const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  *proposal = {};
  const auto &context = *static_cast<Context *>(value);
  if (!context.qualified || !request.shape.exact())
    return Status::Unimplemented(
        "CCCL requires qualified SDK/SM89 exact shapes");
  collectives::Program program;
  HYLE_RETURN_IF_ERROR(
      describe(request.operation, request.inputs, request.outputs, &program));
  cuda::detail::ScopedPrimaryContext scope(context.primary);
  HYLE_RETURN_IF_ERROR(scope.status());
  HYLE_RETURN_IF_ERROR(cuda::status(collectives::configure(&program)));
  for (auto &input : inputs)
    input = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
             .minimum_alignment = 1};
  for (std::size_t i = 0; i < outputs.size(); ++i)
    outputs[i] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                  .minimum_alignment =
                      data_type_size(request.outputs[i].data_type())};
  const auto properties = runtime(program);
  *proposal = {
      .route = 1,
      .runtime = properties,
      .workspace_resource = {.required_properties =
                                 mem::MemoryProperty::kDeviceAccessible,
                             .minimum_alignment = 256},
      .cost = {.workspace = WorkspaceRequirement{program.workspace_bytes, 256},
               .persistent_bytes = sizeof(Prepared),
               .copy_bytes = std::nullopt,
               .repack_bytes = std::nullopt,
               .synchronization_count = properties.blocks_host ? 1u : 0u,
               .dispatch_count = std::nullopt},
      .profile_key = ProviderRouteKey{1}};
  return Status::Ok();
}
void destroy_prepared(void *value, mem::AllocatorRef allocator) noexcept {
  auto *prepared = static_cast<Prepared *>(value);
  if (!prepared)
    return;
  prepared->~Prepared();
  allocator.deallocate(prepared, sizeof(Prepared), alignof(Prepared));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *value, mem::AllocatorRef allocator,
                         mem::BufferView workspace, void **output) noexcept {
  *output = nullptr;
  const auto &prepared = *static_cast<Prepared *>(value);
  void *pointer = nullptr;
  if (prepared.program.workspace_bytes != 0) {
    if (workspace.size() < prepared.program.workspace_bytes)
      return Status::InvalidArgument("CCCL workspace is too small");
    auto native = cuda::detail::mutable_pointer(workspace, 256, 1);
    if (!native)
      return std::move(native).error();
    pointer = *native;
  }
  auto *storage = allocator.allocate(sizeof(Invocation), alignof(Invocation));
  if (!storage)
    return Status::ResourceExhausted("CCCL invocation allocation failed");
  *output = ::new (storage) Invocation{pointer};
  return Status::Ok();
}
void destroy_invocation(void *value, mem::AllocatorRef allocator) noexcept {
  auto *slot = static_cast<Invocation *>(value);
  if (!slot)
    return;
  slot->~Invocation();
  allocator.deallocate(slot, sizeof(Invocation), alignof(Invocation));
}
detail::EnqueueResult submit(void *value, void *slot, Queue &queue,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<Prepared *>(value);
  if (prepared.program.count == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(queue);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  auto input = cuda::detail::const_pointer(inputs[0].storage(), 1, 1);
  if (!input)
    return detail::EnqueueResult::rejected(std::move(input).error());
  const void *index_input = nullptr;
  if (inputs.size() == 2) {
    auto pointer = cuda::detail::const_pointer(inputs[1].storage(), 1, 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    index_input = *pointer;
  }
  auto values = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              prepared.output_alignment, 1);
  if (!values)
    return detail::EnqueueResult::rejected(std::move(values).error());
  void *indices = nullptr;
  if (outputs.size() == 2) {
    auto pointer = cuda::detail::mutable_pointer(outputs[1].storage(),
                                                 alignof(std::int32_t), 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    indices = *pointer;
  }
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  cudaStreamCaptureStatus capture;
  auto status = cuda::status(cudaStreamIsCapturing(*stream, &capture));
  if (!status.ok())
    return detail::EnqueueResult::rejected(std::move(status));
  if (capture != cudaStreamCaptureStatusNone)
    return detail::EnqueueResult::rejected(Status::FailedPrecondition(
        "CCCL managed execution does not accept external capture"));
  bool overflow = false;
  status = cuda::status(collectives::submit(
      prepared.program, *input, index_input, *values, indices,
      static_cast<Invocation *>(slot)->workspace, *stream, &overflow));
  if (!status.ok())
    (void)cudaStreamSynchronize(*stream);
  else if (overflow)
    status = Status::OutOfRange(
        prepared.program.kind == collectives::Kind::kGather
            ? "CCCL Gather index out of range; output unchanged"
            : "CCCL integer scan prefix exceeds int32; output unchanged");
  return detail::EnqueueResult::accepted_command(std::move(status));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy_prepared,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};

ProviderPrepareOutcome prepare(void *value,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  *output = {};
  const auto &context = *static_cast<Context *>(value);
  if (!context.qualified || request.route != 1 || !request.shape.exact())
    return ProviderPrepareOutcome::unsupported(
        Status::Unimplemented("CCCL prepare domain differs"));
  collectives::Program program;
  auto status =
      describe(request.operation, request.inputs, request.outputs, &program);
  if (!status.ok())
    return ProviderPrepareOutcome::unsupported(std::move(status));
  cuda::detail::ScopedPrimaryContext scope(context.primary);
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  status = cuda::status(collectives::configure(&program));
  if (!status.ok())
    return ProviderPrepareOutcome::error(std::move(status));
  auto *storage = allocator.allocate(sizeof(Prepared), alignof(Prepared));
  if (!storage)
    return ProviderPrepareOutcome::resource_declined(
        Status::ResourceExhausted("CCCL prepared allocation failed"));
  auto *prepared = ::new (storage) Prepared{
      context.primary, program, data_type_size(request.outputs[0].data_type())};
  *output = {.vtable = &kPrepared,
             .state = prepared,
             .workspace = {program.workspace_bytes, 256},
             .runtime = runtime(program),
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = sizeof(Invocation)};
  return ProviderPrepareOutcome::prepared();
}
Status drain_context(void *) noexcept { return Status::Ok(); }
const detail::ProviderVTable kProvider{
    .identity = {.id = kCcclProviderId,
                 .name = "cccl-cuda",
                 .version = collectives::version()},
    .create_context = &create_context,
    .destroy_context = &destroy_context,
    .propose = &propose,
    .prepare = &prepare,
    .drain_context = &drain_context};
} // namespace

Provider cccl_provider() noexcept {
  return detail::ProviderAccess::make(&kProvider, std::uint8_t{0});
}
} // namespace hyle::compute::cuda_native
