#include "row_cache_provider.h"

#include <array>
#include "hyle/compute/model_ops.h"
#include "native_context.h"
#include "row_cache_kernel.h"
#include "../compute_batch/row_cache_map.h"

namespace hyle::compute::cuda_native::row_cache {
namespace {
constexpr ProviderRuntimeProperties kRuntime{
    .mode = ProviderRuntimeMode::kSharedQueue,
    .workspace_owner = ProviderWorkspaceOwner::kCaller,
    .replay_class = ProviderReplayClass::kPrepared,
    .blocks_host = false,
    .uses_private_threads = false,
    .supports_concurrent_invocations = true,
    .capture_compatible = true};
struct Prepared final {
  CUcontext primary;
  native::Program program;
  bool padded = false;
};
struct MapPrepared final {
  CUcontext primary;
  native::MapProgram program;
  bool padded = false;
  std::uint32_t columns = 0;
};
Result<native::MapProgram>
select_map(Operation operation, std::span<const TensorDesc> inputs,
           std::span<const TensorDesc> outputs) noexcept {
  const auto padded = operation.custom_as<detail::RowCachePaddedMap>();
  const auto plain = operation.custom_as<detail::RowCacheMap>();
  const auto *map = padded ? &padded->mapping : plain ? &*plain : nullptr;
  if (!map)
    return std::unexpected(
        Status::Unimplemented("CUDA row cache page map differs"));
  std::array<TensorDesc, 2> inferred;
  const auto count = padded ? 2u : 1u;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(
      operation, inputs, std::span(inferred.data(), count)));
  if (outputs.size() != count)
    return std::unexpected(
        Status::InvalidArgument("CUDA row cache map output count differs"));
  for (unsigned i = 0; i < count; ++i)
    if (outputs[i] != inferred[i])
      return std::unexpected(
          Status::InvalidArgument("CUDA row cache map output differs"));
  return native::MapProgram{std::uint64_t(inputs[0].dimensions()[0]),
                            std::uint64_t(inputs[0].dimensions()[1]),
                            map->query_offset,
                            map->kv_offset,
                            map->page_offset,
                            map->tail_offset,
                            map->page_size,
                            map->max_pages,
                            map->max_requests(),
                            map->max_refs()};
}
Result<native::Program> select(Operation operation,
                               std::span<const TensorDesc> inputs,
                               std::span<const TensorDesc> outputs) noexcept {
  native::Program program;
  RowCacheEncoding encoding;
  if (const auto encode = operation.custom_as<model_ops::RowCacheEncode>()) {
    encoding = encode->encoding;
    program.encode = true;
  } else if (const auto decode =
                 operation.custom_as<model_ops::RowCacheGatherDecode>()) {
    encoding = decode->encoding;
    program.checked = decode->bounds == IndexBoundsPolicy::kChecked;
  } else if (const auto padded =
                 operation.custom_as<model_ops::RowCacheGatherDecodePadded>()) {
    encoding = padded->encoding;
    program.checked = true;
  } else
    return std::unexpected(
        Status::Unimplemented("CUDA row cache operation differs"));
  std::array<TensorDesc, 1> inferred;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(operation, inputs, inferred));
  if (outputs.size() != 1 || outputs[0] != inferred[0])
    return std::unexpected(
        Status::InvalidArgument("CUDA row cache output differs"));
  program.rows = static_cast<std::uint64_t>(inputs[0].dimensions()[0]);
  program.float4 = encoding.format() == RowCacheFormat::kFloat4Block32;
  if (!program.encode) {
    HYLE_ASSIGN_OR_RETURN(program.selected, inputs[1].element_count());
  }
  return program;
}
void destroy(void *state, mem::AllocatorRef allocator) noexcept {
  allocator.destroy(static_cast<Prepared *>(state));
}
AliasSupport alias(void *, std::size_t, std::size_t) noexcept { return {}; }
Status create_invocation(void *, mem::AllocatorRef, mem::BufferView,
                         void **output) noexcept {
  if (!output)
    return Status::InvalidArgument("CUDA row cache invocation output is null");
  *output = nullptr;
  return Status::Ok();
}
void destroy_invocation(void *, mem::AllocatorRef) noexcept {}
detail::EnqueueResult submit(void *state, void *, detail::QueueSequence &sequence,
                             std::span<const ConstTensorView> inputs,
                             std::span<const TensorView> outputs) noexcept {
  const auto &prepared = *static_cast<const Prepared *>(state);
  const auto &program = prepared.program;
  if (program.encode ? program.rows == 0 : program.selected == 0)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(std::move(stream).error());
  const void *source = nullptr;
  // Empty history with nonempty all-padding IDs has no readable bank bytes.
  if (program.rows) {
    auto pointer = cuda::detail::const_pointer(inputs[0].storage(),
                                               program.encode ? 2 : 1, 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    source = *pointer;
  }
  const std::int32_t *ids = nullptr;
  if (!program.encode) {
    auto pointer = cuda::detail::const_pointer(inputs[1].storage(), 4, 1);
    if (!pointer)
      return detail::EnqueueResult::rejected(std::move(pointer).error());
    ids = static_cast<const std::int32_t *>(*pointer);
  }
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(),
                                              program.encode ? 1 : 2, 1);
  if (!output)
    return detail::EnqueueResult::rejected(std::move(output).error());
  cuda::detail::ScopedPrimaryContext scope(prepared.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  unsigned *word = nullptr;
  if (program.encode || program.checked) {
    auto initialized = cuda::detail::acquire_device_status(sequence, &word);
    if (!initialized.status.ok())
      return initialized;
  }
  // acquire_device_status may already enqueue a reset. A later launch failure
  // is therefore an accepted prefix, retained by the original Queue completion.
  return detail::EnqueueResult::accepted_command(cuda::status(
      prepared.padded
          ? native::launch_padded(program, source, ids, *output, word, *stream)
          : native::launch(program, source, ids, *output, word, *stream)));
}
const detail::ProviderPreparedVTable kPrepared{
    .destroy_prepared = &destroy,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit};
void destroy_map(void *state, mem::AllocatorRef allocator) noexcept {
  allocator.destroy(static_cast<MapPrepared *>(state));
}
detail::EnqueueResult submit_map(void *state, void *, detail::QueueSequence &sequence,
                                 std::span<const ConstTensorView> inputs,
                                 std::span<const TensorView> outputs) noexcept {
  const auto &p = *static_cast<const MapPrepared *>(state);
  if (!p.program.rows || !p.program.selected)
    return detail::EnqueueResult::no_op();
  auto stream = cuda::stream(sequence);
  if (!stream)
    return detail::EnqueueResult::rejected(stream.error());
  auto ids = cuda::detail::const_pointer(inputs[0].storage(), 4, 1);
  if (!ids)
    return detail::EnqueueResult::rejected(ids.error());
  const std::int32_t *ranges = nullptr;
  float *mask = nullptr;
  if (p.padded) {
    auto r = cuda::detail::const_pointer(inputs[1].storage(), 4, 1);
    if (!r)
      return detail::EnqueueResult::rejected(r.error());
    ranges = static_cast<const std::int32_t *>(*r);
    auto m = cuda::detail::mutable_pointer(outputs[1].storage(), 4, 1);
    if (!m)
      return detail::EnqueueResult::rejected(m.error());
    mask = static_cast<float *>(*m);
  }
  auto metadata =
      cuda::detail::const_pointer(inputs[p.padded ? 2 : 1].storage(), 8, 1);
  if (!metadata)
    return detail::EnqueueResult::rejected(metadata.error());
  auto output = cuda::detail::mutable_pointer(outputs[0].storage(), 4, 1);
  if (!output)
    return detail::EnqueueResult::rejected(output.error());
  cuda::detail::ScopedPrimaryContext scope(p.primary);
  if (!scope)
    return detail::EnqueueResult::rejected(scope.status());
  return detail::EnqueueResult::accepted_command(cuda::status(
      p.padded
          ? native::launch_padded_map(
                p.program, p.columns, static_cast<const std::int32_t *>(*ids),
                ranges, *metadata, static_cast<std::int32_t *>(*output), mask,
                *stream)
          : native::launch_map(
                p.program, static_cast<const std::int32_t *>(*ids), *metadata,
                static_cast<std::int32_t *>(*output), *stream)));
}
const detail::ProviderPreparedVTable kMapPrepared{
    .destroy_prepared = &destroy_map,
    .alias_support = &alias,
    .create_invocation = &create_invocation,
    .destroy_invocation = &destroy_invocation,
    .shared_submit = &submit_map};
} // namespace
bool inspect_access(const detail::ProviderPrepared &prepared,
                    detail::OperationPreparedAccess &access) noexcept {
  if (!prepared.state) return false;
  access = {};
  if (prepared.vtable == &kMapPrepared) return true;
  if (prepared.vtable != &kPrepared) return false;
  const auto &p = *static_cast<const Prepared *>(prepared.state);
  if (p.program.encode || !p.program.checked) return false;
  access.projected_input = 0;
  access.allows_empty_footprint = p.padded || p.program.selected == 0;
  return true;
}
Status propose(const detail::ProviderOperationRequest &request,
               std::span<detail::ResourceBindingRequirement> inputs,
               std::span<detail::ResourceBindingRequirement> outputs,
               detail::ProviderOperationProposal *proposal) noexcept {
  if (!request.shape.exact())
    return Status::Unimplemented("CUDA row cache needs exact shapes");
  const auto padded_map =
      request.operation.custom_as<detail::RowCachePaddedMap>();
  if (request.operation.custom_as<detail::RowCacheMap>() || padded_map) {
    HYLE_ASSIGN_OR_RETURN(
        const auto map,
        select_map(request.operation, request.inputs, request.outputs));
    if (!proposal || inputs.size() != (padded_map ? 3u : 2u) ||
        outputs.size() != (padded_map ? 2u : 1u))
      return Status::InvalidArgument(
          "CUDA row cache map binding counts differ");
    inputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                 .minimum_alignment = 4};
    inputs[1] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                 .minimum_alignment = padded_map ? 4u : 8u};
    if (padded_map)
      inputs[2] = {.required_properties =
                       mem::MemoryProperty::kDeviceAccessible,
                   .minimum_alignment = 8};
    for (auto &binding : outputs)
      binding = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                 .minimum_alignment = 4};
    *proposal = {
        .route = kRoute,
        .runtime = kRuntime,
        .cost = {.workspace = WorkspaceRequirement{},
                 .persistent_bytes = sizeof(MapPrepared),
                 .dispatch_count = map.rows && map.selected
                                       ? std::optional<std::uint32_t>{1}
                                       : std::nullopt},
        .implementation = ProviderImplementationId{kRoute}};
    return Status::Ok();
  }
  HYLE_ASSIGN_OR_RETURN(
      const auto program,
      select(request.operation, request.inputs, request.outputs));
  if (!proposal || inputs.size() != request.inputs.size() ||
      outputs.size() != 1)
    return Status::InvalidArgument("CUDA row cache binding counts differ");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    inputs[i] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                 .minimum_alignment = i == 1           ? 4U
                                      : program.encode ? 2U
                                                       : 1U};
  outputs[0] = {.required_properties = mem::MemoryProperty::kDeviceAccessible,
                .minimum_alignment = program.encode ? 1U : 2U};
  const bool empty = program.encode ? !program.rows : !program.selected;
  *proposal = {
      .route = kRoute,
      .runtime = kRuntime,
      .cost =
          {.workspace = WorkspaceRequirement{},
           .persistent_bytes = sizeof(Prepared),
           .copy_bytes = 0,
           .repack_bytes =
               program.encode ? *request.outputs[0].required_bytes() : 0,
           .synchronization_count = 0,
           // A fresh sequence also resets its status word. A later operation
           // in that same sequence may share the already initialized word.
           .dispatch_count =
               empty ? std::nullopt
                     : std::optional<std::uint32_t>{program.encode ||
                                                            program.checked
                                                        ? 3U
                                                        : 1U}},
      .implementation = ProviderImplementationId{kRoute}};
  return Status::Ok();
}
ProviderPrepareOutcome prepare(void *state,
                               const detail::ProviderPrepareRequest &request,
                               mem::AllocatorRef allocator,
                               detail::ProviderPrepared *output) noexcept {
  if (!output)
    return ProviderPrepareOutcome::error(
        Status::InvalidArgument("CUDA row cache prepared output is null"));
  *output = {};
  const auto padded_map =
      request.operation.custom_as<detail::RowCachePaddedMap>();
  if (request.operation.custom_as<detail::RowCacheMap>() || padded_map) {
    auto program =
        select_map(request.operation, request.inputs, request.outputs);
    if (!program)
      return ProviderPrepareOutcome::error(program.error());
    if (padded_map && program->rows && program->selected) {
      cuda::detail::ScopedPrimaryContext scope(static_cast<CUcontext>(state));
      if (!scope)
        return ProviderPrepareOutcome::error(scope.status());
      const auto configured = native::prepare_padded_map();
      if (configured != cudaSuccess)
        return ProviderPrepareOutcome::error(cuda::status(configured));
    }
    auto *p = allocator.construct<MapPrepared>(
        static_cast<CUcontext>(state), *program, padded_map.has_value(),
        padded_map ? padded_map->score_columns : 0);
    if (!p)
      return ProviderPrepareOutcome::error(
          Status::ResourceExhausted("CUDA row cache map metadata"));
    *output = {.vtable = &kMapPrepared,
               .state = p,
               .workspace = {},
               .runtime = kRuntime,
               .persistent_bytes = sizeof(MapPrepared),
               .invocation_private_bytes = 0};
    return ProviderPrepareOutcome::prepared();
  }
  auto program = select(request.operation, request.inputs, request.outputs);
  if (!program)
    return ProviderPrepareOutcome::error(std::move(program).error());
  cuda::detail::ScopedPrimaryContext scope(static_cast<CUcontext>(state));
  if (!scope)
    return ProviderPrepareOutcome::error(scope.status());
  const bool padded =
      request.operation.custom_as<model_ops::RowCacheGatherDecodePadded>()
          .has_value();
  const auto configured =
      padded ? native::prepare_padded(*program) : native::prepare(*program);
  if (configured != cudaSuccess)
    return ProviderPrepareOutcome::error(cuda::status(configured));
  auto *prepared = allocator.construct<Prepared>(static_cast<CUcontext>(state),
                                                 *program, padded);
  if (!prepared)
    return ProviderPrepareOutcome::error(
        Status::ResourceExhausted("CUDA row cache program allocation"));
  *output = {.vtable = &kPrepared,
             .state = prepared,
             .workspace = {},
             .runtime = kRuntime,
             .persistent_bytes = sizeof(Prepared),
             .invocation_private_bytes = 0};
  return ProviderPrepareOutcome::prepared();
}
} // namespace hyle::compute::cuda_native::row_cache
