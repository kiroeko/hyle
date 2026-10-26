#include "native_slot_store_region.h"

#include "native_region.h"
#include "../compute_route_apple/apple_copy_internal.h"
#include "../graph/slots_region.h"
#include "hyle/core/assert.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/mem/std_allocator.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace hyle::graph::apple::detail {
namespace {
namespace an = compute::apple::detail;
namespace cd = compute::detail;
namespace gd = graph::detail;
namespace sd = graph::slots::detail;
template<class T> using Vector = std::vector<T, mem::StdAllocator<T>>;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged,
    .uses_private_threads = true};
struct Artifact final {
  std::shared_ptr<an::Runtime> runtime;
  sd::StoreRegionFacts facts;
  compute::TensorDesc rows, ids;
  std::uint32_t rows_port = 0;
  std::size_t item_bytes = 0, input_bytes = 0;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef allocator, Artifact &owner)
      : artifact(owner), copies(mem::StdAllocator<an::HostCopyRange>(allocator)) {}
  // This is the sole original lease transaction. Native references and the
  // recording are destroyed first, even when a public Completion is abandoned.
  sd::StoreBinding binding;
  Artifact &artifact;
  Vector<an::HostCopyRange> copies;
  sd::StoreConfiguration configuration;
  compute::ConstTensorView rows;
  compute::ReadLease rows_owner;
  an::InputSource source;
  bool configured = false, bound = false, sealed = false;
  std::shared_ptr<an::CopyRecording> recording;
};
Status open(const Artifact &artifact) noexcept {
  std::lock_guard lock(artifact.runtime->queue_mutex);
  if (!artifact.runtime->accepting)
    return Status::FailedPrecondition("Apple Store original Source is closed");
  return artifact.runtime->failure;
}
bool same_range(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
Status clear_native(Slot &slot) noexcept {
  HYLE_RETURN_IF_ERROR(an::clear_copy_recording(*slot.recording));
  slot.source.reset(); slot.rows_owner.reset(); slot.rows = {};
  slot.bound = false; slot.sealed = false;
  return Status::Ok();
}
Status bind(void *prepared, void *opaque, std::span<const RegionValue> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  const auto &artifact = *static_cast<const Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(opaque);
  slot.sealed = false;
  HYLE_RETURN_IF_ERROR(open(artifact));
  if (!slot.configured || inputs.size() != 2 || !outputs.empty() ||
      inputs[artifact.rows_port].description != artifact.rows ||
      !inputs[artifact.facts.destination_input_port].tensor.valid() ||
      inputs[artifact.facts.destination_input_port].tensor.description() != artifact.ids ||
      !same_range(inputs[artifact.facts.destination_input_port].tensor.storage(),
                  slot.configuration.destination_indices.storage()))
    return Status::InvalidArgument("Apple Store needs its exact configured rows and private IDs");
  HYLE_RETURN_IF_ERROR(slot.binding.validate_configuration(slot.configuration));
  HYLE_RETURN_IF_ERROR(clear_native(slot));
  const auto *transaction = slot.binding.transaction();
  if (!transaction || transaction->destination_count() != slot.copies.size())
    return Status::FailedPrecondition("Apple Store has no current original writers");
  for (std::size_t i = 0; i < slot.copies.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto target, transaction->destinations()[i].view());
    if (target.resource() != artifact.facts.resource || target.size() != artifact.item_bytes)
      return Status::InvalidArgument("Apple Store destination is not its exact original writer");
    slot.copies[i] = {i * artifact.item_bytes, target};
  }
  const auto &input = inputs[artifact.rows_port];
  if (input.reference.valid()) {
    HYLE_ASSIGN_OR_RETURN(slot.source, native_output_source(input.reference, artifact.runtime, artifact.rows));
    HYLE_RETURN_IF_ERROR(an::bind_host_copy_ranges(*slot.recording, slot.source, artifact.rows, slot.copies));
  } else {
    if (!input.tensor.valid() || input.tensor.description() != artifact.rows)
      return Status::InvalidArgument("Apple Store has no concrete or native rows carrier");
    HYLE_ASSIGN_OR_RETURN(auto rows, input.tensor.storage().subview(0, artifact.input_bytes));
    if (input.lease) {
      HYLE_ASSIGN_OR_RETURN(auto owner, input.lease->view());
      if (rows.resource() != owner.resource() || rows.native_handle() != owner.native_handle() ||
          rows.offset() < owner.offset() || rows.offset() - owner.offset() > owner.size() ||
          rows.size() > owner.size() - (rows.offset() - owner.offset()))
        return Status::InvalidArgument("Apple Store lease does not cover its actual rows");
      HYLE_ASSIGN_OR_RETURN(slot.rows_owner, input.lease->subview(rows.offset() - owner.offset(), rows.size()));
    }
    HYLE_RETURN_IF_ERROR(an::bind_host_copy_ranges(*slot.recording, input.tensor, slot.copies,
        std::move(slot.rows_owner)));
    slot.rows = input.tensor;
  }
  slot.bound = true;
  return Status::Ok();
}
const sd::StoreRegionVTable store_table{
    .inspect = [](const ProviderRegionPrepared &prepared, void *opaque) noexcept -> Result<sd::StoreRegionFacts> {
      if (!prepared.state || !opaque || &static_cast<Slot *>(opaque)->artifact != prepared.state)
        return std::unexpected(Status::InvalidArgument("Apple Store original artifact/slot differs"));
      return static_cast<const Artifact *>(prepared.state)->facts;
    },
    .binding = [](void *opaque) noexcept { return &static_cast<Slot *>(opaque)->binding; },
    .configure = [](const ProviderRegionPrepared &prepared, void *opaque,
        const sd::StoreConfiguration &configuration) noexcept -> Status {
      const auto &artifact = *static_cast<const Artifact *>(prepared.state);
      auto &slot = *static_cast<Slot *>(opaque);
      HYLE_RETURN_IF_ERROR(open(artifact));
      HYLE_RETURN_IF_ERROR(an::copy_recording_reuse_gate(*slot.recording));
      HYLE_ASSIGN_OR_RETURN(auto expected, artifact.facts.schema.capacity_description());
      if (!configuration.capacity.valid() || configuration.capacity.description() != expected ||
          configuration.capacity.storage().resource() != artifact.facts.resource ||
          !configuration.destination_indices.valid() ||
          configuration.destination_indices.description() != artifact.ids)
        return Status::InvalidArgument("Apple Store private capacity/IDs configuration differs");
      // Common prepare has not moved the caller's transaction or installed
      // authority yet. Only immutable borrowed descriptors are stored here.
      slot.configuration = configuration; slot.configured = true;
      slot.bound = false; slot.sealed = false;
      return Status::Ok();
    },
    .clear_configuration = [](const ProviderRegionPrepared &, void *opaque) noexcept {
      auto &slot = *static_cast<Slot *>(opaque);
      slot.configuration = {}; slot.configured = false; slot.bound = false; slot.sealed = false;
    }};
const gd::RegionBindingReleaseVTable release_table{
    .validate = [](void *, void *opaque) noexcept {
      return an::copy_recording_retired_gate(*static_cast<Slot *>(opaque)->recording);
    },
    .release = [](void *, void *opaque) noexcept {
      HYLE_FATAL_ASSERT(clear_native(*static_cast<Slot *>(opaque)).ok());
    }};
const gd::RegionOrderedWriteVTable ordered_table{
    .qualify = [](const ProviderRegionPrepared &prepared, void *opaque,
        const cd::CompletionEndpoint *reader) noexcept -> Status {
      if (!prepared.state || !opaque || &static_cast<Slot *>(opaque)->artifact != prepared.state)
        return Status::InvalidArgument("Apple Store ordered artifact/slot differs");
      const auto &artifact = *static_cast<const Artifact *>(prepared.state);
      HYLE_RETURN_IF_ERROR(open(artifact));
      return !reader || imports_region_ordered_endpoint(artifact.runtime, *reader)
          ? Status::Ok() : Status::InvalidArgument("Apple Store reader belongs to another Runtime");
    }};
const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = [](void *prepared, mem::AllocatorRef a, mem::BufferView, void **out) noexcept -> Status {
      *out = nullptr;
      auto &artifact = *static_cast<Artifact *>(prepared);
      auto *slot = a.construct<Slot>(a, artifact);
      if (!slot) return Status::ResourceExhausted("Apple Store invocation metadata");
      *out = slot;
      HYLE_ASSIGN_OR_RETURN(slot->recording, an::prepare_copy_recording_runtime(artifact.runtime,
          static_cast<std::uint32_t>(artifact.facts.destination_count), a,
          artifact.input_bytes, an::CopyRecordingMode::kReplayable));
      try { slot->copies.resize(artifact.facts.destination_count); }
      catch (...) { return Status::ResourceExhausted("Apple Store fixed copy descriptors"); }
      return Status::Ok();
    },
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .delegated_submit = [](void *prepared, void *opaque, std::span<const cd::CompletionDependency> dependencies,
        std::span<const compute::ConstTensorView> inputs, std::span<const compute::TensorView> outputs) noexcept
        -> Result<compute::Completion> {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      auto &slot = *static_cast<Slot *>(opaque);
      if (!slot.sealed || inputs.size() != 2 || !outputs.empty() ||
          !same_range(inputs[artifact.facts.destination_input_port].storage(), slot.configuration.destination_indices.storage()) ||
          (!slot.source.valid() && (!inputs[artifact.rows_port].valid() ||
             !same_range(inputs[artifact.rows_port].storage(), slot.rows.storage()))))
        return std::unexpected(Status::FailedPrecondition("Apple Store ports/transaction are not sealed"));
      for (const auto dependency : dependencies)
        HYLE_RETURN_IF_ERROR(an::qualify_recording_dependency(artifact.runtime, dependency));
      return an::submit_copy_recording(*slot.recording);
    },
    .endpoint = [](void *prepared, void *) noexcept { return region_ordered_endpoint(static_cast<Artifact *>(prepared)->runtime); },
    .can_import_endpoint = [](void *prepared, void *, cd::CompletionEndpoint endpoint) noexcept {
      return imports_region_ordered_endpoint(static_cast<Artifact *>(prepared)->runtime, endpoint);
    },
    .prepare_bindings = &bind,
    .seal_bindings = [](void *, void *opaque) noexcept -> Status {
      auto &slot = *static_cast<Slot *>(opaque); slot.sealed = false;
      if (!slot.configured || !slot.bound)
        return Status::FailedPrecondition("Apple Store has no current native binding");
      HYLE_RETURN_IF_ERROR(slot.binding.validate_configuration(slot.configuration));
      HYLE_RETURN_IF_ERROR(an::copy_recording_reuse_gate(*slot.recording));
      slot.sealed = true;
      return Status::Ok();
    },
    .query_extension = [](void *, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == sd::kStoreBindingExtensionId) return &store_table;
      if (id == gd::kRegionOrderedWriteExtensionId) return &ordered_table;
      if (id == gd::kRegionBindingReleaseExtensionId) return &release_table;
      return nullptr;
    }};
Result<Artifact> describe(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() || request.route != kNativeSlotStoreRoute ||
      request.nodes.size() != 1 || request.inputs.size() != 2 || !request.outputs.empty() ||
      !request.has_unknown_effects || !request.effect_accesses.empty() || !request.effect_bindings.empty())
    return std::unexpected(Status::Unimplemented("Apple Store requires one original zero-output effect"));
  HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(request.nodes[0]));
  const auto store = operation.custom_as<slots::Store>();
  if (!store) return std::unexpected(Status::Unimplemented("not a slot Store"));
  HYLE_ASSIGN_OR_RETURN(auto rows_value, request.graph->node_input(request.nodes[0], 0));
  HYLE_ASSIGN_OR_RETURN(auto ids_value, request.graph->node_input(request.nodes[0], 1));
  const auto rows = std::ranges::find(request.inputs, rows_value), ids = std::ranges::find(request.inputs, ids_value);
  if (rows == request.inputs.end() || ids == request.inputs.end() || ids_value.kind() != ValueKind::kInput)
    return std::unexpected(Status::Unimplemented("Apple Store needs original rows/private IDs boundaries"));
  Artifact artifact;
  HYLE_ASSIGN_OR_RETURN(artifact.rows, request.graph->value_description(rows_value));
  HYLE_ASSIGN_OR_RETURN(artifact.ids, request.graph->value_description(ids_value));
  const std::array descriptions{artifact.rows, artifact.ids};
  HYLE_RETURN_IF_ERROR(store->infer_output_descriptions(descriptions, {}));
  HYLE_ASSIGN_OR_RETURN(artifact.input_bytes, artifact.rows.required_bytes());
  const auto count = static_cast<std::size_t>(artifact.rows.dimensions()[0]);
  if (!count) return std::unexpected(Status::Unimplemented("Apple Store requires positive row count"));
  artifact.item_bytes = artifact.input_bytes / count;
  if (count > UINT32_MAX || !artifact.item_bytes ||
      artifact.input_bytes > static_cast<std::size_t>(PTRDIFF_MAX))
    return std::unexpected(Status::Unimplemented("Apple Store logical rows exceed native copy limits"));
  artifact.rows_port = static_cast<std::uint32_t>(rows - request.inputs.begin());
  artifact.facts = {.store = request.nodes[0], .schema = store->schema,
      .destination_indices = ids_value, .destination_input_port = static_cast<std::uint32_t>(ids - request.inputs.begin()),
      .destination_count = count};
  return artifact;
}
} // namespace
bool native_slot_store_operation(const compute::Operation &operation) noexcept {
  return operation.custom_as<slots::Store>().has_value();
}
Status propose_native_slot_store(void *, const RegionDiscoveryRequest &request,
    std::span<const std::uint8_t> needed, RegionProposalWriter &writer) noexcept {
  for (std::size_t i = 0; i < request.graph.node_count(); ++i) {
    if (!needed[i] || (!request.allowed_nodes.empty() && !request.allowed_nodes[i])) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    if (!native_slot_store_operation(operation)) continue;
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = nodes, .route = kNativeSlotStoreRoute,
        .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{0x534c4f544150504cULL}, .preference = 100}));
    if (!more) break;
  }
  return Status::Ok();
}
Status native_slot_store_resources(void *, const RegionCompileRequest &request,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *proposal) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto artifact, describe(request)); (void)artifact;
  if (inputs.size() != 2 || !outputs.empty()) return Status::InvalidArgument("Apple Store resource boundary differs");
  const cd::ResourceBindingRequirement resource{
      .required_properties = mem::MemoryProperty::kHostAccessible | mem::MemoryProperty::kDeviceAccessible |
          mem::MemoryProperty::kHostDeviceCoherent, .minimum_alignment = 1};
  std::ranges::fill(inputs, RegionInputRequirement{resource, true});
  *proposal = {.runtime = runtime_properties, .workspace_resource = resource, .preserves_effects = true};
  return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_slot_store(void *context,
    const RegionCompileRequest &request, mem::AllocatorRef allocator, ProviderRegionPrepared *output) noexcept {
  using Outcome = compute::ProviderPrepareOutcome;
  *output = {};
  auto described = describe(request);
  if (!described) return described.error().IsUnimplemented()
      ? Outcome::unsupported(described.error()) : Outcome::error(described.error());
  if (request.input_resources.size() != 2 || !an::is_copy_resource(request.input_resources[0]) ||
      request.input_resources[0] != request.input_resources[1])
    return Outcome::unsupported(Status::Unimplemented("Apple Store needs original shared Metal storage"));
  auto *artifact = allocator.construct<Artifact>(std::move(*described));
  if (!artifact) return Outcome::error(Status::ResourceExhausted("Apple Store artifact metadata"));
  artifact->runtime = source_runtime(context);
  artifact->facts.resource = request.input_resources[0];
  *output = {.vtable = &prepared_table, .state = artifact,
      .properties = {.persistent_bytes = sizeof(Artifact), .copy_bytes = artifact->input_bytes,
          .dispatch_count = 1, .runtime = runtime_properties, .preserves_effects = true,
          .invocation_private_bytes = std::nullopt}};
  return Outcome::prepared();
}
} // namespace hyle::graph::apple::detail
