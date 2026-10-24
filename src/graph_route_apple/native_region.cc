#include "native_region.h"
#include "native_output.h"
#include "native_paged_region.h"
#include "native_paged_append_region.h"
#include "native_ragged_region.h"
#include "native_paged_pending.h"
#include "apple_lowering_internal.h"

#include "../graph/native_analysis.h"
#include "../graph/region_discovery.h"
#include "../graph/artifact_packed_region.h"
#include "../graph/static_fp8_region.h"
#include "../compute_route_apple/apple_carrier.h"
#include "../compute_route_apple/apple_copy_internal.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/graph/detail/region_access.h"
#include "../compute_batch/row_cache_planning.h"

#include <algorithm>
#include <mutex>
#include <new>
#include <optional>
#include <vector>

namespace hyle::graph::apple::detail {
namespace ca = compute::apple;
namespace an = ca::detail;
namespace cd = compute::detail;
namespace bd = batch::detail;
namespace mx = mlx::core;
namespace {
template<class T> using Vector = std::vector<T, mem::StdAllocator<T>>;
const unsigned array_type = 0;
const unsigned owner_endpoint = 0;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged,
    .uses_private_threads = true};
extern const ProviderRegionPreparedVTable prepared_table;

struct RowLayout final {
  bool configured = false, empty = false;
  std::uint32_t requests = 0, page_refs = 0;
};
struct Artifact final {
  explicit Artifact(mem::AllocatorRef allocator)
      : input_ports(mem::StdAllocator<std::size_t>(allocator)),
        output_descriptions(mem::StdAllocator<compute::TensorDesc>(allocator)),
        borrowed_constants(mem::StdAllocator<BorrowedRange>(allocator)),
        row_accesses(mem::StdAllocator<PreparedRowAccess>(allocator)),
        pending_inputs(mem::StdAllocator<ValueId>(allocator)),
        pending_ports(mem::StdAllocator<std::uint32_t>(allocator)),
        pending_accesses(mem::StdAllocator<std::size_t>(allocator)),
        pending_descriptions(mem::StdAllocator<compute::TensorDesc>(allocator)) {}
  std::shared_ptr<an::Runtime> runtime;
  ca::Program program;
  compute::ResourceBudget budget;
  Vector<std::size_t> input_ports;
  Vector<compute::TensorDesc> output_descriptions;
  Vector<BorrowedRange> borrowed_constants;
  PreparedRowAccesses row_accesses;
  Vector<ValueId> pending_inputs;
  Vector<std::uint32_t> pending_ports;
  Vector<std::size_t> pending_accesses;
  Vector<compute::TensorDesc> pending_descriptions;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef selected, Artifact &owner)
      : allocator(selected), artifact(owner), bindings(mem::StdAllocator<an::InputBinding>(selected)),
        projected(mem::StdAllocator<std::optional<mx::array>>(selected)),
        exported(mem::StdAllocator<std::optional<compute::ReadLease>>(selected)),
        borrowed(mem::StdAllocator<BorrowedRange>(selected)),
        owned_inputs(mem::StdAllocator<compute::ReadLease>(selected)), sources(mem::StdAllocator<RegionOutputReference>(selected)),
        row_layouts(mem::StdAllocator<RowLayout>(selected)),
        pending_frames(mem::StdAllocator<compute::PagedArenaPendingRead>(selected)),
        pending_reads(selected) {}
  mem::AllocatorRef allocator;
  Artifact &artifact;
  ca::Invocation invocation;
  cd::ResultStorageOwner result;
  Vector<an::InputBinding> bindings;
  Vector<std::optional<mx::array>> projected;
  Vector<std::optional<compute::ReadLease>> exported;
  Vector<BorrowedRange> borrowed;
  Vector<compute::ReadLease> owned_inputs;
  Vector<RegionOutputReference> sources;
  Vector<RowLayout> row_layouts;
  Vector<compute::PagedArenaPendingRead> pending_frames;
  NativePagePendingSource *pending_source = nullptr; // Borrowed from BatchResources.
  compute::LeaseUsePool pending_uses;
  cd::BatchPageReads pending_reads;
  std::uint32_t pending_capacity = 0;
  bool pending_consumed = false;
  std::mutex output_mutex;
};

Status qualify(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() ||
      (request.route != 1 && request.route != kNativeCtPackedRoute &&
       request.route != kNativeStaticFloat8Route) ||
      request.nodes.empty() || request.has_unknown_effects ||
      !request.effect_accesses.empty())
    return Status::Unimplemented("Apple native region requires a fixed pure boundary");
  for (const auto node : request.nodes) {
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(node));
    const auto traits = compute::operation_traits(operation);
    if (traits.effect != compute::EffectKind::kPure || traits.resource_count ||
        (traits.alias == compute::AliasPolicy::kMustAliasInput0 && !operation.get_if<compute::View>()))
      return Status::Unimplemented("Apple native region has unsupported effects/aliasing");
  }
  return Status::Ok();
}

Status propose(void *context, const RegionDiscoveryRequest &request,
               mem::AllocatorRef allocator, RegionProposalWriter &writer) noexcept {
  const auto &graph = request.graph;
  if (!request.allowed_nodes.empty() && request.allowed_nodes.size() != graph.node_count())
    return Status::InvalidArgument("Apple discovery mask size differs");
  if (graph.dimension_count() || !graph.node_count()) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto analysis,
      graph::detail::analyze_native_reachability(graph, allocator));
  HYLE_RETURN_IF_ERROR(propose_native_paged_append(context, request, analysis.needed, writer));
  HYLE_RETURN_IF_ERROR(propose_native_paged(context, request, analysis.needed, writer));
  HYLE_RETURN_IF_ERROR(propose_native_ragged(context, request, analysis.needed, writer));
  using Packed = graph::detail::ArtifactPackedPattern;
  using Static = graph::detail::StaticFloat8Pattern;
  Packed::Uses ct_uses(allocator);
  HYLE_RETURN_IF_ERROR(ct_uses.initialize(graph));
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> ct_allowed(allocator);
  if (!ct_allowed.try_resize(graph.node_count()))
    return Status::ResourceExhausted("Apple CT discovery metadata");
  std::ranges::fill(ct_allowed, std::uint8_t{0});
  const auto ordinary_begin = writer.size();
  HYLE_RETURN_IF_ERROR(graph::detail::discover_regions(graph, allocator,
      [&](NodeId node) -> Result<bool> {
        if (!analysis.needed[node.ordinal()] ||
            (!request.allowed_nodes.empty() && !request.allowed_nodes[node.ordinal()]))
          return false;
        HYLE_ASSIGN_OR_RETURN(const auto &operation, graph.operation(node));
        if (native_paged_operation(operation) || native_paged_append_operation(operation) ||
            native_ragged_operation(context, operation)) return false;
        // Discovery is structural, without per-node prepare/JIT. Actual typed
        // mathematics and device limits remain the final lowerer's admission.
        return graph::detail::validate_native_node(operation).ok();
      },
      [&](std::span<const NodeId> nodes) -> Result<bool> {
        if (nodes.empty()) return true;
        return writer.try_add({.nodes = nodes,
            .cost = {.persistent_bytes = std::nullopt, .copy_bytes = std::nullopt,
                     .repack_bytes = std::nullopt, .synchronization_count = 0, .dispatch_count = 1},
            .route = 1, .replay_class = compute::ProviderReplayClass::kManaged,
            .implementation = compute::ProviderImplementationId{1}, .preference = 200});
        }));
  const auto ordinary_end = writer.size();
  if (!request.allowed_nodes.empty() && !writer.proposal_limit_reached() && !writer.node_limit_reached())
    HYLE_RETURN_IF_ERROR(graph::detail::add_region_singleton_fallbacks(writer, ordinary_begin));
  // Singleton fallbacks belong only to the original route. A packed route
  // must retain the complete recipe and its same-region reader proof.
  for (auto region = ordinary_begin; region < ordinary_end; ++region) {
    const auto nodes = writer.proposals()[region].nodes;
    for (auto node : nodes) ct_allowed[node.ordinal()] = 1;
    bool has_packed = false;
    bool has_grouped_packed = false;
    bool has_static = false;
    for (auto node : nodes) {
      Packed::Pattern pattern;
      if (Packed::pattern(graph, node, pattern) &&
          pattern.descriptor.format != compute::model_ops::WeightArtifactFormat::kNf4DoubleQuant &&
          Packed::closed(graph, pattern, ct_uses, ct_allowed)) {
        has_packed = true;
        has_grouped_packed |= pattern.descriptor.expert_count != 0;
      }
      Static::Pattern fp8;
      has_static |= Static::pattern(graph, node, fp8) &&
          Static::closed(graph, fp8, ct_uses, ct_allowed);
    }
    for (auto node : nodes) ct_allowed[node.ordinal()] = 0;
    // Both native candidates remain available. The packed candidate owns
    // no decoded weight bank; opaque MLX private bounds stay unknown.
    if (has_packed) {
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = writer.proposals()[region].nodes,
          .cost = {.persistent_bytes = std::nullopt, .copy_bytes = std::nullopt,
                   .repack_bytes = std::nullopt, .synchronization_count = 0, .dispatch_count = 1},
          .route = kNativeCtPackedRoute, .replay_class = compute::ProviderReplayClass::kManaged,
          .implementation = compute::ProviderImplementationId{1}, .policy = Packed::kPolicy,
          // A closed grouped recipe avoids the decoded ENK bank by default.
          // This memory choice does not assert a throughput advantage.
          .preference = has_grouped_packed ? 199U : 201U}));
      if (!more) break;
    }
    if (has_static) {
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = writer.proposals()[region].nodes,
          .cost = {.persistent_bytes = std::nullopt, .copy_bytes = std::nullopt,
                   .repack_bytes = std::nullopt, .synchronization_count = 0, .dispatch_count = 1},
          .route = kNativeStaticFloat8Route, .replay_class = compute::ProviderReplayClass::kManaged,
          .implementation = compute::ProviderImplementationId{1}, .policy = Static::kPolicy,
          .preference = 201}));
      if (!more) break;
    }
  }
  return Status::Ok();
}

Status resources(void *context, const RegionCompileRequest &request, mem::AllocatorRef,
    std::span<RegionInputRequirement> inputs,
    std::span<RegionOutputRequirement> outputs, RegionResourceProposal *proposal) noexcept {
  if (request.route == kNativePagedRoute)
    return native_paged_resources(context, request, inputs, outputs, proposal);
  if (request.route == kNativePagedAppendRoute)
    return native_paged_append_resources(context, request, inputs, outputs, proposal);
  if (request.route == kNativeRaggedRoute)
    return native_ragged_resources(context, request, inputs, outputs, proposal);
  HYLE_RETURN_IF_ERROR(qualify(request));
  const cd::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kHostAccessible |
          mem::MemoryProperty::kDeviceAccessible | mem::MemoryProperty::kHostDeviceCoherent,
      .minimum_alignment = 1};
  std::ranges::fill(inputs, RegionInputRequirement{binding, true});
  std::ranges::fill(outputs, RegionOutputRequirement{binding,
      {RegionOutputKind::kDeferredGenerated, RegionOutputProjection::kNativeOnly}});
  *proposal = {.runtime = runtime_properties, .workspace_resource = binding};
  return Status::Ok();
}

Status create_slot(void *prepared, mem::AllocatorRef allocator, mem::BufferView, void **output) noexcept {
  *output = nullptr;
  auto &artifact = *static_cast<Artifact *>(prepared);
  auto *slot = allocator.construct<Slot>(allocator, artifact);
  if (!slot) return Status::ResourceExhausted("Apple region slot metadata");
  *output = slot;
  try {
    HYLE_ASSIGN_OR_RETURN(slot->invocation, artifact.program.create_invocation(allocator));
    slot->bindings.resize(artifact.input_ports.size());
    slot->row_layouts.resize(artifact.row_accesses.size());
    slot->pending_frames.resize(artifact.pending_inputs.size());
    slot->projected.resize(artifact.output_descriptions.size());
    slot->exported.resize(artifact.output_descriptions.size());
    slot->borrowed.reserve(artifact.input_ports.size());
    slot->owned_inputs.reserve(artifact.input_ports.size());
    slot->sources.reserve(artifact.input_ports.size());
    return Status::Ok();
  } catch (const std::bad_alloc &) { return Status::ResourceExhausted("Apple region port metadata"); }
}

void clear_projection(Slot &slot) noexcept {
  for (auto &lease : slot.exported) lease.reset();
  for (auto &array : slot.projected) array.reset();
}
void clear_bound_inputs(Slot &slot) noexcept {
  slot.borrowed.clear(); slot.owned_inputs.clear(); slot.sources.clear();
  for (auto &binding : slot.bindings) { binding.value.reset(); binding.source.reset(); }
}
Status bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
            std::span<const compute::TensorView> outputs) noexcept {
  auto &artifact = *static_cast<Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(invocation);
  if (outputs.size() != artifact.output_descriptions.size())
    return Status::InvalidArgument("Apple generated output arity differs");
  if (slot.pending_consumed)
    return Status::FailedPrecondition("reset the consumed Apple row pending episode before binding");
  slot.result.reset();
  HYLE_RETURN_IF_ERROR(an::InvocationAccess::reuse_gate(slot.invocation));
  clear_projection(slot); clear_bound_inputs(slot);
  try {
    for (std::size_t port = 0; port < artifact.input_ports.size(); ++port) {
      const auto index = artifact.input_ports[port];
      if (index >= inputs.size()) return Status::InvalidArgument("Apple region input arity differs");
      const auto &input = inputs[index];
      for (std::size_t bank = 0; bank < artifact.pending_ports.size(); ++bank) {
        if (artifact.pending_ports[bank] != index || !slot.pending_frames[bank].valid()) continue;
        HYLE_ASSIGN_OR_RETURN(auto frame, cd::PagedArenaAccess::tensor(slot.pending_frames[bank]));
        const auto expected = frame.storage(), actual = input.tensor.storage();
        if (!input.tensor.valid() || expected.resource() != actual.resource() ||
            expected.native_handle() != actual.native_handle() ||
            expected.offset() != actual.offset() || expected.size() != actual.size())
          return Status::InvalidArgument("Apple row input differs from its exact pending frame");
      }
      auto &binding = slot.bindings[port];
      const bool projected_bank = std::ranges::any_of(artifact.row_accesses,
          [&](const auto &access) { return access.input_port == index; });
      if (projected_bank) {
        if (input.reference.valid())
          return Status::InvalidArgument("Apple projected bank needs its exact external storage");
        // Addressable capacity is not a whole-bank ReadLease. The common
        // resource episode admits only the exact typed page guards.
        HYLE_ASSIGN_OR_RETURN(binding.value, an::MemoryAccess::borrow_tensor(input.tensor));
        slot.borrowed.push_back({input.tensor.storage()});
      } else if (input.reference.valid()) {
        const auto &reference = input.reference;
        HYLE_ASSIGN_OR_RETURN(binding.source,
            native_output_source(reference, artifact.runtime, input.description));
        slot.sources.push_back(reference);
      } else if (input.lease) {
        HYLE_ASSIGN_OR_RETURN(auto owned, input.lease->view());
        const auto wanted = input.tensor.storage();
        if (!input.tensor.valid() || owned.resource() != wanted.resource() ||
            owned.native_handle() != wanted.native_handle() || wanted.offset() < owned.offset() ||
            wanted.offset() - owned.offset() > owned.size() ||
            wanted.size() > owned.size() - (wanted.offset() - owned.offset()))
          return Status::InvalidArgument("Apple input lease does not cover its exact physical range");
        HYLE_ASSIGN_OR_RETURN(auto selected,
            input.lease->subview(wanted.offset() - owned.offset(), wanted.size()));
        slot.owned_inputs.push_back(selected);
        HYLE_ASSIGN_OR_RETURN(binding.value, an::MemoryAccess::owned_tensor(
            std::move(selected), input.description, {.allocator = slot.allocator}));
      } else {
        HYLE_ASSIGN_OR_RETURN(binding.value, an::MemoryAccess::borrow_tensor(input.tensor));
        const auto storage = input.tensor.storage();
        slot.borrowed.push_back({storage});
      }
    }
    return an::InvocationAccess::prepare_inputs(slot.invocation, slot.bindings);
  } catch (const std::bad_alloc &) { return Status::ResourceExhausted("Apple input binding metadata"); }
}

Status seal(void *, void *invocation) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  if (slot.pending_consumed)
    return Status::FailedPrecondition("reset the consumed Apple row pending episode before replay preparation");
  slot.result.reset();
  return an::InvocationAccess::reuse_gate(slot.invocation);
}

struct AliasProjection final {
  bool borrowed = false;
  mem::ConstBufferView storage;
  bool owned_overlap = false;
  compute::ReadLease owner;
};
bool same_reference(const RegionOutputReference &a, const RegionOutputReference &b) noexcept {
  return a.vtable == b.vtable && a.prepared == b.prepared &&
      a.invocation == b.invocation && a.port == b.port;
}
Result<AliasProjection> input_alias(Slot &slot, const ca::NativeStorage &storage) noexcept {
  try {
    AliasProjection result;
    const auto overlap = [&](mem::ConstBufferView view) noexcept {
      return view.native_handle() == storage.buffer && view.size() && storage.view_bytes &&
          (view.offset() <= storage.offset_bytes ? storage.offset_bytes - view.offset() < view.size()
                                                : view.offset() - storage.offset_bytes < storage.view_bytes);
    };
    const auto inspect_borrowed = [&](const BorrowedRange &range) -> Status {
      const auto view = range.storage;
      if (!overlap(view)) return Status::Ok();
      result.borrowed = true;
      if (storage.offset_bytes >= view.offset() && storage.offset_bytes - view.offset() <= view.size() &&
          storage.view_bytes <= view.size() - (storage.offset_bytes - view.offset())) {
        HYLE_ASSIGN_OR_RETURN(result.storage, view.subview(storage.offset_bytes - view.offset(), storage.view_bytes));
      }
      return Status::Ok();
    };
    const auto inspect_owner = [&](const compute::ReadLease &owner) -> Status {
      HYLE_ASSIGN_OR_RETURN(auto view, owner.view());
      if (!overlap(view)) return Status::Ok();
      result.owned_overlap = true;
      if (!result.owner.valid() && storage.offset_bytes >= view.offset() &&
          storage.offset_bytes - view.offset() <= view.size() &&
          storage.view_bytes <= view.size() - (storage.offset_bytes - view.offset())) {
        HYLE_ASSIGN_OR_RETURN(result.owner,
            owner.subview(storage.offset_bytes - view.offset(), storage.view_bytes));
      }
      return Status::Ok();
    };
    for (const auto &range : slot.borrowed) HYLE_RETURN_IF_ERROR(inspect_borrowed(range));
    for (const auto &range : slot.artifact.borrowed_constants) HYLE_RETURN_IF_ERROR(inspect_borrowed(range));
    if (result.storage.valid()) return result;
    for (const auto &owner : slot.owned_inputs) HYLE_RETURN_IF_ERROR(inspect_owner(owner));
    Vector<RegionOutputReference> pending{mem::StdAllocator<RegionOutputReference>(slot.allocator)},
        visited{mem::StdAllocator<RegionOutputReference>(slot.allocator)};
    pending.insert(pending.end(), slot.sources.begin(), slot.sources.end());
    while (!pending.empty()) {
      const auto source = pending.back(); pending.pop_back();
      if (std::ranges::any_of(visited, [&](const auto &old) { return same_reference(old, source); })) continue;
      visited.push_back(source);
      if (!source.valid() || !source.prepared || !source.invocation || !source.vtable->query_extension)
        return std::unexpected(Status::InvalidArgument("Apple alias source has no actual output capability"));
      const auto *table = static_cast<const NativeOutputVTable *>(
          source.vtable->query_extension(source.prepared, kNativeOutputExtensionId));
      if (!table || !table->origins)
        return std::unexpected(Status::InvalidArgument("Apple alias source cannot prove its original ranges"));
      HYLE_ASSIGN_OR_RETURN(auto origins, table->origins(
          source.prepared, source.invocation, source.port, storage));
      for (const auto &range : origins.borrowed) HYLE_RETURN_IF_ERROR(inspect_borrowed(range));
      for (const auto &range : origins.constants) HYLE_RETURN_IF_ERROR(inspect_borrowed(range));
      if (result.storage.valid()) return result;
      for (const auto &owner : origins.owned) HYLE_RETURN_IF_ERROR(inspect_owner(owner));
      if (origins.owner.valid()) HYLE_RETURN_IF_ERROR(inspect_owner(origins.owner));
      result.owned_overlap |= origins.owned_overlap;
      pending.insert(pending.end(), origins.sources.begin(), origins.sources.end());
    }
    return result;
  } catch (const std::bad_alloc &) { return std::unexpected(Status::ResourceExhausted("Apple alias owner metadata")); }
}

Status project(Slot &slot, std::size_t port) noexcept {
  if (port >= slot.projected.size()) return Status::OutOfRange("Apple generated output port");
  if (slot.projected[port]) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto array, an::InvocationAccess::output(slot.invocation, port));
  slot.projected[port] = std::move(array);
  return Status::Ok();
}
Status export_port(Slot &slot, std::size_t port) noexcept {
  HYLE_RETURN_IF_ERROR(project(slot, port));
  if (slot.exported[port]) return Status::Ok();
  if (!slot.projected[port]->nbytes())
    return Status::Unimplemented("Apple output has no nonempty readonly byte projection");
  HYLE_RETURN_IF_ERROR(an::MemoryAccess::match_readonly(
      *slot.projected[port], slot.artifact.output_descriptions[port]));
  HYLE_ASSIGN_OR_RETURN(auto storage, an::MemoryAccess::storage_info(*slot.projected[port]));
  HYLE_ASSIGN_OR_RETURN(auto alias, input_alias(slot, storage));
  if (alias.borrowed) return Status::Unimplemented("borrowed Apple input/constant alias has no independent payload owner");
  // Re-exporting an owned view must preserve its original exact lease block,
  // rather than retain a new Array -> lease -> Array chain on every replay.
  if (alias.owner.valid()) {
    slot.exported[port] = std::move(alias.owner);
    return Status::Ok();
  }
  if (alias.owned_overlap)
    return Status::Unimplemented("Apple output alias exceeds its owned input permission");
  HYLE_ASSIGN_OR_RETURN(slot.exported[port], an::MemoryAccess::export_readonly(
      *slot.projected[port], slot.artifact.output_descriptions[port], {.allocator = slot.allocator}));
  return Status::Ok();
}
Result<RegionValue> value(void *, void *invocation, std::size_t port) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  std::lock_guard lock(slot.output_mutex);
  HYLE_RETURN_IF_ERROR(project(slot, port));
  RegionValue result{slot.artifact.output_descriptions[port], {}, &array_type, &*slot.projected[port]};
  if (!slot.projected[port]->nbytes()) return result;
  HYLE_ASSIGN_OR_RETURN(auto storage, an::MemoryAccess::storage_info(*slot.projected[port]));
  HYLE_ASSIGN_OR_RETURN(auto alias, input_alias(slot, storage));
  // Native consumers may accept an original array whose physical layout has
  // no exact byte projection for the public description. Keep that capability;
  // backing/permission inspection above must still succeed before returning it.
  if (!an::MemoryAccess::match_readonly(*slot.projected[port], result.description).ok())
    return result;
  if (alias.borrowed) {
    if (alias.storage.valid()) {
      HYLE_ASSIGN_OR_RETURN(result.tensor, compute::ConstTensorView::create(result.description, alias.storage));
    }
    return result; // A true borrowed view grants no independent owner/lease.
  }
  const auto status = export_port(slot, port);
  if (status.IsUnimplemented()) return result;
  HYLE_RETURN_IF_ERROR(status);
  HYLE_ASSIGN_OR_RETURN(result.tensor, slot.exported[port]->tensor(result.description));
  return result;
}
Result<compute::ReadLease> lease(void *, void *invocation, std::size_t port) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  std::lock_guard lock(slot.output_mutex);
  HYLE_RETURN_IF_ERROR(export_port(slot, port));
  return *slot.exported[port];
}

Result<compute::Completion> submit(void *, void *invocation,
    std::span<const cd::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView>, std::span<const compute::TensorView>) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  for (const auto source : dependencies)
    HYLE_RETURN_IF_ERROR(an::qualify_copy_dependency(slot.artifact.runtime, source));
  slot.result.reset(); clear_projection(slot);
  const bool pending = std::ranges::any_of(slot.pending_frames,
      [](const auto &frame) { return frame.valid(); });
  auto submitted = [&]() -> Result<compute::OwnedResult<ca::Results>> {
    if (!pending) return slot.invocation.submit();
    if (slot.pending_consumed || !slot.pending_source || !slot.pending_uses.valid())
      return std::unexpected(Status::FailedPrecondition("Apple row pending source/episode is unavailable"));
    HYLE_RETURN_IF_ERROR(an::InvocationAccess::reuse_gate(slot.invocation));
    const auto &update = slot.pending_source->source.update();
    HYLE_RETURN_IF_ERROR(qualify_native_pending_source(*slot.pending_source, update));
    HYLE_ASSIGN_OR_RETURN(auto uses, slot.pending_uses.begin());
    slot.pending_reads.reset();
    for (const auto &frame : slot.pending_frames)
      if (frame.valid())
        HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::collect_reads<ca::CopyBatchUpdate,
            an::CopyStreamAccess>(slot.pending_reads, frame, update)));
    HYLE_RETURN_IF_ERROR(slot.pending_reads.admit(uses));
    return an::PagedBatchAccess::submit_pending(slot.invocation, std::move(uses), update);
  }();
  if (!submitted) return std::unexpected(submitted.error());
  if (pending) slot.pending_consumed = true;
  auto transfer = cd::ResultAccess::take_storage(std::move(*submitted));
  slot.result = std::move(transfer.owner);
  return std::move(transfer.completion);
}
const NativeOutputVTable native_output_table{
    [](void *prepared, void *invocation, std::size_t port,
        const std::shared_ptr<an::Runtime> &selected_runtime,
        const compute::TensorDesc &description) noexcept -> Result<an::InputSource> {
      if (!prepared || !invocation)
        return std::unexpected(Status::InvalidArgument("Apple output source has no actual slot"));
      auto &slot = *static_cast<Slot *>(invocation);
      const auto &artifact = *static_cast<Artifact *>(prepared);
      if (&slot.artifact != &artifact || artifact.runtime != selected_runtime ||
          port >= artifact.output_descriptions.size() || artifact.output_descriptions[port] != description)
        return std::unexpected(Status::InvalidArgument("Apple output source Runtime/owner/physical port differs"));
      return an::InvocationAccess::source(slot.invocation, port);
    },
    [](void *prepared, void *invocation, std::size_t port,
        const ca::NativeStorage &) noexcept -> Result<NativeOutputOrigins> {
      if (!prepared || !invocation)
        return std::unexpected(Status::InvalidArgument("Apple output origins have no actual slot"));
      const auto &slot = *static_cast<Slot *>(invocation);
      const auto &artifact = *static_cast<Artifact *>(prepared);
      if (&slot.artifact != &artifact || port >= artifact.output_descriptions.size())
        return std::unexpected(Status::InvalidArgument("Apple output origins owner/physical port differs"));
      const auto ready = an::InvocationAccess::output(slot.invocation, port);
      if (!ready) return std::unexpected(ready.error());
      return NativeOutputOrigins{slot.borrowed, artifact.borrowed_constants,
          slot.owned_inputs, slot.sources, {}, false};
    }};

bool empty_row_layout(const Artifact &artifact, const Slot *slot, std::size_t index) noexcept {
  return slot && &slot->artifact == &artifact && index < slot->row_layouts.size() &&
      (artifact.row_accesses[index].native.kind == an::RowCacheAccessKind::kPaddedCheckedDecode ||
       artifact.row_accesses[index].native.kind == an::RowCacheAccessKind::kPagedIndexer) &&
      slot->row_layouts[index].configured && slot->row_layouts[index].empty;
}
Status inspect_row_port(const ProviderRegionPrepared &prepared, void *invocation,
    std::uint32_t port, cd::OperationPreparedAccess *output) noexcept {
  if (prepared.vtable != &prepared_table || !prepared.state || !output)
    return Status::InvalidArgument("Apple projected row artifact differs");
  const auto &artifact = *static_cast<const Artifact *>(prepared.state);
  for (std::size_t i = 0; i < artifact.row_accesses.size(); ++i) {
    const auto &access = artifact.row_accesses[i];
    if (access.input_port == cd::OperationPreparedAccess::kNoInput || access.input_port != port) continue;
    *output = {.projected_input = port,
        .allows_empty_footprint = empty_row_layout(artifact, static_cast<const Slot *>(invocation), i)};
    return Status::Ok();
  }
  return Status::Unimplemented("Apple port has no selected checked row reader");
}
Status inspect_row_node(const ProviderRegionPrepared &prepared, void *invocation,
    std::uint32_t node, compute::ProviderExtensionId capability,
    graph::detail::RegionNodeAccess *output) noexcept {
  if (prepared.vtable != &prepared_table || !prepared.state || !output)
    return Status::InvalidArgument("Apple projected row artifact differs");
  if (capability != cd::kRowCacheColdPlanningId)
    return Status::Unimplemented("Apple logical access capability differs");
  const auto &artifact = *static_cast<const Artifact *>(prepared.state);
  for (std::size_t i = 0; i < artifact.row_accesses.size(); ++i) {
    const auto &access = artifact.row_accesses[i];
    if (access.node != node) continue;
    *output = {.access = {.projected_input = access.input_port,
        .allows_empty_footprint = empty_row_layout(artifact, static_cast<const Slot *>(invocation), i),
        .requires_configuration = true},
        .allows_pending = access.input_port != cd::OperationPreparedAccess::kNoInput &&
            std::ranges::find(artifact.pending_ports, access.input_port) != artifact.pending_ports.end()};
    return Status::Ok();
  }
  return Status::Unimplemented("Apple node has no selected row access proof");
}
Result<bool> configure_row_node(const ProviderRegionPrepared &prepared, void *invocation,
    std::uint32_t node, compute::ProviderExtensionId capability, const void *configuration) noexcept {
  if (prepared.vtable != &prepared_table || !prepared.state || !invocation || !configuration)
    return std::unexpected(Status::InvalidArgument("Apple row configuration is incomplete"));
  if (capability != cd::kRowCacheColdPlanningId)
    return std::unexpected(Status::Unimplemented("Apple row configuration capability differs"));
  const auto &artifact = *static_cast<const Artifact *>(prepared.state);
  auto &slot = *static_cast<Slot *>(invocation);
  if (&slot.artifact != &artifact || slot.row_layouts.size() != artifact.row_accesses.size())
    return std::unexpected(Status::InvalidArgument("Apple row slot differs from its artifact"));
  const auto position = std::ranges::find(artifact.row_accesses, node, &PreparedRowAccess::node);
  if (position == artifact.row_accesses.end())
    return std::unexpected(Status::Unimplemented("Apple node has no selected row configuration"));
  if (std::ranges::any_of(slot.pending_frames, [](const auto &frame) { return frame.valid(); }))
    return std::unexpected(Status::FailedPrecondition("reset Apple pending row frames before changing their layout"));
  slot.result.reset();
  HYLE_RETURN_IF_ERROR(an::InvocationAccess::reuse_gate(slot.invocation));
  // The old MLX input arrays capture metadata ReadLeases. Release them through
  // the original idle/result gate before the owned mirror upgrades its writer.
  HYLE_RETURN_IF_ERROR(an::InvocationAccess::release_inputs(slot.invocation));
  clear_projection(slot); clear_bound_inputs(slot);
  auto &layout = slot.row_layouts[static_cast<std::size_t>(position - artifact.row_accesses.begin())];
  layout = {};
  const auto &fact = position->native;
  const auto &config = *static_cast<const cd::RowCacheColdLayout *>(configuration);
  const bool mapping = fact.kind == an::RowCacheAccessKind::kMap ||
                       fact.kind == an::RowCacheAccessKind::kPaddedMap ||
                       fact.kind == an::RowCacheAccessKind::kPagedIndexer;
  if (config.query_indptr.size() <= config.requests || config.kv_indptr.size() <= config.requests ||
      config.last_page_lengths.size() < config.requests || config.query_indptr[0] != 0 ||
      config.kv_indptr[0] != 0 || config.query_indptr[config.requests] != std::int64_t(fact.query_rows) ||
      config.kv_indptr[config.requests] < 0 ||
      (mapping && (config.requests > fact.max_requests ||
                   std::uint32_t(config.kv_indptr[config.requests]) > fact.max_refs)))
    return std::unexpected(Status::InvalidArgument("Apple row cold layout extent differs"));
  bool empty = true;
  for (std::uint32_t request = 0; request < config.requests; ++request) {
    const auto begin = config.query_indptr[request], end = config.query_indptr[request + 1];
    const auto first = config.kv_indptr[request], last = config.kv_indptr[request + 1];
    const auto tail = config.last_page_lengths[request];
    if (begin < 0 || end < begin || std::uint64_t(end) > fact.query_rows || first < 0 || last < first ||
        !((last == first && tail == 0) ||
          (last > first && tail > 0 && (!mapping || std::uint32_t(tail) <= fact.page_size))))
      return std::unexpected(Status::InvalidArgument("Apple row cold request span differs"));
    empty = empty && last == first;
  }
  // Only copied scalar facts survive. Device metadata remains the sole map.
  layout = {.configured = true, .empty = empty, .requests = config.requests,
      .page_refs = static_cast<std::uint32_t>(config.kv_indptr[config.requests])};
  return true;
}
const graph::detail::RegionProjectedReadVTable row_projected_table{
    .inspect = &inspect_row_port, .inspect_node = &inspect_row_node, .configure_node = &configure_row_node};

Status reserve_pending_reads(const Artifact &artifact, Slot &slot) noexcept {
  std::uint64_t resources = 0;
  for (const auto access : artifact.pending_accesses)
    resources += slot.row_layouts[access].page_refs;
  if (resources > UINT32_MAX)
    return Status::ResourceExhausted("Apple pending row page guard capacity overflows");
  const auto capacity = static_cast<std::uint32_t>(std::max(std::uint64_t{1}, resources));
  if (slot.pending_uses.valid() && capacity <= slot.pending_capacity) return Status::Ok();
  if (slot.pending_uses.active_batches())
    return Status::Unavailable("Apple pending row guards have not retired");
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({
      .max_batches = 1, .max_resources_per_batch = capacity,
      .allocator = slot.allocator, .resource_budget = artifact.budget}));
  HYLE_RETURN_IF_ERROR(slot.pending_reads.reserve(capacity));
  slot.pending_uses = std::move(uses);
  slot.pending_capacity = capacity;
  return Status::Ok();
}
const bd::NativePendingRegionVTable row_pending_table{
    .inspect = [](void *prepared, void *) noexcept -> Result<bd::NativePendingRegionFacts> {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      return bd::NativePendingRegionFacts{artifact.pending_inputs, artifact.pending_ports,
          {&native_pending_source_table, artifact.runtime.get(),
              [](void *opaque, std::size_t capacity, mem::AllocatorRef allocator,
                  compute::ResourceBudget budget) noexcept -> Result<void *> {
                return create_native_pending_source(static_cast<Artifact *>(opaque)->runtime,
                    capacity, allocator, std::move(budget));
              }}};
    },
    .validate_frame = [](void *prepared, void *invocation, ValueId value,
        const compute::PagedArenaPendingRead &frame,
        const bd::NativePendingSourceOwner &owner) noexcept -> Status {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      auto &slot = *static_cast<Slot *>(invocation);
      if (&slot.artifact != &artifact)
        return Status::InvalidArgument("Apple row pending slot differs from its artifact");
      HYLE_ASSIGN_OR_RETURN(auto source, native_pending_source(owner, artifact.runtime));
      if (slot.pending_source && slot.pending_source != source)
        return Status::InvalidArgument("Apple row pending source differs within this episode");
      const auto position = std::ranges::find(artifact.pending_inputs, value);
      if (position == artifact.pending_inputs.end() || !frame.valid() || slot.pending_consumed)
        return Status::InvalidArgument("Apple row pending input/frame/episode differs");
      const auto bank = static_cast<std::size_t>(position - artifact.pending_inputs.begin());
      if (slot.pending_frames[bank].valid())
        return Status::FailedPrecondition("reset Apple pending row input before rebinding");
      const auto access = artifact.pending_accesses[bank];
      const auto &layout = slot.row_layouts[access];
      const auto facts = frame.layout().facts();
      if (!layout.configured || facts.query_rows != artifact.row_accesses[access].native.query_rows ||
          facts.requests != layout.requests || facts.page_refs != layout.page_refs)
        return Status::InvalidArgument("Apple row pending frame differs from its cold layout");
      HYLE_ASSIGN_OR_RETURN(auto tensor, cd::PagedArenaAccess::tensor(frame));
      const auto &description = artifact.pending_descriptions[bank];
      HYLE_ASSIGN_OR_RETURN(auto frame_bytes, tensor.description().required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto bank_bytes, description.required_bytes());
      // The original arena exposes [pages,page_size,row_bytes]. The checked
      // decoder uses [physical_rows,row_bytes] on exactly the same bytes.
      // This imports only its addressable descriptor; submit admits exact pages.
      if (!an::is_copy_resource(tensor.storage().resource()) ||
          !tensor.description().is_plain() || !tensor.description().is_dense() ||
          tensor.description().data_type() != compute::DataType::kUInt8 ||
          tensor.description().rank() != 3 || description.rank() != 2 ||
          tensor.description().dimensions()[2] != description.dimensions()[1] ||
          frame_bytes != bank_bytes)
        return Status::InvalidArgument("Apple pending row physical bank description/resource differs");
      return reserve_pending_reads(artifact, slot);
    },
    .commit_frame = [](void *prepared, void *invocation, ValueId value,
        compute::PagedArenaPendingRead frame, const bd::NativePendingSourceOwner &owner) noexcept {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      auto &slot = *static_cast<Slot *>(invocation);
      const auto source = native_pending_source(owner, artifact.runtime);
      const auto position = std::ranges::find(artifact.pending_inputs, value);
      HYLE_FATAL_ASSERT(source.has_value() && &slot.artifact == &artifact &&
          position != artifact.pending_inputs.end());
      slot.pending_source = *source;
      slot.pending_frames[static_cast<std::size_t>(position - artifact.pending_inputs.begin())] = std::move(frame);
    },
    .reset = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      for (auto &frame : slot.pending_frames) frame = {};
      slot.pending_source = nullptr;
      slot.pending_consumed = false;
      slot.pending_reads.reset();
    }};

const graph::detail::RegionBindingReleaseVTable binding_release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      const auto &slot = *static_cast<const Slot *>(invocation);
      if (!slot.invocation.valid())
        return slot.result ? Status::FailedPrecondition("native result has no invocation") : Status::Ok();
      return an::InvocationAccess::can_detach_inputs(slot.invocation, slot.result);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.result.reset();
      if (slot.invocation.valid()) an::InvocationAccess::detach_inputs(slot.invocation);
      clear_projection(slot);
      clear_bound_inputs(slot);
    }};

const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *state, mem::AllocatorRef allocator) noexcept { allocator.destroy(static_cast<Artifact *>(state)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = &create_slot,
    .destroy_invocation = [](void *state, mem::AllocatorRef allocator) noexcept { allocator.destroy(static_cast<Slot *>(state)); },
    .delegated_submit = &submit,
    .endpoint = [](void *prepared, void *) noexcept -> cd::CompletionEndpoint {
      return region_ordered_endpoint(static_cast<Artifact *>(prepared)->runtime);
    },
    .can_import_endpoint = [](void *prepared, void *, cd::CompletionEndpoint source) noexcept {
      return imports_region_ordered_endpoint(static_cast<Artifact *>(prepared)->runtime, source);
    },
    .output_contract = [](void *, std::size_t) noexcept {
      return RegionOutputContract{RegionOutputKind::kDeferredGenerated, RegionOutputProjection::kNativeOnly};
    },
    .output_value = &value, .prepare_bindings = &bind, .seal_bindings = &seal, .output_lease = &lease,
    .query_extension = [](void *prepared, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == graph::detail::kRegionBindingReleaseExtensionId) return &binding_release_table;
      if (id == kNativeOutputExtensionId) return &native_output_table;
      if (id == bd::kNativePendingRegionExtensionId &&
          !static_cast<const Artifact *>(prepared)->pending_inputs.empty()) return &row_pending_table;
      if (id == graph::detail::kRegionProjectedReadExtensionId) return &row_projected_table;
      return nullptr;
    }};

compute::ProviderPrepareOutcome compile(void *context, const RegionCompileRequest &request,
    mem::AllocatorRef allocator, ProviderRegionPrepared *result) noexcept {
  using Outcome = compute::ProviderPrepareOutcome;
  *result = {};
  if (request.route == kNativePagedRoute)
    return compile_native_paged(context, request, allocator, result);
  if (request.route == kNativePagedAppendRoute)
    return compile_native_paged_append(context, request, allocator, result);
  if (request.route == kNativeRaggedRoute)
    return compile_native_ragged(context, request, allocator, result);
  auto status = qualify(request);
  if (!status.ok()) return Outcome::unsupported(std::move(status));
  auto *artifact = allocator.construct<Artifact>(allocator);
  if (!artifact) return Outcome::error(Status::ResourceExhausted("Apple native artifact metadata"));
  result->vtable = &prepared_table; result->state = artifact;
  try {
    bool custom_failed = false;
    PreparedConstantStorage prepared_constants;
    const auto build = [&]() -> Status {
    artifact->runtime = source_runtime(context);
    artifact->budget = request.resource_budget;
    Vector<mx::array> constants{mem::StdAllocator<mx::array>(allocator)};
    Vector<ConstantBinding> selected{mem::StdAllocator<ConstantBinding>(allocator)};
    // The common request can carry the complete graph constant table. Capture
    // only actual incoming constant ports of this selected native fragment.
    for (std::size_t port = 0; port < request.inputs.size(); ++port) {
      const auto value = request.inputs[port];
      if (value.kind() != ValueKind::kConstant) {
        artifact->input_ports.push_back(port);
        continue;
      }
      const auto found = std::ranges::find_if(request.constants,
          [&](const ConstantBinding &entry) { return entry.value == value; });
      if (found == request.constants.end()) return Status::InvalidArgument("Apple region constant binding is missing");
      const auto &constant = *found;
      selected.push_back(constant);
      HYLE_ASSIGN_OR_RETURN(auto array, an::MemoryAccess::borrow_tensor(constant.tensor));
      constants.push_back(std::move(array));
      const auto storage = constant.tensor.storage();
      artifact->borrowed_constants.push_back({storage});
    }
    for (const auto output : request.outputs) {
      HYLE_ASSIGN_OR_RETURN(auto description, request.graph->value_description(output));
      artifact->output_descriptions.push_back(description);
    }
    auto boundary = request;
    boundary.constants = selected;
    Vector<ValueId> terminal_outputs{mem::StdAllocator<ValueId>(allocator)};
    if (boundary.outputs.empty()) {
      // A control-only fragment still needs real native terminal arrays for
      // evaluation/checked completion. These stay private to its Program;
      // the common region continues to expose zero tensor output ports.
      const auto last = boundary.nodes.back();
      HYLE_ASSIGN_OR_RETURN(auto count, boundary.graph->node_output_count(last));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, boundary.graph->result(last, port));
        terminal_outputs.push_back(value);
      }
      if (terminal_outputs.empty())
        return Status::Unimplemented("Apple control fragment has no native terminal result");
      boundary.outputs = terminal_outputs;
    }
    HYLE_ASSIGN_OR_RETURN(artifact->program, prepare_region(artifact->runtime, boundary, constants,
        static_cast<const SourceContext *>(context)->custom, &custom_failed, &artifact->row_accesses,
        &prepared_constants));
    for (std::size_t access = 0; access < artifact->row_accesses.size(); ++access) {
      const auto port = artifact->row_accesses[access].input_port;
      if (port == cd::OperationPreparedAccess::kNoInput) continue;
      if (port >= request.inputs.size()) return Status::Internal("Apple row bank boundary port differs");
      const auto value = request.inputs[port];
      if (value.kind() != ValueKind::kInput ||
          std::ranges::find(artifact->pending_inputs, value) != artifact->pending_inputs.end()) continue;
      HYLE_ASSIGN_OR_RETURN(auto description, request.graph->value_description(value));
      artifact->pending_inputs.push_back(value);
      artifact->pending_ports.push_back(port);
      artifact->pending_accesses.push_back(access);
      artifact->pending_descriptions.push_back(description);
    }
    return Status::Ok();
    };
    status = build();
    if (!status.ok()) return !custom_failed && status.IsUnimplemented() ? Outcome::unsupported(std::move(status))
                                                                       : Outcome::error(std::move(status));
    result->properties = {.persistent_bytes = std::nullopt,
                          .packed_constant_count = prepared_constants.count,
                          .packed_constant_bytes = prepared_constants.bytes,
                          .runtime = runtime_properties,
                          .invocation_private_bytes = std::nullopt};
    return Outcome::prepared();
  } catch (const std::bad_alloc &) { return Outcome::error(Status::ResourceExhausted("Apple native preparation metadata")); }
}
const RegionProviderVTable region_table{&propose, &resources, &compile};
} // namespace
const std::shared_ptr<an::Runtime> &source_runtime(void *context) noexcept {
  return static_cast<const SourceContext *>(context)->runtime;
}
const RegionProviderVTable *native_region_provider() noexcept { return &region_table; }
cd::CompletionEndpoint region_ordered_endpoint(const std::shared_ptr<an::Runtime> &runtime) noexcept {
  return {&owner_endpoint, runtime.get()};
}
bool imports_region_ordered_endpoint(const std::shared_ptr<an::Runtime> &runtime,
    cd::CompletionEndpoint source) noexcept {
  return source.authority == &owner_endpoint && source.identity == runtime.get();
}
Result<an::InputSource> native_output_source(const RegionOutputReference &reference,
    const std::shared_ptr<an::Runtime> &runtime, const compute::TensorDesc &description) noexcept {
  if (!reference.valid() || !reference.prepared || !reference.invocation || !reference.vtable->query_extension)
    return std::unexpected(Status::InvalidArgument("Apple source has no actual selected output capability"));
  const auto *table = static_cast<const NativeOutputVTable *>(
      reference.vtable->query_extension(reference.prepared, kNativeOutputExtensionId));
  if (!table || !table->source || !table->origins)
    return std::unexpected(Status::InvalidArgument("Apple source lacks its native output contract"));
  return table->source(reference.prepared, reference.invocation, reference.port, runtime, description);
}

} // namespace hyle::graph::apple::detail
