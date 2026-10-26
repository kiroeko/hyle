#include "slots_region.h"
#include "plan_state.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/transfer.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/slots.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::graph::slots {
namespace c = compute;
namespace cd = c::detail;
namespace gd = ::hyle::graph::detail;
namespace {
bool same_storage(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}

// Only private I32 control bytes, never inference payload or slot ownership.
// Uploads use original leases/QueueSequence/Completion, including failures.
struct Indices final {
  c::LeasePool device_pool, host_pool;
  c::WriteLease writer, staging;
  c::ReadLease ready, source;
  c::LeaseUsePool uses;
  c::Completion pending;
  bool direct = false, failed = false;

  static Result<Indices> create(mem::MemoryResourceRef resource,
      mem::MemoryResourceRef host, std::size_t count, ResourceOptions options) noexcept {
    if (!resource.valid() || !host.valid() || !count ||
        count > SIZE_MAX / sizeof(std::int32_t) ||
        !host.properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("slot IDs need valid storage, host staging and positive count"));
    const auto bytes = count * sizeof(std::int32_t);
    const c::LeasePoolOptions pool_options{.max_blocks = 1, .backing_alignment = 16,
        .allocator = options.allocator, .resource_budget = options.resource_budget};
    Indices result;
    result.direct = resource.properties().has(mem::MemoryProperty::kHostAccessible);
    HYLE_ASSIGN_OR_RETURN(result.device_pool, c::LeasePool::create(resource, bytes, pool_options));
    HYLE_ASSIGN_OR_RETURN(result.writer, result.device_pool.allocate(bytes, 4));
    if (!result.direct) {
      HYLE_ASSIGN_OR_RETURN(result.host_pool, c::LeasePool::create(host, bytes, pool_options));
      HYLE_ASSIGN_OR_RETURN(result.staging, result.host_pool.allocate(bytes, 4));
      HYLE_ASSIGN_OR_RETURN(result.uses, c::LeaseUsePool::create({.max_batches = 1,
          .max_resources_per_batch = 2, .allocator = options.allocator,
          .resource_budget = options.resource_budget}));
    }
    return result;
  }
  Status upload(const cd::QueueRef &queue,
      std::span<const c::LeaseSlotAddress> addresses) noexcept {
    if (failed || pending.valid() || (!direct && uses.active_batches()))
      return Status::FailedPrecondition("slot index upload is failed or not actually retired");
    if (ready.valid()) { HYLE_ASSIGN_OR_RETURN(writer, ready.try_write()); }
    if (source.valid()) { HYLE_ASSIGN_OR_RETURN(staging, source.try_write()); }
    auto &host_writer = direct ? writer : staging;
    HYLE_ASSIGN_OR_RETURN(auto host, host_writer.view());
    if (addresses.size() != host.size() / sizeof(std::int32_t))
      return Status::InvalidArgument("slot index count differs from its fixed carrier");
    HYLE_ASSIGN_OR_RETURN(auto mapping, host.map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes, mapping.writable_bytes());
    for (std::size_t i = 0; i < addresses.size(); ++i) {
      const auto id = static_cast<std::int32_t>(addresses[i].index);
      std::memcpy(bytes.data() + i * sizeof(id), &id, sizeof(id));
    }
    HYLE_RETURN_IF_ERROR(mapping.flush());
    HYLE_RETURN_IF_ERROR(mapping.close());
    if (direct) {
      HYLE_ASSIGN_OR_RETURN(ready, writer.publish());
      return Status::Ok();
    }
    HYLE_ASSIGN_OR_RETURN(source, staging.publish());
    HYLE_ASSIGN_OR_RETURN(auto input, source.view());
    HYLE_ASSIGN_OR_RETURN(auto output, writer.view());
    HYLE_ASSIGN_OR_RETURN(auto use, uses.begin());
    HYLE_RETURN_IF_ERROR(use.read(source));
    HYLE_RETURN_IF_ERROR(use.write(writer));
    HYLE_ASSIGN_OR_RETURN(auto sequence, cd::QueueSequence::begin(queue));
    sequence.record(cd::TransferAccess::enqueue_copy(sequence, output, input));
    HYLE_ASSIGN_OR_RETURN(pending, sequence.finish());
    HYLE_FATAL_ASSERT(use.attach(pending).ok());
    const auto terminal = pending.wait();
    if (!terminal.ok() || uses.active_batches()) {
      failed = true;
      return terminal.ok() ? Status::Unavailable("slot metadata upload retirement is unproven") : terminal;
    }
    pending = {};
    HYLE_ASSIGN_OR_RETURN(ready, writer.publish());
    return Status::Ok();
  }
};

struct Prepared final {
  Store store;
  NodeId node;
  ValueId rows_value, ids_value;
  c::TensorDesc rows, ids;
  std::uint32_t rows_port = 0, ids_port = 1;
  mem::MemoryResourceRef resource;
  c::ProviderRuntimeProperties runtime;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef allocator) noexcept
      : writer_pointers(mem::StdAllocator<const c::WriteLease *>{allocator}),
        destinations(mem::StdAllocator<mem::BufferView>{allocator}) {}
  SlotBindings bindings;
  Indices indices;
  cd::QueueRef queue;
  ResourceOptions options;
  std::vector<const c::WriteLease *, mem::StdAllocator<const c::WriteLease *>> writer_pointers;
  std::vector<mem::BufferView, mem::StdAllocator<mem::BufferView>> destinations;
  // The original state owns this slot. This non-owning back-reference is
  // stable across public Invocation moves and is queried only while it lives.
  const gd::InvocationState *owner = nullptr;
  gd::ProjectedWriter writer{};
  c::ConstTensorView rows, ids;
  bool prepared = false, sealed = false;
};

Status validate_rows(const Prepared &p, Slot &s,
    c::ConstTensorView rows, c::ConstTensorView ids) noexcept {
  if (!s.prepared || !s.bindings.valid() || !s.indices.ready.valid() ||
      rows.description() != p.rows || ids.description() != p.ids)
    return Status::FailedPrecondition("slot store has no current exact transaction and controls");
  HYLE_ASSIGN_OR_RETURN(auto bound,
      gd::RegionAccess::has_write_effect(s.owner, s.writer, s.writer_pointers));
  if (!bound)
    return Status::FailedPrecondition("slot store original write permissions were reset or withdrawn");
  HYLE_ASSIGN_OR_RETURN(auto expected_ids, s.indices.ready.view());
  if (!same_storage(ids.storage(), expected_ids))
    return Status::InvalidArgument("slot store IDs differ from its private validated carrier");
  const auto input = rows.storage();
  if (input.resource() != p.resource || input.offset() > SIZE_MAX - input.size())
    return Status::InvalidArgument("slot store rows differ from its original Queue storage");
  for (std::size_t i = 0; i < s.bindings.destination_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto target, s.bindings.destinations()[i].view());
    if (target.resource() != p.resource || target.offset() > SIZE_MAX - target.size())
      return Status::InvalidArgument("slot store destination range changed");
    if (input.native_handle() == target.native_handle() &&
        input.offset() < target.offset() + target.size() &&
        target.offset() < input.offset() + input.size())
      return Status::InvalidArgument("slot store rows overlap an original destination");
    s.destinations[i] = target;
  }
  return Status::Ok();
}

const gd::RegionBindingReleaseVTable binding_release{
    [](void *, void *) noexcept { return Status::Ok(); },
    [](void *, void *opaque) noexcept {
      auto &slot = *static_cast<Slot *>(opaque);
      slot.rows = {}; slot.ids = {}; slot.sealed = false;
    }, true};
const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Prepared *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return c::AliasSupport{}; },
    .create_invocation = [](void *opaque, mem::AllocatorRef a, mem::BufferView, void **out) noexcept -> Status {
      *out = nullptr;
      auto *slot = a.construct<Slot>(a);
      if (!slot) return Status::ResourceExhausted("slot store invocation metadata");
      try {
        const auto count = static_cast<std::size_t>(static_cast<Prepared *>(opaque)->rows.dimensions()[0]);
        slot->writer_pointers.resize(count);
        slot->destinations.resize(count);
      }
      catch (...) { a.destroy(slot); return Status::ResourceExhausted("slot store selected writer pointers"); }
      *out = slot;
      return Status::Ok();
    },
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .shared_submit = [](void *opaque, void *invocation, cd::QueueSequence &sequence,
        std::span<const c::ConstTensorView> inputs, std::span<const c::TensorView> outputs) noexcept -> cd::EnqueueResult {
      const auto &p = *static_cast<Prepared *>(opaque);
      auto &s = *static_cast<Slot *>(invocation);
      if (inputs.size() != 2 || !outputs.empty() || !s.sealed || !s.queue.same_owner(sequence))
        return cd::EnqueueResult::rejected(Status::FailedPrecondition("slot store needs its original sealed Queue episode"));
      if (inputs[p.rows_port].description() != p.rows || inputs[p.ids_port].description() != p.ids ||
          !same_storage(inputs[p.rows_port].storage(), s.rows.storage()) ||
          !same_storage(inputs[p.ids_port].storage(), s.ids.storage()))
        return cd::EnqueueResult::rejected(Status::FailedPrecondition("slot store inputs differ from their admitted cold spans"));
      const auto bytes = p.rows.required_bytes().value() /
          static_cast<std::size_t>(p.rows.dimensions()[0]);
      bool accepted = false;
      for (std::size_t i = 0; i < s.destinations.size(); ++i) {
        auto source = inputs[p.rows_port].storage().subview(i * bytes, bytes);
        if (!source) return {accepted, source.error()};
        auto command = cd::TransferAccess::enqueue_copy(sequence, s.destinations[i], *source);
        accepted |= command.accepted;
        if (!command.status.ok()) return {accepted, std::move(command.status)};
      }
      return {accepted, Status::Ok()};
    },
    .prepare_bindings = [](void *opaque, void *invocation,
        std::span<const RegionValue> inputs, std::span<const c::TensorView> outputs) noexcept -> Status {
      const auto &p = *static_cast<Prepared *>(opaque);
      auto &s = *static_cast<Slot *>(invocation); s.sealed = false;
      if (inputs.size() != 2 || !outputs.empty()) return Status::InvalidArgument("slot store boundary changed");
      HYLE_RETURN_IF_ERROR(validate_rows(p, s, inputs[p.rows_port].tensor, inputs[p.ids_port].tensor));
      s.rows = inputs[p.rows_port].tensor; s.ids = inputs[p.ids_port].tensor;
      return Status::Ok();
    },
    .seal_bindings = [](void *opaque, void *invocation) noexcept -> Status {
      auto &s = *static_cast<Slot *>(invocation);
      HYLE_RETURN_IF_ERROR(validate_rows(*static_cast<Prepared *>(opaque), s, s.rows, s.ids));
      s.sealed = true; return Status::Ok();
    },
    .query_extension = [](void *, c::ProviderExtensionId id) noexcept -> const void * {
      return id == gd::kRegionBindingReleaseExtensionId ? &binding_release : nullptr;
    }};

struct Description final {
  Store store;
  ValueId rows, ids;
  c::TensorDesc rows_description, ids_description;
  std::uint32_t rows_port = 0, ids_port = 1;
};
Result<Description> describe(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.nodes.size() != 1 || request.inputs.size() != 2 ||
      !request.outputs.empty() || !request.has_unknown_effects)
    return std::unexpected(Status::Unimplemented("slot store needs its original external-write singleton"));
  const auto &graph = *request.graph;
  HYLE_ASSIGN_OR_RETURN(auto operation, graph.operation(request.nodes[0]));
  const auto store = operation.custom_as<Store>();
  if (!store) return std::unexpected(Status::Unimplemented("not a slot store"));
  Description result; result.store = *store;
  HYLE_ASSIGN_OR_RETURN(result.rows, graph.node_input(request.nodes[0], 0));
  HYLE_ASSIGN_OR_RETURN(result.ids, graph.node_input(request.nodes[0], 1));
  if (result.ids.kind() != ValueKind::kInput)
    return std::unexpected(Status::Unimplemented("slot store IDs must be its private caller input"));
  for (std::size_t i = 0; i < graph.output_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto output, graph.output_id(i));
    if (output == result.ids) return std::unexpected(Status::Unimplemented("private slot IDs cannot be published"));
  }
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    if (node == request.nodes[0]) continue;
    HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
    for (std::size_t port = 0; port < count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(node, port));
      if (input == result.ids)
        return std::unexpected(Status::Unimplemented("slot store IDs are shared with another node"));
    }
  }
  HYLE_ASSIGN_OR_RETURN(result.rows_description, graph.value_description(result.rows));
  HYLE_ASSIGN_OR_RETURN(result.ids_description, graph.value_description(result.ids));
  const std::array descriptions{result.rows_description, result.ids_description};
  HYLE_RETURN_IF_ERROR(store->infer_output_descriptions(descriptions, {}));
  const auto rows = std::ranges::find(request.inputs, result.rows);
  const auto ids = std::ranges::find(request.inputs, result.ids);
  if (rows == request.inputs.end() || ids == request.inputs.end())
    return std::unexpected(Status::InvalidArgument("slot store original input boundary changed"));
  result.rows_port = static_cast<std::uint32_t>(rows - request.inputs.begin());
  result.ids_port = static_cast<std::uint32_t>(ids - request.inputs.begin());
  return result;
}
c::ProviderRuntimeProperties runtime(const detail::StoreAdapter &adapter) noexcept {
  return {.mode = c::ProviderRuntimeMode::kSharedQueue,
      .workspace_owner = c::ProviderWorkspaceOwner::kCaller,
      .replay_class = c::ProviderReplayClass::kPrepared,
      .blocks_host = adapter.device_class == c::DeviceClass::kCpu,
      .uses_private_threads = false, .supports_concurrent_invocations = true,
      .capture_compatible = false};
}
Status propose(void *, const RegionDiscoveryRequest &request, mem::AllocatorRef,
    RegionProposalWriter &writer) noexcept {
  for (std::size_t i = 0; i < request.graph.node_count(); ++i) {
    if (!request.allowed_nodes.empty() && !request.allowed_nodes[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto kind, request.graph.operation_kind(node));
    if (kind != c::OperationKind::kCustom) continue;
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    if (!operation.custom_as<Store>()) continue;
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes, .route = 1,
        .replay_class = c::ProviderReplayClass::kPrepared,
        .implementation = c::ProviderImplementationId{0x534c4f5453544f52ULL}}));
    if (!added) break;
  }
  return Status::Ok();
}
Status resources(void *opaque, const RegionCompileRequest &request, mem::AllocatorRef,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *out) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto description, describe(request));
  (void)description;
  if (inputs.size() != 2 || !outputs.empty()) return Status::Unimplemented("slot store resource boundary changed");
  const auto &adapter = *static_cast<detail::StoreAdapter *>(opaque);
  for (auto &input : inputs) {
    input.resource.required_properties = adapter.device_class == c::DeviceClass::kCpu
        ? mem::MemoryProperty::kHostAccessible : mem::MemoryProperty::kDeviceAccessible;
    input.resource.minimum_alignment = 1;
  }
  *out = {.runtime = runtime(adapter), .preserves_effects = true};
  return Status::Ok();
}
c::ProviderPrepareOutcome compile(void *opaque, const RegionCompileRequest &request,
    mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  *out = {};
  auto description = describe(request);
  if (!description) return description.error().IsUnimplemented()
      ? c::ProviderPrepareOutcome::unsupported(description.error())
      : c::ProviderPrepareOutcome::error(description.error());
  const auto &adapter = *static_cast<detail::StoreAdapter *>(opaque);
  if (request.input_resources.size() != 2 ||
      std::ranges::any_of(request.input_resources, [&](const auto &resource) { return resource != adapter.resource; }))
    return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("slot store needs its original Queue resources"));
  auto *prepared = allocator.construct<Prepared>();
  if (!prepared) return c::ProviderPrepareOutcome::error(Status::ResourceExhausted("slot store prepared metadata"));
  prepared->store = description->store; prepared->node = request.nodes[0];
  prepared->rows_value = description->rows; prepared->ids_value = description->ids;
  prepared->rows = description->rows_description; prepared->ids = description->ids_description;
  prepared->rows_port = description->rows_port; prepared->ids_port = description->ids_port;
  prepared->resource = adapter.resource; prepared->runtime = runtime(adapter);
  *out = {.vtable = &prepared_table, .state = prepared,
      .properties = {.workspace = {0, 1}, .persistent_bytes = sizeof(Prepared),
          .dispatch_count = static_cast<std::uint32_t>(prepared->rows.dimensions()[0]),
          .runtime = prepared->runtime, .preserves_effects = true,
          .invocation_private_bytes = std::nullopt}};
  return c::ProviderPrepareOutcome::prepared();
}
const RegionProviderVTable provider_table{&propose, &resources, &compile};
struct Find final {
  NodeId node;
  Slot *slot = nullptr;
  Prepared *prepared = nullptr;
  gd::ProjectedWriter proof{};
  ExecutionLaneId lane;
};
Status find(void *opaque, const gd::RegionAccessView &view) noexcept {
  auto &found = *static_cast<Find *>(opaque);
  if (view.prepared_vtable != &prepared_table) return Status::Ok();
  auto *prepared = static_cast<Prepared *>(view.prepared);
  if (!prepared || prepared->node != found.node) return Status::Ok();
  if (found.slot || view.node_ordinals.size() != 1 ||
      view.node_ordinals[0] != found.node.ordinal() || !view.invocation)
    return Status::InvalidArgument("slot store selected artifact/slot identity differs");
  found.slot = static_cast<Slot *>(view.invocation); found.prepared = prepared;
  found.proof = {found.node, view.step, &prepared_table, prepared}; found.lane = view.lane;
  return Status::Ok();
}
} // namespace

namespace detail {
const RegionProviderVTable *store_provider() noexcept { return &provider_table; }
} // namespace detail

Status prepare(Invocation &invocation, std::span<const QueueBinding> queues,
    PrepareOptions options) noexcept {
  if (!options.allocator.valid() || !options.host_workspace.valid())
    return Status::InvalidArgument("slot preparation needs allocator and host workspace");
  const ResourceOptions resources{options.allocator, options.resource_budget};
  HYLE_RETURN_IF_ERROR(invocation.prepare_resources(resources));
  struct Call final { Invocation *invocation; std::span<const QueueBinding> queues; PrepareOptions options; } call{&invocation, queues, options};
  return gd::RegionAccess::visit(invocation, [](void *opaque, const gd::RegionAccessView &view) noexcept -> Status {
    if (view.prepared_vtable != &prepared_table) return Status::Ok();
    const auto &call = *static_cast<Call *>(opaque);
    const auto &p = *static_cast<Prepared *>(view.prepared);
    auto &s = *static_cast<Slot *>(view.invocation);
    if (s.bindings.valid()) return Status::FailedPrecondition("take original slot bindings before preparation");
    const auto first = std::ranges::find_if(call.queues, [&](const auto &binding) { return binding.lane == view.lane; });
    if (first == call.queues.end() || !first->queue ||
        std::count_if(call.queues.begin(), call.queues.end(), [&](const auto &binding) { return binding.lane == view.lane; }) != 1)
      return Status::InvalidArgument("slot preparation needs one original Queue per selected lane");
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::validate_queue(call.invocation->executable(), view.lane, *first->queue));
    HYLE_ASSIGN_OR_RETURN(auto queue, cd::QueueRef::acquire(*first->queue));
    const ResourceOptions resources{call.options.allocator, call.options.resource_budget};
    HYLE_ASSIGN_OR_RETURN(auto indices, Indices::create(p.resource, call.options.host_workspace,
        static_cast<std::size_t>(p.rows.dimensions()[0]), resources));
    if (!indices.direct) {
      HYLE_ASSIGN_OR_RETURN(auto destination, indices.writer.view());
      HYLE_ASSIGN_OR_RETURN(auto source, indices.staging.view());
      const auto transfer = c::transfer_properties(*first->queue, destination, source);
      if (!transfer)
        return transfer.error().WithContext(
            "slot host_workspace must support metadata transfer on the original Queue");
    }
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(*call.invocation, view.step));
    s.indices = std::move(indices); s.queue = std::move(queue); s.options = resources;
    s.prepared = true; s.sealed = false; s.rows = {}; s.ids = {};
    return Status::Ok();
  }, &call);
}

Status bind(Invocation &invocation, const Graph &graph, NodeId node,
    SlotBindings &&bindings) noexcept {
  Find found{node}; HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation, node, &find, &found));
  if (!found.slot) return Status::Unimplemented("Store has no selected original slot implementation");
  auto &s = *found.slot; const auto &p = *found.prepared;
  if (!s.prepared || !s.queue.is_open() || s.bindings.valid() || !bindings.valid())
    return Status::FailedPrecondition("slot bind needs its prepared open Queue and untaken original transaction");
  const auto layout = bindings.layout();
  HYLE_ASSIGN_OR_RETURN(auto schema, Schema::create(layout.description(), layout.slot_count(), layout.stride_bytes()));
  if (schema != p.store.schema || bindings.destination_count() != static_cast<std::size_t>(p.rows.dimensions()[0]) ||
      bindings.source_addresses().size() != bindings.source_count() ||
      bindings.destination_addresses().size() != bindings.destination_count())
    return Status::InvalidArgument("slot transaction differs from its selected Store geometry");
  for (std::size_t i = 0; i < bindings.source_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto address, layout.locate(bindings.sources()[i]));
    if (address != bindings.source_addresses()[i]) return Status::InvalidArgument("slot source owner/generation changed");
  }
  for (std::size_t i = 0; i < bindings.destination_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto address, layout.locate(bindings.destinations()[i]));
    if (address != bindings.destination_addresses()[i]) return Status::InvalidArgument("slot destination owner/generation changed");
    s.writer_pointers[i] = &bindings.destinations()[i];
  }
  HYLE_ASSIGN_OR_RETURN(auto carrier, cd::LeaseAccess::slot_capacity(layout));
  if (carrier.resource() != p.resource) return Status::InvalidArgument("slot pool differs from its selected Queue resource");
  HYLE_ASSIGN_OR_RETURN(auto description, schema.capacity_description());
  HYLE_ASSIGN_OR_RETURN(auto capacity, c::ConstTensorView::create(description, carrier));
  s.sealed = false;
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation, found.proof.step));
  HYLE_RETURN_IF_ERROR(s.indices.upload(s.queue, bindings.destination_addresses()));
  HYLE_ASSIGN_OR_RETURN(auto ids, s.indices.ready.tensor(p.ids));
  HYLE_RETURN_IF_ERROR(invocation.bind(p.ids_value, ids));
  const std::array controls{s.indices.ready};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_write_effect(invocation, graph, capacity,
      bindings.sources(), s.writer_pointers, {}, found.proof, controls, s.options));
  s.owner = gd::InvocationAccess::state(invocation); s.writer = found.proof;
  s.bindings = std::move(bindings);
  return Status::Ok();
}

Result<SlotBindings> take(Invocation &invocation, NodeId node) noexcept {
  Find found{node}; HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit_retired(invocation, node, &find, &found));
  if (!found.slot || !found.slot->bindings.valid())
    return std::unexpected(Status::InvalidArgument("Store has no retained original slot transaction"));
  HYLE_ASSIGN_OR_RETURN(auto bound, gd::RegionAccess::has_write_effect(
      found.slot->owner, found.slot->writer, found.slot->writer_pointers));
  if (bound)
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_write_effect(invocation, node));
  // A reset/release may already have withdrawn this exact old effect. The
  // retired visitor above still excludes native work/results/observations;
  // returning the original owner neither reopens execution nor publishes it.
  found.slot->sealed = false; found.slot->rows = {}; found.slot->ids = {};
  found.slot->owner = nullptr; found.slot->writer = {};
  return std::move(found.slot->bindings);
}

} // namespace hyle::graph::slots
