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
bool same_configuration(const detail::StoreConfiguration &a,
    const detail::StoreConfiguration &b) noexcept {
  const auto same = [](c::ConstTensorView x, c::ConstTensorView y) noexcept {
    return x.valid() == y.valid() && (!x.valid() ||
        (x.description() == y.description() && same_storage(x.storage(), y.storage())));
  };
  return same(a.capacity, b.capacity) && same(a.source_indices, b.source_indices) &&
      same(a.destination_indices, b.destination_indices);
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
      std::span<const c::LeaseSlotAddress> sources,
      std::span<const c::LeaseSlotAddress> destinations) noexcept {
    if (failed || pending.valid() || (!direct && uses.active_batches()))
      return Status::FailedPrecondition("slot index upload is failed or not actually retired");
    if (ready.valid()) { HYLE_ASSIGN_OR_RETURN(writer, ready.try_write()); }
    if (source.valid()) { HYLE_ASSIGN_OR_RETURN(staging, source.try_write()); }
    auto &host_writer = direct ? writer : staging;
    HYLE_ASSIGN_OR_RETURN(auto host, host_writer.view());
    if (sources.size() > SIZE_MAX - destinations.size() ||
        sources.size() + destinations.size() != host.size() / sizeof(std::int32_t))
      return Status::InvalidArgument("slot index count differs from its fixed carrier");
    HYLE_ASSIGN_OR_RETURN(auto mapping, host.map_host());
    HYLE_ASSIGN_OR_RETURN(auto bytes, mapping.writable_bytes());
    for (std::size_t i = 0; i < sources.size() + destinations.size(); ++i) {
      const auto &address = i < sources.size() ? sources[i] : destinations[i - sources.size()];
      const auto id = static_cast<std::int32_t>(address.index);
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
} // namespace

namespace detail {
struct StoreBindingData final {
  explicit StoreBindingData(mem::AllocatorRef selected) noexcept
      : allocator(selected),
        writer_pointers(mem::StdAllocator<const c::WriteLease *>{selected}) {}
  void destroy() noexcept { const auto selected = allocator; selected.destroy(this); }
  mem::AllocatorRef allocator;
  SlotBindings bindings;
  Indices indices;
  cd::QueueRef queue;
  ResourceOptions options;
  std::vector<const c::WriteLease *, mem::StdAllocator<const c::WriteLease *>> writer_pointers;
  StoreRegionFacts facts;
  StoreSource artifact_source;
  bool logical_source = false;
  StoreConfiguration configuration;
  StoreConfiguration last_configuration;
  // Stable original state, not the movable public Invocation wrapper.
  const gd::InvocationState *owner = nullptr;
  gd::ProjectedWriter writer{};
};
struct StoreBindingAccess final {
  static StoreBindingData *get(StoreBinding &binding) noexcept { return binding.data_; }
  static void replace(StoreBinding &binding, StoreBindingData *data) noexcept {
    if (binding.data_) binding.data_->destroy();
    binding.data_ = data;
  }
};
StoreBinding::StoreBinding(StoreBinding &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)) {}
StoreBinding &StoreBinding::operator=(StoreBinding &&other) noexcept {
  if (this != &other) {
    StoreBinding next(std::move(other)); std::swap(data_, next.data_);
  }
  return *this;
}
StoreBinding::~StoreBinding() { if (data_) data_->destroy(); }
bool StoreBinding::prepared() const noexcept { return data_ != nullptr; }
bool StoreBinding::valid() const noexcept { return data_ && data_->bindings.valid(); }
bool StoreBinding::same_queue(const cd::QueueSequence &sequence) const noexcept {
  return data_ && data_->queue.same_owner(sequence);
}
StoreConfiguration StoreBinding::configuration() const noexcept {
  return data_ ? data_->configuration : StoreConfiguration{};
}
const SlotBindings *StoreBinding::transaction() const noexcept {
  return valid() ? &data_->bindings : nullptr;
}
Status StoreBinding::validate_authority() const noexcept {
  if (!valid() || !data_->indices.ready.valid())
    return Status::FailedPrecondition("slot store has no current original transaction and controls");
  HYLE_ASSIGN_OR_RETURN(auto bound,
      gd::RegionAccess::has_write_effect(data_->owner, data_->writer, data_->writer_pointers));
  if (!bound)
    return Status::FailedPrecondition("slot store original write permissions were reset or withdrawn");
  const auto input = [&](ValueId value, c::ConstTensorView expected) noexcept -> Status {
    HYLE_ASSIGN_OR_RETURN(auto matches, gd::RegionAccess::has_input_binding(data_->owner, value, expected));
    return matches ? Status::Ok() : Status::FailedPrecondition("slot store original pool/IDs input was replaced or withdrawn");
  };
  HYLE_RETURN_IF_ERROR(input(data_->facts.destination_indices, data_->configuration.destination_indices));
  if (data_->facts.source.count) {
    HYLE_RETURN_IF_ERROR(input(data_->facts.source.capacity, data_->configuration.capacity));
    HYLE_RETURN_IF_ERROR(input(data_->facts.source.indices, data_->configuration.source_indices));
    HYLE_ASSIGN_OR_RETURN(auto source, gd::RegionAccess::has_read_projection(
        data_->owner, data_->facts.source.capacity, data_->bindings.sources()));
    if (!source)
      return Status::FailedPrecondition("slot store original source permissions were reset or withdrawn");
  }
  return Status::Ok();
}
Status StoreBinding::validate_configuration(const StoreConfiguration &actual) const noexcept {
  HYLE_RETURN_IF_ERROR(validate_authority());
  if (!same_configuration(actual, data_->configuration))
    return Status::InvalidArgument("slot store differs from its private validated capacity/IDs");
  return Status::Ok();
}
} // namespace detail

namespace {

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
      : destinations(mem::StdAllocator<mem::BufferView>{allocator}) {}
  detail::StoreBinding binding;
  std::vector<mem::BufferView, mem::StdAllocator<mem::BufferView>> destinations;
  c::ConstTensorView rows, ids;
  bool sealed = false;
};

Status validate_rows(const Prepared &p, Slot &s,
    c::ConstTensorView rows, c::ConstTensorView ids) noexcept {
  if (rows.description() != p.rows || ids.description() != p.ids)
    return Status::FailedPrecondition("slot store has no current exact transaction and controls");
  auto configuration = s.binding.configuration();
  configuration.destination_indices = ids;
  HYLE_RETURN_IF_ERROR(s.binding.validate_configuration(configuration));
  const auto &bindings = *s.binding.transaction();
  const auto input = rows.storage();
  if (input.resource() != p.resource || input.offset() > SIZE_MAX - input.size())
    return Status::InvalidArgument("slot store rows differ from its original Queue storage");
  for (std::size_t i = 0; i < bindings.destination_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto target, bindings.destinations()[i].view());
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
const detail::StoreRegionVTable store_binding{
    .inspect = [](const ProviderRegionPrepared &prepared, void *invocation) noexcept
        -> Result<detail::StoreRegionFacts> {
      if (!prepared.state || !invocation)
        return std::unexpected(Status::InvalidArgument("slot store needs its original artifact/slot"));
      const auto &p = *static_cast<Prepared *>(prepared.state);
      return detail::StoreRegionFacts{.store = p.node, .schema = p.store.schema,
          .destination_indices = p.ids_value, .destination_input_port = p.ids_port,
          .destination_count = static_cast<std::size_t>(p.rows.dimensions()[0]),
          .resource = p.resource};
    },
    .binding = [](void *invocation) noexcept {
      return &static_cast<Slot *>(invocation)->binding;
    },
    .configure = [](const ProviderRegionPrepared &prepared, void *invocation,
        const detail::StoreConfiguration &configuration) noexcept -> Status {
      const auto &p = *static_cast<Prepared *>(prepared.state);
      if (!configuration.capacity.valid() ||
          configuration.destination_indices.description() != p.ids)
        return Status::InvalidArgument("slot store private configuration differs");
      auto &s = *static_cast<Slot *>(invocation);
      s.rows = {}; s.ids = {}; s.sealed = false;
      return Status::Ok();
    },
    .clear_configuration = [](const ProviderRegionPrepared &, void *invocation) noexcept {
      auto &s = *static_cast<Slot *>(invocation);
      s.rows = {}; s.ids = {}; s.sealed = false;
    }};
const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Prepared *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return c::AliasSupport{}; },
    .create_invocation = [](void *opaque, mem::AllocatorRef a, mem::BufferView, void **out) noexcept -> Status {
      *out = nullptr;
      auto *slot = a.construct<Slot>(a);
      if (!slot) return Status::ResourceExhausted("slot store invocation metadata");
      try {
        const auto count = static_cast<std::size_t>(static_cast<Prepared *>(opaque)->rows.dimensions()[0]);
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
      if (inputs.size() != 2 || !outputs.empty() || !s.sealed || !s.binding.same_queue(sequence))
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
      if (id == gd::kRegionBindingReleaseExtensionId) return &binding_release;
      if (id == detail::kStoreBindingExtensionId) return &store_binding;
      return nullptr;
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
  detail::StoreBinding *binding = nullptr;
  const detail::StoreRegionVTable *interface = nullptr;
  ProviderRegionPrepared prepared{};
  void *invocation = nullptr;
  detail::StoreRegionFacts facts;
  gd::ProjectedWriter proof{};
  ExecutionLaneId lane;
};
struct LogicalSource final {
  NodeId store;
  Schema schema;
  ValueId destination_indices;
  std::size_t destination_count = 0;
  detail::StoreSource source;
};
bool same_source(const detail::StoreSource &a, const detail::StoreSource &b) noexcept {
  return a.gather == b.gather && a.capacity == b.capacity && a.indices == b.indices &&
      a.count == b.count && a.capacity_input_port == b.capacity_input_port &&
      a.permission_changes_preserve_commands == b.permission_changes_preserve_commands;
}
bool same_logical_source(const detail::StoreSource &a, const detail::StoreSource &b) noexcept {
  return a.gather == b.gather && a.capacity == b.capacity && a.indices == b.indices && a.count == b.count;
}
Status private_input(const Graph &graph, ValueId value, NodeId reader,
    std::size_t reader_port) noexcept {
  if (!value.valid() || value.kind() != ValueKind::kInput)
    return Status::InvalidArgument("slot pool and controls need original Graph inputs");
  for (std::size_t i = 0; i < graph.output_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto output, graph.output_id(i));
    if (output == value) return Status::InvalidArgument("slot pool/control cannot be published");
  }
  bool consumed = false;
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
    for (std::size_t port = 0; port < count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(node, port));
      if (input != value) continue;
      if (node != reader || port != reader_port)
        return Status::InvalidArgument("slot pool/control has an unrelated logical consumer");
      consumed = true;
    }
  }
  return consumed ? Status::Ok() : Status::InvalidArgument("slot pool/control has no original reader");
}
Result<LogicalSource> describe_source(const Graph &graph, SourceBinding binding) noexcept {
  if (!graph.valid() || !binding.store.valid() || !binding.gather.valid() || binding.store == binding.gather)
    return std::unexpected(Status::InvalidArgument("slot source association needs its original Graph and distinct nodes"));
  HYLE_ASSIGN_OR_RETURN(auto store_operation, graph.operation(binding.store));
  const auto store = store_operation.custom_as<Store>();
  if (!store) return std::unexpected(Status::InvalidArgument("slot source association needs an original Store"));
  HYLE_ASSIGN_OR_RETURN(auto gather_operation, graph.operation(binding.gather));
  const auto *gather = gather_operation.get_if<c::Gather>();
  if (!gather || (gather->axis != 0 && gather->axis != -2) ||
      gather->bounds != c::IndexBoundsPolicy::kChecked)
    return std::unexpected(Status::InvalidArgument("slot source needs a checked axis-zero Gather"));
  HYLE_ASSIGN_OR_RETURN(auto pool, graph.node_input(binding.gather, 0));
  HYLE_ASSIGN_OR_RETURN(auto source_ids, graph.node_input(binding.gather, 1));
  HYLE_ASSIGN_OR_RETURN(auto destination_ids, graph.node_input(binding.store, 1));
  if (pool == source_ids || pool == destination_ids || source_ids == destination_ids)
    return std::unexpected(Status::InvalidArgument("slot source and destination controls must be distinct"));
  HYLE_RETURN_IF_ERROR(private_input(graph, pool, binding.gather, 0));
  HYLE_RETURN_IF_ERROR(private_input(graph, source_ids, binding.gather, 1));
  HYLE_RETURN_IF_ERROR(private_input(graph, destination_ids, binding.store, 1));
  HYLE_ASSIGN_OR_RETURN(auto capacity_description, graph.value_description(pool));
  HYLE_ASSIGN_OR_RETURN(auto expected_capacity, store->schema.capacity_description());
  HYLE_ASSIGN_OR_RETURN(auto ids_description, graph.value_description(source_ids));
  HYLE_ASSIGN_OR_RETURN(auto gathered, graph.result(binding.gather, 0));
  HYLE_ASSIGN_OR_RETURN(auto gathered_description, graph.value_description(gathered));
  if (capacity_description != expected_capacity ||
      !ids_description.is_plain() || !ids_description.is_dense() ||
      ids_description.data_type() != c::DataType::kInt32 || ids_description.rank() != 1 ||
      ids_description.dimensions()[0] <= 0 || ids_description.dimensions()[0] > INT32_MAX)
    return std::unexpected(Status::InvalidArgument("slot Gather pool/IDs differ from Store geometry"));
  const auto source_count = static_cast<std::size_t>(ids_description.dimensions()[0]);
  HYLE_ASSIGN_OR_RETURN(auto expected_gathered, c::TensorDesc::dense(store->schema.type(),
      std::array<std::int64_t, 2>{static_cast<std::int64_t>(source_count), store->schema.item_elements()}));
  if (gathered_description != expected_gathered)
    return std::unexpected(Status::InvalidArgument("slot Gather must materialize same-type whole rows"));
  HYLE_ASSIGN_OR_RETURN(auto rows, graph.node_input(binding.store, 0));
  HYLE_ASSIGN_OR_RETURN(auto rows_description, graph.value_description(rows));
  HYLE_ASSIGN_OR_RETURN(auto destination_description, graph.value_description(destination_ids));
  const std::array inputs{rows_description, destination_description};
  HYLE_RETURN_IF_ERROR(store->infer_output_descriptions(inputs, {}));
  return LogicalSource{binding.store, store->schema, destination_ids,
      static_cast<std::size_t>(rows_description.dimensions()[0]),
      {.gather = binding.gather, .capacity = pool, .indices = source_ids, .count = source_count}};
}
Status matches(const detail::StoreRegionFacts &facts, const LogicalSource &logical) noexcept {
  if (facts.store != logical.store || facts.schema != logical.schema ||
      facts.destination_indices != logical.destination_indices || facts.destination_count != logical.destination_count ||
      (facts.source.count && !same_logical_source(facts.source, logical.source)))
    return Status::InvalidArgument("slot logical association differs from its selected Store/source artifact");
  return Status::Ok();
}
const detail::StoreRegionVTable *store_interface(const gd::RegionAccessView &view) noexcept {
  return view.prepared_vtable && view.prepared_vtable->query_extension
      ? static_cast<const detail::StoreRegionVTable *>(view.prepared_vtable->query_extension(
          view.prepared, detail::kStoreBindingExtensionId)) : nullptr;
}
Result<detail::StoreRegionFacts> inspect(const gd::RegionAccessView &view,
    const detail::StoreRegionVTable &interface) noexcept {
  if (!view.prepared || !view.invocation || !interface.inspect || !interface.binding ||
      !interface.configure || !interface.clear_configuration)
    return std::unexpected(Status::InvalidArgument("slot store cold extension is incomplete"));
  const ProviderRegionPrepared prepared{.vtable = view.prepared_vtable, .state = view.prepared};
  HYLE_ASSIGN_OR_RETURN(auto facts, interface.inspect(prepared, view.invocation));
  if (!facts.store.valid() || !facts.schema.valid() || !facts.resource.valid() ||
      !facts.destination_indices.valid() || facts.destination_indices.kind() != ValueKind::kInput ||
      facts.destination_input_port == UINT32_MAX || !facts.destination_count ||
      facts.destination_count > facts.schema.slot_count() ||
      facts.source.count > static_cast<std::size_t>(INT32_MAX) ||
      facts.source.count > SIZE_MAX - facts.destination_count ||
      std::ranges::find(view.node_ordinals, facts.store.ordinal()) == view.node_ordinals.end())
    return std::unexpected(Status::InvalidArgument("slot store cold geometry/artifact facts differ"));
  if (facts.source.count && (!facts.source.gather.valid() ||
      !facts.source.capacity.valid() || !facts.source.indices.valid() ||
      facts.source.capacity.kind() != ValueKind::kInput ||
      facts.source.indices.kind() != ValueKind::kInput ||
      facts.source.capacity == facts.source.indices ||
      facts.source.capacity == facts.destination_indices ||
      facts.source.indices == facts.destination_indices ||
      facts.source.capacity_input_port == UINT32_MAX ||
      std::ranges::find(view.node_ordinals, facts.source.gather.ordinal()) == view.node_ordinals.end()))
    return std::unexpected(Status::InvalidArgument("slot source cold reader/artifact facts differ"));
  return facts;
}
Status find(void *opaque, const gd::RegionAccessView &view) noexcept {
  auto &found = *static_cast<Find *>(opaque);
  const auto *interface = store_interface(view);
  if (!interface) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto facts, inspect(view, *interface));
  if (facts.store != found.node) return Status::Ok();
  if (found.binding)
    return Status::InvalidArgument("slot store selected artifact/slot identity differs");
  found.binding = interface->binding(view.invocation);
  if (!found.binding) return Status::InvalidArgument("slot store has no original binding owner");
  found.interface = interface; found.prepared = {.vtable = view.prepared_vtable, .state = view.prepared};
  found.invocation = view.invocation; found.facts = facts;
  found.proof = {found.node, view.step, view.prepared_vtable, view.prepared}; found.lane = view.lane;
  return Status::Ok();
}
} // namespace

namespace detail {
const RegionProviderVTable *store_provider() noexcept { return &provider_table; }
} // namespace detail

namespace {
Status prepare_impl(Invocation &invocation, const Graph *graph,
    std::span<const QueueBinding> queues, std::span<const SourceBinding> sources,
    PrepareOptions options) noexcept {
  if (!options.allocator.valid() || !options.host_workspace.valid())
    return Status::InvalidArgument("slot preparation needs allocator and host workspace");
  // Reject foreign/duplicate logical associations before replacing any slot's
  // existing carriers. Selected arithmetic remains owned by its original plan.
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (!graph) return Status::InvalidArgument("slot source associations need their original Graph");
    HYLE_ASSIGN_OR_RETURN(auto logical, describe_source(*graph, sources[i]));
    Find selected{sources[i].store};
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation, sources[i].store, &find, &selected));
    if (!selected.binding) return Status::Unimplemented("associated Store has no selected slot implementation");
    HYLE_RETURN_IF_ERROR(matches(selected.facts, logical));
    for (std::size_t j = 0; j < i; ++j) {
      if (sources[i].store == sources[j].store || sources[i].gather == sources[j].gather)
        return Status::InvalidArgument("slot source association duplicates a Store, reader or pool");
      HYLE_ASSIGN_OR_RETURN(auto previous_pool, graph->node_input(sources[j].gather, 0));
      if (logical.source.capacity == previous_pool)
        return Status::InvalidArgument("slot source association duplicates a Store, reader or pool");
    }
  }
  const ResourceOptions resources{options.allocator, options.resource_budget};
  HYLE_RETURN_IF_ERROR(invocation.prepare_resources(resources));
  struct Call final {
    Invocation *invocation; const Graph *graph;
    std::span<const QueueBinding> queues; std::span<const SourceBinding> sources;
    PrepareOptions options;
  } call{&invocation, graph, queues, sources, options};
  return gd::RegionAccess::visit(invocation, [](void *opaque, const gd::RegionAccessView &view) noexcept -> Status {
    const auto *interface = store_interface(view);
    if (!interface) return Status::Ok();
    const auto &call = *static_cast<Call *>(opaque);
    HYLE_ASSIGN_OR_RETURN(auto facts, inspect(view, *interface));
    const auto artifact_source = facts.source;
    bool logical_source = false;
    const auto association = std::ranges::find(call.sources, facts.store, &SourceBinding::store);
    if (association != call.sources.end()) {
      HYLE_ASSIGN_OR_RETURN(auto logical, describe_source(*call.graph, *association));
      HYLE_RETURN_IF_ERROR(matches(facts, logical));
      if (!facts.source.count) { facts.source = logical.source; logical_source = true; }
    }
    auto *binding = interface->binding(view.invocation);
    if (!binding) return Status::InvalidArgument("slot store has no original binding owner");
    if (binding->valid()) return Status::FailedPrecondition("take original slot bindings before preparation");
    const auto first = std::ranges::find_if(call.queues, [&](const auto &binding) { return binding.lane == view.lane; });
    if (first == call.queues.end() || !first->queue ||
        std::count_if(call.queues.begin(), call.queues.end(), [&](const auto &binding) { return binding.lane == view.lane; }) != 1)
      return Status::InvalidArgument("slot preparation needs one original Queue per selected lane");
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::validate_queue(call.invocation->executable(), view.lane, *first->queue));
    HYLE_ASSIGN_OR_RETURN(auto queue, cd::QueueRef::acquire(*first->queue));
    const ResourceOptions resources{call.options.allocator, call.options.resource_budget};
    auto allocator = resources.allocator;
    if (resources.resource_budget.valid()) {
      HYLE_ASSIGN_OR_RETURN(allocator, resources.resource_budget.metadata_allocator());
    }
    auto *data = allocator.construct<detail::StoreBindingData>(allocator);
    if (!data) return Status::ResourceExhausted("slot store original binding metadata");
    detail::StoreBinding next;
    detail::StoreBindingAccess::replace(next, data);
    try { data->writer_pointers.resize(facts.destination_count); }
    catch (...) { return Status::ResourceExhausted("slot store selected writer pointers"); }
    HYLE_ASSIGN_OR_RETURN(auto indices, Indices::create(facts.resource, call.options.host_workspace,
        facts.source.count + facts.destination_count, resources));
    if (!indices.direct) {
      HYLE_ASSIGN_OR_RETURN(auto destination, indices.writer.view());
      HYLE_ASSIGN_OR_RETURN(auto source, indices.staging.view());
      const auto transfer = c::transfer_properties(*first->queue, destination, source);
      if (!transfer)
        return transfer.error().WithContext(
            "slot host_workspace must support metadata transfer on the original Queue");
    }
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(*call.invocation, view.step));
    interface->clear_configuration({.vtable = view.prepared_vtable, .state = view.prepared}, view.invocation);
    data->indices = std::move(indices); data->queue = std::move(queue);
    data->options = resources; data->facts = facts;
    data->artifact_source = artifact_source; data->logical_source = logical_source;
    *binding = std::move(next);
    return Status::Ok();
  }, &call);
}
} // namespace

Status prepare(Invocation &invocation, std::span<const QueueBinding> queues,
    PrepareOptions options) noexcept {
  return prepare_impl(invocation, nullptr, queues, {}, options);
}
Status prepare(Invocation &invocation, const Graph &graph,
    std::span<const QueueBinding> queues, std::span<const SourceBinding> sources,
    PrepareOptions options) noexcept {
  return prepare_impl(invocation, &graph, queues, sources, options);
}

Status bind(Invocation &invocation, const Graph &graph, NodeId node,
    SlotBindings &&bindings) noexcept {
  Find found{node}; HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation, node, &find, &found));
  if (!found.binding) return Status::Unimplemented("Store has no selected original slot implementation");
  auto *data = detail::StoreBindingAccess::get(*found.binding);
  if (!data || !data->queue.is_open() || data->bindings.valid() || !bindings.valid())
    return Status::FailedPrecondition("slot bind needs its prepared open Queue and untaken original transaction");
  auto &s = *data; const auto &p = s.facts;
  if (found.facts.store != p.store || found.facts.schema != p.schema ||
      found.facts.destination_indices != p.destination_indices ||
      found.facts.destination_count != p.destination_count || found.facts.resource != p.resource ||
      !same_source(found.facts.source, s.artifact_source))
    return Status::InvalidArgument("slot preparation differs from its selected artifact");
  if (s.logical_source) {
    // Graph identities name immutable nodes. Preparation already checked the
    // complete logical association; a cold bind only checks this same Graph,
    // without rescanning every layer for every decode step.
    HYLE_ASSIGN_OR_RETURN(auto operation, graph.operation(p.source.gather));
    (void)operation;
  }
  const auto layout = bindings.layout();
  HYLE_ASSIGN_OR_RETURN(auto schema, Schema::create(layout.description(), layout.slot_count(), layout.stride_bytes()));
  if (schema != p.schema || bindings.destination_count() != p.destination_count ||
      (p.source.count && bindings.source_count() != p.source.count) ||
      s.writer_pointers.size() != p.destination_count ||
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
  if (p.source.count) {
    // The projected input adds its guards before bind_write_effect performs
    // its own reservation. Reserve the actual combined footprint first.
    if (p.source.count > (UINT32_MAX - 1u) - p.destination_count - 1u)
      return Status::ResourceExhausted("slot projected permission count overflows");
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::reserve_reads(invocation,
        static_cast<std::uint32_t>(p.source.count + p.destination_count + 1), s.options));
  }
  // Changing only validated indices in a fixed private carrier does not change
  // native arguments. The selected artifact and real backend independently
  // qualify preservation; writer-only fallback retains its original policy.
  detail::StoreConfiguration configuration{.capacity = capacity};
  HYLE_ASSIGN_OR_RETURN(auto destination_description, c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(p.destination_count)}));
  const auto source_addresses = p.source.count ? bindings.source_addresses()
      : std::span<const c::LeaseSlotAddress>{};
  // Withdraw stale borrowed configuration before recycling the metadata
  // lease; no Graph permission or caller transaction is moved by this hook.
  const bool may_preserve = p.source.count && p.source.permission_changes_preserve_commands;
  if (!may_preserve)
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation, found.proof.step));
  else HYLE_RETURN_IF_ERROR(gd::RegionAccess::revoke_preparation(invocation));
  found.interface->clear_configuration(found.prepared, found.invocation);
  HYLE_RETURN_IF_ERROR(s.indices.upload(s.queue, source_addresses, bindings.destination_addresses()));
  HYLE_ASSIGN_OR_RETURN(auto carrier_view, s.indices.ready.view());
  HYLE_ASSIGN_OR_RETURN(auto destination_view, carrier_view.subview(
      p.source.count * sizeof(std::int32_t), p.destination_count * sizeof(std::int32_t)));
  HYLE_ASSIGN_OR_RETURN(configuration.destination_indices,
      c::ConstTensorView::create(destination_description, destination_view));
  if (p.source.count) {
    HYLE_ASSIGN_OR_RETURN(auto source_description, c::TensorDesc::dense(c::DataType::kInt32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(p.source.count)}));
    HYLE_ASSIGN_OR_RETURN(auto source_view, carrier_view.subview(0, p.source.count * sizeof(std::int32_t)));
    HYLE_ASSIGN_OR_RETURN(configuration.source_indices,
        c::ConstTensorView::create(source_description, source_view));
  }
  if (may_preserve && !same_configuration(configuration, s.last_configuration))
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation, found.proof.step));
  auto configured = found.interface->configure(found.prepared, found.invocation, configuration);
  if (!configured.ok()) {
    found.interface->clear_configuration(found.prepared, found.invocation);
    return configured;
  }
  bool source_bound = false;
  const auto install = [&]() noexcept -> Status {
    HYLE_RETURN_IF_ERROR(invocation.bind(p.destination_indices, configuration.destination_indices));
    if (p.source.count) {
      HYLE_RETURN_IF_ERROR(invocation.bind(p.source.indices, configuration.source_indices));
      if (s.logical_source) {
        const std::array readers{p.source.gather};
        HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_read_projection(invocation,
            {p.source.capacity, {}, readers}, capacity, bindings.sources()));
      } else {
        const std::array readers{gd::ProjectedReader{found.proof.step, found.proof.implementation,
            found.proof.prepared, p.source.capacity_input_port,
            p.source.permission_changes_preserve_commands}};
        HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_projected(invocation, p.source.capacity,
            capacity, bindings.sources(), readers));
      }
      source_bound = true;
    }
    // Indexed sources already belong to the original projected input. The
    // writer-only fallback keeps its existing anonymous source controls.
    const auto anonymous_sources = p.source.count ? std::span<const c::ReadLease>{} : bindings.sources();
    const std::array controls{s.indices.ready};
    return gd::RegionAccess::bind_write_effect(invocation, graph, capacity,
        anonymous_sources, s.writer_pointers, {}, found.proof, controls, s.options);
  };
  auto status = install();
  if (!status.ok()) {
    // No writer effect was accepted on a rejecting bind. Undo the only
    // installed permission before returning the untouched caller owner.
    const auto rollback = source_bound
        ? gd::RegionAccess::release_read_projection(invocation, p.source.capacity) : Status::Ok();
    found.interface->clear_configuration(found.prepared, found.invocation);
    if (!rollback.ok()) return rollback.WithContext("slot source rollback after rejected binding");
    return status;
  }
  s.owner = gd::InvocationAccess::state(invocation); s.writer = found.proof;
  s.configuration = configuration; s.last_configuration = configuration;
  s.bindings = std::move(bindings);
  return Status::Ok();
}

Result<SlotBindings> take(Invocation &invocation, NodeId node) noexcept {
  Find found{node}; HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit_retired(invocation, node, &find, &found));
  if (!found.binding || !found.binding->valid())
    return std::unexpected(Status::InvalidArgument("Store has no retained original slot transaction"));
  auto &s = *detail::StoreBindingAccess::get(*found.binding);
  HYLE_ASSIGN_OR_RETURN(auto bound, gd::RegionAccess::has_write_effect(
      s.owner, s.writer, s.writer_pointers));
  bool source_bound = false;
  if (s.facts.source.count) {
    HYLE_ASSIGN_OR_RETURN(source_bound, gd::RegionAccess::has_read_projection(
        s.owner, s.facts.source.capacity, s.bindings.sources()));
  }
  if (source_bound)
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_read_projection(invocation, s.facts.source.capacity));
  if (bound)
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::release_write_effect(invocation, node));
  // A reset/release may already have withdrawn this exact old effect. The
  // retired visitor above still excludes native work/results/observations;
  // returning the original owner neither reopens execution nor publishes it.
  found.interface->clear_configuration(found.prepared, found.invocation);
  s.configuration = {}; s.owner = nullptr; s.writer = {};
  return std::move(s.bindings);
}

} // namespace hyle::graph::slots
