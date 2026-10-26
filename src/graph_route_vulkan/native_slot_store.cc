#include "native_slot_store.h"
#include "native_region.h"
#include "vulkan_source.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <new>
#include <vector>

#include "../compute_route_vulkan/vulkan_copy_internal.h"
#include "../graph/slots_region.h"
#include "hyle/core/assert.h"
#include "hyle/graph/detail/region_access.h"

namespace hyle::graph::vulkan::detail {
namespace {
namespace c = compute;
namespace cv = c::vulkan;
namespace vn = cv::detail;
namespace cd = c::detail;
namespace gd = graph::detail;
namespace sd = slots::detail;
constexpr c::ProviderRuntimeProperties properties{
    .mode = c::ProviderRuntimeMode::kDelegated,
    .workspace_owner = c::ProviderWorkspaceOwner::kProvider,
    .replay_class = c::ProviderReplayClass::kPrepared};
struct Description final {
  slots::Store store;
  ValueId rows, ids;
  c::TensorDesc rows_description, ids_description;
  std::uint32_t rows_port = UINT32_MAX, ids_port = UINT32_MAX;
  std::size_t item_bytes = 0;
};
struct Artifact final {
  std::shared_ptr<vn::Runtime> runtime;
  Description description;
  sd::StoreRegionFacts facts;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef allocator)
      : copies(mem::StdAllocator<cd::CopyBackendRange>(allocator)) {}
  sd::StoreBinding binding;
  sd::StoreConfiguration configuration;
  std::vector<cd::CopyBackendRange, mem::StdAllocator<cd::CopyBackendRange>> copies;
  std::shared_ptr<vn::CopyRecording> recording;
  c::ConstTensorView rows, ids;
  bool configured = false, recorded = false, sealed = false;
};
Status open(const Artifact &a) noexcept {
  std::lock_guard lock(a.runtime->mutex);
  if (a.runtime->closed) return Status::FailedPrecondition("Vulkan slot Store source is closed");
  return a.runtime->failure;
}
bool same(mem::ConstBufferView a, mem::ConstBufferView b) noexcept {
  return a.resource() == b.resource() && a.native_handle() == b.native_handle() &&
      a.offset() == b.offset() && a.size() == b.size();
}
Status copy_alignment(const Artifact &a, mem::ConstBufferView view) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto span, vn::MemoryAccess::copy_span(a.runtime, view));
  const auto alignment = a.description.item_bytes % 4 || a.description.store.schema.stride_bytes() % 4 ? 2u : 4u;
  return span.offset % alignment || view.size() % alignment
      ? Status::Unimplemented("Vulkan slot Store violates its exact copy alignment") : Status::Ok();
}
Result<Description> describe(const RegionCompileRequest &r) noexcept {
  if (!r.graph || r.graph->dimension_count() || r.route != kNativeSlotStoreRoute ||
      r.nodes.size() != 1 || r.inputs.size() != 2 || !r.outputs.empty() ||
      !r.has_unknown_effects || !r.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("Vulkan slot Store needs one exact external write node"));
  HYLE_ASSIGN_OR_RETURN(auto operation, r.graph->operation(r.nodes[0]));
  const auto store = operation.custom_as<slots::Store>();
  if (!store) return std::unexpected(Status::Unimplemented("not an original slot Store"));
  Description d; d.store = *store;
  HYLE_ASSIGN_OR_RETURN(d.rows, r.graph->node_input(r.nodes[0], 0));
  HYLE_ASSIGN_OR_RETURN(d.ids, r.graph->node_input(r.nodes[0], 1));
  if (d.ids.kind() != ValueKind::kInput)
    return std::unexpected(Status::Unimplemented("Vulkan Store IDs must be private input controls"));
  for (std::size_t i = 0; i < r.inputs.size(); ++i) {
    if (r.inputs[i] == d.rows) d.rows_port = static_cast<std::uint32_t>(i);
    if (r.inputs[i] == d.ids) d.ids_port = static_cast<std::uint32_t>(i);
  }
  if (d.rows_port == UINT32_MAX || d.ids_port == UINT32_MAX || d.rows_port == d.ids_port)
    return std::unexpected(Status::Unimplemented("Vulkan Store original input boundary differs"));
  for (std::size_t i = 0; i < r.graph->output_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto value, r.graph->output_id(i));
    if (value == d.ids) return std::unexpected(Status::Unimplemented("private Store IDs cannot escape"));
  }
  for (std::size_t i = 0; i < r.graph->node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, r.graph->node_id(i));
    if (node == r.nodes[0]) continue;
    HYLE_ASSIGN_OR_RETURN(auto count, r.graph->node_input_count(node));
    for (std::size_t port = 0; port < count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto value, r.graph->node_input(node, port));
      if (value == d.ids) return std::unexpected(Status::Unimplemented("Store IDs have another mathematical consumer"));
    }
  }
  HYLE_ASSIGN_OR_RETURN(d.rows_description, r.graph->value_description(d.rows));
  HYLE_ASSIGN_OR_RETURN(d.ids_description, r.graph->value_description(d.ids));
  const std::array inputs{d.rows_description, d.ids_description};
  HYLE_RETURN_IF_ERROR(d.store.infer_output_descriptions(inputs, {}));
  HYLE_ASSIGN_OR_RETURN(auto bytes, d.rows_description.required_bytes());
  if (d.rows_description.dimensions()[0] <= 0)
    return std::unexpected(Status::Unimplemented("Vulkan Store needs nonempty original rows"));
  d.item_bytes = bytes / static_cast<std::size_t>(d.rows_description.dimensions()[0]);
  if (d.item_bytes % 2 || d.store.schema.stride_bytes() % 2)
    return std::unexpected(Status::Unimplemented("Vulkan Store item and stride bytes must be a multiple of two"));
  return d;
}
Status validate(const Artifact &a, Slot &s, c::ConstTensorView rows,
                c::ConstTensorView ids) noexcept {
  if (!s.configured || rows.description() != a.description.rows_description ||
      ids.description() != a.description.ids_description)
    return Status::FailedPrecondition("Vulkan Store original rows/controls are not configured");
  auto configuration = s.configuration; configuration.destination_indices = ids;
  HYLE_RETURN_IF_ERROR(s.binding.validate_configuration(configuration));
  const auto *transaction = s.binding.transaction();
  if (!transaction || transaction->destination_count() != s.copies.size())
    return Status::FailedPrecondition("Vulkan Store original writers differ");
  HYLE_ASSIGN_OR_RETURN(auto bytes, a.description.rows_description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto input, rows.storage().subview(0, bytes));
  HYLE_RETURN_IF_ERROR(copy_alignment(a, input));
  HYLE_ASSIGN_OR_RETURN(auto source, vn::MemoryAccess::copy_span(a.runtime, input));
  for (std::size_t i = 0; i < s.copies.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto target, transaction->destinations()[i].view());
    if (target.size() != a.description.item_bytes)
      return Status::InvalidArgument("Vulkan Store writer is not its original whole item");
    HYLE_RETURN_IF_ERROR(copy_alignment(a, target));
    HYLE_ASSIGN_OR_RETURN(auto destination, vn::MemoryAccess::copy_span(a.runtime, target));
    if (source.physical.address_space == destination.physical.address_space &&
        source.physical.backing == destination.physical.backing &&
        (source.offset < destination.offset ? destination.offset - source.offset < bytes
            : source.offset - destination.offset < a.description.item_bytes))
      return Status::InvalidArgument("Vulkan Store rows overlap an original destination");
    HYLE_ASSIGN_OR_RETURN(s.copies[i].source, input.subview(i * a.description.item_bytes, a.description.item_bytes));
    s.copies[i].destination = target;
  }
  return Status::Ok();
}
const sd::StoreRegionVTable store_table{
    .inspect = [](const ProviderRegionPrepared &p, void *) noexcept -> Result<sd::StoreRegionFacts> {
      if (!p.state) return std::unexpected(Status::InvalidArgument("empty Vulkan Store artifact"));
      return static_cast<const Artifact *>(p.state)->facts;
    },
    .binding = [](void *s) noexcept { return &static_cast<Slot *>(s)->binding; },
    .configure = [](const ProviderRegionPrepared &p, void *opaque,
        const sd::StoreConfiguration &configuration) noexcept -> Status {
      const auto &a = *static_cast<const Artifact *>(p.state); auto &s = *static_cast<Slot *>(opaque);
      HYLE_RETURN_IF_ERROR(open(a));
      if (!configuration.capacity.valid() ||
          configuration.destination_indices.description() != a.description.ids_description)
        return Status::InvalidArgument("Vulkan Store private configuration differs");
      if (s.recording) {
        HYLE_RETURN_IF_ERROR(vn::copy_recording_retired_gate(*s.recording));
        HYLE_RETURN_IF_ERROR(vn::clear_copy_recording(*s.recording));
      }
      s.configuration = configuration; s.rows = {}; s.ids = {};
      s.configured = true; s.recorded = false; s.sealed = false;
      return Status::Ok();
    },
    .clear_configuration = [](const ProviderRegionPrepared &, void *opaque) noexcept {
      auto &s = *static_cast<Slot *>(opaque);
      if (s.recording) HYLE_FATAL_ASSERT(vn::clear_copy_recording(*s.recording).ok());
      s.configuration = {}; s.rows = {}; s.ids = {};
      for (auto &copy : s.copies) copy = {};
      s.configured = false; s.recorded = false; s.sealed = false;
    }};
const gd::RegionBindingReleaseVTable release_table{
    .validate = [](void *, void *opaque) noexcept {
      const auto &s = *static_cast<Slot *>(opaque);
      return s.recording ? vn::copy_recording_retired_gate(*s.recording) : Status::Ok();
    },
    .release = [](void *, void *opaque) noexcept {
      auto &s = *static_cast<Slot *>(opaque);
      if (s.recording) HYLE_FATAL_ASSERT(vn::clear_copy_recording(*s.recording).ok());
      s.rows = {}; s.ids = {}; s.recorded = false; s.sealed = false;
    }};
const gd::RegionOrderedWriteVTable ordered_write{
    [](const ProviderRegionPrepared &p, void *, const cd::CompletionEndpoint *reader) noexcept -> Status {
      const auto &a = *static_cast<const Artifact *>(p.state); HYLE_RETURN_IF_ERROR(open(a));
      return !reader || imports_region_ordered_endpoint(a.runtime, *reader) ? Status::Ok()
          : Status::Unimplemented("Vulkan Store reader belongs to a different Runtime");
    }};
const ProviderRegionPreparedVTable table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return c::AliasSupport{}; },
    .create_invocation = [](void *opaque, mem::AllocatorRef allocator, mem::BufferView, void **out) noexcept -> Status {
      const auto &a = *static_cast<const Artifact *>(opaque); *out = nullptr;
      auto *s = allocator.construct<Slot>(allocator);
      if (!s) return Status::ResourceExhausted("Vulkan Store slot metadata");
      *out = s;
      try { s->copies.resize(a.facts.destination_count); }
      catch (...) { return Status::ResourceExhausted("Vulkan Store exact copy metadata"); }
      HYLE_ASSIGN_OR_RETURN(auto bytes, a.description.rows_description.required_bytes());
      HYLE_ASSIGN_OR_RETURN(s->recording, vn::prepare_copy_recording_runtime(a.runtime,
          static_cast<std::uint32_t>(s->copies.size()), allocator, bytes, vn::CopyRecordingMode::kReplayable,
          a.description.item_bytes % 4 || a.description.store.schema.stride_bytes() % 4
              ? vn::CopyGranularity::kHalfword : vn::CopyGranularity::kWord));
      return Status::Ok();
    },
    .destroy_invocation = [](void *s, mem::AllocatorRef allocator) noexcept { allocator.destroy(static_cast<Slot *>(s)); },
    .delegated_submit = [](void *opaque, void *slot,
        std::span<const cd::CompletionDependency> dependencies,
        std::span<const c::ConstTensorView> inputs,
        std::span<const c::TensorView> outputs) noexcept -> Result<c::Completion> {
      const auto &a = *static_cast<const Artifact *>(opaque); auto &s = *static_cast<Slot *>(slot);
      HYLE_RETURN_IF_ERROR(open(a));
      if (!s.sealed || !s.recorded || inputs.size() != 2 || !outputs.empty() ||
          !same(inputs[a.description.rows_port].storage(), s.rows.storage()) ||
          !same(inputs[a.description.ids_port].storage(), s.ids.storage()))
        return std::unexpected(Status::FailedPrecondition("Vulkan Store exact episode is not sealed"));
      HYLE_RETURN_IF_ERROR(s.binding.validate_authority());
      for (const auto dependency : dependencies) HYLE_RETURN_IF_ERROR(vn::qualify_copy_dependency(a.runtime, dependency));
      return vn::submit_copy_recording(*s.recording);
    },
    .endpoint = [](void *p, void *) noexcept { return region_ordered_endpoint(static_cast<Artifact *>(p)->runtime); },
    .can_import_endpoint = [](void *p, void *, cd::CompletionEndpoint source) noexcept {
      return imports_region_ordered_endpoint(static_cast<Artifact *>(p)->runtime, source);
    },
    .prepare_bindings = [](void *opaque, void *slot, std::span<const RegionValue> inputs,
        std::span<const c::TensorView> outputs) noexcept -> Status {
      const auto &a = *static_cast<const Artifact *>(opaque); auto &s = *static_cast<Slot *>(slot);
      HYLE_RETURN_IF_ERROR(open(a)); s.recorded = false; s.sealed = false;
      if (inputs.size() != 2 || !outputs.empty()) return Status::InvalidArgument("Vulkan Store boundary changed");
      HYLE_RETURN_IF_ERROR(vn::copy_recording_reuse_gate(*s.recording));
      HYLE_RETURN_IF_ERROR(validate(a, s, inputs[a.description.rows_port].tensor, inputs[a.description.ids_port].tensor));
      HYLE_RETURN_IF_ERROR(vn::bind_copy_recording(*s.recording, s.copies));
      s.rows = inputs[a.description.rows_port].tensor; s.ids = inputs[a.description.ids_port].tensor;
      s.recorded = true; return Status::Ok();
    },
    .seal_bindings = [](void *opaque, void *slot) noexcept -> Status {
      const auto &a = *static_cast<const Artifact *>(opaque); auto &s = *static_cast<Slot *>(slot);
      HYLE_RETURN_IF_ERROR(open(a));
      if (!s.recorded) return Status::FailedPrecondition("Vulkan Store copy is not prepared");
      HYLE_RETURN_IF_ERROR(validate(a, s, s.rows, s.ids));
      HYLE_RETURN_IF_ERROR(vn::copy_recording_reuse_gate(*s.recording));
      s.sealed = true; return Status::Ok();
    },
    .query_extension = [](void *, c::ProviderExtensionId id) noexcept -> const void * {
      if (id == sd::kStoreBindingExtensionId) return &store_table;
      if (id == gd::kRegionBindingReleaseExtensionId) return &release_table;
      if (id == gd::kRegionOrderedWriteExtensionId) return &ordered_write;
      return nullptr;
    }};
} // namespace
bool native_slot_store_operation(const c::Operation &operation) noexcept {
  return operation.custom_as<slots::Store>().has_value();
}
Status propose_native_slot_store(const RegionDiscoveryRequest &r, RegionProposalWriter &writer) noexcept {
  for (std::size_t i = 0; i < r.graph.node_count(); ++i) {
    if (!r.allowed_nodes.empty() && !r.allowed_nodes[i]) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, r.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto operation, r.graph.operation(node));
    if (!native_slot_store_operation(operation)) continue;
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = nodes, .route = kNativeSlotStoreRoute,
        .replay_class = c::ProviderReplayClass::kPrepared,
        .implementation = c::ProviderImplementationId{0x534c4f545354564bULL}, .preference = 100}));
    if (!more) break;
  }
  return Status::Ok();
}
Status native_slot_store_resources(void *opaque, const RegionCompileRequest &r,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *out) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto d, describe(r)); (void)d;
  if (inputs.size() != 2 || !outputs.empty()) return Status::InvalidArgument("Vulkan Store resource arity differs");
  const auto &runtime = static_cast<SourceContext *>(opaque)->runtime;
  const cd::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .forbidden_properties = mem::MemoryProperty::kHostAccessible, .minimum_alignment = 1,
      .exact_domain = {runtime->device.get(), static_cast<std::uintptr_t>(cv::MemoryKind::kDevice)},
      .requires_exact_domain = true};
  for (auto &input : inputs) input.resource = binding;
  *out = {.runtime = properties, .preserves_effects = true}; return Status::Ok();
}
c::ProviderPrepareOutcome compile_native_slot_store(void *opaque, const RegionCompileRequest &r,
    mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  using Outcome = c::ProviderPrepareOutcome; *out = {};
  auto d = describe(r);
  if (!d) return d.error().IsUnimplemented() ? Outcome::unsupported(d.error()) : Outcome::error(d.error());
  if (r.input_resources.size() != 2) return Outcome::error(Status::InvalidArgument("Vulkan Store input resources absent"));
  const auto &runtime = static_cast<SourceContext *>(opaque)->runtime;
  for (const auto resource : r.input_resources) {
    const auto admitted = vn::MemoryAccess::validate_resource(runtime, resource);
    if (!admitted.ok()) return Outcome::error(admitted);
  }
  auto *a = allocator.construct<Artifact>();
  if (!a) return Outcome::error(Status::ResourceExhausted("Vulkan Store artifact"));
  a->runtime = runtime; a->description = *d;
  a->facts = {.store = r.nodes[0], .schema = d->store.schema, .destination_indices = d->ids,
      .destination_input_port = d->ids_port,
      .destination_count = static_cast<std::size_t>(d->rows_description.dimensions()[0]),
      .resource = r.input_resources[d->rows_port]};
  *out = {.vtable = &table, .state = a, .properties = {.persistent_bytes = sizeof(Artifact),
      .runtime = properties, .preserves_effects = true, .invocation_private_bytes = std::nullopt}};
  return Outcome::prepared();
}
} // namespace hyle::graph::vulkan::detail
