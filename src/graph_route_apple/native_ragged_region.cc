#include "native_ragged_region.h"
#include "hyle/compute/detail/batch_provider.h"
#include "native_region.h"
#include "native_output.h"

#include "../compute_route_apple/apple_copy_internal.h"
#include "../graph_batch/native_batch_regions.h"
#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/generated_result.h"

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>

namespace hyle::graph::apple::detail {
namespace {
namespace ca = compute::apple;
namespace an = ca::detail;
namespace cd = compute::detail;
namespace bd = graph::batch::detail;
namespace mx = mlx::core;
const unsigned array_type = 0;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged,
    .uses_private_threads = true};

struct Recipe final {
  compute::RaggedAttention attention;
  compute::MlaCacheLayout mla = compute::MlaCacheLayout::kNone;
};
Result<Recipe> recipe(const compute::Operation &operation) noexcept {
  if (const auto attention = operation.custom_as<compute::RaggedAttention>()) {
    if (attention->capacity.key_storage.data_type() == compute::DataType::kFloat8E4M3FN)
      return std::unexpected(Status::Unimplemented("Apple ragged E4 needs its compact MLA operation"));
    return Recipe{*attention};
  }
  if (const auto mla = operation.custom_as<compute::RaggedMlaAttention>()) {
    HYLE_ASSIGN_OR_RETURN(auto capacity, compute::mla_batch_capacity(mla->capacity));
    return Recipe{{.capacity = capacity, .scale = mla->scale,
        .causal = mla->causal, .window_left = mla->window_left}, mla->capacity.layout};
  }
  return std::unexpected(Status::Unimplemented("not ordinary or compact MLA ragged attention"));
}
Result<Recipe> qualify(void *context,
    const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() ||
      request.route != kNativeRaggedRoute || request.nodes.size() != 1 ||
      request.outputs.size() > 1 || request.has_unknown_effects ||
      !request.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("Apple ragged needs one fixed pure typed node"));
  HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(request.nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto selected, recipe(operation));
  HYLE_RETURN_IF_ERROR(an::PagedBatchAccess::validate_ragged(source_runtime(context), selected.attention));
  return selected;
}

struct Artifact final {
  compute::RaggedAttention attention;
  compute::MlaCacheLayout mla = compute::MlaCacheLayout::kNone;
  std::size_t payloads = 3;
  NodeId node;
  std::array<ValueId, 3> inputs{};
  std::array<std::uint32_t, 3> ports{};
  ValueId metadata;
  std::uint32_t metadata_port = 0;
  std::array<compute::TensorDesc, 4> descriptions;
  compute::TensorDesc output;
  std::size_t public_outputs = 0;
  std::shared_ptr<an::Runtime> runtime;
  compute::ResourceBudget budget;
  compute::WorkloadContext workload;
};
struct Slot final {
  Slot(mem::AllocatorRef selected, Artifact &owner)
      : allocator(selected), artifact(owner) {}
  mem::AllocatorRef allocator;
  Artifact &artifact;
  ca::Program program;
  ca::Invocation native;
  compute::BatchLayoutVersion layout;
  std::array<an::InputBinding, 3> bindings;
  cd::ResultStorageOwner result;
  std::optional<mx::array> projected;
  std::optional<compute::ReadLease> exported;
  std::mutex output_mutex;
  bool bound = false;
};
void clear_output(Slot &slot) noexcept {
  slot.exported.reset();
  slot.projected.reset();
}
void clear_bindings(Slot &slot) noexcept {
  for (auto &binding : slot.bindings) {
    binding.value.reset();
    binding.source.reset();
  }
  slot.bound = false;
}
Status reset(void *, void *invocation) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  slot.result.reset();
  clear_output(slot);
  if (slot.native.valid())
    HYLE_RETURN_IF_ERROR(an::InvocationAccess::release_inputs(slot.native));
  clear_bindings(slot);
  return Status::Ok();
}
Status prepare(void *prepared, void *invocation,
    compute::BatchLayoutVersion layout) noexcept {
  auto &artifact = *static_cast<Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(invocation);
  if (!layout.valid() || !cd::BatchLayoutAccess::same_topology(layout.origin_schema().capacity(), artifact.attention.capacity) ||
      layout.origin_schema().mla_layout() != artifact.mla ||
      layout.facts().query_rows > std::uint32_t(artifact.output.dimensions()[0]))
    return Status::InvalidArgument("Apple ragged layout differs from its actual typed Q/output extent");
  HYLE_RETURN_IF_ERROR(compute::validate_batch_attention_layout(layout,
      artifact.attention.causal, artifact.attention.window_left));
  slot.result.reset();
  clear_output(slot);
  if (slot.native.valid())
    HYLE_RETURN_IF_ERROR(an::InvocationAccess::reuse_gate(slot.native));
  clear_bindings(slot);
  const auto same_queries = [&]() noexcept {
    if (!slot.layout.valid() || !slot.layout.origin_schema().owns(layout)) return false;
    const auto before = slot.layout.host().query_indptr;
    const auto after = layout.host().query_indptr;
    return std::ranges::equal(before, after);
  };
  if (slot.native.valid() && same_queries()) {
    HYLE_RETURN_IF_ERROR(ca::update_ragged_batch(slot.native, layout));
  } else {
    HYLE_ASSIGN_OR_RETURN(auto program, an::PagedBatchAccess::prepare_ragged_graph(
        artifact.runtime, layout, artifact.attention,
        std::uint32_t(artifact.output.dimensions()[0]), artifact.workload));
    HYLE_ASSIGN_OR_RETURN(auto native, program.create_invocation(slot.allocator));
    slot.native = std::move(native);
    slot.program = std::move(program);
  }
  slot.layout = std::move(layout);
  return Status::Ok();
}
const bd::NativeBatchRegionVTable layout_table{
    .inspect = [](void *prepared, void *invocation) noexcept -> Result<bd::NativeBatchRegionFacts> {
      const auto &artifact = *static_cast<Artifact *>(prepared);
      const auto &slot = *static_cast<Slot *>(invocation);
      return bd::NativeBatchRegionFacts{.node = artifact.node,
          .capacity = artifact.attention.capacity,
          .mla = artifact.mla,
          .causal = artifact.attention.causal,
          .window_left = artifact.attention.window_left,
          .inputs = std::span(artifact.inputs).first(artifact.payloads),
          .input_ports = std::span(artifact.ports).first(artifact.payloads),
          .metadata_input = artifact.metadata,
          .output_query_rows = std::uint32_t(artifact.output.dimensions()[0]),
          .layout = slot.layout};
    },
    .prepare = &prepare, .reset = &reset};

Status create_slot(void *prepared, mem::AllocatorRef allocator,
    mem::BufferView, void **out) noexcept {
  *out = allocator.construct<Slot>(allocator, *static_cast<Artifact *>(prepared));
  return *out ? Status::Ok() : Status::ResourceExhausted("Apple ragged slot metadata");
}
Status bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  auto &artifact = *static_cast<Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(invocation);
  if (!slot.layout.valid() || !slot.native.valid() || outputs.size() != artifact.public_outputs)
    return Status::FailedPrecondition("Apple ragged layout must be prepared before replay bindings");
  slot.result.reset();
  clear_output(slot);
  HYLE_RETURN_IF_ERROR(an::InvocationAccess::reuse_gate(slot.native));
  clear_bindings(slot);
  if (artifact.metadata_port >= inputs.size() ||
      inputs[artifact.metadata_port].description != artifact.descriptions[artifact.payloads])
    return Status::InvalidArgument("Apple ragged canonical metadata descriptor differs");
  const compute::LeaseImportOptions import{.allocator = slot.allocator,
      .resource_budget = artifact.budget};
  for (std::size_t port = 0; port < artifact.payloads; ++port) {
    if (artifact.ports[port] >= inputs.size())
      return Status::InvalidArgument("Apple ragged physical boundary port differs");
    const auto &input = inputs[artifact.ports[port]];
    const auto &description = artifact.descriptions[port];
    if (input.description != description)
      return Status::InvalidArgument("Apple ragged input description differs");
    if (input.reference.valid()) {
      HYLE_ASSIGN_OR_RETURN(slot.bindings[port].source,
          native_output_source(input.reference, artifact.runtime, description));
    } else if (input.lease) {
      HYLE_ASSIGN_OR_RETURN(auto owner, input.lease->view());
      const auto wanted = input.tensor.storage();
      if (!input.tensor.valid() || owner.resource() != wanted.resource() ||
          owner.native_handle() != wanted.native_handle() || wanted.offset() < owner.offset() ||
          wanted.offset() - owner.offset() > owner.size() ||
          wanted.size() > owner.size() - (wanted.offset() - owner.offset()))
        return Status::InvalidArgument("Apple ragged input lease does not cover its actual tensor");
      HYLE_ASSIGN_OR_RETURN(auto selected,
          input.lease->subview(wanted.offset() - owner.offset(), wanted.size()));
      HYLE_ASSIGN_OR_RETURN(slot.bindings[port].value,
          an::MemoryAccess::owned_tensor(std::move(selected), description, import));
    } else {
      HYLE_ASSIGN_OR_RETURN(slot.bindings[port].value, an::MemoryAccess::borrow_tensor(input.tensor));
    }
  }
  HYLE_RETURN_IF_ERROR(an::InvocationAccess::prepare_inputs(slot.native, std::span(slot.bindings).first(artifact.payloads)));
  slot.bound = true;
  return Status::Ok();
}
Status seal(void *, void *invocation) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  slot.result.reset();
  return an::InvocationAccess::reuse_gate(slot.native);
}
Status project(Slot &slot, std::size_t port) noexcept {
  if (port >= slot.artifact.public_outputs)
    return Status::OutOfRange("Apple ragged output port");
  if (!slot.projected) {
    HYLE_ASSIGN_OR_RETURN(slot.projected, an::InvocationAccess::output(slot.native, port));
  }
  return Status::Ok();
}
Status export_port(Slot &slot, std::size_t port) noexcept {
  HYLE_RETURN_IF_ERROR(project(slot, port));
  if (slot.exported) return Status::Ok();
  if (!slot.projected->nbytes())
    return Status::Unimplemented("empty Apple ragged output has no payload lease");
  HYLE_ASSIGN_OR_RETURN(slot.exported,
      an::MemoryAccess::export_readonly(*slot.projected, slot.artifact.output,
          {.allocator = slot.allocator, .resource_budget = slot.artifact.budget}));
  return Status::Ok();
}
Result<RegionValue> value(void *, void *invocation, std::size_t port) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  std::lock_guard lock(slot.output_mutex);
  HYLE_RETURN_IF_ERROR(project(slot, port));
  RegionValue result{slot.artifact.output, {}, &array_type, &*slot.projected};
  if (!slot.projected->nbytes()) return result;
  HYLE_RETURN_IF_ERROR(export_port(slot, port));
  HYLE_ASSIGN_OR_RETURN(result.tensor, slot.exported->tensor(result.description));
  return result;
}
Result<compute::ReadLease> lease(void *, void *invocation, std::size_t port) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  std::lock_guard lock(slot.output_mutex);
  HYLE_RETURN_IF_ERROR(export_port(slot, port));
  return *slot.exported;
}
const NativeOutputVTable output_table{
    [](void *prepared, void *invocation, std::size_t port,
        const std::shared_ptr<an::Runtime> &runtime,
        const compute::TensorDesc &description) noexcept -> Result<an::InputSource> {
      if (!prepared || !invocation)
        return std::unexpected(Status::InvalidArgument("Apple ragged source has no actual slot"));
      const auto &artifact = *static_cast<Artifact *>(prepared);
      const auto &slot = *static_cast<Slot *>(invocation);
      if (&slot.artifact != &artifact || artifact.runtime != runtime ||
          description != artifact.output || port >= artifact.public_outputs || !slot.native.valid())
        return std::unexpected(Status::InvalidArgument("Apple ragged source owner/Runtime/port differs"));
      return an::InvocationAccess::source(slot.native, port);
    },
    [](void *prepared, void *invocation, std::size_t port,
        const ca::NativeStorage &wanted) noexcept -> Result<NativeOutputOrigins> {
      if (!prepared || !invocation)
        return std::unexpected(Status::InvalidArgument("Apple ragged output has no actual slot"));
      const auto &artifact = *static_cast<Artifact *>(prepared);
      auto &slot = *static_cast<Slot *>(invocation);
      if (&slot.artifact != &artifact || port >= artifact.public_outputs)
        return std::unexpected(Status::InvalidArgument("Apple ragged output owner/port differs"));
      std::lock_guard lock(slot.output_mutex);
      HYLE_RETURN_IF_ERROR(project(slot, port));
      HYLE_ASSIGN_OR_RETURN(auto actual, an::MemoryAccess::storage_info(*slot.projected));
      NativeOutputOrigins result;
      if (!actual.view_bytes || !wanted.view_bytes || actual.buffer != wanted.buffer ||
          !(actual.offset_bytes <= wanted.offset_bytes ?
              wanted.offset_bytes - actual.offset_bytes < actual.view_bytes :
              actual.offset_bytes - wanted.offset_bytes < wanted.view_bytes)) return result;
      result.owned_overlap = true;
      HYLE_RETURN_IF_ERROR(export_port(slot, port));
      result.owner = *slot.exported;
      return result;
    }};
Result<compute::Completion> submit(void *, void *invocation,
    std::span<const cd::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView>, std::span<const compute::TensorView>) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  if (!slot.bound)
    return std::unexpected(Status::FailedPrecondition("Apple ragged inputs are not bound"));
  for (const auto dependency : dependencies)
    HYLE_RETURN_IF_ERROR(an::qualify_recording_dependency(slot.artifact.runtime, dependency));
  slot.result.reset();
  clear_output(slot);
  HYLE_ASSIGN_OR_RETURN(auto accepted, slot.native.submit());
  auto transfer = cd::ResultAccess::take_storage(std::move(accepted));
  slot.result = std::move(transfer.owner);
  return std::move(transfer.completion);
}
const graph::detail::RegionBindingReleaseVTable binding_release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      const auto &slot = *static_cast<const Slot *>(invocation);
      if (!slot.native.valid())
        return slot.result ? Status::FailedPrecondition("native result has no invocation") : Status::Ok();
      return an::InvocationAccess::can_detach_inputs(slot.native, slot.result);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.result.reset();
      if (slot.native.valid()) an::InvocationAccess::detach_inputs(slot.native);
      clear_output(slot);
      clear_bindings(slot);
    }};

const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = &create_slot,
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .delegated_submit = &submit,
    .endpoint = [](void *p, void *) noexcept { return region_ordered_endpoint(static_cast<Artifact *>(p)->runtime); },
    .can_import_endpoint = [](void *p, void *, cd::CompletionEndpoint source) noexcept {
      return imports_region_ordered_endpoint(static_cast<Artifact *>(p)->runtime, source);
    },
    .output_contract = [](void *, std::size_t) noexcept {
      return RegionOutputContract{RegionOutputKind::kDeferredGenerated, RegionOutputProjection::kNativeOnly};
    },
    .output_value = &value, .prepare_bindings = &bind, .seal_bindings = &seal, .output_lease = &lease,
    .query_extension = [](void *, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == graph::detail::kRegionBindingReleaseExtensionId) return &binding_release_table;
      if (id == bd::kNativeBatchRegionExtensionId) return &layout_table;
      if (id == kNativeOutputExtensionId) return &output_table;
      return nullptr;
    }};
} // namespace

bool native_ragged_operation(void *context, const compute::Operation &operation) noexcept {
  const auto selected = recipe(operation);
  return selected && an::PagedBatchAccess::validate_ragged(source_runtime(context), selected->attention).ok();
}
Status propose_native_ragged(void *context, const RegionDiscoveryRequest &request,
    std::span<const std::uint8_t> needed, RegionProposalWriter &writer) noexcept {
  for (std::size_t ordinal = 0; ordinal < request.graph.node_count(); ++ordinal) {
    if (!needed[ordinal] || (!request.allowed_nodes.empty() && !request.allowed_nodes[ordinal])) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(ordinal));
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    auto selected = recipe(operation);
    if (!selected) { if (selected.error().IsUnimplemented()) continue; return selected.error(); }
    const auto admitted = an::PagedBatchAccess::validate_ragged(source_runtime(context), selected->attention);
    if (!admitted.ok()) { if (admitted.IsUnimplemented()) continue; return admitted; }
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes,
        .route = kNativeRaggedRoute, .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{0x5241474745444150ULL}, .preference = 100}));
    if (!added) break;
  }
  return Status::Ok();
}
Status native_ragged_resources(void *context, const RegionCompileRequest &request,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *proposal) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto selected, qualify(context, request));
  if (inputs.size() != request.inputs.size() || outputs.size() != request.outputs.size())
    return Status::InvalidArgument("Apple ragged resource boundary differs");
  const cd::ResourceBindingRequirement resource{
      .required_properties = mem::MemoryProperty::kHostAccessible | mem::MemoryProperty::kDeviceAccessible |
          mem::MemoryProperty::kHostDeviceCoherent, .minimum_alignment = 1};
  std::ranges::fill(inputs, RegionInputRequirement{resource, false});
  for (std::size_t port = 0; port < (selected.mla == compute::MlaCacheLayout::kNone ? 3u : 2u); ++port) {
    HYLE_ASSIGN_OR_RETURN(auto input, request.graph->node_input(request.nodes[0], port));
    const auto found = std::ranges::find(request.inputs, input);
    if (found == request.inputs.end()) return Status::InvalidArgument("Apple ragged payload boundary is missing");
    inputs[std::size_t(found - request.inputs.begin())].accepts_reference = true;
  }
  std::ranges::fill(outputs, RegionOutputRequirement{resource,
      {RegionOutputKind::kDeferredGenerated, RegionOutputProjection::kNativeOnly}});
  *proposal = {.runtime = runtime_properties, .workspace_resource = resource};
  return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_ragged(void *context,
    const RegionCompileRequest &request, mem::AllocatorRef allocator,
    ProviderRegionPrepared *out) noexcept {
  using Outcome = compute::ProviderPrepareOutcome;
  *out = {};
  auto attention = qualify(context, request);
  if (!attention) return attention.error().IsUnimplemented() ?
      Outcome::unsupported(attention.error()) : Outcome::error(attention.error());
  auto *artifact = allocator.construct<Artifact>();
  if (!artifact) return Outcome::error(Status::ResourceExhausted("Apple ragged artifact metadata"));
  *out = {.vtable = &prepared_table, .state = artifact};
  const auto build = [&]() noexcept -> Status {
    artifact->attention = attention->attention;
    artifact->mla = attention->mla;
    artifact->payloads = artifact->mla == compute::MlaCacheLayout::kNone ? 3u : 2u;
    artifact->node = request.nodes[0];
    artifact->runtime = source_runtime(context);
    artifact->workload = request.workload;
    artifact->budget = request.resource_budget;
    artifact->public_outputs = request.outputs.size();
    for (std::size_t port = 0; port < artifact->payloads + 1; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, request.graph->node_input(artifact->node, port));
      const auto found = std::ranges::find(request.inputs, input);
      if (found == request.inputs.end()) return Status::InvalidArgument("Apple ragged boundary input is missing");
      HYLE_ASSIGN_OR_RETURN(artifact->descriptions[port], request.graph->value_description(input));
      if (port == artifact->payloads) {
        artifact->metadata = input;
        artifact->metadata_port = std::uint32_t(found - request.inputs.begin());
      } else {
        artifact->inputs[port] = input;
        artifact->ports[port] = std::uint32_t(found - request.inputs.begin());
      }
    }
    compute::TensorDesc expected;
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(artifact->node));
    HYLE_RETURN_IF_ERROR(compute::infer_output_descriptions(operation,
        std::span(artifact->descriptions).first(artifact->payloads + 1), std::span(&expected, 1)));
    HYLE_ASSIGN_OR_RETURN(auto output, request.graph->result(artifact->node));
    HYLE_ASSIGN_OR_RETURN(artifact->output, request.graph->value_description(output));
    if (artifact->output != expected)
      return Status::InvalidArgument("Apple ragged concrete Q/output descriptions differ");
    out->properties = {.persistent_bytes = std::nullopt, .runtime = runtime_properties,
        .invocation_private_bytes = std::nullopt};
    return Status::Ok();
  };
  const auto status = build();
  return status.ok() ? Outcome::prepared() : status.IsUnimplemented() ?
      Outcome::unsupported(status) : Outcome::error(status);
}
} // namespace hyle::graph::apple::detail
