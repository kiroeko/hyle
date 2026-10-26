#include "native_ragged_region.h"
#include "hyle/compute/detail/batch_provider.h"
#include "native_region.h"
#include "vulkan_source.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

#include "../compute_route_vulkan/vulkan_composition_access.h"
#include "hyle/compute/detail/vulkan_copy_stream_access.h"
#include "../compute_route_vulkan/vulkan_memory_internal.h"
#include "../compute_route_vulkan/vulkan_program_internal.h"
#include "../graph_batch/native_batch_regions.h"
#include "hyle/compute/detail/vulkan_batch_access.h"

namespace hyle::graph::vulkan::detail {
namespace {
namespace cv = compute::vulkan;
namespace vn = cv::detail;
namespace cd = compute::detail;
namespace bd = graph::batch::detail;
const unsigned tensor_type = 0, observed_endpoint = 0;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged};

struct Recipe final {
  compute::RaggedAttention operation;
  compute::MlaCacheLayout mla = compute::MlaCacheLayout::kNone;
};
Result<Recipe> recipe(const compute::Operation &operation) noexcept {
  if (const auto d = operation.custom_as<compute::RaggedAttention>())
    return Recipe{*d};
  if (const auto d = operation.custom_as<compute::RaggedMlaAttention>()) {
    HYLE_ASSIGN_OR_RETURN(auto capacity, compute::mla_batch_capacity(d->capacity));
    return Recipe{compute::RaggedAttention{.capacity=capacity,.scale=d->scale,
        .causal=d->causal,.window_left=d->window_left},d->capacity.layout};
  }
  return std::unexpected(Status::Unimplemented("not native ragged attention"));
}
struct Artifact final {
  std::shared_ptr<vn::Runtime> runtime;
  cv::InvocationAllocation allocation;
  compute::RaggedAttention operation;
  compute::MlaCacheLayout mla = compute::MlaCacheLayout::kNone;
  std::size_t payload_count = 3;
  compute::WorkloadContext workload;
  NodeId node;
  std::array<ValueId, 4> inputs{};
  std::array<std::uint32_t, 4> ports{};
  std::array<compute::TensorDesc, 4> descriptions{};
  std::size_t boundary_count = 0, public_outputs = 0;
  mem::MemoryResourceRef output_resource;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef selected) : allocator(selected) {}
  mem::AllocatorRef allocator;
  cv::Invocation native;
  compute::BatchLayoutVersion layout;
  cv::Tensor output;
  mem::Buffer backing;
  cd::ResultStorageOwner result;
  bool prepared = false, sealed = false, needs_seal = false;
};

Status qualify(const compute::RaggedAttention &operation) noexcept {
  HYLE_RETURN_IF_ERROR(compute::validate_batch_capacity(operation.capacity));
  HYLE_RETURN_IF_ERROR(compute::validate_batch_attention_options(operation.causal, operation.window_left));
  if (operation.capacity.storage != compute::BatchStorage::kRagged || !std::isfinite(operation.scale))
    return Status::InvalidArgument("native Vulkan ragged options/storage differ");
  if (!vn::PagedBatchAccess::cache_domain(operation.capacity))
    return Status::Unimplemented("native Vulkan ragged cache dtype/geometry is unsupported");
  return Status::Ok();
}
Result<Recipe> qualify(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() || request.route != kNativeRaggedRoute ||
      request.nodes.size() != 1 || request.outputs.size() > 1 ||
      request.has_unknown_effects || !request.effect_accesses.empty())
    return std::unexpected(Status::Unimplemented("native Vulkan ragged needs one fixed pure typed node"));
  HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(request.nodes[0]));
  HYLE_ASSIGN_OR_RETURN(auto selected, recipe(operation));
  HYLE_RETURN_IF_ERROR(qualify(selected.operation));
  return selected;
}
Status create_slot(void *, mem::AllocatorRef allocator, mem::BufferView, void **out) noexcept {
  *out = allocator.construct<Slot>(allocator);
  return *out ? Status::Ok() : Status::ResourceExhausted("native Vulkan ragged slot metadata");
}
Status refresh_output(const Artifact &a, Slot &s) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto output, vn::CompositionAccess::output(s.native, 0));
  mem::Buffer backing;
  if (a.public_outputs) {
    HYLE_ASSIGN_OR_RETURN(backing, vn::MemoryAccess::export_storage(
        a.runtime, a.output_resource, output, s.allocator));
  }
  // Publish both cold projections together without moving their Slot objects.
  s.output = std::move(output); s.backing = std::move(backing);
  return Status::Ok();
}
Status prepare_layout(void *prepared, void *invocation, compute::BatchLayoutVersion layout) noexcept {
  auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (!layout.valid() || !cd::BatchLayoutAccess::same_topology(layout.origin_schema().capacity(), a.operation.capacity) ||
      layout.origin_schema().mla_layout() != a.mla ||
      layout.facts().query_rows > std::uint32_t(a.descriptions[0].dimensions()[0]))
    return Status::InvalidArgument("native Vulkan ragged owned layout exceeds concrete Q/output extent");
  HYLE_RETURN_IF_ERROR(compute::validate_batch_attention_layout(layout,
      a.operation.causal, a.operation.window_left));
  s.result.reset();
  if (s.native.valid()) HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_bindings(s.native));
  s.prepared = false; s.needs_seal = false;
  bool reuse = false;
  if (s.native.valid()) {
    const auto &contract = vn::Access::invocation_state(s.native)->program->ragged;
    if (contract && contract->initial.origin_schema().owns(layout)) {
      const auto first = contract->initial.host().query_indptr, next = layout.host().query_indptr;
      reuse = first.size() == next.size() && std::equal(first.begin(), first.end(), next.begin());
    }
  }
  if (reuse) {
    HYLE_RETURN_IF_ERROR(cv::update_ragged_batch(s.native, layout));
  } else {
    HYLE_ASSIGN_OR_RETURN(auto program, vn::PagedBatchAccess::prepare_ragged_graph(
        a.runtime, layout, a.operation,
        std::uint32_t(a.descriptions[0].dimensions()[0]), a.workload));
    HYLE_ASSIGN_OR_RETURN(auto native, program.create_invocation(s.allocator, a.allocation));
    bool refresh = false;
    HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_outputs(native, &refresh));
    HYLE_ASSIGN_OR_RETURN(auto output, vn::CompositionAccess::output(native, 0));
    mem::Buffer backing;
    if (a.public_outputs) {
      HYLE_ASSIGN_OR_RETURN(backing, vn::MemoryAccess::export_storage(
          a.runtime, a.output_resource, output, s.allocator));
    }
    // A replacement has no accepted work. Commit only after its complete
    // initial projection succeeds; final seal follows the actual input bind.
    s.native = std::move(native); // The original Invocation retains its Program.
    s.output = std::move(output); s.backing = std::move(backing); s.sealed = false;
  }
  s.layout = std::move(layout); s.prepared = true;
  return Status::Ok();
}
Status reset(void *, void *invocation) noexcept {
  auto &s = *static_cast<Slot *>(invocation);
  s.result.reset();
  if (s.native.valid()) HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_bindings(s.native));
  s.prepared = false; s.layout = {}; s.needs_seal = false;
  if (s.native.valid()) HYLE_RETURN_IF_ERROR(vn::PagedBatchAccess::release_inputs(s.native));
  return Status::Ok();
}
Status bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
    std::span<const compute::TensorView> outputs) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (!s.prepared || !s.layout.valid() || !s.native.valid())
    return Status::FailedPrecondition("prepare the native ragged owned layout before replay bindings");
  if (inputs.size() != a.boundary_count || outputs.size() != a.public_outputs)
    return Status::InvalidArgument("native ragged physical boundary arity differs");
  const auto &bound_metadata = inputs[a.ports[a.payload_count]];
  // The common cold-layout coordinator binds and protects this exact owned
  // metadata input. An authorized Graph copy may change its physical owner;
  // the original native Program consumes only the already prepared layout.
  if (bound_metadata.description != a.descriptions[a.payload_count])
    return Status::InvalidArgument("native ragged metadata description differs");
  s.result.reset();
  bool refresh = false;
  HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_outputs(s.native, &refresh));
  s.needs_seal = false;
  if (refresh || !s.output.valid()) HYLE_RETURN_IF_ERROR(refresh_output(a, s));
  const compute::LeaseImportOptions import{.allocator = s.allocator,
      .resource_budget = a.allocation.resource_budget};
  std::array<cv::Tensor, 3> native;
  for (std::size_t port = 0; port < a.payload_count; ++port) {
    const auto &input = inputs[a.ports[port]];
    if (input.description != a.descriptions[port])
      return Status::InvalidArgument("native ragged Q/K/V capacity description differs");
    if (input.lease) {
      HYLE_ASSIGN_OR_RETURN(native[port], vn::MemoryAccess::lease_tensor(a.runtime,
          input.tensor.storage().resource(), *input.lease, input.description, import));
    } else {
      HYLE_ASSIGN_OR_RETURN(native[port], vn::MemoryAccess::tensor_view(a.runtime,
          input.tensor, s.allocator, import.resource_budget));
    }
  }
  HYLE_RETURN_IF_ERROR(s.native.bind(std::span(native).first(a.payload_count)));
  s.needs_seal = true;
  return Status::Ok();
}
Status seal(void *, void *invocation) noexcept {
  auto &s = *static_cast<Slot *>(invocation); s.result.reset();
  if (!s.prepared) return Status::FailedPrecondition("native ragged layout is not prepared");
  if (s.sealed && !s.needs_seal) return vn::PagedBatchAccess::reuse_gate(s.native);
  HYLE_RETURN_IF_ERROR(s.sealed ? vn::PagedBatchAccess::reseal_composition(s.native)
                               : vn::CompositionAccess::seal(s.native));
  s.sealed = true; s.needs_seal = false;
  return Status::Ok();
}
Result<RegionValue> value(void *prepared, void *invocation, std::size_t port) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (!s.prepared || port >= a.public_outputs || !s.output.valid() || !s.backing.valid())
    return std::unexpected(Status::FailedPrecondition("native ragged output is not cold prepared"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, compute::ConstTensorView::create(s.output.description(), s.backing.view()));
  return RegionValue{s.output.description(), tensor, &tensor_type, &s.output};
}
Result<compute::ReadLease> lease(void *prepared, void *invocation, std::size_t port) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (!s.result || port >= a.public_outputs)
    return std::unexpected(Status::FailedPrecondition("native ragged has no successful output owner"));
  const auto &run = vn::Access::invocation_state(s.native)->run;
  {
    std::lock_guard lock(a.runtime->mutex);
    if (run->pending || run->retiring || !run->terminal.ok() ||
        !run->observation_error.ok() || !run->submission_failure.ok())
      return std::unexpected(Status::FailedPrecondition("native ragged output has not retired successfully"));
  }
  HYLE_ASSIGN_OR_RETURN(auto backing, vn::MemoryAccess::export_storage(
      a.runtime, a.output_resource, s.output, s.allocator));
  return compute::ReadLease::adopt(std::move(backing), {.allocator = s.allocator});
}
cd::CompletionEndpoint endpoint(void *prepared, void *invocation) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared); const auto &s = *static_cast<Slot *>(invocation);
  if (!s.native.valid()) return {&observed_endpoint, &s};
  const auto &run = *vn::Access::invocation_state(s.native)->run;
  if (run.async || run.needs_drain || std::ranges::any_of(run.stages, [](const auto &stage) { return bool(stage.iree); }))
    return {&observed_endpoint, &run};
  return region_ordered_endpoint(a.runtime);
}
bool imports(void *prepared, void *invocation, cd::CompletionEndpoint source) noexcept {
  const auto own = endpoint(prepared, invocation);
  return own.authority != &observed_endpoint &&
      imports_region_ordered_endpoint(static_cast<Artifact *>(prepared)->runtime, source);
}
Result<compute::Completion> submit(void *prepared, void *invocation,
    std::span<const cd::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView>, std::span<const compute::TensorView>) noexcept {
  const auto &a = *static_cast<Artifact *>(prepared); auto &s = *static_cast<Slot *>(invocation);
  if (!s.prepared) return std::unexpected(Status::FailedPrecondition("native ragged layout is not prepared"));
  for (const auto dependency : dependencies) HYLE_RETURN_IF_ERROR(vn::qualify_copy_dependency(a.runtime, dependency));
  s.result.reset();
  HYLE_ASSIGN_OR_RETURN(auto accepted, s.native.submit());
  auto transfer = cd::ResultAccess::take_storage(std::move(accepted));
  s.result = std::move(transfer.owner);
  return std::move(transfer.completion);
}

// The selected-layout protocol table is shared with the original native batch
// preparation coordinator. No separate queue, controller or executor lives here.
const bd::NativeBatchRegionVTable batch_table{
    .inspect = [](void *prepared, void *invocation) noexcept -> Result<bd::NativeBatchRegionFacts> {
      const auto &a = *static_cast<Artifact *>(prepared); const auto &s = *static_cast<Slot *>(invocation);
      return bd::NativeBatchRegionFacts{
          .node = a.node, .paged_program = nullptr, .capacity = a.operation.capacity,
          .mla = a.mla, .causal = a.operation.causal,
          .window_left = a.operation.window_left,
          .inputs = std::span(a.inputs).first(a.payload_count),
          .input_ports = std::span(a.ports).first(a.payload_count),
          .metadata_input = a.inputs[a.payload_count],
          .output_query_rows = std::uint32_t(a.descriptions[0].dimensions()[0]),
          .layout = s.layout};
    },
    .prepare = &prepare_layout,
    .reset = &reset};
const graph::detail::RegionBindingReleaseVTable binding_release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      const auto &slot = *static_cast<const Slot *>(invocation);
      if (!slot.native.valid())
        return slot.result ? Status::FailedPrecondition("native result has no invocation") : Status::Ok();
      return vn::PagedBatchAccess::can_detach_inputs(slot.native, slot.result);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.result.reset();
      if (slot.native.valid()) vn::PagedBatchAccess::detach_inputs(slot.native);
      slot.needs_seal = false;
    }};

const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Artifact *>(p)); },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept { return compute::AliasSupport{}; },
    .create_invocation = &create_slot,
    .destroy_invocation = [](void *p, mem::AllocatorRef a) noexcept { a.destroy(static_cast<Slot *>(p)); },
    .delegated_submit = &submit, .endpoint = &endpoint, .can_import_endpoint = &imports,
    .output_contract = [](void *, std::size_t) noexcept { return RegionOutputContract{
        .kind = RegionOutputKind::kBoundGenerated, .projection = RegionOutputProjection::kTensorView}; },
    .output_value = &value, .prepare_bindings = &bind, .seal_bindings = &seal, .output_lease = &lease,
    .query_extension = [](void *, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == graph::detail::kRegionBindingReleaseExtensionId) return &binding_release_table;
      if (id == bd::kNativeBatchRegionExtensionId) return &batch_table;
      return nullptr;
    }};

} // namespace

bool native_ragged_operation(const compute::Operation &operation) noexcept {
  return operation.custom_as<compute::RaggedAttention>() ||
         operation.custom_as<compute::RaggedMlaAttention>();
}
Status propose_native_ragged(const RegionDiscoveryRequest &request,
    std::span<const std::uint8_t> needed, RegionProposalWriter &writer) noexcept {
  for (std::size_t i = 0; i < request.graph.node_count(); ++i) {
    if (!needed[i] || (!request.allowed_nodes.empty() && !request.allowed_nodes[i])) continue;
    HYLE_ASSIGN_OR_RETURN(auto node, request.graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph.operation(node));
    if (!native_ragged_operation(operation)) continue;
    HYLE_ASSIGN_OR_RETURN(auto selected, recipe(operation));
    const auto admitted = qualify(selected.operation);
    if (!admitted.ok()) { if (admitted.IsUnimplemented()) continue; return admitted; }
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added, writer.try_add({.nodes = nodes, .route = kNativeRaggedRoute,
        .replay_class = compute::ProviderReplayClass::kManaged,
        .implementation = compute::ProviderImplementationId{0x524147474544564bULL}, .preference = 100}));
    if (!added) break;
  }
  return Status::Ok();
}
Status native_ragged_resources(void *context, const RegionCompileRequest &request,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *proposal) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto operation, qualify(request));
  (void)operation;
  if (inputs.size() != request.inputs.size() || outputs.size() != request.outputs.size())
    return Status::InvalidArgument("native ragged resource boundary differs");
  const auto &runtime = static_cast<SourceContext *>(context)->runtime;
  const cd::ResourceBindingRequirement requirement{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .forbidden_properties = mem::MemoryProperty::kHostAccessible, .minimum_alignment = 1,
      .exact_domain = {runtime->device.get(), static_cast<std::uintptr_t>(cv::MemoryKind::kDevice)},
      .requires_exact_domain = true};
  std::ranges::fill(inputs, RegionInputRequirement{.resource = requirement});
  std::ranges::fill(outputs, RegionOutputRequirement{.resource = requirement,
      .value = {.kind = RegionOutputKind::kBoundGenerated, .projection = RegionOutputProjection::kTensorView}});
  *proposal = {.runtime = runtime_properties, .workspace_resource = requirement};
  return Status::Ok();
}
compute::ProviderPrepareOutcome compile_native_ragged(void *context,
    const RegionCompileRequest &request, mem::AllocatorRef allocator, ProviderRegionPrepared *out) noexcept {
  using Outcome = compute::ProviderPrepareOutcome;
  *out = {};
  auto selected = qualify(request);
  if (!selected) return selected.error().IsUnimplemented() ? Outcome::unsupported(selected.error()) : Outcome::error(selected.error());
  if (request.input_resources.size() != request.inputs.size() ||
      request.output_resources.size() != request.outputs.size())
    return Outcome::error(Status::InvalidArgument("native ragged physical resource arity differs"));
  auto *a = allocator.construct<Artifact>();
  if (!a) return Outcome::error(Status::ResourceExhausted("native ragged artifact metadata"));
  *out = {.vtable = &prepared_table, .state = a};
  a->runtime = static_cast<SourceContext *>(context)->runtime;
  a->operation = selected->operation; a->mla = selected->mla;
  a->payload_count = a->mla == compute::MlaCacheLayout::kNone ? 3u : 2u;
  a->node = request.nodes[0]; a->workload = request.workload;
  a->allocation.resource_budget = request.resource_budget;
  a->boundary_count = request.inputs.size(); a->public_outputs = request.outputs.size();
  const auto build = [&]() noexcept -> Status {
    const auto consider = [&](mem::MemoryResourceRef resource) {
      if (!vn::MemoryAccess::validate_resource(a->runtime, resource).ok()) return;
      if (resource.properties().has(mem::MemoryProperty::kHostAccessible)) {
        if (!a->allocation.staging.valid()) a->allocation.staging = resource;
      } else if (!a->allocation.device.valid()) a->allocation.device = resource;
    };
    consider(request.workspace_resource);
    for (const auto &domain : request.available_storage_domains) consider(domain.resource);
    if (!a->allocation.device.valid())
      return Status::Unimplemented("native ragged requires an actual Vulkan device domain");
    a->output_resource = a->public_outputs ? request.output_resources[0] : a->allocation.device;
    for (std::size_t port = 0; port <= a->payload_count; ++port) {
      HYLE_ASSIGN_OR_RETURN(a->inputs[port], request.graph->node_input(a->node, port));
      if (port == a->payload_count && a->inputs[port].kind() != ValueKind::kInput)
        return Status::Unimplemented("native ragged metadata needs its owned external layout binding");
      HYLE_ASSIGN_OR_RETURN(a->descriptions[port], request.graph->value_description(a->inputs[port]));
      const auto found = std::ranges::find(request.inputs, a->inputs[port]);
      if (found == request.inputs.end()) return Status::InvalidArgument("native ragged physical input is missing");
      a->ports[port] = static_cast<std::uint32_t>(found - request.inputs.begin());
      HYLE_RETURN_IF_ERROR(vn::MemoryAccess::validate_resource(a->runtime, request.input_resources[a->ports[port]]));
    }
    compute::TensorDesc expected;
    HYLE_ASSIGN_OR_RETURN(auto original, request.graph->operation(a->node));
    HYLE_RETURN_IF_ERROR(compute::infer_output_descriptions(original,
        std::span(a->descriptions).first(a->payload_count+1), std::span(&expected, 1)));
    HYLE_ASSIGN_OR_RETURN(auto output, request.graph->result(a->node));
    HYLE_ASSIGN_OR_RETURN(auto description, request.graph->value_description(output));
    if (description != expected)
      return Status::InvalidArgument("native Vulkan ragged concrete Q/output descriptions differ");
    out->properties = {.persistent_bytes = sizeof(Artifact), .runtime = runtime_properties,
        .invocation_private_bytes = std::nullopt};
    return Status::Ok();
  };
  const auto status = build();
  if (!status.ok()) return status.IsUnimplemented() ? Outcome::unsupported(status) : Outcome::error(status);
  return Outcome::prepared();
}
} // namespace hyle::graph::vulkan::detail
