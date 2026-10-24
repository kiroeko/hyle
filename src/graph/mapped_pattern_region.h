#pragma once

#include "mapped_native_region.h"
#include "closed_region.h"
#include "hyle/graph/region_provider.h"
#include "../compute/provider_internal.h"

#include <algorithm>
#include <array>
#include <limits>

namespace hyle::graph::detail {
namespace c = compute;

// Exact cold recipes share only their native resource/prepared adapter. A new
// recipe owns its mathematical matcher; queues and hot execution are unchanged.
template <const c::detail::ProviderVTable &(*Original)() noexcept, class... Recipes>
struct MappedPatternRegions final {
static_assert([] {
  constexpr std::array routes{Recipes::kRoute...};
  for (std::size_t a = 0; a < routes.size(); ++a)
    for (std::size_t b = a + 1; b < routes.size(); ++b)
      if (routes[a] == routes[b]) return false;
  return true;
}(), "mapped pattern routes must be distinct");
static const c::detail::ProviderVTable &original() noexcept { return Original(); }

template <class P, std::size_t Ports>
static Status native_proposal(void *context, const Graph &graph, const P &p,
    std::array<c::TensorDesc, Ports> &inputs, c::TensorDesc &output,
    std::array<c::detail::ResourceBindingRequirement, Ports> &requirements,
    c::detail::ResourceBindingRequirement &destination,
    c::detail::ProviderOperationProposal &proposal, const c::WorkloadContext &workload = {}) noexcept {
  for (std::size_t n = 0; n < p.input_count; ++n)
    inputs[n] = graph.value_description(p.inputs[n]).value();
  output = graph.value_description(p.output).value();
  std::array<c::TensorDesc, 1> inferred;
  const auto op = p.composite();
  HYLE_RETURN_IF_ERROR(c::infer_output_descriptions(op, std::span(inputs).first(p.input_count), inferred));
  if (inferred[0] != output)
    return Status::Unimplemented("mapped pattern composite output differs from the public graph");
  HYLE_RETURN_IF_ERROR(original().propose(context,
      {.operation = op, .inputs = std::span(inputs).first(p.input_count),
       .outputs = std::span(&output, 1), .workload = workload},
      std::span(requirements).first(p.input_count), std::span(&destination, 1), &proposal));
  if (proposal.cost.persistent_bytes) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, mapped_native_region::artifact_bytes(p.input_count, 1));
    if (*proposal.cost.persistent_bytes > SIZE_MAX - bytes)
      return Status::OutOfRange("mapped pattern region metadata bytes overflow");
    *proposal.cost.persistent_bytes += bytes;
  }
  return Status::Ok();
}
template <class Recipe>
static Status propose_one(void *context, const RegionDiscoveryRequest &request,
    const ClosedRegionPattern::Uses &uses, RegionProposalWriter &writer) noexcept {
  using Pattern = typename Recipe::Pattern;
  constexpr auto kMaxInputs = Recipe::kMaxInputs;
  const auto &graph = request.graph;
  if (!request.allowed_nodes.empty() && request.allowed_nodes.size() != graph.node_count())
    return Status::InvalidArgument("mapped pattern allowed-node count differs");
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    Pattern p;
    if (!Recipe::pattern(graph, node, p) || !Recipe::closed(graph, p, uses, request.allowed_nodes)) continue;
    std::array<c::TensorDesc, kMaxInputs> inputs;
    c::TensorDesc output;
    std::array<c::detail::ResourceBindingRequirement, kMaxInputs> requirements;
    c::detail::ResourceBindingRequirement destination;
    c::detail::ProviderOperationProposal proposal;
    const auto status = native_proposal(context, graph, p, inputs, output, requirements, destination, proposal);
    if (!status.ok()) {
      if (status.IsUnimplemented() || status.IsInvalidArgument()) continue;
      return status;
    }
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = p.span(), .cost = proposal.cost,
        .route = Recipe::kRoute, .replay_class = proposal.runtime.replay_class,
        .implementation = c::ProviderImplementationId{1}, .policy = Recipe::kPolicy,
        .preference = proposal.preference == (std::numeric_limits<std::uint32_t>::max)()
            ? 200 : proposal.preference}));
    if (!more) return Status::Ok();
  }
  return Status::Ok();
}

static bool same_requirement(const c::ResourceRequirement &a, const c::ResourceRequirement &b) noexcept {
  return a.required_properties == b.required_properties && a.forbidden_properties == b.forbidden_properties &&
      a.minimum_alignment == b.minimum_alignment && a.exact_resource == b.exact_resource &&
      a.exact_domain == b.exact_domain && a.requires_exact_domain == b.requires_exact_domain;
}
template <class Recipe>
static Status resources_one(void *context, const RegionCompileRequest &request, mem::AllocatorRef,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *result) noexcept {
  using Pattern = typename Recipe::Pattern;
  constexpr auto kMaxInputs = Recipe::kMaxInputs;
  *result = {};
  Pattern p;
  HYLE_RETURN_IF_ERROR(Recipe::qualify(request, p));
  if (inputs.size() != request.inputs.size() || outputs.size() != 1)
    return Status::InvalidArgument("mapped pattern resource count differs");
  std::array<c::TensorDesc, kMaxInputs> descriptions;
  c::TensorDesc output;
  std::array<c::detail::ResourceBindingRequirement, kMaxInputs> requirements;
  c::detail::ResourceBindingRequirement destination;
  c::detail::ProviderOperationProposal proposal;
  HYLE_RETURN_IF_ERROR(native_proposal(context, *request.graph, p, descriptions, output,
      requirements, destination, proposal, request.workload));
  const auto logical_inputs = std::span(p.inputs).first(p.input_count);
  for (std::size_t port = 0; port < request.inputs.size(); ++port) {
    const auto first = std::size_t(std::ranges::find(
        logical_inputs, request.inputs[port]) - logical_inputs.begin());
    for (std::size_t n = first + 1; n < p.input_count; ++n)
      if (p.inputs[n] == request.inputs[port] && !same_requirement(requirements[first], requirements[n]))
        return Status::Unimplemented("mapped pattern shared port has different native resource requirements");
    inputs[port] = {.resource = requirements[first]};
  }
  outputs[0] = {.resource = destination};
  *result = {.runtime = proposal.runtime, .workspace_resource = proposal.workspace_resource,
      .compilation_storage = proposal.cost.compilation_storage};
  return Status::Ok();
}

template <class Recipe>
static c::ProviderPrepareOutcome compile_one(void *context, const RegionCompileRequest &request,
    mem::AllocatorRef allocator, ProviderRegionPrepared *result) noexcept {
  using Pattern = typename Recipe::Pattern;
  constexpr auto kMaxInputs = Recipe::kMaxInputs;
  *result = {};
  Pattern p;
  const auto qualification = Recipe::qualify(request, p);
  if (!qualification.ok()) return c::ProviderPrepareOutcome::unsupported(qualification);
  if (request.input_resources.size() != request.inputs.size() || request.output_resources.size() != 1)
    return c::ProviderPrepareOutcome::error(Status::InvalidArgument("mapped pattern physical resource count differs"));
  std::array<c::TensorDesc, kMaxInputs> inputs;
  c::TensorDesc output;
  std::array<c::detail::ResourceBindingRequirement, kMaxInputs> requirements;
  c::detail::ResourceBindingRequirement destination;
  c::detail::ProviderOperationProposal proposal;
  const auto status = native_proposal(context, *request.graph, p, inputs, output,
      requirements, destination, proposal, request.workload);
  if (!status.ok()) return status.IsUnimplemented() ? c::ProviderPrepareOutcome::unsupported(status)
                                                  : c::ProviderPrepareOutcome::error(status);
  std::array<std::size_t, kMaxInputs> ports;
  std::array<mem::MemoryResourceRef, kMaxInputs> resources;
  for (std::size_t n = 0; n < p.input_count; ++n) {
    ports[n] = std::size_t(std::ranges::find(request.inputs, p.inputs[n]) - request.inputs.begin());
    resources[n] = request.input_resources[ports[n]];
  }
  return mapped_native_region::prepare(original(), context,
      {.operation = p.composite(), .inputs = std::span(inputs).first(p.input_count),
       .outputs = std::span(&output, 1), .input_resources = std::span(resources).first(p.input_count),
       .output_resources = request.output_resources, .workspace_resource = request.workspace_resource,
       .route = proposal.route, .workload = request.workload}, proposal,
      std::span(ports).first(p.input_count), request.inputs.size(), allocator, result);
}
static Status propose(void *context, const RegionDiscoveryRequest &r,
    mem::AllocatorRef allocator, RegionProposalWriter &writer) noexcept {
  ClosedRegionPattern::Uses uses(allocator);
  HYLE_RETURN_IF_ERROR(uses.initialize(r.graph));
  Status status = Status::Ok();
  ((status.ok() ? status = propose_one<Recipes>(context, r, uses, writer) : status), ...);
  return status;
}
static Status resources(void *context, const RegionCompileRequest &r, mem::AllocatorRef allocator,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *result) noexcept {
  Status status = Status::Unimplemented("unknown mapped pattern route");
  const bool found = ((r.route == Recipes::kRoute
      ? (status = resources_one<Recipes>(context, r, allocator, inputs, outputs, result), true)
      : false) || ...);
  (void)found;
  return status;
}
static c::ProviderPrepareOutcome compile(void *context, const RegionCompileRequest &r,
    mem::AllocatorRef allocator, ProviderRegionPrepared *result) noexcept {
  auto outcome = c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("unknown mapped pattern route"));
  const bool found = ((r.route == Recipes::kRoute
      ? (outcome = compile_one<Recipes>(context, r, allocator, result), true) : false) || ...);
  (void)found;
  return outcome;
}
inline static const RegionProviderVTable kRegion{.propose = &propose, .propose_resources = &resources, .compile = &compile};
static const void *query_extension(void *context, c::ProviderExtensionId id) noexcept {
  if (id == kRegionProviderExtensionId) return &kRegion;
  return original().query_extension ? original().query_extension(context, id) : nullptr;
}
static const c::detail::ProviderVTable &decorated() noexcept {
  static const auto table = [] {
    auto value = original(); value.query_extension = &query_extension; return value;
  }();
  return table;
}
static c::Provider decorate(c::Provider provider) noexcept {
  return c::detail::ProviderAccess::with_vtable(std::move(provider), &decorated());
}
};
} // namespace hyle::graph::detail
