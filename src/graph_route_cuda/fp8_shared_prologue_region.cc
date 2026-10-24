#include "fp8_shared_prologue_region.h"
#include "../graph/mapped_native_region.h"

#include "hyle/compute/providers/cuda.h"
#include "hyle/container/small_vector.h"
#include "../compute/provider_internal.h"
#include "../compute_provider_cuda/fp8_shared_linear.h"

#include <algorithm>
#include <array>
#include <bit>
#include <tuple>

namespace hyle::graph::cuda::internal {
namespace {
namespace c = compute;
using c::detail::ProviderAccess;
using Linear = c::model_ops::Float8BlockLinear;
using Pair = c::cuda_native::internal::Float8BlockLinearPair;
inline constexpr c::RoutePolicyId kPolicy{0x48594c4546503853ULL, 1, 1};

const c::detail::ProviderVTable &original() noexcept {
  static const auto *table = ProviderAccess::vtable(c::cuda_native::sglang_provider());
  return *table;
}
struct Pattern final {
  std::array<NodeId, 2> nodes;
  std::array<ValueId, 3> inputs;
  std::array<ValueId, 2> outputs;
  Pair descriptor;
  c::Operation composite() const noexcept { return c::Operation::custom(descriptor); }
  std::size_t boundary_count() const noexcept { return inputs[1] == inputs[2] ? 2 : 3; }
};
bool linear(const Graph &graph, NodeId node, Linear &result) noexcept {
  const auto operation = graph.operation(node);
  if (!operation) return false;
  const auto value = operation->custom_as<Linear>();
  if (!value || graph.node_input_count(node).value() != 2 ||
      graph.node_output_count(node).value() != 1 || graph.node_resource_count(node).value() != 0)
    return false;
  result = *value;
  return true;
}
bool pattern(const Graph &graph, std::array<NodeId, 2> nodes, Pattern &p) noexcept {
  if (nodes[0] == nodes[1]) return false;
  std::ranges::sort(nodes, {}, &NodeId::ordinal);
  Linear first, second;
  if (!linear(graph, nodes[0], first) || !linear(graph, nodes[1], second) ||
      first.scale_policy != second.scale_policy ||
      std::bit_cast<std::uint32_t>(first.absmax_floor) != std::bit_cast<std::uint32_t>(second.absmax_floor))
    return false;
  const auto a = graph.node_input(nodes[0], 0).value();
  if (graph.node_input(nodes[1], 0).value() != a) return false;
  p = {nodes, {a, graph.node_input(nodes[0], 1).value(), graph.node_input(nodes[1], 1).value()},
      {graph.result(nodes[0]).value(), graph.result(nodes[1]).value()},
      {first.scale_policy, first.absmax_floor}};
  // The native carrier reads three external inputs before either output.
  return std::ranges::none_of(p.inputs, [&](ValueId value) {
    return value == p.outputs[0] || value == p.outputs[1];
  });
}
Status native_proposal(void *context, const Graph &graph, const Pattern &p,
    std::array<c::TensorDesc, 3> &inputs, std::array<c::TensorDesc, 2> &outputs,
    std::array<c::detail::ResourceBindingRequirement, 3> &requirements,
    std::array<c::detail::ResourceBindingRequirement, 2> &destinations,
    c::detail::ProviderOperationProposal &proposal, const c::WorkloadContext &workload = {}) noexcept {
  for (std::size_t n = 0; n < inputs.size(); ++n) inputs[n] = graph.value_description(p.inputs[n]).value();
  for (std::size_t n = 0; n < outputs.size(); ++n) outputs[n] = graph.value_description(p.outputs[n]).value();
  std::array<c::TensorDesc, 2> inferred;
  const auto operation = p.composite();
  HYLE_RETURN_IF_ERROR(c::infer_output_descriptions(operation, inputs, inferred));
  if (inferred != outputs) return Status::Unimplemented("FP8 shared prologue changes an output boundary");
  HYLE_RETURN_IF_ERROR(original().propose(context,
      {.operation = operation, .inputs = inputs, .outputs = outputs, .workload = workload},
      requirements, destinations, &proposal));
  if (proposal.cost.persistent_bytes) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, graph::detail::mapped_native_region::artifact_bytes(inputs.size(), outputs.size()));
    if (*proposal.cost.persistent_bytes > SIZE_MAX - bytes)
      return Status::OutOfRange("FP8 shared prologue metadata bytes overflow");
    *proposal.cost.persistent_bytes += bytes;
  }
  return Status::Ok();
}

struct Entry final {
  NodeId node;
  ValueId activation;
  Linear operation;
  auto key() const noexcept {
    return std::tuple{activation.ordinal(), std::bit_cast<std::uint32_t>(operation.absmax_floor),
        operation.scale_policy};
  }
};
Status propose(void *context, const RegionDiscoveryRequest &request,
    mem::AllocatorRef allocator, RegionProposalWriter &writer) noexcept {
  const auto &graph = request.graph;
  if (!request.allowed_nodes.empty() && request.allowed_nodes.size() != graph.node_count())
    return Status::InvalidArgument("FP8 shared prologue allowed-node count differs");
  container::SmallVector<Entry, 16, mem::AllocatorRef> entries(allocator);
  for (std::size_t n = 0; n < graph.node_count(); ++n) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(n));
    if (!request.allowed_nodes.empty() && !request.allowed_nodes[n]) continue;
    Linear operation;
    if (!linear(graph, node, operation)) continue;
    if (!entries.try_push_back(Entry{node, graph.node_input(node, 0).value(), operation}))
      return Status::ResourceExhausted("FP8 shared prologue discovery metadata allocation failed");
  }
  // Group by exact logical producer/quantization contract once, rather than
  // scanning the whole graph again for each projection or comparing pointers.
  std::ranges::sort(entries, [](const Entry &a, const Entry &b) {
    return a.key() != b.key() ? a.key() < b.key() : a.node.ordinal() < b.node.ordinal();
  });
  for (std::size_t n = 1; n < entries.size(); ++n) {
    if (entries[n - 1].key() != entries[n].key()) continue;
    Pattern p;
    if (!pattern(graph, {entries[n - 1].node, entries[n].node}, p)) continue;
    HYLE_ASSIGN_OR_RETURN(auto contractible, is_region_contractible(graph, p.nodes, allocator));
    if (!contractible) continue;
    std::array<c::TensorDesc, 3> inputs;
    std::array<c::TensorDesc, 2> outputs;
    std::array<c::detail::ResourceBindingRequirement, 3> requirements;
    std::array<c::detail::ResourceBindingRequirement, 2> destinations;
    c::detail::ProviderOperationProposal proposal;
    const auto status = native_proposal(context, graph, p, inputs, outputs, requirements, destinations, proposal);
    if (!status.ok()) {
      if (status.IsUnimplemented() || status.IsInvalidArgument()) continue;
      return status;
    }
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = p.nodes, .cost = proposal.cost,
        .route = 1, .replay_class = proposal.runtime.replay_class,
        .implementation = c::ProviderImplementationId{1}, .policy = kPolicy, .preference = 200}));
    if (!more) break;
  }
  return Status::Ok();
}
Status qualify(const RegionCompileRequest &request, Pattern &p) noexcept {
  if (!request.graph || request.route != 1 || request.nodes.size() != 2 || request.outputs.size() != 2 ||
      request.has_unknown_effects || !request.effect_accesses.empty() ||
      !pattern(*request.graph, {request.nodes[0], request.nodes[1]}, p) ||
      request.inputs.size() != p.boundary_count())
    return Status::Unimplemented("FP8 shared prologue boundary differs");
  if (!std::ranges::all_of(p.inputs, [&](ValueId value) {
        return std::ranges::count(request.inputs, value) == 1;
      }) || !std::ranges::all_of(p.outputs, [&](ValueId value) {
        return std::ranges::count(request.outputs, value) == 1;
      }))
    return Status::Unimplemented("FP8 shared prologue misses an original boundary");
  return Status::Ok();
}
Status resources(void *context, const RegionCompileRequest &request, mem::AllocatorRef,
    std::span<RegionInputRequirement> inputs, std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *result) noexcept {
  *result = {};
  Pattern p;
  HYLE_RETURN_IF_ERROR(qualify(request, p));
  if (inputs.size() != request.inputs.size() || outputs.size() != 2)
    return Status::InvalidArgument("FP8 shared prologue resource count differs");
  std::array<c::TensorDesc, 3> descriptions;
  std::array<c::TensorDesc, 2> results;
  std::array<c::detail::ResourceBindingRequirement, 3> requirements;
  std::array<c::detail::ResourceBindingRequirement, 2> destinations;
  c::detail::ProviderOperationProposal proposal;
  HYLE_RETURN_IF_ERROR(native_proposal(context, *request.graph, p, descriptions, results,
      requirements, destinations, proposal, request.workload));
  for (std::size_t n = 0; n < p.inputs.size(); ++n) {
    const auto port = std::size_t(std::ranges::find(request.inputs, p.inputs[n]) - request.inputs.begin());
    inputs[port] = {.resource = requirements[n]};
  }
  for (std::size_t n = 0; n < p.outputs.size(); ++n) {
    const auto port = std::size_t(std::ranges::find(request.outputs, p.outputs[n]) - request.outputs.begin());
    outputs[port] = {.resource = destinations[n]};
  }
  *result = {.runtime = proposal.runtime, .workspace_resource = proposal.workspace_resource,
      .compilation_storage = proposal.cost.compilation_storage};
  return Status::Ok();
}
c::ProviderPrepareOutcome compile(void *context, const RegionCompileRequest &request,
    mem::AllocatorRef allocator, ProviderRegionPrepared *result) noexcept {
  *result = {};
  Pattern p;
  const auto qualification = qualify(request, p);
  if (!qualification.ok()) return c::ProviderPrepareOutcome::unsupported(qualification);
  if (request.input_resources.size() != request.inputs.size() || request.output_resources.size() != 2)
    return c::ProviderPrepareOutcome::error(Status::InvalidArgument("FP8 shared prologue physical resource count differs"));
  std::array<c::TensorDesc, 3> inputs;
  std::array<c::TensorDesc, 2> outputs;
  std::array<c::detail::ResourceBindingRequirement, 3> requirements;
  std::array<c::detail::ResourceBindingRequirement, 2> destinations;
  c::detail::ProviderOperationProposal proposal;
  const auto status = native_proposal(context, *request.graph, p, inputs, outputs, requirements, destinations, proposal, request.workload);
  if (!status.ok()) return status.IsUnimplemented() ? c::ProviderPrepareOutcome::unsupported(status)
                                                  : c::ProviderPrepareOutcome::error(status);
  std::array<std::size_t, 3> ports;
  std::array<std::size_t, 2> output_ports;
  std::array<mem::MemoryResourceRef, 3> sources;
  std::array<mem::MemoryResourceRef, 2> targets;
  for (std::size_t n = 0; n < p.inputs.size(); ++n) {
    ports[n] = std::size_t(std::ranges::find(request.inputs, p.inputs[n]) - request.inputs.begin());
    sources[n] = request.input_resources[ports[n]];
  }
  for (std::size_t n = 0; n < p.outputs.size(); ++n) {
    output_ports[n] = std::size_t(std::ranges::find(request.outputs, p.outputs[n]) - request.outputs.begin());
    targets[n] = request.output_resources[output_ports[n]];
  }
  return graph::detail::mapped_native_region::prepare(original(), context,
      {.operation = p.composite(), .inputs = inputs, .outputs = outputs,
       .input_resources = sources, .output_resources = targets, .workspace_resource = request.workspace_resource,
       .route = proposal.route, .workload = request.workload}, proposal, ports, request.inputs.size(), allocator, result, output_ports);
}
const RegionProviderVTable kRegion{.propose = &propose, .propose_resources = &resources, .compile = &compile};
const void *query_extension(void *context, c::ProviderExtensionId id) noexcept {
  if (id == kRegionProviderExtensionId) return &kRegion;
  return original().query_extension ? original().query_extension(context, id) : nullptr;
}
const c::detail::ProviderVTable &decorated() noexcept {
  static const auto table = [] {
    auto value = original(); value.query_extension = &query_extension; return value;
  }();
  return table;
}
} // namespace
c::Provider fp8_shared_prologue_provider() noexcept {
  return ProviderAccess::with_vtable(c::cuda_native::sglang_provider(), &decorated());
}
} // namespace hyle::graph::cuda::internal
