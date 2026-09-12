#include "hyle/graph/executable.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <utility>

#include "graph_state.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "plan_state.h"

namespace hyle::graph {
namespace {

Status plan_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph plan metadata allocation failed");
}

Status invocation_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph invocation allocation failed");
}

void mark_invocation_terminal(void *opaque, const Status &terminal) noexcept {
  auto *state = static_cast<detail::InvocationState *>(opaque);
  state->outputs_ready.store(terminal.ok(), std::memory_order_release);
  state->in_flight.store(false, std::memory_order_release);
}

void release_invocation_lifetime(void *opaque) noexcept {
  detail::release(static_cast<detail::InvocationState *>(opaque));
}

bool owns(const detail::PlanState &plan, ValueId value) noexcept {
  if (!value.valid() ||
      detail::GraphIdentity::generation(value) != plan.generation ||
      value.ordinal() >= plan.descriptions.size()) {
    return false;
  }
  const std::uint8_t role = plan.binding_roles[value.ordinal()];
  const ValueKind expected =
      (role & detail::kExternalInput) != 0
          ? ValueKind::kInput
          : ((role & detail::kConstant) != 0 ? ValueKind::kConstant
                                             : ValueKind::kNodeOutput);
  return value.kind() == expected;
}

struct ByteRange final {
  mem::MemoryResourceRef resource{};
  void *handle = nullptr;
  std::size_t offset = 0;
  std::size_t size = 0;
};

ByteRange byte_range(mem::ConstBufferView view) noexcept {
  return {view.resource(), view.native_handle(), view.offset(), view.size()};
}

ByteRange byte_range(mem::BufferView view) noexcept {
  return {view.resource(), view.native_handle(), view.offset(), view.size()};
}

bool same_allocation(const ByteRange &left, const ByteRange &right) noexcept {
  return left.resource == right.resource && left.handle == right.handle;
}

bool equal_range(const ByteRange &left, const ByteRange &right) noexcept {
  return same_allocation(left, right) && left.offset == right.offset &&
         left.size == right.size;
}

bool overlaps(const ByteRange &left, const ByteRange &right) noexcept {
  if (!same_allocation(left, right) || left.size == 0 || right.size == 0)
    return false;
  if (left.offset <= right.offset)
    return right.offset - left.offset < left.size;
  return left.offset - right.offset < right.size;
}

bool matches_alias_range(ByteRange source, ByteRange output,
                         std::size_t byte_offset) noexcept {
  return same_allocation(source, output) && output.offset >= source.offset &&
         output.offset - source.offset == byte_offset &&
         byte_offset <= source.size && output.size <= source.size - byte_offset;
}

bool matches_planned_alias(const detail::PlanState &plan,
                           std::size_t left_ordinal, ByteRange left,
                           std::size_t right_ordinal,
                           ByteRange right) noexcept {
  if (plan.alias_sources[left_ordinal] == right_ordinal) {
    return matches_alias_range(right, left, plan.alias_offsets[left_ordinal]);
  }
  if (plan.alias_sources[right_ordinal] == left_ordinal) {
    return matches_alias_range(left, right, plan.alias_offsets[right_ordinal]);
  }
  return false;
}

ByteRange bound_range(const detail::InvocationState &invocation,
                      std::size_t ordinal) noexcept {
  const std::uint8_t role = invocation.plan->binding_roles[ordinal];
  if ((role & detail::kConstant) != 0)
    return byte_range(invocation.readable[ordinal].storage());
  if ((role & detail::kExternalOutput) != 0)
    return byte_range(invocation.writable[ordinal].storage());
  return byte_range(invocation.readable[ordinal].storage());
}

Status validate_external_alias(const detail::InvocationState &invocation,
                               std::size_t ordinal,
                               ByteRange candidate) noexcept {
  const detail::PlanState &plan = *invocation.plan;
  const bool candidate_is_output =
      (plan.binding_roles[ordinal] & detail::kExternalOutput) != 0;

  auto check = [&](ValueId other_value) noexcept -> Status {
    const std::size_t other = other_value.ordinal();
    if (other == ordinal || invocation.bound[other] == 0)
      return Status::Ok();
    const bool other_is_output =
        (plan.binding_roles[other] & detail::kExternalOutput) != 0;
    if (!candidate_is_output && !other_is_output)
      return Status::Ok();
    const ByteRange existing = bound_range(invocation, other);
    if (!overlaps(candidate, existing))
      return Status::Ok();
    if (matches_planned_alias(plan, ordinal, candidate, other, existing)) {
      return Status::Ok();
    }
    return Status::InvalidArgument(
        "external graph tensor bindings overlap unsafely");
  };

  if (candidate_is_output) {
    for (ValueId input : plan.input_ids) {
      Status status = check(input);
      if (!status.ok())
        return status;
    }
    for (ValueId constant : plan.constant_ids) {
      Status status = check(constant);
      if (!status.ok())
        return status;
    }
  }
  for (ValueId output : plan.output_ids) {
    Status status = check(output);
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

Status initialize_plan_metadata(detail::PlanState &plan,
                                const detail::GraphState &graph) noexcept {
  if (!plan.steps.try_reserve(graph.nodes.size()) ||
      !plan.input_ids.try_reserve(graph.inputs.size()) ||
      !plan.constant_ids.try_reserve(graph.constants.size()) ||
      !plan.constant_bindings.try_resize(graph.constants.size()) ||
      !plan.output_ids.try_reserve(graph.outputs.size()) ||
      !plan.node_inputs.try_reserve(graph.edges.size()) ||
      !plan.node_outputs.try_reserve(graph.values.size()) ||
      !plan.descriptions.try_resize(graph.values.size()) ||
      !plan.workspace_requests.try_resize(graph.values.size()) ||
      !plan.node_steps.try_resize(graph.nodes.size()) ||
      !plan.value_materialized.try_resize(graph.values.size()) ||
      !plan.alias_sources.try_resize(graph.values.size()) ||
      !plan.alias_offsets.try_resize(graph.values.size()) ||
      !plan.binding_roles.try_resize(graph.values.size())) {
    return plan_allocation_failure();
  }

  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    plan.descriptions[index] = graph.values[index].description;
    plan.workspace_requests[index] = detail::kNoWorkspaceRequest;
    plan.value_materialized[index] = 0;
    plan.alias_sources[index] = detail::kNoInPlaceSource;
    plan.alias_offsets[index] = 0;
    plan.binding_roles[index] = 0;
  }
  for (std::size_t index = 0; index < plan.node_steps.size(); ++index)
    plan.node_steps[index] = detail::kNoNode;
  plan.logical_node_count = graph.nodes.size();
  for (std::uint32_t ordinal : graph.inputs) {
    if (!plan.input_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kExternalInput;
  }
  for (std::uint32_t ordinal : graph.constants) {
    if (!plan.constant_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kConstant;
  }
  for (std::uint32_t ordinal : graph.outputs) {
    if (!plan.output_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kExternalOutput;
  }
  return Status::Ok();
}

Status bind_constants(detail::PlanState &plan, const detail::GraphState &graph,
                      CompileOptions options) noexcept {
  if (options.constants.size() != graph.constants.size()) {
    return Status::InvalidArgument(
        "compile requires exactly one binding for every graph constant");
  }
  using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  using Flags = container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;
  Ordinals slots(plan.allocator);
  Flags seen(plan.allocator);
  if (!slots.try_resize(graph.values.size()) ||
      !seen.try_resize(graph.constants.size())) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < slots.size(); ++index)
    slots[index] = detail::kNoWorkspaceRequest;
  for (std::size_t index = 0; index < graph.constants.size(); ++index) {
    slots[graph.constants[index]] = static_cast<std::uint32_t>(index);
    seen[index] = 0;
  }

  for (const ConstantBinding &binding : options.constants) {
    if (!detail::owns(graph, binding.value) ||
        binding.value.kind() != ValueKind::kConstant) {
      return Status::InvalidArgument(
          "constant binding does not belong to this graph");
    }
    const std::uint32_t slot = slots[binding.value.ordinal()];
    if (slot == detail::kNoWorkspaceRequest || seen[slot] != 0)
      return Status::InvalidArgument("constant binding is duplicated");
    if (!binding.tensor.valid() ||
        binding.tensor.description() !=
            graph.values[binding.value.ordinal()].description) {
      return Status::InvalidArgument(
          "constant binding does not match its graph description");
    }
    plan.constant_bindings[slot] = binding.tensor;
    seen[slot] = 1;
  }
  for (std::uint8_t present : seen) {
    if (present == 0)
      return Status::InvalidArgument("graph constant binding is missing");
  }
  return Status::Ok();
}

struct RegionExtension final {
  compute::detail::ProviderExtensionRef extension{};
  std::uint32_t provider_index = 0;
};

struct RegionCandidate final {
  RegionProposal proposal{};
  std::uint32_t extension_index = 0;
  std::uint32_t proposal_order = 0;
};

using RegionExtensions =
    container::SmallVector<RegionExtension, 4, mem::AllocatorRef>;
using RegionCandidates =
    container::SmallVector<RegionCandidate, 16, mem::AllocatorRef>;

Status discover_region_candidates(const Graph &published_graph,
                                  const detail::GraphState &graph,
                                  const compute::Context &context,
                                  CompileOptions options,
                                  mem::AllocatorRef allocator,
                                  RegionExtensions &extensions,
                                  RegionCandidates &candidates) noexcept {
  if (options.region_selection == RegionSelectionPolicy::kDisabled ||
      graph.nodes.size() < 2) {
    return Status::Ok();
  }
  if (options.max_region_proposals == 0) {
    return Status::InvalidArgument(
        "enabled graph region selection requires a non-zero proposal budget");
  }
  if (options.max_region_proposals_per_provider == 0) {
    return Status::InvalidArgument(
        "enabled graph region selection requires a non-zero per-provider "
        "proposal budget");
  }
  if (options.max_region_proposals >
          (std::numeric_limits<std::uint32_t>::max)() ||
      options.max_region_proposals_per_provider >
          (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::OutOfRange(
        "graph region proposal budget exceeds the supported identity space");
  }
  if (context.provider_count() > (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::ResourceExhausted(
        "compute provider identity space is exhausted");
  }
  for (std::size_t index = 0; index < context.provider_count(); ++index) {
    auto extension = compute::detail::ProviderExtensionAccess::query(
        context, index, kRegionProviderExtensionId);
    if (!extension) {
      if (extension.error().IsUnimplemented())
        continue;
      return std::move(extension).error();
    }
    if (!extensions.try_push_back(RegionExtension{
            std::move(*extension), static_cast<std::uint32_t>(index)})) {
      return plan_allocation_failure();
    }
  }
  if (extensions.empty())
    return Status::Ok();

  container::SmallVector<RegionProposal, 32, mem::AllocatorRef> storage(
      allocator);
  if (!storage.try_resize(options.max_region_proposals) ||
      !candidates.try_reserve(options.max_region_proposals)) {
    return plan_allocation_failure();
  }

  std::size_t used = 0;
  for (std::size_t extension_index = 0; extension_index < extensions.size();
       ++extension_index) {
    const std::size_t remaining = storage.size() - used;
    const std::size_t provider_capacity =
        (std::min)(remaining, options.max_region_proposals_per_provider);
    RegionProposalWriter writer(
        std::span<RegionProposal>(storage).subspan(used, provider_capacity));
    struct Episode final {
      const Graph *graph = nullptr;
      RegionProposalWriter *writer = nullptr;
    } episode{&published_graph, &writer};
    const Status proposed = compute::detail::ProviderExtensionAccess::call(
        extensions[extension_index].extension,
        +[](void *provider_context, const void *interface_value,
            void *opaque) noexcept -> Status {
          const auto *interface =
              static_cast<const RegionProviderVTable *>(interface_value);
          auto &call = *static_cast<Episode *>(opaque);
          if (interface == nullptr || interface->propose == nullptr ||
              interface->compile == nullptr) {
            return Status::Internal(
                "graph region provider interface is incomplete");
          }
          return interface->propose(provider_context, *call.graph,
                                    *call.writer);
        },
        &episode);
    if (!proposed.ok())
      return proposed;

    for (const RegionProposal &proposal : writer.proposals()) {
      if (proposal.nodes.count < 2 ||
          proposal.nodes.end() > graph.nodes.size()) {
        return Status::InvalidArgument(
            "provider proposed an invalid graph region interval");
      }
      if (options.region_selection == RegionSelectionPolicy::kConservative &&
          proposal.cost.estimated_net_savings_ns == 0) {
        continue;
      }
      if (proposal.cost.dispatch_count == 0) {
        return Status::InvalidArgument(
            "provider proposed a graph region with no dispatch");
      }
      if (!candidates.try_push_back(RegionCandidate{
              proposal, static_cast<std::uint32_t>(extension_index),
              static_cast<std::uint32_t>(candidates.size())})) {
        return plan_allocation_failure();
      }
    }
    used += writer.size();
  }
  return Status::Ok();
}

struct RegionSelectionScore final {
  std::uint64_t savings = 0;
  std::uint64_t covered_nodes = 0;
};

template <class Unsigned>
Unsigned saturating_add(Unsigned left, Unsigned right) noexcept {
  const Unsigned limit = (std::numeric_limits<Unsigned>::max)();
  return right > limit - left ? limit : left + right;
}

bool better_score(RegionSelectionScore left,
                  RegionSelectionScore right) noexcept {
  return left.savings > right.savings ||
         (left.savings == right.savings &&
          left.covered_nodes > right.covered_nodes);
}

Status select_regions(RegionCandidates &candidates, mem::AllocatorRef allocator,
                      RegionCandidates &selected) noexcept {
  std::sort(
      candidates.begin(), candidates.end(),
      [](const RegionCandidate &left, const RegionCandidate &right) noexcept {
        if (left.proposal.nodes.end() != right.proposal.nodes.end())
          return left.proposal.nodes.end() < right.proposal.nodes.end();
        if (left.proposal.nodes.begin != right.proposal.nodes.begin)
          return left.proposal.nodes.begin < right.proposal.nodes.begin;
        if (left.extension_index != right.extension_index)
          return left.extension_index < right.extension_index;
        if (left.proposal.route != right.proposal.route)
          return left.proposal.route < right.proposal.route;
        return left.proposal_order < right.proposal_order;
      });
  using Indices = container::SmallVector<std::uint32_t, 32, mem::AllocatorRef>;
  using Scores =
      container::SmallVector<RegionSelectionScore, 32, mem::AllocatorRef>;
  using Decisions = container::SmallVector<std::uint8_t, 32, mem::AllocatorRef>;
  Indices predecessors(allocator);
  Scores scores(allocator);
  Decisions take(allocator);
  if (!predecessors.try_resize(candidates.size()) ||
      !scores.try_resize(candidates.size() + 1) ||
      !take.try_resize(candidates.size())) {
    return plan_allocation_failure();
  }
  scores[0] = {};
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const std::uint64_t begin = candidates[index].proposal.nodes.begin;
    const auto previous_end = std::upper_bound(
        candidates.begin(), candidates.begin() + index, begin,
        [](std::uint64_t value, const RegionCandidate &candidate) noexcept {
          return value < candidate.proposal.nodes.end();
        });
    const std::size_t previous_count =
        static_cast<std::size_t>(previous_end - candidates.begin());
    predecessors[index] = static_cast<std::uint32_t>(previous_count);
    RegionSelectionScore include = scores[previous_count];
    include.savings = saturating_add(
        include.savings,
        candidates[index].proposal.cost.estimated_net_savings_ns);
    include.covered_nodes = saturating_add(
        include.covered_nodes,
        static_cast<std::uint64_t>(candidates[index].proposal.nodes.count));
    const RegionSelectionScore exclude = scores[index];
    if (better_score(include, exclude)) {
      scores[index + 1] = include;
      take[index] = 1;
    } else {
      scores[index + 1] = exclude;
      take[index] = 0;
    }
  }

  std::size_t cursor = candidates.size();
  while (cursor != 0) {
    const std::size_t index = cursor - 1;
    if (take[index] == 0) {
      --cursor;
      continue;
    }
    if (!selected.try_push_back(candidates[index]))
      return plan_allocation_failure();
    cursor = predecessors[index];
  }
  std::reverse(selected.begin(), selected.end());
  return Status::Ok();
}

Status append_operation_step(detail::PlanState &plan,
                             const detail::GraphState &graph,
                             const compute::Context &context,
                             std::uint32_t node_index,
                             CompileFallbackReason fallback_reason) noexcept {
  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions input_descriptions(plan.allocator);
  Descriptions output_descriptions(plan.allocator);
  const detail::NodeRecord &node = graph.nodes[node_index];
  if (!input_descriptions.try_resize(node.input_count) ||
      !output_descriptions.try_resize(node.output_count)) {
    return plan_allocation_failure();
  }
  const std::uint32_t plan_input_begin =
      static_cast<std::uint32_t>(plan.node_inputs.size());
  const std::uint32_t plan_output_begin =
      static_cast<std::uint32_t>(plan.node_outputs.size());
  for (std::size_t index = 0; index < node.input_count; ++index) {
    const std::uint32_t ordinal = graph.edges[node.input_begin + index];
    input_descriptions[index] = graph.values[ordinal].description;
    if (!plan.node_inputs.try_push_back(ordinal))
      return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < node.output_count; ++index) {
    const std::uint32_t ordinal =
        node.output_begin + static_cast<std::uint32_t>(index);
    output_descriptions[index] = graph.values[ordinal].description;
    if (!plan.node_outputs.try_push_back(ordinal))
      return plan_allocation_failure();
    plan.value_materialized[ordinal] = 1;
  }
  auto prepared = compute::prepare(context, node.operation, input_descriptions,
                                   output_descriptions);
  if (!prepared)
    return std::move(prepared).error();
  const bool materializes =
      compute::operation_traits(node.operation).materializes;
  const std::uint32_t step_index =
      static_cast<std::uint32_t>(plan.steps.size());
  if (!plan.steps.try_emplace_back(std::move(*prepared), node_index,
                                   plan_input_begin, node.input_count,
                                   plan_output_begin, node.output_count,
                                   materializes, fallback_reason)) {
    return plan_allocation_failure();
  }
  plan.node_steps[node_index] = step_index;
  return Status::Ok();
}

Status append_region_step(
    detail::PlanState &plan, const Graph &published_graph,
    const detail::GraphState &graph, const RegionExtension &extension,
    const RegionProposal &proposal,
    std::span<const std::uint32_t> last_consumers,
    container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> &marks,
    std::uint32_t marker, CompileOptions options) noexcept {
  using Values = container::SmallVector<ValueId, 8, mem::AllocatorRef>;
  Values inputs(plan.allocator);
  Values outputs(plan.allocator);
  const std::uint32_t begin = proposal.nodes.begin;
  const std::uint32_t end =
      begin + static_cast<std::uint32_t>(proposal.nodes.count);

  for (std::uint32_t node_index = begin; node_index < end; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t ordinal = graph.edges[node.input_begin + input_index];
      const std::uint32_t producer = graph.values[ordinal].producer;
      if (producer != detail::kNoNode && producer >= begin && producer < end)
        continue;
      if (marks[ordinal] == marker)
        continue;
      marks[ordinal] = marker;
      if (!inputs.try_push_back(detail::value_id(graph, ordinal)))
        return plan_allocation_failure();
    }
  }
  for (std::uint32_t node_index = begin; node_index < end; ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t ordinal =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      const bool published =
          (plan.binding_roles[ordinal] & detail::kExternalOutput) != 0;
      const bool consumed_after = last_consumers[ordinal] != detail::kNoNode &&
                                  last_consumers[ordinal] >= end;
      if ((published || consumed_after) &&
          !outputs.try_push_back(detail::value_id(graph, ordinal))) {
        return plan_allocation_failure();
      }
    }
  }

  RegionCompileRequest request{
      .graph = &published_graph,
      .nodes = proposal.nodes,
      .inputs = inputs,
      .outputs = outputs,
      .constants = options.constants,
      .route = proposal.route,
  };
  ProviderRegionPrepared provider_prepared{};
  struct Episode final {
    const RegionCompileRequest *request = nullptr;
    mem::AllocatorRef allocator{};
    ProviderRegionPrepared *prepared = nullptr;
  } episode{&request, plan.allocator, &provider_prepared};
  const Status compiled = compute::detail::ProviderExtensionAccess::call(
      extension.extension,
      +[](void *provider_context, const void *interface_value,
          void *opaque) noexcept -> Status {
        const auto *interface =
            static_cast<const RegionProviderVTable *>(interface_value);
        auto &call = *static_cast<Episode *>(opaque);
        if (interface == nullptr || interface->compile == nullptr)
          return Status::Internal("graph region compile callback is missing");
        return interface->compile(provider_context, *call.request,
                                  call.allocator, call.prepared);
      },
      &episode);
  if (!compiled.ok()) {
    if (provider_prepared.vtable != nullptr &&
        provider_prepared.vtable->destroy_prepared != nullptr) {
      provider_prepared.vtable->destroy_prepared(provider_prepared.state,
                                                 plan.allocator);
    }
    return compiled;
  }
  const ProviderRegionPreparedVTable *table = provider_prepared.vtable;
  const compute::WorkspaceRequirement workspace =
      provider_prepared.properties.workspace;
  if (table == nullptr || table->destroy_prepared == nullptr ||
      table->alias_support == nullptr || table->create_invocation == nullptr ||
      table->destroy_invocation == nullptr || table->submit == nullptr ||
      workspace.alignment == 0 ||
      (workspace.alignment & (workspace.alignment - 1)) != 0 ||
      provider_prepared.properties.dispatch_count == 0 ||
      provider_prepared.properties.packed_constant_count >
          options.constants.size()) {
    if (table != nullptr && table->destroy_prepared != nullptr)
      table->destroy_prepared(provider_prepared.state, plan.allocator);
    return Status::Internal(
        "graph region provider returned an invalid compiled artifact");
  }

  const std::uint32_t plan_input_begin =
      static_cast<std::uint32_t>(plan.node_inputs.size());
  const std::uint32_t plan_output_begin =
      static_cast<std::uint32_t>(plan.node_outputs.size());
  for (ValueId input : inputs) {
    if (!plan.node_inputs.try_push_back(input.ordinal())) {
      table->destroy_prepared(provider_prepared.state, plan.allocator);
      return plan_allocation_failure();
    }
  }
  for (ValueId output : outputs) {
    if (!plan.node_outputs.try_push_back(output.ordinal())) {
      table->destroy_prepared(provider_prepared.state, plan.allocator);
      return plan_allocation_failure();
    }
    plan.value_materialized[output.ordinal()] = 1;
  }

  detail::PreparedRegion prepared(extension.extension, provider_prepared,
                                  plan.allocator);
  const std::uint32_t step_index =
      static_cast<std::uint32_t>(plan.steps.size());
  if (!plan.steps.try_emplace_back(
          std::move(prepared), begin, proposal.nodes.count, plan_input_begin,
          static_cast<std::uint32_t>(inputs.size()), plan_output_begin,
          static_cast<std::uint32_t>(outputs.size()))) {
    return plan_allocation_failure();
  }
  for (std::uint32_t node_index = begin; node_index < end; ++node_index)
    plan.node_steps[node_index] = step_index;
  ++plan.region_count;
  return Status::Ok();
}

Status prepare_steps(detail::PlanState &plan, const Graph &published_graph,
                     const detail::GraphState &graph,
                     const compute::Context &context,
                     CompileOptions options) noexcept {
  RegionExtensions extensions(plan.allocator);
  RegionCandidates candidates(plan.allocator);
  RegionCandidates selected(plan.allocator);
  Status status =
      discover_region_candidates(published_graph, graph, context, options,
                                 plan.allocator, extensions, candidates);
  if (!status.ok())
    return status;
  status = select_regions(candidates, plan.allocator, selected);
  if (!status.ok())
    return status;
  if (options.operation_fallback == OperationFallbackPolicy::kReject) {
    std::size_t covered_nodes = 0;
    for (const RegionCandidate &candidate : selected) {
      covered_nodes = saturating_add(
          covered_nodes,
          static_cast<std::size_t>(candidate.proposal.nodes.count));
    }
    if (covered_nodes != graph.nodes.size()) {
      return Status::Unimplemented(
          "graph specialization requires a single-operation fallback");
    }
  }

  container::SmallVector<std::int64_t, 32, mem::AllocatorRef>
      eligible_region_coverage(plan.allocator);
  if (!eligible_region_coverage.try_resize(graph.nodes.size() + 1))
    return plan_allocation_failure();
  for (std::int64_t &count : eligible_region_coverage)
    count = 0;
  for (const RegionCandidate &candidate : candidates) {
    ++eligible_region_coverage[candidate.proposal.nodes.begin];
    --eligible_region_coverage[static_cast<std::size_t>(
        candidate.proposal.nodes.end())];
  }
  for (std::size_t index = 1; index < eligible_region_coverage.size();
       ++index) {
    eligible_region_coverage[index] += eligible_region_coverage[index - 1];
  }

  container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> last_consumers(
      plan.allocator);
  container::SmallVector<std::uint32_t, 16, mem::AllocatorRef> marks(
      plan.allocator);
  if (!last_consumers.try_resize(graph.values.size()) ||
      !marks.try_resize(graph.values.size())) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    last_consumers[index] = detail::kNoNode;
    marks[index] = 0;
  }
  for (std::uint32_t node_index = 0; node_index < graph.nodes.size();
       ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      last_consumers[graph.edges[node.input_begin + input_index]] = node_index;
    }
  }

  std::size_t selection_index = 0;
  std::uint32_t node_index = 0;
  std::uint32_t marker = 1;
  while (node_index < graph.nodes.size()) {
    if (selection_index < selected.size() &&
        selected[selection_index].proposal.nodes.begin == node_index) {
      const RegionCandidate &candidate = selected[selection_index];
      status = append_region_step(
          plan, published_graph, graph, extensions[candidate.extension_index],
          candidate.proposal, last_consumers, marks, marker++, options);
      if (!status.ok())
        return status;
      node_index += candidate.proposal.nodes.count;
      ++selection_index;
      continue;
    }
    const CompileFallbackReason reason =
        options.region_selection == RegionSelectionPolicy::kDisabled
            ? CompileFallbackReason::kRegionSelectionDisabled
            : (eligible_region_coverage[node_index] == 0
                   ? CompileFallbackReason::kNoEligibleRegion
                   : CompileFallbackReason::kRegionNotSelected);
    status = append_operation_step(plan, graph, context, node_index, reason);
    if (!status.ok())
      return status;
    ++node_index;
  }
  return Status::Ok();
}

std::uint32_t alias_root(const detail::PlanState &plan,
                         std::uint32_t value) noexcept {
  while (plan.alias_sources[value] != detail::kNoInPlaceSource)
    value = plan.alias_sources[value];
  return value;
}

bool has_external_storage_root(const detail::PlanState &plan,
                               std::uint32_t value) noexcept {
  const std::uint32_t root = alias_root(plan, value);
  const std::uint8_t role = plan.binding_roles[root];
  return (role & (detail::kExternalInput | detail::kExternalOutput |
                  detail::kConstant)) != 0;
}

Status plan_workspace(detail::PlanState &plan,
                      const detail::GraphState &graph) noexcept {
  using LastUses = container::SmallVector<std::uint64_t, 16, mem::AllocatorRef>;
  using Requests =
      container::SmallVector<mem::LifetimeRequest, 8, mem::AllocatorRef>;

  LastUses last_uses(plan.allocator);
  Requests requests(plan.allocator);
  if (!last_uses.try_resize(graph.values.size()) ||
      !plan.step_workspace_requests.try_resize(plan.steps.size()) ||
      !requests.try_reserve(graph.values.size() + plan.steps.size())) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < last_uses.size(); ++index) {
    last_uses[index] = 0;
  }
  for (std::size_t index = 0; index < plan.step_workspace_requests.size();
       ++index) {
    plan.step_workspace_requests[index] = detail::kNoWorkspaceRequest;
  }

  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    const std::uint64_t terminal = step_index + 1;
    for (std::size_t output_index = 0; output_index < step.output_count;
         ++output_index) {
      const std::uint32_t output =
          plan.node_outputs[step.output_begin + output_index];
      last_uses[output] = (std::max)(last_uses[output], terminal);
    }
    for (std::size_t input_index = 0; input_index < step.input_count;
         ++input_index) {
      const std::uint32_t input =
          plan.node_inputs[step.input_begin + input_index];
      last_uses[input] = (std::max)(last_uses[input], terminal);
    }
  }

  // The schema supplies only a broad permission. A prepared provider narrows
  // it for the selected route and layout. Required view aliases preserve
  // storage without consuming the source; optional in-place aliases require
  // equal descriptions and the source's final read.
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    // Region aliases are optional in v1 and are not selected until G39 can
    // report every affected boundary port. A region must therefore support
    // distinct boundary outputs even if it advertises optional aliases.
    if (step.kind == detail::PlanStepKind::kRegion)
      continue;
    const detail::NodeRecord &node = graph.nodes[step.node_begin];
    const compute::OperationTraits traits =
        compute::operation_traits(node.operation);
    const compute::AliasPolicy alias = traits.alias;
    if (alias == compute::AliasPolicy::kNone || node.input_count == 0)
      continue;
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t output =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      if (alias == compute::AliasPolicy::kMustAliasInput0) {
        auto support = step.operation.alias_support(output_index, 0);
        if (!support || !support->supported) {
          return Status::Internal(
              "prepared provider omitted a required graph alias");
        }
        const std::uint32_t input = graph.edges[node.input_begin];
        plan.alias_sources[output] = input;
        plan.alias_offsets[output] = support->byte_offset;
        if ((plan.binding_roles[output] & detail::kExternalOutput) != 0) {
          if (has_external_storage_root(plan, input)) {
            plan.binding_roles[output] |= detail::kSourceAliasedOutput;
          } else {
            plan.binding_roles[output] |= detail::kInvocationOwnedOutput;
            plan.binding_roles[alias_root(plan, input)] |=
                detail::kRetainedOutputRoot;
          }
        }
        continue;
      }

      const bool output_is_external =
          (plan.binding_roles[output] & detail::kExternalOutput) != 0;
      const std::size_t candidate_count =
          alias == compute::AliasPolicy::kMayAliasAnyInput ? node.input_count
                                                           : std::size_t{1};
      for (std::size_t input_index = 0; input_index < candidate_count;
           ++input_index) {
        const std::uint32_t input = graph.edges[node.input_begin + input_index];
        auto support = step.operation.alias_support(output_index, input_index);
        if (!support)
          return std::move(support).error();
        if (!support->supported || support->byte_offset != 0)
          continue;
        const bool input_is_external =
            (plan.binding_roles[input] & detail::kExternalInput) != 0;
        const bool internal_pair =
            !output_is_external && !input_is_external &&
            graph.values[input].kind == ValueKind::kNodeOutput &&
            (plan.binding_roles[input] & detail::kExternalOutput) == 0 &&
            !has_external_storage_root(plan, input);
        const bool external_pair = output_is_external && input_is_external;
        if ((internal_pair || external_pair) &&
            last_uses[input] == step_index + 1 &&
            graph.values[input].description ==
                graph.values[output].description) {
          plan.alias_sources[output] = input;
          plan.alias_offsets[output] = 0;
          break;
        }
      }
    }
  }

  for (std::size_t ordinal = 0; ordinal < graph.values.size(); ++ordinal) {
    const detail::ValueRecord &value = graph.values[ordinal];
    if (value.kind != ValueKind::kNodeOutput ||
        plan.value_materialized[ordinal] == 0 ||
        ((plan.binding_roles[ordinal] & detail::kExternalOutput) != 0 &&
         (plan.binding_roles[ordinal] & detail::kInvocationOwnedOutput) == 0)) {
      continue;
    }
    auto size = value.description.required_bytes();
    if (!size)
      return std::move(size).error();
    if (requests.size() >= detail::kNoWorkspaceRequest) {
      return Status::ResourceExhausted(
          "graph workspace identity space is exhausted");
    }
    const std::uint32_t request = static_cast<std::uint32_t>(requests.size());
    const std::uint32_t producer_step = plan.node_steps[value.producer];
    if (producer_step == detail::kNoNode)
      return Status::Internal("graph value has no compiled producer step");
    const std::uint64_t begin = producer_step;
    const std::uint64_t end = (std::max)(begin + 1, last_uses[ordinal]);
    const std::uint32_t alias_source = plan.alias_sources[ordinal];
    const std::uint32_t alias_request =
        alias_source == detail::kNoInPlaceSource
            ? mem::kNoLifetimeAlias
            : plan.workspace_requests[alias_source];
    if (alias_source != detail::kNoInPlaceSource &&
        alias_request == detail::kNoWorkspaceRequest) {
      const bool materializes = plan.steps[producer_step].materializes;
      if (!materializes &&
          has_external_storage_root(plan, static_cast<std::uint32_t>(ordinal)))
        continue;
      return Status::Internal("graph internal alias source has no workspace");
    }
    if (!requests.try_push_back(mem::LifetimeRequest{
            .resource_slot = 0,
            .size = *size,
            // A derived view only promises its tensor's required alignment;
            // forcing the root's preferred 64-byte kernel alignment onto a
            // non-zero slice offset would reject otherwise valid aliases.
            .alignment =
                alias_request == mem::kNoLifetimeAlias
                    ? (std::max)(std::size_t{64},
                                 value.description.required_alignment())
                    : value.description.required_alignment(),
            .begin = begin,
            .end = end,
            .alias_of = alias_request,
            .alias_offset = plan.alias_offsets[ordinal],
            .flags =
                (plan.binding_roles[ordinal] & detail::kRetainedOutputRoot) != 0
                    ? mem::LifetimeFlag::kOutput
                    : mem::LifetimeFlag::kNone,
        })) {
      return plan_allocation_failure();
    }
    plan.workspace_requests[ordinal] = request;
  }

  // Provider scratchpads are per Invocation, not shared prepared state. Step
  // execution is ordered, so disjoint [step,step+1) requests may reuse one
  // backing while every operation/region invocation retains a stable view.
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    if (!plan.steps[step_index].materializes)
      continue;
    const compute::WorkspaceRequirement requirement =
        plan.steps[step_index].workspace_requirement();
    if (requirement.size == 0)
      continue;
    if (requests.size() >= detail::kNoWorkspaceRequest) {
      return Status::ResourceExhausted(
          "graph workspace identity space is exhausted");
    }
    const std::uint32_t request = static_cast<std::uint32_t>(requests.size());
    if (!requests.try_push_back(mem::LifetimeRequest{
            .resource_slot = 0,
            .size = requirement.size,
            .alignment = requirement.alignment,
            .begin = step_index,
            .end = step_index + 1,
        })) {
      return plan_allocation_failure();
    }
    plan.step_workspace_requests[step_index] = request;
  }

  const std::array resources{mem::system_resource()};
  const std::span<const mem::MemoryResourceRef> selected_resources =
      requests.empty() ? std::span<const mem::MemoryResourceRef>{}
                       : std::span<const mem::MemoryResourceRef>(resources);
  auto lifetime_plan = mem::LifetimePlanner::build(
      std::span<const mem::LifetimeRequest>(requests.data(), requests.size()),
      selected_resources);
  if (!lifetime_plan)
    return std::move(lifetime_plan).error();

  plan.workspace_plan = std::move(*lifetime_plan);
  if (!requests.empty()) {
    auto resource = plan.workspace_plan.resource_plan(0);
    if (!resource)
      return std::move(resource).error();
    plan.workspace_bytes = resource->peak_bytes;
    plan.workspace_backing_count = resource->backing_count;
  }
  return Status::Ok();
}

std::size_t step_constant_count(const detail::PlanState &plan,
                                const detail::PlanStep &step) noexcept {
  std::size_t count = 0;
  for (std::size_t index = 0; index < step.input_count; ++index) {
    const std::uint32_t ordinal = plan.node_inputs[step.input_begin + index];
    if ((plan.binding_roles[ordinal] & detail::kConstant) != 0)
      ++count;
  }
  return count;
}

Result<CompileReport> build_report(const detail::PlanState &plan,
                                   const compute::Context &context) noexcept {
  void *storage = plan.allocator.allocate(sizeof(detail::ReportState),
                                          alignof(detail::ReportState));
  if (storage == nullptr)
    return std::unexpected(plan_allocation_failure());
  auto *state = ::new (storage) detail::ReportState(plan.allocator);
  if (!state->steps.try_resize(plan.steps.size())) {
    detail::release(state);
    return std::unexpected(plan_allocation_failure());
  }

  state->backend = context.device().backend_id();
  state->workspace_bytes = plan.workspace_bytes;
  state->workspace_backing_count = plan.workspace_backing_count;
  state->node_count = plan.logical_node_count;
  state->region_count = plan.region_count;
  state->constant_count = plan.constant_ids.size();
  // PlanState conservatively keeps every compile-time binding because the v1
  // provider contract reports packed counts, not exact constant identities.
  state->borrowed_constant_count = plan.constant_ids.size();

  for (ValueId output : plan.output_ids) {
    if ((plan.binding_roles[output.ordinal()] &
         detail::kInvocationOwnedOutput) != 0) {
      ++state->invocation_owned_output_count;
    }
  }

  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const detail::PlanStep &step = plan.steps[index];
    CompileStepReport step_report{
        .kind = step.kind == detail::PlanStepKind::kRegion
                    ? CompileStepKind::kProviderRegion
                    : CompileStepKind::kOperationFallback,
        .fallback_reason = step.fallback_reason,
        .provider = step.provider(),
        .node_begin = step.node_begin,
        .node_count = step.node_count,
        .workspace = step.workspace_requirement(),
        .borrowed_constant_count = step_constant_count(plan, step),
        .dispatch_count = step.materializes ? 1u : 0u,
    };
    if (step.kind == detail::PlanStepKind::kRegion) {
      const RegionArtifactProperties properties = step.region.properties();
      if (properties.packed_constant_count >
          step_report.borrowed_constant_count) {
        detail::release(state);
        return std::unexpected(Status::Internal(
            "graph region packed more constants than its boundary contains"));
      }
      step_report.persistent_bytes = properties.persistent_bytes;
      step_report.packed_constant_count = properties.packed_constant_count;
      step_report.packed_constant_bytes = properties.packed_constant_bytes;
      step_report.copy_bytes = properties.copy_bytes;
      step_report.synchronization_count = properties.synchronization_count;
      step_report.dispatch_count = properties.dispatch_count;
      step_report.blocks_host = properties.blocks_host;
      step_report.uses_private_threads = properties.uses_private_threads;
      step_report.supports_concurrent_invocations =
          properties.supports_concurrent_invocations;
      state->packed_constant_count = saturating_add(
          state->packed_constant_count, step_report.packed_constant_count);
      state->packed_constant_bytes = saturating_add(
          state->packed_constant_bytes, step_report.packed_constant_bytes);
      state->copy_bytes =
          saturating_add(state->copy_bytes, step_report.copy_bytes);
      state->synchronization_count = saturating_add(
          state->synchronization_count, step_report.synchronization_count);
    } else {
      ++state->fallback_operation_count;
    }
    state->steps[index] = step_report;

    if (!state->mixed_providers) {
      if (!state->provider.valid()) {
        state->provider = step_report.provider;
      } else if (state->provider.id != step_report.provider.id ||
                 state->provider.version != step_report.provider.version) {
        state->provider = {};
        state->mixed_providers = true;
      }
    }
  }
  return detail::CompileReportAccess::adopt(state);
}

} // namespace

CompileReport::CompileReport(const CompileReport &other) noexcept
    : state_(other.state_) {
  detail::retain(state_);
}

CompileReport &CompileReport::operator=(const CompileReport &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

CompileReport::CompileReport(CompileReport &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

CompileReport &CompileReport::operator=(CompileReport &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

CompileReport::~CompileReport() { detail::release(state_); }

compute::BackendId CompileReport::backend() const noexcept {
  return state_ != nullptr ? state_->backend : compute::BackendId{};
}

compute::ProviderIdentity CompileReport::provider() const noexcept {
  return state_ != nullptr ? state_->provider : compute::ProviderIdentity{};
}

std::size_t CompileReport::workspace_bytes() const noexcept {
  return state_ != nullptr ? state_->workspace_bytes : 0;
}

std::size_t CompileReport::workspace_backing_count() const noexcept {
  return state_ != nullptr ? state_->workspace_backing_count : 0;
}

std::size_t CompileReport::node_count() const noexcept {
  return state_ != nullptr ? state_->node_count : 0;
}

std::size_t CompileReport::step_count() const noexcept {
  return state_ != nullptr ? state_->steps.size() : 0;
}

std::size_t CompileReport::region_count() const noexcept {
  return state_ != nullptr ? state_->region_count : 0;
}

std::size_t CompileReport::fallback_operation_count() const noexcept {
  return state_ != nullptr ? state_->fallback_operation_count : 0;
}

std::size_t CompileReport::constant_count() const noexcept {
  return state_ != nullptr ? state_->constant_count : 0;
}

std::size_t CompileReport::borrowed_constant_count() const noexcept {
  return state_ != nullptr ? state_->borrowed_constant_count : 0;
}

std::size_t CompileReport::packed_constant_count() const noexcept {
  return state_ != nullptr ? state_->packed_constant_count : 0;
}

std::size_t CompileReport::packed_constant_bytes() const noexcept {
  return state_ != nullptr ? state_->packed_constant_bytes : 0;
}

std::size_t CompileReport::invocation_owned_output_count() const noexcept {
  return state_ != nullptr ? state_->invocation_owned_output_count : 0;
}

std::size_t CompileReport::copy_bytes() const noexcept {
  return state_ != nullptr ? state_->copy_bytes : 0;
}

std::uint32_t CompileReport::synchronization_count() const noexcept {
  return state_ != nullptr ? state_->synchronization_count : 0;
}

bool CompileReport::mixed_providers() const noexcept {
  return state_ != nullptr && state_->mixed_providers;
}

Result<CompileStepReport>
CompileReport::step(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("compile report is invalid"));
  }
  if (index >= state_->steps.size()) {
    return std::unexpected(
        Status::OutOfRange("compile report step index is out of range"));
  }
  return state_->steps[index];
}

Executable::Executable(const Executable &other) noexcept
    : state_(other.state_) {
  detail::retain(state_);
}

Executable &Executable::operator=(const Executable &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

Executable::Executable(Executable &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Executable &Executable::operator=(Executable &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Executable::~Executable() { detail::release(state_); }

std::size_t Executable::input_count() const noexcept {
  return state_ != nullptr ? state_->input_ids.size() : 0;
}

std::size_t Executable::output_count() const noexcept {
  return state_ != nullptr ? state_->output_ids.size() : 0;
}

std::size_t Executable::node_count() const noexcept {
  return state_ != nullptr ? state_->logical_node_count : 0;
}

std::size_t Executable::step_count() const noexcept {
  return state_ != nullptr ? state_->steps.size() : 0;
}

std::size_t Executable::region_count() const noexcept {
  return state_ != nullptr ? state_->region_count : 0;
}

std::size_t Executable::workspace_bytes() const noexcept {
  return state_ != nullptr ? state_->workspace_bytes : 0;
}

std::size_t Executable::workspace_backing_count() const noexcept {
  return state_ != nullptr ? state_->workspace_backing_count : 0;
}

Result<ValueId> Executable::input_id(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (index >= state_->input_ids.size()) {
    return std::unexpected(
        Status::OutOfRange("executable input index is out of range"));
  }
  return state_->input_ids[index];
}

Result<ValueId> Executable::output_id(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (index >= state_->output_ids.size()) {
    return std::unexpected(
        Status::OutOfRange("executable output index is out of range"));
  }
  return state_->output_ids[index];
}

Result<OutputOwnership>
Executable::output_ownership(std::size_t index) const noexcept {
  auto value = output_id(index);
  if (!value)
    return std::unexpected(std::move(value).error());
  const std::uint8_t role = state_->binding_roles[value->ordinal()];
  if ((role & detail::kInvocationOwnedOutput) != 0)
    return OutputOwnership::kInvocation;
  if ((role & (detail::kSourceAliasedOutput | detail::kExternalInput |
               detail::kConstant)) != 0) {
    return OutputOwnership::kSourceAlias;
  }
  return OutputOwnership::kCaller;
}

Result<Invocation> Executable::create_invocation() const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }

  const std::array resources{mem::system_resource()};
  const std::span<const mem::MemoryResourceRef> selected_resources =
      state_->workspace_plan.resource_count() == 0
          ? std::span<const mem::MemoryResourceRef>{}
          : std::span<const mem::MemoryResourceRef>(resources);
  auto workspace = state_->workspace_plan.instantiate(selected_resources);
  if (!workspace)
    return std::unexpected(std::move(workspace).error());

  void *storage = state_->allocator.allocate(sizeof(detail::InvocationState),
                                             alignof(detail::InvocationState));
  if (storage == nullptr) {
    return std::unexpected(invocation_allocation_failure());
  }
  auto *invocation =
      ::new (storage) detail::InvocationState(state_, std::move(*workspace));
  const std::size_t value_count = state_->descriptions.size();
  if (!invocation->readable.try_resize(value_count) ||
      !invocation->writable.try_resize(value_count) ||
      !invocation->bound.try_resize(value_count) ||
      !invocation->invocations.try_reserve(state_->steps.size()) ||
      !invocation->submission_inputs.try_resize(state_->node_inputs.size()) ||
      !invocation->submission_outputs.try_resize(state_->node_outputs.size())) {
    detail::release(invocation);
    return std::unexpected(invocation_allocation_failure());
  }
  for (std::size_t ordinal = 0; ordinal < value_count; ++ordinal) {
    invocation->bound[ordinal] = 0;
    const std::uint32_t request = state_->workspace_requests[ordinal];
    if (request == detail::kNoWorkspaceRequest)
      continue;
    auto storage_view = invocation->workspace.view(request);
    if (!storage_view) {
      Status error = std::move(storage_view).error();
      detail::release(invocation);
      return std::unexpected(std::move(error));
    }
    auto tensor = compute::TensorView::create(state_->descriptions[ordinal],
                                              *storage_view);
    if (!tensor) {
      Status error = std::move(tensor).error();
      detail::release(invocation);
      return std::unexpected(std::move(error));
    }
    invocation->writable[ordinal] = *tensor;
    invocation->readable[ordinal] = tensor->as_const();
  }
  for (std::size_t index = 0; index < state_->constant_ids.size(); ++index) {
    const std::size_t ordinal = state_->constant_ids[index].ordinal();
    invocation->readable[ordinal] = state_->constant_bindings[index];
    invocation->bound[ordinal] = 1;
  }
  for (std::size_t step_index = 0; step_index < state_->steps.size();
       ++step_index) {
    mem::BufferView step_workspace;
    const std::uint32_t request = state_->step_workspace_requests[step_index];
    if (request != detail::kNoWorkspaceRequest) {
      auto view = invocation->workspace.view(request);
      if (!view) {
        Status error = std::move(view).error();
        detail::release(invocation);
        return std::unexpected(std::move(error));
      }
      step_workspace = *view;
    }
    detail::StepInvocation created_step;
    const detail::PlanStep &step = state_->steps[step_index];
    if (step.materializes) {
      if (step.kind == detail::PlanStepKind::kRegion) {
        auto created = step.region.create_invocation(step_workspace);
        if (!created) {
          Status error = std::move(created).error();
          detail::release(invocation);
          return std::unexpected(std::move(error));
        }
        created_step.region = std::move(*created);
      } else {
        auto created = step.operation.create_invocation(step_workspace);
        if (!created) {
          Status error = std::move(created).error();
          detail::release(invocation);
          return std::unexpected(std::move(error));
        }
        created_step.operation = std::move(*created);
      }
    }
    if (!invocation->invocations.try_push_back(std::move(created_step))) {
      detail::release(invocation);
      return std::unexpected(invocation_allocation_failure());
    }
  }
  return detail::InvocationAccess::adopt(invocation);
}

Invocation::Invocation(Invocation &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Invocation &Invocation::operator=(Invocation &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Invocation::~Invocation() { detail::release(state_); }

Status Invocation::bind(ValueId value,
                        compute::ConstTensorView tensor) noexcept {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("invocation is invalid");
  }
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "invocation bindings cannot change while work is in flight");
  }
  if (!tensor.valid())
    return Status::InvalidArgument("tensor binding is invalid");
  const detail::PlanState &plan = *state_->plan;
  if (!owns(plan, value)) {
    return Status::InvalidArgument("value does not belong to this executable");
  }
  const std::size_t ordinal = value.ordinal();
  if ((plan.binding_roles[ordinal] & detail::kConstant) != 0)
    return Status::InvalidArgument("graph constants are bound at compile time");
  if ((plan.binding_roles[ordinal] &
       (detail::kInvocationOwnedOutput | detail::kSourceAliasedOutput)) != 0) {
    return Status::FailedPrecondition(
        "graph output storage is fixed by its compiled ownership policy");
  }
  if ((plan.binding_roles[ordinal] & detail::kExternalInput) == 0) {
    if ((plan.binding_roles[ordinal] & detail::kExternalOutput) != 0) {
      return Status::InvalidArgument(
          "graph output requires mutable tensor storage");
    }
    return Status::InvalidArgument("internal graph values cannot be bound");
  }
  if (tensor.description() != plan.descriptions[ordinal]) {
    return Status::InvalidArgument(
        "input tensor does not match the executable description");
  }
  Status alias =
      validate_external_alias(*state_, ordinal, byte_range(tensor.storage()));
  if (!alias.ok())
    return alias;
  state_->readable[ordinal] = tensor;
  state_->bound[ordinal] = 1;
  return Status::Ok();
}

Status Invocation::bind(ValueId value, compute::TensorView tensor) noexcept {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("invocation is invalid");
  }
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "invocation bindings cannot change while work is in flight");
  }
  if (!tensor.valid())
    return Status::InvalidArgument("tensor binding is invalid");
  const detail::PlanState &plan = *state_->plan;
  if (!owns(plan, value)) {
    return Status::InvalidArgument("value does not belong to this executable");
  }
  const std::size_t ordinal = value.ordinal();
  if ((plan.binding_roles[ordinal] & detail::kConstant) != 0)
    return Status::InvalidArgument("graph constants are bound at compile time");
  if ((plan.binding_roles[ordinal] & detail::kExternalOutput) != 0) {
    if ((plan.binding_roles[ordinal] & (detail::kInvocationOwnedOutput |
                                        detail::kSourceAliasedOutput)) != 0) {
      return Status::FailedPrecondition(
          "graph output storage is fixed by its compiled ownership policy");
    }
    if (tensor.description() != plan.descriptions[ordinal]) {
      return Status::InvalidArgument(
          "output tensor does not match the executable description");
    }
    Status alias =
        validate_external_alias(*state_, ordinal, byte_range(tensor.storage()));
    if (!alias.ok())
      return alias;
    state_->writable[ordinal] = tensor;
    state_->readable[ordinal] = tensor.as_const();
    state_->bound[ordinal] = 1;
    return Status::Ok();
  }
  if ((plan.binding_roles[ordinal] & detail::kExternalInput) != 0) {
    return bind(value, tensor.as_const());
  }
  return Status::InvalidArgument("internal graph values cannot be bound");
}

bool Invocation::in_flight() const noexcept {
  return state_ != nullptr && state_->in_flight.load(std::memory_order_acquire);
}

Status Invocation::reset() noexcept {
  if (state_ == nullptr)
    return Status::FailedPrecondition("invocation is invalid");
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return Status::FailedPrecondition(
        "invocation cannot reset while work is in flight");
  }
  state_->outputs_ready.store(false, std::memory_order_release);
  for (ValueId value : state_->plan->input_ids) {
    const std::size_t ordinal = value.ordinal();
    state_->readable[ordinal] = {};
    state_->bound[ordinal] = 0;
  }
  for (ValueId value : state_->plan->output_ids) {
    const std::size_t ordinal = value.ordinal();
    if ((state_->plan->binding_roles[ordinal] & detail::kConstant) != 0)
      continue;
    state_->readable[ordinal] = {};
    state_->writable[ordinal] = {};
    state_->bound[ordinal] = 0;
  }
  return Status::Ok();
}

Result<compute::ConstTensorView>
Invocation::output(ValueId value) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  if (state_->in_flight.load(std::memory_order_acquire)) {
    return std::unexpected(Status::FailedPrecondition(
        "graph outputs are unavailable while work is in flight"));
  }
  if (!state_->outputs_ready.load(std::memory_order_acquire)) {
    return std::unexpected(Status::FailedPrecondition(
        "graph outputs require a successful completed submission"));
  }
  const detail::PlanState &plan = *state_->plan;
  if (!owns(plan, value)) {
    return std::unexpected(
        Status::InvalidArgument("value does not belong to this executable"));
  }
  const std::size_t ordinal = value.ordinal();
  if ((plan.binding_roles[ordinal] & detail::kExternalOutput) == 0) {
    return std::unexpected(
        Status::InvalidArgument("value is not a published graph output"));
  }
  if (!state_->readable[ordinal].valid()) {
    return std::unexpected(
        Status::Internal("published graph output has no readable storage"));
  }
  return state_->readable[ordinal];
}

Result<compute::Completion> Invocation::submit(compute::Queue &queue) noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  bool idle = false;
  if (!state_->in_flight.compare_exchange_strong(
          idle, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
    return std::unexpected(
        Status::FailedPrecondition("invocation already has work in flight"));
  }
  state_->outputs_ready.store(false, std::memory_order_release);
  struct RunRollback final {
    detail::InvocationState *state = nullptr;
    bool armed = true;
    ~RunRollback() {
      if (armed)
        state->in_flight.store(false, std::memory_order_release);
    }
    void disarm() noexcept { armed = false; }
  } rollback{state_};

  const detail::PlanState &plan = *state_->plan;
  for (ValueId value : plan.input_ids) {
    if (state_->bound[value.ordinal()] == 0) {
      return std::unexpected(Status::FailedPrecondition(
          "invocation input bindings are incomplete"));
    }
  }
  for (ValueId value : plan.output_ids) {
    const std::size_t ordinal = value.ordinal();
    if ((plan.binding_roles[ordinal] & (detail::kInvocationOwnedOutput |
                                        detail::kSourceAliasedOutput)) == 0 &&
        state_->bound[ordinal] == 0) {
      return std::unexpected(Status::FailedPrecondition(
          "invocation output bindings are incomplete"));
    }
  }

  // Resolve every non-materializing view and construct every step's binding
  // spans before accepting any Queue work.  A malformed alias or missing
  // internal binding therefore cannot strand an already-submitted prefix.
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const detail::PlanStep &step = plan.steps[step_index];
    if (!step.materializes) {
      for (std::size_t output_index = 0; output_index < step.output_count;
           ++output_index) {
        const std::uint32_t output =
            plan.node_outputs[step.output_begin + output_index];
        const std::uint32_t source = plan.alias_sources[output];
        if (source == detail::kNoInPlaceSource ||
            !state_->readable[source].valid()) {
          return std::unexpected(Status::Internal(
              "non-materializing graph output has no readable source"));
        }
        auto bytes = plan.descriptions[output].required_bytes();
        if (!bytes)
          return std::unexpected(std::move(bytes).error());
        auto storage = state_->readable[source].storage().subview(
            plan.alias_offsets[output], *bytes);
        if (!storage)
          return std::unexpected(std::move(storage).error());
        auto derived = compute::ConstTensorView::create(
            plan.descriptions[output], *storage);
        if (!derived)
          return std::unexpected(std::move(derived).error());
        if ((plan.binding_roles[output] & detail::kSourceAliasedOutput) != 0) {
          state_->readable[output] = *derived;
        } else if ((plan.binding_roles[output] & detail::kExternalOutput) !=
                   0) {
          if (!equal_range(byte_range(derived->storage()),
                           byte_range(state_->writable[output].storage()))) {
            return std::unexpected(Status::InvalidArgument(
                "published graph view does not bind its inferred storage"));
          }
        } else {
          state_->readable[output] = *derived;
        }
      }
      continue;
    }
    for (std::size_t index = 0; index < step.input_count; ++index) {
      const compute::ConstTensorView binding =
          state_->readable[plan.node_inputs[step.input_begin + index]];
      if (!binding.valid()) {
        return std::unexpected(Status::Internal(
            "materializing graph step has no readable input storage"));
      }
      state_->submission_inputs[step.input_begin + index] = binding;
    }
    for (std::size_t index = 0; index < step.output_count; ++index) {
      const compute::TensorView binding =
          state_->writable[plan.node_outputs[step.output_begin + index]];
      if (!binding.valid()) {
        return std::unexpected(Status::Internal(
            "materializing graph step has no writable output storage"));
      }
      state_->submission_outputs[step.output_begin + index] = binding;
    }
  }

  compute::Completion final_completion = compute::Completion::completed();
  if (!plan.steps.empty()) {
    auto sequence = compute::detail::QueueSequence::begin(queue);
    if (!sequence)
      return std::unexpected(std::move(sequence).error());

    bool accepted_prefix = false;
    const auto all_inputs = std::span<const compute::ConstTensorView>(
        state_->submission_inputs.data(), state_->submission_inputs.size());
    const auto all_outputs = std::span<const compute::TensorView>(
        state_->submission_outputs.data(), state_->submission_outputs.size());
    for (std::size_t step_index = 0; step_index < plan.steps.size();
         ++step_index) {
      const detail::PlanStep &step = plan.steps[step_index];
      if (!step.materializes)
        continue;
      const auto step_inputs =
          all_inputs.subspan(step.input_begin, step.input_count);
      const auto step_outputs =
          all_outputs.subspan(step.output_begin, step.output_count);
      Result<compute::Completion> completion =
          step.kind == detail::PlanStepKind::kRegion
              ? state_->invocations[step_index].region.submit(
                    *sequence, step_inputs, step_outputs)
              : compute::detail::OperationInvocationAccess::submit(
                    state_->invocations[step_index].operation, *sequence,
                    step_inputs, step_outputs);
      if (!completion) {
        Status rejected = std::move(completion).error();
        if (!accepted_prefix)
          return std::unexpected(std::move(rejected));
        compute::detail::CompletionAccess::append_terminal(final_completion,
                                                           std::move(rejected));
        break;
      }
      accepted_prefix = true;
      // A non-blocking observation catches synchronous CPU/provider failure
      // before a dependent command is submitted.  Pending work continues on
      // the ordered Queue; there is deliberately no host wait here.
      auto observed = completion->poll();
      final_completion = std::move(*completion);
      if (!observed)
        break;
    }
  }

  if (!compute::detail::CompletionAccess::has_async_state(final_completion)) {
    mark_invocation_terminal(state_, final_completion.wait());
    rollback.disarm();
    return final_completion;
  }

  detail::retain(state_);
  state_->completion_lifetime = compute::detail::CompletionLifetime{
      .state = state_,
      .terminal = &mark_invocation_terminal,
      .release = &release_invocation_lifetime,
  };
  Status attached = compute::detail::CompletionAccess::attach_lifetime(
      final_completion, state_->completion_lifetime);
  if (!attached.ok())
    return std::unexpected(std::move(attached));
  rollback.disarm();
  return final_completion;
}

Result<CompiledGraph> compile(const Graph &graph,
                              const compute::Context &context,
                              CompileOptions options) noexcept {
  if (!graph.valid()) {
    return std::unexpected(Status::InvalidArgument("graph is invalid"));
  }
  if (!context.valid() || !context.is_open()) {
    return std::unexpected(
        Status::FailedPrecondition("compute context is invalid or closed"));
  }
  if (options.region_selection != RegionSelectionPolicy::kDisabled &&
      options.region_selection != RegionSelectionPolicy::kConservative) {
    return std::unexpected(
        Status::InvalidArgument("graph region selection policy is invalid"));
  }
  if (options.operation_fallback != OperationFallbackPolicy::kAllow &&
      options.operation_fallback != OperationFallbackPolicy::kReject) {
    return std::unexpected(
        Status::InvalidArgument("graph operation fallback policy is invalid"));
  }
  if (context.device().device_class() != compute::DeviceClass::kCpu) {
    return std::unexpected(Status::Unimplemented(
        "graph v2 currently supports only the CPU backend"));
  }
  const detail::GraphState *logical = detail::GraphAccess::state(graph);
  if (logical == nullptr) {
    return std::unexpected(Status::InvalidArgument("graph is invalid"));
  }

  const mem::AllocatorRef allocator = context.metadata_allocator();
  if (!allocator.valid()) {
    return std::unexpected(Status::FailedPrecondition(
        "compute context has no metadata allocator"));
  }
  void *storage =
      allocator.allocate(sizeof(detail::PlanState), alignof(detail::PlanState));
  if (storage == nullptr)
    return std::unexpected(plan_allocation_failure());
  auto *plan =
      ::new (storage) detail::PlanState(allocator, logical->generation);

  Status status = initialize_plan_metadata(*plan, *logical);
  if (status.ok())
    status = bind_constants(*plan, *logical, options);
  if (status.ok())
    status = prepare_steps(*plan, graph, *logical, context, options);
  if (status.ok())
    status = plan_workspace(*plan, *logical);
  if (!status.ok()) {
    detail::release(plan);
    return std::unexpected(std::move(status));
  }
  auto report = build_report(*plan, context);
  if (!report) {
    detail::release(plan);
    return std::unexpected(std::move(report).error());
  }

  CompiledGraph result;
  result.report = std::move(*report);
  result.executable = detail::ExecutableAccess::adopt(plan);
  return result;
}

} // namespace hyle::graph
