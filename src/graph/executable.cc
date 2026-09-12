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
#include "plan_state.h"

namespace hyle::graph {
namespace {

Status plan_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph plan metadata allocation failed");
}

Status invocation_allocation_failure() noexcept {
  return Status::ResourceExhausted("graph invocation allocation failed");
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
  if (options.max_region_proposals >
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
    RegionProposalWriter writer(std::span<RegionProposal>(storage).subspan(
        used, storage.size() - used));
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
                             std::uint32_t node_index) noexcept {
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
  if (!plan.steps.try_emplace_back(
          std::move(*prepared), node_index, plan_input_begin, node.input_count,
          plan_output_begin, node.output_count, materializes)) {
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
    status = append_operation_step(plan, graph, context, node_index);
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
        if ((plan.binding_roles[output] & detail::kExternalOutput) != 0 &&
            !has_external_storage_root(plan, input)) {
          return Status::Unimplemented(
              "a published view of private workspace needs a runtime-owned "
              "output contract");
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
        (plan.binding_roles[ordinal] & detail::kExternalOutput) != 0) {
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
            .alignment = (std::max)(std::size_t{64},
                                    value.description.required_alignment()),
            .begin = begin,
            .end = end,
            .alias_of = alias_request,
            .alias_offset = plan.alias_offsets[ordinal],
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

} // namespace

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
    detail::destroy(invocation);
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
      detail::destroy(invocation);
      return std::unexpected(std::move(error));
    }
    auto tensor = compute::TensorView::create(state_->descriptions[ordinal],
                                              *storage_view);
    if (!tensor) {
      Status error = std::move(tensor).error();
      detail::destroy(invocation);
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
        detail::destroy(invocation);
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
          detail::destroy(invocation);
          return std::unexpected(std::move(error));
        }
        created_step.region = std::move(*created);
      } else {
        auto created = step.operation.create_invocation(step_workspace);
        if (!created) {
          Status error = std::move(created).error();
          detail::destroy(invocation);
          return std::unexpected(std::move(error));
        }
        created_step.operation = std::move(*created);
      }
    }
    if (!invocation->invocations.try_push_back(std::move(created_step))) {
      detail::destroy(invocation);
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
    detail::destroy(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Invocation::~Invocation() { detail::destroy(state_); }

Status Invocation::bind(ValueId value,
                        compute::ConstTensorView tensor) noexcept {
  if (state_ == nullptr) {
    return Status::FailedPrecondition("invocation is invalid");
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

void Invocation::reset() noexcept {
  if (state_ == nullptr)
    return;
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
}

Result<compute::Completion> Invocation::submit(compute::Queue &queue) noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  const detail::PlanState &plan = *state_->plan;
  for (ValueId value : plan.input_ids) {
    if (state_->bound[value.ordinal()] == 0) {
      return std::unexpected(Status::FailedPrecondition(
          "invocation input bindings are incomplete"));
    }
  }
  for (ValueId value : plan.output_ids) {
    if (state_->bound[value.ordinal()] == 0) {
      return std::unexpected(Status::FailedPrecondition(
          "invocation output bindings are incomplete"));
    }
  }

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
        if ((plan.binding_roles[output] & detail::kExternalOutput) != 0) {
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
      state_->submission_inputs[step.input_begin + index] =
          state_->readable[plan.node_inputs[step.input_begin + index]];
    }
    for (std::size_t index = 0; index < step.output_count; ++index) {
      state_->submission_outputs[step.output_begin + index] =
          state_->writable[plan.node_outputs[step.output_begin + index]];
    }
    const auto all_inputs = std::span<const compute::ConstTensorView>(
        state_->submission_inputs.data(), state_->submission_inputs.size());
    const auto all_outputs = std::span<const compute::TensorView>(
        state_->submission_outputs.data(), state_->submission_outputs.size());
    const auto step_inputs =
        all_inputs.subspan(step.input_begin, step.input_count);
    const auto step_outputs =
        all_outputs.subspan(step.output_begin, step.output_count);
    Result<compute::Completion> completion =
        step.kind == detail::PlanStepKind::kRegion
            ? state_->invocations[step_index].region.submit(queue, step_inputs,
                                                            step_outputs)
            : state_->invocations[step_index].operation.submit(
                  queue, step_inputs, step_outputs);
    if (!completion)
      return std::unexpected(std::move(completion).error());
    Status terminal = completion->wait();
    if (!terminal.ok()) {
      return compute::Completion::completed(std::move(terminal));
    }
  }
  return compute::Completion::completed();
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

  CompiledGraph result;
  compute::ProviderIdentity graph_provider{};
  bool mixed_providers = false;
  std::size_t packed_constant_count = 0;
  std::size_t packed_constant_bytes = 0;
  std::size_t copy_bytes = 0;
  std::uint32_t synchronization_count = 0;
  for (const detail::PlanStep &step : plan->steps) {
    const compute::ProviderIdentity node_provider = step.provider();
    if (step.kind == detail::PlanStepKind::kRegion) {
      const RegionArtifactProperties properties = step.region.properties();
      packed_constant_count = saturating_add(packed_constant_count,
                                             properties.packed_constant_count);
      packed_constant_bytes = saturating_add(packed_constant_bytes,
                                             properties.packed_constant_bytes);
      copy_bytes = saturating_add(copy_bytes, properties.copy_bytes);
      synchronization_count = saturating_add(synchronization_count,
                                             properties.synchronization_count);
    }
    if (mixed_providers) {
      continue;
    } else if (!graph_provider.valid()) {
      graph_provider = node_provider;
    } else if (graph_provider.id != node_provider.id) {
      graph_provider = {};
      mixed_providers = true;
    }
  }
  result.report = CompileReport{
      .backend = context.device().backend_id(),
      .provider = graph_provider,
      .workspace_bytes = plan->workspace_bytes,
      .workspace_backing_count = plan->workspace_backing_count,
      .node_count = plan->logical_node_count,
      .step_count = plan->steps.size(),
      .region_count = plan->region_count,
      .fallback_operation_count = plan->steps.size() - plan->region_count,
      .constant_count = plan->constant_ids.size(),
      .borrowed_constant_count = plan->constant_ids.size(),
      .packed_constant_count = packed_constant_count,
      .packed_constant_bytes = packed_constant_bytes,
      .copy_bytes = copy_bytes,
      .synchronization_count = synchronization_count,
      .mixed_providers = mixed_providers,
  };
  result.executable = detail::ExecutableAccess::adopt(plan);
  return result;
}

} // namespace hyle::graph
