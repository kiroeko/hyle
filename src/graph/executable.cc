#include "hyle/graph/executable.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
  if (!plan.nodes.try_reserve(graph.nodes.size()) ||
      !plan.input_ids.try_reserve(graph.inputs.size()) ||
      !plan.constant_ids.try_reserve(graph.constants.size()) ||
      !plan.constant_bindings.try_resize(graph.constants.size()) ||
      !plan.output_ids.try_reserve(graph.outputs.size()) ||
      !plan.node_inputs.try_reserve(graph.edges.size()) ||
      !plan.node_outputs.try_reserve(graph.values.size()) ||
      !plan.descriptions.try_resize(graph.values.size()) ||
      !plan.workspace_requests.try_resize(graph.values.size()) ||
      !plan.alias_sources.try_resize(graph.values.size()) ||
      !plan.alias_offsets.try_resize(graph.values.size()) ||
      !plan.binding_roles.try_resize(graph.values.size())) {
    return plan_allocation_failure();
  }

  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    plan.descriptions[index] = graph.values[index].description;
    plan.workspace_requests[index] = detail::kNoWorkspaceRequest;
    plan.alias_sources[index] = detail::kNoInPlaceSource;
    plan.alias_offsets[index] = 0;
    plan.binding_roles[index] = 0;
  }
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

Status prepare_nodes(detail::PlanState &plan, const detail::GraphState &graph,
                     const compute::Context &context) noexcept {
  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions input_descriptions(plan.allocator);
  Descriptions output_descriptions(plan.allocator);
  for (const detail::NodeRecord &node : graph.nodes) {
    input_descriptions.clear();
    output_descriptions.clear();
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
    }
    auto prepared = compute::prepare(context, node.operation,
                                     input_descriptions, output_descriptions);
    if (!prepared)
      return std::move(prepared).error();
    const bool materializes =
        compute::operation_traits(node.operation).materializes;
    if (!plan.nodes.try_emplace_back(std::move(*prepared), plan_input_begin,
                                     node.input_count, plan_output_begin,
                                     node.output_count, materializes)) {
      return plan_allocation_failure();
    }
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
      !plan.operation_workspace_requests.try_resize(plan.nodes.size()) ||
      !requests.try_reserve(graph.values.size() + plan.nodes.size())) {
    return plan_allocation_failure();
  }
  for (std::size_t index = 0; index < last_uses.size(); ++index) {
    last_uses[index] = 0;
  }
  for (std::size_t index = 0; index < plan.operation_workspace_requests.size();
       ++index) {
    plan.operation_workspace_requests[index] = detail::kNoWorkspaceRequest;
  }

  for (std::size_t node_index = 0; node_index < graph.nodes.size();
       ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    const std::uint64_t terminal = node_index + 1;
    for (std::size_t output_index = 0; output_index < node.output_count;
         ++output_index) {
      const std::uint32_t output =
          node.output_begin + static_cast<std::uint32_t>(output_index);
      last_uses[output] = (std::max)(last_uses[output], terminal);
    }
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t input = graph.edges[node.input_begin + input_index];
      last_uses[input] = (std::max)(last_uses[input], terminal);
    }
  }

  // The schema supplies only a broad permission. A prepared provider narrows
  // it for the selected route and layout. Required view aliases preserve
  // storage without consuming the source; optional in-place aliases require
  // equal descriptions and the source's final read.
  for (std::size_t node_index = 0; node_index < graph.nodes.size();
       ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
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
        auto support =
            plan.nodes[node_index].prepared.alias_support(output_index, 0);
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
        auto support = plan.nodes[node_index].prepared.alias_support(
            output_index, input_index);
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
            last_uses[input] == node_index + 1 &&
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
    const std::uint64_t begin = value.producer;
    const std::uint64_t end = (std::max)(begin + 1, last_uses[ordinal]);
    const std::uint32_t alias_source = plan.alias_sources[ordinal];
    const std::uint32_t alias_request =
        alias_source == detail::kNoInPlaceSource
            ? mem::kNoLifetimeAlias
            : plan.workspace_requests[alias_source];
    if (alias_source != detail::kNoInPlaceSource &&
        alias_request == detail::kNoWorkspaceRequest) {
      const bool materializes = plan.nodes[value.producer].materializes;
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

  // Provider scratchpads are per Invocation, not shared prepared state. Node
  // execution is ordered, so disjoint [node,node+1) requests may reuse one
  // backing while every OperationInvocation retains its own stable view.
  for (std::size_t node_index = 0; node_index < plan.nodes.size();
       ++node_index) {
    if (!plan.nodes[node_index].materializes)
      continue;
    const compute::WorkspaceRequirement requirement =
        plan.nodes[node_index].prepared.workspace_requirement();
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
            .begin = node_index,
            .end = node_index + 1,
        })) {
      return plan_allocation_failure();
    }
    plan.operation_workspace_requests[node_index] = request;
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
  return state_ != nullptr ? state_->nodes.size() : 0;
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
      !invocation->operations.try_reserve(state_->nodes.size()) ||
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
  for (std::size_t node_index = 0; node_index < state_->nodes.size();
       ++node_index) {
    mem::BufferView operation_workspace;
    const std::uint32_t request =
        state_->operation_workspace_requests[node_index];
    if (request != detail::kNoWorkspaceRequest) {
      auto view = invocation->workspace.view(request);
      if (!view) {
        Status error = std::move(view).error();
        detail::destroy(invocation);
        return std::unexpected(std::move(error));
      }
      operation_workspace = *view;
    }
    compute::OperationInvocation operation;
    if (state_->nodes[node_index].materializes) {
      auto created = state_->nodes[node_index].prepared.create_invocation(
          operation_workspace);
      if (!created) {
        Status error = std::move(created).error();
        detail::destroy(invocation);
        return std::unexpected(std::move(error));
      }
      operation = std::move(*created);
    }
    if (!invocation->operations.try_push_back(std::move(operation))) {
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

  for (std::size_t node_index = 0; node_index < plan.nodes.size();
       ++node_index) {
    const detail::PlanNode &node = plan.nodes[node_index];
    if (!node.materializes) {
      for (std::size_t output_index = 0; output_index < node.output_count;
           ++output_index) {
        const std::uint32_t output =
            plan.node_outputs[node.output_begin + output_index];
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
    for (std::size_t index = 0; index < node.input_count; ++index) {
      state_->submission_inputs[node.input_begin + index] =
          state_->readable[plan.node_inputs[node.input_begin + index]];
    }
    for (std::size_t index = 0; index < node.output_count; ++index) {
      state_->submission_outputs[node.output_begin + index] =
          state_->writable[plan.node_outputs[node.output_begin + index]];
    }
    const auto all_inputs = std::span<const compute::ConstTensorView>(
        state_->submission_inputs.data(), state_->submission_inputs.size());
    const auto all_outputs = std::span<const compute::TensorView>(
        state_->submission_outputs.data(), state_->submission_outputs.size());
    auto completion = state_->operations[node_index].submit(
        queue, all_inputs.subspan(node.input_begin, node.input_count),
        all_outputs.subspan(node.output_begin, node.output_count));
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
    status = prepare_nodes(*plan, *logical, context);
  if (status.ok())
    status = plan_workspace(*plan, *logical);
  if (!status.ok()) {
    detail::release(plan);
    return std::unexpected(std::move(status));
  }

  CompiledGraph result;
  compute::ProviderIdentity graph_provider{};
  bool mixed_providers = false;
  for (const detail::PlanNode &node : plan->nodes) {
    const compute::ProviderIdentity node_provider = node.prepared.provider();
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
      .node_count = plan->nodes.size(),
      .constant_count = plan->constant_ids.size(),
      .borrowed_constant_count = plan->constant_ids.size(),
      .packed_constant_count = 0,
      .packed_constant_bytes = 0,
      .copy_bytes = 0,
      .synchronization_count = 0,
      .mixed_providers = mixed_providers,
  };
  result.executable = detail::ExecutableAccess::adopt(plan);
  return result;
}

} // namespace hyle::graph
