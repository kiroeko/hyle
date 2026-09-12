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
  const ValueKind expected = (role & detail::kExternalInput) != 0
                                 ? ValueKind::kInput
                                 : ValueKind::kNodeOutput;
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

bool allows_exact_in_place(const detail::PlanState &plan, std::size_t left,
                           std::size_t right) noexcept {
  return (plan.in_place_sources[left] == right ||
          plan.in_place_sources[right] == left);
}

ByteRange bound_range(const detail::InvocationState &invocation,
                      std::size_t ordinal) noexcept {
  const std::uint8_t role = invocation.plan->binding_roles[ordinal];
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
    if (equal_range(candidate, existing) &&
        allows_exact_in_place(plan, ordinal, other)) {
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
      !plan.output_ids.try_reserve(graph.outputs.size()) ||
      !plan.descriptions.try_resize(graph.values.size()) ||
      !plan.workspace_requests.try_resize(graph.values.size()) ||
      !plan.in_place_sources.try_resize(graph.values.size()) ||
      !plan.binding_roles.try_resize(graph.values.size())) {
    return plan_allocation_failure();
  }

  for (std::size_t index = 0; index < graph.values.size(); ++index) {
    plan.descriptions[index] = graph.values[index].description;
    plan.workspace_requests[index] = detail::kNoWorkspaceRequest;
    plan.in_place_sources[index] = detail::kNoInPlaceSource;
    plan.binding_roles[index] = 0;
  }
  for (std::uint32_t ordinal : graph.inputs) {
    if (!plan.input_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kExternalInput;
  }
  for (std::uint32_t ordinal : graph.outputs) {
    if (!plan.output_ids.try_push_back(detail::value_id(graph, ordinal))) {
      return plan_allocation_failure();
    }
    plan.binding_roles[ordinal] |= detail::kExternalOutput;
  }
  return Status::Ok();
}

Status prepare_nodes(detail::PlanState &plan, const detail::GraphState &graph,
                     const compute::Context &context) noexcept {
  for (const detail::NodeRecord &node : graph.nodes) {
    std::array<compute::TensorDesc, 2> input_descriptions{};
    std::array<std::uint32_t, 2> input_ordinals{};
    for (std::size_t index = 0; index < node.input_count; ++index) {
      const std::uint32_t ordinal = graph.edges[node.input_begin + index];
      input_ordinals[index] = ordinal;
      input_descriptions[index] = graph.values[ordinal].description;
    }
    const std::array output_descriptions{graph.values[node.output].description};
    auto prepared =
        compute::prepare(context, node.operation,
                         std::span<const compute::TensorDesc>(
                             input_descriptions.data(), node.input_count),
                         output_descriptions);
    if (!prepared)
      return std::move(prepared).error();
    if (!plan.nodes.try_emplace_back(
            std::move(*prepared),
            std::span<const std::uint32_t>(input_ordinals.data(),
                                           node.input_count),
            node.output)) {
      return plan_allocation_failure();
    }
  }
  return Status::Ok();
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
    last_uses[node.output] = (std::max)(last_uses[node.output], terminal);
    for (std::size_t input_index = 0; input_index < node.input_count;
         ++input_index) {
      const std::uint32_t input = graph.edges[node.input_begin + input_index];
      last_uses[input] = (std::max)(last_uses[input], terminal);
    }
  }

  // The current catalog has one legal alias: an externally bound RMSNorm
  // output may exactly reuse its first external input after that input's final
  // read. Partial overlap and all other cross-value aliases remain invalid.
  for (std::size_t node_index = 0; node_index < graph.nodes.size();
       ++node_index) {
    const detail::NodeRecord &node = graph.nodes[node_index];
    if (node.operation.kind() != compute::OperationKind::kRmsNorm ||
        (plan.binding_roles[node.output] & detail::kExternalOutput) == 0) {
      continue;
    }
    const std::uint32_t input = graph.edges[node.input_begin];
    if ((plan.binding_roles[input] & detail::kExternalInput) != 0 &&
        last_uses[input] == node_index + 1) {
      plan.in_place_sources[node.output] = input;
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
    if (!requests.try_push_back(mem::LifetimeRequest{
            .resource_slot = 0,
            .size = *size,
            .alignment = (std::max)(std::size_t{64},
                                    value.description.required_alignment()),
            .begin = begin,
            .end = end,
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
      !invocation->operations.try_reserve(state_->nodes.size())) {
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
    auto operation = state_->nodes[node_index].prepared.create_invocation(
        operation_workspace);
    if (!operation) {
      Status error = std::move(operation).error();
      detail::destroy(invocation);
      return std::unexpected(std::move(error));
    }
    if (!invocation->operations.try_push_back(std::move(*operation))) {
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
    std::array<compute::ConstTensorView, 2> inputs{};
    for (std::size_t index = 0; index < node.input_count; ++index) {
      inputs[index] = state_->readable[node.inputs[index]];
    }
    const compute::TensorView output = state_->writable[node.output];
    auto completion = state_->operations[node_index].submit(
        queue,
        std::span<const compute::ConstTensorView>(inputs.data(),
                                                  node.input_count),
        std::span<const compute::TensorView>(&output, 1));
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
                              const compute::Context &context) noexcept {
  if (!graph.valid()) {
    return std::unexpected(Status::InvalidArgument("graph is invalid"));
  }
  if (!context.valid() || !context.is_open()) {
    return std::unexpected(
        Status::FailedPrecondition("compute context is invalid or closed"));
  }
  if (context.device().backend() != compute::BackendKind::kCpu) {
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
    status = prepare_nodes(*plan, *logical, context);
  if (status.ok())
    status = plan_workspace(*plan, *logical);
  if (!status.ok()) {
    detail::release(plan);
    return std::unexpected(std::move(status));
  }

  CompiledGraph result;
  compute::ProviderKind graph_provider = compute::ProviderKind::kUnknown;
  bool used_fallback = false;
  for (const detail::PlanNode &node : plan->nodes) {
    const compute::ProviderKind node_provider = node.prepared.provider();
    if (graph_provider == compute::ProviderKind::kUnknown) {
      graph_provider = node_provider;
    } else if (graph_provider != node_provider) {
      graph_provider = compute::ProviderKind::kMixed;
    }
    used_fallback = used_fallback || node.prepared.used_fallback();
  }
  result.report = CompileReport{
      .backend = context.device().backend(),
      .provider = graph_provider,
      .workspace_bytes = plan->workspace_bytes,
      .workspace_backing_count = plan->workspace_backing_count,
      .node_count = plan->nodes.size(),
      .copy_bytes = 0,
      .synchronization_count = 0,
      .used_fallback = used_fallback,
  };
  result.executable = detail::ExecutableAccess::adopt(plan);
  return result;
}

} // namespace hyle::graph
