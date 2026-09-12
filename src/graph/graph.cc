#include "hyle/graph/graph.h"

#include <array>
#include <atomic>
#include <limits>
#include <new>
#include <span>
#include <utility>

#include "graph_state.h"

namespace hyle::graph {
namespace {

std::uint64_t next_generation() noexcept {
  static std::atomic<std::uint64_t> next{1};
  std::uint64_t generation = next.fetch_add(1, std::memory_order_relaxed);
  // Zero is the invalid/moved-from sentinel. Wraparound would require more
  // builders than a process can physically create, but keep the invariant
  // explicit instead of ever publishing a forgeable invalid identity.
  while (generation == 0) {
    generation = next.fetch_add(1, std::memory_order_relaxed);
  }
  return generation;
}

Status invalid_graph() noexcept {
  return Status::FailedPrecondition("graph is invalid");
}

Status invalid_builder() noexcept {
  return Status::FailedPrecondition("graph builder is invalid");
}

Status allocation_failure() noexcept {
  return Status::ResourceExhausted("graph metadata allocation failed");
}

Status canonicalize_dependencies(detail::BuilderState &state) noexcept {
  using Counts = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  Counts counts(state.allocator);
  Counts cursors(state.allocator);
  Counts seen(state.allocator);
  if (!counts.try_resize(state.nodes.size()) ||
      !cursors.try_resize(state.nodes.size()) ||
      !seen.try_resize(state.nodes.size()) ||
      !state.dependencies.try_resize(state.control_edges.size())) {
    return allocation_failure();
  }
  for (std::size_t index = 0; index < state.nodes.size(); ++index) {
    counts[index] = 0;
    seen[index] = detail::kNoNode;
  }
  for (const detail::ControlEdgeRecord edge : state.control_edges) {
    if (counts[edge.node] == (std::numeric_limits<std::uint32_t>::max)()) {
      return Status::ResourceExhausted(
          "graph control dependency count is exhausted");
    }
    ++counts[edge.node];
  }

  std::uint32_t begin = 0;
  for (std::size_t index = 0; index < state.nodes.size(); ++index) {
    state.nodes[index].dependency_begin = begin;
    state.nodes[index].dependency_count = counts[index];
    cursors[index] = begin;
    begin += counts[index];
  }
  for (const detail::ControlEdgeRecord edge : state.control_edges)
    state.dependencies[cursors[edge.node]++] = edge.prerequisite;

  std::size_t write = 0;
  for (std::size_t node_index = 0; node_index < state.nodes.size();
       ++node_index) {
    detail::NodeRecord &node = state.nodes[node_index];
    const std::uint32_t old_begin = node.dependency_begin;
    const std::uint32_t old_count = node.dependency_count;
    node.dependency_begin = static_cast<std::uint32_t>(write);
    const std::uint32_t marker = static_cast<std::uint32_t>(node_index);
    for (std::size_t index = 0; index < old_count; ++index) {
      const std::uint32_t prerequisite = state.dependencies[old_begin + index];
      if (seen[prerequisite] == marker)
        continue;
      seen[prerequisite] = marker;
      state.dependencies[write++] = prerequisite;
    }
    node.dependency_count =
        static_cast<std::uint32_t>(write - node.dependency_begin);
  }
  state.dependencies.try_resize(write);
  return Status::Ok();
}

Status validate_effect_order(const detail::BuilderState &state) noexcept {
  using Effects = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
  Effects latest_reachable(state.allocator);
  if (!latest_reachable.try_resize(state.nodes.size()))
    return allocation_failure();
  for (std::size_t index = 0; index < latest_reachable.size(); ++index)
    latest_reachable[index] = detail::kNoNode;

  std::uint32_t last_effect = detail::kNoNode;
  for (std::size_t node_index = 0; node_index < state.nodes.size();
       ++node_index) {
    const detail::NodeRecord &node = state.nodes[node_index];
    std::uint32_t reachable = detail::kNoNode;
    auto absorb = [&](std::uint32_t prerequisite) noexcept {
      const bool prerequisite_is_effect =
          compute::operation_traits(state.nodes[prerequisite].operation)
              .effect == compute::EffectKind::kExternalSideEffect;
      const std::uint32_t candidate = prerequisite_is_effect
                                          ? prerequisite
                                          : latest_reachable[prerequisite];
      if (candidate != detail::kNoNode &&
          (reachable == detail::kNoNode || candidate > reachable)) {
        reachable = candidate;
      }
    };
    for (std::size_t index = 0; index < node.input_count; ++index) {
      const std::uint32_t value = state.edges[node.input_begin + index];
      const std::uint32_t producer = state.values[value].producer;
      if (producer != detail::kNoNode)
        absorb(producer);
    }
    for (std::size_t index = 0; index < node.dependency_count; ++index)
      absorb(state.dependencies[node.dependency_begin + index]);

    const bool is_effect = compute::operation_traits(node.operation).effect ==
                           compute::EffectKind::kExternalSideEffect;
    if (is_effect && last_effect != detail::kNoNode &&
        reachable != last_effect) {
      return Status::FailedPrecondition(
          "external side-effect nodes require an explicit dependency chain");
    }
    if (is_effect) {
      last_effect = static_cast<std::uint32_t>(node_index);
      reachable = last_effect;
    }
    latest_reachable[node_index] = reachable;
  }
  return Status::Ok();
}

constexpr bool fits_value_ordinal(std::size_t size) noexcept {
  return size <= (std::numeric_limits<std::uint32_t>::max)();
}

constexpr bool fits_node_ordinal(std::size_t size) noexcept {
  // UINT32_MAX is reserved as ValueRecord::producer's "no producer" value.
  return size < (std::numeric_limits<std::uint32_t>::max)();
}

static_assert(fits_value_ordinal(
    static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)())));
static_assert(!fits_node_ordinal(
    static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)())));

} // namespace

Graph::Graph(const Graph &other) noexcept : state_(other.state_) {
  detail::retain(state_);
}

Graph &Graph::operator=(const Graph &other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

Graph::Graph(Graph &&other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Graph &Graph::operator=(Graph &&other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Graph::~Graph() { detail::release(state_); }

std::size_t Graph::value_count() const noexcept {
  return state_ != nullptr ? state_->values.size() : 0;
}

std::size_t Graph::input_count() const noexcept {
  return state_ != nullptr ? state_->inputs.size() : 0;
}

std::size_t Graph::constant_count() const noexcept {
  return state_ != nullptr ? state_->constants.size() : 0;
}

std::size_t Graph::node_count() const noexcept {
  return state_ != nullptr ? state_->nodes.size() : 0;
}

std::size_t Graph::output_count() const noexcept {
  return state_ != nullptr ? state_->outputs.size() : 0;
}

Result<ValueId> Graph::input_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->inputs.size()) {
    return std::unexpected(
        Status::OutOfRange("graph input index is out of range"));
  }
  return detail::value_id(*state_, state_->inputs[index]);
}

Result<const compute::TensorDesc &>
Graph::input_description(std::size_t index) const noexcept {
  auto id = input_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  return state_->values[id->ordinal()].description;
}

Result<ValueId> Graph::constant_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->constants.size()) {
    return std::unexpected(
        Status::OutOfRange("graph constant index is out of range"));
  }
  return detail::value_id(*state_, state_->constants[index]);
}

Result<const compute::TensorDesc &>
Graph::constant_description(std::size_t index) const noexcept {
  auto id = constant_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  return state_->values[id->ordinal()].description;
}

Result<NodeId> Graph::node_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->nodes.size()) {
    return std::unexpected(
        Status::OutOfRange("graph node index is out of range"));
  }
  return detail::GraphIdentity::node(state_->generation,
                                     static_cast<std::uint32_t>(index));
}

Result<const compute::Operation &>
Graph::operation(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return state_->nodes[node.ordinal()].operation;
}

Result<std::size_t> Graph::node_input_count(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return state_->nodes[node.ordinal()].input_count;
}

Result<ValueId> Graph::node_input(NodeId node,
                                  std::size_t input_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (input_index >= record.input_count) {
    return std::unexpected(
        Status::OutOfRange("graph node input index is out of range"));
  }
  return detail::value_id(*state_,
                          state_->edges[record.input_begin + input_index]);
}

Result<std::size_t> Graph::node_dependency_count(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return state_->nodes[node.ordinal()].dependency_count;
}

Result<NodeId>
Graph::node_dependency(NodeId node,
                       std::size_t dependency_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (dependency_index >= record.dependency_count) {
    return std::unexpected(
        Status::OutOfRange("graph node dependency index is out of range"));
  }
  return detail::GraphIdentity::node(
      state_->generation,
      state_->dependencies[record.dependency_begin + dependency_index]);
}

Result<std::size_t> Graph::node_output_count(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return state_->nodes[node.ordinal()].output_count;
}

Result<ValueId> Graph::result(NodeId node,
                              std::size_t output_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (output_index >= record.output_count) {
    return std::unexpected(
        Status::OutOfRange("graph node output index is out of range"));
  }
  return detail::value_id(
      *state_, record.output_begin + static_cast<std::uint32_t>(output_index));
}

Result<NodeId> Graph::producer(ValueId value) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, value)) {
    return std::unexpected(
        Status::InvalidArgument("value does not belong to this graph"));
  }
  const std::uint32_t producer = state_->values[value.ordinal()].producer;
  if (producer == detail::kNoNode) {
    return std::unexpected(Status::FailedPrecondition(
        "external graph value has no producer node"));
  }
  return detail::GraphIdentity::node(state_->generation, producer);
}

Result<ValueId> Graph::output_id(std::size_t index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (index >= state_->outputs.size()) {
    return std::unexpected(
        Status::OutOfRange("graph output index is out of range"));
  }
  return detail::value_id(*state_, state_->outputs[index]);
}

Result<const compute::TensorDesc &>
Graph::output_description(std::size_t index) const noexcept {
  auto id = output_id(index);
  if (!id)
    return std::unexpected(std::move(id).error());
  return state_->values[id->ordinal()].description;
}

Result<const compute::TensorDesc &>
Graph::value_description(ValueId value) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, value)) {
    return std::unexpected(
        Status::InvalidArgument("value does not belong to this graph"));
  }
  return state_->values[value.ordinal()].description;
}

GraphBuilder::GraphBuilder(GraphBuilder &&other) noexcept
    : state_(other.state_) {
  other.state_ = nullptr;
}

GraphBuilder &GraphBuilder::operator=(GraphBuilder &&other) noexcept {
  if (this != &other) {
    detail::destroy(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

GraphBuilder::~GraphBuilder() { detail::destroy(state_); }

Result<GraphBuilder>
GraphBuilder::create(GraphBuilderOptions options) noexcept {
  mem::AllocatorRef allocator = options.metadata_allocator;
  if (!allocator.valid())
    allocator = mem::system_allocator();
  void *storage = allocator.allocate(sizeof(detail::BuilderState),
                                     alignof(detail::BuilderState));
  if (storage == nullptr)
    return std::unexpected(allocation_failure());
  auto *state =
      ::new (storage) detail::BuilderState(next_generation(), allocator);
  return GraphBuilder(state);
}

Result<ValueId> GraphBuilder::input(compute::TensorDesc description) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (!description.valid()) {
    return std::unexpected(
        Status::InvalidArgument("graph input description is invalid"));
  }
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::unexpected(std::move(bytes).error());
  if (!fits_value_ordinal(state_->values.size())) {
    return std::unexpected(
        Status::ResourceExhausted("graph value identity space is exhausted"));
  }
  if (!state_->values.try_reserve(state_->values.size() + 1) ||
      !state_->inputs.try_reserve(state_->inputs.size() + 1)) {
    return std::unexpected(allocation_failure());
  }

  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(state_->values.size());
  const bool value_added = state_->values.try_push_back(detail::ValueRecord{
      .description = description,
      .kind = ValueKind::kInput,
      .producer = detail::kNoNode,
  });
  const bool input_added = state_->inputs.try_push_back(ordinal);
  if (!value_added || !input_added) {
    return std::unexpected(allocation_failure());
  }
  return detail::value_id(*state_, ordinal);
}

Result<ValueId>
GraphBuilder::constant(compute::TensorDesc description) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (!description.valid()) {
    return std::unexpected(
        Status::InvalidArgument("graph constant description is invalid"));
  }
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::unexpected(std::move(bytes).error());
  if (!fits_value_ordinal(state_->values.size())) {
    return std::unexpected(
        Status::ResourceExhausted("graph value identity space is exhausted"));
  }
  if (!state_->values.try_reserve(state_->values.size() + 1) ||
      !state_->constants.try_reserve(state_->constants.size() + 1)) {
    return std::unexpected(allocation_failure());
  }

  const std::uint32_t ordinal =
      static_cast<std::uint32_t>(state_->values.size());
  const bool value_added = state_->values.try_push_back(detail::ValueRecord{
      .description = description,
      .kind = ValueKind::kConstant,
      .producer = detail::kNoNode,
  });
  const bool constant_added = state_->constants.try_push_back(ordinal);
  if (!value_added || !constant_added)
    return std::unexpected(allocation_failure());
  return detail::value_id(*state_, ordinal);
}

Result<NodeId>
GraphBuilder::add_node(compute::Operation operation,
                       std::span<const ValueId> inputs) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  const compute::OperationTraits traits = compute::operation_traits(operation);
  if (!operation.valid() || !traits.inputs.accepts(inputs.size())) {
    return std::unexpected(
        Status::InvalidArgument("graph operation input count is invalid"));
  }
  if (inputs.size() > (std::numeric_limits<std::uint16_t>::max)())
    return std::unexpected(Status::ResourceExhausted(
        "graph operation input identity space is exhausted"));
  const std::size_t output_count = traits.outputs;
  if (!fits_node_ordinal(state_->nodes.size()) ||
      (output_count != 0 &&
       state_->values.size() >
           (std::numeric_limits<std::uint32_t>::max)() - (output_count - 1)) ||
      state_->edges.size() >
          (std::numeric_limits<std::uint32_t>::max)() - inputs.size()) {
    return std::unexpected(
        Status::ResourceExhausted("graph identity space is exhausted"));
  }

  using Descriptions =
      container::SmallVector<compute::TensorDesc, 4, mem::AllocatorRef>;
  Descriptions descriptions(state_->allocator);
  Descriptions outputs(state_->allocator);
  if (!descriptions.try_resize(inputs.size()) ||
      !outputs.try_resize(output_count)) {
    return std::unexpected(allocation_failure());
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!detail::owns(*state_, inputs[index])) {
      return std::unexpected(Status::InvalidArgument(
          "operation input belongs to another builder"));
    }
    descriptions[index] = state_->values[inputs[index].ordinal()].description;
  }
  Status inferred =
      compute::infer_output_descriptions(operation, descriptions, outputs);
  if (!inferred.ok())
    return std::unexpected(std::move(inferred));

  if (!state_->edges.try_reserve(state_->edges.size() + inputs.size()) ||
      !state_->nodes.try_reserve(state_->nodes.size() + 1) ||
      !state_->values.try_reserve(state_->values.size() + output_count)) {
    return std::unexpected(allocation_failure());
  }

  const std::uint32_t node_ordinal =
      static_cast<std::uint32_t>(state_->nodes.size());
  const std::uint32_t edge_begin =
      static_cast<std::uint32_t>(state_->edges.size());
  const std::uint32_t output_begin =
      static_cast<std::uint32_t>(state_->values.size());
  for (ValueId input_value : inputs) {
    if (!state_->edges.try_push_back(input_value.ordinal())) {
      return std::unexpected(allocation_failure());
    }
  }
  if (!state_->nodes.try_push_back(detail::NodeRecord{
          .operation = operation,
          .input_begin = edge_begin,
          .output_begin = output_begin,
          .input_count = static_cast<std::uint16_t>(inputs.size()),
          .output_count = static_cast<std::uint8_t>(output_count),
      })) {
    return std::unexpected(allocation_failure());
  }
  for (const compute::TensorDesc &output : outputs) {
    if (!state_->values.try_push_back(detail::ValueRecord{
            .description = output,
            .kind = ValueKind::kNodeOutput,
            .producer = node_ordinal,
        })) {
      return std::unexpected(allocation_failure());
    }
  }
  return detail::GraphIdentity::node(state_->generation, node_ordinal);
}

Result<ValueId> GraphBuilder::result(NodeId node,
                                     std::size_t output_index) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this builder"));
  }
  const detail::NodeRecord &record = state_->nodes[node.ordinal()];
  if (output_index >= record.output_count) {
    return std::unexpected(
        Status::OutOfRange("graph node output index is out of range"));
  }
  return detail::value_id(
      *state_, record.output_begin + static_cast<std::uint32_t>(output_index));
}

Status GraphBuilder::depends_on(NodeId node, NodeId prerequisite) noexcept {
  if (state_ == nullptr)
    return invalid_builder();
  if (!detail::owns(*state_, node) || !detail::owns(*state_, prerequisite)) {
    return Status::InvalidArgument(
        "control dependency node belongs to another builder");
  }
  if (prerequisite.ordinal() >= node.ordinal()) {
    return Status::InvalidArgument(
        "control dependency must point to an earlier node");
  }
  if (state_->control_edges.size() >=
      (std::numeric_limits<std::uint32_t>::max)()) {
    return Status::ResourceExhausted(
        "graph control dependency identity space is exhausted");
  }
  if (!state_->control_edges.try_push_back(detail::ControlEdgeRecord{
          .node = node.ordinal(), .prerequisite = prerequisite.ordinal()})) {
    return allocation_failure();
  }
  return Status::Ok();
}

Status GraphBuilder::publish(ValueId value) noexcept {
  if (state_ == nullptr)
    return invalid_builder();
  if (!detail::owns(*state_, value)) {
    return Status::InvalidArgument("graph output belongs to another builder");
  }
  for (std::uint32_t published : state_->outputs) {
    if (published == value.ordinal()) {
      return Status::AlreadyExists("graph output is already published");
    }
  }
  if (!state_->outputs.try_push_back(value.ordinal())) {
    return allocation_failure();
  }
  return Status::Ok();
}

Result<Graph> GraphBuilder::finish() && noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (state_->nodes.empty() && state_->outputs.empty()) {
    return std::unexpected(
        Status::FailedPrecondition("graph requires a node or published value"));
  }
  Status dependencies = canonicalize_dependencies(*state_);
  if (!dependencies.ok())
    return std::unexpected(std::move(dependencies));
  Status effects = validate_effect_order(*state_);
  if (!effects.ok())
    return std::unexpected(std::move(effects));

  const mem::AllocatorRef allocator = state_->allocator;
  void *storage = allocator.allocate(sizeof(detail::GraphState),
                                     alignof(detail::GraphState));
  if (storage == nullptr)
    return std::unexpected(allocation_failure());
  auto *published = ::new (storage) detail::GraphState(*state_);
  detail::destroy(state_);
  state_ = nullptr;
  return detail::GraphAccess::adopt(published);
}

} // namespace hyle::graph
