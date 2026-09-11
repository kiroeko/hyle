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

bool fits_ordinal(std::size_t size) noexcept {
  return size <= (std::numeric_limits<std::uint32_t>::max)();
}

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

Result<ValueId> Graph::node_output(NodeId node) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_graph());
  if (!detail::owns(*state_, node)) {
    return std::unexpected(
        Status::InvalidArgument("node does not belong to this graph"));
  }
  return detail::value_id(*state_, state_->nodes[node.ordinal()].output);
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
    return std::unexpected(
        Status::FailedPrecondition("graph input has no producer node"));
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
  if (!fits_ordinal(state_->values.size())) {
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

Result<ValueId> GraphBuilder::add(compute::Operation operation,
                                  std::span<const ValueId> inputs) noexcept {
  if (state_ == nullptr)
    return std::unexpected(invalid_builder());
  if (inputs.empty() || inputs.size() > 2) {
    return std::unexpected(
        Status::InvalidArgument("graph operation input count is unsupported"));
  }
  if (!fits_ordinal(state_->nodes.size()) ||
      !fits_ordinal(state_->values.size()) ||
      state_->edges.size() >
          (std::numeric_limits<std::uint32_t>::max)() - inputs.size()) {
    return std::unexpected(
        Status::ResourceExhausted("graph identity space is exhausted"));
  }

  std::array<compute::TensorDesc, 2> descriptions{};
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    if (!detail::owns(*state_, inputs[index])) {
      return std::unexpected(Status::InvalidArgument(
          "operation input belongs to another builder"));
    }
    descriptions[index] = state_->values[inputs[index].ordinal()].description;
  }
  auto output = compute::infer_output_description(
      operation,
      std::span<const compute::TensorDesc>(descriptions.data(), inputs.size()));
  if (!output)
    return std::unexpected(std::move(output).error());

  if (!state_->edges.try_reserve(state_->edges.size() + inputs.size()) ||
      !state_->nodes.try_reserve(state_->nodes.size() + 1) ||
      !state_->values.try_reserve(state_->values.size() + 1)) {
    return std::unexpected(allocation_failure());
  }

  const std::uint32_t node_ordinal =
      static_cast<std::uint32_t>(state_->nodes.size());
  const std::uint32_t edge_begin =
      static_cast<std::uint32_t>(state_->edges.size());
  const std::uint32_t output_ordinal =
      static_cast<std::uint32_t>(state_->values.size());
  for (ValueId input_value : inputs) {
    if (!state_->edges.try_push_back(input_value.ordinal())) {
      return std::unexpected(allocation_failure());
    }
  }
  if (!state_->nodes.try_push_back(detail::NodeRecord{
          .operation = operation,
          .input_begin = edge_begin,
          .output = output_ordinal,
          .input_count = static_cast<std::uint8_t>(inputs.size()),
      }) ||
      !state_->values.try_push_back(detail::ValueRecord{
          .description = *output,
          .kind = ValueKind::kNodeOutput,
          .producer = node_ordinal,
      })) {
    return std::unexpected(allocation_failure());
  }
  return detail::value_id(*state_, output_ordinal);
}

Result<ValueId> GraphBuilder::add(compute::RmsNorm operation,
                                  ValueId input_value) noexcept {
  const std::array inputs{input_value};
  return add(compute::Operation(operation), inputs);
}

Result<ValueId> GraphBuilder::add(compute::RmsNorm operation,
                                  ValueId input_value,
                                  ValueId weight) noexcept {
  const std::array inputs{input_value, weight};
  return add(compute::Operation(operation), inputs);
}

Result<ValueId> GraphBuilder::add(compute::MatMul operation, ValueId left,
                                  ValueId right) noexcept {
  const std::array inputs{left, right};
  return add(compute::Operation(operation), inputs);
}

Result<ValueId> GraphBuilder::add(compute::SiluAndMul operation,
                                  ValueId input_value) noexcept {
  const std::array inputs{input_value};
  return add(compute::Operation(operation), inputs);
}

Status GraphBuilder::output(ValueId value) noexcept {
  if (state_ == nullptr)
    return invalid_builder();
  if (!detail::owns(*state_, value)) {
    return Status::InvalidArgument("graph output belongs to another builder");
  }
  if (value.kind() != ValueKind::kNodeOutput) {
    return Status::InvalidArgument(
        "a graph output must be produced by an operation");
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
  if (state_->nodes.empty() || state_->outputs.empty()) {
    return std::unexpected(Status::FailedPrecondition(
        "graph requires at least one operation and published output"));
  }

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
