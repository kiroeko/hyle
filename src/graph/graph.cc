#include "hyle/graph/graph.h"

#include <array>
#include <atomic>
#include <cmath>
#include <utility>

namespace hyle::graph {
namespace {

std::uint64_t next_generation() noexcept {
  static std::atomic<std::uint64_t> next{1};
  const std::uint64_t generation =
      next.fetch_add(1, std::memory_order_relaxed);
  // Exhausting a 64-bit process-local identity space is not a recoverable
  // runtime condition. Zero remains the invalid/moved-from sentinel.
  return generation == 0
             ? next.fetch_add(1, std::memory_order_relaxed)
             : generation;
}

}  // namespace

Graph::Graph(std::uint64_t generation, compute::Operation operation,
             std::span<const compute::TensorDesc> inputs) noexcept
    : generation_(generation), operation_(operation), output_(inputs.front()),
      input_count_(static_cast<std::uint8_t>(inputs.size())) {
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    inputs_[index] = inputs[index];
  }
}

Result<ValueId> Graph::input_id(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph is invalid"));
  }
  if (index >= input_count_) {
    return std::unexpected(Status::OutOfRange("graph input index is out of range"));
  }
  return ValueId(generation_, ValueKind::kInput,
                 static_cast<std::uint32_t>(index));
}

Result<const compute::TensorDesc&> Graph::input_description(
    std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph is invalid"));
  }
  if (index >= input_count_) {
    return std::unexpected(Status::OutOfRange("graph input index is out of range"));
  }
  return inputs_[index];
}

Result<NodeId> Graph::node_id(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph is invalid"));
  }
  if (index != 0) {
    return std::unexpected(Status::OutOfRange("graph node index is out of range"));
  }
  return NodeId(generation_, 0);
}

Result<const compute::Operation&> Graph::operation(
    std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph is invalid"));
  }
  if (index != 0) {
    return std::unexpected(Status::OutOfRange("graph node index is out of range"));
  }
  return operation_;
}

Result<ValueId> Graph::output_id(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph is invalid"));
  }
  if (index != 0) {
    return std::unexpected(Status::OutOfRange("graph output index is out of range"));
  }
  return ValueId(generation_, ValueKind::kOutput, 0);
}

Result<const compute::TensorDesc&> Graph::output_description(
    std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph is invalid"));
  }
  if (index != 0) {
    return std::unexpected(Status::OutOfRange("graph output index is out of range"));
  }
  return output_;
}

GraphBuilder::GraphBuilder() noexcept : generation_(next_generation()) {}

GraphBuilder::GraphBuilder(GraphBuilder&& other) noexcept
    : generation_(other.generation_), operation_(other.operation_),
      inputs_(other.inputs_), input_count_(other.input_count_),
      has_node_(other.has_node_), has_output_(other.has_output_),
      finished_(other.finished_) {
  other.generation_ = 0;
  other.input_count_ = 0;
  other.has_node_ = false;
  other.has_output_ = false;
  other.finished_ = true;
}

GraphBuilder& GraphBuilder::operator=(GraphBuilder&& other) noexcept {
  if (this != &other) {
    generation_ = other.generation_;
    operation_ = other.operation_;
    inputs_ = other.inputs_;
    input_count_ = other.input_count_;
    has_node_ = other.has_node_;
    has_output_ = other.has_output_;
    finished_ = other.finished_;
    other.generation_ = 0;
    other.input_count_ = 0;
    other.has_node_ = false;
    other.has_output_ = false;
    other.finished_ = true;
  }
  return *this;
}

Result<ValueId> GraphBuilder::input(
    compute::TensorDesc description) noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph builder is invalid"));
  }
  if (has_node_) {
    return std::unexpected(Status::FailedPrecondition(
        "graph inputs must be declared before the operation"));
  }
  if (!description.valid()) {
    return std::unexpected(
        Status::InvalidArgument("graph input description is invalid"));
  }
  if (input_count_ == inputs_.size()) {
    return std::unexpected(
        Status::Unimplemented("one-op graph supports at most two inputs"));
  }
  const std::uint32_t index = input_count_;
  inputs_[input_count_++] = description;
  return ValueId(generation_, ValueKind::kInput, index);
}

bool GraphBuilder::owns_input(ValueId value) const noexcept {
  return value.valid() && value.generation_ == generation_ &&
         value.kind_ == ValueKind::kInput && value.ordinal_ < input_count_;
}

Result<ValueId> GraphBuilder::add(compute::Operation operation,
                                  std::span<const ValueId> inputs) noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph builder is invalid"));
  }
  if (has_node_) {
    return std::unexpected(
        Status::AlreadyExists("one-op graph already has an operation"));
  }
  if (inputs.size() != 1 && inputs.size() != 2) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm graph operation requires one input and optional weight"));
  }
  if (inputs.size() != input_count_) {
    return std::unexpected(Status::InvalidArgument(
        "one-op graph must consume every declared input"));
  }
  for (const ValueId value : inputs) {
    if (!owns_input(value)) {
      return std::unexpected(Status::InvalidArgument(
          "graph operation input belongs to another builder or role"));
    }
  }
  if (inputs.front().ordinal_ != 0 ||
      (inputs.size() == 2 && inputs[1].ordinal_ != 1)) {
    return std::unexpected(Status::InvalidArgument(
        "graph operation inputs are not in declaration order"));
  }
  if (inputs.size() == 2 && inputs_[1] != inputs_[0]) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm weight description must match its input"));
  }
  if (!std::isfinite(operation.rms_norm().epsilon) ||
      operation.rms_norm().epsilon < 0.0F) {
    return std::unexpected(Status::InvalidArgument(
        "RMSNorm epsilon must be finite and non-negative"));
  }

  operation_ = operation;
  has_node_ = true;
  return ValueId(generation_, ValueKind::kOutput, 0);
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

Status GraphBuilder::output(ValueId value) noexcept {
  if (!valid()) return Status::FailedPrecondition("graph builder is invalid");
  if (!has_node_) {
    return Status::FailedPrecondition("graph has no operation output");
  }
  if (has_output_) return Status::AlreadyExists("graph output is already set");
  if (!value.valid() || value.generation_ != generation_ ||
      value.kind_ != ValueKind::kOutput || value.ordinal_ != 0) {
    return Status::InvalidArgument(
        "graph output belongs to another builder or role");
  }
  has_output_ = true;
  return Status::Ok();
}

Result<Graph> GraphBuilder::finish() && noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition("graph builder is invalid"));
  }
  if (!has_node_ || !has_output_) {
    return std::unexpected(Status::FailedPrecondition(
        "graph requires one operation and one published output"));
  }
  Graph result(generation_, operation_,
               std::span<const compute::TensorDesc>(inputs_.data(), input_count_));
  finished_ = true;
  return result;
}

}  // namespace hyle::graph
