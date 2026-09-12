#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/operation.h"
#include "hyle/error/result.h"
#include "hyle/graph/visibility.h"
#include "hyle/mem/allocator.h"

namespace hyle::graph {

namespace detail {
struct GraphIdentity;
struct GraphState;
struct BuilderState;
struct GraphAccess;
} // namespace detail

enum class ValueKind : std::uint8_t {
  kInput = 0,
  kNodeOutput,
};

// Graph-local identity. The private generation prevents IDs from another
// builder/graph being accepted merely because their ordinals happen to match.
// ordinal() is useful for diagnostics; runtime binding remains O(1).
class ValueId final {
public:
  constexpr ValueId() noexcept = default;

  constexpr bool valid() const noexcept { return generation_ != 0; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr ValueKind kind() const noexcept { return kind_; }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(ValueId, ValueId) noexcept = default;

private:
  friend struct detail::GraphIdentity;
  constexpr ValueId(std::uint64_t generation, ValueKind kind,
                    std::uint32_t ordinal) noexcept
      : generation_(generation), ordinal_(ordinal), kind_(kind) {}

  std::uint64_t generation_ = 0;
  std::uint32_t ordinal_ = 0;
  ValueKind kind_ = ValueKind::kInput;
};

class NodeId final {
public:
  constexpr NodeId() noexcept = default;

  constexpr bool valid() const noexcept { return generation_ != 0; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(NodeId, NodeId) noexcept = default;

private:
  friend struct detail::GraphIdentity;
  constexpr NodeId(std::uint64_t generation, std::uint32_t ordinal) noexcept
      : generation_(generation), ordinal_(ordinal) {}

  std::uint64_t generation_ = 0;
  std::uint32_t ordinal_ = 0;
};

// Immutable, self-contained logical DAG metadata. Copies share one intrusive
// state; no Graph owns tensor payload, Context, Queue or provider artifacts.
class Graph final {
public:
  Graph() noexcept = default;
  HYLE_GRAPH_API Graph(const Graph &other) noexcept;
  HYLE_GRAPH_API Graph &operator=(const Graph &other) noexcept;
  HYLE_GRAPH_API Graph(Graph &&other) noexcept;
  HYLE_GRAPH_API Graph &operator=(Graph &&other) noexcept;
  HYLE_GRAPH_API ~Graph();

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_API std::size_t value_count() const noexcept;
  HYLE_GRAPH_API std::size_t input_count() const noexcept;
  HYLE_GRAPH_API std::size_t node_count() const noexcept;
  HYLE_GRAPH_API std::size_t output_count() const noexcept;

  HYLE_GRAPH_API Result<ValueId> input_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc &>
  input_description(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<NodeId> node_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::Operation &>
  operation(NodeId node) const noexcept;
  HYLE_GRAPH_API Result<std::size_t>
  node_input_count(NodeId node) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  node_input(NodeId node, std::size_t input_index) const noexcept;
  HYLE_GRAPH_API Result<ValueId> node_output(NodeId node) const noexcept;
  HYLE_GRAPH_API Result<NodeId> producer(ValueId value) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  output_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc &>
  output_description(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc &>
  value_description(ValueId value) const noexcept;

private:
  friend struct detail::GraphAccess;
  explicit Graph(detail::GraphState *state) noexcept : state_(state) {}
  detail::GraphState *state_ = nullptr;
};

struct GraphBuilderOptions final {
  // Empty selects mem::system_allocator(). The backing must outlive the
  // builder and every Graph published from it.
  mem::AllocatorRef metadata_allocator{};
};

// Fallible, single-owner DAG construction. add() accepts only values already
// produced by this builder, making cycles unrepresentable. A failed mutator
// leaves logical content unchanged; capacity growth may remain reserved.
class GraphBuilder final {
public:
  GraphBuilder() noexcept = default;
  GraphBuilder(const GraphBuilder &) = delete;
  GraphBuilder &operator=(const GraphBuilder &) = delete;
  HYLE_GRAPH_API GraphBuilder(GraphBuilder &&other) noexcept;
  HYLE_GRAPH_API GraphBuilder &operator=(GraphBuilder &&other) noexcept;
  HYLE_GRAPH_API ~GraphBuilder();

  HYLE_GRAPH_API static Result<GraphBuilder>
  create(GraphBuilderOptions options = {}) noexcept;

  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  HYLE_GRAPH_API Result<ValueId>
  input(compute::TensorDesc description) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(compute::Operation operation,
                                     std::span<const ValueId> inputs) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(compute::RmsNorm operation,
                                     ValueId input) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(compute::RmsNorm operation, ValueId input,
                                     ValueId weight) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(compute::MatMul operation, ValueId left,
                                     ValueId right) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(compute::GatedActivation operation,
                                     ValueId input) noexcept;
  HYLE_GRAPH_API Status output(ValueId value) noexcept;
  HYLE_GRAPH_API Result<Graph> finish() && noexcept;

private:
  explicit GraphBuilder(detail::BuilderState *state) noexcept : state_(state) {}
  detail::BuilderState *state_ = nullptr;
};

} // namespace hyle::graph
