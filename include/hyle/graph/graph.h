#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

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
  kConstant,
  kNodeOutput,
};

// Graph-local symbolic dimension identity. Reusing one ID is the complete
// equality constraint supported by the bounded Graph IR; arbitrary affine
// equations deliberately remain with the graph producer.
class DimensionId final {
public:
  constexpr DimensionId() noexcept = default;

  constexpr bool valid() const noexcept { return generation_ != 0; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(DimensionId, DimensionId) noexcept = default;

private:
  friend struct detail::GraphIdentity;
  constexpr DimensionId(std::uint64_t generation,
                        std::uint32_t ordinal) noexcept
      : generation_(generation), ordinal_(ordinal) {}

  std::uint64_t generation_ = 0;
  std::uint32_t ordinal_ = 0;
};

struct DimensionBounds final {
  std::int64_t minimum = 1;
  std::int64_t maximum = 1;
  std::int64_t multiple_of = 1;

  friend constexpr bool operator==(DimensionBounds,
                                   DimensionBounds) noexcept = default;
};

// One logical extent is either a non-negative fixed value or one graph-local
// bounded symbol. An invalid fixed value remains representable so fallible
// TensorSpec construction can diagnose it without exceptions.
class Dimension final {
public:
  constexpr Dimension() noexcept = default;
  constexpr explicit Dimension(DimensionId symbol) noexcept : symbol_(symbol) {}

  static constexpr Dimension fixed(std::int64_t value) noexcept {
    Dimension result;
    result.fixed_ = value;
    return result;
  }

  constexpr bool valid() const noexcept {
    return symbol_.valid() || fixed_ >= 0;
  }
  constexpr bool is_fixed() const noexcept { return !symbol_.valid(); }
  constexpr bool is_symbol() const noexcept { return symbol_.valid(); }
  constexpr std::int64_t fixed_value() const noexcept { return fixed_; }
  constexpr DimensionId symbol() const noexcept { return symbol_; }

  friend constexpr bool operator==(Dimension, Dimension) noexcept = default;

private:
  DimensionId symbol_{};
  std::int64_t fixed_ = -1;
};

enum class TensorSpecKind : std::uint8_t {
  kInvalid = 0,
  kConcrete,
  kDense,
  kBlockedDense,
};

// Logical graph-time tensor metadata. Concrete preserves every TensorDesc
// layout. Symbolic descriptions intentionally support only dense and
// blocked-dense storage; arbitrary symbolic strides and opaque layouts need a
// provider-owned typed contract instead of guessed generic arithmetic.
class TensorSpec final {
public:
  TensorSpec() noexcept = default;

  HYLE_GRAPH_API static Result<TensorSpec>
  concrete(compute::TensorDesc description) noexcept;
  HYLE_GRAPH_API static Result<TensorSpec>
  dense(compute::DataType type, std::span<const Dimension> dimensions) noexcept;
  HYLE_GRAPH_API static Result<TensorSpec>
  blocked_dense(compute::DataType logical_type,
                std::span<const Dimension> dimensions,
                compute::StorageFormat format) noexcept;

  bool valid() const noexcept { return kind_ != TensorSpecKind::kInvalid; }
  explicit operator bool() const noexcept { return valid(); }
  TensorSpecKind kind() const noexcept { return kind_; }
  compute::DataType data_type() const noexcept { return type_; }
  const compute::StorageFormat &storage_format() const noexcept {
    return format_;
  }
  std::size_t rank() const noexcept { return rank_; }
  std::span<const Dimension> dimensions() const noexcept {
    return {dimensions_.data(), rank_};
  }
  HYLE_GRAPH_API bool is_symbolic() const noexcept;
  // Succeeds for an original concrete description or a fixed dense/
  // blocked-dense spec. Symbolic specs require graph specialization.
  HYLE_GRAPH_API Result<compute::TensorDesc>
  concrete_description() const noexcept;

  friend constexpr bool operator==(const TensorSpec &left,
                                   const TensorSpec &right) noexcept {
    if (left.kind_ != right.kind_ || left.type_ != right.type_ ||
        left.rank_ != right.rank_ || left.format_ != right.format_ ||
        left.concrete_ != right.concrete_) {
      return false;
    }
    for (std::size_t index = 0; index < left.rank_; ++index) {
      if (left.dimensions_[index] != right.dimensions_[index])
        return false;
    }
    return true;
  }

private:
  std::array<Dimension, compute::kMaxTensorRank> dimensions_{};
  compute::TensorDesc concrete_{};
  compute::StorageFormat format_{};
  compute::DataType type_ = compute::DataType::kUnknown;
  std::uint8_t rank_ = 0;
  TensorSpecKind kind_ = TensorSpecKind::kInvalid;
};

struct ShapeBinding final {
  DimensionId dimension{};
  std::int64_t value = 0;

  friend constexpr bool operator==(ShapeBinding,
                                   ShapeBinding) noexcept = default;
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
  HYLE_GRAPH_API std::size_t dimension_count() const noexcept;
  HYLE_GRAPH_API std::size_t input_count() const noexcept;
  HYLE_GRAPH_API std::size_t constant_count() const noexcept;
  HYLE_GRAPH_API std::size_t node_count() const noexcept;
  HYLE_GRAPH_API std::size_t output_count() const noexcept;

  HYLE_GRAPH_API Result<DimensionId>
  dimension_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<DimensionBounds>
  dimension_bounds(DimensionId dimension) const noexcept;

  HYLE_GRAPH_API Result<ValueId> input_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc &>
  input_description(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const TensorSpec &>
  input_spec(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  constant_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc &>
  constant_description(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const TensorSpec &>
  constant_spec(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<NodeId> node_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::Operation &>
  operation(NodeId node) const noexcept;
  HYLE_GRAPH_API Result<std::size_t>
  node_input_count(NodeId node) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  node_input(NodeId node, std::size_t input_index) const noexcept;
  HYLE_GRAPH_API Result<std::size_t>
  node_dependency_count(NodeId node) const noexcept;
  HYLE_GRAPH_API Result<NodeId>
  node_dependency(NodeId node, std::size_t dependency_index) const noexcept;
  HYLE_GRAPH_API Result<std::size_t>
  node_output_count(NodeId node) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  result(NodeId node, std::size_t output_index = 0) const noexcept;
  HYLE_GRAPH_API Result<NodeId> producer(ValueId value) const noexcept;
  HYLE_GRAPH_API Result<ValueId>
  output_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc &>
  output_description(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const TensorSpec &>
  output_spec(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc &>
  value_description(ValueId value) const noexcept;
  HYLE_GRAPH_API Result<const TensorSpec &>
  value_spec(ValueId value) const noexcept;

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

// Fallible, single-owner DAG construction. add_node() accepts only values
// already produced by this builder, making cycles unrepresentable. A failed
// mutator leaves logical content unchanged; capacity growth may remain
// reserved.
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

  HYLE_GRAPH_API Result<DimensionId> dimension(DimensionBounds bounds) noexcept;

  HYLE_GRAPH_API Result<ValueId>
  input(compute::TensorDesc description) noexcept;
  HYLE_GRAPH_API Result<ValueId> input(TensorSpec spec) noexcept;
  HYLE_GRAPH_API Result<ValueId>
  constant(compute::TensorDesc description) noexcept;
  HYLE_GRAPH_API Result<ValueId> constant(TensorSpec spec) noexcept;
  HYLE_GRAPH_API Result<NodeId>
  add_node(compute::Operation operation,
           std::span<const ValueId> inputs) noexcept;
  HYLE_GRAPH_API Result<NodeId>
  add_node(compute::Operation operation, std::span<const ValueId> inputs,
           std::span<const TensorSpec> outputs) noexcept;
  HYLE_GRAPH_API Result<ValueId>
  result(NodeId node, std::size_t output_index = 0) const noexcept;
  HYLE_GRAPH_API Status depends_on(NodeId node, NodeId prerequisite) noexcept;
  template <class Descriptor, class... Inputs>
    requires std::constructible_from<compute::Operation, Descriptor> &&
             (std::same_as<std::remove_cvref_t<Inputs>, ValueId> && ...)
  Result<ValueId> add(Descriptor operation, Inputs... inputs) noexcept {
    const compute::Operation erased(std::move(operation));
    if (compute::operation_traits(erased).outputs != 1) {
      return std::unexpected(Status::InvalidArgument(
          "multi-output operation requires add_node() and result()"));
    }
    const std::array<ValueId, sizeof...(Inputs)> values{inputs...};
    auto node = add_node(erased, values);
    if (!node)
      return std::unexpected(std::move(node).error());
    return result(*node);
  }
  template <class Descriptor, class... Inputs>
    requires std::constructible_from<compute::Operation, Descriptor> &&
             (std::same_as<std::remove_cvref_t<Inputs>, ValueId> && ...)
  Result<ValueId> add(Descriptor operation, TensorSpec output,
                      Inputs... inputs) noexcept {
    const compute::Operation erased(std::move(operation));
    if (compute::operation_traits(erased).outputs != 1) {
      return std::unexpected(Status::InvalidArgument(
          "multi-output operation requires add_node() and result()"));
    }
    const std::array<ValueId, sizeof...(Inputs)> values{inputs...};
    const std::array<TensorSpec, 1> outputs{output};
    auto node = add_node(erased, values, outputs);
    if (!node)
      return std::unexpected(std::move(node).error());
    return result(*node);
  }
  HYLE_GRAPH_API Status publish(ValueId value) noexcept;
  HYLE_GRAPH_API Result<Graph> finish() && noexcept;

private:
  explicit GraphBuilder(detail::BuilderState *state) noexcept : state_(state) {}
  detail::BuilderState *state_ = nullptr;
};

} // namespace hyle::graph
