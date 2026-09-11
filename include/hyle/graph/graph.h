#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/compute/operation.h"
#include "hyle/error/result.h"
#include "hyle/graph/visibility.h"

namespace hyle::graph {

enum class ValueKind : std::uint8_t {
  kInput = 0,
  kOutput,
};

// Typed graph-local identity. Equality includes a private builder generation,
// so an ID from another builder cannot accidentally bind a structurally similar
// graph. ordinal() is diagnostic only and is never used without the generation.
class ValueId final {
 public:
  constexpr ValueId() noexcept = default;

  constexpr bool valid() const noexcept { return generation_ != 0; }
  constexpr explicit operator bool() const noexcept { return valid(); }
  constexpr ValueKind kind() const noexcept { return kind_; }
  constexpr std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend constexpr bool operator==(ValueId, ValueId) noexcept = default;

 private:
  friend class GraphBuilder;
  friend class Graph;
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
  friend class GraphBuilder;
  friend class Graph;
  constexpr NodeId(std::uint64_t generation, std::uint32_t ordinal) noexcept
      : generation_(generation), ordinal_(ordinal) {}

  std::uint64_t generation_ = 0;
  std::uint32_t ordinal_ = 0;
};

class Graph final {
 public:
  Graph() noexcept = default;

  bool valid() const noexcept { return generation_ != 0; }
  explicit operator bool() const noexcept { return valid(); }
  std::size_t input_count() const noexcept { return input_count_; }
  std::size_t node_count() const noexcept { return valid() ? 1 : 0; }
  std::size_t output_count() const noexcept { return valid() ? 1 : 0; }

  HYLE_GRAPH_API Result<ValueId> input_id(
      std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc&> input_description(
      std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<NodeId> node_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::Operation&> operation(
      std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<ValueId> output_id(std::size_t index = 0) const noexcept;
  HYLE_GRAPH_API Result<const compute::TensorDesc&> output_description(
      std::size_t index = 0) const noexcept;

 private:
  friend class GraphBuilder;
  Graph(std::uint64_t generation, compute::Operation operation,
        std::span<const compute::TensorDesc> inputs) noexcept;

  std::uint64_t generation_ = 0;
  compute::Operation operation_{compute::RmsNorm{}};
  std::array<compute::TensorDesc, 2> inputs_{};
  compute::TensorDesc output_{};
  std::uint8_t input_count_ = 0;
};

// Single-owner construction surface. v1 deliberately admits one typed
// RMSNorm node, while IDs and immutable publication already have the shape a
// multi-node DAG will extend. A failed mutator leaves the builder unchanged.
class GraphBuilder final {
 public:
  HYLE_GRAPH_API GraphBuilder() noexcept;
  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder& operator=(const GraphBuilder&) = delete;
  HYLE_GRAPH_API GraphBuilder(GraphBuilder&& other) noexcept;
  HYLE_GRAPH_API GraphBuilder& operator=(GraphBuilder&& other) noexcept;

  bool valid() const noexcept { return generation_ != 0 && !finished_; }
  explicit operator bool() const noexcept { return valid(); }

  HYLE_GRAPH_API Result<ValueId> input(
      compute::TensorDesc description) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(
      compute::Operation operation, std::span<const ValueId> inputs) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(compute::RmsNorm operation,
                                     ValueId input) noexcept;
  HYLE_GRAPH_API Result<ValueId> add(compute::RmsNorm operation, ValueId input,
                                     ValueId weight) noexcept;
  HYLE_GRAPH_API Status output(ValueId value) noexcept;
  HYLE_GRAPH_API Result<Graph> finish() && noexcept;

 private:
  bool owns_input(ValueId value) const noexcept;

  std::uint64_t generation_ = 0;
  compute::Operation operation_{compute::RmsNorm{}};
  std::array<compute::TensorDesc, 2> inputs_{};
  std::uint8_t input_count_ = 0;
  bool has_node_ = false;
  bool has_output_ = false;
  bool finished_ = false;
};

}  // namespace hyle::graph
