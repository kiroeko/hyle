#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hyle/container/small_vector.h"
#include "hyle/graph/graph.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::detail {

inline constexpr std::uint32_t kNoNode = UINT32_MAX;

struct GraphIdentity final {
  static constexpr ValueId value(std::uint64_t generation, ValueKind kind,
                                 std::uint32_t ordinal) noexcept {
    return ValueId(generation, kind, ordinal);
  }
  static constexpr NodeId node(std::uint64_t generation,
                               std::uint32_t ordinal) noexcept {
    return NodeId(generation, ordinal);
  }
  static constexpr std::uint64_t generation(ValueId value) noexcept {
    return value.generation_;
  }
  static constexpr std::uint64_t generation(NodeId node) noexcept {
    return node.generation_;
  }
};

struct ValueRecord final {
  compute::TensorDesc description{};
  ValueKind kind = ValueKind::kInput;
  std::uint32_t producer = kNoNode;
};

struct NodeRecord final {
  compute::Operation operation{compute::RmsNorm{}};
  std::uint32_t input_begin = 0;
  std::uint32_t output = 0;
  std::uint8_t input_count = 0;
};

using ValueRecords = container::SmallVector<ValueRecord, 8, mem::AllocatorRef>;
using NodeRecords = container::SmallVector<NodeRecord, 8, mem::AllocatorRef>;
using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;

struct BuilderState final {
  BuilderState(std::uint64_t generation_value,
               mem::AllocatorRef allocator_value) noexcept
      : allocator(allocator_value), generation(generation_value),
        values(allocator_value), nodes(allocator_value), edges(allocator_value),
        inputs(allocator_value), outputs(allocator_value) {}

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  ValueRecords values;
  NodeRecords nodes;
  Ordinals edges;
  Ordinals inputs;
  Ordinals outputs;
};

struct GraphState final : util::RefCounted<GraphState> {
  explicit GraphState(BuilderState &builder) noexcept
      : allocator(builder.allocator), generation(builder.generation),
        values(std::move(builder.values)), nodes(std::move(builder.nodes)),
        edges(std::move(builder.edges)), inputs(std::move(builder.inputs)),
        outputs(std::move(builder.outputs)) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~GraphState();
    owner.deallocate(this, sizeof(GraphState), alignof(GraphState));
  }

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  ValueRecords values;
  NodeRecords nodes;
  Ordinals edges;
  Ordinals inputs;
  Ordinals outputs;
};

struct GraphAccess final {
  static Graph adopt(GraphState *state) noexcept { return Graph(state); }
  static const GraphState *state(const Graph &graph) noexcept {
    return graph.state_;
  }
};

inline void retain(GraphState *state) noexcept {
  if (state != nullptr)
    state->incref();
}

inline void release(GraphState *state) noexcept {
  if (state != nullptr)
    state->decref();
}

inline void destroy(BuilderState *state) noexcept {
  if (state == nullptr)
    return;
  const mem::AllocatorRef allocator = state->allocator;
  state->~BuilderState();
  allocator.deallocate(state, sizeof(BuilderState), alignof(BuilderState));
}

inline ValueId value_id(const GraphState &state,
                        std::uint32_t ordinal) noexcept {
  return GraphIdentity::value(state.generation, state.values[ordinal].kind,
                              ordinal);
}

inline ValueId value_id(const BuilderState &state,
                        std::uint32_t ordinal) noexcept {
  return GraphIdentity::value(state.generation, state.values[ordinal].kind,
                              ordinal);
}

inline bool owns(const GraphState &state, ValueId value) noexcept {
  return value.valid() &&
         GraphIdentity::generation(value) == state.generation &&
         value.ordinal() < state.values.size() &&
         value.kind() == state.values[value.ordinal()].kind;
}

inline bool owns(const BuilderState &state, ValueId value) noexcept {
  return value.valid() &&
         GraphIdentity::generation(value) == state.generation &&
         value.ordinal() < state.values.size() &&
         value.kind() == state.values[value.ordinal()].kind;
}

} // namespace hyle::graph::detail
