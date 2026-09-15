#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>

#include "../compute/shape_program.h"
#include "hyle/container/small_vector.h"
#include "hyle/core/assert.h"
#include "hyle/graph/graph.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::detail {

inline constexpr std::uint32_t kNoNode = UINT32_MAX;

struct GraphIdentity final {
  static constexpr EffectResourceId
  effect_resource(std::uint64_t generation, std::uint32_t ordinal) noexcept {
    return EffectResourceId(generation, ordinal);
  }
  static constexpr std::uint64_t
  generation(EffectResourceId resource) noexcept {
    return resource.generation_;
  }
  static constexpr DimensionId dimension(std::uint64_t generation,
                                         std::uint32_t ordinal) noexcept {
    return DimensionId(generation, ordinal);
  }
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
  static constexpr std::uint64_t generation(DimensionId dimension) noexcept {
    return dimension.generation_;
  }
};

using DimensionRecord = compute::detail::ShapeDimension;

struct ValueRecord final {
  TensorSpec spec{};
  // Valid exactly when spec is concrete. Keeping the resolved value beside
  // the spec preserves O(1) legacy concrete queries and lets the existing
  // compiler consume concrete-only graphs without a second representation.
  compute::TensorDesc description{};
  ValueKind kind = ValueKind::kInput;
  std::uint32_t producer = kNoNode;
};

struct NodeRecord final {
  // The logical node never pretends a symbolic ViewSpec is a fixed View.
  // Providers and physical planning consume a specialized concrete copy.
  std::variant<compute::Operation, std::uint32_t> descriptor{};
  std::uint32_t input_begin = 0;
  std::uint32_t output_begin = 0;
  std::uint32_t dependency_begin = 0;
  std::uint32_t dependency_count = 0;
  std::uint32_t resource_begin = kNoNode;
  std::uint16_t input_count = 0;
  std::uint8_t output_count = 0;

  const compute::Operation &operation() const noexcept {
    const auto *result = std::get_if<compute::Operation>(&descriptor);
    HYLE_FATAL_ASSERT(result != nullptr);
    return *result;
  }
  const std::uint32_t *view_index() const noexcept {
    return std::get_if<std::uint32_t>(&descriptor);
  }
  compute::OperationKind kind() const noexcept {
    return view_index() ? compute::OperationKind::kView : operation().kind();
  }
  compute::OperationTraits traits() const noexcept {
    // Only the View schema's shape-independent traits are queried here; no
    // placeholder descriptor is ever supplied to inference or a Provider.
    return view_index()
               ? compute::operation_traits(compute::Operation(compute::View{}))
               : compute::operation_traits(operation());
  }
};

// Metadata-only cold binding. IDs have already been checked against the
// owning builder; values includes independent and derived program slots.
Result<compute::Operation>
bind_operation(const GraphState &graph, const NodeRecord &node,
               std::span<const std::int64_t> values) noexcept;
Result<compute::View> bind_view(const ViewSpec &view,
                                std::span<const std::int64_t> values) noexcept;

struct ControlEdgeRecord final {
  std::uint32_t node = 0;
  std::uint32_t prerequisite = 0;
};

using ValueRecords = container::SmallVector<ValueRecord, 8, mem::AllocatorRef>;
using DimensionRecords =
    container::SmallVector<DimensionRecord, 4, mem::AllocatorRef>;
using ShapeGuards =
    container::SmallVector<compute::detail::ShapeGuard, 4, mem::AllocatorRef>;
using NodeRecords = container::SmallVector<NodeRecord, 8, mem::AllocatorRef>;
using ViewSpecs = container::SmallVector<ViewSpec, 0, mem::AllocatorRef>;
using Ordinals = container::SmallVector<std::uint32_t, 16, mem::AllocatorRef>;
using ControlEdgeRecords =
    container::SmallVector<ControlEdgeRecord, 8, mem::AllocatorRef>;

struct BuilderState final {
  BuilderState(std::uint64_t generation_value,
               mem::AllocatorRef allocator_value) noexcept
      : allocator(allocator_value), generation(generation_value),
        dimensions(allocator_value), shape_guards(allocator_value),
        values(allocator_value), nodes(allocator_value), views(allocator_value),
        edges(allocator_value), resource_edges(allocator_value),
        inputs(allocator_value), constants(allocator_value),
        outputs(allocator_value), control_edges(allocator_value),
        dependencies(allocator_value) {}

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  DimensionRecords dimensions;
  ShapeGuards shape_guards;
  ValueRecords values;
  NodeRecords nodes;
  ViewSpecs views;
  Ordinals edges;
  Ordinals resource_edges;
  std::uint32_t effect_resource_count = 0;
  Ordinals inputs;
  Ordinals constants;
  Ordinals outputs;
  ControlEdgeRecords control_edges;
  Ordinals dependencies;
};

struct GraphState final : util::RefCounted<GraphState> {
  explicit GraphState(BuilderState &builder) noexcept
      : allocator(builder.allocator), generation(builder.generation),
        dimensions(std::move(builder.dimensions)),
        shape_guards(std::move(builder.shape_guards)),
        values(std::move(builder.values)), nodes(std::move(builder.nodes)),
        views(std::move(builder.views)), edges(std::move(builder.edges)),
        resource_edges(std::move(builder.resource_edges)),
        effect_resource_count(builder.effect_resource_count),
        inputs(std::move(builder.inputs)),
        constants(std::move(builder.constants)),
        outputs(std::move(builder.outputs)),
        dependencies(std::move(builder.dependencies)) {}

  void destroy() noexcept {
    const mem::AllocatorRef owner = allocator;
    this->~GraphState();
    owner.deallocate(this, sizeof(GraphState), alignof(GraphState));
  }

  mem::AllocatorRef allocator{};
  std::uint64_t generation = 0;
  DimensionRecords dimensions;
  ShapeGuards shape_guards;
  ValueRecords values;
  NodeRecords nodes;
  ViewSpecs views;
  Ordinals edges;
  Ordinals resource_edges;
  std::uint32_t effect_resource_count = 0;
  Ordinals inputs;
  Ordinals constants;
  Ordinals outputs;
  Ordinals dependencies;
};

struct GraphAccess final {
  static Graph adopt(GraphState *state) noexcept { return Graph(state); }
  static const GraphState *state(const Graph &graph) noexcept {
    return graph.state_;
  }
};

template <class State>
inline const ViewSpec *view_spec(const State &graph,
                                 const NodeRecord &node) noexcept {
  const auto *index = node.view_index();
  if (index == nullptr)
    return nullptr;
  HYLE_FATAL_ASSERT(*index < graph.views.size());
  return &graph.views[*index];
}

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

inline bool owns(const GraphState &state, DimensionId dimension) noexcept {
  return dimension.valid() &&
         GraphIdentity::generation(dimension) == state.generation &&
         dimension.ordinal() < state.dimensions.size();
}

inline bool owns(const GraphState &state, NodeId node) noexcept {
  return node.valid() && GraphIdentity::generation(node) == state.generation &&
         node.ordinal() < state.nodes.size();
}

inline bool owns(const BuilderState &state, ValueId value) noexcept {
  return value.valid() &&
         GraphIdentity::generation(value) == state.generation &&
         value.ordinal() < state.values.size() &&
         value.kind() == state.values[value.ordinal()].kind;
}

inline bool owns(const BuilderState &state, DimensionId dimension) noexcept {
  return dimension.valid() &&
         GraphIdentity::generation(dimension) == state.generation &&
         dimension.ordinal() < state.dimensions.size();
}

inline bool owns(const BuilderState &state, NodeId node) noexcept {
  return node.valid() && GraphIdentity::generation(node) == state.generation &&
         node.ordinal() < state.nodes.size();
}

template <class State>
inline bool owns(const State &state, EffectResourceId resource) noexcept {
  return resource.valid() &&
         GraphIdentity::generation(resource) == state.generation &&
         resource.ordinal() < state.effect_resource_count;
}

Status validate_effect_order(const BuilderState &state) noexcept;

} // namespace hyle::graph::detail
