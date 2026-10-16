#pragma once

#include "hyle/graph/graph.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/std_allocator.h"

#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace hyle::graph::detail {

// Cold native-only projection. Legality belongs to GraphBuilder::finish; this
// preserves both data and control reachability for native artifact formation.
struct NativeAnalysis final {
  using Flags = std::vector<std::uint8_t, mem::StdAllocator<std::uint8_t>>;
  explicit NativeAnalysis(mem::AllocatorRef allocator)
      : needed(mem::StdAllocator<std::uint8_t>(allocator)),
        boundary(mem::StdAllocator<std::uint8_t>(allocator)),
        control_predecessor(mem::StdAllocator<std::uint8_t>(allocator)) {}
  Flags needed, boundary, control_predecessor;
};

inline Result<NativeAnalysis> analyze_native_pure(
    const Graph &graph, mem::AllocatorRef allocator = mem::system_allocator(),
    std::span<const std::uint8_t> cpu_nodes = {}) noexcept {
  if (!graph || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("native analysis requires graph and allocator"));
  if (!cpu_nodes.empty() && cpu_nodes.size() != graph.node_count())
    return std::unexpected(Status::InvalidArgument("CPU effect admission mask size differs"));
  try {
    NativeAnalysis result(allocator);
    result.needed.resize(graph.node_count());
    result.boundary.resize(graph.node_count());
    result.control_predecessor.resize(graph.node_count());
    NativeAnalysis::Flags live{mem::StdAllocator<std::uint8_t>(allocator)};
    live.resize(graph.value_count());
    for (std::size_t i = 0; i < graph.output_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto value, graph.output_id(i));
      live[value.ordinal()] = 1;
    }
    for (std::size_t i = graph.node_count(); i != 0; --i) {
      HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i - 1));
      HYLE_ASSIGN_OR_RETURN(const auto &operation, graph.operation(node));
      const auto traits = compute::operation_traits(operation);
      // Even dead effects cannot be silently erased. A native effect needs
      // actual ports, permissions and a preserves_effects carrier first.
      if (traits.effect != compute::EffectKind::kPure || traits.resource_count != 0) {
        if (cpu_nodes.empty() || !cpu_nodes[i - 1])
          return std::unexpected(Status::Unimplemented("native Graph has no qualified effect carrier"));
        result.needed[i - 1] = result.boundary[i - 1] = 1;
      }
      HYLE_ASSIGN_OR_RETURN(auto count, graph.node_output_count(node));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.result(node, port));
        result.needed[i - 1] |= live[value.ordinal()];
      }
      if (!result.needed[i - 1]) continue;
      if (traits.alias == compute::AliasPolicy::kMustAliasInput0 &&
          operation.kind() != compute::OperationKind::kView &&
          (cpu_nodes.empty() || !cpu_nodes[i - 1]))
        return std::unexpected(Status::Unimplemented("native custom schema has no qualified mandatory alias carrier"));
      HYLE_ASSIGN_OR_RETURN(count, graph.node_input_count(node));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, port));
        live[value.ordinal()] = 1;
      }
      HYLE_ASSIGN_OR_RETURN(count, graph.node_dependency_count(node));
      for (std::size_t edge = 0; edge < count; ++edge) {
        HYLE_ASSIGN_OR_RETURN(auto prerequisite, graph.node_dependency(node, edge));
        result.needed[prerequisite.ordinal()] = 1;
        result.boundary[prerequisite.ordinal()] = result.boundary[i - 1] = 1;
        result.control_predecessor[prerequisite.ordinal()] = 1;
      }
    }
    return result;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("native reachability allocation failed"));
  }
}

// A fragment may absorb a same-target successor only when that successor does
// not add a new external prerequisite at its entry. Otherwise an independent
// prefix would inherit the wait of a later join. Keep explicit native/effect
// boundaries and project original data AND control edges, including dead effects.
struct NativePartition final {
  struct Fragment final {
    std::size_t begin = 0, end = 0;
    std::size_t dependency_begin = 0, dependency_count = 0;
  };
  template<class T> using Vector = std::vector<T, mem::StdAllocator<T>>;
  explicit NativePartition(mem::AllocatorRef allocator)
      : fragments(mem::StdAllocator<Fragment>(allocator)),
        dependencies(mem::StdAllocator<std::size_t>(allocator)) {}
  Vector<Fragment> fragments;
  Vector<std::size_t> dependencies;
  std::span<const std::size_t> prerequisites(const Fragment &fragment) const noexcept {
    return std::span<const std::size_t>(dependencies).subspan(
        fragment.dependency_begin, fragment.dependency_count);
  }
};

inline Result<NativePartition> partition_native_joint(
    const Graph &graph, const NativeAnalysis &analysis,
    std::span<const ExecutionTargetId> targets,
    mem::AllocatorRef allocator = mem::system_allocator()) noexcept {
  if (!graph || !allocator.valid() || targets.size() != graph.node_count() ||
      analysis.needed.size() != graph.node_count() ||
      analysis.boundary.size() != graph.node_count())
    return std::unexpected(Status::InvalidArgument("joint partition inputs differ"));
  try {
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    NativePartition result(allocator);
    NativePartition::Vector<std::size_t> producers(
        graph.value_count(), none, mem::StdAllocator<std::size_t>(allocator));
    NativePartition::Vector<std::size_t> node_fragments(
        graph.node_count(), none, mem::StdAllocator<std::size_t>(allocator));
    NativePartition::Vector<std::size_t> dependency_stamp(
        graph.node_count(), none, mem::StdAllocator<std::size_t>(allocator));
    result.fragments.reserve(graph.node_count());
    // Every logical value has at most one original producer. Inputs/constants
    // stay outside the fragment graph; their readiness is a boundary stage.
    for (std::size_t i = 0; i < graph.node_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
      HYLE_ASSIGN_OR_RETURN(auto count, graph.node_output_count(node));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.result(node, port));
        producers[value.ordinal()] = i;
      }
    }
    auto visit_prerequisites = [&](NodeId node, auto &&visit) -> Status {
      HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, port));
        const auto producer = producers[value.ordinal()];
        if (producer != none) {
          if (producer >= node.ordinal() || node_fragments[producer] == none)
            return Status::Internal("joint input producer is not live and earlier");
          visit(node_fragments[producer]);
        }
      }
      HYLE_ASSIGN_OR_RETURN(count, graph.node_dependency_count(node));
      for (std::size_t edge = 0; edge < count; ++edge) {
        HYLE_ASSIGN_OR_RETURN(auto source, graph.node_dependency(node, edge));
        if (source.ordinal() >= node.ordinal() || node_fragments[source.ordinal()] == none)
          return Status::Internal("joint control producer is not live and earlier");
        visit(node_fragments[source.ordinal()]);
      }
      return Status::Ok();
    };
    for (std::size_t i = 0; i < graph.node_count(); ++i) {
      if (!analysis.needed[i]) continue;
      if (!targets[i].valid())
        return std::unexpected(Status::InvalidArgument("joint node target is invalid"));
      HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
      auto active = result.fragments.empty() ? none : result.fragments.size() - 1;
      bool split = active == none || analysis.boundary[i];
      if (active != none) {
        const auto begin = result.fragments[active].begin;
        split = split || analysis.boundary[begin] || targets[begin] != targets[i];
        if (!split) {
          HYLE_RETURN_IF_ERROR(visit_prerequisites(node, [&](std::size_t source) {
            if (source != active && dependency_stamp[source] != active) split = true;
          }));
        }
      }
      if (split) {
        if (active != none) result.fragments[active].end = i;
        active = result.fragments.size();
        result.fragments.push_back({i, graph.node_count(), result.dependencies.size(), 0});
        HYLE_RETURN_IF_ERROR(visit_prerequisites(node, [&](std::size_t source) {
          if (dependency_stamp[source] == active) return;
          dependency_stamp[source] = active;
          result.dependencies.push_back(source);
          ++result.fragments.back().dependency_count;
        }));
      }
      node_fragments[i] = active;
    }
    return result;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("joint partition metadata allocation"));
  }
}

} // namespace hyle::graph::detail
