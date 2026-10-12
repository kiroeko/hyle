#pragma once

#include "hyle/graph/graph.h"
#include "hyle/mem/std_allocator.h"

#include <cstdint>
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

} // namespace hyle::graph::detail
