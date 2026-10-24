#pragma once

#include <cstdint>
#include <span>

#include "hyle/container/small_vector.h"
#include "hyle/graph/graph.h"
#include "hyle/graph/region_provider.h"

namespace hyle::graph::detail {

// Native-only sources need leaf candidates when partial placement makes their
// maximal regions overlap. Reuse the fixed writer's original proposals; all
// maximal regions keep their rank, and every fallback still requires real
// backend admission. Call only for implementations that admit single nodes.
inline Status add_region_singleton_fallbacks(RegionProposalWriter &writer,
                                             std::size_t begin) noexcept {
  const auto maximal = writer.proposals().subspan(begin);
  for (const auto &region : maximal) {
    if (region.nodes.size() < 2) continue;
    for (const auto node : region.nodes) {
      auto leaf = region;
      leaf.nodes = std::span(&node, 1);
      HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add(leaf));
      if (!more) return Status::Ok();
    }
  }
  return Status::Ok();
}

// A native region must not leave and re-enter through another provider.
// Discover maximal legal prefixes in topological order rather than rejecting
// the entire eligible set. Independent interleaved nodes do not split a region.
// A later selected node cannot add a previously uncovered external producer
// to the entry frontier: doing so would delay the independent earlier prefix.
// Only direct data/control producers establish this conservative proof; no
// transitive closure is inferred. Each edge is visited at most twice; epoch
// stamps avoid clearing graph-sized scratch after every boundary. The compiler
// still validates every emitted proposal.
template <class Eligible, class Emit>
Status discover_regions(const Graph &graph, mem::AllocatorRef allocator,
                        Eligible eligible, Emit emit) noexcept {
  container::SmallVector<NodeId, 16, mem::AllocatorRef> nodes(allocator);
  container::SmallVector<std::int64_t, 16, mem::AllocatorRef> marks(allocator);
  container::SmallVector<std::int64_t, 16, mem::AllocatorRef> incoming(allocator);
  if (!nodes.try_resize(graph.node_count()) ||
      !marks.try_resize(graph.node_count()) ||
      !incoming.try_resize(graph.node_count()))
    return Status::ResourceExhausted("graph region discovery metadata");
  std::int64_t epoch = 1;
  std::size_t size = 0;
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto selected, eligible(node));
    // Revisit only this node's direct edges after a cut, so its entry set is
    // recorded against the new epoch rather than the abandoned prefix.
    const auto visit_predecessors = [&](auto &&visit) -> Status {
      HYLE_ASSIGN_OR_RETURN(auto inputs, graph.node_input_count(node));
      for (std::size_t input = 0; input < inputs; ++input) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, input));
        if (value.kind() == ValueKind::kNodeOutput) {
          HYLE_ASSIGN_OR_RETURN(auto producer, graph.producer(value));
          visit(producer);
        }
      }
      HYLE_ASSIGN_OR_RETURN(auto dependencies, graph.node_dependency_count(node));
      for (std::size_t dependency = 0; dependency < dependencies; ++dependency) {
        HYLE_ASSIGN_OR_RETURN(auto before,
                              graph.node_dependency(node, dependency));
        visit(before);
      }
      return Status::Ok();
    };
    bool from_inside = false, from_outside = false, new_frontier = false;
    HYLE_RETURN_IF_ERROR(visit_predecessors([&](NodeId id) noexcept {
      from_inside |= marks[id.ordinal()] == epoch;
      from_outside |= marks[id.ordinal()] == -epoch;
      new_frontier |= marks[id.ordinal()] != epoch && incoming[id.ordinal()] != epoch;
    }));
    if (selected && size != 0 && (from_outside || new_frontier)) {
      HYLE_ASSIGN_OR_RETURN(auto more, emit(std::span(nodes).first(size)));
      if (!more)
        return Status::Ok();
      size = 0;
      ++epoch;
    }
    if (selected) {
      HYLE_RETURN_IF_ERROR(visit_predecessors([&](NodeId id) noexcept {
        if (marks[id.ordinal()] != epoch) incoming[id.ordinal()] = epoch;
      }));
      nodes[size++] = node;
      marks[i] = epoch;
    } else {
      marks[i] = from_inside || from_outside ? -epoch : 0;
    }
  }
  if (size != 0) {
    HYLE_ASSIGN_OR_RETURN(auto more, emit(std::span(nodes).first(size)));
    (void)more;
  }
  return Status::Ok();
}

} // namespace hyle::graph::detail
