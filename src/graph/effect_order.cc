#include "graph_state.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "hyle/core/assert.h"

namespace hyle::graph::detail {
namespace {

using Bytes = container::SmallVector<std::uint8_t, 16, mem::AllocatorRef>;
constexpr std::uint8_t kRead = 1, kWrite = 2;

Status no_memory() noexcept {
  return Status::ResourceExhausted("graph effect analysis allocation failed");
}

template <class Visitor>
void predecessors(const BuilderState &graph, std::size_t ordinal,
                  Visitor &&visit) noexcept {
  const NodeRecord &node = graph.nodes[ordinal];
  for (std::size_t port = 0; port < node.input_count; ++port) {
    const auto producer =
        graph.values[graph.edges[node.input_begin + port]].producer;
    if (producer != kNoNode)
      visit(producer);
  }
  for (std::size_t edge = 0; edge < node.dependency_count; ++edge)
    visit(graph.dependencies[node.dependency_begin + edge]);
}

// Writers must form a chain. Reads can form an antichain between writers;
// checking only the largest reader ordinal would silently lose WAR edges.
Status validate_domain(const BuilderState &graph,
                       std::span<const std::uint8_t> modes, Ordinals &latest,
                       Bytes &seen, Ordinals &stack) noexcept {
  std::fill(latest.begin(), latest.end(), kNoNode);
  std::fill(seen.begin(), seen.end(), std::uint8_t{0});
  std::uint32_t last_writer = kNoNode;
  std::size_t pending_reads = 0;
  for (std::size_t ordinal = 0; ordinal < graph.nodes.size(); ++ordinal) {
    std::uint32_t reachable = kNoNode;
    predecessors(graph, ordinal, [&](std::uint32_t predecessor) noexcept {
      const auto candidate = latest[predecessor];
      if (candidate != kNoNode &&
          (reachable == kNoNode || candidate > reachable))
        reachable = candidate;
    });
    latest[ordinal] = reachable;
    if (modes[ordinal] == 0)
      continue;
    if (last_writer != kNoNode && reachable != last_writer)
      return Status::FailedPrecondition(
          "conflicting resource access requires a data/control dependency");
    if ((modes[ordinal] & kWrite) == 0) {
      ++pending_reads;
      continue;
    }
    if (pending_reads != 0) {
      stack.clear();
      auto enqueue = [&](std::uint32_t predecessor) noexcept {
        if ((last_writer != kNoNode && predecessor <= last_writer) ||
            seen[predecessor])
          return;
        seen[predecessor] = 1;
        // Capacity is graph.nodes.size(); each node is pushed at most once.
        const bool pushed = stack.try_push_back(predecessor);
        HYLE_FATAL_ASSERT(pushed);
      };
      predecessors(graph, ordinal, enqueue);
      std::size_t reached_reads = 0;
      while (!stack.empty()) {
        const auto predecessor = stack.back();
        stack.pop_back();
        if (modes[predecessor] == kRead)
          ++reached_reads;
        predecessors(graph, predecessor, enqueue);
      }
      if (reached_reads != pending_reads)
        return Status::FailedPrecondition(
            "resource writer must depend on every preceding reader");
    }
    pending_reads = 0;
    last_writer = static_cast<std::uint32_t>(ordinal);
    latest[ordinal] = last_writer;
  }
  return Status::Ok();
}

struct Access final {
  std::uint32_t node = 0;
  std::uint8_t mode = 0;
};
using Accesses = container::SmallVector<Access, 16, mem::AllocatorRef>;

} // namespace

Status validate_effect_order(const BuilderState &graph) noexcept {
  Ordinals counts(graph.allocator), starts(graph.allocator),
      cursors(graph.allocator);
  Ordinals unknown(graph.allocator);
  if (!counts.try_resize(graph.effect_resource_count))
    return no_memory();
  std::fill(counts.begin(), counts.end(), std::uint32_t{0});
  std::size_t access_count = 0;
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    const auto &node = graph.nodes[index];
    const auto traits = compute::operation_traits(node.operation);
    if (traits.effect == compute::EffectKind::kExternalSideEffect &&
        !unknown.try_push_back(static_cast<std::uint32_t>(index)))
      return no_memory();
    if (traits.resource_count == 0)
      continue;
    if (node.resource_begin == kNoNode)
      return Status::FailedPrecondition(
          "graph operation has unbound resource ports");
    for (std::size_t port = 0; port < traits.resource_count; ++port)
      ++counts[graph.resource_edges[node.resource_begin + port]];
    access_count += traits.resource_count;
  }
  if (access_count == 0 && unknown.empty())
    return Status::Ok();
  Accesses accesses(graph.allocator);
  Ordinals latest(graph.allocator), stack(graph.allocator);
  Bytes modes(graph.allocator), seen(graph.allocator);
  if (!starts.try_resize(counts.size()) || !cursors.try_resize(counts.size()) ||
      !accesses.try_resize(access_count) ||
      !latest.try_resize(graph.nodes.size()) ||
      !stack.try_reserve(graph.nodes.size()) ||
      !modes.try_resize(graph.nodes.size()) ||
      !seen.try_resize(graph.nodes.size()))
    return no_memory();
  std::uint32_t start = 0;
  for (std::size_t resource = 0; resource < counts.size(); ++resource) {
    starts[resource] = start;
    cursors[resource] = start;
    start += counts[resource];
  }
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    const auto &node = graph.nodes[index];
    const auto count = compute::operation_traits(node.operation).resource_count;
    for (std::size_t port = 0; port < count; ++port) {
      auto access = compute::operation_resource_access(node.operation, port);
      if (!access)
        return std::move(access).error();
      const std::uint8_t mode =
          *access == compute::ResourceAccess::kRead
              ? kRead
              : (*access == compute::ResourceAccess::kWrite ? kWrite
                                                            : kRead | kWrite);
      const auto resource = graph.resource_edges[node.resource_begin + port];
      accesses[cursors[resource]++] = {static_cast<std::uint32_t>(index), mode};
    }
  }
  bool any_domain = false;
  for (std::size_t resource = 0; resource < counts.size(); ++resource) {
    if (counts[resource] == 0)
      continue;
    any_domain = true;
    if (unknown.empty()) {
      bool writes = false, multiple_nodes = false;
      const auto first_node = accesses[starts[resource]].node;
      for (std::size_t index = starts[resource]; index < cursors[resource];
           ++index) {
        writes |= (accesses[index].mode & kWrite) != 0;
        multiple_nodes |= accesses[index].node != first_node;
      }
      // No conflicting pair exists in a read-only or single-node domain.
      // Many independent resources therefore need O(V + D + A), not D DAG
      // traversals. Unknown effects deliberately disable this shortcut.
      if (!writes || !multiple_nodes)
        continue;
    }
    std::fill(modes.begin(), modes.end(), std::uint8_t{0});
    for (std::size_t index = starts[resource]; index < cursors[resource];
         ++index)
      modes[accesses[index].node] |= accesses[index].mode;
    for (auto index : unknown)
      modes[index] = kRead | kWrite;
    HYLE_RETURN_IF_ERROR(validate_domain(graph, modes, latest, seen, stack));
  }
  if (!any_domain) {
    std::fill(modes.begin(), modes.end(), std::uint8_t{0});
    for (auto index : unknown)
      modes[index] = kWrite;
    return validate_domain(graph, modes, latest, seen, stack);
  }
  return Status::Ok();
}

} // namespace hyle::graph::detail
