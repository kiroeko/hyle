#pragma once

#include "hyle/graph/region_provider.h"
#include "hyle/container/small_vector.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace hyle::graph::detail {

// Cold data/control reader proof shared by exact, single-output graph chains.
// It carries no format, provider, runtime permissions or payload addresses.
struct ClosedRegionPattern {
template <class T>
static std::optional<T> operation(const Graph &graph, NodeId node) noexcept {
  if (!node.valid()) return std::nullopt;
  const auto op = graph.operation(node);
  if (!op) return std::nullopt;
  const auto *value = op->get_if<T>();
  return value ? std::optional<T>{*value} : std::nullopt;
}
static NodeId producer(const Graph &graph, ValueId value) noexcept {
  if (value.kind() != ValueKind::kNodeOutput) return {};
  const auto node = graph.producer(value);
  return node ? *node : NodeId{};
}
static bool arity(const Graph &graph, NodeId node, std::size_t inputs) noexcept {
  if (!node.valid()) return false;
  const auto in = graph.node_input_count(node), out = graph.node_output_count(node),
             resources = graph.node_resource_count(node);
  return in && out && resources && *in == inputs && *out == 1 && *resources == 0;
}

// One cold reader index; neither permissions nor payload addresses are cached.
struct Uses final {
  container::SmallVector<std::uint8_t, 16, mem::AllocatorRef> data;
  container::SmallVector<NodeId, 16, mem::AllocatorRef> reader;
  container::SmallVector<std::size_t, 16, mem::AllocatorRef> control;
  explicit Uses(mem::AllocatorRef allocator) noexcept
      : data(allocator), reader(allocator), control(allocator) {}
  Status initialize(const Graph &graph) noexcept {
    if (!data.try_resize(graph.value_count()) || !reader.try_resize(graph.value_count()) ||
        !control.try_resize(graph.node_count()))
      return Status::ResourceExhausted("closed region discovery metadata allocation failed");
    std::ranges::fill(data, std::uint8_t{0});
    std::ranges::fill(reader, NodeId{});
    std::ranges::fill(control, std::size_t{0});
    for (std::size_t i = 0; i < graph.node_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
      HYLE_ASSIGN_OR_RETURN(auto inputs, graph.node_input_count(node));
      for (std::size_t n = 0; n < inputs; ++n) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(node, n));
        auto &count = data[value.ordinal()];
        if (count == 0) reader[value.ordinal()] = node;
        if (count < 2) ++count;
      }
      HYLE_ASSIGN_OR_RETURN(auto dependencies, graph.node_dependency_count(node));
      for (std::size_t n = 0; n < dependencies; ++n) {
        HYLE_ASSIGN_OR_RETURN(auto before, graph.node_dependency(node, n));
        if (control[before.ordinal()] == SIZE_MAX)
          return Status::OutOfRange("closed region control reader count overflows");
        ++control[before.ordinal()];
      }
    }
    for (std::size_t n = 0; n < graph.output_count(); ++n) {
      HYLE_ASSIGN_OR_RETURN(auto value, graph.output_id(n));
      data[value.ordinal()] = 2;
    }
    return Status::Ok();
  }
};
template <class Pattern>
static bool closed(const Graph &graph, const Pattern &p, const Uses &uses,
            std::span<const std::uint8_t> allowed) noexcept {
  const auto terminal = producer(graph, p.output);
  for (const auto node : p.span()) {
    if (!allowed.empty() && !allowed[node.ordinal()]) return false;
    if (node == terminal) continue;
    const auto output = graph.result(node).value();
    if (uses.data[output.ordinal()] != 1 || !p.contains(uses.reader[output.ordinal()]))
      return false;
    std::size_t internal_controls = 0;
    for (const auto after : p.span())
      for (std::size_t n = 0; n < graph.node_dependency_count(after).value(); ++n)
        internal_controls += graph.node_dependency(after, n).value() == node;
    if (uses.control[node.ordinal()] != internal_controls) return false;
  }
  return true;
}

};
} // namespace hyle::graph::detail
