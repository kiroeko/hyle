#pragma once

#include "hyle/graph/detail/region_access.h"

namespace hyle::graph::batch::detail {
// Mathematical proof for this materialized reference recipe. The original
// private Invocation binds indices from its immutable owned layout version;
// no caller can replace that binding through MaterializedInvocation.
inline Status bind_materialized_pages(Invocation &invocation, const Graph &graph,
    ValueId capacity, ValueId indices, compute::ConstTensorView tensor,
    std::span<const compute::ReadLease> pages) noexcept {
  if (!graph.valid() || capacity == indices || capacity.kind() != ValueKind::kInput ||
      indices.kind() != ValueKind::kInput || !tensor.valid() || !tensor.description().is_dense())
    return Status::InvalidArgument("materialized projection requires its exact dense inputs");
  HYLE_ASSIGN_OR_RETURN(auto source, graph.value_description(capacity));
  HYLE_ASSIGN_OR_RETURN(auto index_description, graph.value_description(indices));
  (void)index_description;
  if (source != tensor.description())
    return Status::InvalidArgument("materialized capacity differs from its immutable source");
  NodeId reshape{}, gather{}; ValueId alias{};
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
    for (std::size_t port = 0; port < count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(node, port));
      if (input != capacity) continue;
      HYLE_ASSIGN_OR_RETURN(auto operation, graph.operation(node));
      const auto *view = operation.get_if<compute::View>();
      if (reshape.valid() || !view || view->kind() != compute::ViewKind::kReshape || count != 1)
        return Status::InvalidArgument("materialized capacity has a reader outside its unique reshape");
      HYLE_ASSIGN_OR_RETURN(auto outputs, graph.node_output_count(node));
      if (outputs != 1) return Status::InvalidArgument("materialized capacity reshape arity");
      reshape = node; HYLE_ASSIGN_OR_RETURN(alias, graph.result(node));
    }
  }
  if (!reshape.valid()) return Status::InvalidArgument("materialized capacity has no unique reshape");
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto count, graph.node_input_count(node));
    for (std::size_t port = 0; port < count; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto input, graph.node_input(node, port));
      if (input != alias) continue;
      HYLE_ASSIGN_OR_RETURN(auto operation, graph.operation(node));
      const auto *op = operation.get_if<compute::Gather>();
      if (gather.valid() || !op || op->axis != 0 || op->bounds != compute::IndexBoundsPolicy::kChecked || count != 2 || port != 0)
        return Status::InvalidArgument("materialized alias has a reader outside its unique checked Gather");
      HYLE_ASSIGN_OR_RETURN(auto supplied, graph.node_input(node, 1));
      if (supplied != indices) return Status::InvalidArgument("materialized Gather uses foreign metadata");
      gather = node;
    }
  }
  if (!gather.valid()) return Status::InvalidArgument("materialized capacity has no checked Gather");
  const std::array aliases{alias}; const std::array readers{gather};
  return graph::detail::RegionAccess::bind_read_projection(invocation,
      {capacity, aliases, readers}, tensor, pages);
}
} // namespace hyle::graph::batch::detail
