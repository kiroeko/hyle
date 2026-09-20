#pragma once

#include <hyle/graph/apple.h>

#include "../../compute/cpp/apple_recurrent.h"

namespace sample_apple_graph {
inline hyle::Status recurrent(const hyle::compute::apple::Context &context) {
  namespace c = hyle::compute;
  namespace g = hyle::graph;
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  const auto examples = sample_apple_recurrent::inputs();
  std::array<g::ValueId, 6> ports;
  for (std::size_t i = 0; i < ports.size(); ++i) {
    const std::vector<std::int64_t> shape(examples[i].shape().begin(),
                                         examples[i].shape().end());
    HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32, shape));
    HYLE_ASSIGN_OR_RETURN(ports[i], builder.input(desc));
  }
  // MLX -> original whole ggml recurrent region -> MLX, with explicit state.
  HYLE_ASSIGN_OR_RETURN(ports[0], builder.add(c::Binary{}, ports[0], ports[0]));
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(c::Operation(c::GatedDeltaNet{}), ports));
  HYLE_ASSIGN_OR_RETURN(auto score, builder.result(node, 0));
  HYLE_ASSIGN_OR_RETURN(auto state, builder.result(node, 1));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Binary{}, score, ports[2]));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_RETURN_IF_ERROR(builder.publish(state));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto program, g::apple::compile(context, graph));
  graph = {};
  return sample_apple_recurrent::replay(program, true);
}
} // namespace sample_apple_graph
