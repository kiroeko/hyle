#pragma once

#include <hyle/graph/vulkan.h>

#include "../../compute/cpp/vulkan_mature.h"

namespace sample_vulkan_mature_graph {
namespace s = sample_vulkan_mature;
namespace g = hyle::graph;

inline hyle::Status run(const s::v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto cases, s::examples(context));
  for (const auto &example : cases) {
    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    std::vector<g::ValueId> inputs;
    for (const auto &input : example.inputs) {
      HYLE_ASSIGN_OR_RETURN(auto id, builder.input(input.description()));
      inputs.push_back(id);
    }
    HYLE_ASSIGN_OR_RETURN(auto node,
                          builder.add_node(example.operation, inputs));
    for (std::size_t port = 0; port < example.expected.size(); ++port) {
      HYLE_ASSIGN_OR_RETURN(auto result, builder.result(node, port));
      // All floating examples have positive outputs. Abs additionally exercises
      // an original ggml region after IREE, then the ordinary native copy edge.
      if (example.floating) {
        HYLE_ASSIGN_OR_RETURN(
            auto positive,
            builder.add(s::c::Unary{s::c::UnaryKind::kAbs}, result));
        result = positive;
      }
      HYLE_ASSIGN_OR_RETURN(auto output, builder.add(s::c::Copy{}, result));
      HYLE_RETURN_IF_ERROR(builder.publish(output));
    }
    HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(auto program, g::vulkan::compile(context, graph));
    HYLE_RETURN_IF_ERROR(s::replay(context, example, program));
  }
  return hyle::Status::Ok();
}
} // namespace sample_vulkan_mature_graph
