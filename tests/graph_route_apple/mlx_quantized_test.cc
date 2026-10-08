#include "../compute_route_apple/packed_test_support.h"
#include "hyle/graph/apple.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mx = mlx::core;
hyle::Result<c::apple::Program> prepare(const c::apple::Context &context,
    c::Operation operation, const std::vector<mx::array> &inputs) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  std::vector<g::ValueId> ids;
  for (const auto &input : inputs) {
    const auto type = input.dtype() == mx::uint32 ? c::DataType::kUInt32 :
                      input.dtype() == mx::uint8 ? c::DataType::kUInt8 :
                      apple_tensor_test::logical(input.dtype());
    const std::vector<std::int64_t> shape(input.shape().begin(), input.shape().end());
    HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(type, shape));
    HYLE_ASSIGN_OR_RETURN(auto id, builder.input(description));
    ids.push_back(id);
  }
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(operation, ids));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  return g::apple::compile(context, graph);
}
TEST(AppleGraphMlxQuantized, NativeFormatsUseOrdinaryEntryAndCheckedExperts) {
  apple_packed_test::formats(prepare);
}
TEST(AppleGraphMlxQuantized, EmbeddingSelectsPackedRowsAndChecksChangedIds) {
  apple_packed_test::embedding(prepare);
}
} // namespace
