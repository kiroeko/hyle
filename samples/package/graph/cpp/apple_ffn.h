#pragma once

#include <hyle/graph/apple.h>
#include "apple_recurrent.h"

#include "../../compute/cpp/apple_ffn.h"

namespace sample_apple_graph {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace a = c::apple;

inline hyle::Result<g::Graph> build(int rows) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  const std::array<std::int64_t, 2> activation_shape{rows, 64};
  const std::array<std::int64_t, 2> scale_shape{64, 1};
  const std::array<std::int64_t, 2> down_shape{64, 16};
  HYLE_ASSIGN_OR_RETURN(
      auto input_desc,
      c::TensorDesc::dense(c::DataType::kFloat32, activation_shape));
  HYLE_ASSIGN_OR_RETURN(auto q4_desc, a::affine_q4_description(64, 64));
  HYLE_ASSIGN_OR_RETURN(
      auto scale_desc,
      c::TensorDesc::dense(c::DataType::kFloat32, scale_shape));
  HYLE_ASSIGN_OR_RETURN(
      auto down_desc, c::TensorDesc::dense(c::DataType::kFloat32, down_shape));
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(input_desc));
  std::array<g::ValueId, 6> constants;
  for (std::size_t i = 0; i < constants.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(constants[i],
                          builder.constant(i % 3 == 0 ? q4_desc : scale_desc));
  }
  HYLE_ASSIGN_OR_RETURN(auto down, builder.constant(down_desc));
  const c::QuantizedMatMul qmm{.mode =
                                   c::QuantizedWeightMode::kUnsignedAffine4};
  HYLE_ASSIGN_OR_RETURN(
      auto gate, builder.add(qmm, x, constants[0], constants[1], constants[2]));
  HYLE_ASSIGN_OR_RETURN(
      auto up, builder.add(qmm, x, constants[3], constants[4], constants[5]));
  HYLE_ASSIGN_OR_RETURN(
      auto product,
      builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
                  gate, up));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::MatMul{}, product, down));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  return std::move(builder).finish();
}

inline hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_RETURN_IF_ERROR(sample_apple_direct::verify_base_kernel(context));
  HYLE_RETURN_IF_ERROR(recurrent(context));
  sample_apple_direct::Weights weights;
  for (int rows : {1, 4}) {
    HYLE_ASSIGN_OR_RETURN(auto graph, build(rows));
    g::apple::CompileReport report;
    HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
    HYLE_ASSIGN_OR_RETURN(auto program,
                          g::apple::compile(context, graph, weights.constants,
                                            {.report = &report,
                                             .workload = workload}));
    if (program.workload() != workload)
      return hyle::Status::Internal("Apple Graph workload was not retained");
    if (!report.status.ok() || report.nodes.size() != 4 ||
        report.logical_constant_bytes == 0 || report.private_storage_bytes)
      return hyle::Status::Internal("incomplete Apple compilation report");
    graph = {}; // only the compiled native program is needed for repetition
    HYLE_RETURN_IF_ERROR(sample_apple_direct::replay(program, weights, rows));
  }
  return context.close();
}
} // namespace sample_apple_graph
