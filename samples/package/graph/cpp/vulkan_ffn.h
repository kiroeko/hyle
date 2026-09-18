#pragma once

#include <hyle/graph/vulkan.h>

#include "../../compute/cpp/vulkan_ffn.h"

namespace sample_vulkan_graph {
namespace s = sample_vulkan_direct;
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gv = g::vulkan;

struct Affine final {
  static constexpr c::OperationId operation_id{0x68796c65766b7361ULL, 1};
  static constexpr auto operation_name = "sample-vulkan-affine";
  c::OperationTraits operation_traits() const noexcept {
    return {.inputs = {1, 1}, .outputs = 1};
  }
  hyle::Status
  infer_output_descriptions(std::span<const c::TensorDesc> inputs,
                            std::span<c::TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1)
      return hyle::Status::InvalidArgument("sample affine arity");
    outputs[0] = inputs[0];
    return hyle::Status::Ok();
  }
};

inline hyle::Result<v::Program> prepare(const v::Context &context,
                                        const s::Weights &weights, int rows) {
  HYLE_ASSIGN_OR_RETURN(auto constants, weights.constants(context));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(s::dense({rows, s::Weights::k})));
  HYLE_ASSIGN_OR_RETURN(auto gate, builder.constant(s::dense(
                                       {s::Weights::k, s::Weights::hidden})));
  HYLE_ASSIGN_OR_RETURN(
      auto up, builder.constant(s::dense({s::Weights::k, s::Weights::hidden})));
  HYLE_ASSIGN_OR_RETURN(
      auto down,
      builder.constant(s::dense({s::Weights::hidden, s::Weights::output})));
  HYLE_ASSIGN_OR_RETURN(auto a, builder.add(c::MatMul{}, x, gate));
  HYLE_ASSIGN_OR_RETURN(auto b, builder.add(c::MatMul{}, x, up));
  HYLE_ASSIGN_OR_RETURN(
      auto product,
      builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, a,
                  b));
  HYLE_ASSIGN_OR_RETURN(auto projected,
                        builder.add(c::MatMul{}, product, down));
  HYLE_ASSIGN_OR_RETURN(auto output,
                        builder.add(c::Operation::custom(Affine{}), projected));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  gv::CompileReport report;
  auto program = gv::compile(
      context, graph, constants,
      {.custom = [](const v::Context &owner, const c::Operation &operation,
                    std::span<const c::TensorDesc>,
                    const c::TensorDesc &description)
           -> hyle::Result<gv::CustomDispatch> {
         if (!operation.custom_as<Affine>())
           return std::unexpected(
               hyle::Status::Unimplemented("unknown application schema"));
         HYLE_ASSIGN_OR_RETURN(auto kernel, s::affine(owner));
         HYLE_ASSIGN_OR_RETURN(auto elements, description.element_count());
         const auto count = static_cast<std::uint32_t>(elements);
         return gv::CustomDispatch{std::move(kernel),
                                   {(count + 63) / 64, 1, 1},
                                   s::count_push(count)};
       },
       .report = &report});
  if (!program)
    return program;
  if (!report.status.ok() || report.nodes.size() != 5 ||
      report.native_dispatch_count != 4 || report.managed_region_count != 1 ||
      report.invocation_storage_block_bytes.empty() ||
      report.opaque_storage_bytes)
    return std::unexpected(
        hyle::Status::Internal("incomplete Vulkan compilation report"));
  // Native/managed resources and constants survive this local Graph and tensors.
  return program;
}

inline hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  const s::Weights weights;
  for (int rows : {1, 16}) {
    HYLE_ASSIGN_OR_RETURN(auto program, prepare(context, weights, rows));
    HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
    for (std::uint32_t iteration = 1; iteration <= 3; ++iteration) {
      const auto host =
          s::data(static_cast<std::size_t>(rows) * s::Weights::k, iteration);
      HYLE_ASSIGN_OR_RETURN(auto input,
                            context.upload(s::dense({rows, s::Weights::k}),
                                           std::as_bytes(std::span(host))));
      HYLE_RETURN_IF_ERROR(slot.bind(input));
      HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
      HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
      HYLE_RETURN_IF_ERROR(
          s::check(context, result->tensors()[0], weights.oracle(host, rows)));
    }
  }
  return context.close();
}
} // namespace sample_vulkan_graph
