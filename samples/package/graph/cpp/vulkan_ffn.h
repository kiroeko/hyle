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
  HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
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
       .report = &report,
       .workload = workload});
  if (!program)
    return program;
  if (program->workload() != workload)
    return std::unexpected(
        hyle::Status::Internal("Vulkan Graph workload was not retained"));
  if (!report.status.ok() || report.nodes.size() != 5 ||
      report.native_dispatch_count != 4 || report.managed_region_count != 1 ||
      report.invocation_storage_block_bytes.empty() ||
      report.opaque_storage_bytes)
    return std::unexpected(
        hyle::Status::Internal("incomplete Vulkan compilation report"));
  // Native/managed resources and constants survive this local Graph and
  // tensors.
  return program;
}

inline hyle::Status topk(const v::Context &context) {
  namespace t = sample_vulkan_topk;
  HYLE_ASSIGN_OR_RETURN(auto desc, t::description());
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto positive,
                        builder.add(c::Unary{c::UnaryKind::kAbs}, input));
  HYLE_ASSIGN_OR_RETURN(
      auto node, builder.add_node(c::Operation(c::TopK{.count = t::selected}),
                                  std::array{positive}));
  HYLE_ASSIGN_OR_RETURN(auto values, builder.result(node, 0));
  HYLE_ASSIGN_OR_RETURN(auto indices, builder.result(node, 1));
  HYLE_ASSIGN_OR_RETURN(auto negative,
                        builder.add(c::Unary{c::UnaryKind::kNeg}, values));
  HYLE_ASSIGN_OR_RETURN(auto copied_indices, builder.add(c::Copy{}, indices));
  HYLE_RETURN_IF_ERROR(builder.publish(negative));
  HYLE_RETURN_IF_ERROR(builder.publish(copied_indices));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto program, gv::compile(context, graph));
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  graph = {};
  for (int phase = 0; phase < 3; ++phase) {
    auto host = t::scores(phase);
    HYLE_ASSIGN_OR_RETURN(auto tensor,
                          context.upload(desc, std::as_bytes(std::span(host))));
    HYLE_RETURN_IF_ERROR(slot.bind(tensor));
    HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    for (auto &value : host)
      value = std::abs(value);
    HYLE_RETURN_IF_ERROR(
        t::check(context, result->tensors(), host, true, true));
  }
  const auto host = t::scores(1);
  for (std::int64_t origin : {1, 4, 7}) {
    HYLE_ASSIGN_OR_RETURN(auto tensor,
                          t::captured_scores(context, host, origin));
    HYLE_ASSIGN_OR_RETURN(auto constants, g::GraphBuilder::create());
    HYLE_ASSIGN_OR_RETURN(auto source,
                          constants.constant(tensor.description()));
    HYLE_ASSIGN_OR_RETURN(
        auto chosen,
        constants.add_node(c::Operation(c::TopK{.count = t::selected}),
                           std::array{source}));
    for (std::uint32_t port = 0; port < 2; ++port) {
      HYLE_ASSIGN_OR_RETURN(auto output, constants.result(chosen, port));
      HYLE_RETURN_IF_ERROR(constants.publish(output));
    }
    HYLE_ASSIGN_OR_RETURN(auto constant_graph, std::move(constants).finish());
    HYLE_ASSIGN_OR_RETURN(
        auto fixed, gv::compile(context, constant_graph, std::array{tensor}));
    HYLE_ASSIGN_OR_RETURN(auto fixed_slot, fixed.create_invocation());
    HYLE_RETURN_IF_ERROR(fixed_slot.bind(std::span<const v::Tensor>{}));
    HYLE_ASSIGN_OR_RETURN(auto work, fixed_slot.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    HYLE_RETURN_IF_ERROR(t::check(context, result->tensors(), host, true));
  }
  return hyle::Status::Ok();
}

inline hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_RETURN_IF_ERROR(topk(context));
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
