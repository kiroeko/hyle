#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/apple.h"

#include "hyle/obs/trace.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace hyle::graph::apple {
namespace mx = mlx::core;
namespace ca = compute::apple;

namespace {
struct Node {
  compute::Operation operation;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
};

struct Plan {
  compute::WorkloadContext workload;
  std::vector<Node> nodes;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> constants;
  std::vector<std::size_t> outputs;
  std::vector<compute::TensorDesc> descriptions;
  std::vector<mx::array> values;
  CustomLowering custom;
};

bool fixed_ggml_model_operation(const compute::Operation &operation) noexcept {
  return operation.get_if<compute::GatedDeltaNet>() ||
         operation.custom_as<compute::model_ops::GroupedMatMul>() ||
         operation.custom_as<
             compute::model_ops::HyperConnectionCombine>() ||
         operation.custom_as<compute::model_ops::HyperConnectionPre>() ||
         operation.custom_as<compute::model_ops::HyperConnectionPost>() ||
         operation.custom_as<compute::model_ops::LightningIndexer>();
}

Result<ca::InputSpec> native_spec(const compute::TensorDesc &description) {
  return ca::input_spec(description);
}

bool ggml_model_operation(const Plan &plan, const Node &node) noexcept {
  return fixed_ggml_model_operation(node.operation) ||
         (node.operation.get_if<compute::MatMul>() && node.inputs.size() == 2 &&
          compute::gguf_quantized_type(
              plan.descriptions[node.inputs[1]].storage_format())
              .has_value());
}

Status match(const compute::TensorDesc &description, const mx::array &value) {
  HYLE_ASSIGN_OR_RETURN(auto spec, native_spec(description));
  if (spec.shape != value.shape() || spec.dtype != value.dtype())
    return Status::InvalidArgument(
        "Apple Graph native array shape/dtype disagrees with schema");
  return Status::Ok();
}

Result<std::vector<mx::array>> trace(const Plan &plan,
                                     std::span<const Node> nodes,
                                     std::span<const std::size_t> input_ids,
                                     std::span<const std::size_t> output_ids,
                                     const std::vector<mx::array> &inputs,
                                     mx::Stream stream) {
  // This storage exists only during cold MLX tracing. Hot runs invoke its
  // compiled artifact; they never allocate or interpret this HYLE value table.
  std::vector<std::optional<mx::array>> values(plan.descriptions.size());
  for (std::size_t i = 0; i < inputs.size(); ++i)
    values[input_ids[i]] = inputs[i];
  for (std::size_t i = 0; i < plan.constants.size(); ++i)
    values[plan.constants[i]] = plan.values[i];
  std::vector<mx::array> arguments;
  for (const auto &node : nodes) {
    arguments.clear();
    for (auto index : node.inputs)
      arguments.push_back(*values[index]);
    const bool mature_custom =
        node.operation.custom_as<compute::model_ops::CausalConv1d>()
            .has_value();
    auto outputs =
        node.operation.kind() == compute::OperationKind::kCustom &&
                !mature_custom
            ? plan.custom(node.operation, arguments, stream)
            : ca::lower(node.operation, arguments, stream, plan.workload);
    if (!outputs)
      return std::unexpected(std::move(outputs).error());
    if (outputs->size() != node.outputs.size())
      return std::unexpected(Status::InvalidArgument(
          "Apple custom lowering output arity mismatch"));
    for (std::size_t i = 0; i < outputs->size(); ++i) {
      const auto index = node.outputs[i];
      HYLE_RETURN_IF_ERROR(match(plan.descriptions[index], (*outputs)[i]));
      values[index] = std::move((*outputs)[i]);
    }
  }
  std::vector<mx::array> result;
  result.reserve(output_ids.size());
  for (auto index : output_ids)
    result.push_back(std::move(*values[index]));
  return result;
}

Result<ca::Program> compile_regions(const ca::Context &context,
                                    std::shared_ptr<const Plan> plan) {
  ca::detail::RegionPlan native;
  native.workload = plan->workload;
  native.values.resize(plan->descriptions.size());
  native.inputs = plan->inputs;
  native.outputs = plan->outputs;
  native.constants = plan->constants;
  native.constant_values = plan->values;
  for (std::size_t id = 0; id < plan->descriptions.size(); ++id)
    if (plan->descriptions[id].valid()) {
      HYLE_ASSIGN_OR_RETURN(native.values[id],
                            native_spec(plan->descriptions[id]));
    }
  std::vector<std::size_t> last_use(native.values.size());
  std::vector<bool> constant(native.values.size(), false);
  for (auto id : plan->constants)
    constant[id] = true;
  for (std::size_t i = 0; i < plan->nodes.size(); ++i)
    for (auto id : plan->nodes[i].inputs)
      last_use[id] = i;
  for (auto id : plan->outputs)
    last_use[id] = plan->nodes.size();

  for (std::size_t begin = 0; begin < plan->nodes.size();) {
    ca::detail::Region region;
    const auto &first = plan->nodes[begin];
    if (ggml_model_operation(*plan, first)) {
      region.kind = ca::detail::RegionKind::kGgml;
      region.operation = first.operation;
      region.inputs = first.inputs;
      region.outputs = first.outputs;
      for (auto id : first.inputs)
        region.native_inputs.push_back(plan->descriptions[id]);
      native.regions.push_back(std::move(region));
      ++begin;
      continue;
    }
    auto end = begin + 1;
    while (end < plan->nodes.size() &&
           !ggml_model_operation(*plan, plan->nodes[end]))
      ++end;
    std::vector<bool> internal(native.values.size(), false);
    std::vector<bool> argument(native.values.size(), false);
    for (auto i = begin; i < end; ++i) {
      for (auto id : plan->nodes[i].inputs)
        if (!internal[id] && !constant[id] && !argument[id]) {
          region.inputs.push_back(id);
          argument[id] = true;
        }
      for (auto id : plan->nodes[i].outputs) {
        internal[id] = true;
        if (last_use[id] >= end)
          region.outputs.push_back(id);
      }
    }
    // Only cross-region/public values become MLX outputs. Publishing every
    // intermediate here would prevent the original compiler's fusion.
    region.mlx = [plan, begin, end, inputs = region.inputs,
                  outputs = region.outputs](const auto &values,
                                            mx::Stream stream) {
      return trace(
          *plan, std::span<const Node>(plan->nodes).subspan(begin, end - begin),
          inputs, outputs, values, stream);
    };
    native.regions.push_back(std::move(region));
    begin = end;
  }
  return ca::detail::ContextAccess::compile(context, std::move(native));
}
} // namespace

static Result<ca::Program> compile_impl(const ca::Context &context,
                                        const Graph &graph,
                                        std::span<const mx::array> constants,
                                        CompileOptions options) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.graph.apple.compile");
  if (!options.workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple workload"));
  if (!context || !graph || graph.output_count() == 0)
    return std::unexpected(Status::InvalidArgument(
        "Apple Graph needs a Context and published outputs"));
  if (graph.dimension_count() != 0 || graph.effect_resource_count() != 0)
    return std::unexpected(
        Status::Unimplemented("Apple Graph admits fixed, pure regions only"));
  if (constants.size() != graph.constant_count())
    return std::unexpected(
        Status::InvalidArgument("Apple Graph constant count mismatch"));
  try {
    auto *report = options.report;
    if (report != nullptr) {
      HYLE_ASSIGN_OR_RETURN(const auto &environment, context.environment());
      report->environment = environment;
      report->implementation_policy = "One MLX GPU region; fixed typed "
                                      "lowering plus same-runtime custom nodes";
      report->numerical_policy =
          "Typed F32/F16/BF16 tensor, F32 gate/norm/reduce/softmax and stable "
          "TopK; "
          "I32/I64/Bool storage, checked Gather/ordered Scatter/Scan and "
          "batched MatMul, exact-position RoPE, normalized Hadamard, GQA SDPA "
          "and affine Q4/group64; "
          "each node retains its "
          "schema contract; MLX private intermediates unknown";
      report->boundary_policy =
          "Caller inputs/constants must be ready; no HYLE shadow output copy; "
          "native result leases retain MLX outputs; private GPU algorithm "
          "status is read after completion; internal fusion/copies "
          "unknown";
      report->concurrency_policy =
          "Independent Invocation/results; one Context owner executes FIFO "
          "and checks native completion before releasing admitted resources";
      report->declared_constant_count = constants.size();
    }
    auto plan = std::make_shared<Plan>();
    plan->workload = options.workload;
    plan->custom = std::move(options.custom);
    plan->descriptions.resize(graph.value_count());
    std::vector<ca::InputSpec> specs;
    specs.reserve(graph.input_count());
    auto describe = [&](ValueId id) -> Status {
      HYLE_ASSIGN_OR_RETURN(const auto &spec, graph.value_spec(id));
      HYLE_ASSIGN_OR_RETURN(auto description, spec.concrete_description());
      HYLE_ASSIGN_OR_RETURN(auto native, native_spec(description));
      plan->descriptions[id.ordinal()] = description;
      return Status::Ok();
    };
    for (std::size_t i = 0; i < graph.input_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.input_id(i));
      HYLE_RETURN_IF_ERROR(describe(id));
      HYLE_ASSIGN_OR_RETURN(auto spec,
                            native_spec(plan->descriptions[id.ordinal()]));
      specs.push_back(std::move(spec));
      plan->inputs.push_back(id.ordinal());
    }
    for (std::size_t i = 0; i < constants.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.constant_id(i));
      HYLE_RETURN_IF_ERROR(describe(id));
      if (constants[i].status() != mx::array::available)
        return std::unexpected(Status::FailedPrecondition(
            "Apple Graph constants must be successfully ready"));
      HYLE_RETURN_IF_ERROR(
          match(plan->descriptions[id.ordinal()], constants[i]));
      plan->constants.push_back(id.ordinal());
      plan->values.push_back(constants[i]);
      if (report != nullptr) {
        HYLE_ASSIGN_OR_RETURN(
            auto bytes, plan->descriptions[id.ordinal()].required_bytes());
        if (bytes > SIZE_MAX - report->logical_constant_bytes)
          return std::unexpected(Status::ResourceExhausted(
              "Apple report logical constant bytes overflow"));
        report->logical_constant_bytes += bytes;
      }
    }
    std::vector<bool> live(graph.value_count(), false);
    for (std::size_t i = 0; i < graph.output_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.output_id(i));
      plan->outputs.push_back(id.ordinal());
      live[id.ordinal()] = true;
    }
    // GraphBuilder guarantees topological order. Reverse reachability removes
    // dead pure nodes without splitting the mature runtime's fusion region.
    std::vector<std::optional<Node>> nodes(graph.node_count());
    for (std::size_t i = graph.node_count(); i > 0; --i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.node_id(i - 1));
      HYLE_ASSIGN_OR_RETURN(const auto &operation, graph.operation(id));
      HYLE_ASSIGN_OR_RETURN(auto dependencies, graph.node_dependency_count(id));
      const auto traits = compute::operation_traits(operation);
      if (traits.effect != compute::EffectKind::kPure ||
          traits.resource_count != 0 || dependencies != 0)
        return std::unexpected(
            Status::Unimplemented("Apple Graph cannot lower effects or "
                                  "explicit control dependencies"));
      HYLE_ASSIGN_OR_RETURN(auto output_count, graph.node_output_count(id));
      bool needed = false;
      Node node;
      node.operation = operation;
      for (std::size_t port = 0; port < output_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.result(id, port));
        node.outputs.push_back(value.ordinal());
        needed = needed || live[value.ordinal()];
        HYLE_RETURN_IF_ERROR(describe(value));
      }
      if (!needed)
        continue;
      if (operation.kind() == compute::OperationKind::kCustom &&
          !operation.custom_as<compute::model_ops::CausalConv1d>() &&
          !fixed_ggml_model_operation(operation) &&
          !plan->custom)
        return std::unexpected(Status::Unimplemented(
            "Apple custom operation needs a cold native lowering"));
      HYLE_ASSIGN_OR_RETURN(auto input_count, graph.node_input_count(id));
      for (std::size_t port = 0; port < input_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(id, port));
        node.inputs.push_back(value.ordinal());
        live[value.ordinal()] = true;
      }
      nodes[i - 1] = std::move(node);
    }
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      auto &node = nodes[index];
      if (!node)
        continue;
      if (report != nullptr)
        report->nodes.push_back(CompileNodeReport{
            .ordinal = static_cast<std::uint32_t>(index),
            .kind = node->operation.kind(),
            .traits = compute::operation_traits(node->operation),
        });
      plan->nodes.push_back(std::move(*node));
    }
    if (report != nullptr) {
      for (const auto input : plan->inputs)
        report->inputs.push_back(plan->descriptions[input]);
      for (const auto output : plan->outputs)
        report->outputs.push_back(plan->descriptions[output]);
    }
    for (const auto &node : plan->nodes) {
      const bool has_gguf = std::ranges::any_of(node.inputs, [&](auto id) {
        return compute::gguf_quantized_type(
                   plan->descriptions[id].storage_format())
            .has_value();
      });
      if (has_gguf && !ggml_model_operation(*plan, node))
        return std::unexpected(Status::Unimplemented(
            "Apple Graph GGUF carrier requires native MatMul or GroupedMatMul"));
    }
    const bool has_native_model_region =
        std::ranges::any_of(plan->nodes, [&](const auto &node) {
          return ggml_model_operation(*plan, node);
        });
    if (has_native_model_region) {
      if (report != nullptr) {
        report->implementation_policy =
            "Original ggml Metal model regions and maximal MLX "
            "compiled regions; fixed cold binding";
        report->numerical_policy +=
            "; provider-owned grouped expert/mHC/recurrent F32 regions, "
            "explicit non-mutating initial/final state";
        report->boundary_policy +=
            "; checked host completion at runtime boundaries; native MTLBuffer "
            "import; strided inputs use original MLX contiguous copy";
      }
      return compile_regions(context, std::move(plan));
    }
    return context.compile(
        [plan = std::move(plan)](const auto &inputs, mx::Stream stream) {
          return trace(*plan, plan->nodes, plan->inputs, plan->outputs, inputs,
                       stream);
        },
        specs, options.workload);
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Apple Graph preparation allocation failed"));
  } catch (const std::exception &error) {
    return std::unexpected(Status::Internal(error.what()));
  } catch (...) {
    return std::unexpected(Status::Internal("Apple Graph preparation failed"));
  }
}
Result<ca::Program> compile(const ca::Context &context, const Graph &graph,
                            std::span<const mx::array> constants,
                            CompileOptions options) noexcept {
  auto *report = options.report;
  if (report != nullptr)
    *report = {};
  auto program = compile_impl(context, graph, constants, std::move(options));
  if (report != nullptr) {
    if (!program) {
      *report = {};
      report->status = program.error();
    } else {
      report->status = Status::Ok();
    }
  }
  return program;
}

} // namespace hyle::graph::apple
