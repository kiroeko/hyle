#include "hyle/graph/apple.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
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
  std::vector<Node> nodes;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> constants;
  std::vector<std::size_t> outputs;
  std::vector<compute::TensorDesc> descriptions;
  std::vector<mx::array> values;
  CustomLowering custom;
};

Result<ca::InputSpec> native_spec(const compute::TensorDesc &description) {
  if (!description.valid() || !description.is_dense() ||
      description.data_type() != compute::DataType::kFloat32)
    return std::unexpected(Status::Unimplemented(
        "Apple Graph requires dense F32 logical tensors"));
  ca::InputSpec result;
  for (auto dimension : description.dimensions()) {
    if (dimension <= 0 || dimension > (std::numeric_limits<int>::max)())
      return std::unexpected(Status::InvalidArgument(
          "Apple Graph dimension is outside native range"));
    result.shape.push_back(static_cast<int>(dimension));
  }
  if (!description.is_plain()) {
    if (description.rank() != 2 ||
        description.storage_format().id() != ca::kAffineQ4Format)
      return std::unexpected(Status::Unimplemented(
          "Apple Graph storage encoding is not admitted"));
    HYLE_ASSIGN_OR_RETURN(
        auto expected, ca::affine_q4_description(description.dimensions()[0],
                                                 description.dimensions()[1]));
    if (description != expected)
      return std::unexpected(Status::InvalidArgument(
          "Apple Q4 layout disagrees with native packing"));
    result.shape.back() /= 8;
    result.dtype = mx::uint32;
  }
  return result;
}

Status match(const compute::TensorDesc &description, const mx::array &value) {
  HYLE_ASSIGN_OR_RETURN(auto spec, native_spec(description));
  if (spec.shape != value.shape() || spec.dtype != value.dtype())
    return Status::InvalidArgument(
        "Apple Graph native array shape/dtype disagrees with schema");
  return Status::Ok();
}

Result<std::vector<mx::array>> trace(const Plan &plan,
                                     const std::vector<mx::array> &inputs,
                                     mx::Stream stream) {
  // This storage exists only during cold MLX tracing. Hot runs invoke its
  // compiled artifact; they never allocate or interpret this HYLE value table.
  std::vector<std::optional<mx::array>> values(plan.descriptions.size());
  for (std::size_t i = 0; i < inputs.size(); ++i)
    values[plan.inputs[i]] = inputs[i];
  for (std::size_t i = 0; i < plan.constants.size(); ++i)
    values[plan.constants[i]] = plan.values[i];
  std::vector<mx::array> arguments;
  for (const auto &node : plan.nodes) {
    arguments.clear();
    for (auto index : node.inputs)
      arguments.push_back(*values[index]);
    auto outputs = node.operation.kind() == compute::OperationKind::kCustom
                       ? plan.custom(node.operation, arguments, stream)
                       : ca::lower(node.operation, arguments, stream);
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
  result.reserve(plan.outputs.size());
  for (auto index : plan.outputs)
    result.push_back(std::move(*values[index]));
  return result;
}
} // namespace

Result<ca::Program> compile(const ca::Context &context, const Graph &graph,
                            std::span<const mx::array> constants,
                            CompileOptions options) noexcept {
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
    auto plan = std::make_shared<Plan>();
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
      if (operation.kind() == compute::OperationKind::kCustom && !plan->custom)
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
    for (auto &node : nodes) {
      if (node)
        plan->nodes.push_back(std::move(*node));
    }
    return context.compile(
        [plan = std::move(plan)](const auto &inputs, mx::Stream stream) {
          return trace(*plan, inputs, stream);
        },
        specs);
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Apple Graph preparation allocation failed"));
  } catch (const std::exception &error) {
    return std::unexpected(Status::Internal(error.what()));
  } catch (...) {
    return std::unexpected(Status::Internal("Apple Graph preparation failed"));
  }
}
} // namespace hyle::graph::apple
