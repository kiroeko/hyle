#include "hyle/graph/vulkan.h"

#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <utility>

#include "hyle/compute/detail/vulkan_plan.h"

namespace hyle::graph::vulkan {
namespace cv = compute::vulkan;

Result<cv::Program> compile(const cv::Context &context, const Graph &graph,
                            std::span<const cv::Tensor> constants,
                            CompileOptions options) noexcept {
  if (!context || !graph || graph.output_count() == 0 ||
      graph.value_count() > (std::numeric_limits<std::uint32_t>::max)() / 2)
    return std::unexpected(Status::InvalidArgument(
        "Vulkan Graph needs a Context and published outputs"));
  if (graph.dimension_count() != 0 || graph.effect_resource_count() != 0)
    return std::unexpected(
        Status::Unimplemented("Vulkan Graph admits fixed, pure regions only"));
  if (constants.size() != graph.constant_count())
    return std::unexpected(
        Status::InvalidArgument("Vulkan Graph constant count mismatch"));
  try {
    cv::detail::Plan plan;
    plan.values.resize(graph.value_count());
    auto describe = [&](ValueId id) -> Status {
      HYLE_ASSIGN_OR_RETURN(const auto &spec, graph.value_spec(id));
      HYLE_ASSIGN_OR_RETURN(auto description, spec.concrete_description());
      plan.values[id.ordinal()].description = description;
      return Status::Ok();
    };
    for (std::size_t index = 0; index < graph.input_count(); ++index) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.input_id(index));
      HYLE_RETURN_IF_ERROR(describe(id));
      plan.inputs.push_back(id.ordinal());
    }
    for (std::size_t index = 0; index < constants.size(); ++index) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.constant_id(index));
      HYLE_RETURN_IF_ERROR(describe(id));
      if (!constants[index])
        return std::unexpected(Status::InvalidArgument(
            "Vulkan Graph constant has no native owner"));
      plan.values[id.ordinal()].constant = constants[index];
    }
    std::vector<bool> live(graph.value_count(), false);
    for (std::size_t index = 0; index < graph.output_count(); ++index) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.output_id(index));
      plan.outputs.push_back(id.ordinal());
      live[id.ordinal()] = true;
    }
    // GraphBuilder already fixes topological order. Remove dead pure nodes
    // cold; reject explicit effects even when no published tensor uses them.
    std::vector<std::optional<cv::detail::PlanStep>> nodes(graph.node_count());
    for (std::size_t index = graph.node_count(); index > 0; --index) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.node_id(index - 1));
      HYLE_ASSIGN_OR_RETURN(const auto &operation, graph.operation(id));
      HYLE_ASSIGN_OR_RETURN(auto dependencies, graph.node_dependency_count(id));
      const auto traits = compute::operation_traits(operation);
      if (traits.effect != compute::EffectKind::kPure ||
          traits.resource_count != 0 || dependencies != 0)
        return std::unexpected(
            Status::Unimplemented("Vulkan Graph cannot lower effects or "
                                  "explicit control dependencies"));
      HYLE_ASSIGN_OR_RETURN(auto output_count, graph.node_output_count(id));
      bool needed = false;
      for (std::size_t port = 0; port < output_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.result(id, port));
        HYLE_RETURN_IF_ERROR(describe(value));
        needed = needed || live[value.ordinal()];
      }
      if (!needed)
        continue;
      if (output_count != 1 || traits.alias != compute::AliasPolicy::kNone)
        return std::unexpected(Status::Unimplemented(
            "Vulkan Graph requires non-aliasing single-output nodes"));
      cv::detail::PlanStep step;
      step.operation = operation;
      HYLE_ASSIGN_OR_RETURN(auto output, graph.result(id, 0));
      step.output = output.ordinal();
      HYLE_ASSIGN_OR_RETURN(auto input_count, graph.node_input_count(id));
      for (std::size_t port = 0; port < input_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(id, port));
        step.inputs.push_back(value.ordinal());
        live[value.ordinal()] = true;
      }
      nodes[index - 1] = std::move(step);
    }
    for (auto &node : nodes) {
      if (!node)
        continue;
      if (node->operation.kind() == compute::OperationKind::kCustom) {
        if (!options.custom)
          return std::unexpected(Status::Unimplemented(
              "Vulkan custom schema needs a cold native lowering"));
        std::vector<compute::TensorDesc> descriptions;
        descriptions.reserve(node->inputs.size());
        for (const auto input : node->inputs)
          descriptions.push_back(plan.values[input].description);
        HYLE_ASSIGN_OR_RETURN(
            auto native, options.custom(context, node->operation, descriptions,
                                        plan.values[node->output].description));
        node->custom = std::move(native.kernel);
        node->groups = native.groups;
        node->push_constants = std::move(native.push_constants);
      }
      plan.steps.push_back(std::move(*node));
    }
    return cv::detail::prepare_plan(context, plan);
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted(
        "Vulkan Graph preparation allocation failed"));
  } catch (const std::exception &error) {
    return std::unexpected(Status::Internal(error.what()));
  } catch (...) {
    return std::unexpected(
        Status::Internal("Vulkan Graph custom lowering failed"));
  }
}

} // namespace hyle::graph::vulkan
