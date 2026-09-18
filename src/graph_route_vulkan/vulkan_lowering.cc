#include "hyle/graph/vulkan.h"

#include "hyle/obs/trace.h"

#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <utility>

#include "hyle/compute/detail/vulkan_plan.h"

namespace hyle::graph::vulkan {
namespace cv = compute::vulkan;

static Result<cv::Program> compile_impl(const cv::Context &context,
                                        const Graph &graph,
                                        std::span<const cv::Tensor> constants,
                                        CompileOptions options) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.graph.vulkan.compile");
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
    auto *report = options.report;
    if (report != nullptr) {
      report->environment = context.environment();
      report->implementation_policy =
          "Fixed managed ggml dense F32 norm/gate/pointwise/TopK regions and "
          "native tensor "
          "shaders plus cold custom "
          "SPIR-V; "
          "device feature/limit matching, no runtime timing or provider search";
      report->numerical_policy =
          "F32/F16/BF16 tensor arithmetic with F32 evaluation and storage RNE; "
          "I32/I64/Bool exact storage operations; "
          "custom nodes retain their declared schema contract";
      report->boundary_policy =
          "Ready same-device inputs/constants; cold weight packing; "
          "explicit compute visibility barriers; no host output copy; "
          "imported buffers require a completed visibility handoff";
      report->concurrency_policy =
          "Each Invocation owns native commands, managed work and mutable "
          "storage; admission and provider retirement are serialized on the "
          "Context while normal fence waits permit independent producers; "
          "retained results keep their slot until released";
    }
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
      if ((operation.kind() == compute::OperationKind::kCustom &&
           output_count != 1) ||
          (traits.alias == compute::AliasPolicy::kMustAliasInput0 &&
           operation.kind() != compute::OperationKind::kView))
        return std::unexpected(
            Status::Unimplemented("Vulkan Graph requires supported alias "
                                  "semantics and single-output custom nodes"));
      cv::detail::PlanStep step;
      step.operation = operation;
      for (std::size_t port = 0; port < output_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto output, graph.result(id, port));
        step.outputs.push_back(output.ordinal());
      }
      HYLE_ASSIGN_OR_RETURN(auto input_count, graph.node_input_count(id));
      for (std::size_t port = 0; port < input_count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto value, graph.node_input(id, port));
        step.inputs.push_back(value.ordinal());
        live[value.ordinal()] = true;
      }
      nodes[index - 1] = std::move(step);
    }
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      auto &node = nodes[index];
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
            auto native,
            options.custom(context, node->operation, descriptions,
                           plan.values[node->outputs[0]].description));
        node->custom = std::move(native.kernel);
        node->groups = native.groups;
        node->push_constants = std::move(native.push_constants);
      }
      if (report != nullptr)
        report->nodes.push_back(CompileNodeReport{
            .ordinal = static_cast<std::uint32_t>(index),
            .kind = node->operation.kind(),
            .traits = compute::operation_traits(node->operation),
        });
      plan.steps.push_back(std::move(*node));
    }
    cv::detail::PlanFacts facts;
    auto program = cv::detail::prepare_plan(
        context, plan, report != nullptr ? &facts : nullptr);
    if (!program)
      return program;
    if (report != nullptr) {
      for (const auto input : plan.inputs)
        report->inputs.push_back(plan.values[input].description);
      for (const auto output : plan.outputs)
        report->outputs.push_back(plan.values[output].description);
      report->native_dispatch_count = facts.native_dispatch_count;
      report->managed_region_count = facts.managed_region_count;
      report->invocation_storage_block_bytes =
          std::move(facts.invocation_storage_block_bytes);
      report->retained_constant_owner_count =
          facts.retained_constant_owner_count;
      report->retained_constant_capacity_bytes =
          facts.retained_constant_capacity_bytes;
      report->owned_constant_allocation_bytes =
          facts.owned_constant_allocation_bytes;
      report->has_borrowed_constants = facts.has_borrowed_constants;
    }
    return program;
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

Result<cv::Program> compile(const cv::Context &context, const Graph &graph,
                            std::span<const cv::Tensor> constants,
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

} // namespace hyle::graph::vulkan
