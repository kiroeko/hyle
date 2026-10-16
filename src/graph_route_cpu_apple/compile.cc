#include "plan.h"
#include "../graph/native_analysis.h"
#include "../graph_batch/native_paged_nodes.h"
#include "hyle/compute/detail/apple_batch_access.h"

#include <algorithm>
#include <array>
#include <exception>
#include <new>

#include "hyle/compute/backends/cpu.h"

namespace hyle::graph::cpu_apple {
namespace {
using detail::Plan;
using detail::Vector;
namespace ca = compute::apple;

Result<compute::TensorDesc> carrier(compute::TensorDesc description) noexcept {
  if (description.is_plain())
    return compute::TensorDesc::dense(description.data_type(), description.dimensions());
  // Encoded values preserve their complete physical contract. GGUF [K,N]
  // weights are contiguous native byte carriers even when is_dense() is false
  // for the logical axes. The original native provider qualifies that layout.
  return description;
}

Status copy_constant(Plan &plan, detail::Value &value) noexcept {
  if (value.native_constant) return Status::Ok();
  auto source = value.constant;
  mem::Buffer normalized;
  compute::ResourceBudget::Reservation charge;
  if (source.description() != value.carrier) {
    HYLE_ASSIGN_OR_RETURN(auto builder, GraphBuilder::create({plan.allocator}));
    HYLE_ASSIGN_OR_RETURN(auto input, builder.input(source.description()));
    HYLE_ASSIGN_OR_RETURN(auto output, builder.add(compute::Copy{}, input));
    HYLE_RETURN_IF_ERROR(builder.publish(output));
    HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(auto compiled, graph::compile(graph, *plan.environment.cpu,
        {.resource_budget = plan.budget, .workload = plan.workload}));
    HYLE_ASSIGN_OR_RETURN(auto bytes, value.carrier.required_bytes());
    if (plan.budget) {
      HYLE_ASSIGN_OR_RETURN(charge, plan.budget.reserve(plan.environment.host, bytes, compute::ResourceBudgetCategory::kConstant));
    }
    HYLE_ASSIGN_OR_RETURN(normalized, mem::Buffer::create(plan.environment.host, bytes, value.carrier.required_alignment()));
    HYLE_ASSIGN_OR_RETURN(auto destination, compute::TensorView::create(value.carrier, normalized.view()));
    HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(run.bind(input, source));
    HYLE_RETURN_IF_ERROR(run.bind(output, destination));
    HYLE_ASSIGN_OR_RETURN(auto queue, plan.environment.cpu->create_queue());
    HYLE_ASSIGN_OR_RETURN(auto completion, run.submit(queue));
    HYLE_RETURN_IF_ERROR(completion.wait());
    source = destination.as_const();
  }
  HYLE_ASSIGN_OR_RETURN(auto native, plan.environment.native->allocate(value.carrier, plan.budget,
      compute::ResourceBudgetCategory::kConstant));
  HYLE_RETURN_IF_ERROR(plan.environment.native->copy_from_host(native, source));
  value.native_constant = std::move(native);
  return Status::Ok();
}

Result<std::shared_ptr<const Plan>> compile_plan(const Graph &graph, Environment environment,
                                                CompileOptions options, CompileReport *report) noexcept {
  if (!graph.valid() || environment.cpu == nullptr ||
      !environment.cpu->is_open() || environment.apple == nullptr || !environment.apple->valid() ||
      environment.native == nullptr || !environment.native->valid() || !environment.host.valid() ||
      environment.cpu->device().backend_id() != compute::cpu::kBackendId ||
      !environment.host.properties().has(mem::MemoryProperty::kHostAccessible) ||
      !options.workload.valid())
    return std::unexpected(Status::InvalidArgument("CPU+Apple requires live native endpoints and host storage"));
  if (graph.dimension_count() != 0)
    return std::unexpected(Status::Unimplemented("CPU+Apple admits fixed pure DAGs only"));
  auto allocator = options.allocator;
  if (options.resource_budget) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  if (!allocator.valid()) return std::unexpected(Status::InvalidArgument("invalid CPU+Apple metadata allocator"));
  try {
    auto plan = std::allocate_shared<Plan>(mem::StdAllocator<Plan>(allocator), allocator);
    plan->budget = options.resource_budget;
    plan->workload = options.workload;
    plan->environment = environment;
    HYLE_ASSIGN_OR_RETURN(plan->cpu_context, compute::detail::ContextRef::acquire(*environment.cpu));
    plan->values.resize(graph.value_count());
    auto describe = [&](ValueId id) -> Status {
      auto &value = plan->values[id.ordinal()];
      value.id = id;
      HYLE_ASSIGN_OR_RETURN(const auto &spec, graph.value_spec(id));
      HYLE_ASSIGN_OR_RETURN(value.description, spec.concrete_description());
      HYLE_ASSIGN_OR_RETURN(value.carrier, carrier(value.description));
      return Status::Ok();
    };
    for (std::size_t i = 0; i < graph.input_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.input_id(i));
      HYLE_RETURN_IF_ERROR(describe(id));
    }
    if (options.constants.size() != graph.constant_count())
      return std::unexpected(Status::InvalidArgument("CPU+Apple constant count differs"));
    for (std::size_t i = 0; i < graph.constant_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.constant_id(i));
      HYLE_RETURN_IF_ERROR(describe(id));
      const ConstantBinding *binding = nullptr;
      for (const auto &candidate : options.constants)
        if (candidate.value == id) {
          if (binding != nullptr) return std::unexpected(Status::InvalidArgument("duplicate CPU+Apple constant"));
          binding = &candidate;
        }
      auto &value = plan->values[id.ordinal()];
      if (binding == nullptr || !binding->tensor.valid() || binding->tensor.description() != value.description ||
          !binding->tensor.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
        return std::unexpected(Status::InvalidArgument("CPU+Apple constant ID/description/storage differs"));
      value.constant = binding->tensor;
      value.constant_policy = ConstantPolicy::kBorrow;
      if (binding->policy == ConstantPolicy::kMaterialize) {
        HYLE_ASSIGN_OR_RETURN(auto bytes, value.description.required_bytes());
        if (plan->budget) {
          HYLE_ASSIGN_OR_RETURN(value.constant_charge, plan->budget.reserve(environment.host, bytes,
              compute::ResourceBudgetCategory::kConstant));
        }
        HYLE_ASSIGN_OR_RETURN(value.constant_storage, mem::Buffer::create(environment.host, bytes,
            std::max(alignof(std::max_align_t), value.description.required_alignment())));
        HYLE_RETURN_IF_ERROR(value.constant_storage.view().copy_from(binding->tensor.storage()));
        HYLE_ASSIGN_OR_RETURN(value.constant, compute::ConstTensorView::create(value.description, value.constant_storage.view()));
      }
    }
    Vector<bool> live{mem::StdAllocator<bool>(allocator)};
    Vector<bool> needed{mem::StdAllocator<bool>(allocator)};
    Vector<std::size_t> last_use{mem::StdAllocator<std::size_t>(allocator)};
    Vector<ExecutionTargetId> targets{mem::StdAllocator<ExecutionTargetId>(allocator)};
    live.resize(graph.value_count()); needed.resize(graph.node_count());
    last_use.resize(graph.value_count()); targets.resize(graph.node_count(), kAppleTarget);
    for (std::size_t i = 0; i < options.node_placements.size(); ++i) {
      const auto &placement = options.node_placements[i];
      if (placement.target != kCpuTarget && placement.target != kAppleTarget)
        return std::unexpected(Status::InvalidArgument("unknown CPU+Apple execution target"));
      HYLE_ASSIGN_OR_RETURN(auto actual, graph.node_id(placement.node.ordinal()));
      if (placement.node != actual) return std::unexpected(Status::InvalidArgument("foreign CPU+Apple node placement"));
      for (std::size_t j = 0; j < i; ++j)
        if (options.node_placements[j].node == placement.node)
          return std::unexpected(Status::InvalidArgument("duplicate CPU+Apple node placement"));
      targets[placement.node.ordinal()] = placement.target;
    }
    graph::detail::NativeAnalysis::Flags cpu_nodes{mem::StdAllocator<std::uint8_t>(allocator)};
    cpu_nodes.resize(graph.node_count());
    for (std::size_t i = 0; i < targets.size(); ++i) cpu_nodes[i] = targets[i] == kCpuTarget;
    HYLE_ASSIGN_OR_RETURN(auto analysis, graph::detail::analyze_native_pure(graph, allocator, cpu_nodes));
    HYLE_RETURN_IF_ERROR(batch::detail::audit_native_paged_nodes(graph,options.paged_nodes,
        analysis.needed,targets,kAppleTarget));
    for(const auto& binding:options.paged_nodes)analysis.boundary[binding.node.ordinal()]=1;
    HYLE_RETURN_IF_ERROR(graph::detail::collect_effect_bindings(
        graph, options.effect_bindings, options.constants, allocator, plan->effect_bindings));
    for (std::size_t i = 0; i < graph.output_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.output_id(i));
      HYLE_RETURN_IF_ERROR(describe(id));
      live[id.ordinal()] = true;
      plan->values[id.ordinal()].published = true;
      last_use[id.ordinal()] = graph.node_count();
    }
    for (std::size_t i = graph.node_count(); i != 0; --i) {
      HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(i - 1));
      needed[i - 1] = analysis.needed[i - 1] != 0;
      HYLE_ASSIGN_OR_RETURN(auto count, graph.node_output_count(node));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto id, graph.result(node, port));
        HYLE_RETURN_IF_ERROR(describe(id));
        plan->values[id.ordinal()].producer = targets[i - 1];
        needed[i - 1] = needed[i - 1] || live[id.ordinal()];
      }
      if (!needed[i - 1]) continue;
      HYLE_ASSIGN_OR_RETURN(count, graph.node_input_count(node));
      for (std::size_t port = 0; port < count; ++port) {
        HYLE_ASSIGN_OR_RETURN(auto id, graph.node_input(node, port));
        live[id.ordinal()] = true;
        last_use[id.ordinal()] = std::max(last_use[id.ordinal()], i - 1);
        if (targets[i - 1] == kCpuTarget) plan->values[id.ordinal()].cpu_use = true;
        else plan->values[id.ordinal()].apple_use = true;
      }
    }
    for (std::size_t i = 0; i < graph.input_count(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto id, graph.input_id(i));
      const auto &value = plan->values[id.ordinal()];
      if (value.cpu_use || value.apple_use || value.published)
        plan->inputs.push_back(id.ordinal());
    }
    HYLE_ASSIGN_OR_RETURN(auto partition, graph::detail::partition_native_joint(
        graph, analysis, targets, allocator));
    for (const auto &range : partition.fragments) {
      const auto begin = range.begin, end = range.end;
      const auto target = targets[begin];
      const auto dependencies = partition.prerequisites(range);
      const auto page_binding=std::find_if(options.paged_nodes.begin(),options.paged_nodes.end(),
          [&](const auto& b){return b.node.ordinal()==begin;});
      if(page_binding!=options.paged_nodes.end()){
        detail::Fragment fragment(allocator);fragment.target=kAppleTarget;
        fragment.dependencies.assign(dependencies.begin(), dependencies.end());
        for(std::size_t port=0;port<3;++port){
          HYLE_ASSIGN_OR_RETURN(auto input,graph.node_input(page_binding->node,port));
          fragment.inputs.push_back(input.ordinal());
          if(input.kind()==ValueKind::kConstant)HYLE_RETURN_IF_ERROR(copy_constant(*plan,plan->values[input.ordinal()]));
        }
        HYLE_ASSIGN_OR_RETURN(auto output,graph.result(page_binding->node));
        fragment.outputs.push_back(output.ordinal());plan->values[output.ordinal()].fragment_output=true;
        if(plan->budget){HYLE_ASSIGN_OR_RETURN(fragment.private_storage,plan->budget.reserve_unattributed(std::nullopt));}
        HYLE_ASSIGN_OR_RETURN(auto native,compute::apple::detail::PagedBatchAccess::prepare_program(
            *environment.apple,page_binding->program,page_binding->layout,plan->workload));
        plan->paged.push_back({plan->fragments.size(),*page_binding,std::move(native)});
        if(report!=nullptr)report->fragments.push_back({.target=kAppleTarget,.node_count=1});
        plan->fragments.push_back(std::move(fragment));continue;
      }
      HYLE_ASSIGN_OR_RETURN(auto builder, GraphBuilder::create({allocator}));
      Vector<ValueId> mapped{mem::StdAllocator<ValueId>(allocator)};
      mapped.resize(graph.value_count());
      Vector<compute::TensorDesc> physical{mem::StdAllocator<compute::TensorDesc>(allocator)};
      physical.resize(graph.value_count());
      Vector<ConstantBinding> constants{mem::StdAllocator<ConstantBinding>(allocator)};
      std::vector<mlx::core::array> native_constants;
      Vector<EffectResourceBinding> effects{mem::StdAllocator<EffectResourceBinding>(allocator)};
      Vector<EffectResourceId> resources{mem::StdAllocator<EffectResourceId>(allocator)};
      resources.resize(graph.effect_resource_count());
      detail::Fragment fragment(allocator);
      fragment.target = target;
      fragment.dependencies.assign(dependencies.begin(), dependencies.end());
      std::size_t node_count = 0;
      for (auto index = begin; index < end; ++index) {
        if (!needed[index]) continue;
        ++node_count;
        HYLE_ASSIGN_OR_RETURN(auto node, graph.node_id(index));
        HYLE_ASSIGN_OR_RETURN(auto input_count, graph.node_input_count(node));
        Vector<ValueId> arguments{mem::StdAllocator<ValueId>(allocator)};
        Vector<compute::TensorDesc> input_descriptions{mem::StdAllocator<compute::TensorDesc>(allocator)};
        for (std::size_t port = 0; port < input_count; ++port) {
          HYLE_ASSIGN_OR_RETURN(auto original, graph.node_input(node, port));
          auto &value = plan->values[original.ordinal()];
          auto &local = mapped[original.ordinal()];
          if (!local.valid()) {
            const auto description = target == kAppleTarget ? value.carrier
                : original.kind() == ValueKind::kInput || original.kind() == ValueKind::kConstant
                    ? value.description : value.carrier;
            if (original.kind() == ValueKind::kConstant) {
              HYLE_ASSIGN_OR_RETURN(local, builder.constant(description));
              if (target == kCpuTarget) constants.push_back({local, value.constant, value.constant_policy});
              else {
                HYLE_RETURN_IF_ERROR(copy_constant(*plan, value));
                native_constants.push_back(*value.native_constant);
              }
            } else {
              HYLE_ASSIGN_OR_RETURN(local, builder.input(description));
              fragment.inputs.push_back(original.ordinal());
              fragment.cpu_inputs.push_back(local);
            }
          }
          if (physical[original.ordinal()].valid() == false) {
            physical[original.ordinal()] = target == kAppleTarget ? value.carrier
                : original.kind() == ValueKind::kInput || original.kind() == ValueKind::kConstant
                    ? value.description : value.carrier;
          }
          arguments.push_back(local);
          input_descriptions.push_back(physical[original.ordinal()]);
        }
        HYLE_ASSIGN_OR_RETURN(const auto &operation, graph.operation(node));
        HYLE_ASSIGN_OR_RETURN(auto local_node, builder.add_node(operation, arguments));
        HYLE_ASSIGN_OR_RETURN(auto resource_count, graph.node_resource_count(node));
        Vector<EffectResourceId> ports{mem::StdAllocator<EffectResourceId>(allocator)};
        for (std::size_t port = 0; port < resource_count; ++port) {
          HYLE_ASSIGN_OR_RETURN(auto original, graph.node_resource(node, port));
          auto &local = resources[original.ordinal()];
          if (!local.valid()) {
            HYLE_ASSIGN_OR_RETURN(local, builder.effect_resource());
            const auto binding = std::ranges::find_if(plan->effect_bindings,
                [&](const auto &entry) { return entry.resource == original; });
            if (binding == plan->effect_bindings.end())
              return std::unexpected(Status::Internal("preflight effect footprint is missing"));
            effects.push_back({local, binding->footprint});
          }
          ports.push_back(local);
        }
        if (!ports.empty()) HYLE_RETURN_IF_ERROR(builder.bind_resources(local_node, ports));

        HYLE_ASSIGN_OR_RETURN(auto output_count, graph.node_output_count(node));
        Vector<compute::TensorDesc> output_descriptions{mem::StdAllocator<compute::TensorDesc>(allocator)};
        output_descriptions.resize(output_count);
        HYLE_RETURN_IF_ERROR(compute::infer_output_descriptions(operation, input_descriptions, output_descriptions));
        for (std::size_t port = 0; port < output_count; ++port) {
          HYLE_ASSIGN_OR_RETURN(auto original, graph.result(node, port));
          physical[original.ordinal()] = output_descriptions[port];
          HYLE_ASSIGN_OR_RETURN(mapped[original.ordinal()], builder.result(local_node, port));
          if (last_use[original.ordinal()] < end && !analysis.control_predecessor[index]) continue;
          auto output = mapped[original.ordinal()];
          auto &value = plan->values[original.ordinal()];
          value.fragment_output = true;
          if (target == kCpuTarget && (physical[original.ordinal()] != value.carrier ||
                                       operation.kind() == compute::OperationKind::kView)) {
            HYLE_ASSIGN_OR_RETURN(output, builder.add(compute::Copy{}, output));
          }
          HYLE_RETURN_IF_ERROR(builder.publish(output));
          fragment.outputs.push_back(original.ordinal());
          fragment.cpu_outputs.push_back(output);
        }
      }
      HYLE_ASSIGN_OR_RETURN(auto native_graph, std::move(builder).finish());
      FragmentReport facts{.target = target, .node_count = node_count};
      if (target == kCpuTarget) {
        HYLE_ASSIGN_OR_RETURN(auto compiled, graph::compile(native_graph, *environment.cpu,
            {.constants = constants, .resource_budget = plan->budget, .effect_bindings = effects, .workload = plan->workload}));
        fragment.cpu = std::move(compiled.executable);
        facts.cpu = std::move(compiled.report);
        if (fragment.cpu.workspace_bytes() > environment.host_workspace_budget_bytes - plan->host_workspace_bytes)
          return std::unexpected(Status::ResourceExhausted("CPU+Apple CPU workspace exceeds limit"));
        plan->host_workspace_bytes += fragment.cpu.workspace_bytes();
      } else {
        if (plan->budget) {
          HYLE_ASSIGN_OR_RETURN(fragment.private_storage, plan->budget.reserve_unattributed(std::nullopt));
        }
        HYLE_ASSIGN_OR_RETURN(fragment.apple, apple::compile(*environment.apple, native_graph, native_constants,
            {.custom = options.custom, .report = report != nullptr ? &facts.apple : nullptr,
             .workload = plan->workload, .contiguous_outputs = true}));
      }
      if (report != nullptr) report->fragments.push_back(std::move(facts));
      plan->fragments.push_back(std::move(fragment));
    }
    for (const auto &fragment : plan->fragments) {
      using Lane = graph::detail::JointPipelinePlan::Lane;
      plan->pipeline.append(fragment.target == kCpuTarget ? Lane::kCpu : Lane::kNative,
          fragment.dependencies);
    }
    HYLE_RETURN_IF_ERROR(plan->pipeline.prepare());
    // Only strided host graph inputs crossing into Apple need an additional
    // prepared CPU Copy. Ordinary CPU-only inputs keep their original layout.
    for (auto &value : plan->values) {
      if (value.id.kind() != ValueKind::kInput || !value.apple_use || value.description == value.carrier) continue;
      HYLE_ASSIGN_OR_RETURN(auto builder, GraphBuilder::create({allocator}));
      HYLE_ASSIGN_OR_RETURN(value.repack_input, builder.input(value.description));
      HYLE_ASSIGN_OR_RETURN(value.repack_output, builder.add(compute::Copy{}, value.repack_input));
      HYLE_RETURN_IF_ERROR(builder.publish(value.repack_output));
      HYLE_ASSIGN_OR_RETURN(auto repack_graph, std::move(builder).finish());
      HYLE_ASSIGN_OR_RETURN(auto compiled, graph::compile(repack_graph, *environment.cpu,
          {.resource_budget = plan->budget, .workload = plan->workload}));
      value.repack = std::move(compiled.executable);
      if (value.repack.workspace_bytes() > environment.host_workspace_budget_bytes - plan->host_workspace_bytes)
        return std::unexpected(Status::ResourceExhausted("CPU+Apple repack workspace exceeds limit"));
      plan->host_workspace_bytes += value.repack.workspace_bytes();
    }
    if (report != nullptr) {
      report->host_workspace_bytes = plan->host_workspace_bytes;
      report->has_unknown_native_storage = std::ranges::any_of(plan->fragments, [](const auto &f) { return f.target == kAppleTarget; });
    }
    return std::shared_ptr<const Plan>(std::move(plan));
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("CPU+Apple preparation metadata allocation failed"));
  } catch (const std::exception &error) {
    return std::unexpected(Status::Internal(error.what()));
  } catch (...) {
    return std::unexpected(Status::Internal("CPU+Apple preparation failed"));
  }
}
} // namespace

Result<Executable> compile(const Graph &graph, Environment environment,
                           CompileOptions options, CompileReport *report) noexcept {
  if (report != nullptr) *report = {};
  auto plan = [&]() -> Result<std::shared_ptr<const Plan>> {
    if (graph.dimension_count() == 0 && options.shape.empty())
      return compile_plan(graph, environment, std::move(options), report);
    auto allocator = options.allocator;
    if (options.resource_budget) {
      HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
    }
    HYLE_ASSIGN_OR_RETURN(auto exact, specialize_exact(graph, options.shape, allocator));
    options.shape = {};
    return compile_plan(exact.graph, environment, std::move(options), report);
  }();
  if (!plan) {
    if (report != nullptr) *report = {};
    return std::unexpected(std::move(plan).error());
  }
  return Executable(std::move(*plan));
}
} // namespace hyle::graph::cpu_apple
