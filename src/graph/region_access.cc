#include "hyle/graph/detail/region_access.h"
#include <algorithm>
#include "../compute/provider_internal.h"
#include "plan_state.h"

namespace hyle::graph::detail {
bool RegionAccess::preserves_capture_bindings(
    const RegionInvocation &invocation) noexcept {
  if (!invocation.valid() || !invocation.prepared_->valid() || !invocation.state_)
    return false;
  const auto &prepared = invocation.prepared_->prepared_;
  const auto *table = prepared.vtable->query_extension
      ? static_cast<const RegionCaptureBindingsVTable *>(
          prepared.vtable->query_extension(prepared.state,
              kRegionCaptureBindingsExtensionId))
      : nullptr;
  return table && table->preserves_private_arguments &&
      table->preserves_private_arguments(prepared.state, invocation.state_);
}

Result<RegionNodeAccess>
RegionAccess::inspect_projection_node(const PlanState &plan,
    const InvocationState *invocation, NodeId node,
    compute::ProviderExtensionId capability) noexcept {
  if (!capability.valid() || GraphIdentity::generation(node) != plan.generation ||
      node.ordinal() >= plan.node_steps.size())
    return std::unexpected(Status::InvalidArgument("projection access node is foreign"));
  const auto index = plan.node_steps[node.ordinal()];
  if (index >= plan.steps.size())
    return std::unexpected(Status::Unimplemented("projection node has no selected implementation"));
  const auto &step = plan.steps[index];
  if (step.kind == PlanStepKind::kOperation && step.node_count == 1 &&
      step.node_begin == node.ordinal()) {
    HYLE_ASSIGN_OR_RETURN(auto access,
        compute::detail::OperationPreparationAccess::inspect_access(step.operation, capability));
    // Preserve the original typed shared-Queue pending admission contract.
    return RegionNodeAccess{access, true};
  }
  if (step.kind != PlanStepKind::kRegion || !step.region.valid())
    return std::unexpected(Status::Unimplemented("projection node has no selected access implementation"));
  if (step.node_set_begin > plan.step_nodes.size() ||
      step.node_count > plan.step_nodes.size() - step.node_set_begin)
    return std::unexpected(Status::Internal("selected region node set is outside the plan"));
  const auto nodes = std::span<const std::uint32_t>(plan.step_nodes)
      .subspan(step.node_set_begin, step.node_count);
  if (std::ranges::find(nodes, node.ordinal()) == nodes.end())
    return std::unexpected(Status::InvalidArgument("projection node is outside its selected region"));
  void *slot = nullptr;
  if (invocation) {
    if (invocation->plan != &plan || index >= invocation->invocations.size() ||
        !invocation->invocations[index].region.valid())
      return std::unexpected(Status::InvalidArgument("projection slot differs from its selected region"));
    slot = invocation->invocations[index].region.state_;
  }
  struct Call final {
    const ProviderRegionPrepared *prepared;
    void *slot;
    std::uint32_t node;
    compute::ProviderExtensionId capability;
    RegionNodeAccess facts{};
  } call{&step.region.prepared_, slot, node.ordinal(), capability, {}};
  HYLE_RETURN_IF_ERROR(step.region.source_.call(
      [](void *, const void *, void *opaque) noexcept -> Status {
        auto &call = *static_cast<Call *>(opaque);
        const auto &prepared = *call.prepared;
        const auto *table = prepared.vtable->query_extension
            ? static_cast<const RegionProjectedReadVTable *>(
                prepared.vtable->query_extension(prepared.state, kRegionProjectedReadExtensionId))
            : nullptr;
        if (!table || !table->inspect_node)
          return Status::Unimplemented("selected region has no logical access inspector");
        HYLE_RETURN_IF_ERROR(table->inspect_node(prepared, call.slot,
            call.node, call.capability, &call.facts));
        if (call.facts.access.requires_configuration && !table->configure_node)
          return Status::Internal("selected region layout capability has no configuration callback");
        return Status::Ok();
      }, &call));
  if (call.facts.access.projected_input != compute::detail::OperationPreparedAccess::kNoInput &&
      call.facts.access.projected_input >= step.input_count)
    return std::unexpected(Status::Internal("region projection port exceeds its boundary inputs"));
  return call.facts;
}

Result<bool> RegionAccess::configure_node(Invocation&invocation,NodeId node,
    compute::ProviderExtensionId extension,const void*configuration)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!state || !configuration || !extension.valid() ||
      GraphIdentity::generation(node)!=state->plan->generation || node.ordinal()>=state->plan->node_steps.size())
    return std::unexpected(Status::InvalidArgument("cold operation node is foreign"));
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  const auto index=state->plan->node_steps[node.ordinal()];
  HYLE_ASSIGN_OR_RETURN(auto access,
      inspect_projection_node(*state->plan, state, node, extension));
  const auto &step = state->plan->steps[index];
  if (!access.access.requires_configuration) return false;
  // The selected callback may replace private plans before returning failure.
  // No old recording may retain those resources across that mutation.
  invalidate_step_recording(*state, index);
  if (step.kind == PlanStepKind::kOperation)
    return compute::detail::OperationInvocationAccess::configure_cold(
        state->invocations[index].operation, extension, configuration);
  struct Call final {
    const ProviderRegionPrepared *prepared;
    void *slot;
    std::uint32_t node;
    compute::ProviderExtensionId capability;
    const void *configuration;
    bool configured = false;
  } call{&step.region.prepared_, state->invocations[index].region.state_,
      node.ordinal(), extension, configuration, false};
  HYLE_RETURN_IF_ERROR(step.region.source_.call(
      [](void *, const void *, void *opaque) noexcept -> Status {
        auto &call = *static_cast<Call *>(opaque);
        const auto &prepared = *call.prepared;
        const auto *table = prepared.vtable->query_extension
            ? static_cast<const RegionProjectedReadVTable *>(
                prepared.vtable->query_extension(prepared.state, kRegionProjectedReadExtensionId))
            : nullptr;
        if (!table || !table->configure_node)
          return Status::Unimplemented("selected region has no logical access configuration");
        HYLE_ASSIGN_OR_RETURN(call.configured, table->configure_node(prepared,
            call.slot, call.node, call.capability, call.configuration));
        if (!call.configured)
          return Status::Internal("selected region did not apply its required configuration");
        return Status::Ok();
      }, &call));
  return call.configured;
}
Status RegionAccess::invalidate_commands(Invocation &invocation,
    std::uint32_t selected_step) noexcept {
  auto *state = InvocationAccess::state(invocation);
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  if (selected_step >= state->plan->steps.size())
    return Status::InvalidArgument("selected command step is outside this invocation");
  invalidate_step_recording(*state, selected_step);
  return Status::Ok();
}
Status RegionAccess::validate_queue(const Executable&executable,ExecutionLaneId lane,const compute::Queue&queue)noexcept{
  const auto*plan=ExecutableAccess::state(executable);
  if(!plan || !lane.valid() || lane.ordinal()>=plan->lanes.size() ||
     !ExecutionSourceAccess::owns(
         plan->targets[plan->lanes[lane.ordinal()].target.ordinal()].source,queue))
    return Status::InvalidArgument("Graph adapter Queue belongs to a different execution target");
  return Status::Ok();
}
Result<ExecutionSource> RegionAccess::execution_source(
    const Executable &executable, ExecutionLaneId lane) noexcept {
  const auto *plan = ExecutableAccess::state(executable);
  if (!plan || !lane.valid() || lane.ordinal() >= plan->lanes.size())
    return std::unexpected(Status::InvalidArgument("execution source lane is foreign"));
  const auto target = plan->lanes[lane.ordinal()].target;
  if (!target.valid() || target.ordinal() >= plan->targets.size() ||
      !plan->targets[target.ordinal()].source.valid())
    return std::unexpected(Status::Internal("execution lane has no retained source"));
  return plan->targets[target.ordinal()].source;
}
Result<mem::MemoryResourceRef> RegionAccess::input_resource(
    const Executable &executable, ValueId value) noexcept {
  const auto *plan = ExecutableAccess::state(executable);
  if (!plan || !value.valid() || value.kind() != ValueKind::kInput ||
      GraphIdentity::generation(value) != plan->generation ||
      value.ordinal() >= plan->binding_roles.size() ||
      !(plan->binding_roles[value.ordinal()] & kExternalInput) ||
      (plan->binding_roles[value.ordinal()] & kExternalOutput))
    return std::unexpected(Status::InvalidArgument("configuration value is not a private external input"));
  const auto ordinal = value.ordinal();
  if (ordinal >= plan->external_instances.size() || ordinal >= plan->external_storage.size())
    return std::unexpected(Status::Internal("graph input has no external storage mapping"));
  const auto instance = plan->external_instances[ordinal];
  const auto storage = plan->external_storage[ordinal];
  if (instance >= plan->instance_storage.size() ||
      plan->instance_storage[instance] != storage || storage >= plan->storage_domains.size())
    return std::unexpected(Status::Internal("graph input physical storage mapping is invalid"));
  return plan->storage_domains[storage].resource;
}
Status RegionAccess::visit(const Executable&executable,RegionAccessFn callback,void*user)noexcept{
  auto*plan=ExecutableAccess::state(executable);
  if(!plan || !callback)return Status::InvalidArgument("region access requires an executable and callback");
  for(std::size_t i=0;i<plan->steps.size();++i){
    const auto&step=plan->steps[i];if(step.kind!=PlanStepKind::kRegion)continue;
    RegionAccessView view{std::uint32_t(i),plan->generation,std::span<const std::uint32_t>(plan->step_nodes).subspan(step.node_set_begin,step.node_count),
      step.region.source_.shared_extension(),step.region.prepared_.state,nullptr,step.lane,
      {},{},step.region.prepared_.vtable};
    HYLE_RETURN_IF_ERROR(callback(user,view));
  }return Status::Ok();
}
Status RegionAccess::visit(Invocation&invocation,RegionAccessFn callback,void*user)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!state || !callback)return Status::InvalidArgument("region access requires an invocation and callback");
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  auto*plan=state->plan;
  for(std::size_t i=0;i<plan->steps.size();++i){
    const auto&step=plan->steps[i];if(step.kind!=PlanStepKind::kRegion)continue;
    const auto&slot=state->invocations[i].region;
    RegionAccessView view{std::uint32_t(i),plan->generation,std::span<const std::uint32_t>(plan->step_nodes).subspan(step.node_set_begin,step.node_count),
      step.region.source_.shared_extension(),step.region.prepared_.state,slot.state_,step.lane,
      {},{},step.region.prepared_.vtable};
    HYLE_RETURN_IF_ERROR(callback(user,view));
  }return Status::Ok();
}
Status RegionAccess::visit(Invocation&invocation,NodeId node,RegionAccessFn callback,void*user)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!state || !callback)return Status::InvalidArgument("region access requires an invocation and callback");
  HYLE_RETURN_IF_ERROR(resource_control_ready(state));
  const auto&plan=*state->plan;
  if(GraphIdentity::generation(node)!=plan.generation || node.ordinal()>=plan.node_steps.size())
    return Status::InvalidArgument("region access node is foreign");
  const auto index=plan.node_steps[node.ordinal()];
  if(index>=plan.steps.size())return Status::Unimplemented("node has no selected region");
  const auto&step=plan.steps[index];
  if(step.kind!=PlanStepKind::kRegion || !step.region.valid())return Status::Unimplemented("node has no selected region");
  if(step.node_set_begin>plan.step_nodes.size() || step.node_count>plan.step_nodes.size()-step.node_set_begin ||
      index>=state->invocations.size() || !state->invocations[index].region.valid())
    return Status::Internal("selected region node set or slot is invalid");
  const auto nodes=std::span<const std::uint32_t>(plan.step_nodes).subspan(step.node_set_begin,step.node_count);
  if(std::ranges::find(nodes,node.ordinal())==nodes.end())return Status::InvalidArgument("node is outside its selected region");
  RegionAccessView view{index,plan.generation,nodes,step.region.source_.shared_extension(),
    step.region.prepared_.state,state->invocations[index].region.state_,step.lane,{}, {},step.region.prepared_.vtable};
  return callback(user,view);
}
} // namespace hyle::graph::detail
