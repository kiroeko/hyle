#include "hyle/graph/detail/region_access.h"
#include "plan_state.h"

namespace hyle::graph::detail {
Status RegionAccess::validate_queue(const Executable&executable,ExecutionLaneId lane,const compute::Queue&queue)noexcept{
  const auto*plan=ExecutableAccess::state(executable);
  if(!plan || !lane.valid() || lane.ordinal()>=plan->lanes.size() ||
     !plan->targets[plan->lanes[lane.ordinal()].target.ordinal()].context.owns(queue))
    return Status::InvalidArgument("Graph adapter Queue belongs to a different execution target");
  return Status::Ok();
}
Status RegionAccess::visit(const Executable&executable,RegionAccessFn callback,void*user)noexcept{
  auto*plan=ExecutableAccess::state(executable);
  if(!plan || !callback)return Status::InvalidArgument("region access requires an executable and callback");
  for(std::size_t i=0;i<plan->steps.size();++i){
    const auto&step=plan->steps[i];if(step.kind!=PlanStepKind::kRegion)continue;
    RegionAccessView view{std::uint32_t(i),plan->generation,std::span<const std::uint32_t>(plan->step_nodes).subspan(step.node_set_begin,step.node_count),
      &step.region.extension_,step.region.prepared_.state,nullptr,step.lane};
    HYLE_RETURN_IF_ERROR(callback(user,view));
  }return Status::Ok();
}
Status RegionAccess::visit(Invocation&invocation,RegionAccessFn callback,void*user)noexcept{
  auto*state=InvocationAccess::state(invocation);
  if(!state || !callback)return Status::InvalidArgument("region access requires an invocation and callback");
  if(state->phase.load(std::memory_order_acquire)!=InvocationPhase::kIdle)
    return Status::FailedPrecondition("region access requires an idle successful invocation");
  auto*plan=state->plan;
  for(std::size_t i=0;i<plan->steps.size();++i){
    const auto&step=plan->steps[i];if(step.kind!=PlanStepKind::kRegion)continue;
    const auto&slot=state->invocations[i].region;
    RegionAccessView view{std::uint32_t(i),plan->generation,std::span<const std::uint32_t>(plan->step_nodes).subspan(step.node_set_begin,step.node_count),
      &step.region.extension_,step.region.prepared_.state,slot.state_,step.lane};
    HYLE_RETURN_IF_ERROR(callback(user,view));
  }return Status::Ok();
}
} // namespace hyle::graph::detail
