#include "hyle/graph/detail/region_access.h"
#include "plan_state.h"

namespace hyle::graph::detail {
Status RegionAccess::visit(const Executable&executable,RegionAccessFn callback,void*user)noexcept{
  auto*plan=ExecutableAccess::state(executable);
  if(!plan || !callback)return Status::InvalidArgument("region access requires an executable and callback");
  for(std::size_t i=0;i<plan->steps.size();++i){
    const auto&step=plan->steps[i];if(step.kind!=PlanStepKind::kRegion)continue;
    RegionAccessView view{std::uint32_t(i),plan->generation,std::span<const std::uint32_t>(plan->step_nodes).subspan(step.node_set_begin,step.node_count),
      &step.region.extension_,step.region.prepared_.state,nullptr};
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
      &step.region.extension_,step.region.prepared_.state,slot.state_};
    HYLE_RETURN_IF_ERROR(callback(user,view));
  }return Status::Ok();
}
} // namespace hyle::graph::detail
