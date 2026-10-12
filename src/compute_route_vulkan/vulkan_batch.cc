#include "hyle/compute/vulkan_batch.h"
#include "hyle/compute/detail/vulkan_plan.h"

#include <new>

namespace hyle::compute::vulkan {
Result<Program> prepare_materialized_batch(const Context&context,
    const MaterializedBatchProgram&program,const WorkloadContext&workload)noexcept{
  if(!context.valid() || !program.valid())return std::unexpected(Status::InvalidArgument("Vulkan materialized batch context/program is empty"));
  try {
    detail::Plan plan;plan.workload=workload;plan.inputs={0,1,2,3,4};plan.outputs={program.output_value()};
    plan.values.reserve(program.descriptions().size());plan.steps.reserve(program.steps().size());
    for(const auto&description:program.descriptions())plan.values.push_back({description,{}});
    for(const auto&step:program.steps()){
      detail::PlanStep native;native.operation=step.operation;native.outputs={step.output};
      native.inputs.assign(step.inputs.begin(),step.inputs.begin()+step.input_count);
      plan.steps.push_back(std::move(native));
    }
    return detail::prepare_plan(context,plan);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("Vulkan materialized batch cold recipe metadata"));}
}
} // namespace hyle::compute::vulkan
