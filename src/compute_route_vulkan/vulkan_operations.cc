#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <new>
#include <utility>

namespace hyle::compute::vulkan::detail {

Status lower_operation(ProgramState &program,
                       const PlanStep &step) noexcept {
  try {
    if (managed_elementwise_limits(program, step))
      return lower_managed_elementwise(program, step);
    if (step.operation.kind() == OperationKind::kReduce ||
        step.operation.kind() == OperationKind::kSoftmax)
      return lower_managed_reduction(program, step);
    if (step.operation.kind() == OperationKind::kLayerNorm)
      return lower_layer_norm(program, step);
    if (step.operation.kind() == OperationKind::kTopK)
      return lower_topk(program, step);
    if (step.operation.kind() == OperationKind::kGather)
      return lower_gather(program, step);
    if (step.operation.kind() == OperationKind::kScan)
      return lower_scan(program, step);
    if (step.operation.kind() == OperationKind::kScatter &&
        mature_scatter_add_operation(program, step))
      return lower_mature_scatter_add(program, step);
    if (step.operation.kind() == OperationKind::kScatter)
      return lower_scatter(program, step);
    if (step.outputs.size() != 1)
      return Status::Unimplemented("Vulkan native operation output arity");
    if (step.operation.kind() == OperationKind::kCustom) {
      const auto &pipeline = Access::pipeline(step.custom);
      if (!pipeline)
        return Status::Unimplemented(
            "Vulkan custom schema needs a cold native pipeline");
      if (pipeline->device != program.runtime->device ||
          pipeline->bindings != step.inputs.size() + 1 ||
          pipeline->push_bytes != step.push_constants.size())
        return Status::InvalidArgument(
            "Vulkan custom pipeline device or argument layout");
      for (std::size_t axis = 0; axis < 3; ++axis)
        if (step.groups[axis] == 0 ||
            step.groups[axis] >
                program.runtime->device->properties.limits
                    .maxComputeWorkGroupCount[axis])
          return Status::InvalidArgument(
              "Vulkan custom dispatch group count");
      Dispatch dispatch;
      dispatch.pipeline = pipeline;
      dispatch.values = step.inputs;
      dispatch.values.push_back(step.outputs[0]);
      dispatch.groups = step.groups;
      dispatch.push = step.push_constants;
      program.dispatches.push_back(std::move(dispatch));
      return Status::Ok();
    }
    return lower_tensor(program, step);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted(
        "Vulkan typed operation metadata");
  }
}

} // namespace hyle::compute::vulkan::detail
