#include "vulkan_program_internal.h"
#include "vulkan_carrier.h"
#include "vulkan_tensor.h"
#include "hyle/compute/sampling_ops.h"
#include "hyle/compute/model_ops.h"
#include "../compute_packed_shared/artifact_packed_linear.h"
#include "../compute_packed_shared/static_fp8_linear.h"
#include "../compute_batch/row_cache_map.h"

#include <new>
#include <utility>

namespace hyle::compute::vulkan::detail {

Status lower_operation(ProgramState &program, const PlanStep &step) noexcept {
  return lower_operation(program, step, {});
}

Status lower_operation(ProgramState &program, const PlanStep &step,
                       ColdCustomResolver resolver,
                       RowCacheAccessFacts *access) noexcept {
  if (access) *access = {};
  try {
    if (step.operation.custom_as<model_ops::BlockwiseW4A8Linear>() ||
        step.operation.custom_as<model_ops::GroupedBlockwiseW4A8Linear>())
      return lower_blockwise_w4a8(program, step);
    if (step.operation.custom_as<packed::internal::StaticFloat8Linear>())
      return lower_static_fp8(program, step);
    if (step.operation.custom_as<packed::internal::ArtifactPackedLinear>())
      return lower_ct_packed(program, step);
    if (const auto artifact = step.operation.custom_as<model_ops::ArtifactWeightDecode>();
        artifact && artifact->format == model_ops::WeightArtifactFormat::kGptqV1Int4)
      return lower_gptq_decode(program, step);
    if (step.operation.custom_as<model_ops::Mxfp4Linear>() ||
        step.operation.custom_as<model_ops::GroupedMxfp4Linear>() ||
        step.operation.custom_as<model_ops::Nvfp4Linear>() ||
        step.operation.custom_as<model_ops::GroupedNvfp4Linear>() ||
        step.operation.custom_as<model_ops::LocalNvfp4Linear>() ||
        step.operation.custom_as<model_ops::GroupedLocalNvfp4Linear>())
      return lower_nvfp4(program, step);
    if (step.operation.custom_as<model_ops::RowCacheIndexerScores>() ||
        step.operation.custom_as<compute::detail::RowCachePagedIndexerScores>())
      return lower_row_cache_indexer(program, step, access);
    if (step.operation.custom_as<model_ops::RowCacheEncode>() ||
        step.operation.custom_as<model_ops::RowCacheGatherDecode>() ||
        step.operation.custom_as<model_ops::RowCacheGatherDecodePadded>() ||
        step.operation.custom_as<compute::detail::RowCachePaddedMap>() ||
        step.operation.custom_as<compute::detail::RowCacheMap>())
      return lower_row_cache(program, step, access);
    if (step.operation.custom_as<sampling_ops::RandomBits>() ||
        step.operation.custom_as<sampling_ops::Categorical>() ||
        single_topk_operation(program, step))
      return lower_sampling(program, step);
    if (managed_elementwise_limits(program, step))
      return lower_managed_elementwise(program, step);
    if (step.operation.kind() == OperationKind::kReduce ||
        step.operation.kind() == OperationKind::kSoftmax)
      return lower_managed_reduction(program, step);
    if (step.operation.kind() == OperationKind::kLayerNorm)
      return lower_layer_norm(program, step);
    if (step.operation.kind() == OperationKind::kTopK)
      return lower_topk(program, step);
    if (step.operation.custom_as<model_ops::GatherElements>())
      return lower_gather_elements(program, step);
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
      ColdCustomDispatch native;
      if (step.custom) {
        native = {step.custom, step.groups, step.push_constants};
      } else if (resolver.resolve) {
        std::vector<TensorDesc> inputs, outputs;
        inputs.reserve(step.inputs.size());
        outputs.reserve(step.outputs.size());
        for (const auto id : step.inputs)
          inputs.push_back(program.values[id].description);
        for (const auto id : step.outputs)
          outputs.push_back(program.values[id].description);
        HYLE_ASSIGN_OR_RETURN(native, resolver.resolve(
            resolver.state, step.operation, inputs, outputs));
      } else {
        return Status::Unimplemented(
            "Vulkan custom schema needs a cold native pipeline");
      }
      const auto &pipeline = Access::pipeline(native.kernel);
      if (!pipeline || pipeline->device != program.runtime->device ||
          pipeline->bindings != step.inputs.size() + 1 ||
          pipeline->push_bytes != native.push_constants.size())
        return Status::InvalidArgument(
            "Vulkan custom pipeline device or argument layout");
      for (std::size_t axis = 0; axis < 3; ++axis)
        if (native.groups[axis] == 0 ||
            native.groups[axis] >
                program.runtime->device->properties.limits
                    .maxComputeWorkGroupCount[axis])
          return Status::InvalidArgument(
              "Vulkan custom dispatch group count");
      Dispatch dispatch;
      dispatch.pipeline = pipeline;
      dispatch.values = step.inputs;
      dispatch.values.push_back(step.outputs[0]);
      dispatch.groups = native.groups;
      dispatch.push = std::move(native.push_constants);
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
