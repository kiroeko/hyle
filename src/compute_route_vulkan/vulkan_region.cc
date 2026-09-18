#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <new>

namespace hyle::compute::vulkan::detail {
namespace provider = ggml_vulkan::detail;

bool managed_operation(const ProgramState &program,
                       const PlanStep &step) noexcept {
  if (step.custom || step.inputs.size() > 3)
    return false;
  std::array<TensorDesc, 3> inputs;
  for (std::size_t i = 0; i < step.inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  const auto &output = program.values[step.output].description;
  if (!provider::region_supports(
          step.operation, std::span(inputs).first(step.inputs.size()), output))
    return false;
  // Original RMSNorm uses ne1/ne2/ne3 as workgroup counts. Other admitted
  // elementwise kernels split their flattened domain into bounded XYZ tiles.
  if (step.operation.get_if<RmsNorm>()) {
    const auto dimensions = output.dimensions();
    for (std::size_t axis = 1; axis < dimensions.size(); ++axis)
      if (static_cast<std::uint64_t>(dimensions[dimensions.size() - axis - 1]) >
          program.runtime->device->properties.limits
              .maxComputeWorkGroupCount[axis - 1])
        return false;
  }
  return true;
}

Status lower_region(ProgramState &program, const Plan &plan, std::size_t first,
                    std::size_t end,
                    std::span<const std::size_t> last_uses) noexcept {
  try {
    std::vector<std::uint32_t> ids;
    for (auto i = first; i < end; ++i) {
      ids.insert(ids.end(), plan.steps[i].inputs.begin(),
                 plan.steps[i].inputs.end());
      ids.push_back(plan.steps[i].output);
    }
    std::ranges::sort(ids);
    const auto duplicate = std::ranges::unique(ids);
    ids.erase(duplicate.begin(), duplicate.end());
    const auto local = [&](std::uint32_t value) {
      return static_cast<std::uint32_t>(std::ranges::lower_bound(ids, value) -
                                        ids.begin());
    };
    auto recipe = std::make_shared<provider::RegionRecipe>();
    for (const auto id : ids)
      recipe->values.push_back(program.values[id].description);
    std::vector<bool> produced(ids.size(), false);
    for (auto i = first; i < end; ++i) {
      const auto &step = plan.steps[i];
      provider::RegionNode node{step.operation, {}, local(step.output)};
      for (const auto input : step.inputs)
        node.inputs.push_back(local(input));
      recipe->nodes.push_back(std::move(node));
      produced[local(step.output)] = true;
      if (last_uses[step.output] >= end ||
          std::ranges::find(plan.outputs, step.output) != plan.outputs.end())
        recipe->outputs.push_back(local(step.output));
    }
    for (std::uint32_t i = 0; i < ids.size(); ++i)
      if (!produced[i])
        recipe->inputs.push_back(i);
    // A dead-only region can occur only in a manually supplied private Plan;
    // still keep a concrete output so the original allocator can own its work.
    if (recipe->outputs.empty())
      recipe->outputs.push_back(local(plan.steps[end - 1].output));
    Dispatch dispatch;
    while (program.aliases.size() < program.values.size()) {
      const auto value = static_cast<std::uint32_t>(program.aliases.size());
      program.aliases.push_back({value, 0});
    }
    for (const auto input : recipe->inputs) {
      const auto value = ids[input];
      const auto alias = program.aliases[value];
      const auto &constant = program.values[value].constant;
      const bool aligned =
          constant ? constant.native_offset() % sizeof(float) == 0
                   : std::ranges::find(program.inputs, alias.root) ==
                             program.inputs.end() &&
                         alias.byte_offset % sizeof(float) == 0;
      if (aligned) {
        dispatch.values.push_back(value);
        continue;
      }
      // A dense description says nothing about a bound View's byte origin.
      // Freeze a storage adapter now, using the existing byte-addressed Copy;
      // binding never changes Provider or rebuilds the native region graph.
      const auto adapted = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({program.values[value].description, {}});
      program.aliases.push_back({adapted, 0});
      HYLE_RETURN_IF_ERROR(
          lower_tensor(program, {Operation(Copy{}), {value}, adapted,
                                 {}, {1, 1, 1}, {}}));
      dispatch.values.push_back(adapted);
    }
    for (const auto output : recipe->outputs)
      dispatch.values.push_back(ids[output]);
    dispatch.region = std::move(recipe);
    program.dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan region boundary plan");
  }
}

void refresh_region_bindings(Run &run) noexcept {
  for (auto &stage : run.stages) {
    if (stage.managed) {
      for (std::size_t port = 0; port < stage.values.size(); ++port) {
        const auto &buffer = run.retained_buffers[stage.values[port]];
        stage.bindings[port] =
            buffer && buffer->handle
                ? provider::RegionBinding{buffer->handle,
                                          static_cast<std::size_t>(
                                              buffer->offset),
                                          static_cast<std::size_t>(
                                              buffer->bytes),
                                          buffer}
                : provider::RegionBinding{};
      }
    }
  }
}

Status bind_regions(Run &run) noexcept {
  refresh_region_bindings(run);
  for (auto &stage : run.stages) {
    if (stage.managed) {
      HYLE_RETURN_IF_ERROR(stage.managed->bind(stage.bindings));
    } else {
      for (std::size_t i = 0; i < stage.values.size(); ++i)
        stage.native->retained_buffers[i] =
            run.retained_buffers[stage.values[i]];
      if (origins_changed(*stage.native, stage.encoding->dispatches))
        HYLE_RETURN_IF_ERROR(
            encode_commands(*stage.native, stage.encoding->dispatches));
    }
  }
  return Status::Ok();
}

Status record_regions(const std::shared_ptr<Run> &run,
                      std::span<const PlanValue> values,
                      std::span<const Dispatch> dispatches,
                      std::vector<DescriptorBinding> *dynamic) noexcept {
  try {
    for (std::size_t first = 0; first < dispatches.size();) {
      RunStage stage;
      if (dispatches[first].region) {
        const auto &dispatch = dispatches[first++];
        stage.values = dispatch.values;
        stage.bindings.resize(stage.values.size());
        std::lock_guard lock(run->runtime->mutex);
        HYLE_ASSIGN_OR_RETURN(
            auto region, provider::Region::create(
                             run->runtime->device->provider, *dispatch.region));
        stage.managed = std::make_unique<provider::Region>(std::move(region));
      } else {
        auto end = first + 1;
        while (end < dispatches.size() && !dispatches[end].region)
          ++end;
        for (auto i = first; i < end; ++i)
          stage.values.insert(stage.values.end(), dispatches[i].values.begin(),
                              dispatches[i].values.end());
        std::ranges::sort(stage.values);
        const auto duplicate = std::ranges::unique(stage.values);
        stage.values.erase(duplicate.begin(), duplicate.end());
        stage.encoding = std::make_shared<NativeEncoding>();
        stage.encoding->dispatches.assign(dispatches.begin() + first,
                                          dispatches.begin() + end);
        for (auto &dispatch : stage.encoding->dispatches)
          for (auto &value : dispatch.values)
            value = static_cast<std::uint32_t>(
                std::ranges::lower_bound(stage.values, value) -
                stage.values.begin());
        HYLE_ASSIGN_OR_RETURN(stage.native, create_run(run->runtime));
        std::vector<PlanValue> compact;
        for (const auto value : stage.values) {
          compact.push_back(values[value]);
          stage.native->retained_buffers.push_back(
              run->retained_buffers[value]);
        }
        std::vector<DescriptorBinding> bindings;
        HYLE_RETURN_IF_ERROR(record(stage.native, compact,
                                    stage.encoding->dispatches,
                                    dynamic ? &bindings : nullptr));
        if (dynamic)
          for (auto &binding : bindings) {
            binding.value = stage.values[binding.value];
            dynamic->push_back(binding);
          }
        first = end;
      }
      run->stages.push_back(std::move(stage));
    }
    std::lock_guard lock(run->runtime->mutex);
    HYLE_RETURN_IF_ERROR(bind_regions(*run));
    const auto &last = run->stages.back();
    run->completion_fence = last.managed ? last.managed->fence() : run->fence;
    run->encoding_valid = true;
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan composed native regions");
  }
}
} // namespace hyle::compute::vulkan::detail
