#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <new>

namespace hyle::compute::vulkan::detail {
namespace provider = ggml_vulkan::detail;

bool managed_operation(const ProgramState &program,
                       const PlanStep &step) noexcept {
  if (step.custom || step.inputs.size() > 3 || step.outputs.size() > 2)
    return false;
  std::array<TensorDesc, 3> inputs;
  for (std::size_t i = 0; i < step.inputs.size(); ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  std::array<TensorDesc, 2> outputs;
  for (std::size_t i = 0; i < step.outputs.size(); ++i)
    outputs[i] = program.values[step.outputs[i]].description;
  if (!provider::region_supports(step.operation,
                                 std::span(inputs).first(step.inputs.size()),
                                 std::span(outputs).first(step.outputs.size())))
    return false;
  // Original RMSNorm uses ne1/ne2/ne3 as workgroup counts. Other admitted
  // elementwise kernels split their flattened domain into bounded XYZ tiles.
  if (step.operation.get_if<RmsNorm>()) {
    const auto dimensions = outputs[0].dimensions();
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
      ids.insert(ids.end(), plan.steps[i].outputs.begin(),
                 plan.steps[i].outputs.end());
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
    struct ValueUses final {
      bool produced = false;
      bool descriptor_aligned = false;
    };
    std::vector<ValueUses> uses(ids.size());
    for (auto i = first; i < end; ++i) {
      const auto &step = plan.steps[i];
      provider::RegionNode node{step.operation, {}, {}};
      for (const auto input : step.inputs)
        node.inputs.push_back(local(input));
      // Original Argsort has no shader-side descriptor offset. Other admitted
      // operations retain their four-byte, offset-aware input contract.
      if (step.operation.get_if<TopK>())
        uses[local(step.inputs[0])].descriptor_aligned = true;
      for (const auto output : step.outputs) {
        node.outputs.push_back(local(output));
        uses[local(output)].produced = true;
        if (last_uses[output] >= end ||
            std::ranges::find(plan.outputs, output) != plan.outputs.end())
          recipe->outputs.push_back(local(output));
      }
      recipe->nodes.push_back(std::move(node));
    }
    for (std::uint32_t i = 0; i < ids.size(); ++i)
      if (!uses[i].produced)
        recipe->inputs.push_back(i);
    // A dead-only region can occur only in a manually supplied private Plan;
    // still keep a concrete output so the original allocator can own its work.
    if (recipe->outputs.empty())
      recipe->outputs.push_back(local(plan.steps[end - 1].outputs[0]));
    Dispatch dispatch;
    while (program.aliases.size() < program.values.size()) {
      const auto value = static_cast<std::uint32_t>(program.aliases.size());
      program.aliases.push_back({value, 0});
    }
    for (const auto input : recipe->inputs) {
      const auto value = ids[input];
      const auto alias = program.aliases[value];
      const auto &constant = program.values[alias.root].constant;
      const auto alignment =
          uses[input].descriptor_aligned
              ? (std::max)(VkDeviceSize{sizeof(float)},
                           program.runtime->device->properties.limits
                               .minStorageBufferOffsetAlignment)
              : sizeof(float);
      // Private layout aliases retain the root's storage, not a separate
      // constant Tensor. Include both origins, without overflowing their sum.
      const auto origin_remainder =
          (constant ? constant.native_offset() % alignment : 0) +
          alias.byte_offset % alignment;
      const bool aligned = std::ranges::find(program.inputs, alias.root) ==
                               program.inputs.end() &&
                           origin_remainder % alignment == 0;
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
      HYLE_RETURN_IF_ERROR(lower_tensor(
          program, {Operation(Copy{}), {value}, {adapted}, {}, {1, 1, 1}, {}}));
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

Status lower_topk(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto input = program.values[step.inputs[0]].description;
    if (input.data_type() != DataType::kFloat32)
      return Status::Unimplemented("Vulkan TopK requires F32 router scores");
    if (*input.element_count() == 0)
      return Status::Ok(); // Distinct empty published owners, no GPU payload.
    if (*input.element_count() >
        program.runtime->device->properties.limits.maxStorageBufferRange / 4)
      return Status::Unimplemented(
          "Vulkan TopK dense adapter exceeds device limits");
    auto selection = *step.operation.get_if<TopK>();
    const auto rank = input.rank();
    const auto axis = static_cast<std::size_t>(
        selection.axis < 0 ? selection.axis + static_cast<std::int32_t>(rank)
                           : selection.axis);
    const auto width = input.dimensions()[axis];
    const auto rows = static_cast<std::int64_t>(*input.element_count()) / width;
    selection.axis = -1;
    while (program.aliases.size() < program.values.size()) {
      const auto id = static_cast<std::uint32_t>(program.aliases.size());
      program.aliases.push_back({id, 0});
    }
    const auto append = [&](const TensorDesc &description) {
      const auto id = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({description, {}});
      program.aliases.push_back({id, 0});
      return id;
    };
    const auto alias = [&](const TensorDesc &description,
                           std::uint32_t source) {
      const auto owner = program.aliases[source];
      const auto id = append(description);
      program.aliases[id] = owner;
      return id;
    };
    std::array<std::int32_t, kMaxTensorRank> axes{}, inverse{};
    std::size_t next = 0;
    for (std::size_t i = 0; i < rank; ++i)
      if (i != axis)
        axes[next++] = static_cast<std::int32_t>(i);
    axes[next] = static_cast<std::int32_t>(axis);
    for (std::size_t i = 0; i < rank; ++i)
      inverse[axes[i]] = static_cast<std::int32_t>(i);
    HYLE_ASSIGN_OR_RETURN(auto permute,
                          View::permute(std::span(axes).first(rank)));
    HYLE_ASSIGN_OR_RETURN(
        auto permuted,
        infer_single_output(Operation(permute), std::span(&input, 1)));
    auto source = alias(permuted, step.inputs[0]);
    if (!permuted.is_dense()) {
      HYLE_ASSIGN_OR_RETURN(
          auto dense,
          TensorDesc::dense(DataType::kFloat32, permuted.dimensions()));
      const auto packed = append(dense);
      HYLE_RETURN_IF_ERROR(lower_tensor(
          program, {Operation(Copy{}), {source}, {packed}, {}, {1, 1, 1}, {}}));
      source = packed;
    }
    HYLE_ASSIGN_OR_RETURN(
        auto matrix,
        TensorDesc::dense(DataType::kFloat32,
                          std::array<std::int64_t, 2>{rows, width}));
    const auto matrix_input = alias(matrix, source);
    std::array<TensorDesc, 2> results;
    HYLE_RETURN_IF_ERROR(infer_output_descriptions(
        Operation(selection), std::span(&matrix, 1), results));
    std::array<std::uint32_t, 2> output_ids{};
    for (std::size_t port = 0; port < output_ids.size(); ++port)
      output_ids[port] = axis + 1 == rank
                             ? alias(results[port], step.outputs[port])
                             : append(results[port]);
    Plan local;
    local.outputs.assign(output_ids.begin(), output_ids.end());
    local.steps.push_back({Operation(selection),
                           {matrix_input},
                           local.outputs,
                           {},
                           {1, 1, 1},
                           {}});
    std::vector<std::size_t> last_uses(program.values.size(), 0);
    HYLE_RETURN_IF_ERROR(lower_region(program, local, 0, 1, last_uses));
    if (axis + 1 == rank)
      return Status::Ok();
    // Provider rows are the other axes in their original order. Restore the
    // selected axis with ordinary View/Copy; both materializations are counted.
    std::array<std::int64_t, kMaxTensorRank> shape{};
    std::ranges::copy(permuted.dimensions(), shape.begin());
    shape[rank - 1] = static_cast<std::int64_t>(selection.count);
    HYLE_ASSIGN_OR_RETURN(auto restore,
                          View::permute(std::span(inverse).first(rank)));
    for (std::size_t port = 0; port < output_ids.size(); ++port) {
      HYLE_ASSIGN_OR_RETURN(auto expanded,
                            TensorDesc::dense(results[port].data_type(),
                                              std::span(shape).first(rank)));
      HYLE_ASSIGN_OR_RETURN(
          auto reordered,
          infer_single_output(Operation(restore), std::span(&expanded, 1)));
      const auto view = alias(reordered, output_ids[port]);
      HYLE_RETURN_IF_ERROR(lower_tensor(program, {Operation(Copy{}),
                                                  {view},
                                                  {step.outputs[port]},
                                                  {},
                                                  {1, 1, 1},
                                                  {}}));
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan TopK layout metadata");
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
