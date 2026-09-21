#include "vulkan_iree.h"
#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <new>

namespace hyle::compute::vulkan::detail {
namespace provider = ggml_vulkan::detail;
namespace {
bool layout_limits(const ProgramState &program, const PlanStep &step) noexcept {
  const auto &limits = program.runtime->device->properties.limits;
  const auto fits = [&](std::uint32_t value) {
    const auto count = *program.values[value].description.element_count();
    return count <= limits.maxStorageBufferRange / sizeof(float) &&
           (count + 262143) / 262144 <= limits.maxComputeWorkGroupCount[2];
  };
  // The original cont/repeat/concat adapters use 512-lane flattened dispatches.
  // Check the intermediate F32 materialization as well as the public output.
  return limits.maxComputeWorkGroupInvocations >= 512 &&
         limits.maxComputeWorkGroupSize[0] >= 512 &&
         limits.maxComputeWorkGroupCount[0] >= 512 &&
         limits.maxComputeWorkGroupCount[1] >= 512 &&
         std::ranges::all_of(step.inputs, fits) &&
         std::ranges::all_of(step.outputs, fits);
}
bool topk_limits(const ProgramState &program, const TensorDesc &input,
                 const TopK &selection) noexcept {
  const auto axis = static_cast<std::size_t>(
      selection.axis < 0
          ? selection.axis + static_cast<std::int32_t>(input.rank())
          : selection.axis);
  const auto width = static_cast<std::uint32_t>(input.dimensions()[axis]);
  const auto rows = *input.element_count() / width;
  const auto padded = std::bit_ceil(width);
  const auto &limits = program.runtime->device->properties.limits;
  const auto max_group = std::bit_floor(limits.maxComputeWorkGroupInvocations);
  if (max_group == 0 || limits.maxComputeWorkGroupCount[1] == 0)
    return false;
  const auto exponent = std::bit_width(padded) - 1u;
  const auto pipeline = (std::min)(exponent, 10u);
  const auto block = (std::min)(1u << pipeline, max_group);
  const bool small =
      padded <= max_group &&
      std::uint64_t{block} * 8 <= limits.maxComputeSharedMemorySize;
  if (small)
    return padded == block && block <= limits.maxComputeWorkGroupSize[0] &&
           limits.maxComputeWorkGroupCount[0] != 0;
  // The large kernel stores ivec2 for each padded element and unrolls two
  // elements per lane. Later cross-workgroup passes use pipeline index - 2.
  const auto smallest_block =
      padded <= max_group
          ? block
          : (std::min)(1u << (pipeline > 2 ? pipeline - 2 : 0), max_group);
  const auto lanes = block > 1 ? block / 2 : 1;
  return std::uint64_t{padded} * rows <= limits.maxStorageBufferRange / 8 &&
         lanes <= limits.maxComputeWorkGroupSize[0] &&
         (padded + smallest_block - 1) / smallest_block <=
             limits.maxComputeWorkGroupCount[0];
}

bool scan_limits(const ProgramState &program,
                 const TensorDesc &matrix) noexcept {
  const auto &limits = program.runtime->device->properties.limits;
  const std::uint64_t elements = *matrix.element_count();
  const auto width = static_cast<std::uint64_t>(matrix.dimensions().back());
  const auto rows = elements / width;
  const auto lanes = width <= 512 ? 128U : 256U;
  if (elements > INT32_MAX || elements > limits.maxStorageBufferRange / 4 ||
      lanes > limits.maxComputeWorkGroupInvocations ||
      lanes > limits.maxComputeWorkGroupSize[0] ||
      (std::min)(rows, std::uint64_t{512}) >
          limits.maxComputeWorkGroupCount[0] ||
      (std::min)((rows + 511) / 512, std::uint64_t{512}) >
          limits.maxComputeWorkGroupCount[1] ||
      (rows + 262143) / 262144 > limits.maxComputeWorkGroupCount[2])
    return false;
  // The original device rule may select two passes for wide, short matrices.
  // Admit only shapes legal for either native choice; never dispatch then
  // discover that the selected pipeline exceeds the physical grid limits.
  return width <= 4096 ||
         ((width + 127) / 128 <= limits.maxComputeWorkGroupCount[0] &&
          rows <= limits.maxComputeWorkGroupCount[1]);
}
bool region_operation(const ProgramState &program, const PlanStep &step,
                      bool promote_bfloat) noexcept {
  if (step.custom || step.inputs.size() > 6 || step.outputs.size() > 2)
    return false;
  std::array<TensorDesc, 6> inputs;
  std::size_t input_count = 0;
  for (const auto input : step.inputs) {
    const auto &description = program.values[input].description;
    // Empty Concat ports retain their public binding contract but have no
    // storage to import into the original backend.
    if (!step.operation.get_if<Concat>() || *description.element_count() != 0)
      inputs[input_count++] = description;
  }
  std::array<TensorDesc, 2> outputs;
  for (std::size_t i = 0; i < step.outputs.size(); ++i)
    outputs[i] = program.values[step.outputs[i]].description;
  const auto &features = program.runtime->device->provider.native();
  const auto low_precision = [](const TensorDesc &description) {
    return description.data_type() == DataType::kFloat16 ||
           description.data_type() == DataType::kBFloat16;
  };
  if ((std::ranges::any_of(std::span(inputs).first(input_count),
                           low_precision) ||
       std::ranges::any_of(std::span(outputs).first(step.outputs.size()),
                           low_precision)) &&
      (!features.shader_float16 || !features.storage_buffer16))
    return false;
  if (promote_bfloat) {
    for (std::size_t i = 0; i < input_count; ++i) {
      if (!inputs[i].is_plain() || !inputs[i].is_dense() ||
          (inputs[i].data_type() != DataType::kFloat32 &&
           !low_precision(inputs[i])))
        return false;
      const auto wide =
          TensorDesc::dense(DataType::kFloat32, inputs[i].dimensions());
      if (!wide)
        return false;
      inputs[i] = *wide;
    }
    const auto wide =
        TensorDesc::dense(DataType::kFloat32, outputs[0].dimensions());
    if (!wide)
      return false;
    outputs[0] = *wide;
  }
  if (!provider::region_supports(step.operation,
                                 std::span(inputs).first(input_count),
                                 std::span(outputs).first(step.outputs.size())))
    return false;
  if (step.operation.get_if<MatMul>()) {
    const auto &limits = program.runtime->device->properties.limits;
    const auto rows = static_cast<std::uint64_t>(inputs[0].dimensions()[0]);
    const auto columns = static_cast<std::uint64_t>(inputs[1].dimensions()[1]);
    // Original matmul may choose up to eight split-K pieces. Bound both its
    // F32 partial workspace and all grids before native graph allocation.
    if (rows > limits.maxComputeWorkGroupCount[1] ||
        columns > limits.maxComputeWorkGroupCount[0] / 8 ||
        rows * columns > limits.maxStorageBufferRange / (8 * sizeof(float)) ||
        rows * columns > limits.maxComputeWorkGroupCount[0])
      return false;
  }
  if (step.operation.get_if<GatedDeltaNet>()) {
    const auto dimensions = inputs[2].dimensions();
    const auto &limits = program.runtime->device->properties.limits;
    const auto packed_elements =
        *outputs[0].element_count() + *outputs[1].element_count();
    if (static_cast<std::uint64_t>(dimensions[2]) >
            limits.maxComputeWorkGroupCount[0] ||
        static_cast<std::uint64_t>(dimensions[0]) >
            limits.maxComputeWorkGroupCount[1] ||
        static_cast<std::uint64_t>(dimensions[3]) >
            limits.maxComputeWorkGroupCount[2] ||
        limits.maxPerStageDescriptorStorageBuffers < 7 ||
        limits.maxDescriptorSetStorageBuffers < 7 ||
        packed_elements > limits.maxStorageBufferRange / sizeof(float))
      return false;
    for (const auto &input : inputs)
      if (*input.element_count() > limits.maxStorageBufferRange / sizeof(float))
        return false;
  }
  if ((step.operation.get_if<Unary>() || step.operation.get_if<Binary>() ||
       step.operation.get_if<GatedActivation>()) &&
      !managed_elementwise_limits(program, step))
    return false;
  if ((promote_bfloat || step.operation.get_if<Copy>() ||
       step.operation.get_if<Cast>() || step.operation.get_if<Concat>() ||
       step.operation.get_if<Binary>() || step.operation.get_if<Reduce>() ||
       std::ranges::any_of(std::span(inputs).first(input_count),
                           low_precision) ||
       low_precision(outputs[0])) &&
      !layout_limits(program, step))
    return false;
  // Original RMSNorm uses ne1/ne2/ne3 as workgroup counts. Other admitted
  // elementwise kernels split their flattened domain into bounded XYZ tiles.
  if (step.operation.get_if<RmsNorm>()) {
    const auto dimensions = outputs[0].dimensions();
    const auto &limits = program.runtime->device->properties.limits;
    if (dimensions.size() > 4) {
      if (*outputs[0].element_count() / dimensions.back() >
          limits.maxComputeWorkGroupCount[0])
        return false;
    } else {
      for (std::size_t axis = 1; axis < dimensions.size(); ++axis)
        if (static_cast<std::uint64_t>(
                dimensions[dimensions.size() - axis - 1]) >
            limits.maxComputeWorkGroupCount[axis - 1])
          return false;
    }
  }
  if (step.operation.get_if<Gather>()) {
    const auto &limits = program.runtime->device->properties.limits;
    if (limits.maxComputeWorkGroupInvocations < 512 ||
        limits.maxComputeWorkGroupSize[0] < 512 ||
        static_cast<std::uint64_t>(inputs[0].dimensions()[1]) >
            std::uint64_t{limits.maxComputeWorkGroupCount[0]} * 512)
      return false;
  }
  if (step.operation.get_if<Scan>() && !scan_limits(program, inputs[0]))
    return false;
  if (step.operation.get_if<LayerNorm>() &&
      !layer_norm_limits(program, inputs[0]))
    return false;
  if ((step.operation.get_if<Reduce>() || step.operation.get_if<Softmax>()) &&
      !managed_reduction_limits(program, step.operation,
                                program.values[step.inputs[0]].description))
    return false;
  if (const auto *selection = step.operation.get_if<TopK>();
      selection && !topk_limits(program, inputs[0], *selection))
    return false;
  return true;
}
} // namespace

bool managed_operation(const ProgramState &program,
                       const PlanStep &step) noexcept {
  return region_operation(program, step, false);
}

bool bfloat_region_operation(const ProgramState &program,
                             const PlanStep &step) noexcept {
  if (step.outputs.size() != 1 ||
      program.values[step.outputs[0]].description.data_type() !=
          DataType::kBFloat16 ||
      !(step.operation.get_if<Unary>() || step.operation.get_if<Binary>() ||
        step.operation.get_if<GatedActivation>() ||
        step.operation.get_if<RmsNorm>() ||
        step.operation.get_if<LayerNorm>() || step.operation.get_if<Reduce>() ||
        step.operation.get_if<Softmax>()))
    return false;
  const auto wide = TensorDesc::dense(
      DataType::kFloat32,
      program.values[step.outputs[0]].description.dimensions());
  if (!wide || !iree_precision_supported(*program.runtime->device, *wide))
    return false;
  return region_operation(program, step, true);
}

Status lower_bfloat_region(ProgramState &program,
                           const PlanStep &step) noexcept {
  try {
    const auto append =
        [&](const TensorDesc &description) -> Result<std::uint32_t> {
      HYLE_ASSIGN_OR_RETURN(
          auto wide,
          TensorDesc::dense(DataType::kFloat32, description.dimensions()));
      const auto id = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({wide, {}});
      program.aliases.push_back({id, 0});
      return id;
    };
    Plan local;
    PlanStep wide = step;
    for (auto &input : wide.inputs) {
      if (program.values[input].description.data_type() == DataType::kFloat32)
        continue;
      HYLE_ASSIGN_OR_RETURN(auto converted,
                            append(program.values[input].description));
      local.steps.push_back({Operation(Cast{DataType::kFloat32}),
                             {input},
                             {converted},
                             {},
                             {1, 1, 1},
                             {}});
      input = converted;
    }
    HYLE_ASSIGN_OR_RETURN(auto result,
                          append(program.values[step.outputs[0]].description));
    wide.outputs = {result};
    local.steps.push_back(std::move(wide));
    local.outputs = {result};
    const std::vector<std::size_t> last_uses(program.values.size(), 0);
    HYLE_RETURN_IF_ERROR(
        lower_region(program, local, 0, local.steps.size(), last_uses));
    // The public BF16 port is a rounding boundary. Its qualified conversion
    // belongs to the original IREE program on this same native device.
    return lower_iree(program, {Operation(Cast{DataType::kBFloat16}),
                                {result},
                                step.outputs,
                                {},
                                {1, 1, 1},
                                {}});
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan BF16 region metadata");
  }
}

Status lower_region(ProgramState &program, const Plan &plan, std::size_t first,
                    std::size_t end,
                    std::span<const std::size_t> last_uses) noexcept {
  try {
    std::vector<std::uint32_t> ids;
    const auto has_payload = [&](const PlanStep &step, std::uint32_t input) {
      return !step.operation.get_if<Concat>() ||
             *program.values[input].description.element_count() != 0;
    };
    for (auto i = first; i < end; ++i) {
      for (const auto input : plan.steps[i].inputs)
        if (has_payload(plan.steps[i], input))
          ids.push_back(input);
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
        if (has_payload(step, input))
          node.inputs.push_back(local(input));
      // Argsort, GroupNorm, Softmax and multi-pass Scan need aligned
      // descriptors. Other admitted operations retain their offset-aware input
      // contract.
      if (step.operation.get_if<TopK>() || step.operation.get_if<Scan>() ||
          step.operation.get_if<LayerNorm>())
        uses[local(step.inputs[0])].descriptor_aligned = true;
      if (step.operation.get_if<Softmax>())
        for (const auto input : step.inputs)
          uses[local(input)].descriptor_aligned = true;
      if (step.operation.get_if<GatedDeltaNet>())
        for (const auto input : step.inputs)
          uses[local(input)].descriptor_aligned = true;
      if (step.operation.get_if<MatMul>())
        for (const auto input : step.inputs)
          uses[local(input)].descriptor_aligned = true;
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
      if (!program.values[value].description.is_plain()) {
        // Preserve compressed bytes. Borrowed compressed ports require native
        // descriptor alignment, checked before binding; no scalar Copy can
        // interpret a block encoding as its logical F16 elements.
        dispatch.values.push_back(value);
        continue;
      }
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

Status lower_gather(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto input = program.values[step.inputs[0]].description;
    const auto indices = program.values[step.inputs[1]].description;
    const auto output = program.values[step.outputs[0]].description;
    const auto gather = *step.operation.get_if<Gather>();
    const auto count = *indices.element_count();
    if (count == 0)
      return Status::Ok();
    const auto &limits = program.runtime->device->properties.limits;
    if (count > limits.maxStorageBufferRange / 4)
      return Status::Unimplemented("Vulkan Gather indices exceed device range");
    const auto axis = static_cast<std::size_t>(
        gather.axis < 0 ? gather.axis + static_cast<std::int32_t>(input.rank())
                        : gather.axis);
    const auto extent = static_cast<std::uint64_t>(input.dimensions()[axis]);
    const bool checked = gather.bounds == IndexBoundsPolicy::kChecked;
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
    if (checked && !program.status_value) {
      HYLE_ASSIGN_OR_RETURN(auto status,
                            TensorDesc::dense(DataType::kInt32, {}));
      program.status_value = append(status);
    }
    HYLE_ASSIGN_OR_RETURN(
        auto safe_desc,
        TensorDesc::dense(DataType::kInt32, indices.dimensions()));
    const auto safe = append(safe_desc);
    HYLE_RETURN_IF_ERROR(
        lower_index_check(program, step.inputs[1], safe, extent, checked));
    // Nonempty indices still require validation when another dimension is
    // empty. Never hand an empty source axis to the unchecked vendor kernel.
    if (*input.element_count() == 0 || *output.element_count() == 0)
      return Status::Ok();
    if (limits.maxComputeWorkGroupInvocations < 512 ||
        limits.maxComputeWorkGroupSize[0] < 512)
      return Status::Unimplemented("ggml Vulkan GetRows requires 512 lanes");
    const auto size = data_type_size(input.data_type());
    const auto row_elements = *input.element_count() / extent;
    const auto row_bytes = row_elements * size;
    const auto padded_bytes = (row_bytes + 3) / 4 * 4;
    if (extent > limits.maxStorageBufferRange / padded_bytes ||
        count > limits.maxStorageBufferRange / padded_bytes ||
        padded_bytes / 4 >
            std::uint64_t{limits.maxComputeWorkGroupCount[0]} * 512 ||
        extent * (padded_bytes / size) > INT32_MAX ||
        count * (padded_bytes / size) > INT32_MAX)
      return Status::Unimplemented(
          "Vulkan Gather row adapter exceeds buffer or dispatch range");
    std::array<std::int32_t, kMaxTensorRank> axes{};
    axes[0] = static_cast<std::int32_t>(axis);
    std::size_t next = 1;
    for (std::size_t dim = 0; dim < input.rank(); ++dim)
      if (dim != axis)
        axes[next++] = static_cast<std::int32_t>(dim);
    HYLE_ASSIGN_OR_RETURN(auto permute,
                          View::permute(std::span(axes).first(input.rank())));
    HYLE_ASSIGN_OR_RETURN(auto rows, infer_single_output(Operation(permute),
                                                         std::span(&input, 1)));
    auto source = alias(rows, step.inputs[0]);
    if (!rows.is_dense()) {
      HYLE_ASSIGN_OR_RETURN(
          auto dense, TensorDesc::dense(input.data_type(), rows.dimensions()));
      const auto packed = append(dense);
      HYLE_RETURN_IF_ERROR(lower_tensor(
          program, {Operation(Copy{}), {source}, {packed}, {}, {1, 1, 1}, {}}));
      source = packed;
    }
    HYLE_ASSIGN_OR_RETURN(
        auto matrix,
        TensorDesc::dense(input.data_type(),
                          std::array<std::int64_t, 2>{
                              static_cast<std::int64_t>(extent),
                              static_cast<std::int64_t>(row_elements)}));
    source = alias(matrix, source);
    const auto padding = (padded_bytes - row_bytes) / size;
    if (padding != 0) {
      HYLE_ASSIGN_OR_RETURN(
          auto tail_desc,
          TensorDesc::dense(
              input.data_type(),
              std::array<std::int64_t, 2>{static_cast<std::int64_t>(extent),
                                          static_cast<std::int64_t>(padding)}));
      const auto tail = append(tail_desc);
      HYLE_ASSIGN_OR_RETURN(auto tail_shape,
                            TensorShape::create(tail_desc.dimensions()));
      HYLE_RETURN_IF_ERROR(lower_tensor(
          program, {Operation(Fill{
                        .output_shape = tail_shape,
                        .output_type = input.data_type(),
                        .value = ScalarValue::from_bits(input.data_type(), 0)}),
                    {},
                    {tail},
                    {},
                    {1, 1, 1},
                    {}}));
      HYLE_ASSIGN_OR_RETURN(
          auto padded_desc,
          TensorDesc::dense(
              input.data_type(),
              std::array<std::int64_t, 2>{
                  static_cast<std::int64_t>(extent),
                  static_cast<std::int64_t>(row_elements + padding)}));
      const auto padded = append(padded_desc);
      HYLE_RETURN_IF_ERROR(lower_tensor(program, {Operation(Concat{-1}),
                                                  {source, tail},
                                                  {padded},
                                                  {},
                                                  {1, 1, 1},
                                                  {}}));
      source = padded;
    }
    HYLE_ASSIGN_OR_RETURN(
        auto raw_desc,
        TensorDesc::dense(DataType::kInt32,
                          std::array<std::int64_t, 2>{
                              static_cast<std::int64_t>(extent),
                              static_cast<std::int64_t>(padded_bytes / 4)}));
    const auto raw = alias(raw_desc, source);
    HYLE_ASSIGN_OR_RETURN(
        auto flat_indices,
        TensorDesc::dense(
            DataType::kInt32,
            std::array<std::int64_t, 1>{static_cast<std::int64_t>(count)}));
    const auto index_rows = alias(flat_indices, safe);
    HYLE_ASSIGN_OR_RETURN(
        auto gathered_desc,
        TensorDesc::dense(DataType::kInt32,
                          std::array<std::int64_t, 2>{
                              static_cast<std::int64_t>(count),
                              static_cast<std::int64_t>(padded_bytes / 4)}));
    const auto gathered = append(gathered_desc);
    Plan local;
    local.outputs = {gathered};
    local.steps.push_back({Operation(Gather{0, IndexBoundsPolicy::kUnchecked}),
                           {raw, index_rows},
                           {gathered},
                           {},
                           {1, 1, 1},
                           {}});
    std::vector<std::size_t> last_uses(program.values.size(), 0);
    HYLE_RETURN_IF_ERROR(lower_region(program, local, 0, 1, last_uses));
    // The provider publishes indices-first rows. Reinterpret exact scalar
    // bits, discard only row padding, and restore prefix/indices/suffix axes.
    std::array<std::int64_t, kMaxTensorRank> shape{}, strides{};
    next = indices.rank();
    std::ranges::copy(indices.dimensions(), shape.begin());
    for (std::size_t dim = 0; dim < input.rank(); ++dim)
      if (dim != axis)
        shape[next++] = input.dimensions()[dim];
    auto stride = static_cast<std::int64_t>(size);
    for (auto dim = next; dim > indices.rank(); --dim) {
      strides[dim - 1] = stride;
      stride *= shape[dim - 1];
    }
    stride = static_cast<std::int64_t>(padded_bytes);
    for (auto dim = indices.rank(); dim > 0; --dim) {
      strides[dim - 1] = stride;
      stride *= shape[dim - 1];
    }
    HYLE_ASSIGN_OR_RETURN(auto expanded,
                          TensorDesc::strided_bytes(
                              input.data_type(), std::span(shape).first(next),
                              std::span(strides).first(next)));
    std::size_t position = 0;
    for (std::size_t dim = 0; dim < axis; ++dim)
      axes[position++] = static_cast<std::int32_t>(indices.rank() + dim);
    for (std::size_t dim = 0; dim < indices.rank(); ++dim)
      axes[position++] = static_cast<std::int32_t>(dim);
    for (auto dim = indices.rank() + axis; dim < next; ++dim)
      axes[position++] = static_cast<std::int32_t>(dim);
    HYLE_ASSIGN_OR_RETURN(auto restore,
                          View::permute(std::span(axes).first(next)));
    HYLE_ASSIGN_OR_RETURN(
        auto reordered,
        infer_single_output(Operation(restore), std::span(&expanded, 1)));
    const auto result = alias(reordered, gathered);
    if (checked)
      return lower_checked_copy(program, result, step.outputs[0]);
    return lower_tensor(
        program,
        {Operation(Copy{}), {result}, {step.outputs[0]}, {}, {1, 1, 1}, {}});
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan Gather adapter metadata");
  }
}

Status lower_scan(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto input = program.values[step.inputs[0]].description;
    const bool integer = input.data_type() == DataType::kInt32;
    if (!integer && input.data_type() != DataType::kFloat32)
      return Status::Unimplemented("Vulkan Scan requires I32 or F32");
    if (*input.element_count() == 0)
      return Status::Ok();
    const auto scan = *step.operation.get_if<Scan>();
    const auto rank = input.rank();
    const auto axis = static_cast<std::size_t>(
        scan.axis < 0 ? scan.axis + static_cast<std::int32_t>(rank)
                      : scan.axis);
    const auto width = input.dimensions()[axis];
    const auto rows = static_cast<std::int64_t>(*input.element_count()) / width;
    HYLE_ASSIGN_OR_RETURN(
        auto matrix,
        TensorDesc::dense(input.data_type(),
                          std::array<std::int64_t, 2>{rows, width}));
    if (!integer && !scan_limits(program, matrix))
      return Status::Unimplemented(
          "Vulkan Scan exceeds native buffer or grid limits");
    if (scan.exclusive && width == 1) {
      HYLE_ASSIGN_OR_RETURN(auto shape,
                            TensorShape::create(input.dimensions()));
      return lower_tensor(
          program, {Operation(Fill{
                        .output_shape = shape,
                        .output_type = input.data_type(),
                        .value = ScalarValue::from_bits(input.data_type(), 0)}),
                    {},
                    step.outputs,
                    {},
                    {1, 1, 1},
                    {}});
    }
    if (integer && width == 1)
      return lower_tensor(
          program,
          {Operation(Copy{}), step.inputs, step.outputs, {}, {1, 1, 1}, {}});
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
          TensorDesc::dense(input.data_type(), permuted.dimensions()));
      const auto packed = append(dense);
      HYLE_RETURN_IF_ERROR(lower_tensor(
          program, {Operation(Copy{}), {source}, {packed}, {}, {1, 1, 1}, {}}));
      source = packed;
    }
    source = alias(matrix, source);
    const bool final_layout = axis + 1 == rank;
    auto result = !integer && !scan.exclusive && final_layout
                      ? alias(matrix, step.outputs[0])
                      : append(matrix);
    if (integer) {
      HYLE_RETURN_IF_ERROR(
          lower_integer_scan(program, source, result, scan.exclusive));
    } else {
      Plan local;
      local.outputs = {result};
      local.steps.push_back(
          {Operation(Scan{}), {source}, {result}, {}, {1, 1, 1}, {}});
      std::vector<std::size_t> last_uses(program.values.size(), 0);
      HYLE_RETURN_IF_ERROR(lower_region(program, local, 0, 1, last_uses));
    }
    if (!integer && scan.exclusive) {
      HYLE_ASSIGN_OR_RETURN(auto prefix,
                            TensorDesc::strided_bytes(
                                DataType::kFloat32,
                                std::array<std::int64_t, 2>{rows, width - 1},
                                std::array<std::int64_t, 2>{width * 4, 4}));
      const auto prefix_value = alias(prefix, result);
      HYLE_ASSIGN_OR_RETURN(
          auto zero_desc,
          TensorDesc::dense(DataType::kFloat32,
                            std::array<std::int64_t, 2>{rows, 1}));
      const auto zero = append(zero_desc);
      HYLE_ASSIGN_OR_RETURN(auto shape,
                            TensorShape::create(zero_desc.dimensions()));
      HYLE_RETURN_IF_ERROR(lower_tensor(
          program, {Operation(Fill{.output_shape = shape,
                                   .output_type = DataType::kFloat32,
                                   .value = ScalarValue(0.0F)}),
                    {},
                    {zero},
                    {},
                    {1, 1, 1},
                    {}}));
      result = final_layout ? alias(matrix, step.outputs[0]) : append(matrix);
      HYLE_RETURN_IF_ERROR(lower_tensor(program, {Operation(Concat{-1}),
                                                  {zero, prefix_value},
                                                  {result},
                                                  {},
                                                  {1, 1, 1},
                                                  {}}));
    }
    if (!final_layout) {
      HYLE_ASSIGN_OR_RETURN(
          auto expanded,
          TensorDesc::dense(input.data_type(), permuted.dimensions()));
      HYLE_ASSIGN_OR_RETURN(auto restore,
                            View::permute(std::span(inverse).first(rank)));
      HYLE_ASSIGN_OR_RETURN(
          auto reordered,
          infer_single_output(Operation(restore), std::span(&expanded, 1)));
      result = alias(reordered, result);
      if (!integer)
        HYLE_RETURN_IF_ERROR(lower_tensor(
            program,
            {Operation(Copy{}), {result}, step.outputs, {}, {1, 1, 1}, {}}));
    }
    if (integer) {
      if (final_layout) {
        HYLE_ASSIGN_OR_RETURN(
            auto expanded,
            TensorDesc::dense(input.data_type(), input.dimensions()));
        result = alias(expanded, result);
      }
      // Restore and publish only after every global prefix has been checked.
      return lower_checked_copy(program, result, step.outputs[0]);
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan Scan layout metadata");
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
    if (!topk_limits(program, input, selection))
      return Status::Unimplemented(
          "Vulkan TopK padded sort exceeds native limits");
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
    } else if (stage.iree) {
      HYLE_RETURN_IF_ERROR(stage.iree->bind());
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
      if (dispatches[first].iree) {
        const auto &dispatch = dispatches[first++];
        stage.values = dispatch.values;
        HYLE_ASSIGN_OR_RETURN(stage.iree,
                              create_iree_stage(run, dispatch, values));
      } else if (dispatches[first].region) {
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
        while (end < dispatches.size() && !dispatches[end].region &&
               !dispatches[end].iree)
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
