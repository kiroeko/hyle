#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <new>

namespace hyle::compute::vulkan::detail {
namespace {
std::array<bool, kMaxTensorRank> reduction_axes(const Operation &operation,
                                                std::size_t rank) noexcept {
  std::array<bool, kMaxTensorRank> selected{};
  const auto mark = [&](int axis) {
    selected[axis < 0 ? axis + static_cast<int>(rank) : axis] = true;
  };
  if (const auto *reduce = operation.get_if<Reduce>()) {
    if (reduce->axes.is_all())
      selected.fill(true);
    else
      for (auto axis : reduce->axes.values())
        mark(axis);
  } else {
    mark(operation.get_if<Softmax>()->axis);
  }
  return selected;
}
} // namespace

bool managed_reduction_limits(const ProgramState &program,
                              const Operation &operation,
                              const TensorDesc &input) noexcept {
  const auto count = input.element_count();
  if (!count || *count == 0 || *count > INT32_MAX || input.rank() == 0)
    return false;
  const auto *reduce = operation.get_if<Reduce>();
  const auto *softmax = operation.get_if<Softmax>();
  if ((reduce && reduce->kind != ReductionKind::kSum &&
       reduce->kind != ReductionKind::kMean) ||
      (softmax && softmax->kind != SoftmaxKind::kSoftmax))
    return false;
  const auto accumulation =
      reduce ? reduce->accumulation_type : softmax->accumulation_type;
  if ((accumulation != DataType::kFloat32 &&
       accumulation != DataType::kUnknown) ||
      (input.data_type() != DataType::kFloat32 &&
       (accumulation != DataType::kFloat32 ||
        (input.data_type() != DataType::kFloat16 &&
         input.data_type() != DataType::kBFloat16))))
    return false;
  const auto selected = reduction_axes(operation, input.rank());
  std::uint64_t width = 1;
  for (std::size_t axis = 0; axis < input.rank(); ++axis)
    if (selected[axis])
      width *= static_cast<std::uint64_t>(input.dimensions()[axis]);
  const auto rows = *count / width;
  const auto &device = *program.runtime->device;
  const auto &limits = device.properties.limits;
  const bool large = softmax && width > 16384;
  const auto lanes = large ? 128U
                     : softmax && width > 1024
                         ? 512U
                         : device.environment.subgroup_size;
  if (!std::has_single_bit(lanes) ||
      lanes > limits.maxComputeWorkGroupInvocations ||
      lanes > limits.maxComputeWorkGroupSize[0] ||
      std::uint64_t{lanes} * (large ? 8 : 4) >
          limits.maxComputeSharedMemorySize ||
      *count > limits.maxStorageBufferRange / sizeof(float))
    return false;
  if (large) {
    const auto blocks = (width + 511) / 512;
    return blocks <= limits.maxComputeWorkGroupCount[0] &&
           rows <= limits.maxComputeWorkGroupCount[1] &&
           blocks * rows <= limits.maxStorageBufferRange / sizeof(float);
  }
  return (std::min)(rows, std::uint64_t{512}) <=
             limits.maxComputeWorkGroupCount[0] &&
         (std::min)((rows + 511) / 512, std::uint64_t{512}) <=
             limits.maxComputeWorkGroupCount[1] &&
         (rows + 262143) / 262144 <= limits.maxComputeWorkGroupCount[2];
}

Status lower_managed_reduction(ProgramState &program,
                               const PlanStep &step) noexcept {
  try {
    const auto input = program.values[step.inputs[0]].description;
    const auto output = program.values[step.outputs[0]].description;
    if (const auto *probability = step.operation.get_if<Softmax>();
        probability && step.inputs.size() == 2 &&
        program.values[step.inputs[1]].description.data_type() !=
            (probability->mask == MaskKind::kBoolean ? DataType::kBool
                                                     : input.data_type()))
      return lower_tensor(program, step);
    if (!managed_reduction_limits(program, step.operation, input))
      return lower_tensor(program, step);
    const auto *reduce = step.operation.get_if<Reduce>();
    auto softmax = reduce ? Softmax{} : *step.operation.get_if<Softmax>();
    const auto rank = input.rank();
    const auto selected = reduction_axes(step.operation, rank);
    std::array<std::int32_t, kMaxTensorRank> axes{}, inverse{};
    std::size_t next = 0;
    std::int64_t width = 1;
    for (bool reducing : {false, true})
      for (std::size_t axis = 0; axis < rank; ++axis)
        if (selected[axis] == reducing) {
          axes[next++] = static_cast<std::int32_t>(axis);
          if (reducing)
            width *= input.dimensions()[axis];
        }
    for (std::size_t axis = 0; axis < rank; ++axis)
      inverse[axes[axis]] = static_cast<std::int32_t>(axis);
    const auto rows = static_cast<std::int64_t>(*input.element_count()) / width;
    HYLE_ASSIGN_OR_RETURN(auto permutation,
                          View::permute(std::span(axes).first(rank)));
    HYLE_ASSIGN_OR_RETURN(
        auto permuted,
        infer_single_output(Operation(permutation), std::span(&input, 1)));
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
    const auto emit = [&](Operation operation,
                          std::vector<std::uint32_t> inputs,
                          std::uint32_t destination) {
      return lower_tensor(
          program,
          {operation, std::move(inputs), {destination}, {}, {1, 1, 1}, {}});
    };
    const auto wide = [&](std::uint32_t source) -> Result<std::uint32_t> {
      const auto description = program.values[source].description;
      if (description.is_dense() &&
          description.data_type() == DataType::kFloat32)
        return source;
      HYLE_ASSIGN_OR_RETURN(
          auto dense,
          TensorDesc::dense(DataType::kFloat32, description.dimensions()));
      const auto result = append(dense);
      HYLE_RETURN_IF_ERROR(
          emit(Operation(Cast{DataType::kFloat32}), {source}, result));
      return result;
    };
    const auto scalar = [&](float value) -> Result<std::uint32_t> {
      HYLE_ASSIGN_OR_RETURN(auto description,
                            TensorDesc::dense(DataType::kFloat32, {}));
      HYLE_ASSIGN_OR_RETURN(auto shape, TensorShape::create({}));
      const auto result = append(description);
      HYLE_RETURN_IF_ERROR(
          emit(Operation(Fill{shape, DataType::kFloat32, ScalarValue(value)}),
               {}, result));
      return result;
    };
    HYLE_ASSIGN_OR_RETURN(auto source, wide(alias(permuted, step.inputs[0])));
    HYLE_ASSIGN_OR_RETURN(
        auto matrix,
        TensorDesc::dense(DataType::kFloat32,
                          std::array<std::int64_t, 2>{rows, width}));
    std::vector<std::uint32_t> inputs;
    if (!reduce && softmax.mask != MaskKind::kNone) {
      // Preserve F32 multiplication before masking; negative/zero scale must
      // never be applied to the Boolean mask's negative infinity sentinel.
      if (softmax.scale != 1.0F) {
        HYLE_ASSIGN_OR_RETURN(auto factor, scalar(softmax.scale));
        const auto scaled = append(program.values[source].description);
        HYLE_RETURN_IF_ERROR(emit(Operation(Binary{BinaryKind::kMultiply}),
                                  {source, factor}, scaled));
        source = scaled;
        softmax.scale = 1.0F;
      }
      HYLE_ASSIGN_OR_RETURN(auto shape,
                            TensorShape::create(input.dimensions()));
      HYLE_ASSIGN_OR_RETURN(auto broadcast, View::broadcast_to(shape));
      const auto mask = program.values[step.inputs[1]].description;
      HYLE_ASSIGN_OR_RETURN(
          auto expanded,
          infer_single_output(Operation(broadcast), std::span(&mask, 1)));
      HYLE_ASSIGN_OR_RETURN(
          auto reordered,
          infer_single_output(Operation(permutation), std::span(&expanded, 1)));
      auto mask_value = alias(reordered, step.inputs[1]);
      if (softmax.mask == MaskKind::kBoolean) {
        HYLE_ASSIGN_OR_RETURN(auto excluded,
                              scalar(-std::numeric_limits<float>::infinity()));
        const auto masked = append(program.values[source].description);
        HYLE_RETURN_IF_ERROR(
            emit(Operation(Select{}), {mask_value, source, excluded}, masked));
        source = masked;
        softmax.mask = MaskKind::kNone;
      } else {
        HYLE_ASSIGN_OR_RETURN(mask_value, wide(mask_value));
        inputs.push_back(alias(matrix, mask_value));
      }
    }
    inputs.insert(inputs.begin(), alias(matrix, source));
    HYLE_ASSIGN_OR_RETURN(
        auto result_matrix,
        TensorDesc::dense(DataType::kFloat32, std::array<std::int64_t, 2>{
                                                  rows, reduce ? 1 : width}));
    const bool final_layout = reduce || selected[rank - 1];
    const bool direct =
        output.data_type() == DataType::kFloat32 && final_layout;
    const auto result =
        direct ? alias(result_matrix, step.outputs[0]) : append(result_matrix);
    HYLE_ASSIGN_OR_RETURN(auto last_axis, Axes::from(std::array{-1}));
    softmax.axis = -1;
    softmax.accumulation_type = DataType::kFloat32;
    const auto operation = reduce ? Operation(Reduce{reduce->kind, last_axis,
                                                     true, DataType::kFloat32})
                                  : Operation(softmax);
    Plan local;
    local.outputs = {result};
    local.steps.push_back(
        {operation, std::move(inputs), {result}, {}, {1, 1, 1}, {}});
    const std::vector<std::size_t> last_uses(program.values.size(), 0);
    HYLE_RETURN_IF_ERROR(lower_region(program, local, 0, 1, last_uses));
    if (!direct) {
      HYLE_ASSIGN_OR_RETURN(auto restored,
                            TensorDesc::dense(DataType::kFloat32,
                                              reduce ? output.dimensions()
                                                     : permuted.dimensions()));
      if (!reduce) {
        HYLE_ASSIGN_OR_RETURN(auto restore,
                              View::permute(std::span(inverse).first(rank)));
        HYLE_ASSIGN_OR_RETURN(
            restored,
            infer_single_output(Operation(restore), std::span(&restored, 1)));
      }
      HYLE_RETURN_IF_ERROR(emit(Operation(Cast{output.data_type()}),
                                {alias(restored, result)}, step.outputs[0]));
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan reduction adapter metadata");
  }
}
} // namespace hyle::compute::vulkan::detail
