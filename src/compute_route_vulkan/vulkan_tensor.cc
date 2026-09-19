#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

#include "vulkan_shaders.h"

namespace hyle::compute::vulkan::detail {
std::optional<std::uint32_t> tensor_storage_type(DataType type) noexcept {
  switch (type) {
  case DataType::kFloat32:
    return 0;
  case DataType::kFloat16:
    return 1;
  case DataType::kBFloat16:
    return 2;
  case DataType::kInt32:
    return 3;
  case DataType::kInt64:
    return 4;
  case DataType::kBool:
    return 5;
  default:
    return std::nullopt;
  }
}

namespace {
Status tensor_layout(const ProgramState &program,
                     std::span<const std::uint32_t> inputs,
                     std::uint32_t output_id,
                     std::span<SpecializationConstant> specialization,
                     Dispatch &dispatch) {
  const auto &output = program.values[output_id].description;
  for (std::uint32_t index = 0; index < specialization.size(); ++index)
    specialization[index].id = index;
  specialization[4].value = *tensor_storage_type(output.data_type());
  specialization[6].value = static_cast<std::uint32_t>(*output.element_count());
  for (std::size_t axis = 0; axis < 8; ++axis) {
    const auto padding = 8 - output.rank();
    specialization[10 + axis].value =
        axis < padding
            ? 1u
            : static_cast<std::uint32_t>(output.dimensions()[axis - padding]);
  }
  dispatch.values.resize(4, output_id);
  for (std::size_t port = 0; port < inputs.size(); ++port) {
    const auto &input = program.values[inputs[port]].description;
    specialization[1 + port].value = *tensor_storage_type(input.data_type());
    dispatch.values[port] = inputs[port];
    if (*input.element_count() == 0)
      continue; // An empty Concat input has no reachable address.
    for (std::size_t axis = 0; axis < input.rank(); ++axis) {
      const auto stride =
          input.dimensions()[axis] == 1 ? 0 : input.byte_strides()[axis];
      if (!std::in_range<std::uint32_t>(stride))
        return Status::Unimplemented("Vulkan tensor byte stride exceeds U32");
      specialization[18 + port * 8 + 8 - input.rank() + axis].value =
          static_cast<std::uint32_t>(stride);
    }
  }
  return Status::Ok();
}

Status
append_tensor_dispatch(ProgramState &program, Dispatch dispatch,
                       std::span<const std::uint32_t> shader,
                       std::span<const SpecializationConstant> specialization,
                       std::uint32_t groups) {
  HYLE_ASSIGN_OR_RETURN(auto pipeline,
                        create_pipeline(program.runtime->device,
                                        {shader, 4, 12, specialization}));
  dispatch.pipeline = std::move(pipeline);
  dispatch.tensor_byte_origins = true;
  dispatch.push.resize(12);
  const auto &limits = program.runtime->device->properties.limits;
  const auto x = (std::min)(groups, limits.maxComputeWorkGroupCount[0]);
  if (x == 0 || (groups + x - 1) / x > limits.maxComputeWorkGroupCount[1])
    return Status::OutOfRange("Vulkan tensor dispatch exceeds device limits");
  dispatch.groups = {x, (groups + x - 1) / x, 1};
  program.dispatches.push_back(std::move(dispatch));
  return Status::Ok();
}

Status lower_reduction(ProgramState &program, const PlanStep &step) {
  static_assert(static_cast<unsigned>(ReductionKind::kMaximum) == 3);
  static_assert(static_cast<unsigned>(SoftmaxKind::kLogSoftmax) == 1);
  static_assert(static_cast<unsigned>(MaskKind::kBoolean) == 2);
  // Copy descriptions: an internal row-statistics value can grow values.
  const auto input = program.values[step.inputs[0]].description;
  const auto output = program.values[step.outputs[0]].description;
  const auto *reduce = step.operation.get_if<Reduce>();
  const auto *softmax = step.operation.get_if<Softmax>();
  const auto type = *tensor_storage_type(input.data_type());
  const auto accumulation =
      reduce ? reduce->accumulation_type : softmax->accumulation_type;
  if (type > 2 || output.data_type() != input.data_type() ||
      (accumulation != DataType::kFloat32 &&
       accumulation != DataType::kUnknown) ||
      (type != 0 && accumulation != DataType::kFloat32))
    return Status::Unimplemented("Vulkan reduction requires F32/F16/BF16 with "
                                 "F32 accumulation, explicit for half");
  if (softmax && step.inputs.size() == 2 &&
      program.values[step.inputs[1]].description.data_type() !=
          (softmax->mask == MaskKind::kBoolean ? DataType::kBool
                                               : input.data_type()))
    return Status::Unimplemented(
        "Vulkan Softmax requires Bool or same-type additive mask");
  const auto count = *output.element_count();
  if (count == 0)
    return Status::Ok();
  const auto input_count = *input.element_count();
  std::array<SpecializationConstant, 49> specialization{};
  Dispatch dispatch;
  HYLE_RETURN_IF_ERROR(tensor_layout(program, step.inputs, step.outputs[0],
                                     specialization, dispatch));
  specialization[0].value = reduce ? 10u : 11u;
  specialization[5].value = reduce ? static_cast<std::uint32_t>(reduce->kind)
                                   : static_cast<std::uint32_t>(softmax->kind);
  const auto rows_per_group = type == 0 ? 1u : 2u;
  specialization[9].value = rows_per_group;
  std::uint64_t width = 0, rows = count, inner = 1;
  if (reduce) {
    width = input_count / count;
    if (reduce->axes.is_all())
      specialization[45].value = ((1u << input.rank()) - 1)
                                 << (8 - input.rank());
    else
      for (auto requested : reduce->axes.values()) {
        const auto axis = requested < 0
                              ? requested + static_cast<int>(input.rank())
                              : requested;
        specialization[45].value |= 1u << (8 - input.rank() + axis);
      }
    // Reduce addresses input coordinates, while count still names the output.
    // Empty inputs only write identities and need no narrowed shape metadata.
    for (std::size_t axis = 0; axis < 8; ++axis)
      specialization[10 + axis].value =
          input_count == 0 || axis < 8 - input.rank()
              ? 1u
              : static_cast<std::uint32_t>(
                    input.dimensions()[axis - (8 - input.rank())]);
  } else {
    const auto requested = softmax->axis;
    const auto axis = static_cast<std::size_t>(
        requested < 0 ? requested + static_cast<int>(input.rank()) : requested);
    width = static_cast<std::uint64_t>(input.dimensions()[axis]);
    rows = count / width;
    for (std::size_t dim = axis + 1; dim < input.rank(); ++dim)
      inner *= static_cast<std::uint64_t>(input.dimensions()[dim]);
    specialization[7].value = std::bit_cast<std::uint32_t>(softmax->scale);
    specialization[46].value = static_cast<std::uint32_t>(softmax->mask);
  }
  specialization[42].value = static_cast<std::uint32_t>(width);
  specialization[43].value = static_cast<std::uint32_t>(rows);
  specialization[44].value = static_cast<std::uint32_t>(inner);
  const auto &limits = program.runtime->device->properties.limits;
  const bool cached = softmax && width <= 2048 &&
                      width * 8 + 2048 <= limits.maxComputeSharedMemorySize;
  specialization[47].value = cached ? static_cast<std::uint32_t>(width) : 1u;
  specialization[48].value = cached ? 1u : 0u;
  const auto groups =
      static_cast<std::uint32_t>((rows + rows_per_group - 1) / rows_per_group);
  if (softmax && type != 0 && inner > 1 && inner % 2 != 0) {
    // Odd inner strides pair output halves from different mathematical rows.
    // Store only row statistics, then assign one writer per final output word.
    if (rows > limits.maxStorageBufferRange / 8)
      return Status::Unimplemented(
          "Vulkan Softmax row statistics exceed device range");
    HYLE_ASSIGN_OR_RETURN(
        auto statistics_desc,
        TensorDesc::dense(
            DataType::kFloat32,
            std::array<std::int64_t, 2>{static_cast<std::int64_t>(rows), 2}));
    const auto statistics_id =
        static_cast<std::uint32_t>(program.values.size());
    program.values.push_back({statistics_desc, {}});
    auto statistics_dispatch = dispatch;
    statistics_dispatch.values[3] = statistics_id;
    specialization[0].value = 12;
    HYLE_RETURN_IF_ERROR(
        append_tensor_dispatch(program, std::move(statistics_dispatch),
                               shaders::reduction(), specialization, groups));
    dispatch.values[2] = statistics_id;
    specialization[0].value = 13;
    specialization[47].value = 1;
    specialization[48].value = 0;
    const auto words = (*output.required_bytes() + 3) / 4;
    return append_tensor_dispatch(
        program, std::move(dispatch), shaders::reduction(), specialization,
        static_cast<std::uint32_t>((words + 127) / 128));
  }
  return append_tensor_dispatch(program, std::move(dispatch),
                                shaders::reduction(), specialization, groups);
}

Status lower_concat(ProgramState &program, const PlanStep &step) {
  const auto &output = program.values[step.outputs[0]].description;
  const auto elements = *output.element_count();
  if (elements == 0)
    return Status::Ok();
  auto selected_axis = step.operation.get_if<Concat>()->axis;
  if (selected_axis < 0)
    selected_axis += static_cast<std::int32_t>(output.rank());
  const auto axis = static_cast<std::size_t>(selected_axis);
  std::uint64_t inner = 1;
  for (std::size_t dim = axis + 1; dim < output.rank(); ++dim)
    inner *= static_cast<std::uint64_t>(output.dimensions()[dim]);
  const auto row_elements =
      inner * static_cast<std::uint64_t>(output.dimensions()[axis]);
  const auto outer = elements / row_elements;
  const auto size = data_type_size(output.data_type());
  const auto row_bytes = row_elements * size;
  std::uint64_t prefix = 0;
  for (std::size_t begin = 0; begin < step.inputs.size(); begin += 3) {
    const auto inputs =
        std::span(step.inputs)
            .subspan(begin,
                     (std::min)(std::size_t{3}, step.inputs.size() - begin));
    const auto group_begin = prefix;
    std::array<std::uint64_t, 3> starts{}, ends{};
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      starts[port] = prefix;
      prefix += static_cast<std::uint64_t>(
          program.values[inputs[port]].description.dimensions()[axis]);
      ends[port] = prefix;
    }
    for (auto port = inputs.size(); port < 3; ++port)
      ends[port] = prefix;
    if (prefix == group_begin)
      continue;
    std::array<SpecializationConstant, 53> specialization{};
    Dispatch dispatch;
    HYLE_RETURN_IF_ERROR(tensor_layout(program, inputs, step.outputs[0],
                                       specialization, dispatch));
    specialization[0].value = 9;
    specialization[42].value = static_cast<std::uint32_t>(inner);
    specialization[43].value = static_cast<std::uint32_t>(row_elements);
    specialization[44].value = static_cast<std::uint32_t>(group_begin);
    for (std::size_t port = 0; port < 3; ++port) {
      specialization[45 + port].value = static_cast<std::uint32_t>(ends[port]);
      if (port < inputs.size()) {
        const auto stride =
            specialization[18 + port * 8 + 8 - output.rank() + axis].value;
        // U32 modular subtraction cancels the global axis prefix. The selected
        // input's final address is within its independently validated span.
        specialization[48 + port].value =
            static_cast<std::uint32_t>(starts[port] * stride);
      }
    }
    const auto start_bytes = group_begin * inner * size;
    const auto segment_bytes = (prefix - group_begin) * inner * size;
    std::uint64_t max_prefix = 0;
    for (std::uint64_t row = 0; row < (std::min)(outer, std::uint64_t{4});
         ++row)
      max_prefix = (std::max)(max_prefix, (row * row_bytes + start_bytes) % 4);
    const auto word_slots = (max_prefix + segment_bytes + 3) / 4;
    const auto work = outer * word_slots;
    if (!std::in_range<std::uint32_t>(work))
      return Status::Unimplemented("Vulkan Concat word schedule exceeds U32");
    specialization[51].value = static_cast<std::uint32_t>(word_slots);
    specialization[52].value = static_cast<std::uint32_t>(work);
    HYLE_RETURN_IF_ERROR(append_tensor_dispatch(
        program, std::move(dispatch), shaders::concat(), specialization,
        static_cast<std::uint32_t>((work + 127) / 128)));
  }
  return Status::Ok();
}
} // namespace

Status lower_index_check(ProgramState &program, std::uint32_t input,
                         std::uint32_t output, std::uint64_t extent,
                         bool checked) {
  std::array<SpecializationConstant, 42> specialization{};
  Dispatch dispatch;
  HYLE_RETURN_IF_ERROR(tensor_layout(program, std::array{input}, output,
                                     specialization, dispatch));
  specialization[0].value = 14;
  specialization[5].value = checked ? 1u : 0u;
  specialization[7].value = static_cast<std::uint32_t>(extent);
  specialization[8].value = static_cast<std::uint32_t>(extent >> 32);
  if (checked) {
    dispatch.values[2] = *program.status_value;
    dispatch.checked_status = true;
  }
  const auto count = *program.values[output].description.element_count();
  return append_tensor_dispatch(
      program, std::move(dispatch), shaders::checked_tensor(), specialization,
      static_cast<std::uint32_t>((count + 127) / 128));
}

Status lower_checked_copy(ProgramState &program, std::uint32_t input,
                          std::uint32_t output) {
  const auto bytes = *program.values[output].description.required_bytes();
  if (bytes == 0)
    return Status::Ok();
  std::array<SpecializationConstant, 42> specialization{};
  Dispatch dispatch;
  HYLE_RETURN_IF_ERROR(tensor_layout(program, std::array{input}, output,
                                     specialization, dispatch));
  specialization[5].value = 1;
  dispatch.values[2] = *program.status_value;
  dispatch.checked_status = true;
  return append_tensor_dispatch(
      program, std::move(dispatch), shaders::checked_tensor(), specialization,
      static_cast<std::uint32_t>(((bytes + 3) / 4 + 127) / 128));
}

Status lower_tensor(ProgramState &program, const PlanStep &step) {
  if (step.operation.kind() == OperationKind::kReduce ||
      step.operation.kind() == OperationKind::kSoftmax)
    return lower_reduction(program, step);
  if (step.operation.kind() == OperationKind::kConcat)
    return lower_concat(program, step);
  std::uint32_t operation = 0, mode = 0;
  std::uint64_t fill = 0;
  switch (step.operation.kind()) {
  case OperationKind::kCopy:
    operation = 0;
    break;
  case OperationKind::kFill:
    operation = 1;
    fill = step.operation.get_if<Fill>()->value.bits();
    break;
  case OperationKind::kCast:
    operation = 2;
    break;
  case OperationKind::kUnary:
    operation = 3;
    mode = static_cast<std::uint32_t>(step.operation.get_if<Unary>()->kind);
    break;
  case OperationKind::kBinary:
    operation = 4;
    mode = static_cast<std::uint32_t>(step.operation.get_if<Binary>()->kind);
    break;
  case OperationKind::kSelect:
    operation = 5;
    break;
  case OperationKind::kGatedActivation: {
    operation = 6;
    const auto activation =
        step.operation.get_if<GatedActivation>()->activation;
    mode = activation == GatedActivationKind::kSilu   ? 11u
           : activation == GatedActivationKind::kGelu ? 9u
                                                      : 10u;
    break;
  }
  case OperationKind::kRmsNorm:
  case OperationKind::kLayerNorm: {
    const auto *rms = step.operation.get_if<RmsNorm>();
    const auto *layer = step.operation.get_if<LayerNorm>();
    const auto accumulation =
        rms ? rms->accumulation_type : layer->accumulation_type;
    if (accumulation != DataType::kFloat32)
      return Status::Unimplemented(
          "Vulkan normalization requires F32 accumulation");
    operation = rms ? 7u : 8u;
    fill = std::bit_cast<std::uint32_t>(rms ? rms->epsilon : layer->epsilon) |
           (static_cast<std::uint64_t>(step.inputs.size()) << 32);
    break;
  }
  default:
    return Status::Unimplemented(
        "operation has no admitted Vulkan tensor lowering");
  }
  static_assert(static_cast<unsigned>(UnaryKind::kSoftplus) == 12);
  static_assert(static_cast<unsigned>(BinaryKind::kMaximum) == 5);
  const auto &output = program.values[step.outputs[0]].description;
  const auto output_type = tensor_storage_type(output.data_type());
  if (!output_type || !output.is_dense())
    return Status::Unimplemented(
        "Vulkan tensor output requires admitted dense storage");
  if ((operation >= 2 && operation <= 4) || operation >= 6) {
    if (*output_type > 2)
      return Status::Unimplemented(
          "Vulkan tensor arithmetic requires F32/F16/BF16");
    for (auto value : step.inputs) {
      const auto type =
          tensor_storage_type(program.values[value].description.data_type());
      if (!type || *type > 2)
        return Status::Unimplemented(
            "Vulkan tensor arithmetic requires F32/F16/BF16");
    }
  }
  const auto elements = *output.element_count();
  if (elements == 0)
    return Status::Ok();
  const bool normalization = operation == 7 || operation == 8;
  const auto rows_per_group =
      normalization && *output_type != 0 && output.dimensions().back() % 2 != 0
          ? 2u
          : 1u;
  if (operation == 6 && step.inputs.size() == 1) {
    const auto &input = program.values[step.inputs[0]].description;
    const auto half = static_cast<std::uint64_t>(output.dimensions().back()) *
                      static_cast<std::uint64_t>(input.byte_strides().back());
    if (!std::in_range<std::uint32_t>(half))
      return Status::Unimplemented("Vulkan packed gate offset exceeds U32");
    fill = half | (std::uint64_t{1} << 32);
  }
  // Specialization freezes types, modes and address mapping into the native
  // pipeline. No layout metadata upload or interpretation is added to replay.
  std::array<SpecializationConstant, 42> specialization{};
  Dispatch dispatch;
  HYLE_RETURN_IF_ERROR(tensor_layout(program, step.inputs, step.outputs[0],
                                     specialization, dispatch));
  specialization[0].value = operation;
  specialization[5].value = mode;
  specialization[7].value = static_cast<std::uint32_t>(fill);
  specialization[8].value = static_cast<std::uint32_t>(fill >> 32);
  // Odd-width half rows share an output word. One group owns both rows.
  specialization[9].value = rows_per_group;
  const auto words = (*output.required_bytes() + 3) / 4;
  const auto groups = static_cast<std::uint32_t>(
      normalization
          ? (elements / output.dimensions().back() + rows_per_group - 1) /
                rows_per_group
          : (words + 127) / 128);
  return append_tensor_dispatch(program, std::move(dispatch),
                                normalization ? shaders::normalization()
                                              : shaders::tensor(),
                                specialization, groups);
}
} // namespace hyle::compute::vulkan::detail
