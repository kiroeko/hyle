#include "vulkan_program_internal.h"
#include "vulkan_shaders.h"
#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <new>

namespace hyle::compute::vulkan::detail {
namespace {

inline constexpr std::array kNvidiaSubgroupPredicates{
    RouteAttributePredicate::number(route_attribute::kDeviceVendorId, 0x10de),
    RouteAttributePredicate::number(route_attribute::kDeviceId, 0x2786),
    RouteAttributePredicate::number(route_attribute::kSubgroupSize, 32),
    RouteAttributePredicate::number(kSubgroupArithmeticAttribute, 1)};
inline constexpr RouteRule kNvidiaSubgroupRule{
    .route = NativeRoute::kVulkan,
    .attributes = kNvidiaSubgroupPredicates,
    .operation = OperationKind::kMatMul,
    .input0_type = DataType::kFloat32,
    .input1_type = DataType::kFloat32,
    .output0_type = DataType::kFloat32};
inline constexpr std::array kAppleSubgroupPredicates{
    RouteAttributePredicate::number(route_attribute::kDeviceVendorId, 0x106b),
    RouteAttributePredicate::string(route_attribute::kDeviceName, "Apple M4"),
    RouteAttributePredicate::number(route_attribute::kSubgroupSize, 32),
    RouteAttributePredicate::number(kSubgroupArithmeticAttribute, 1)};
inline constexpr RouteRule kAppleSubgroupRule{
    .route = NativeRoute::kVulkan,
    .attributes = kAppleSubgroupPredicates,
    .operation = OperationKind::kMatMul,
    .input0_type = DataType::kFloat32,
    .input1_type = DataType::kFloat32,
    .output0_type = DataType::kFloat32};

// Push layouts, fast division and non-cooperative F32 tile recipes follow
// ggml 456172ec733a. See third_party/ggml_vulkan_shaders/NOTICE.md and LICENSE.
std::array<std::uint32_t, 2> fastdiv(std::uint32_t divisor) noexcept {
  const auto shift = static_cast<std::uint32_t>(std::bit_width(divisor - 1));
  const auto multiplier = static_cast<std::uint32_t>(
      ((std::uint64_t{1} << 32) * ((std::uint64_t{1} << shift) - divisor)) /
          divisor +
      1);
  return {multiplier, shift};
}

template <std::size_t N>
std::vector<std::byte> push_bytes(const std::array<std::uint32_t, N> &words) {
  std::vector<std::byte> bytes(sizeof(words));
  std::memcpy(bytes.data(), words.data(), bytes.size());
  return bytes;
}

// The upstream flat shaders use z*262144 + y*512 + x, with local_size_x=512.
// This mapping also covers large weight transposes beyond 65535 X workgroups.
std::array<std::uint32_t, 3> flat_groups(std::uint32_t elements) noexcept {
  return {1, (std::min)((elements + 511) / 512, 512u),
          (elements + 262143) / 262144};
}

Status check_flat_shader(const Device &device) noexcept {
  if (device.properties.limits.maxComputeWorkGroupInvocations < 512 ||
      device.properties.limits.maxComputeWorkGroupSize[0] < 512)
    return Status::Unavailable(
        "ggml Vulkan flat shader requires a 512-lane workgroup");
  return Status::Ok();
}

bool f32_matrix(const TensorDesc &description) noexcept {
  return description.valid() && description.data_type() == DataType::kFloat32 &&
         description.is_plain() && description.rank() == 2 &&
         description.dimensions()[0] > 0 && description.dimensions()[1] > 0;
}

// Interpret a physical row-major [K,N] source through an [N,K] logical copy
// view; the original public MatMul argument remains [K,N].
Result<Dispatch> transpose_dispatch(ProgramState &program, std::uint32_t source,
                                    std::uint32_t destination, std::uint32_t k,
                                    std::uint32_t n) noexcept {
  auto status = check_flat_shader(*program.runtime->device);
  if (!status.ok())
    return std::unexpected(status);
  auto pipeline = create_pipeline(program.runtime->device,
                                  {shaders::copy_f32(), 2, 120, {}});
  if (!pipeline)
    return std::unexpected(std::move(pipeline).error());
  try {
    std::array<std::uint32_t, 30> words{};
    const auto elements = k * n;
    words[0] = elements;
    // Source: ne00..03, nb00..03, measured in F32 elements.
    words[1] = k;
    words[2] = n;
    words[3] = 1;
    words[4] = 1;
    words[5] = n;
    words[6] = 1;
    words[7] = elements;
    words[8] = elements;
    words[9] = k;
    words[10] = n;
    words[11] = 1;
    words[12] = 1;
    words[13] = 1;
    words[14] = k;
    words[15] = elements;
    words[16] = elements;
    const auto outer = fastdiv(elements);
    const auto inner = fastdiv(k);
    for (std::size_t base : {22u, 26u}) {
      words[base] = outer[0];
      words[base + 1] = outer[0];
      words[base + 2] = inner[0];
      words[base + 3] = outer[1] | (outer[1] << 8) | (inner[1] << 16);
    }
    Dispatch dispatch;
    dispatch.pipeline = std::move(*pipeline);
    dispatch.values = {source, destination};
    dispatch.push = push_bytes(words);
    dispatch.groups = flat_groups(elements);
    return dispatch;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan layout preparation"));
  }
}

Result<Tensor> pack_constant(ProgramState &program, const Tensor &source,
                             const TensorDesc &description,
                             Dispatch dispatch) noexcept {
  try {
    auto bytes = description.required_bytes();
    if (!bytes)
      return std::unexpected(std::move(bytes).error());
    auto buffer = allocate_buffer(program.runtime->device, *bytes, false);
    if (!buffer)
      return std::unexpected(std::move(buffer).error());
    auto output = Access::tensor(description, *buffer);
    auto run = create_run(program.runtime);
    if (!run)
      return std::unexpected(std::move(run).error());
    const std::array<PlanValue, 2> values{
        {{source.description(), source}, {description, output}}};
    dispatch.values = {0, 1};
    (*run)->retained_buffers = {Access::buffer(source), *buffer};
    // The pipeline must survive even when an observation error quarantines
    // accepted native work past this cold preparation call.
    (*run)->recording_owner = dispatch.pipeline;
    auto status = record(*run, values, std::span<const Dispatch>(&dispatch, 1));
    if (!status.ok())
      return std::unexpected(status);
    Result<Completion> completion;
    {
      std::lock_guard lock(program.runtime->mutex);
      completion = submit_locked(*run);
    }
    if (!completion)
      return std::unexpected(std::move(completion).error());
    status = completion->wait();
    if (!status.ok())
      return std::unexpected(status);
    return output;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan constant packing metadata"));
  }
}

Status
lower_matmul(ProgramState &program, const PlanStep &step,
             std::vector<std::pair<std::uint32_t, std::uint32_t>> &packs) {
  const auto *operation = step.operation.get_if<MatMul>();
  if (operation == nullptr ||
      operation->accumulation_type != DataType::kFloat32 ||
      step.inputs.size() != 2)
    return Status::Unimplemented(
        "Vulkan MatMul requires two F32 inputs and F32 accumulation");
  const auto a = program.values[step.inputs[0]].description;
  const auto b = program.values[step.inputs[1]].description;
  if (!f32_matrix(a) || !a.is_dense() || !f32_matrix(b))
    return Status::Unimplemented(
        "Vulkan MatMul requires a dense 2-D F32 activation");
  const auto m = static_cast<std::uint32_t>(a.dimensions()[0]);
  const auto k = static_cast<std::uint32_t>(a.dimensions()[1]);
  const auto n = static_cast<std::uint32_t>(b.dimensions()[1]);
  std::uint32_t weights = step.inputs[1];
  const bool packed = b.byte_strides()[0] == 4 &&
                      b.byte_strides()[1] == static_cast<std::int64_t>(k) * 4;
  if (!packed) {
    if (!b.is_dense())
      return Status::Unimplemented(
          "Vulkan MatMul RHS requires dense or transposed dense F32");
    const auto constant = program.values[weights].constant;
    bool reused = false;
    if (constant) {
      for (auto [source, result] : packs) {
        const auto &previous = program.values[source].constant;
        if (previous.native_buffer() == constant.native_buffer() &&
            previous.native_offset() == constant.native_offset() &&
            previous.description() == constant.description()) {
          weights = result;
          reused = true;
          break;
        }
      }
    }
    if (!reused) {
      const std::array<std::int64_t, 2> dimensions{k, n};
      const std::array<std::int64_t, 2> strides{
          4, static_cast<std::int64_t>(k) * 4};
      auto description =
          TensorDesc::strided_bytes(DataType::kFloat32, dimensions, strides);
      if (!description)
        return std::move(description).error();
      const auto original = weights;
      weights = static_cast<std::uint32_t>(program.values.size());
      auto transpose = transpose_dispatch(program, original, weights, k, n);
      if (!transpose)
        return std::move(transpose).error();
      if (constant) {
        auto converted = pack_constant(program, constant, *description,
                                       std::move(*transpose));
        if (!converted)
          return std::move(converted).error();
        program.values.push_back({*description, std::move(*converted)});
        packs.emplace_back(original, weights);
      } else {
        program.values.push_back({*description, {}});
        program.dispatches.push_back(std::move(*transpose));
      }
    }
  }
  auto &device = *program.runtime->device;
  Dispatch dispatch;
  if (m <= 8) {
    // Preserve mature single/few-vector reduction. Qualified NVIDIA shapes
    // use the upstream larger-group rule; unknown devices keep the legal
    // shared-memory baseline until their own performance qualification.
    const auto facts = route_facts(device.environment);
    const std::array route_inputs{a, b};
    const std::array route_outputs{program.values[step.outputs[0]].description};
    const RouteMatchInput route_input{.platform = kBuildRoutePlatform,
                                      .route = NativeRoute::kVulkan,
                                      .attributes = facts.attributes,
                                      .model =
                                          program.workload.model.description(),
                                      .phase = program.workload.phase,
                                      .region = program.workload.region,
                                      .operation = step.operation,
                                      .inputs = route_inputs,
                                      .outputs = route_outputs};
    const bool nvidia_subgroup =
        match_route_rule(kNvidiaSubgroupRule, route_input).matched();
    const bool known_subgroup =
        nvidia_subgroup ||
        match_route_rule(kAppleSubgroupRule, route_input).matched();
    const bool large = nvidia_subgroup && n <= 8192 && k >= 1024;
    const std::uint32_t block = large ? 128u : 32u;
    const auto shader = !known_subgroup ? shaders::mv_f32()
                        : large         ? shaders::mv_f32_hybrid()
                                        : shaders::mv_f32_subgroup();
    if (block > device.properties.limits.maxComputeWorkGroupInvocations ||
        block > device.properties.limits.maxComputeWorkGroupSize[0] ||
        block * m * sizeof(float) >
            device.properties.limits.maxComputeSharedMemorySize)
      return Status::Unavailable("Vulkan matvec workgroup limits");
    const std::array<SpecializationConstant, 3> specialization{
        {{0, block}, {1, 1}, {2, m}}};
    auto pipeline = create_pipeline(program.runtime->device,
                                    {shader, 5, 52, specialization});
    if (!pipeline)
      return std::move(pipeline).error();
    const std::array<std::uint32_t, 13> push{k, k, k, n, k * n, k, n,
                                             0, 0, 1, 1, 1,     1};
    dispatch.pipeline = std::move(*pipeline);
    dispatch.values = {weights, step.inputs[0], step.outputs[0], step.inputs[0],
                       step.inputs[0]};
    dispatch.push = push_bytes(push);
    const auto groups_x =
        (std::min)(n, device.properties.limits.maxComputeWorkGroupCount[0]);
    dispatch.groups = {groups_x, 1, (n + groups_x - 1) / groups_x};
  } else {
    std::uint32_t tile = m <= 32 || n <= 32   ? 32u
                         : m <= 64 || n <= 64 ? 64u
                                              : 128u;
    // DATA_A_F32 hard-codes BK=32 in the shader. The host recipe's ID3=16 is
    // inactive for this variant; limits must follow the actual shared arrays.
    constexpr std::uint32_t reduction_tile = 32;
    constexpr std::uint32_t shared_row_bytes = (reduction_tile / 2 + 1) * 8;
    while (tile > 32 &&
           2 * tile * shared_row_bytes > device.environment.shared_memory_bytes)
      tile /= 2;
    const std::uint32_t block = tile == 32 ? 32u : 128u;
    if (2 * tile * shared_row_bytes > device.environment.shared_memory_bytes ||
        block > device.properties.limits.maxComputeWorkGroupInvocations ||
        block > device.properties.limits.maxComputeWorkGroupSize[0])
      return Status::Unavailable(
          "Vulkan F32 matmul tile exceeds device limits");
    // Upstream aligned loads omit edge checks. A native borrowed range carries
    // no hidden padding promise, so only complete tiles may use that variant.
    // The native bind/borrow contract then guarantees vec4-aligned offsets.
    const bool aligned =
        m % tile == 0 && n % tile == 0 && k % tile == 0 &&
        device.properties.limits.minStorageBufferOffsetAlignment >= 16;
    const std::array<std::uint32_t, 12> recipe{block,
                                               tile,
                                               tile,
                                               16, // Upstream ID3; inactive for F32.
                                               tile == 128 ? 64u : 32u,
                                               tile == 128 ? 64u : 32u,
                                               2,
                                               tile == 32 ? 2u : 4u,
                                               tile == 128 ? 4u : 2u,
                                               1,
                                               32,
                                               aligned ? 1u : 0u};
    std::array<SpecializationConstant, 12> specialization{};
    for (std::uint32_t index = 0; index < specialization.size(); ++index)
      specialization[index] = {index, recipe[index]};
    auto pipeline = create_pipeline(program.runtime->device,
                                    {shaders::mm_f32(), 3, 64, specialization});
    if (!pipeline)
      return std::move(pipeline).error();
    // One Z workgroup owns the entire reduction. ggml's k_split is the
    // reduction span, not the number of partitions; 1 would compute one term.
    const std::array<std::uint32_t, 16> push{
        n, m, k, k, k, n, k * n, k * m, n * m, 0, 1, k, 1, 1, 1, 1};
    dispatch.pipeline = std::move(*pipeline);
    dispatch.values = {weights, step.inputs[0], step.outputs[0]};
    dispatch.push = push_bytes(push);
    dispatch.groups = {(n + tile - 1) / tile, (m + tile - 1) / tile, 1};
  }
  program.dispatches.push_back(std::move(dispatch));
  return Status::Ok();
}

bool mature_gate_eligible(const ProgramState &program, const PlanStep &step) {
  const auto &operation = *step.operation.get_if<GatedActivation>();
  const auto &output = program.values[step.outputs[0]].description;
  if (operation.activation != GatedActivationKind::kSilu ||
      output.data_type() != DataType::kFloat32 || output.rank() > 4 ||
      *output.element_count() == 0 ||
      !check_flat_shader(*program.runtime->device).ok())
    return false;
  const auto alignment =
      (std::max)(VkDeviceSize{4}, program.runtime->device->properties.limits
                                      .minStorageBufferOffsetAlignment);
  for (const auto input : step.inputs) {
    const auto &value = program.values[input];
    if (!value.description.is_dense())
      return false;
    if (value.constant) {
      if (value.constant.native_offset() % alignment != 0)
        return false;
    } else {
      const auto alias = program.aliases[input];
      if (std::ranges::find(program.inputs, alias.root) !=
              program.inputs.end() ||
          alias.byte_offset % alignment != 0)
        return false; // Dynamic base origins are not known during preparation.
    }
  }
  return true;
}

Status lower_gate(ProgramState &program, const PlanStep &step) {
  const auto *operation = step.operation.get_if<GatedActivation>();
  if (operation == nullptr ||
      operation->activation != GatedActivationKind::kSilu ||
      program.values[step.inputs[0]].description.data_type() !=
          DataType::kFloat32)
    return Status::Unimplemented(
        "Vulkan gated activation currently supports SiLU");
  auto status = check_flat_shader(*program.runtime->device);
  if (!status.ok())
    return status;
  const auto &output = program.values[step.outputs[0]].description;
  const auto &a = program.values[step.inputs[0]].description;
  const auto &b = program.values[step.inputs.back()].description;
  if (a.rank() == 0 || a.rank() > 4 || !a.is_dense() || !b.is_dense() ||
      !output.is_dense())
    return Status::Unimplemented(
        "Vulkan SiLU gate requires dense F32 tensors of rank 1..4");
  auto pipeline = create_pipeline(program.runtime->device,
                                  {shaders::swiglu_f32(), 3, 108, {}});
  if (!pipeline)
    return std::move(pipeline).error();
  std::array<std::uint32_t, 27> push{};
  const auto elements = static_cast<std::uint32_t>(*output.element_count());
  push[0] = elements;
  push[1] = static_cast<std::uint32_t>(a.dimensions().back());
  push[2] = static_cast<std::uint32_t>(output.dimensions().back());
  push[3] = operation->inputs == GatedInputKind::kSeparate ? 2u : 0u;
  std::array<std::uint32_t, 4> dimensions{1, 1, 1, 1};
  for (std::size_t index = 0; index < output.rank(); ++index)
    dimensions[index] = static_cast<std::uint32_t>(
        output.dimensions()[output.rank() - index - 1]);
  const std::array<const TensorDesc *, 3> descriptions{&a, &b, &output};
  for (std::size_t port = 0; port < descriptions.size(); ++port) {
    const auto &description = *descriptions[port];
    const auto storage_elements =
        static_cast<std::uint32_t>(*description.element_count());
    for (std::size_t axis = 0; axis < 4; ++axis)
      push[6 + port * 4 + axis] =
          axis < description.rank()
              ? static_cast<std::uint32_t>(
                    description.byte_strides()[description.rank() - axis - 1] /
                    4)
              : storage_elements;
  }
  push[18] = dimensions[1];
  push[19] = dimensions[2];
  const auto outer = fastdiv(dimensions[2] * dimensions[1] * dimensions[0]);
  const auto middle = fastdiv(dimensions[1] * dimensions[0]);
  const auto inner = fastdiv(dimensions[0]);
  push[21] = outer[0];
  push[22] = outer[1];
  push[23] = middle[0];
  push[24] = middle[1];
  push[25] = inner[0];
  push[26] = inner[1];
  Dispatch dispatch;
  dispatch.pipeline = std::move(*pipeline);
  dispatch.values = {step.inputs[0], step.inputs.back(), step.outputs[0]};
  dispatch.push = push_bytes(push);
  dispatch.groups = flat_groups(elements);
  program.dispatches.push_back(std::move(dispatch));
  return Status::Ok();
}

} // namespace

Status lower_operation(
    ProgramState &program, const PlanStep &step,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> &packs) noexcept {
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
            step.groups[axis] > program.runtime->device->properties.limits
                                    .maxComputeWorkGroupCount[axis])
          return Status::InvalidArgument("Vulkan custom dispatch group count");
      Dispatch dispatch;
      dispatch.pipeline = pipeline;
      dispatch.values = step.inputs;
      dispatch.values.push_back(step.outputs[0]);
      dispatch.groups = step.groups;
      dispatch.push = step.push_constants;
      program.dispatches.push_back(std::move(dispatch));
      return Status::Ok();
    }
    if (step.operation.kind() == OperationKind::kMatMul)
      return lower_matmul(program, step, packs);
    if (step.operation.kind() == OperationKind::kGatedActivation &&
        mature_gate_eligible(program, step))
      return lower_gate(program, step);
    return lower_tensor(program, step);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan typed operation metadata");
  }
}

} // namespace hyle::compute::vulkan::detail
