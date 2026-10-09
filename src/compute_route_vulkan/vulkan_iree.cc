#include "vulkan_iree.h"

#include <algorithm>
#include <cstring>
#include <new>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace hyle::compute::vulkan::detail {
namespace iv = iree_vulkan;
namespace {

std::uint64_t admitted_elements(const Device &device) noexcept {
  const auto &limits = device.properties.limits;
  const auto grid = (std::min)({limits.maxComputeWorkGroupCount[0],
                                limits.maxComputeWorkGroupCount[1],
                                limits.maxComputeWorkGroupCount[2]});
  return (std::min)(std::uint64_t{grid},
                    std::uint64_t{limits.maxStorageBufferRange} / 8);
}

std::uint64_t precision_elements(const Device &device) noexcept {
  const auto &limits = device.properties.limits;
  // Both locked conversion entries have 64 lanes and one flat logical tile
  // per workgroup. Retain the compiler target's 65535-group bound as well.
  const auto grid = (std::min)({limits.maxComputeWorkGroupCount[0],
                                limits.maxComputeWorkGroupCount[1],
                                limits.maxComputeWorkGroupCount[2], 65535U});
  return (std::min)({std::uint64_t{grid} * 64,
                     std::uint64_t{limits.maxStorageBufferRange} / 4,
                     std::uint64_t{INT32_MAX} / 8});
}

bool fp8_dispatch_fits(const VkPhysicalDeviceLimits &limits,
                       const iv::ProgramSpec &spec) noexcept {
  if (spec.rows <= 0 || spec.columns <= 0 || spec.depth <= 0 ||
      limits.maxComputeWorkGroupCount[1] < 1 ||
      limits.maxComputeWorkGroupCount[2] < 1)
    return false;
  const auto m = static_cast<std::uint64_t>(spec.rows);
  const auto n = static_cast<std::uint64_t>(spec.columns);
  const auto k = static_cast<std::uint64_t>(spec.depth);
  // Retain the original program's signed 32-bit indexing domain, including
  // both F32 matrices and 64-byte padding in their shared allocation.
  constexpr auto max_elements = std::uint64_t{INT32_MAX - 128} / 8;
  const auto fits = [](std::uint64_t a, std::uint64_t b) {
    return a <= max_elements / b;
  };
  if (!fits(m, k) || !fits(n, k) || !fits(m, n))
    return false;
  const auto align64 = [](std::uint64_t bytes) {
    return (bytes + 63) & ~std::uint64_t{63};
  };
  const auto workspace = align64(4 * m * k) + align64(4 * n * k);
  if (workspace > limits.maxStorageBufferRange ||
      4 * m * n > limits.maxStorageBufferRange)
    return false;
  // Locked fp8.vmfb: conversion A, conversion W, then MatMul. Each uses a
  // 64-lane X grid, capped at 65535 with the original compiler's strided loop.
  // The two scale modes have this same dispatch shape; no shader is changed.
  const auto grid_fits = [&](std::uint64_t rows, std::uint64_t columns) {
    return (std::min)(rows * ((columns + 63) / 64), std::uint64_t{65535}) <=
           limits.maxComputeWorkGroupCount[0];
  };
  return grid_fits(m, k) && grid_fits(n, k) && grid_fits(m, n);
}

iv::ArtifactScalar artifact_scalar(DataType type) noexcept {
  return type == DataType::kFloat16   ? iv::ArtifactScalar::Float16
         : type == DataType::kBFloat16 ? iv::ArtifactScalar::BFloat16
                                      : iv::ArtifactScalar::Float32;
}

bool artifact_dispatch_fits(const Device &device, const ProgramState &program,
                            const PlanStep &step) noexcept {
  const auto &limits = device.properties.limits;
  const auto &output = program.values[step.outputs[0]].description;
  const auto elements = *output.element_count();
  if (elements > std::uint64_t{INT32_MAX} / 8 ||
      (std::min)((elements + 63) / 64, std::uint64_t{65535}) >
          limits.maxComputeWorkGroupCount[0] ||
      limits.maxComputeWorkGroupCount[1] < 1 ||
      limits.maxComputeWorkGroupCount[2] < 1)
    return false;
  // Include descriptor alignment in the original allocator's combined input
  // and output domain. Reject before native allocation, not after an OOM probe.
  std::uint64_t bytes = (*output.required_bytes() + 63) & ~std::uint64_t{63};
  for (const auto input : step.inputs) {
    const auto size = *program.values[input].description.required_bytes();
    if (size > limits.maxStorageBufferRange)
      return false;
    bytes += (size + 63) & ~std::uint64_t{63};
  }
  return bytes <= limits.maxStorageBufferRange;
}

std::optional<Float8ScaleMode> float8_mode(const TensorDesc &description,
                                           bool weight) noexcept {
  if (description.rank() != 2 || description.is_plain())
    return std::nullopt;
  for (const auto mode : {Float8ScaleMode::kFloat32Block128,
                          Float8ScaleMode::kUnsignedExponent8Group32}) {
    const auto shape = description.dimensions();
    auto expected =
        weight ? float8_weight_description(shape[0], shape[1], mode)
               : float8_activation_description(shape[0], shape[1], mode);
    if (expected && *expected == description)
      return mode;
  }
  return std::nullopt;
}

Status begin_copy(Run &run) noexcept {
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  // The slot's pool allows implicit reset on Begin; the parent proves
  // retirement before a repeated invocation can re-record any of these command
  // buffers.
  HYLE_RETURN_IF_ERROR(native_status(
      run.runtime->device->fn.BeginCommandBuffer(run.command, &begin),
      "vkBeginCommandBuffer(IREE boundary)"));
  memory_barrier(*run.runtime->device, run.command,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
  return Status::Ok();
}

Status end_copy(Run &run) noexcept {
  memory_barrier(*run.runtime->device, run.command,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                     VK_PIPELINE_STAGE_HOST_BIT,
                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_HOST_READ_BIT);
  return native_status(run.runtime->device->fn.EndCommandBuffer(run.command),
                       "vkEndCommandBuffer(IREE boundary)");
}

Status copy_result(IreeStage &stage, bool flag) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto source, stage.program->output(flag ? 1 : 0));
  const auto &destination =
      flag ? stage.flag : stage.parent->retained_buffers[stage.output];
  const auto bytes = flag ? sizeof(std::uint32_t) : stage.output_bytes;
  if (source.bytes != bytes || !destination || destination->bytes < bytes)
    return Status::Internal(
        "IREE result storage does not match its bound contract");
  auto &run = *(flag ? stage.flag_copy : stage.output_copy);
  HYLE_RETURN_IF_ERROR(begin_copy(run));
  const VkBufferCopy copy{source.offset, destination->offset, bytes};
  run.runtime->device->fn.CmdCopyBuffer(run.command, source.buffer,
                                        destination->handle, 1, &copy);
  HYLE_RETURN_IF_ERROR(end_copy(run));
  const bool last =
      stage.parent->async->next_stage + 1 == stage.parent->stages.size();
  return submit_native(*run.runtime->device, run.command,
                       !flag && last ? stage.parent->fence : VK_NULL_HANDLE);
}

void fail_sequence(Run &run, const Status &status) noexcept {
  if (run.submission_failure.ok())
    run.submission_failure = status;
  // Only the observer may drain. A proactor callback must never block on GPU
  // work or on another callback that needs this same original runner.
  run.needs_drain = true;
  run.async->finished = true;
  if (status.native_error() ==
      NativeError{NativeErrorDomain::Vulkan, VK_ERROR_DEVICE_LOST}) {
    run.runtime->failure = status;
    run.runtime->device->provider.note_device_lost();
  }
}

} // namespace

bool named_compressed_storage(const TensorDesc &description) noexcept {
  if (gguf_quantized_type(description.storage_format()))
    return true;
  if (description.rank() != 2)
    return false;
  if (float8_mode(description, false) || float8_mode(description, true))
    return true;
  return false;
}

bool iree_precision_supported(const Device &device,
                              const TensorDesc &input) noexcept {
  const auto &features = device.provider.native();
  return (input.data_type() == DataType::kFloat32 ||
          input.data_type() == DataType::kFloat16) &&
         input.is_plain() && input.is_dense() && input.element_count() &&
         *input.element_count() > 0 &&
         *input.element_count() <= precision_elements(device) &&
         features.shader_int16 && features.storage_buffer16 &&
         features.preserve_float32_specials &&
         (input.data_type() != DataType::kFloat16 || features.shader_float16) &&
         device.properties.limits.maxComputeWorkGroupInvocations >= 64 &&
         device.properties.limits.maxComputeWorkGroupSize[0] >= 64;
}

bool iree_operation(const ProgramState &program,
                    const PlanStep &step) noexcept {
  if (step.custom || step.inputs.empty() || step.outputs.size() != 1)
    return false;
  if (step.operation.custom_as<model_ops::ArtifactWeightDecode>())
    return true; // Domain diagnostics belong to cold lowering, never a shader fallback.
  const auto &input = program.values[step.inputs[0]].description;
  if (const auto *cast = step.operation.get_if<Cast>())
    return step.inputs.size() == 1 &&
           cast->output_type == DataType::kBFloat16 &&
           iree_precision_supported(*program.runtime->device, input);
  if (const auto *scan = step.operation.get_if<Scan>())
    return input.data_type() == DataType::kInt32 && input.is_dense() &&
           *input.element_count() != 0 &&
           *input.element_count() <=
               admitted_elements(*program.runtime->device) &&
           (scan->axis == -1 ||
            scan->axis == static_cast<std::int32_t>(input.rank()) - 1);
  return step.operation.get_if<MatMul>() && step.inputs.size() == 2 &&
         float8_mode(input, false).has_value();
}

Status lower_iree(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto &input = program.values[step.inputs[0]].description;
    iv::ProgramSpec spec;
    const auto *scan = step.operation.get_if<Scan>();
    const bool precision = step.operation.get_if<Cast>() != nullptr;
    const auto artifact = step.operation.custom_as<model_ops::ArtifactWeightDecode>();
    if (artifact) {
      using Format = model_ops::WeightArtifactFormat;
      spec.kind = artifact->format == Format::kNf4 ? iv::ProgramKind::Nf4Decode
                  : artifact->format == Format::kNf4DoubleQuant
                      ? iv::ProgramKind::Nf4NestedDecode
                  : artifact->format == Format::kAwqGemm4
                      ? iv::ProgramKind::AwqDecode
                      : iv::ProgramKind::Symmetric4Decode;
      const auto &output = program.values[step.outputs[0]].description;
      spec.rows = output.dimensions()[0];
      spec.columns = output.dimensions()[1];
      spec.group_size = artifact->group_size;
      spec.output_type = artifact_scalar(output.data_type());
      if (spec.kind == iv::ProgramKind::AwqDecode ||
          spec.kind == iv::ProgramKind::Symmetric4Decode)
        spec.scale_type = artifact_scalar(
            program.values[step.inputs[spec.kind == iv::ProgramKind::AwqDecode ? 2 : 1]]
                .description.data_type());
    } else if (precision) {
      if (!iree_operation(program, step))
        return Status::Unimplemented("Vulkan IREE conversion domain");
      spec.kind = input.data_type() == DataType::kFloat16
                      ? iv::ProgramKind::Float16ToBFloat16
                      : iv::ProgramKind::Float32ToBFloat16;
      spec.rows = 1;
      spec.columns = static_cast<std::int64_t>(*input.element_count());
    } else if (scan) {
      spec.kind = scan->exclusive ? iv::ProgramKind::ScanExclusive
                                  : iv::ProgramKind::ScanInclusive;
      spec.columns = input.dimensions().back();
      spec.rows =
          static_cast<std::int64_t>(*input.element_count()) / spec.columns;
    } else {
      const auto mode = float8_mode(input, false);
      const auto &weight = program.values[step.inputs[1]].description;
      if (!mode || float8_mode(weight, true) != mode ||
          step.operation.get_if<MatMul>()->accumulation_type !=
              DataType::kFloat32)
        return Status::Unimplemented(
            "Vulkan FP8 requires matching named scales and F32 accumulation");
      spec.kind = *mode == Float8ScaleMode::kFloat32Block128
                      ? iv::ProgramKind::Float8Block128
                      : iv::ProgramKind::Float8Group32;
      spec.rows = input.dimensions()[0];
      spec.depth = input.dimensions()[1];
      spec.columns = weight.dimensions()[1];
    }
    auto &device = *program.runtime->device;
    const auto &features = device.provider.native();
    if ((scan && !features.shader_int64) ||
        (precision && (!features.shader_int16 || !features.storage_buffer16 ||
                       !features.preserve_float32_specials ||
                       (input.data_type() == DataType::kFloat16 &&
                        !features.shader_float16))) ||
        (!scan && !precision &&
         (!features.shader_float16 || !features.storage_buffer16)) ||
        (artifact && (!features.shader_int16 ||
                      !features.preserve_float32_specials)))
      return Status::Unimplemented(
          "Vulkan IREE program requires enabled numeric device features");
    const auto &limits = device.properties.limits;
    // Scan retains its conservative domain; precision and FP8 use their
    // separately proven locked-program bounds before any native allocation.
    const auto maximum =
        precision ? precision_elements(device) : admitted_elements(device);
    const auto fits = [maximum](std::int64_t a, std::int64_t b) {
      return a > 0 && b > 0 &&
             static_cast<std::uint64_t>(a) <=
                 maximum / static_cast<std::uint64_t>(b);
    };
    if (limits.maxComputeWorkGroupInvocations < 64 ||
        limits.maxComputeWorkGroupSize[0] < 64 ||
        ((scan || precision) ? !fits(spec.rows, spec.columns)
         : artifact ? !artifact_dispatch_fits(device, program, step)
                    : !fp8_dispatch_fits(limits, spec)))
      return Status::Unimplemented("Vulkan IREE program exceeds admitted "
                                   "native grid or workspace limits");
    {
      std::lock_guard lock(program.runtime->mutex);
      if (program.runtime->closed)
        return Status::FailedPrecondition("Vulkan context is closed");
      if (!device.iree) {
        HYLE_ASSIGN_OR_RETURN(device.iree, iv::NativeDevice::create(
                                               device.provider, device.quota));
      }
    }
    Dispatch dispatch;
    dispatch.iree = spec;
    dispatch.values = step.inputs;
    dispatch.values.insert(dispatch.values.end(), step.outputs.begin(),
                           step.outputs.end());
    program.dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan IREE preparation metadata");
  }
}

Result<std::shared_ptr<IreeStage>>
create_iree_stage(const std::shared_ptr<Run> &run, const Dispatch &dispatch,
                  std::span<const PlanValue> values) noexcept {
  try {
    auto stage = std::make_shared<IreeStage>();
    stage->parent = run.get();
    const auto spec = *dispatch.iree;
    HYLE_ASSIGN_OR_RETURN(
        stage->program,
        iv::NativeProgram::create(run->runtime->device->iree, spec));
    stage->input_count = stage->program->input_count();
    stage->output = dispatch.values.back();
    stage->output_bytes = *values[stage->output].description.required_bytes();
    for (std::size_t port = 0; port < stage->input_count; ++port) {
      auto &input = stage->inputs[port];
      input.value = dispatch.values[iv::artifact_program(spec.kind) ? port : port % 2];
      if (!iv::artifact_program(spec.kind) && port >= 2)
        input.offset =
            static_cast<VkDeviceSize>(spec.depth) *
            static_cast<VkDeviceSize>(port == 2 ? spec.rows : spec.columns);
      HYLE_ASSIGN_OR_RETURN(input.destination, stage->program->input(port));
    }
    HYLE_ASSIGN_OR_RETURN(stage->input_copy, create_run(run->runtime));
    HYLE_ASSIGN_OR_RETURN(stage->output_copy, create_run(run->runtime));
    if (spec.kind == iv::ProgramKind::ScanInclusive ||
        spec.kind == iv::ProgramKind::ScanExclusive) {
      HYLE_ASSIGN_OR_RETURN(stage->flag_copy, create_run(run->runtime));
      HYLE_ASSIGN_OR_RETURN(stage->flag,
                            allocate_buffer(run->runtime->device, 4, true));
    }
    if (!run->async)
      run->async = std::make_unique<AsyncRun>();
    return stage;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan IREE invocation boundary"));
  }
}

Status IreeStage::bind() noexcept {
  for (std::size_t port = 0; port < input_count; ++port) {
    const auto &input = inputs[port];
    const auto &buffer = parent->retained_buffers[input.value];
    if (!buffer || buffer->handle == VK_NULL_HANDLE)
      return Status::Ok(); // Cold unbound invocation; submission stays
                           // disabled.
    if (input.offset > buffer->bytes ||
        input.destination.bytes > buffer->bytes - input.offset ||
        buffer->offset > UINT64_MAX - input.offset)
      return Status::InvalidArgument("Vulkan IREE input range");
  }
  HYLE_RETURN_IF_ERROR(begin_copy(*input_copy));
  for (std::size_t port = 0; port < input_count; ++port) {
    const auto &input = inputs[port];
    const auto &buffer = parent->retained_buffers[input.value];
    const VkBufferCopy copy{buffer->offset + input.offset,
                            input.destination.offset, input.destination.bytes};
    parent->runtime->device->fn.CmdCopyBuffer(
        input_copy->command, buffer->handle, input.destination.buffer, 1,
        &copy);
  }
  return end_copy(*input_copy);
}

void IreeStage::completed(void *opaque, const Status &completion) noexcept {
  auto &stage = *static_cast<IreeStage *>(opaque);
  auto &run = *stage.parent;
  std::lock_guard lock(run.runtime->mutex);
  auto status = completion;
  if (status.ok() && stage.flag && !stage.reading_flag) {
    status = copy_result(stage, true);
    if (status.ok()) {
      stage.reading_flag = true;
      status = stage.program->after_native_submissions(completed, &stage);
      if (status.ok())
        return;
    }
  } else if (status.ok() && stage.reading_flag) {
    const auto &buffer = stage.flag;
    if ((buffer->memory_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
      VkMappedMemoryRange range{};
      range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      range.memory = buffer->memory;
      range.size = VK_WHOLE_SIZE;
      status =
          native_status(run.runtime->device->fn.InvalidateMappedMemoryRanges(
                            run.runtime->device->native.device, 1, &range),
                        "vkInvalidateMappedMemoryRanges(IREE Scan flag)");
    }
    std::uint32_t flag = 0;
    if (status.ok()) {
      std::memcpy(&flag, buffer->mapped, sizeof(flag));
      if (flag != 0)
        status = Status::OutOfRange("Vulkan Scan prefix exceeds I32");
    }
  }
  if (status.ok() && run.cancelled) {
    // Cancellation stops publication but does not poison a reusable slot.
    // A completed checked operation still reports its numerical error first.
    run.needs_drain = true;
    run.async->finished = true;
    return;
  }
  if (status.ok())
    status = copy_result(stage, false);
  if (!status.ok()) {
    fail_sequence(run, status);
    return;
  }
  ++run.async->next_stage;
  (void)advance_async_locked(run);
}

bool advance_async_locked(Run &run) noexcept {
  bool accepted = false;
  auto &device = *run.runtime->device;
  while (run.async->next_stage < run.stages.size()) {
    if (run.cancelled) {
      run.needs_drain = true;
      run.async->finished = true;
      return accepted;
    }
    const auto index = run.async->next_stage;
    auto &stage = run.stages[index];
    Status status;
    if (stage.iree) {
      stage.iree->reading_flag = false;
      status = submit_native(device, stage.iree->input_copy->command,
                             VK_NULL_HANDLE);
      accepted = accepted || status.ok();
      if (status.ok()) {
        const auto submitted =
            stage.iree->program->invoke(IreeStage::completed, stage.iree.get());
        status = submitted.status;
        if (submitted.pending)
          return true; // Completion, failure and flag readback advance the
                       // cursor.
      }
    } else if (stage.managed) {
      const auto submitted = stage.managed->submit();
      status = submitted.status;
      accepted = accepted || submitted.pending;
      if (submitted.device_lost)
        run.runtime->failure =
            native_status(VK_ERROR_DEVICE_LOST, "Vulkan managed submit lost");
    } else {
      status = submit_native(device, stage.native->command,
                             index + 1 == run.stages.size() ? run.fence
                                                            : VK_NULL_HANDLE);
      accepted = accepted || status.ok();
    }
    if (!status.ok()) {
      fail_sequence(run, status);
      return accepted;
    }
    ++run.async->next_stage;
  }
  run.async->finished = true;
  return accepted;
}

} // namespace hyle::compute::vulkan::detail
