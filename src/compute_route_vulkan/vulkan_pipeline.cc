#include "vulkan_program_internal.h"

#include <algorithm>
#include <limits>
#include <new>

namespace hyle::compute::vulkan::detail {

Pipeline::~Pipeline() {
  if (handle != VK_NULL_HANDLE)
    device->fn.DestroyPipeline(device->native.device, handle, nullptr);
  if (layout != VK_NULL_HANDLE)
    device->fn.DestroyPipelineLayout(device->native.device, layout, nullptr);
  if (descriptor_layout != VK_NULL_HANDLE)
    device->fn.DestroyDescriptorSetLayout(device->native.device,
                                          descriptor_layout, nullptr);
}

Result<std::shared_ptr<Pipeline>>
create_pipeline(const std::shared_ptr<Device> &device,
                const SpirvDescription &description) noexcept {
  const auto &limits = device->properties.limits;
  const auto &indexing = device->descriptor_properties;
  const bool rebindable = device->environment.storage_update_after_bind;
  const auto binding_limit =
      rebindable
          ? (std::min)({indexing
                            .maxPerStageDescriptorUpdateAfterBindStorageBuffers,
                        indexing.maxDescriptorSetUpdateAfterBindStorageBuffers,
                        indexing.maxPerStageUpdateAfterBindResources})
          : (std::min)({limits.maxPerStageDescriptorStorageBuffers,
                        limits.maxDescriptorSetStorageBuffers,
                        limits.maxPerStageResources});
  if (description.code.size() < 5 || description.code[0] != 0x07230203 ||
      description.code[1] > 0x00010500 || description.storage_bindings == 0 ||
      description.storage_bindings > binding_limit ||
      description.push_constant_bytes > limits.maxPushConstantsSize ||
      description.push_constant_bytes % 4 != 0 ||
      description.specialization.size() >
          (std::numeric_limits<std::uint32_t>::max)() / 4)
    return std::unexpected(
        Status::InvalidArgument("Vulkan shader description exceeds domain"));
  try {
    auto pipeline = std::make_shared<Pipeline>();
    pipeline->device = device;
    pipeline->bindings = description.storage_bindings;
    pipeline->push_bytes = description.push_constant_bytes;
    std::vector<VkDescriptorSetLayoutBinding> bindings(pipeline->bindings);
    std::vector<VkDescriptorBindingFlags> binding_flags(
        pipeline->bindings,
        rebindable ? VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT} : 0u);
    for (std::uint32_t index = 0; index < pipeline->bindings; ++index) {
      bindings[index].binding = index;
      bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[index].descriptorCount = 1;
      bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo flags{};
    flags.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flags.bindingCount = pipeline->bindings;
    flags.pBindingFlags = binding_flags.data();
    VkDescriptorSetLayoutCreateInfo descriptors{};
    descriptors.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptors.pNext = &flags;
    descriptors.flags =
        rebindable ? VkDescriptorSetLayoutCreateFlags{VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT}
                   : 0u;
    descriptors.bindingCount = pipeline->bindings;
    descriptors.pBindings = bindings.data();
    auto status = native_status(device->fn.CreateDescriptorSetLayout(
                                    device->native.device, &descriptors,
                                    nullptr, &pipeline->descriptor_layout),
                                "vkCreateDescriptorSetLayout");
    if (!status.ok())
      return std::unexpected(status);
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = pipeline->push_bytes;
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &pipeline->descriptor_layout;
    layout.pushConstantRangeCount = push.size == 0 ? 0u : 1u;
    layout.pPushConstantRanges = push.size == 0 ? nullptr : &push;
    status = native_status(
        device->fn.CreatePipelineLayout(device->native.device, &layout, nullptr,
                                        &pipeline->layout),
        "vkCreatePipelineLayout");
    if (!status.ok())
      return std::unexpected(status);
    std::vector<VkSpecializationMapEntry> entries;
    std::vector<std::uint32_t> constants;
    entries.reserve(description.specialization.size());
    constants.reserve(description.specialization.size());
    for (const auto &entry : description.specialization) {
      if (std::ranges::any_of(entries, [&](const auto &existing) {
            return existing.constantID == entry.id;
          }))
        return std::unexpected(
            Status::InvalidArgument("duplicate Vulkan specialization ID"));
      entries.push_back(
          {entry.id, static_cast<std::uint32_t>(constants.size() * 4), 4});
      constants.push_back(entry.value);
    }
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = static_cast<std::uint32_t>(entries.size());
    specialization.pMapEntries = entries.data();
    specialization.dataSize = constants.size() * sizeof(std::uint32_t);
    specialization.pData = constants.data();
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = description.code.size_bytes();
    shader_info.pCode = description.code.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    status = native_status(device->fn.CreateShaderModule(device->native.device,
                                                         &shader_info, nullptr,
                                                         &shader),
                           "vkCreateShaderModule");
    if (!status.ok())
      return std::unexpected(status);
    VkComputePipelineCreateInfo compute{};
    compute.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute.layout = pipeline->layout;
    compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute.stage.module = shader;
    compute.stage.pName = "main";
    compute.stage.pSpecializationInfo =
        entries.empty() ? nullptr : &specialization;
    status = native_status(device->fn.CreateComputePipelines(
                               device->native.device, VK_NULL_HANDLE, 1,
                               &compute, nullptr, &pipeline->handle),
                           "vkCreateComputePipelines");
    device->fn.DestroyShaderModule(device->native.device, shader, nullptr);
    if (!status.ok())
      return std::unexpected(status);
    return pipeline;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan pipeline metadata"));
  }
}

namespace {
std::uint32_t origin(const Buffer *buffer) noexcept {
  if (!buffer)
    return 0;
  const auto alignment =
      (std::max)(VkDeviceSize{4}, buffer->device->properties.limits
                                      .minStorageBufferOffsetAlignment);
  return static_cast<std::uint32_t>(buffer->offset % alignment);
}
std::array<std::uint32_t, 3> origins(const Run &run,
                                     const Dispatch &dispatch) noexcept {
  std::array<std::uint32_t, 3> result{};
  if (dispatch.tensor_byte_origins)
    for (std::size_t port = 0; port < result.size(); ++port)
      result[port] = origin(run.retained_buffers[dispatch.values[port]].get());
  return result;
}
} // namespace

Result<VkDescriptorBufferInfo>
descriptor_buffer(const Buffer *buffer, const TensorDesc &description,
                  bool byte_origin, VkDeviceSize relative_offset) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
  VkDescriptorBufferInfo result{};
  VkDeviceSize prefix = 0;
  if (buffer) {
    const auto alignment =
        (std::max)(VkDeviceSize{4}, buffer->device->properties.limits
                                        .minStorageBufferOffsetAlignment);
    const auto logical = buffer->offset + relative_offset;
    prefix = logical % alignment;
    if (prefix != 0 && !byte_origin)
      return std::unexpected(
          Status::Unimplemented("Vulkan native fragment requires an aligned "
                                "range; materialize the View"));
    result.buffer = buffer->handle;
    result.offset = logical - prefix;
    if (buffer->handle != VK_NULL_HANDLE &&
        (relative_offset > buffer->bytes ||
         bytes > buffer->bytes - relative_offset))
      return std::unexpected(
          Status::InvalidArgument("Vulkan view exceeds its native buffer"));
  }
  HYLE_ASSIGN_OR_RETURN(result.range, buffer_capacity(prefix + bytes));
  if (buffer &&
      (result.range > buffer->device->properties.limits.maxStorageBufferRange ||
       (buffer->handle != VK_NULL_HANDLE &&
        result.range > prefix + buffer->bytes - relative_offset)))
    return std::unexpected(Status::InvalidArgument(
        "Vulkan padded descriptor exceeds its native range"));
  return result;
}

bool origins_changed(const Run &run,
                     std::span<const Dispatch> dispatches) noexcept {
  if (!run.encoding_valid)
    return true;
  for (std::size_t index = 0; index < dispatches.size(); ++index)
    if (origins(run, dispatches[index]) != run.recorded_origins[index])
      return true;
  return false;
}

Status encode_commands(Run &run,
                       std::span<const Dispatch> dispatches) noexcept {
  auto &device = *run.runtime->device;
  run.encoding_valid = false;
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  HYLE_RETURN_IF_ERROR(
      native_status(device.fn.BeginCommandBuffer(run.command, &begin),
                    "vkBeginCommandBuffer"));
  memory_barrier(device, run.command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                 VK_ACCESS_MEMORY_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  for (std::size_t index = 0; index < dispatches.size(); ++index) {
    const auto &dispatch = dispatches[index];
    const auto offsets = origins(run, dispatch);
    device.fn.CmdBindPipeline(run.command, VK_PIPELINE_BIND_POINT_COMPUTE,
                              dispatch.pipeline->handle);
    device.fn.CmdBindDescriptorSets(run.command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    dispatch.pipeline->layout, 0, 1,
                                    &run.descriptor_sets[index], 0, nullptr);
    if (!dispatch.push.empty())
      device.fn.CmdPushConstants(
          run.command, dispatch.pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
          0, static_cast<std::uint32_t>(dispatch.push.size()),
          dispatch.tensor_byte_origins
              ? static_cast<const void *>(offsets.data())
              : dispatch.push.data());
    device.fn.CmdDispatch(run.command, dispatch.groups[0], dispatch.groups[1],
                          dispatch.groups[2]);
    memory_barrier(device, run.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    run.recorded_origins[index] = offsets;
  }
  memory_barrier(device, run.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                 VK_ACCESS_MEMORY_READ_BIT);
  if (run.checked)
    memory_barrier(device, run.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                   VK_ACCESS_HOST_READ_BIT);
  HYLE_RETURN_IF_ERROR(native_status(device.fn.EndCommandBuffer(run.command),
                                     "vkEndCommandBuffer"));
  run.encoding_valid = true;
  return Status::Ok();
}

Status record(const std::shared_ptr<Run> &run,
              std::span<const PlanValue> values,
              std::span<const Dispatch> dispatches,
              std::vector<DescriptorBinding> *dynamic_bindings) noexcept {
  auto &device = *run->runtime->device;
  if (dispatches.size() > (std::numeric_limits<std::uint32_t>::max)() ||
      run->retained_buffers.size() != values.size() ||
      run->descriptor_pool != VK_NULL_HANDLE)
    return Status::InvalidArgument("invalid Vulkan recording recipe");
  try {
    for (const auto &dispatch : dispatches) {
      if (!dispatch.checked_status)
        continue;
      if (dispatch.values.size() < 3 || dispatch.values[2] >= values.size())
        return Status::Internal("Vulkan checked status binding");
      const auto &buffer = run->retained_buffers[dispatch.values[2]];
      if (!buffer || !buffer->mapped || buffer->bytes != 4 || buffer->offset)
        return Status::Internal("Vulkan checked status storage");
      if (!run->checked)
        run->checked = std::make_unique<CheckedStatus>();
      if (std::ranges::find(run->checked->buffers, buffer) ==
          run->checked->buffers.end())
        run->checked->buffers.push_back(buffer);
    }
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan checked status metadata");
  }
  if (std::ranges::any_of(dispatches, [](const auto &dispatch) {
        return dispatch.region != nullptr || dispatch.iree.has_value();
      }))
    return record_regions(run, values, dispatches, dynamic_bindings);
  try {
    std::uint64_t descriptors = 0;
    for (const auto &dispatch : dispatches) {
      if (!dispatch.pipeline ||
          dispatch.pipeline->device != run->runtime->device ||
          dispatch.values.size() != dispatch.pipeline->bindings ||
          dispatch.push.size() != dispatch.pipeline->push_bytes)
        return Status::InvalidArgument(
            "Vulkan dispatch pipeline or binding arity");
      for (std::size_t axis = 0; axis < 3; ++axis)
        if (dispatch.groups[axis] == 0 ||
            dispatch.groups[axis] >
                device.properties.limits.maxComputeWorkGroupCount[axis])
          return Status::OutOfRange("Vulkan dispatch group count");
      for (const auto value : dispatch.values) {
        if (value >= values.size())
          return Status::InvalidArgument("Vulkan dispatch value index");
        if ((!run->retained_buffers[value] ||
             run->retained_buffers[value]->handle == VK_NULL_HANDLE) &&
            dynamic_bindings == nullptr)
          return Status::InvalidArgument(
              "Vulkan fixed dispatch has an unbound input");
      }
      descriptors += dispatch.pipeline->bindings;
    }
    if (descriptors > (std::numeric_limits<std::uint32_t>::max)())
      return Status::ResourceExhausted("Vulkan descriptor pool size overflows");
    const bool rebindable = device.environment.storage_update_after_bind;
    if (dynamic_bindings != nullptr && !rebindable)
      return Status::Unavailable(
          "Vulkan input rebinding requires storage-buffer update-after-bind");
    if (rebindable) {
      const auto limit =
          device.descriptor_properties.maxUpdateAfterBindDescriptorsInAllPools;
      auto current =
          device.descriptors_in_pools.load(std::memory_order_relaxed);
      do {
        if (descriptors > limit || current > limit - descriptors)
          return Status::ResourceExhausted(
              "Vulkan update-after-bind descriptor budget");
      } while (!device.descriptors_in_pools.compare_exchange_weak(
          current, current + descriptors, std::memory_order_relaxed));
      run->descriptor_count = static_cast<std::uint32_t>(descriptors);
    }
    auto status = Status::Ok();
    if (!dispatches.empty()) {
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                static_cast<std::uint32_t>(descriptors)};
      VkDescriptorPoolCreateInfo pool{};
      pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      pool.flags =
          rebindable ? VkDescriptorPoolCreateFlags{VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT} : 0u;
      pool.maxSets = static_cast<std::uint32_t>(dispatches.size());
      pool.poolSizeCount = 1;
      pool.pPoolSizes = &size;
      status = native_status(
          device.fn.CreateDescriptorPool(device.native.device, &pool, nullptr,
                                         &run->descriptor_pool),
          "vkCreateDescriptorPool");
      if (!status.ok())
        return status;
    }
    run->descriptor_sets.reserve(dispatches.size());
    run->recorded_origins.resize(dispatches.size());
    for (const auto &dispatch : dispatches) {
      VkDescriptorSetAllocateInfo allocate{};
      allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocate.descriptorPool = run->descriptor_pool;
      allocate.descriptorSetCount = 1;
      allocate.pSetLayouts = &dispatch.pipeline->descriptor_layout;
      VkDescriptorSet descriptor = VK_NULL_HANDLE;
      status = native_status(device.fn.AllocateDescriptorSets(
                                 device.native.device, &allocate, &descriptor),
                             "vkAllocateDescriptorSets");
      if (!status.ok())
        return status;
      run->descriptor_sets.push_back(descriptor);
      for (std::uint32_t binding = 0; binding < dispatch.pipeline->bindings;
           ++binding) {
        const auto value = dispatch.values[binding];
        DescriptorBinding entry;
        entry.value = value;
        entry.byte_origin = dispatch.tensor_byte_origins && binding < 3;
        const auto &buffer = run->retained_buffers[value];
        HYLE_ASSIGN_OR_RETURN(entry.buffer,
                              descriptor_buffer(buffer.get(),
                                                values[value].description,
                                                entry.byte_origin));
        entry.write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        entry.write.dstSet = descriptor;
        entry.write.dstBinding = binding;
        entry.write.descriptorCount = 1;
        entry.write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (buffer && buffer->handle != VK_NULL_HANDLE) {
          entry.write.pBufferInfo = &entry.buffer;
          device.fn.UpdateDescriptorSets(device.native.device, 1, &entry.write,
                                         0, nullptr);
        } else {
          dynamic_bindings->push_back(entry);
        }
      }
    }
    return encode_commands(*run, dispatches);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan command recording metadata");
  }
}

} // namespace hyle::compute::vulkan::detail

namespace hyle::compute::vulkan {

Result<Kernel> Kernel::create(const Context &context,
                              const SpirvDescription &description) noexcept {
  const auto &runtime = detail::Access::runtime(context);
  if (!runtime)
    return std::unexpected(Status::InvalidArgument("invalid Vulkan context"));
  {
    std::lock_guard lock(runtime->mutex);
    if (runtime->closed)
      return std::unexpected(
          Status::FailedPrecondition("Vulkan context is closed"));
    if (!runtime->failure.ok())
      return std::unexpected(runtime->failure);
  }
  auto pipeline = detail::create_pipeline(runtime->device, description);
  if (!pipeline)
    return std::unexpected(std::move(pipeline).error());
  return detail::Access::kernel(std::move(*pipeline));
}

} // namespace hyle::compute::vulkan
