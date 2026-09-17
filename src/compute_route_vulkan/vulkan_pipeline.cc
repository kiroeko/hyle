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
        rebindable ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0u);
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
        rebindable ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
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

Status record(const std::shared_ptr<Run> &run,
              std::span<const PlanValue> values,
              std::span<const Dispatch> dispatches,
              std::vector<DescriptorBinding> *dynamic_bindings) noexcept {
  auto &device = *run->runtime->device;
  if (dispatches.empty() ||
      dispatches.size() > (std::numeric_limits<std::uint32_t>::max)() ||
      run->retained_buffers.size() != values.size() ||
      run->descriptor_pool != VK_NULL_HANDLE)
    return Status::InvalidArgument("invalid Vulkan recording recipe");
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
        if (!run->retained_buffers[value] && dynamic_bindings == nullptr)
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
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              static_cast<std::uint32_t>(descriptors)};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.flags =
        rebindable ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0u;
    pool.maxSets = static_cast<std::uint32_t>(dispatches.size());
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    auto status = native_status(
        device.fn.CreateDescriptorPool(device.native.device, &pool, nullptr,
                                       &run->descriptor_pool),
        "vkCreateDescriptorPool");
    if (!status.ok())
      return status;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    status = native_status(device.fn.BeginCommandBuffer(run->command, &begin),
                           "vkBeginCommandBuffer");
    if (!status.ok())
      return status;
    memory_barrier(device, run->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_ACCESS_MEMORY_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
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
      for (std::uint32_t binding = 0; binding < dispatch.pipeline->bindings;
           ++binding) {
        const auto value = dispatch.values[binding];
        const auto bytes = values[value].description.required_bytes();
        if (!bytes || *bytes == 0 ||
            *bytes > device.properties.limits.maxStorageBufferRange)
          return Status::InvalidArgument("Vulkan descriptor range");
        DescriptorBinding entry;
        entry.value = value;
        entry.buffer.range = *bytes;
        entry.write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        entry.write.dstSet = descriptor;
        entry.write.dstBinding = binding;
        entry.write.descriptorCount = 1;
        entry.write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        const auto &buffer = run->retained_buffers[value];
        if (buffer) {
          entry.buffer.buffer = buffer->handle;
          entry.buffer.offset = buffer->offset;
          entry.write.pBufferInfo = &entry.buffer;
          device.fn.UpdateDescriptorSets(device.native.device, 1, &entry.write,
                                         0, nullptr);
        } else {
          dynamic_bindings->push_back(entry);
        }
      }
      device.fn.CmdBindPipeline(run->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                dispatch.pipeline->handle);
      device.fn.CmdBindDescriptorSets(
          run->command, VK_PIPELINE_BIND_POINT_COMPUTE,
          dispatch.pipeline->layout, 0, 1, &descriptor, 0, nullptr);
      if (!dispatch.push.empty())
        device.fn.CmdPushConstants(
            run->command, dispatch.pipeline->layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0,
            static_cast<std::uint32_t>(dispatch.push.size()),
            dispatch.push.data());
      device.fn.CmdDispatch(run->command, dispatch.groups[0],
                            dispatch.groups[1], dispatch.groups[2]);
      memory_barrier(device, run->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    memory_barrier(device, run->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_ACCESS_MEMORY_READ_BIT);
    return native_status(device.fn.EndCommandBuffer(run->command),
                         "vkEndCommandBuffer");
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
