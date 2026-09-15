// Research only: native recorded compute and memory dependencies, no HYLE facade.
#include <volk.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) {
    std::fprintf(stderr, "%s VkResult=%d\n", operation, static_cast<int>(result));
    throw std::runtime_error(operation);
  }
}
constexpr std::uint32_t kN = 1024;
struct Buffer { VkBuffer buffer{}; VkDeviceMemory memory{}; float* data{}; bool coherent{}; };

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected SPIR-V path");
    check(volkInitialize(), "volkInitialize");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "hyle-four-route-research";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance{};
    check(vkCreateInstance(&ici, nullptr, &instance), "create instance");
    volkLoadInstance(instance);
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate count");
    if (!count) throw std::runtime_error("no Vulkan device");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate devices");
    VkPhysicalDevice physical{};
    std::uint32_t family = UINT32_MAX;
    for (auto candidate : devices) {
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
      std::vector<VkQueueFamilyProperties> queues(count);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, queues.data());
      for (std::uint32_t i = 0; i < count; ++i) {
        if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          physical = candidate; family = i; break;
        }
      }
      if (physical) break;
    }
    if (!physical) throw std::runtime_error("no compute queue");
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    std::printf("device=%s queue_family=%u API=%u.%u.%u driver=%u\n",
                properties.deviceName, family, VK_VERSION_MAJOR(properties.apiVersion),
                VK_VERSION_MINOR(properties.apiVersion), VK_VERSION_PATCH(properties.apiVersion), properties.driverVersion);
    const float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice device{};
    check(vkCreateDevice(physical, &dci, nullptr, &device), "create device");
    volkLoadDevice(device);
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    std::array<Buffer, 3> buffers{};
    for (auto& b : buffers) {
      VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bci.size = kN * sizeof(float); bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      check(vkCreateBuffer(device, &bci, nullptr, &b.buffer), "create buffer");
      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(device, b.buffer, &requirements);
      std::uint32_t type = UINT32_MAX;
      for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
          type = i; break;
        }
      }
      if (type == UINT32_MAX) throw std::runtime_error("no compatible host-visible memory");
      b.coherent = (memory_properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
      VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      ai.allocationSize = requirements.size; ai.memoryTypeIndex = type;
      check(vkAllocateMemory(device, &ai, nullptr, &b.memory), "allocate memory");
      check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "bind memory");
      void* mapped = nullptr;
      check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &mapped), "map memory");
      b.data = static_cast<float*>(mapped);
    }
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % 4 != 0) throw std::runtime_error("invalid shader file");
    std::vector<std::uint32_t> code(static_cast<std::size_t>(file.tellg()) / 4);
    file.seekg(0); file.read(reinterpret_cast<char*>(code.data()), code.size() * 4);
    if (!file) throw std::runtime_error("shader read failed");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size() * 4; smci.pCode = code.data();
    VkShaderModule shader{};
    check(vkCreateShaderModule(device, &smci, nullptr, &shader), "create shader");
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t i = 0; i < 2; ++i)
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 2; slci.pBindings = bindings.data();
    VkDescriptorSetLayout layout{};
    check(vkCreateDescriptorSetLayout(device, &slci, nullptr, &layout), "create descriptor layout");
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &layout;
    VkPipelineLayout pipeline_layout{};
    check(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout), "create pipeline layout");
    VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = shader; pci.stage.pName = "main";
    pci.layout = pipeline_layout;
    VkPipeline pipeline{};
    check(vkCreateComputePipelines(device, {}, 1, &pci, nullptr, &pipeline), "create pipeline");
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2; dpci.poolSizeCount = 1; dpci.pPoolSizes = &pool_size;
    VkDescriptorPool pool{};
    check(vkCreateDescriptorPool(device, &dpci, nullptr, &pool), "create descriptor pool");
    const std::array layouts{layout, layout};
    std::array<VkDescriptorSet, 2> sets{};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool; dsai.descriptorSetCount = 2; dsai.pSetLayouts = layouts.data();
    check(vkAllocateDescriptorSets(device, &dsai, sets.data()), "allocate sets");
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
      for (std::uint32_t port = 0; port < 2; ++port) {
        VkDescriptorBufferInfo bi{buffers[pass + port].buffer, 0, kN * sizeof(float)};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[pass]; write.dstBinding = port;
        write.descriptorCount = 1; write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
      }
    }
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = family;
    VkCommandPool command_pool{};
    check(vkCreateCommandPool(device, &cpci, nullptr, &command_pool), "create command pool");
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = command_pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(device, &cbai, &command), "allocate command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command, &begin), "begin command");
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &sets[pass], 0, nullptr);
      vkCmdDispatch(command, kN / 64, 1, 1);
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = pass == 0 ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           pass == 0 ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_HOST_BIT,
                           0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    check(vkEndCommandBuffer(command), "end command");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    check(vkCreateFence(device, &fci, nullptr, &fence), "create fence");
    for (int run = 0; run < 3; ++run) {
      for (std::uint32_t i = 0; i < kN; ++i) buffers[0].data[i] = static_cast<float>(i + run * 17);
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buffers[0].memory; range.size = VK_WHOLE_SIZE;
      if (!buffers[0].coherent) check(vkFlushMappedMemoryRanges(device, 1, &range), "flush input");
      check(vkResetFences(device, 1, &fence), "reset fence");
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
      check(vkQueueSubmit(queue, 1, &submit, fence), "submit");
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ULL), "wait fence");
      range.memory = buffers[2].memory;
      if (!buffers[2].coherent) check(vkInvalidateMappedMemoryRanges(device, 1, &range), "invalidate output");
      for (std::uint32_t i = 0; i < kN; ++i)
        if (buffers[2].data[i] != 4.0f * static_cast<float>(i + run * 17) + 9.0f)
          throw std::runtime_error("two-pass oracle mismatch");
      std::printf("run=%d values=%u record_count=1 result=PASS\n", run, kN);
    }
    check(vkDeviceWaitIdle(device), "terminal device wait");
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    for (auto& b : buffers) {
      vkUnmapMemory(device, b.memory);
      vkDestroyBuffer(device, b.buffer, nullptr); vkFreeMemory(device, b.memory, nullptr);
    }
    vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    std::puts("result=PASS fixture_resource_creations_in_replay=0 rerecords=0 native_driver_cost=UNMEASURED timing=NOT_RUN");
    return 0;
  } catch (const std::exception& error) {
    // Isolated process failure: not a production RAII/OOM/fault-injection test.
    std::fprintf(stderr, "result=FAIL error=%s\n", error.what());
    return 1;
  }
}
