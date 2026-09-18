#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "../compute_provider_ggml_vulkan/native_device.h"
#include "../compute_provider_ggml_vulkan/region.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/vulkan.h"

namespace hyle::compute::vulkan::detail {
namespace cd = hyle::compute::detail;

// Native PFNs are resolved once for the exact instance/device. There are no
// process-global Vulkan entry points and no per-dispatch provider lookup.
#define HYLE_VULKAN_INSTANCE_FUNCTIONS(X)                                      \
  X(GetPhysicalDeviceProperties2)                                              \
  X(GetPhysicalDeviceMemoryProperties)                                         \
  X(EnumerateDeviceExtensionProperties)                                        \
  X(GetDeviceProcAddr)

#define HYLE_VULKAN_DEVICE_FUNCTIONS(X)                                        \
  X(CreateBuffer)                                                              \
  X(DestroyBuffer)                                                             \
  X(GetBufferMemoryRequirements)                                               \
  X(AllocateMemory)                                                            \
  X(FreeMemory)                                                                \
  X(BindBufferMemory)                                                          \
  X(MapMemory)                                                                 \
  X(UnmapMemory)                                                               \
  X(FlushMappedMemoryRanges)                                                   \
  X(InvalidateMappedMemoryRanges)                                              \
  X(CreateCommandPool)                                                         \
  X(DestroyCommandPool)                                                        \
  X(AllocateCommandBuffers)                                                    \
  X(BeginCommandBuffer)                                                        \
  X(EndCommandBuffer)                                                          \
  X(CmdCopyBuffer)                                                             \
  X(CmdPipelineBarrier)                                                        \
  X(CreateFence)                                                               \
  X(DestroyFence)                                                              \
  X(ResetFences)                                                               \
  X(GetFenceStatus)                                                            \
  X(WaitForFences)                                                             \
  X(QueueSubmit)                                                               \
  X(QueueWaitIdle)                                                             \
  X(CreateShaderModule)                                                        \
  X(DestroyShaderModule)                                                       \
  X(CreateDescriptorSetLayout)                                                 \
  X(DestroyDescriptorSetLayout)                                                \
  X(CreatePipelineLayout)                                                      \
  X(DestroyPipelineLayout)                                                     \
  X(CreateComputePipelines)                                                    \
  X(DestroyPipeline)                                                           \
  X(CreateDescriptorPool)                                                      \
  X(DestroyDescriptorPool)                                                     \
  X(AllocateDescriptorSets)                                                    \
  X(UpdateDescriptorSets)                                                      \
  X(CmdBindPipeline)                                                           \
  X(CmdBindDescriptorSets)                                                     \
  X(CmdPushConstants)                                                          \
  X(CmdDispatch)

struct InstanceFunctions final {
#define HYLE_VULKAN_DECLARE(name) PFN_vk##name name = nullptr;
  HYLE_VULKAN_INSTANCE_FUNCTIONS(HYLE_VULKAN_DECLARE)
#undef HYLE_VULKAN_DECLARE
};
struct DeviceFunctions final {
#define HYLE_VULKAN_DECLARE(name) PFN_vk##name name = nullptr;
  HYLE_VULKAN_DEVICE_FUNCTIONS(HYLE_VULKAN_DECLARE)
#undef HYLE_VULKAN_DECLARE
};

struct Device final {
  // One native owner for HYLE commands and complete managed ggml regions.
  // It outlives every Buffer/Run and owns loader, instance, device and queues.
  ggml_vulkan::detail::NativeDevice provider;
  InstanceFunctions instance_fn;
  DeviceFunctions fn;
  NativeDevice native;
  VkQueue queue = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkPhysicalDeviceDescriptorIndexingProperties descriptor_properties{};
  std::atomic<std::uint64_t> descriptors_in_pools{0};
  Environment environment;
  mem::QuotaAllocator *quota = nullptr;
};

struct Buffer final {
  std::shared_ptr<Device> device;
  // Destroy memory before releasing its quota reservation.
  mem::QuotaAllocator::Reservation charge;
  std::shared_ptr<const void> external_owner;
  VkBuffer handle = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkDeviceSize bytes = 0;
  VkDeviceSize allocation_bytes = 0;
  VkMemoryPropertyFlags memory_flags = 0;
  void *mapped = nullptr;
  bool is_view = false;
  ~Buffer();
};

struct Run;
struct NativeEncoding;
struct RunStage final {
  std::shared_ptr<Run> native;
  std::shared_ptr<NativeEncoding> encoding;
  std::unique_ptr<ggml_vulkan::detail::Region> managed;
  std::vector<std::uint32_t>
      values; // Compact local port/value -> parent value.
  std::vector<ggml_vulkan::detail::RegionBinding> bindings;
};
struct Runtime final {
  std::shared_ptr<Device> device;
  std::mutex mutex;
  Run *pending = nullptr;
  bool closed = false;
  Status failure;
};

// One cold-allocated native submission slot. Active ownership is embedded, so
// hot acceptance/abandonment does not grow a container or allocate a node.
// All mutable execution state and membership are protected by Runtime::mutex.
struct Run final {
  std::shared_ptr<Runtime> runtime;
  std::shared_ptr<Run> pending_owner;
  std::shared_ptr<Run> completion_owner;
  std::vector<std::shared_ptr<Buffer>> retained_buffers;
  std::vector<VkDescriptorSet> descriptor_sets;
  std::vector<std::array<std::uint32_t, 3>> recorded_origins;
  std::shared_ptr<const void> recording_owner;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkFence completion_fence =
      VK_NULL_HANDLE;           // Borrowed last actual batch fence.
  std::vector<RunStage> stages; // Empty for the original single native batch.
  Run *next = nullptr;
  Run **previous = nullptr;
  cd::CompletionLifetime abandoned;
  Status hint;
  Status observation_error;
  Status terminal;
  Status submission_failure;
  bool needs_drain = false;
  bool pending = false;
  bool cancelled = false;
  bool retiring = false;
  bool encoding_valid = false;
  std::uint32_t observers = 0;
  std::uint32_t descriptor_count = 0;
  ~Run();
};

struct Access final {
  static const std::shared_ptr<Runtime> &
  runtime(const Context &context) noexcept {
    return context.state_;
  }
  static const std::shared_ptr<Buffer> &buffer(const Tensor &tensor) noexcept {
    return tensor.buffer_;
  }
  static Tensor tensor(TensorDesc description,
                       std::shared_ptr<Buffer> buffer) noexcept {
    return Tensor(description, std::move(buffer));
  }
  static const std::shared_ptr<Pipeline> &
  pipeline(const Kernel &kernel) noexcept {
    return kernel.state_;
  }
  static Kernel kernel(std::shared_ptr<Pipeline> pipeline) noexcept {
    return Kernel(std::move(pipeline));
  }
  static const std::shared_ptr<ProgramState> &
  program(const Program &program) noexcept {
    return program.state_;
  }
  static Program program(std::shared_ptr<ProgramState> state) noexcept {
    return Program(std::move(state));
  }
  static Invocation
  invocation(std::shared_ptr<InvocationState> state) noexcept {
    return Invocation(std::move(state));
  }
  static Results results(std::shared_ptr<ResultValues> values) noexcept {
    return Results(std::move(values));
  }
};

Status native_status(VkResult result, const char *operation) noexcept;
// Shader storage is U32-addressable, even for sub-word/empty logical tensors.
Result<VkDeviceSize> buffer_capacity(VkDeviceSize logical_bytes) noexcept;
void bind_view_buffer(Buffer &view, const std::shared_ptr<Buffer> &root,
                      VkDeviceSize byte_offset) noexcept;
Result<std::shared_ptr<Buffer>>
allocate_buffer(const std::shared_ptr<Device> &device, VkDeviceSize bytes,
                bool staging) noexcept;
Result<std::shared_ptr<Run>>
create_run(const std::shared_ptr<Runtime> &runtime) noexcept;
Result<Completion> submit_locked(const std::shared_ptr<Run> &run,
                                 bool allow_closed = false) noexcept;
Status collect(const std::shared_ptr<Runtime> &runtime, bool drain) noexcept;
void memory_barrier(Device &device, VkCommandBuffer command,
                    VkPipelineStageFlags source_stage,
                    VkAccessFlags source_access,
                    VkPipelineStageFlags destination_stage,
                    VkAccessFlags destination_access) noexcept;

} // namespace hyle::compute::vulkan::detail
