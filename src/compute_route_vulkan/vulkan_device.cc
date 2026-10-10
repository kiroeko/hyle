#include "vulkan_build.h"
#include "vulkan_internal.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string_view>

#include "../compute_ops/view_alias.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::vulkan::detail {

struct BufferBudget final {
  mem::AllocatorRef allocator;
  compute::ResourceBudget::Reservation reservation;
};

Status native_status(VkResult result, const char *operation) noexcept {
  if (result == VK_SUCCESS)
    return Status::Ok();
  StatusCode code = StatusCode::Unknown;
  switch (result) {
  case VK_ERROR_OUT_OF_HOST_MEMORY:
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
  case VK_ERROR_TOO_MANY_OBJECTS:
  case VK_ERROR_OUT_OF_POOL_MEMORY:
    code = StatusCode::ResourceExhausted;
    break;
  case VK_ERROR_DEVICE_LOST:
    code = StatusCode::DataLoss;
    break;
  case VK_ERROR_FEATURE_NOT_PRESENT:
  case VK_ERROR_EXTENSION_NOT_PRESENT:
  case VK_ERROR_LAYER_NOT_PRESENT:
  case VK_ERROR_INCOMPATIBLE_DRIVER:
  case VK_ERROR_INITIALIZATION_FAILED:
    code = StatusCode::Unavailable;
    break;
  default:
    break;
  }
  return Status::FromNative(code, {NativeErrorDomain::Vulkan, result},
                            operation);
}

Buffer::~Buffer() {
  if (memory != VK_NULL_HANDLE && mapped != nullptr)
    device->fn.UnmapMemory(device->native.device, memory);
  if (handle != VK_NULL_HANDLE && !external_owner)
    device->fn.DestroyBuffer(device->native.device, handle, nullptr);
  if (memory != VK_NULL_HANDLE)
    device->fn.FreeMemory(device->native.device, memory, nullptr);
  if (budget != nullptr) {
    const auto allocator = budget->allocator;
    std::destroy_at(budget);
    allocator.deallocate(budget, sizeof(BufferBudget), alignof(BufferBudget));
  }
}

namespace {

bool has_extension(std::span<const VkExtensionProperties> extensions,
                   std::string_view name) noexcept {
  return std::ranges::any_of(extensions, [&](const auto &extension) {
    return name == extension.extensionName;
  });
}

Result<std::shared_ptr<Device>> create_device(ContextOptions options) noexcept {
  try {
    auto device = std::make_shared<Device>();
    device->quota = options.memory_quota;
    HYLE_ASSIGN_OR_RETURN(device->provider,
        ggml_vulkan::detail::NativeDevice::create({
            options.loader_path, options.device_index,
            options.allow_software_device, options.validation,
            options.memory_quota}));
    const auto &owner = device->provider.native();
    device->native = {owner.instance, owner.physical, owner.device,
                      owner.queue_family};
    device->queue = owner.queue;
    device->environment.loader_version = owner.loader_version;
    device->environment.storage_update_after_bind =
        owner.storage_update_after_bind;
    const auto get = owner.get_instance_proc_addr;
#define HYLE_VULKAN_LOAD_INSTANCE(name)                                        \
  device->instance_fn.name = reinterpret_cast<PFN_vk##name>(                   \
      get(device->native.instance, "vk" #name));                               \
  if (device->instance_fn.name == nullptr)                                     \
    return std::unexpected(Status::Unavailable("missing Vulkan vk" #name));
    HYLE_VULKAN_INSTANCE_FUNCTIONS(HYLE_VULKAN_LOAD_INSTANCE)
#undef HYLE_VULKAN_LOAD_INSTANCE
    auto &instance = device->instance_fn;
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    device->descriptor_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
    driver.pNext = &device->descriptor_properties;
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup.pNext = &driver;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &subgroup;
    instance.GetPhysicalDeviceProperties2(device->native.physical_device,
                                          &properties);
    device->properties = properties.properties;
    auto &environment = device->environment;
    environment.shader_compiler = kShaderCompiler;
    environment.shader_revision = kShaderRevision;
    environment.sdk_version = kVulkanSdkVersion;
    environment.ggml_revision = kGgmlRevision;
    environment.iree_revision = kIreeRevision;
    environment.device_name = device->properties.deviceName;
    environment.driver_name = driver.driverName;
    environment.driver_info = driver.driverInfo;
    environment.vendor_id = device->properties.vendorID;
    environment.device_id = device->properties.deviceID;
    environment.driver_id = static_cast<std::uint32_t>(driver.driverID);
    environment.driver_version = device->properties.driverVersion;
    environment.api_version = device->properties.apiVersion;
    environment.subgroup_size = subgroup.subgroupSize;
    environment.subgroup_arithmetic =
        (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) !=
            0;
    environment.shared_memory_bytes =
        device->properties.limits.maxComputeSharedMemorySize;
    environment.software_device =
        device->properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    instance.GetPhysicalDeviceMemoryProperties(device->native.physical_device,
                                               &device->memory_properties);
    std::uint32_t count = 0;
    auto status = native_status(
        instance.EnumerateDeviceExtensionProperties(
            device->native.physical_device, nullptr, &count, nullptr),
        "vkEnumerateDeviceExtensionProperties(count)");
    if (!status.ok())
      return std::unexpected(status);
    std::vector<VkExtensionProperties> extensions(count);
    status = native_status(
        instance.EnumerateDeviceExtensionProperties(
            device->native.physical_device, nullptr, &count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");
    if (!status.ok())
      return std::unexpected(status);
    extensions.resize(count);
    environment.portability_subset =
        has_extension(extensions, "VK_KHR_portability_subset");
#define HYLE_VULKAN_LOAD_DEVICE(name)                                          \
  device->fn.name = reinterpret_cast<PFN_vk##name>(                            \
      instance.GetDeviceProcAddr(device->native.device, "vk" #name));          \
  if (device->fn.name == nullptr)                                              \
    return std::unexpected(Status::Unavailable("missing Vulkan vk" #name));
    HYLE_VULKAN_DEVICE_FUNCTIONS(HYLE_VULKAN_LOAD_DEVICE)
#undef HYLE_VULKAN_LOAD_DEVICE
    if (device->queue == VK_NULL_HANDLE)
      return std::unexpected(Status::Unavailable("null Vulkan compute queue"));
    return device;
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan context metadata"));
  }
}

} // namespace

Result<VkDeviceSize> buffer_capacity(VkDeviceSize logical_bytes) noexcept {
  if (logical_bytes > (std::numeric_limits<VkDeviceSize>::max)() - 3)
    return std::unexpected(
        Status::OutOfRange("Vulkan storage capacity overflows"));
  return (std::max)(VkDeviceSize{4}, (logical_bytes + 3) & ~VkDeviceSize{3});
}

void bind_view_buffer(Buffer &view, const std::shared_ptr<Buffer> &root,
                      VkDeviceSize byte_offset) noexcept {
  view.external_owner = root;
  view.handle = root ? root->handle : VK_NULL_HANDLE;
  view.offset = (root ? root->offset : 0) + byte_offset;
  view.bytes = root ? root->bytes - byte_offset : 0;
  view.allocation_bytes = root ? root->allocation_bytes : 0;
  view.is_view = true;
}

Result<std::shared_ptr<Buffer>>
allocate_buffer(const std::shared_ptr<Device> &device, VkDeviceSize bytes,
                bool staging, bool shader_access, BufferAllocation allocation) noexcept {
  HYLE_ASSIGN_OR_RETURN(bytes, buffer_capacity(bytes));
  if (bytes > (std::numeric_limits<std::size_t>::max)() ||
      ((!staging || shader_access) &&
       bytes > device->properties.limits.maxStorageBufferRange))
    return std::unexpected(
        Status::InvalidArgument("Vulkan buffer size exceeds domain"));
  try {
    auto allocator = allocation.allocator;
    compute::ResourceBudget::Reservation reservation;
    if (allocation.budget) {
      HYLE_ASSIGN_OR_RETURN(allocator, allocation.budget.metadata_allocator());
      HYLE_ASSIGN_OR_RETURN(reservation, allocation.budget.reserve(allocation.resource,
          static_cast<std::size_t>(bytes), allocation.category));
    }
    if (!allocator.valid())
      return std::unexpected(Status::InvalidArgument("invalid Vulkan buffer metadata allocator"));
    auto buffer = std::allocate_shared<Buffer>(mem::StdAllocator<Buffer>(allocator));
    if (allocation.budget) {
      auto *owner = allocator.allocate(sizeof(BufferBudget), alignof(BufferBudget));
      if (owner == nullptr)
        return std::unexpected(Status::ResourceExhausted("Vulkan buffer budget metadata"));
      buffer->budget = std::construct_at(static_cast<BufferBudget *>(owner), allocator, std::move(reservation));
    }
    buffer->device = device;
    buffer->bytes = bytes;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!staging || shader_access)
      info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto status =
        native_status(device->fn.CreateBuffer(device->native.device, &info,
                                              nullptr, &buffer->handle),
                      "vkCreateBuffer");
    if (!status.ok())
      return std::unexpected(status);
    VkMemoryRequirements requirements{};
    device->fn.GetBufferMemoryRequirements(device->native.device,
                                           buffer->handle, &requirements);
    const VkMemoryPropertyFlags required =
        staging ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    std::uint32_t selected = 0;
    int selected_score = -1;
    for (std::uint32_t index = 0;
         index < device->memory_properties.memoryTypeCount; ++index) {
      const auto flags =
          device->memory_properties.memoryTypes[index].propertyFlags;
      if ((requirements.memoryTypeBits & (1u << index)) == 0 ||
          (flags & required) != required)
        continue;
      const int score =
          staging
              ? ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 ? 2 : 0) +
                    ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0 ? 1 : 0)
              : ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 ? 1 : 0);
      if (score > selected_score) {
        selected = index;
        selected_score = score;
        buffer->memory_flags = flags;
      }
    }
    if (selected_score < 0)
      return std::unexpected(
          Status::Unavailable("Vulkan memory type unavailable"));
    if (requirements.size > (std::numeric_limits<std::size_t>::max)())
      return std::unexpected(
          Status::ResourceExhausted("Vulkan allocation size overflows"));
    if (device->quota != nullptr) {
      auto charge = device->quota->try_reserve_bytes(
          static_cast<std::size_t>(requirements.size));
      if (!charge)
        return std::unexpected(
            Status::ResourceExhausted("Vulkan device memory quota"));
      buffer->charge = std::move(*charge);
    }
    VkMemoryAllocateInfo native_allocation{};
    native_allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    native_allocation.allocationSize = requirements.size;
    native_allocation.memoryTypeIndex = selected;
    status = native_status(device->fn.AllocateMemory(device->native.device,
                                                     &native_allocation, nullptr,
                                                     &buffer->memory),
                           "vkAllocateMemory");
    if (!status.ok())
      return std::unexpected(status);
    buffer->allocation_bytes = requirements.size;
    status = native_status(device->fn.BindBufferMemory(device->native.device,
                                                       buffer->handle,
                                                       buffer->memory, 0),
                           "vkBindBufferMemory");
    if (!status.ok())
      return std::unexpected(status);
    if (staging) {
      status = native_status(
          device->fn.MapMemory(device->native.device, buffer->memory, 0,
                               VK_WHOLE_SIZE, 0, &buffer->mapped),
          "vkMapMemory");
      if (!status.ok())
        return std::unexpected(status);
    }
    return buffer;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan buffer metadata"));
  }
}

void memory_barrier(Device &device, VkCommandBuffer command,
                    VkPipelineStageFlags source_stage,
                    VkAccessFlags source_access,
                    VkPipelineStageFlags destination_stage,
                    VkAccessFlags destination_access) noexcept {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = source_access;
  barrier.dstAccessMask = destination_access;
  device.fn.CmdPipelineBarrier(command, source_stage, destination_stage, 0, 1,
                               &barrier, 0, nullptr, 0, nullptr);
}

} // namespace hyle::compute::vulkan::detail

namespace hyle::compute::vulkan {

VkBuffer Tensor::native_buffer() const noexcept {
  return buffer_ ? buffer_->handle : VK_NULL_HANDLE;
}
VkDeviceSize Tensor::native_offset() const noexcept {
  return buffer_ ? buffer_->offset : 0;
}
std::shared_ptr<const void> Tensor::native_owner() const noexcept {
  return buffer_;
}

Result<Tensor> Tensor::view(const compute::View &operation) const noexcept {
  if (!buffer_)
    return std::unexpected(
        Status::InvalidArgument("invalid Vulkan view input"));
  HYLE_ASSIGN_OR_RETURN(
      auto description,
      infer_single_output(Operation(operation), std::span(&description_, 1)));
  HYLE_ASSIGN_OR_RETURN(auto offset, compute::detail::view_alias_offset(
                                         operation, description_, description));
  try {
    auto alias = std::make_shared<detail::Buffer>();
    alias->device = buffer_->device;
    detail::bind_view_buffer(*alias, buffer_, offset);
    return Tensor(description, std::move(alias));
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan view metadata"));
  }
}

Context &Context::operator=(Context &&other) noexcept {
  if (this != &other) {
    close().IgnoreError();
    state_ = std::move(other.state_);
  }
  return *this;
}
Context::~Context() { close().IgnoreError(); }

Result<Context> Context::create(ContextOptions options) noexcept {
  auto device = detail::create_device(options);
  if (!device)
    return std::unexpected(std::move(device).error());
  try {
    auto runtime = std::make_shared<detail::Runtime>();
    runtime->device = std::move(*device);
    return Context(std::move(runtime));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan runtime metadata"));
  }
}

const Environment &Context::environment() const noexcept {
  HYLE_FATAL_ASSERT(valid());
  return state_->device->environment;
}
NativeDevice Context::native_device() const noexcept {
  return state_ ? state_->device->native : NativeDevice{};
}

Result<Tensor> Context::allocate(const TensorDesc &description) const noexcept {
  if (!state_ || !description.valid())
    return std::unexpected(
        Status::InvalidArgument("invalid Vulkan allocation"));
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::unexpected(std::move(bytes).error());
  std::lock_guard lock(state_->mutex);
  if (state_->closed)
    return std::unexpected(
        Status::FailedPrecondition("Vulkan context is closed"));
  if (!state_->failure.ok())
    return std::unexpected(state_->failure);
  auto buffer = detail::allocate_buffer(state_->device, *bytes, false);
  if (!buffer)
    return std::unexpected(std::move(buffer).error());
  return detail::Access::tensor(description, std::move(*buffer));
}

Result<Tensor>
Context::borrow_ready(const TensorDesc &description, VkBuffer buffer,
                      VkDeviceSize offset, VkDeviceSize available_bytes,
                      std::shared_ptr<const void> owner) const noexcept {
  if (!state_ || !description.valid() || buffer == VK_NULL_HANDLE || !owner)
    return std::unexpected(
        Status::InvalidArgument("invalid Vulkan borrowed buffer"));
  auto bytes = description.required_bytes();
  if (!bytes)
    return std::unexpected(std::move(bytes).error());
  const auto &limits = state_->device->properties.limits;
  HYLE_ASSIGN_OR_RETURN(auto capacity, detail::buffer_capacity(*bytes));
  if (capacity > available_bytes || capacity > limits.maxStorageBufferRange ||
      offset % (std::max)(VkDeviceSize{4},
                          limits.minStorageBufferOffsetAlignment) !=
          0 ||
      available_bytes > (std::numeric_limits<VkDeviceSize>::max)() - offset)
    return std::unexpected(
        Status::InvalidArgument("Vulkan borrowed buffer range/alignment"));
  std::lock_guard lock(state_->mutex);
  if (state_->closed)
    return std::unexpected(
        Status::FailedPrecondition("Vulkan context is closed"));
  if (!state_->failure.ok())
    return std::unexpected(state_->failure);
  try {
    auto native = std::make_shared<detail::Buffer>();
    native->device = state_->device;
    native->external_owner = std::move(owner);
    native->handle = buffer;
    native->offset = offset;
    native->bytes = capacity;
    return detail::Access::tensor(description, std::move(native));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan borrowed buffer metadata"));
  }
}

Status Context::with_queue(
    const std::function<Status(VkQueue)> &function) const noexcept {
  if (!state_ || !function)
    return Status::InvalidArgument("invalid Vulkan queue access");
  std::lock_guard lock(state_->mutex);
  if (state_->closed)
    return Status::FailedPrecondition("Vulkan context is closed");
  if (!state_->failure.ok())
    return state_->failure;
  try {
    return state_->device->provider.with_queue(function);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan native queue callback");
  } catch (const std::exception &error) {
    return Status::Internal(error.what());
  } catch (...) {
    return Status::Unknown("Vulkan native queue callback threw");
  }
}

Status Context::collect() const noexcept {
  return state_ ? detail::collect(state_, false) : Status::Ok();
}
Status Context::close() noexcept {
  if (!state_)
    return Status::Ok();
  {
    std::lock_guard lock(state_->mutex);
    state_->closed = true;
  }
  return detail::collect(state_, true);
}

} // namespace hyle::compute::vulkan
