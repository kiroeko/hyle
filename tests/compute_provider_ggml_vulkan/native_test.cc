#include "native_device.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <thread>

#include <gtest/gtest.h>
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "hyle/platform/dynlib.h"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
namespace bridge = hyle::compute::ggml_vulkan::bridge;
using hyle::compute::ggml_vulkan::detail::NativeDevice;
using hyle::compute::ggml_vulkan::detail::NativeOptions;

class GgmlVulkanNative : public testing::Test {
  void SetUp() override {
#if defined(_WIN32)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
                 SEM_NOOPENFILEERRORBOX);
#endif
  }
};

TEST_F(GgmlVulkanNative, RejectsInvalidLoaderAndPhysicalIndex) {
  NativeOptions options;
  options.loader_path = "hyle-nonexistent-vulkan-loader";
  EXPECT_FALSE(NativeDevice::create(options));
  options.loader_path = nullptr;
  options.device_index = UINT32_MAX - 1;
  auto result = NativeDevice::create(options);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code(), hyle::StatusCode::OutOfRange);
}

TEST_F(GgmlVulkanNative, ExplicitValidationUsesItsOwnInstance) {
  if (!std::getenv("HYLE_TEST_VULKAN_VALIDATE"))
    GTEST_SKIP() << "requires the declared validation-layer environment";
  NativeOptions options;
  options.validation = true;
  auto result = NativeDevice::create(options);
  ASSERT_TRUE(result);
  EXPECT_NE(result->native().instance, VK_NULL_HANDLE);
}

TEST_F(GgmlVulkanNative, QuotaAndBorrowedOwnerRetireExactNativeStorage) {
  hyle::mem::QuotaAllocator empty(hyle::mem::system_allocator(), 0);
  NativeOptions options;
  options.quota = &empty;
  auto rejected = NativeDevice::create(options);
  ASSERT_TRUE(rejected);
  auto allocation = rejected->allocate_buffer(4096);
  ASSERT_FALSE(allocation);
  EXPECT_EQ(allocation.error().code(), hyle::StatusCode::ResourceExhausted);
  EXPECT_EQ(empty.used(), 0u);

  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1 << 20);
  options.quota = &quota;
  auto device = NativeDevice::create(options);
  ASSERT_TRUE(device);
  auto active = device->activate();
  auto storage = device->allocate_buffer(4096);
  ASSERT_TRUE(storage);
  std::shared_ptr<ggml_backend_buffer> owner(*storage, ggml_backend_buffer_free);
  const auto charged = quota.used();
  EXPECT_GE(charged, 4096u);
  auto borrowed = device->borrow(device->buffer_handle(owner.get()), 4096, owner);
  ASSERT_TRUE(borrowed);
  EXPECT_EQ(quota.used(), charged);
  std::weak_ptr<ggml_backend_buffer> alive = owner;
  owner.reset();
  EXPECT_FALSE(alive.expired());
  ggml_backend_buffer_free(*borrowed);
  EXPECT_TRUE(alive.expired());
  EXPECT_EQ(quota.used(), 0u);
}

TEST_F(GgmlVulkanNative, IndependentOwnersAndNestedThreadScopes) {
  const auto terminate = std::get_terminate();
  auto first = NativeDevice::create();
  auto second = NativeDevice::create();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_NE(first->native().instance, second->native().instance);
  EXPECT_NE(first->native().device, second->native().device);
  EXPECT_EQ(std::get_terminate(), terminate);
  std::array<bool, 2> passed{};
  auto work = [&](NativeDevice &device, std::size_t index) {
    auto active = device.activate();
    auto backend = device.create_backend();
    if (!backend) return;
    auto buffer = device.allocate_buffer(4096);
    if (buffer) {
      passed[index] = device.buffer_handle(*buffer) != VK_NULL_HANDLE;
      ggml_backend_buffer_free(*buffer);
    }
    ggml_backend_free(*backend);
  };
  std::thread a([&] { work(*first, 0); });
  std::thread b([&] { work(*second, 1); });
  a.join();
  b.join();
  EXPECT_TRUE(passed[0]);
  EXPECT_TRUE(passed[1]);
  auto outer = first->activate();
  ggml_backend_buffer_type_t second_type = nullptr;
  {
    auto inner = second->activate();
    auto backend = second->create_backend();
    ASSERT_TRUE(backend);
    second_type = ggml_backend_get_default_buffer_type(*backend);
    ggml_backend_free(*backend);
  }
  auto backend = first->create_backend();
  ASSERT_TRUE(backend);
  EXPECT_TRUE(ggml_backend_supports_buft(
      *backend, ggml_backend_get_default_buffer_type(*backend)));
  EXPECT_FALSE(ggml_backend_supports_buft(*backend, second_type));
  ggml_backend_free(*backend);
}

// Faults wrap the real loader and real objects. There is no mock backend or
// successful fake GPU computation. Calls remain serial in this test fixture.
struct Faults {
  PFN_vkGetInstanceProcAddr get = nullptr;
  PFN_vkCreateDevice create_device = nullptr;
  PFN_vkDestroyDevice destroy_device = nullptr;
  PFN_vkCreateFence create_fence = nullptr;
  PFN_vkDestroyFence destroy_fence = nullptr;
  PFN_vkCreateBuffer create_buffer = nullptr;
  PFN_vkDestroyBuffer destroy_buffer = nullptr;
  PFN_vkAllocateMemory allocate_memory = nullptr;
  PFN_vkFreeMemory free_memory = nullptr;
  PFN_vkBindBufferMemory bind_buffer = nullptr;
  int devices = 0, fences = 0, buffers = 0, memories = 0;
  int fence_calls = 0, fail_fence = 0;
  bool fail_bind = false;
};
Faults *faults = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL create_device(VkPhysicalDevice physical,
    const VkDeviceCreateInfo *info, const VkAllocationCallbacks *allocator,
    VkDevice *result) {
  auto status = faults->create_device(physical, info, allocator, result);
  if (status == VK_SUCCESS) ++faults->devices;
  return status;
}
VKAPI_ATTR void VKAPI_CALL destroy_device(VkDevice device,
    const VkAllocationCallbacks *allocator) {
  if (device) --faults->devices;
  faults->destroy_device(device, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL create_fence(VkDevice device,
    const VkFenceCreateInfo *info, const VkAllocationCallbacks *allocator,
    VkFence *result) {
  if (++faults->fence_calls == faults->fail_fence)
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  auto status = faults->create_fence(device, info, allocator, result);
  if (status == VK_SUCCESS) ++faults->fences;
  return status;
}
VKAPI_ATTR void VKAPI_CALL destroy_fence(VkDevice device, VkFence fence,
    const VkAllocationCallbacks *allocator) {
  if (fence) --faults->fences;
  faults->destroy_fence(device, fence, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice device,
    const VkBufferCreateInfo *info, const VkAllocationCallbacks *allocator,
    VkBuffer *result) {
  auto status = faults->create_buffer(device, info, allocator, result);
  if (status == VK_SUCCESS) ++faults->buffers;
  return status;
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer,
    const VkAllocationCallbacks *allocator) {
  if (buffer) --faults->buffers;
  faults->destroy_buffer(device, buffer, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device,
    const VkMemoryAllocateInfo *info, const VkAllocationCallbacks *allocator,
    VkDeviceMemory *result) {
  auto status = faults->allocate_memory(device, info, allocator, result);
  if (status == VK_SUCCESS) ++faults->memories;
  return status;
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory memory,
    const VkAllocationCallbacks *allocator) {
  if (memory) --faults->memories;
  faults->free_memory(device, memory, allocator);
}
VKAPI_ATTR VkResult VKAPI_CALL bind_buffer(VkDevice device, VkBuffer buffer,
    VkDeviceMemory memory, VkDeviceSize offset) {
  if (faults->fail_bind) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  return faults->bind_buffer(device, buffer, memory, offset);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get(VkInstance instance,
                                            const char *name) {
  auto original = faults->get(instance, name);
  if (!original) return nullptr;
#define INTERCEPT(VkName, member) \
  if (std::strcmp(name, #VkName) == 0) { \
    faults->member = reinterpret_cast<PFN_##VkName>(original); \
    return reinterpret_cast<PFN_vkVoidFunction>(member); \
  }
  INTERCEPT(vkCreateDevice, create_device)
  INTERCEPT(vkDestroyDevice, destroy_device)
  INTERCEPT(vkCreateFence, create_fence)
  INTERCEPT(vkDestroyFence, destroy_fence)
  INTERCEPT(vkCreateBuffer, create_buffer)
  INTERCEPT(vkDestroyBuffer, destroy_buffer)
  INTERCEPT(vkAllocateMemory, allocate_memory)
  INTERCEPT(vkFreeMemory, free_memory)
  INTERCEPT(vkBindBufferMemory, bind_buffer)
#undef INTERCEPT
  return original;
}

TEST_F(GgmlVulkanNative, RealNativeCreationFailuresRollBackHandles) {
#if defined(_WIN32)
  const char *name = "vulkan-1.dll";
#elif defined(__APPLE__)
  const char *name = "libvulkan.1.dylib";
#else
  const char *name = "libvulkan.so.1";
#endif
  auto loader = hyle::platform::DynLib::open(name);
  ASSERT_TRUE(loader);
  auto lookup = loader->symbol_as<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  ASSERT_TRUE(lookup);
  Faults state;
  state.get = *lookup;
  faults = &state;
  struct Reset { ~Reset() { faults = nullptr; } } reset;
  bridge::Options options;
  options.get_instance_proc_addr = get;
  bridge::Runtime *runtime = nullptr;
  bridge::Native native;
  state.fail_fence = 1;
  auto failure = bridge::create(options, &runtime, &native);
  EXPECT_EQ(failure.native, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  EXPECT_EQ(runtime, nullptr);
  EXPECT_EQ(state.devices, 0);
  EXPECT_EQ(state.fences, 0);

  state.fail_fence = 0;
  ASSERT_EQ(bridge::create(options, &runtime, &native).error, bridge::Error::None);
  std::unique_ptr<bridge::Runtime, decltype(&bridge::destroy)> owner(runtime, bridge::destroy);
  state.fail_fence = state.fence_calls + 2; // second backend fence, after the first succeeded
  ggml_backend *backend = nullptr;
  failure = bridge::create_backend(runtime, &backend);
  EXPECT_EQ(failure.native, VK_ERROR_OUT_OF_DEVICE_MEMORY);
  EXPECT_EQ(backend, nullptr);
  EXPECT_EQ(state.fences, 1); // only the persistent device fence
  state.fail_fence = 0;
  state.fail_bind = true;
  ggml_backend_buffer *buffer = nullptr;
  EXPECT_NE(bridge::allocate_buffer(runtime, 4096, &buffer).error, bridge::Error::None);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(state.buffers, 0);
  EXPECT_EQ(state.memories, 0);
  owner.reset();
  EXPECT_EQ(state.devices, 0);
  EXPECT_EQ(state.fences, 0);
}
} // namespace
