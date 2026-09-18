#include "native_device.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

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
  PFN_vkQueueSubmit queue_submit = nullptr;
  PFN_vkQueueWaitIdle queue_wait_idle = nullptr;
  PFN_vkGetFenceStatus get_fence_status = nullptr;
  int devices = 0, fences = 0, buffers = 0, memories = 0;
  int fence_calls = 0, fail_fence = 0;
  int submit_calls = 0, accepted_submits = 0, fail_submit = 0;
  int fenced_submits = 0, empty_submits = 0;
  int idle_calls = 0;
  bool fail_bind = false;
  bool fail_idle = false;
  bool fail_observation = false;
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
VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, uint32_t count,
    const VkSubmitInfo *submits, VkFence fence) {
  if (++faults->submit_calls == faults->fail_submit)
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  const auto result = faults->queue_submit(queue, count, submits, fence);
  if (result == VK_SUCCESS) {
    ++faults->accepted_submits;
    if (fence) ++faults->fenced_submits;
    bool empty = true;
    for (uint32_t i = 0; i < count; ++i)
      empty = empty && submits[i].commandBufferCount == 0;
    if (empty) ++faults->empty_submits;
  }
  return result;
}
VKAPI_ATTR VkResult VKAPI_CALL queue_wait_idle(VkQueue queue) {
  ++faults->idle_calls;
  if (faults->fail_idle) return VK_ERROR_OUT_OF_HOST_MEMORY;
  return faults->queue_wait_idle(queue);
}
VKAPI_ATTR VkResult VKAPI_CALL get_fence_status(VkDevice device, VkFence fence) {
  if (faults->fail_observation) return VK_ERROR_OUT_OF_HOST_MEMORY;
  return faults->get_fence_status(device, fence);
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
  INTERCEPT(vkQueueSubmit, queue_submit)
  INTERCEPT(vkQueueWaitIdle, queue_wait_idle)
  INTERCEPT(vkGetFenceStatus, get_fence_status)
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

TEST_F(GgmlVulkanNative, PartialRegionSubmitRetainsResourcesUntilDrainIsProved) {
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
  ASSERT_EQ(bridge::create(options, &runtime, &native).error, bridge::Error::None);
  std::unique_ptr<bridge::Runtime, decltype(&bridge::destroy)> owner(runtime, bridge::destroy);
  {
    struct Scope {
      void *previous;
      ~Scope() { bridge::leave(previous); }
    } scope{bridge::enter(runtime)};
    constexpr int nodes = 256;
    auto arena = hyle::compute::ggml_vulkan::detail::MetadataArena::create(
        (nodes + 8) * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes + 8, false),
        hyle::mem::system_allocator());
    ASSERT_TRUE(arena);
    auto *context = arena->get();
    auto *input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
    auto *weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
    auto *output = input;
    for (int i = 0; i < nodes; ++i) output = ggml_add(context, output, weight);
    auto *graph = ggml_new_graph_custom(context, nodes + 8, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer *buffer = nullptr;
    ASSERT_EQ(bridge::allocate_context(runtime, context, &buffer).error, bridge::Error::None);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
        storage(buffer, ggml_backend_buffer_free);
    const std::vector<float> values(2048, 0.25f);
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(weight, values.data(), 0, values.size() * sizeof(float));
    bridge::Work *work = nullptr;
    ASSERT_EQ(bridge::create_work(runtime, &work).error, bridge::Error::None);
    std::unique_ptr<bridge::Work, decltype(&bridge::destroy_work)> execution(work, bridge::destroy_work);

    const auto accepted_before = state.accepted_submits;
    state.fail_submit = state.submit_calls + 2;
    const auto submitted = bridge::submit_work(work, graph);
    ASSERT_EQ(submitted.failure.native, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    EXPECT_TRUE(submitted.pending);
    EXPECT_GT(state.accepted_submits, accepted_before); // a real accepted prefix
    const auto submissions_after_failure = state.submit_calls;
    const auto buffers_while_pending = state.buffers;
    const auto memories_while_pending = state.memories;
    state.fail_idle = true;
    auto observed = bridge::observe_work(work, false);
    EXPECT_TRUE(observed.pending);
    EXPECT_EQ(state.idle_calls, 0); // nonblocking poll does not queue-drain
    observed = bridge::observe_work(work, true);
    EXPECT_TRUE(observed.pending);
    EXPECT_EQ(observed.failure.native, VK_ERROR_OUT_OF_DEVICE_MEMORY); // first diagnosis
    EXPECT_GT(state.idle_calls, 0);
    EXPECT_EQ(state.buffers, buffers_while_pending);
    EXPECT_EQ(state.memories, memories_while_pending);
    state.fail_idle = false;
    observed = bridge::observe_work(work, true);
    EXPECT_FALSE(observed.pending);
    EXPECT_EQ(observed.failure.native, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    EXPECT_EQ(state.submit_calls, submissions_after_failure); // never submit residual encoding
    EXPECT_FALSE(bridge::submit_work(work, graph).pending);
    EXPECT_EQ(state.submit_calls, submissions_after_failure);
  }
  owner.reset();
  EXPECT_EQ(state.devices, 0);
  EXPECT_EQ(state.fences, 0);
  EXPECT_EQ(state.buffers, 0);
  EXPECT_EQ(state.memories, 0);
}
TEST_F(GgmlVulkanNative, FenceObservationFailureKeepsTheAcceptedRegionPending) {
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
  ASSERT_EQ(bridge::create(options, &runtime, &native).error, bridge::Error::None);
  std::unique_ptr<bridge::Runtime, decltype(&bridge::destroy)> owner(runtime, bridge::destroy);
  {
    struct Scope {
      void *previous;
      ~Scope() { bridge::leave(previous); }
    } scope{bridge::enter(runtime)};
    auto arena = hyle::compute::ggml_vulkan::detail::MetadataArena::create(
        8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false),
        hyle::mem::system_allocator());
    ASSERT_TRUE(arena);
    auto *context = arena->get();
    auto *input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2048);
    auto *output = ggml_silu(context, input);
    auto *graph = ggml_new_graph_custom(context, 8, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer *buffer = nullptr;
    ASSERT_EQ(bridge::allocate_context(runtime, context, &buffer).error, bridge::Error::None);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
        storage(buffer, ggml_backend_buffer_free);
    const std::vector<float> values(2048, 0.25f);
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    bridge::Work *work = nullptr;
    ASSERT_EQ(bridge::create_work(runtime, &work).error, bridge::Error::None);
    std::unique_ptr<bridge::Work, decltype(&bridge::destroy_work)> execution(work, bridge::destroy_work);

    const auto fences_before = state.fenced_submits;
    const auto markers_before = state.empty_submits;
    const auto idle_before = state.idle_calls;
    const auto submitted = bridge::submit_work(work, graph);
    ASSERT_EQ(submitted.failure.error, bridge::Error::None);
    ASSERT_TRUE(submitted.pending);
    EXPECT_GT(state.fenced_submits, fences_before);
    RecordProperty("region_empty_submits", state.empty_submits - markers_before);
    EXPECT_EQ(state.idle_calls, idle_before);
    const auto submitted_count = state.submit_calls;
    const auto buffers_before = state.buffers;
    const auto memories_before = state.memories;
    state.fail_observation = true;
    auto observed = bridge::observe_work(work, false);
    EXPECT_TRUE(observed.pending);
    EXPECT_EQ(observed.failure.native, VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(state.idle_calls, idle_before);
    state.fail_observation = false;
    state.fail_idle = true;
    observed = bridge::observe_work(work, true);
    EXPECT_TRUE(observed.pending);
    EXPECT_EQ(state.buffers, buffers_before);
    EXPECT_EQ(state.memories, memories_before);
    state.fail_idle = false;
    observed = bridge::observe_work(work, true);
    EXPECT_FALSE(observed.pending);
    EXPECT_EQ(observed.failure.native, VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(state.submit_calls, submitted_count);
  }
  owner.reset();
  EXPECT_EQ(state.devices, 0);
  EXPECT_EQ(state.fences, 0);
  EXPECT_EQ(state.buffers, 0);
  EXPECT_EQ(state.memories, 0);
}
} // namespace
