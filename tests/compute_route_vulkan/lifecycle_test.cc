#include "hyle/compute/vulkan.h"

#include <array>
#include <atomic>
#include <barrier>
#include <memory>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "fixture.h"
#include "vulkan_internal.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
namespace t = vulkan_test;

// Test-only interception of one real Context's native PFNs. All successful
// execution is still actual Vulkan work on the device. No replacement Provider
// or product test hook is introduced; each injected failure is at a specified
// non-accepting allocation/submit or non-terminal observation boundary.
struct NativeBoundary final {
  d::Device &device;
  d::DeviceFunctions original;
  VkPhysicalDeviceMemoryProperties original_memory;
  std::atomic<int> waits{0}, resets{0};
  int flushes = 0, invalidations = 0;
  bool fail_submit = false, fail_poll = false, fail_wait = false,
       fail_idle = false;
  bool fail_bind_memory = false, fail_descriptors = false;
  static inline NativeBoundary *active = nullptr;

  explicit NativeBoundary(const v::Context &context)
      : device(*d::Access::runtime(context)->device), original(device.fn),
        original_memory(device.memory_properties) {
    active = this;
    device.fn.QueueSubmit = queue_submit;
    device.fn.GetFenceStatus = get_fence;
    device.fn.WaitForFences = wait_fences;
    device.fn.QueueWaitIdle = wait_idle;
    device.fn.ResetFences = reset_fences;
    device.fn.BindBufferMemory = bind_memory;
    device.fn.AllocateDescriptorSets = allocate_descriptors;
    device.fn.FlushMappedMemoryRanges = flush_ranges;
    device.fn.InvalidateMappedMemoryRanges = invalidate_ranges;
  }
  ~NativeBoundary() {
    device.fn = original;
    device.memory_properties = original_memory;
    active = nullptr;
  }
  static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue,
                                                     std::uint32_t count,
                                                     const VkSubmitInfo *info,
                                                     VkFence fence) {
    if (std::exchange(active->fail_submit, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.QueueSubmit(queue, count, info, fence);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL get_fence(VkDevice device,
                                                  VkFence fence) {
    if (std::exchange(active->fail_poll, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.GetFenceStatus(device, fence);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL wait_fences(VkDevice device,
                                                    std::uint32_t count,
                                                    const VkFence *fences,
                                                    VkBool32 all,
                                                    std::uint64_t timeout) {
    ++active->waits;
    if (std::exchange(active->fail_wait, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.WaitForFences(device, count, fences, all, timeout);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkQueue queue) {
    if (std::exchange(active->fail_idle, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.QueueWaitIdle(queue);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL reset_fences(VkDevice device,
                                                     std::uint32_t count,
                                                     const VkFence *fences) {
    ++active->resets;
    return active->original.ResetFences(device, count, fences);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL bind_memory(VkDevice device,
                                                    VkBuffer buffer,
                                                    VkDeviceMemory memory,
                                                    VkDeviceSize offset) {
    if (std::exchange(active->fail_bind_memory, false))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return active->original.BindBufferMemory(device, buffer, memory, offset);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  allocate_descriptors(VkDevice device, const VkDescriptorSetAllocateInfo *info,
                       VkDescriptorSet *sets) {
    if (std::exchange(active->fail_descriptors, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.AllocateDescriptorSets(device, info, sets);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL flush_ranges(
      VkDevice device, std::uint32_t count, const VkMappedMemoryRange *ranges) {
    ++active->flushes;
    return active->original.FlushMappedMemoryRanges(device, count, ranges);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL invalidate_ranges(
      VkDevice device, std::uint32_t count, const VkMappedMemoryRange *ranges) {
    ++active->invalidations;
    return active->original.InvalidateMappedMemoryRanges(device, count, ranges);
  }
};

TEST(VulkanLifetime, NoncoherentTransferBranchUsesRealFlushAndInvalidate) {
  auto context = t::context();
  ASSERT_TRUE(context);
  NativeBoundary boundary(*context);
  // Exercise the required flush/invalidate mechanism even when this machine
  // only exposes coherent host memory. The actual allocation and Vulkan calls
  // remain real; this does not qualify a physical noncoherent memory type.
  for (std::uint32_t index = 0;
       index < boundary.device.memory_properties.memoryTypeCount; ++index)
    boundary.device.memory_properties.memoryTypes[index].propertyFlags &=
        ~static_cast<VkMemoryPropertyFlags>(
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  const auto host = t::data(257, 3);
  auto tensor = t::upload(*context, t::dense({257}), host);
  ASSERT_TRUE(tensor) << tensor.error().message();
  auto actual = t::read(*context, *tensor);
  ASSERT_TRUE(actual) << actual.error().message();
  EXPECT_EQ(*actual, host);
  EXPECT_EQ(boundary.flushes, 1);
  EXPECT_EQ(boundary.invalidations, 1);
}

TEST(VulkanLifetime, SubmitOomDoesNotWaitForUnacceptedFenceAndCanRetry) {
  auto context = t::context();
  ASSERT_TRUE(context) << context.error().message();
  const auto description = t::dense({2, 2});
  auto input = t::upload(*context, description, t::data(4, 1));
  ASSERT_TRUE(input);
  auto program = v::prepare(*context, c::MatMul{}, description, description);
  ASSERT_TRUE(program);
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot) << slot.error().message();
  ASSERT_TRUE(slot->bind(*input, *input).ok());
  NativeBoundary boundary(*context);
  boundary.fail_submit = true;
  auto refused = slot->submit();
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().IsResourceExhausted());
  EXPECT_EQ(boundary.waits.load(), 0);
  EXPECT_EQ(d::Access::runtime(*context)->pending, nullptr);
  auto retry = slot->submit();
  ASSERT_TRUE(retry) << retry.error().message();
  ASSERT_TRUE(retry->wait());
}

TEST(VulkanLifetime, ObservationOomRetainsWorkUntilActualNativeRetirement) {
  auto context = t::context();
  ASSERT_TRUE(context);
  const auto description = t::dense({2, 2});
  auto input = t::upload(*context, description, t::data(4, 1));
  ASSERT_TRUE(input);
  auto program = v::prepare(*context, c::MatMul{}, description, description);
  ASSERT_TRUE(program);
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot);
  ASSERT_TRUE(slot->bind(*input, *input).ok());
  NativeBoundary boundary(*context);
  auto work = slot->submit();
  ASSERT_TRUE(work);
  boundary.fail_poll = true;
  auto observed = work->poll();
  ASSERT_TRUE(observed);
  EXPECT_EQ(*observed, c::CompletionPoll::kPending);
  boundary.fail_poll = true;
  EXPECT_FALSE(work->result());
  EXPECT_NE(d::Access::runtime(*context)->pending, nullptr);
  EXPECT_FALSE(slot->bind(*input, *input).ok());
  boundary.fail_wait = true;
  boundary.fail_idle = true;
  const auto incomplete = context->close();
  EXPECT_TRUE(incomplete.IsResourceExhausted());
  EXPECT_NE(d::Access::runtime(*context)->pending, nullptr);
  EXPECT_EQ(boundary.resets.load(), 1);
  // Now an actual native fence wait proves completion. Preserve the first
  // observation diagnosis, and never publish a successful generated result.
  auto terminal = work->wait();
  ASSERT_FALSE(terminal);
  EXPECT_TRUE(terminal.error().IsResourceExhausted());
  EXPECT_FALSE(work->result());
  EXPECT_EQ(d::Access::runtime(*context)->pending, nullptr);
  EXPECT_TRUE(context->close().ok());
}

TEST(VulkanLifetime,
     PartialAllocationFailureReleasesNativeQuotaAndDescriptors) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1024 * 1024);
  auto context = t::context(&quota);
  ASSERT_TRUE(context);
  const auto description = t::dense({2, 2});
  auto program = v::prepare(*context, c::MatMul{}, description, description);
  ASSERT_TRUE(program);
  NativeBoundary boundary(*context);
  const auto before = quota.used();
  boundary.fail_bind_memory = true;
  auto failed_buffer = context->allocate(description);
  ASSERT_FALSE(failed_buffer);
  EXPECT_TRUE(failed_buffer.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), before);
  boundary.fail_descriptors = true;
  auto failed_slot = program->create_invocation();
  ASSERT_FALSE(failed_slot);
  EXPECT_TRUE(failed_slot.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), before);
  EXPECT_EQ(d::Access::runtime(*context)->device->descriptors_in_pools.load(),
            0u);
  EXPECT_TRUE(program->create_invocation());
}

TEST(VulkanLifetime, PackedProgramDoesNotRetainAnUnusedDenseWeightCopy) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1024 * 1024);
  auto context = t::context(&quota);
  ASSERT_TRUE(context);
  const auto host = t::data(67 * 37, 2);
  auto weights = t::upload(*context, t::dense({67, 37}), host);
  ASSERT_TRUE(weights);
  const auto one_allocation = quota.used();
  std::weak_ptr<const void> original = weights->native_owner();
  auto program = v::prepare(*context, c::MatMul{}, t::dense({1, 67}), *weights);
  ASSERT_TRUE(program) << program.error().message();
  *weights = {};
  EXPECT_TRUE(original.expired());
  EXPECT_EQ(quota.used(), one_allocation);
  const auto input_host = t::data(67, 3);
  auto input = t::upload(*context, t::dense({1, 67}), input_host);
  ASSERT_TRUE(input);
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot);
  ASSERT_TRUE(slot->bind(*input).ok());
  auto work = slot->submit();
  ASSERT_TRUE(work);
  auto lease = work->wait();
  ASSERT_TRUE(lease);
  auto actual = t::read(*context, (*lease)->tensors()[0]);
  ASSERT_TRUE(actual);
  const auto expected = t::matmul(input_host, host, 1, 67, 37);
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR((*actual)[i], expected[i], 1e-4f);
}

TEST(VulkanLifetime,
     CancellationAndAbandonmentKeepOwnersWithoutDestructorWait) {
  auto context = t::context();
  ASSERT_TRUE(context);
  const auto description = t::dense({2, 2});
  auto input = t::upload(*context, description, t::data(4, 1));
  ASSERT_TRUE(input);
  std::weak_ptr<const void> owner = input->native_owner();
  auto program = v::prepare(*context, c::MatMul{}, description, description);
  ASSERT_TRUE(program);
  auto first = program->create_invocation();
  auto next = program->create_invocation();
  ASSERT_TRUE(first && next);
  ASSERT_TRUE(first->bind(*input, *input).ok());
  ASSERT_TRUE(next->bind(*input, *input).ok());
  NativeBoundary boundary(*context);
  auto cancelled = first->submit();
  ASSERT_TRUE(cancelled);
  EXPECT_TRUE(cancelled->request_cancel().ok());
  auto failure = cancelled->wait();
  ASSERT_FALSE(failure);
  EXPECT_TRUE(failure.error().IsCancelled());
  cancelled = c::OwnedResult<v::Results>{};
  boundary.waits.store(0);
  auto abandoned = next->submit();
  ASSERT_TRUE(abandoned);
  abandoned = c::OwnedResult<v::Results>{};
  first = v::Invocation{};
  next = v::Invocation{};
  program = v::Program{};
  input = v::Tensor{};
  EXPECT_EQ(boundary.waits.load(), 0);
  EXPECT_FALSE(owner.expired());
  EXPECT_TRUE(context->close().ok());
  EXPECT_TRUE(owner.expired());
}

TEST(VulkanLifetime, ExportedTensorAndNativeTokenPreventSlotOverwrite) {
  auto context = t::context();
  ASSERT_TRUE(context);
  const auto description = t::dense({2, 2});
  auto input = t::upload(*context, description, t::data(4, 1));
  ASSERT_TRUE(input);
  auto program = v::prepare(*context, c::MatMul{}, description, description);
  ASSERT_TRUE(program);
  auto slot = program->create_invocation();
  ASSERT_TRUE(slot);
  ASSERT_TRUE(slot->bind(*input, *input).ok());
  v::Tensor exported;
  {
    auto work = slot->submit();
    ASSERT_TRUE(work);
    auto result = work->wait();
    ASSERT_TRUE(result);
    exported = result->get()->tensors()[0];
  }
  EXPECT_FALSE(slot->submit());
  auto token = exported.native_owner();
  exported = v::Tensor{};
  EXPECT_FALSE(slot->submit());
  token.reset();
  auto replay = slot->submit();
  ASSERT_TRUE(replay) << replay.error().message();
  auto result = replay->wait();
  ASSERT_TRUE(result);
  exported = result->get()->tensors()[0];
  EXPECT_TRUE(context->close().ok());
  EXPECT_TRUE(t::read(*context, exported));
}

TEST(VulkanLifetime, IndependentThreadsReplayWhileContextCollects) {
  auto context = t::context();
  ASSERT_TRUE(context);
  const auto description = t::dense({2, 2});
  auto input = t::upload(*context, description, t::data(4, 1));
  ASSERT_TRUE(input);
  auto program = v::prepare(*context, c::MatMul{}, description, description);
  ASSERT_TRUE(program);
  auto first = program->create_invocation();
  auto second = program->create_invocation();
  ASSERT_TRUE(first && second);
  std::barrier start(3);
  std::atomic<int> failures{0}, finished{0};
  auto execute = [&](v::Invocation &slot) {
    start.arrive_and_wait();
    for (int repetition = 0; repetition < 32; ++repetition) {
      if (!slot.bind(*input, *input).ok()) {
        ++failures;
        break;
      }
      auto run = slot.submit();
      if (!run || !run->wait()) {
        ++failures;
        break;
      }
    }
    ++finished;
  };
  std::thread left(execute, std::ref(*first));
  std::thread right(execute, std::ref(*second));
  start.arrive_and_wait();
  while (finished.load() != 2) {
    if (!context->collect().ok())
      ++failures;
    std::this_thread::yield();
  }
  left.join();
  right.join();
  EXPECT_EQ(failures.load(), 0);
  EXPECT_TRUE(context->close().ok());
  EXPECT_EQ(d::Access::runtime(*context)->pending, nullptr);
}

} // namespace
