#include "hyle/compute/vulkan.h"

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "fixture.h"
#include "hyle/compute/detail/vulkan_plan.h"
#include "scan_test_support.h"
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
  std::atomic<bool> hold_wait{false}, waiting{false}, release_wait{false},
      wait_timed_out{false};
  int submissions = 0, fail_submit_on = 0;
  int flushes = 0, invalidations = 0;
  bool fail_submit = false, fail_poll = false, fail_wait = false,
       fail_idle = false;
  bool fail_bind_memory = false, fail_descriptors = false;
  bool fail_flush = false, fail_invalidate = false;
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
    if (++active->submissions == active->fail_submit_on ||
        std::exchange(active->fail_submit, false))
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
    if (active->hold_wait.exchange(false)) {
      active->waiting.store(true);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!active->release_wait.load() &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      active->wait_timed_out.store(!active->release_wait.load());
      active->waiting.store(false);
    }
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
    if (std::exchange(active->fail_flush, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.FlushMappedMemoryRanges(device, count, ranges);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL invalidate_ranges(
      VkDevice device, std::uint32_t count, const VkMappedMemoryRange *ranges) {
    ++active->invalidations;
    if (std::exchange(active->fail_invalidate, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.InvalidateMappedMemoryRanges(device, count, ranges);
  }
};

TEST(VulkanLifetime, IreeCheckedScanCopyFailuresDrainAcceptedPrefix) {
  for (int rejected_copy : {1, 2, 3}) {
    SCOPED_TRACE(rejected_copy);
    auto context = t::context().value();
    const auto desc = vulkan_tensor_test::dense(c::DataType::kInt32, {257});
    const auto data = vulkan_scan_test::integer_pattern(257);
    const auto input = vulkan_scan_test::make_integer(context, desc, data);
    auto program = v::prepare(context, c::Scan{}, desc).value();
    auto slot = program.create_invocation().value();
    ASSERT_TRUE(slot.bind(input).ok());
    NativeBoundary boundary(context);
    // Original IREE submissions use their own PFNs. These failures reject
    // only the HYLE input copy, flag readback, or final output copy.
    boundary.fail_submit_on = rejected_copy;
    auto work = slot.submit();
    if (rejected_copy == 1) {
      ASSERT_FALSE(work);
      EXPECT_TRUE(work.error().IsResourceExhausted());
    } else {
      ASSERT_TRUE(work) << work.error().message();
      auto result = work->wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsResourceExhausted());
    }
    EXPECT_TRUE(slot.bind(input).IsResourceExhausted());
    auto fresh = program.create_invocation().value();
    ASSERT_TRUE(fresh.bind(input).ok());
    auto retry = fresh.submit().value();
    auto result = retry.wait();
    ASSERT_TRUE(result) << result.error().message();
    vulkan_scan_test::expect_integer_prefix(
        context, result->get()->tensors()[0], desc, data, c::Scan{});
    EXPECT_TRUE(context.close().ok());
  }
}

TEST(VulkanLifetime, IreeCancellationReusesSlotAndCloseDrainsAbandonment) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  128 * 1024 * 1024);
  {
    auto context = t::context(&quota).value();
    const auto desc = vulkan_tensor_test::dense(c::DataType::kInt32, {257});
    const auto data = vulkan_scan_test::integer_pattern(257);
    const auto input = vulkan_scan_test::make_integer(context, desc, data);
    auto program = v::prepare(context, c::Scan{}, desc).value();
    auto first = program.create_invocation().value();
    auto second = program.create_invocation().value();
    ASSERT_TRUE(first.bind(input).ok());
    ASSERT_TRUE(second.bind(input).ok());
    for (int phase = 0; phase < 2; ++phase) {
      {
        auto work = first.submit().value();
        ASSERT_TRUE(work.request_cancel().ok());
        auto result = work.wait();
        ASSERT_FALSE(result);
        EXPECT_TRUE(result.error().IsCancelled());
      }
      ASSERT_TRUE(first.bind(input).ok());
      auto work = first.submit().value();
      ASSERT_TRUE(work.wait());
    }
    {
      auto abandoned = second.submit().value();
    }
    EXPECT_TRUE(context.close().ok());
    EXPECT_FALSE(first.submit());
  }
  EXPECT_EQ(quota.used(), 0U);
}

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

TEST(VulkanLifetime, ManagedPrefixSurvivesRejectedNativeSuffixUntilDrain) {
  for (bool selection : {false, true}) {
    SCOPED_TRACE(selection);
    for (bool abandon : {false, true}) {
      SCOPED_TRACE(abandon);
      auto context = t::context().value();
      const auto description = t::dense({17, selection ? 288 : 2048});
      auto input = t::upload(context, description,
                             t::data(*description.element_count(), 1))
                       .value();
      std::weak_ptr<const void> owner = input.native_owner();
      d::Plan plan;
      plan.values.assign(4, {description, {}});
      plan.inputs = {0};
      plan.outputs = {3};
      plan.steps = {{c::Operation(c::Unary{c::UnaryKind::kRelu}),
                     {0},
                     {1},
                     {},
                     {1, 1, 1},
                     {}},
                    {c::Operation(c::RmsNorm{}), {1}, {2}, {}, {1, 1, 1}, {}},
                    {c::Operation(c::Unary{c::UnaryKind::kRelu}),
                     {2},
                     {3},
                     {},
                     {1, 1, 1},
                     {}}};
      if (selection) {
        const auto count =
            static_cast<std::size_t>(description.dimensions().back());
        plan.values.push_back(
            {c::TensorDesc::dense(c::DataType::kInt32, description.dimensions())
                 .value(),
             {}});
        plan.steps[1].operation = c::Operation(c::TopK{.count = count});
        plan.steps[1].outputs.push_back(4);
        plan.outputs.push_back(4);
      }
      d::PlanFacts facts;
      auto program = d::prepare_plan(context, plan, &facts).value();
      ASSERT_EQ(facts.native_dispatch_count, 2U);
      ASSERT_EQ(facts.managed_region_count, 1U);
      auto slot = program.create_invocation().value();
      ASSERT_TRUE(slot.bind(input).ok());
      NativeBoundary boundary(context);
      // Provider submits use their original PFNs. This rejects only the HYLE
      // suffix after the real HYLE prefix and real managed region were
      // accepted.
      boundary.fail_submit_on = 2;
      auto work = slot.submit();
      ASSERT_TRUE(work) << work.error().message();
      ASSERT_EQ(boundary.submissions, 2);
      ASSERT_NE(d::Access::runtime(context)->pending, nullptr);
      auto observed = work->poll();
      ASSERT_TRUE(observed);
      EXPECT_EQ(*observed, c::CompletionPoll::kPending);
      EXPECT_EQ(boundary.waits.load(), 0);
      if (abandon) {
        work = c::OwnedResult<v::Results>{};
        slot = {};
        program = {};
        input = {};
        EXPECT_FALSE(owner.expired());
        EXPECT_EQ(boundary.waits.load(), 0);
        boundary.fail_idle = true;
        EXPECT_TRUE(context.close().IsResourceExhausted());
        EXPECT_NE(d::Access::runtime(context)->pending, nullptr);
        EXPECT_FALSE(owner.expired());
        EXPECT_TRUE(context.close()
                        .IsResourceExhausted()); // Original submit diagnosis.
        EXPECT_EQ(d::Access::runtime(context)->pending, nullptr);
        EXPECT_TRUE(owner.expired());
      } else {
        auto failed = work->wait();
        ASSERT_FALSE(failed);
        EXPECT_TRUE(failed.error().IsResourceExhausted());
        EXPECT_FALSE(work->result());
        EXPECT_EQ(d::Access::runtime(context)->pending, nullptr);
        work = c::OwnedResult<v::Results>{};
        EXPECT_TRUE(slot.bind(input).IsResourceExhausted());
        auto retry = slot.submit();
        ASSERT_FALSE(retry);
        EXPECT_TRUE(retry.error().IsResourceExhausted());
      }
      EXPECT_TRUE(context.close().ok());
    }
  }
}

TEST(VulkanLifetime,
     ManagedWaitAllowsIndependentProducerAndCancelledSlotReuse) {
  auto context = t::context().value();
  const auto description = t::dense({17, 2048});
  auto input = t::upload(context, description, t::data(17 * 2048, 3)).value();
  auto program = v::prepare(context, c::RmsNorm{}, description).value();
  auto first = program.create_invocation().value();
  auto second = program.create_invocation().value();
  ASSERT_TRUE(first.bind(input).ok());
  ASSERT_TRUE(second.bind(input).ok());
  NativeBoundary boundary(context);
  {
    auto work = first.submit().value();
    boundary.hold_wait.store(true);
    std::atomic<bool> succeeded{false};
    std::thread waiter([&] { succeeded.store(work.wait().has_value()); });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!boundary.waiting.load() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    EXPECT_TRUE(boundary.waiting.load());
    auto independent = second.submit();
    EXPECT_TRUE(
        boundary.waiting.load()); // Submit completed during the held wait.
    boundary.release_wait.store(true);
    waiter.join();
    EXPECT_FALSE(boundary.wait_timed_out.load());
    EXPECT_TRUE(succeeded.load());
    ASSERT_TRUE(independent) << independent.error().message();
    ASSERT_TRUE(independent->wait());
  }
  {
    auto cancelled = first.submit().value();
    EXPECT_TRUE(cancelled.request_cancel().ok());
    auto result = cancelled.wait();
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsCancelled());
  }
  {
    auto replay = first.submit();
    ASSERT_TRUE(replay) << replay.error().message();
    ASSERT_TRUE(replay->wait());
  }
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanLifecycle, CheckedGatherFlushInvalidateCancelAndAbandon) {
  auto context = t::context().value();
  const auto source_desc = t::dense({5, 7});
  const auto source = t::upload(context, source_desc, t::data(35, 4)).value();
  const auto index_desc =
      c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{2})
          .value();
  const std::array<std::int32_t, 2> good_positions{4, 0}, bad_positions{4, -1};
  const auto good =
      context.upload(index_desc, std::as_bytes(std::span(good_positions)))
          .value();
  const auto bad =
      context.upload(index_desc, std::as_bytes(std::span(bad_positions)))
          .value();
  auto program = v::prepare(context, c::Gather{0}, source, index_desc).value();
  NativeBoundary boundary(context);
  // Exercise non-coherent control flow on real device memory. Clearing the
  // cached property does not claim that this physical device lacks coherence.
  for (std::uint32_t i = 0;
       i < boundary.device.memory_properties.memoryTypeCount; ++i)
    boundary.device.memory_properties.memoryTypes[i].propertyFlags &=
        ~VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  auto first = program.create_invocation().value();
  auto second = program.create_invocation().value();
  ASSERT_TRUE(first.bind(good).ok());
  ASSERT_TRUE(second.bind(bad).ok());
  {
    auto work = first.submit().value();
    ASSERT_TRUE(work.wait());
    EXPECT_GT(boundary.flushes, 0);
    EXPECT_GT(boundary.invalidations, 0);
  }
  {
    auto work = second.submit().value();
    ASSERT_TRUE(work.request_cancel().ok());
    const auto result = work.wait();
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsOutOfRange());
  }
  EXPECT_TRUE(second.bind(good).IsOutOfRange());
  {
    auto work = first.submit().value();
    ASSERT_TRUE(work.request_cancel().ok());
    const auto result = work.wait();
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsCancelled());
  }
  {
    auto work = first.submit().value();
    ASSERT_TRUE(work.wait());
  }
  auto abandoned = program.create_invocation().value();
  ASSERT_TRUE(abandoned.bind(bad).ok());
  {
    auto work = abandoned.submit().value();
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  auto drained = context.collect();
  while (drained.ok() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    drained = context.collect();
  }
  EXPECT_TRUE(drained.IsOutOfRange()) << drained.message();
  EXPECT_TRUE(abandoned.bind(good).IsOutOfRange());
  auto fresh = program.create_invocation().value();
  ASSERT_TRUE(fresh.bind(good).ok());
  auto result = fresh.submit().value();
  ASSERT_TRUE(result.wait());
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanLifecycle,
     CheckedGatherPartialAcceptanceDrainsAndPreservesNativeFailure) {
  auto context = t::context().value();
  const auto source_desc = t::dense({5, 7});
  const auto source = t::upload(context, source_desc, t::data(35, 2)).value();
  const auto index_desc =
      c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{2})
          .value();
  const std::array<std::int32_t, 2> positions{4, -1};
  const auto indices =
      context.upload(index_desc, std::as_bytes(std::span(positions))).value();
  auto program = v::prepare(context, c::Gather{0}, source, index_desc).value();
  auto slot = program.create_invocation().value();
  ASSERT_TRUE(slot.bind(indices).ok());
  NativeBoundary boundary(context);
  // Native check and original managed GetRows accept work; final HYLE output
  // publication rejects admission. Keep that native error and drain the prefix.
  boundary.fail_submit_on = 2;
  auto work = slot.submit();
  ASSERT_TRUE(work) << work.error().message();
  const auto result = work->wait();
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsResourceExhausted());
  EXPECT_TRUE(slot.bind(indices).IsResourceExhausted());
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanLifecycle,
     CheckedGatherMappedFailuresKeepAdmissionAndRetirementDistinct) {
  auto context = t::context().value();
  const auto source_desc = t::dense({5, 7});
  const auto source = t::upload(context, source_desc, t::data(35, 2)).value();
  const auto index_desc =
      c::TensorDesc::dense(c::DataType::kInt32, std::array<std::int64_t, 1>{1})
          .value();
  const std::array<std::int32_t, 1> position{4};
  const auto index =
      context.upload(index_desc, std::as_bytes(std::span(position))).value();
  auto program = v::prepare(context, c::Gather{0}, source, index_desc).value();
  NativeBoundary boundary(context);
  for (std::uint32_t i = 0;
       i < boundary.device.memory_properties.memoryTypeCount; ++i)
    boundary.device.memory_properties.memoryTypes[i].propertyFlags &=
        ~VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  auto slot = program.create_invocation().value();
  ASSERT_TRUE(slot.bind(index).ok());
  boundary.fail_flush = true;
  auto rejected = slot.submit();
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_EQ(boundary.submissions, 0);
  {
    auto accepted = slot.submit();
    ASSERT_TRUE(accepted) << accepted.error().message();
    boundary.fail_invalidate = true;
    auto result = accepted->wait();
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsResourceExhausted());
  }
  EXPECT_TRUE(slot.bind(index).IsResourceExhausted());
  auto fresh = program.create_invocation().value();
  ASSERT_TRUE(fresh.bind(index).ok());
  auto work = fresh.submit().value();
  ASSERT_TRUE(work.wait());
  EXPECT_TRUE(context.close().ok());
}

} // namespace
