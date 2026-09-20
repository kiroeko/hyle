#include "native_program.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
using hyle::compute::iree_vulkan::BufferSpan;
using hyle::compute::iree_vulkan::NativeDevice;
using hyle::compute::iree_vulkan::NativeProgram;
using hyle::compute::iree_vulkan::ProgramKind;
using hyle::compute::iree_vulkan::ProgramSpec;
using GgmlDevice = hyle::compute::ggml_vulkan::detail::NativeDevice;

void check(VkResult result) {
  if (result != VK_SUCCESS)
    throw std::runtime_error("test Vulkan transfer failed");
}

// Independent test transport. The product receives device-resident buffers;
// no IREE upload/download helper can accidentally hide a host math fallback.
struct Transfer final {
  GgmlDevice owner;
  VkDevice device;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  void *mapped = nullptr;
  PFN_vkGetDeviceProcAddr get;
  template <class T> T function(const char *name) const {
    return reinterpret_cast<T>(get(device, name));
  }
  explicit Transfer(GgmlDevice native)
      : owner(std::move(native)), device(owner.native().device) {
    const auto &vk = owner.native();
    get = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        vk.get_instance_proc_addr(vk.instance, "vkGetDeviceProcAddr"));
    VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    p.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    p.queueFamilyIndex = vk.queue_family;
    check(function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(
        device, &p, nullptr, &pool));
    VkCommandBufferAllocateInfo a{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = pool;
    a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    a.commandBufferCount = 1;
    check(function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(
        device, &a, &command));
    VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    b.size = 131072;
    b.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(function<PFN_vkCreateBuffer>("vkCreateBuffer")(device, &b, nullptr,
                                                         &buffer));
    VkMemoryRequirements requirements;
    function<PFN_vkGetBufferMemoryRequirements>(
        "vkGetBufferMemoryRequirements")(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties properties;
    reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        vk.get_instance_proc_addr(
            vk.instance, "vkGetPhysicalDeviceMemoryProperties"))(vk.physical,
                                                                 &properties);
    uint32_t index = 0;
    constexpr auto required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (; index < properties.memoryTypeCount; ++index)
      if ((requirements.memoryTypeBits & (1u << index)) &&
          (properties.memoryTypes[index].propertyFlags & required) == required)
        break;
    if (index == properties.memoryTypeCount)
      throw std::runtime_error("test requires coherent transfer memory");
    VkMemoryAllocateInfo m{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    m.allocationSize = requirements.size;
    m.memoryTypeIndex = index;
    check(function<PFN_vkAllocateMemory>("vkAllocateMemory")(device, &m,
                                                             nullptr, &memory));
    check(function<PFN_vkBindBufferMemory>("vkBindBufferMemory")(device, buffer,
                                                                 memory, 0));
    check(function<PFN_vkMapMemory>("vkMapMemory")(device, memory, 0,
                                                   VK_WHOLE_SIZE, 0, &mapped));
    VkFenceCreateInfo f{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(function<PFN_vkCreateFence>("vkCreateFence")(device, &f, nullptr,
                                                       &fence));
  }
  ~Transfer() {
    function<PFN_vkDestroyFence>("vkDestroyFence")(device, fence, nullptr);
    function<PFN_vkUnmapMemory>("vkUnmapMemory")(device, memory);
    function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(device, buffer, nullptr);
    function<PFN_vkFreeMemory>("vkFreeMemory")(device, memory, nullptr);
    function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(device, pool,
                                                               nullptr);
  }
  void copy(BufferSpan source, BufferSpan destination, std::size_t bytes) {
    check(
        function<PFN_vkResetCommandBuffer>("vkResetCommandBuffer")(command, 0));
    check(function<PFN_vkResetFences>("vkResetFences")(device, 1, &fence));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(command,
                                                                     &begin));
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    auto barriers = function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    barriers(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0,
             nullptr);
    VkBufferCopy region{source.offset, destination.offset, bytes};
    function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(
        command, source.buffer, destination.buffer, 1, &region);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    barriers(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0,
             nullptr);
    check(function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    auto result = owner.with_queue([&](VkQueue queue) {
      check(function<PFN_vkQueueSubmit>("vkQueueSubmit")(queue, 1, &submit,
                                                         fence));
      return hyle::Status::Ok();
    });
    if (!result.ok())
      throw std::runtime_error("native queue rejected test copy");
    check(function<PFN_vkWaitForFences>("vkWaitForFences")(
        device, 1, &fence, VK_TRUE, 10000000000ULL));
  }
  void upload(BufferSpan destination, const void *data) {
    ASSERT_LE(destination.bytes, 131072u);
    std::memcpy(mapped, data, destination.bytes);
    copy({buffer, 0, destination.bytes}, destination, destination.bytes);
  }
  void download(BufferSpan source, void *data) {
    ASSERT_LE(source.bytes, 131072u);
    copy(source, {buffer, 0, source.bytes}, source.bytes);
    std::memcpy(data, mapped, source.bytes);
  }
};

class IreeNative : public testing::Test {
  void SetUp() override {
#if defined(_WIN32)
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
                 SEM_NOOPENFILEERRORBOX);
#endif
  }
};

TEST_F(IreeNative, QuotaFailureRollsBackAndReleasesBorrowedOwner) {
  auto native = GgmlDevice::create();
  ASSERT_TRUE(native);
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  {
    auto device = NativeDevice::create(*native, &quota);
    if (device) {
      auto program =
          NativeProgram::create(*device, {ProgramKind::ScanInclusive, 2, 17});
      ASSERT_FALSE(program);
      EXPECT_EQ(program.error().code(), hyle::StatusCode::ResourceExhausted);
    } else {
      EXPECT_EQ(device.error().code(), hyle::StatusCode::ResourceExhausted);
    }
  }
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_FALSE(NativeDevice::create({}));
}

TEST_F(IreeNative, OriginalCheckedScansUseSharedDeviceAndRepeat) {
  auto native = GgmlDevice::create();
  ASSERT_TRUE(native);
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1ull << 30);
  {
    auto device = NativeDevice::create(*native, &quota);
    ASSERT_TRUE(device);
    Transfer transport(*native);
    for (auto kind : {ProgramKind::ScanInclusive, ProgramKind::ScanExclusive}) {
      auto program = NativeProgram::create(*device, {kind, 2, 5});
      ASSERT_TRUE(program) << program.error().message();
      ASSERT_FALSE((*program)->output(0));
      const auto input = (*program)->input(0);
      ASSERT_TRUE(input);
      for (int repeat = 0; repeat != 3; ++repeat) {
        const std::array<int32_t, 10> values{1, 2,  -3, 4,  5,
                                             7, -1, 3,  -4, repeat};
        transport.upload(*input, values.data());
        auto submitted = (*program)->invoke();
        ASSERT_TRUE(submitted.status.ok());
        auto completed = (*program)->wait();
        ASSERT_TRUE(completed.status.ok()) << completed.status.message();
        ASSERT_FALSE(completed.pending);
        auto output = (*program)->output(0), flag = (*program)->output(1);
        ASSERT_TRUE(output);
        ASSERT_TRUE(flag);
        std::array<int32_t, 10> actual{};
        int32_t overflow = -1;
        transport.download(*output, actual.data());
        transport.download(*flag, &overflow);
        EXPECT_EQ(overflow, 0);
        for (std::size_t row = 0; row != 2; ++row) {
          int64_t sum = 0;
          for (std::size_t column = 0; column != 5; ++column) {
            const auto i = row * 5 + column;
            const auto expected =
                kind == ProgramKind::ScanExclusive ? sum : sum + values[i];
            EXPECT_EQ(actual[i], expected);
            sum += values[i];
          }
        }
      }
    }
  }
  EXPECT_GT(quota.peak(), 0u);
  EXPECT_EQ(quota.used(), 0u);
}

TEST_F(IreeNative, OriginalFp8ProgramsUseCompressedInputAndIndependentSlots) {
  auto native = GgmlDevice::create();
  ASSERT_TRUE(native);
  auto device = NativeDevice::create(*native);
  ASSERT_TRUE(device);
  Transfer transport(*native);
  for (auto kind : {ProgramKind::Float8Block128, ProgramKind::Float8Group32}) {
    const ProgramSpec spec{kind, 2, 128, 128};
    auto first = NativeProgram::create(*device, spec),
         second = NativeProgram::create(*device, spec);
    ASSERT_TRUE(first) << first.error().message();
    ASSERT_TRUE(second) << second.error().message();
    for (auto *program : {first->get(), second->get()}) {
      ASSERT_EQ(program->input_count(), 4u);
      for (std::size_t i = 0; i != 4; ++i) {
        auto input = program->input(i);
        ASSERT_TRUE(input);
        if (i < 2 || kind == ProgramKind::Float8Group32) {
          const uint8_t activation = program == first->get() ? 0x38 : 0x40;
          std::vector<uint8_t> values(input->bytes, i == 0   ? activation
                                                    : i == 1 ? 0x40
                                                             : 127);
          transport.upload(*input, values.data());
        } else {
          std::vector<float> values(input->bytes / sizeof(float), 1.0f);
          transport.upload(*input, values.data());
        }
      }
    }
    const auto first_submission = (*first)->invoke();
    ASSERT_TRUE(first_submission.status.ok())
        << first_submission.status.message();
    const auto second_submission = (*second)->invoke();
    ASSERT_TRUE(second_submission.status.ok())
        << second_submission.status.message();
    for (auto *program : {first->get(), second->get()}) {
      const auto completion = program->wait();
      ASSERT_TRUE(completion.status.ok()) << completion.status.message();
      auto output = program->output(0);
      ASSERT_TRUE(output);
      ASSERT_EQ(output->bytes, 256 * sizeof(float));
      std::array<float, 256> actual;
      transport.download(*output, actual.data());
      for (float value : actual)
        EXPECT_FLOAT_EQ(value, program == first->get() ? 256.0f : 512.0f);
    }
  }
}

TEST_F(IreeNative,
       OriginalProactorAdvancesIndependentProgramsWithoutHostPolling) {
  auto native = GgmlDevice::create();
  ASSERT_TRUE(native);
  auto device = NativeDevice::create(*native);
  ASSERT_TRUE(device);
  Transfer transport(*native);
  auto first =
      NativeProgram::create(*device, {ProgramKind::ScanInclusive, 1, 3});
  auto second =
      NativeProgram::create(*device, {ProgramKind::ScanExclusive, 1, 3});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  const std::array<int32_t, 3> values{3, 4, -2};
  transport.upload(*(*first)->input(0), values.data());
  transport.upload(*(*second)->input(0), values.data());
  struct Chain {
    NativeProgram *first;
    NativeProgram *second;
    std::promise<hyle::Status> completion;
    bool output_ready = false;
  } chain{first->get(), second->get()};
  auto future = chain.completion.get_future();
  auto result = (*first)->invoke(
      [](void *data, const hyle::Status &status) noexcept {
        auto &chain = *static_cast<Chain *>(data);
        if (!status.ok()) {
          chain.completion.set_value(status);
          return;
        }
        chain.output_ready = chain.first->output(0).has_value();
        auto next = chain.second->invoke(
            [](void *data, const hyle::Status &status) noexcept {
              static_cast<Chain *>(data)->completion.set_value(status);
            },
            &chain);
        if (!next.status.ok())
          chain.completion.set_value(next.status);
      },
      &chain);
  ASSERT_TRUE(result.status.ok());
  ASSERT_EQ(future.wait_for(std::chrono::seconds(10)),
            std::future_status::ready);
  EXPECT_TRUE(future.get().ok());
  EXPECT_TRUE(chain.output_ready);
  EXPECT_TRUE((*first)->wait().status.ok());
  EXPECT_TRUE((*second)->wait().status.ok());
  std::array<int32_t, 3> actual{};
  transport.download(*(*second)->output(0), actual.data());
  EXPECT_EQ(actual, (std::array<int32_t, 3>{0, 3, 7}));
}

TEST_F(IreeNative, CheckedScanReportsIntermediateAndExclusiveFinalOverflow) {
  auto native = GgmlDevice::create();
  ASSERT_TRUE(native);
  auto device = NativeDevice::create(*native);
  ASSERT_TRUE(device);
  Transfer transport(*native);
  for (auto kind : {ProgramKind::ScanInclusive, ProgramKind::ScanExclusive}) {
    auto program = NativeProgram::create(*device, {kind, 1, 3});
    ASSERT_TRUE(program);
    for (const auto values : {std::array<int32_t, 3>{INT32_MAX, 1, -1},
                              std::array<int32_t, 3>{INT32_MAX, -1, 2},
                              std::array<int32_t, 3>{INT32_MIN, -1, 1}}) {
      transport.upload(*(*program)->input(0), values.data());
      ASSERT_TRUE((*program)->invoke().status.ok());
      ASSERT_TRUE((*program)->wait().status.ok());
      int32_t flag = 0;
      transport.download(*(*program)->output(1), &flag);
      EXPECT_EQ(flag, 1);
      std::array<int32_t, 3> unchanged;
      transport.download(*(*program)->input(0), unchanged.data());
      EXPECT_EQ(unchanged, values);
    }
  }
}

TEST_F(IreeNative, RuntimeQuotaFailureKeepsOutputsPrivateAndReleasesResources) {
  auto native = GgmlDevice::create();
  ASSERT_TRUE(native);
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1ull << 30);
  {
    auto device = NativeDevice::create(*native, &quota);
    ASSERT_TRUE(device);
    // A wide intermediate exceeds unused space in the original input slab.
    // A tiny scan can legitimately reuse that slab without any new charge.
    auto program = NativeProgram::create(
        *device, {ProgramKind::ScanInclusive, 1024, 8192});
    ASSERT_TRUE(program);
    Transfer transport(*native);
    const std::array<int32_t, 32768> values{};
    const auto input = *(*program)->input(0);
    for (std::size_t offset = 0; offset < input.bytes; offset += sizeof(values))
      transport.upload({input.buffer, input.offset + offset, sizeof(values)},
                       values.data());
    auto occupied = quota.try_reserve_bytes(quota.remaining());
    ASSERT_TRUE(occupied);
    const auto submitted = (*program)->invoke();
    const auto completed = submitted.pending ? (*program)->wait() : submitted;
    EXPECT_EQ(completed.status.code(), hyle::StatusCode::ResourceExhausted)
        << completed.status.message();
    EXPECT_FALSE(completed.pending);
    EXPECT_FALSE((*program)->output(0));
    auto stopped =
        NativeProgram::create(*device, {ProgramKind::ScanInclusive, 1, 3});
    ASSERT_FALSE(stopped);
    EXPECT_EQ(stopped.error().code(), hyle::StatusCode::FailedPrecondition);
    // Stopping the IREE wrapper does not destroy or poison its borrowed queue.
    std::array<int32_t, 3> unchanged{1, 1, 1};
    transport.download({input.buffer, input.offset, sizeof(unchanged)},
                       unchanged.data());
    EXPECT_EQ(unchanged, (std::array<int32_t, 3>{0, 0, 0}));
  }
  EXPECT_EQ(quota.used(), 0u);
}

TEST_F(IreeNative, RetirementIncludesTheNotificationReturning) {
  auto native = GgmlDevice::create();
  ASSERT_TRUE(native);
  auto device = NativeDevice::create(*native);
  ASSERT_TRUE(device);
  auto program =
      NativeProgram::create(*device, {ProgramKind::ScanInclusive, 1, 3});
  ASSERT_TRUE(program);
  Transfer transport(*native);
  const std::array<int32_t, 3> values{1, 2, 3};
  transport.upload(*(*program)->input(0), values.data());
  std::promise<void> release;
  struct Tail {
    std::promise<void> entered;
    std::shared_future<void> release;
  } tail{{}, release.get_future().share()};
  auto entered = tail.entered.get_future();
  auto submitted = (*program)->invoke(
      [](void *data, const hyle::Status &) noexcept {
        auto &tail = *static_cast<Tail *>(data);
        tail.entered.set_value();
        // Deliberately hold a host callback to test lifetime accounting. Never
        // wait on the same invocation or perform a GPU wait from this callback.
        tail.release.wait();
      },
      &tail);
  const auto started = entered.wait_for(std::chrono::seconds(10));
  EXPECT_TRUE(submitted.status.ok());
  EXPECT_EQ(started, std::future_status::ready);
  if (started == std::future_status::ready) {
    EXPECT_TRUE((*program)->poll().pending);
    EXPECT_EQ((*program)->invoke().status.code(),
              hyle::StatusCode::FailedPrecondition);
  }
  release.set_value();
  EXPECT_TRUE((*program)->wait().status.ok());
  EXPECT_FALSE((*program)->poll().pending);
}
} // namespace
