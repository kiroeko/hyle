#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <utility>

#include "hyle/compute/device.h"

namespace {

using hyle::compute::BackendKind;
using hyle::compute::Context;
using hyle::compute::ContextOptions;
using hyle::compute::Device;
using hyle::compute::Queue;
using hyle::mem::AllocatorRef;

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::size_t allocations = 0;
  std::size_t deallocations = 0;
  bool fail = false;
};

void* probe_allocate(void* opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto& probe = *static_cast<AllocatorProbe*>(opaque);
  if (probe.fail) return nullptr;
  void* result = probe.backing.allocate(size, alignment);
  if (result != nullptr) ++probe.allocations;
  return result;
}

void probe_deallocate(void* opaque, void* pointer, std::size_t size,
                      std::size_t alignment) noexcept {
  auto& probe = *static_cast<AllocatorProbe*>(opaque);
  if (pointer != nullptr) ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void* probe_reallocate(void* opaque, void* pointer, std::size_t old_size,
                       std::size_t new_size,
                       std::size_t alignment) noexcept {
  auto& probe = *static_cast<AllocatorProbe*>(opaque);
  if (probe.fail) return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe& probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_deallocate, &probe_reallocate,
                      &probe);
}

TEST(Device, CpuDiscoveryUsesExplicitBufferFillSemantics) {
  auto count = hyle::compute::enumerate_devices();
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 1u);

  std::array<Device, 1> devices{};
  count = hyle::compute::enumerate_devices(devices);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 1u);
  EXPECT_EQ(devices.front(), Device::cpu());

  auto properties = hyle::compute::query_device(devices.front());
  ASSERT_TRUE(properties.has_value());
  EXPECT_EQ(properties->device, Device::cpu());
  EXPECT_EQ(properties->name, "HYLE host CPU");
  EXPECT_TRUE(properties->unified_memory);
  EXPECT_FALSE(properties->asynchronous_queues);
  EXPECT_EQ(hyle::compute::backend_name(BackendKind::kCpu), "cpu");

  auto invalid = hyle::compute::query_device(Device{});
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
}

TEST(Context, ExplicitAllocatorAndRetainedHandlesHaveOneColdAllocation) {
  AllocatorProbe probe;
  Queue queue;
  {
    auto created = Context::create(
        Device::cpu(), ContextOptions{.metadata_allocator = probe_allocator(probe)});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    Context context = std::move(*created);
    EXPECT_EQ(probe.allocations, 1u);
    EXPECT_EQ(context.metadata_allocator().state(), &probe);

    auto queue_result = context.create_queue();
    ASSERT_TRUE(queue_result.has_value());
    queue = std::move(*queue_result);
    EXPECT_TRUE(queue.is_open());
    EXPECT_EQ(queue.device(), Device::cpu());

    Context moved = std::move(context);
    EXPECT_FALSE(context.valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(queue.is_open());
    EXPECT_TRUE(moved.close().ok());
    EXPECT_FALSE(queue.is_open());
    EXPECT_EQ(probe.deallocations, 0u);
  }
  EXPECT_EQ(probe.deallocations, 0u);
  queue = Queue{};
  EXPECT_EQ(probe.deallocations, 1u);
}

TEST(Context, RejectsInvalidDeviceAndReportsMetadataOom) {
  auto invalid = Context::create(Device{});
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());

  AllocatorProbe probe;
  probe.fail = true;
  auto oom = Context::create(
      Device::cpu(), ContextOptions{.metadata_allocator = probe_allocator(probe)});
  ASSERT_FALSE(oom.has_value());
  EXPECT_TRUE(oom.error().IsResourceExhausted());
  EXPECT_EQ(probe.allocations, 0u);
}

TEST(Context, MoveAssignmentClosesTheReplacedRuntime) {
  Context first = Context::create(Device::cpu()).value();
  Queue first_queue = first.create_queue().value();
  Context second = Context::create(Device::cpu()).value();
  Queue second_queue = second.create_queue().value();

  second = std::move(first);
  EXPECT_TRUE(second.is_open());
  EXPECT_TRUE(first_queue.is_open());
  EXPECT_FALSE(second_queue.is_open());
  EXPECT_FALSE(first.valid());  // NOLINT(bugprone-use-after-move)
}

TEST(Queue, CloseIsLocalButContextCloseInvalidatesEveryQueue) {
  Context context = Context::create(Device::cpu()).value();
  Queue first = context.create_queue().value();
  Queue second = context.create_queue().value();

  EXPECT_TRUE(first.close().ok());
  EXPECT_FALSE(first.is_open());
  EXPECT_TRUE(second.is_open());
  EXPECT_TRUE(context.close().ok());
  EXPECT_FALSE(second.is_open());

  auto late = context.create_queue();
  ASSERT_FALSE(late.has_value());
  EXPECT_TRUE(late.error().IsFailedPrecondition());
}

TEST(Queue, InvalidAndMovedFromOperationsAreObservable) {
  Queue invalid;
  EXPECT_FALSE(invalid.is_open());
  EXPECT_TRUE(invalid.close().IsFailedPrecondition());

  Context context = Context::create(Device::cpu()).value();
  Queue original = context.create_queue().value();
  Queue moved = std::move(original);
  EXPECT_FALSE(original.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(moved.is_open());
}

}  // namespace
