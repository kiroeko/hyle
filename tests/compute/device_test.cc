#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <utility>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/transfer.h"
#include "hyle/mem/buffer.h"

namespace {

using hyle::compute::Context;
using hyle::compute::ContextOptions;
using hyle::compute::Device;
using hyle::compute::DeviceClass;
using hyle::compute::Queue;
using hyle::mem::AllocatorRef;

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::size_t allocations = 0;
  std::size_t deallocations = 0;
  bool fail = false;
};

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (probe.fail)
    return nullptr;
  void *result = probe.backing.allocate(size, alignment);
  if (result != nullptr)
    ++probe.allocations;
  return result;
}

void probe_deallocate(void *opaque, void *pointer, std::size_t size,
                      std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (pointer != nullptr)
    ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *opaque, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (probe.fail)
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_deallocate, &probe_reallocate,
                      &probe);
}

TEST(Device, CpuDiscoveryIsScopedToTheExplicitBackend) {
  const auto backend = hyle::compute::cpu::backend();
  ASSERT_TRUE(backend.valid());
  EXPECT_EQ(backend.id(), hyle::compute::cpu::kBackendId);
  EXPECT_EQ(backend.name(), "cpu");
  EXPECT_EQ(backend.device_class(), DeviceClass::kCpu);

  auto count = backend.device_count();
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 1u);

  auto discovered = backend.device(0);
  ASSERT_TRUE(discovered.has_value());
  EXPECT_EQ(*discovered, hyle::compute::cpu::device());
  auto out_of_range = backend.device(1);
  ASSERT_FALSE(out_of_range.has_value());
  EXPECT_TRUE(out_of_range.error().IsOutOfRange());

  auto properties = discovered->properties();
  ASSERT_TRUE(properties.has_value());
  EXPECT_EQ(properties->backend, hyle::compute::cpu::kBackendId);
  EXPECT_EQ(properties->device_class, DeviceClass::kCpu);
  EXPECT_EQ(properties->backend_name, "cpu");
  EXPECT_EQ(properties->name, "HYLE host CPU");
  EXPECT_TRUE(properties->unified_memory);
  EXPECT_FALSE(properties->asynchronous_queues);

  auto invalid = Device{}.properties();
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsFailedPrecondition());
}

TEST(Context, ExplicitAllocatorAndRetainedHandlesHaveOneColdAllocation) {
  AllocatorProbe probe;
  Queue queue;
  {
    auto created = Context::create(
        hyle::compute::cpu::device(),
        ContextOptions{.metadata_allocator = probe_allocator(probe)});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    Context context = std::move(*created);
    EXPECT_EQ(probe.allocations, 1u);
    EXPECT_EQ(context.metadata_allocator().state(), &probe);

    auto queue_result = context.create_queue();
    ASSERT_TRUE(queue_result.has_value());
    queue = std::move(*queue_result);
    EXPECT_EQ(probe.allocations, 2u);
    EXPECT_TRUE(queue.is_open());
    EXPECT_EQ(queue.device(), hyle::compute::cpu::device());

    Context moved = std::move(context);
    EXPECT_FALSE(context.valid()); // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(queue.is_open());
    EXPECT_TRUE(moved.close().ok());
    EXPECT_FALSE(queue.is_open());
    EXPECT_EQ(probe.deallocations, 0u);
  }
  EXPECT_EQ(probe.deallocations, 0u);
  queue = Queue{};
  EXPECT_EQ(probe.deallocations, 2u);
}

TEST(Context, RejectsInvalidDeviceAndReportsMetadataOom) {
  auto invalid = Context::create(Device{});
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());

  AllocatorProbe probe;
  probe.fail = true;
  auto oom = Context::create(
      hyle::compute::cpu::device(),
      ContextOptions{.metadata_allocator = probe_allocator(probe)});
  ASSERT_FALSE(oom.has_value());
  EXPECT_TRUE(oom.error().IsResourceExhausted());
  EXPECT_EQ(probe.allocations, 0u);
}

TEST(Context, MoveAssignmentClosesTheReplacedRuntime) {
  Context first = Context::create(hyle::compute::cpu::device()).value();
  Queue first_queue = first.create_queue().value();
  Context second = Context::create(hyle::compute::cpu::device()).value();
  Queue second_queue = second.create_queue().value();

  second = std::move(first);
  EXPECT_TRUE(second.is_open());
  EXPECT_TRUE(first_queue.is_open());
  EXPECT_FALSE(second_queue.is_open());
  EXPECT_FALSE(first.valid()); // NOLINT(bugprone-use-after-move)
}

TEST(Queue, CloseIsLocalButContextCloseInvalidatesEveryQueue) {
  Context context = Context::create(hyle::compute::cpu::device()).value();
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

  Context context = Context::create(hyle::compute::cpu::device()).value();
  Queue original = context.create_queue().value();
  Queue moved = std::move(original);
  EXPECT_FALSE(original.valid()); // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(moved.is_open());
}

TEST(Queue, CpuAsyncByteOperationsUseTheOrderedCompletionContract) {
  Context context = Context::create(hyle::compute::cpu::device()).value();
  Queue queue = context.create_queue().value();
  auto source = hyle::mem::Buffer::create(hyle::mem::system_resource(), 8)
                    .value();
  auto destination = hyle::mem::Buffer::create(hyle::mem::system_resource(), 8)
                         .value();
  const std::array<std::byte, 8> bytes{
      std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4},
      std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
  ASSERT_TRUE(source.view().write(bytes).ok());

  auto properties = hyle::compute::transfer_properties(
      queue, destination.view(), source.view());
  ASSERT_TRUE(properties.has_value()) << properties.error().message();
  EXPECT_EQ(properties->mode, hyle::compute::TransferMode::kImmediate);
  EXPECT_FALSE(properties->peer);

  auto copied = hyle::compute::copy_async(queue, destination.view(),
                                          source.view());
  ASSERT_TRUE(copied.has_value()) << copied.error().message();
  EXPECT_TRUE(copied->wait().ok());
  std::array<std::byte, 8> observed{};
  EXPECT_TRUE(destination.view().read(observed).ok());
  EXPECT_EQ(observed, bytes);

  auto filled = hyle::compute::fill_async(queue, destination.view(),
                                          std::byte{0x5a});
  ASSERT_TRUE(filled.has_value()) << filled.error().message();
  EXPECT_TRUE(filled->wait().ok());
  EXPECT_TRUE(destination.view().read(observed).ok());
  for (std::byte value : observed)
    EXPECT_EQ(value, std::byte{0x5a});

  auto short_destination = destination.view().subview(0, 4).value();
  EXPECT_TRUE(hyle::compute::copy_async(queue, short_destination, source.view())
                  .error()
                  .IsInvalidArgument());
  auto overlapping_source = source.view().subview(0, 6).value();
  auto overlapping_destination = source.view().subview(2, 6).value();
  EXPECT_TRUE(hyle::compute::copy_async(queue, overlapping_destination,
                                        overlapping_source)
                  .error()
                  .IsInvalidArgument());
}

} // namespace
