#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/backend_extension.h"
#include "hyle/compute/detail/copy_backend.h"
#include "../compute/context_state.h"
#include "copy_fault_test_support.h"

namespace {
using namespace hyle::compute;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;

// Native ABI/byte-route test only. F2's public lease controller, partial
// acceptance and mathematical consumption have separate qualification gates.
struct NativeBatch final {
  Queue *queue = nullptr;
  detail::BackendExtensionRef extension;
  const detail::CopyBackendVTable *table = nullptr;
  void *program = nullptr, *slot = nullptr;
  std::size_t program_bytes = 0, slot_bytes = 0;
  std::uint64_t calls = 0;
  ~NativeBatch() {
    if (queue) EXPECT_TRUE(queue->drain().ok());
    if (slot) table->destroy_slot(slot, mem::system_allocator());
    if (program) table->destroy_program(program, mem::system_allocator());
  }
  hyle::Status create(const Context &context, Queue &selected) noexcept {
    queue = &selected;
    HYLE_ASSIGN_OR_RETURN(extension, detail::BackendExtensionAccess::query(context, detail::kCopyBatchExtensionId));
    return detail::BackendExtensionAccess::call(extension, [](void *context, const void *interface, void *opaque) noexcept -> hyle::Status {
      auto &self = *static_cast<NativeBatch *>(opaque);
      self.table = static_cast<const detail::CopyBackendVTable *>(interface);
      HYLE_RETURN_IF_ERROR(self.table->prepare(context, mem::system_allocator(), &self.program, &self.program_bytes));
      return self.table->create_slot(context, detail::ContextAccess::backend_queue(*self.queue), self.program,
          4, 4, mem::system_allocator(), &self.slot, &self.slot_bytes);
    }, this);
  }
  hyle::Status bind(std::span<const detail::CopyBackendRange> ranges) noexcept {
    struct Call { NativeBatch *self; std::span<const detail::CopyBackendRange> ranges; } call{this, ranges};
    return detail::BackendExtensionAccess::call(extension, [](void *context, const void *, void *opaque) noexcept {
      auto &call = *static_cast<Call *>(opaque);
      return call.self->table->bind_slot(context, call.self->program, call.self->slot, call.ranges);
    }, &call);
  }
  hyle::Result<Completion> submit(std::size_t first, std::size_t count) noexcept {
    struct Call { NativeBatch *self; std::size_t first, count; } call{this, first, count};
    return detail::BackendExtensionAccess::submit(extension, *queue,
        [](void *context, const void *, void *queue, void *opaque) noexcept {
      auto &call = *static_cast<Call *>(opaque);
      return call.self->table->enqueue(context, queue, call.self->program, call.self->slot,
          call.first, call.count, &call.self->calls);
    }, &call);
  }
};

TEST(CudaCopyBatch, BidirectionalSameDeviceByteRoutesAndColdSlotReplay) {
  auto context = Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(4).ok());
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  auto device = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto input = mem::Buffer::create(host.ref(), 32).value();
  auto output = mem::Buffer::create(host.ref(), 32).value();
  auto first = mem::Buffer::create(device.ref(), 32).value();
  auto second = mem::Buffer::create(device.ref(), 32).value();
  NativeBatch batch;
  ASSERT_TRUE(batch.create(context, queue).ok());
  EXPECT_GT(batch.program_bytes, 0u); EXPECT_GT(batch.slot_bytes, 0u);
  for (unsigned replay = 0; replay < 3; ++replay) {
    std::array<std::byte, 32> bytes, observed;
    for (unsigned index = 0; index < bytes.size(); ++index)
      bytes[index] = std::byte((index * 7 + replay * 19) & 255u);
    ASSERT_TRUE(input.write(0, bytes).ok());
    const std::array<detail::CopyBackendRange, 4> ranges{{
        {first.view().subview(0, 16).value(), input.view().subview(0, 16).value()},
        {first.view().subview(16, 16).value(), input.view().subview(16, 16).value()},
        {second.view(), first.view()}, {output.view(), second.view()}}};
    ASSERT_TRUE(batch.bind(ranges).ok());
    const auto before = batch.calls;
    auto upload = batch.submit(0, 2).value();
    auto resident = batch.submit(2, 1).value();
    auto download = batch.submit(3, 1).value();
    ASSERT_TRUE(download.wait().ok());
    ASSERT_TRUE(upload.wait().ok()); ASSERT_TRUE(resident.wait().ok());
    ASSERT_TRUE(output.read(0, observed).ok());
    EXPECT_EQ(observed, bytes);
    // Actual native batch is 3 calls; the cold original-memcpy fallback is 4.
    EXPECT_GE(batch.calls - before, 3u); EXPECT_LE(batch.calls - before, 4u);
  }
  RecordProperty("copy_native_calls", std::to_string(batch.calls));
  const auto before = batch.calls;
  ASSERT_TRUE(batch.bind({}).ok());
  ASSERT_TRUE(batch.submit(0, 0)->wait().ok());
  EXPECT_EQ(batch.calls, before);
  EXPECT_FALSE(batch.submit(0, 1).has_value());
}

TEST(CudaCopyBatch, PhysicalIdentityUsesByteAddressAndRejectsForeignContext) {
  auto context = Context::create(cuda::device(0).value()).value();
  auto other = Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value();
  auto resource = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto foreign = cuda::MemoryResource::create(other, cuda::MemoryKind::kDevice).value();
  auto backing = mem::Buffer::create(resource.ref(), 64).value();
  auto foreign_backing = mem::Buffer::create(foreign.ref(), 64).value();
  NativeBatch batch;
  ASSERT_TRUE(batch.create(context, queue).ok());
  struct Call { NativeBatch *self; mem::ConstBufferView left, right, foreign; } call{
      &batch, backing.view(), backing.view().subview(16, 32).value(), foreign_backing.view()};
  ASSERT_TRUE(detail::BackendExtensionAccess::call(batch.extension,
      [](void *context, const void *, void *opaque) noexcept {
    auto &call = *static_cast<Call *>(opaque);
    auto left = call.self->table->physical(context, call.left);
    auto right = call.self->table->physical(context, call.right);
    EXPECT_TRUE(left.has_value()); EXPECT_TRUE(right.has_value());
    if (left && right) {
      EXPECT_EQ(left->address_space, right->address_space);
      EXPECT_EQ(left->backing, right->backing);
      EXPECT_EQ(right->origin, left->origin + 16);
      EXPECT_EQ(right->capacity, 32u);
    }
    EXPECT_FALSE(call.self->table->physical(context, call.foreign).has_value());
    return hyle::Status::Ok();
  }, &call).ok());
  const std::array<detail::CopyBackendRange, 1> invalid{{{backing.view(), foreign_backing.view()}}};
  EXPECT_FALSE(batch.bind(invalid).ok());
  EXPECT_FALSE(batch.submit(0, 1).has_value());
  auto wrong_queue = context.create_queue().value();
  auto wrong_endpoint = detail::BackendExtensionAccess::submit(batch.extension, wrong_queue,
      [](void *context, const void *, void *queue, void *opaque) noexcept {
    auto &batch = *static_cast<NativeBatch *>(opaque);
    return batch.table->enqueue(context, queue, batch.program, batch.slot, 0, 0, &batch.calls);
  }, &batch);
  EXPECT_FALSE(wrong_endpoint.has_value());
  EXPECT_EQ(batch.calls, 0u);
}
#if defined(HYLE_TESTING)
TEST(CudaCopyBatch, ColdAbsentBatchUsesRealSameStreamCopiesWithoutReprobing) {
  auto context = Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value();
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  auto device = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto source = mem::Buffer::create(host.ref(), 32).value();
  auto first = mem::Buffer::create(device.ref(), 32).value();
  auto second = mem::Buffer::create(device.ref(), 32).value();
  auto output = mem::Buffer::create(host.ref(), 32).value();
  copy_fault_test::NativeScope decorated;
  NativeBatch batch;
  ASSERT_TRUE(batch.create(context, queue).ok());
  for (unsigned replay = 0; replay < 3; ++replay) {
    std::array<std::byte, 32> bytes, observed;
    bytes.fill(std::byte(29 + replay));
    ASSERT_TRUE(source.write(0, bytes).ok());
    const std::array<detail::CopyBackendRange, 4> ranges{{
        {first.view().subview(0, 16).value(), source.view().subview(0, 16).value()},
        {first.view().subview(16, 16).value(), source.view().subview(16, 16).value()},
        {second.view(), first.view()}, {output.view(), second.view()}}};
    ASSERT_TRUE(batch.bind(ranges).ok());
    auto upload = batch.submit(0, 2).value(), resident = batch.submit(2, 1).value(), readback = batch.submit(3, 1).value();
    ASSERT_TRUE(readback.wait().ok()); ASSERT_TRUE(upload.wait().ok()); ASSERT_TRUE(resident.wait().ok());
    ASSERT_TRUE(output.read(0, observed).ok()); EXPECT_EQ(observed, bytes);
    EXPECT_EQ(copy_fault_test::probes, 1u);
    EXPECT_EQ(batch.calls, 4u * (replay + 1));
    EXPECT_EQ(copy_fault_test::copies, batch.calls);
  }
  RecordProperty("native_fallback_copies", copy_fault_test::copies);
  RecordProperty("cold_proc_probes", copy_fault_test::probes);
}
#endif
} // namespace
