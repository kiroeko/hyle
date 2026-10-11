#include <gtest/gtest.h>

#include <array>
#include <string>

#include "apple_copy_internal.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace d = a::detail;
namespace mem = hyle::mem;
struct DrainGuard final { a::Context &context; ~DrainGuard() { context.close().IgnoreError(); } };

TEST(AppleCopyRecording, OriginalGpuStreamBlitThreeStagesAndTwoColdSlotsPreserveBytes) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value();
  RecordProperty("device_name", std::string(context.environment()->device_name));
  auto input = mem::Buffer::create(memory.ref(), 32).value(); auto output = mem::Buffer::create(memory.ref(), 32).value();
  auto first = mem::Buffer::create(memory.ref(), 32).value(); auto second = mem::Buffer::create(memory.ref(), 32).value();
  mem::QuotaAllocator metadata(mem::system_allocator()); DrainGuard drain{context};
  auto one = d::prepare_copy_recording(context, 4, metadata.get()).value();
  auto two = d::prepare_copy_recording(context, 4, metadata.get()).value();
  auto three = d::prepare_copy_recording(context, 4, metadata.get()).value();
  const auto cold_bytes = metadata.used();
  for (unsigned replay = 0; replay < 3; ++replay) {
    std::array<std::byte, 32> bytes, observed;
    for (unsigned index = 0; index < bytes.size(); ++index) bytes[index] = std::byte((index * 7 + replay * 19) & 255u);
    ASSERT_TRUE(input.write(0, bytes).ok());
    const std::array<c::detail::CopyBackendRange, 2> upload{{
        {first.view().subview(0, 16).value(), input.view().subview(0, 16).value()},
        {first.view().subview(16, 16).value(), input.view().subview(16, 16).value()}}};
    const std::array<c::detail::CopyBackendRange, 1> copy{{{second.view(), first.view()}}};
    const std::array<c::detail::CopyBackendRange, 1> download{{{output.view(), second.view()}}};
    ASSERT_TRUE(d::bind_copy_recording(*one, upload).ok()); ASSERT_TRUE(d::bind_copy_recording(*two, copy).ok());
    ASSERT_TRUE(d::bind_copy_recording(*three, download).ok());
    auto x = d::submit_copy_recording(*one).value(), y = d::submit_copy_recording(*two).value(), z = d::submit_copy_recording(*three).value();
    ASSERT_TRUE(z.wait().ok()); ASSERT_TRUE(x.wait().ok()); ASSERT_TRUE(y.wait().ok());
    ASSERT_TRUE(output.read(0, observed).ok()); EXPECT_EQ(observed, bytes);
    EXPECT_EQ(d::copy_recording_report(*one).native_calls, 2u);
    EXPECT_EQ(d::copy_recording_report(*two).native_calls, 1u); EXPECT_EQ(d::copy_recording_report(*three).native_calls, 1u);
    EXPECT_EQ(d::copy_recording_report(*one).phase_completions, 1u);
    EXPECT_EQ(d::copy_recording_report(*one).host_waits, 1u); EXPECT_TRUE(d::copy_recording_report(*one).native_metadata_unknown);
    EXPECT_EQ(metadata.used(), cold_bytes);
  }
  ASSERT_TRUE(d::bind_copy_recording(*one, {}).ok()); auto empty = d::submit_copy_recording(*one).value();
  ASSERT_TRUE(empty.wait().ok()); EXPECT_EQ(d::copy_recording_report(*one).native_calls, 0u);
  EXPECT_EQ(d::copy_recording_report(*one).phase_completions, 0u);
}

TEST(AppleCopyRecording, RepeatedSourceAliasAndNativeTailConstraintsRejectBeforeJobAdmission) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value();
  auto input = mem::Buffer::create(memory.ref(), 32).value(); auto first = mem::Buffer::create(memory.ref(), 32).value();
  auto second = mem::Buffer::create(memory.ref(), 32).value(); ASSERT_TRUE(input.fill(0, 32, std::byte{49}).ok());
  auto slot = d::prepare_copy_recording(context, 2).value();
  const std::array<c::detail::CopyBackendRange, 2> copies{{{first.view(), input.view()}, {second.view(), input.view()}}};
  ASSERT_TRUE(d::bind_copy_recording(*slot, copies).ok()); auto submitted = d::submit_copy_recording(*slot).value();
  ASSERT_TRUE(submitted.wait().ok()); std::array<std::byte, 32> a, b;
  ASSERT_TRUE(first.read(0, a).ok()); ASSERT_TRUE(second.read(0, b).ok()); EXPECT_EQ(a, b);
  for (auto byte : a) EXPECT_EQ(byte, std::byte{49});
  const std::array<c::detail::CopyBackendRange, 2> overlap{{
      {first.view().subview(0, 16).value(), input.view().subview(0, 16).value()},
      {first.view().subview(8, 16).value(), input.view().subview(16, 16).value()}}};
  EXPECT_FALSE(d::bind_copy_recording(*slot, overlap).ok()); EXPECT_FALSE(d::submit_copy_recording(*slot).has_value());
  const std::array<c::detail::CopyBackendRange, 1> unaligned{{{first.view().subview(0, 8).value(), input.view().subview(2, 8).value()}}};
  EXPECT_FALSE(d::bind_copy_recording(*slot, unaligned).ok());
  const std::array<c::detail::CopyBackendRange, 1> tail{{{first.view().subview(0, 2).value(), input.view().subview(0, 2).value()}}};
  EXPECT_FALSE(d::bind_copy_recording(*slot, tail).ok());
  const std::array<c::detail::CopyBackendRange, 1> self{{{first.view().subview(2, 8).value(), first.view().subview(2, 8).value()}}};
  ASSERT_TRUE(d::bind_copy_recording(*slot, self).ok()); submitted = d::submit_copy_recording(*slot).value();
  ASSERT_TRUE(submitted.wait().ok()); EXPECT_EQ(d::copy_recording_report(*slot).no_op_copies, 1u);
  EXPECT_EQ(d::copy_recording_report(*slot).native_calls, 0u);
}

TEST(AppleCopyRecording, ReadonlyImportedNativeViewKeepsOriginalPhysicalIdentity) {
  auto context = a::Context::create().value(); auto memory = a::MemoryResource::create(context).value();
  auto writer = c::LeasePool::create(memory.ref(), 512).value();
  auto block = writer.allocate(32).value(); ASSERT_TRUE(block.view()->fill(std::byte{71}).ok());
  auto ready = block.publish().value(), slice = ready.subview(4, 16).value();
  auto description = c::TensorDesc::dense(c::DataType::kUInt8, std::array<std::int64_t, 1>{16}).value();
  auto native = memory.binding(slice, description).value(); auto imported = memory.import_readonly(native, description).value();
  auto output = mem::Buffer::create(memory.ref(), 16).value(); auto slot = d::prepare_copy_recording(context, 1).value();
  const std::array<c::detail::CopyBackendRange, 1> copies{{{output.view(), imported.view().value()}}};
  ASSERT_TRUE(d::bind_copy_recording(*slot, copies).ok()); auto completed = d::submit_copy_recording(*slot).value();
  ASSERT_TRUE(completed.wait().ok()); std::array<std::byte, 16> bytes; ASSERT_TRUE(output.read(0, bytes).ok());
  for (auto byte : bytes) EXPECT_EQ(byte, std::byte{71});
}
} // namespace
