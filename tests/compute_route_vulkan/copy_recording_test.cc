#include <gtest/gtest.h>

#include <array>
#include <string>

#include "fixture.h"
#include "vulkan_copy_internal.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;
namespace mem = hyle::mem;
struct DrainGuard final {
  v::Context &context;
  ~DrainGuard() { context.close().IgnoreError(); }
};

TEST(VulkanCopyRecording, ThreeNativeBytePhasesAndPairRegionsReuseColdMetadata) {
  auto context = vulkan_test::context().value();
  ASSERT_FALSE(context.environment().software_device);
  RecordProperty("device_name", context.environment().device_name);
  RecordProperty("driver_name", context.environment().driver_name);
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto input = mem::Buffer::create(staging.ref(), 32).value();
  auto output = mem::Buffer::create(staging.ref(), 32).value();
  auto first = mem::Buffer::create(device.ref(), 32).value();
  auto second = mem::Buffer::create(device.ref(), 32).value();
  mem::QuotaAllocator metadata(mem::system_allocator());
  DrainGuard drain{context}; // native Run may retain this allocator after an assertion exits
  auto upload = d::prepare_copy_recording(context, 4, metadata.get()).value();
  auto resident = d::prepare_copy_recording(context, 4, metadata.get()).value();
  auto download = d::prepare_copy_recording(context, 4, metadata.get()).value();
  const auto cold_bytes = metadata.used();
  for (unsigned replay = 0; replay < 3; ++replay) {
    std::array<std::byte, 32> bytes, observed;
    for (unsigned index = 0; index < bytes.size(); ++index)
      bytes[index] = std::byte((index * 7 + replay * 19) & 255u);
    ASSERT_TRUE(input.write(0, bytes).ok());
    const std::array<c::detail::CopyBackendRange, 2> to_device{{
        {first.view().subview(0, 16).value(), input.view().subview(0, 16).value()},
        {first.view().subview(16, 16).value(), input.view().subview(16, 16).value()}}};
    const std::array<c::detail::CopyBackendRange, 1> same_device{{{second.view(), first.view()}}};
    const std::array<c::detail::CopyBackendRange, 1> to_host{{{output.view(), second.view()}}};
    ASSERT_TRUE(d::bind_copy_recording(*upload, to_device, true).ok());
    ASSERT_TRUE(d::bind_copy_recording(*resident, same_device).ok());
    ASSERT_TRUE(d::bind_copy_recording(*download, to_host).ok());
    EXPECT_EQ(d::copy_recording_report(*upload).native_calls, 1u);
    EXPECT_EQ(d::copy_recording_report(*upload).active_copies, 2u);
    auto a = d::submit_copy_recording(*upload).value();
    auto b = d::submit_copy_recording(*resident).value();
    auto completed = d::submit_copy_recording(*download).value();
    ASSERT_TRUE(completed.wait().ok()); ASSERT_TRUE(a.wait().ok()); ASSERT_TRUE(b.wait().ok());
    ASSERT_TRUE(output.read(0, observed).ok()); EXPECT_EQ(observed, bytes);
    EXPECT_EQ(d::copy_recording_report(*upload).phase_completions, 1u);
    EXPECT_EQ(d::copy_recording_report(*resident).native_calls, 1u);
    EXPECT_EQ(d::copy_recording_report(*download).native_calls, 1u);
    EXPECT_EQ(metadata.used(), cold_bytes);
  }
  ASSERT_TRUE(d::bind_copy_recording(*upload, {}).ok());
  auto empty = d::submit_copy_recording(*upload).value(); ASSERT_TRUE(empty.wait().ok());
  EXPECT_EQ(d::copy_recording_report(*upload).native_calls, 0u);
  EXPECT_EQ(d::copy_recording_report(*upload).phase_completions, 0u);
}

TEST(VulkanCopyRecording, IndependentBufferPairsShareOneNativePhaseAndSourceCanRepeat) {
  auto context = vulkan_test::context().value(); ASSERT_FALSE(context.environment().software_device);
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto input = mem::Buffer::create(staging.ref(), 32).value(); ASSERT_TRUE(input.fill(0, 32, std::byte{51}).ok());
  auto first = mem::Buffer::create(device.ref(), 32).value();
  auto second = mem::Buffer::create(device.ref(), 32).value();
  auto first_read = mem::Buffer::create(staging.ref(), 32).value();
  auto second_read = mem::Buffer::create(staging.ref(), 32).value();
  auto batch = d::prepare_copy_recording(context, 4).value();
  const std::array<c::detail::CopyBackendRange, 2> copies{{{first.view(), input.view()}, {second.view(), input.view()}}};
  ASSERT_TRUE(d::bind_copy_recording(*batch, copies, true).ok());
  EXPECT_EQ(d::copy_recording_report(*batch).native_calls, 2u);
  auto submitted = d::submit_copy_recording(*batch).value(); ASSERT_TRUE(submitted.wait().ok());
  EXPECT_EQ(d::copy_recording_report(*batch).phase_completions, 1u);
  const std::array<c::detail::CopyBackendRange, 2> reads{{{first_read.view(), first.view()}, {second_read.view(), second.view()}}};
  ASSERT_TRUE(d::bind_copy_recording(*batch, reads).ok());
  submitted = d::submit_copy_recording(*batch).value(); ASSERT_TRUE(submitted.wait().ok());
  std::array<std::byte, 32> a, b;
  ASSERT_TRUE(first_read.read(0, a).ok()); ASSERT_TRUE(second_read.read(0, b).ok());
  for (auto byte : a) EXPECT_EQ(byte, std::byte{51}); EXPECT_EQ(a, b);
}

TEST(VulkanCopyRecording, AliasAlignmentForeignBackingAndSelfAreCheckedBeforeNativeAcceptance) {
  auto context = vulkan_test::context().value(); ASSERT_FALSE(context.environment().software_device);
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto input = mem::Buffer::create(staging.ref(), 32).value();
  auto output = mem::Buffer::create(device.ref(), 32).value();
  auto batch = d::prepare_copy_recording(context, 2).value();
  const std::array<c::detail::CopyBackendRange, 2> overlap{{
      {output.view().subview(0, 16).value(), input.view().subview(0, 16).value()},
      {output.view().subview(8, 16).value(), input.view().subview(16, 16).value()}}};
  EXPECT_FALSE(d::bind_copy_recording(*batch, overlap).ok()); EXPECT_FALSE(d::submit_copy_recording(*batch).has_value());
  const std::array<c::detail::CopyBackendRange, 1> unaligned{{
      {output.view().subview(0, 8).value(), input.view().subview(2, 8).value()}}};
  EXPECT_FALSE(d::bind_copy_recording(*batch, unaligned).ok());
  const std::array<c::detail::CopyBackendRange, 1> self{{
      {output.view().subview(2, 8).value(), output.view().subview(2, 8).value()}}};
  ASSERT_TRUE(d::bind_copy_recording(*batch, self).ok());
  auto completed = d::submit_copy_recording(*batch).value(); ASSERT_TRUE(completed.wait().ok());
  EXPECT_EQ(d::copy_recording_report(*batch).no_op_copies, 1u);
  EXPECT_EQ(d::copy_recording_report(*batch).native_calls, 0u);
  auto foreign_context = vulkan_test::context().value();
  auto foreign = v::MemoryResource::create(foreign_context, v::MemoryKind::kDevice).value();
  auto foreign_bytes = mem::Buffer::create(foreign.ref(), 32).value();
  const std::array<c::detail::CopyBackendRange, 1> wrong_device{{{foreign_bytes.view(), input.view()}}};
  EXPECT_FALSE(d::bind_copy_recording(*batch, wrong_device).ok());
  EXPECT_FALSE(d::submit_copy_recording(*batch).has_value());
}
} // namespace
