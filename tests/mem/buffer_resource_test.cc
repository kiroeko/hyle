#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {

using hyle::Result;
using hyle::Status;
using namespace hyle::mem;

class TestMappedResource final : public MemoryResource {
 public:
  std::string_view name() const noexcept override { return "test.mapped"; }
  MemoryProperties properties() const noexcept override {
    return MemoryProperties{
        .flags = MemoryProperty::kHostAccessible,
        .minimum_alignment = 16,
        .maximum_allocation = storage_.size(),
        .domain = MemoryDomainId{this, 7},
    };
  }

  Result<ResourceAllocation> allocate(std::size_t size,
                                      std::size_t alignment) noexcept override {
    if (live_ || size > storage_.size() || alignment > 64) {
      return std::unexpected(Status::ResourceExhausted());
    }
    live_ = true;
    ++allocations_;
    return ResourceAllocation{size == 0 ? nullptr : storage_.data(), size,
                              alignment};
  }

  void deallocate(ResourceAllocation allocation) noexcept override {
    EXPECT_TRUE(live_);
    EXPECT_EQ(allocation.native_handle,
              allocation.size == 0 ? nullptr : storage_.data());
    live_ = false;
    ++deallocations_;
  }

  Result<ResourceHostMapping> map_host(
      void* native_handle, std::size_t offset, std::size_t size,
      HostAccess) noexcept override {
    ++maps_;
    if (return_null_mapping_) return ResourceHostMapping{nullptr, this};
    return ResourceHostMapping{
        size == 0 ? nullptr
                  : static_cast<void*>(
                        static_cast<std::byte*>(native_handle) + offset),
        nullptr};
  }

  Status unmap_host(void*, void*, std::size_t, std::size_t,
                    HostAccess) noexcept override {
    ++unmaps_;
    return Status::Ok();
  }

  Status flush(void*, std::size_t, std::size_t) noexcept override {
    ++flushes_;
    return Status::Ok();
  }
  Status invalidate(void*, std::size_t, std::size_t) noexcept override {
    ++invalidates_;
    return Status::Ok();
  }

  int allocations() const noexcept { return allocations_; }
  int deallocations() const noexcept { return deallocations_; }
  int maps() const noexcept { return maps_; }
  int unmaps() const noexcept { return unmaps_; }
  int flushes() const noexcept { return flushes_; }
  int invalidates() const noexcept { return invalidates_; }
  void return_null_mapping(bool value) noexcept {
    return_null_mapping_ = value;
  }

 private:
  alignas(64) std::array<std::byte, 256> storage_{};
  bool live_ = false;
  int allocations_ = 0;
  int deallocations_ = 0;
  int maps_ = 0;
  int unmaps_ = 0;
  int flushes_ = 0;
  int invalidates_ = 0;
  bool return_null_mapping_ = false;
};

TEST(MemoryResource, SystemPropertiesAndIdentityAreStable) {
  MemoryResourceRef first = system_resource();
  MemoryResourceRef second = system_resource();
  ASSERT_TRUE(first.valid());
  EXPECT_EQ(first, second);
  EXPECT_EQ(first.name(), "hyle.system");
  const MemoryProperties properties = first.properties();
  EXPECT_TRUE(properties.has(MemoryProperty::kHostAccessible));
  EXPECT_TRUE(properties.has(MemoryProperty::kHostDeviceCoherent));
  EXPECT_FALSE(properties.has(MemoryProperty::kDeviceAccessible));
  EXPECT_EQ(first.domain(), second.domain());
}

TEST(Buffer, SystemResourceSupportsOwnedRangeOperations) {
  auto created = Buffer::create(system_resource(), 64, 64);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  Buffer buffer = std::move(*created);
  ASSERT_TRUE(buffer.valid());
  EXPECT_EQ(buffer.size(), 64u);
  EXPECT_EQ(buffer.alignment(), 64u);

  std::array<std::byte, 16> input{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::byte>(index + 1);
  }
  ASSERT_TRUE(buffer.write(8, input).ok());

  std::array<std::byte, 16> output{};
  ASSERT_TRUE(buffer.read(8, output).ok());
  EXPECT_EQ(output, input);

  auto middle = buffer.view().subview(8, input.size());
  ASSERT_TRUE(middle.has_value());
  ASSERT_TRUE(middle->fill(std::byte{0x5a}).ok());
  ASSERT_TRUE(buffer.read(8, output).ok());
  for (std::byte value : output) EXPECT_EQ(value, std::byte{0x5a});

  auto mapped = middle->map_host();
  ASSERT_TRUE(mapped.has_value());
  auto writable = mapped->writable_bytes();
  ASSERT_TRUE(writable.has_value());
  (*writable)[0] = std::byte{0x33};
  EXPECT_TRUE(mapped->flush().ok());
  EXPECT_TRUE(mapped->close().ok());
  ASSERT_TRUE(buffer.read(8, output).ok());
  EXPECT_EQ(output[0], std::byte{0x33});
}

TEST(Buffer, ViewsRemainBoundToAllocationAcrossOwnerMove) {
  Buffer original = Buffer::create(system_resource(), 32).value();
  BufferView before_move = original.view();
  Buffer moved = std::move(original);
  EXPECT_FALSE(original.valid());
  ASSERT_TRUE(moved.valid());
  EXPECT_TRUE(before_move.fill(std::byte{0x6d}).ok());

  std::array<std::byte, 32> bytes{};
  EXPECT_TRUE(moved.read(0, bytes).ok());
  for (std::byte value : bytes) EXPECT_EQ(value, std::byte{0x6d});
}

TEST(Buffer, RejectsInvalidResourceAlignmentAndRanges) {
  auto invalid_resource = Buffer::create({}, 16);
  ASSERT_FALSE(invalid_resource.has_value());
  EXPECT_TRUE(invalid_resource.error().IsInvalidArgument());

  auto invalid_alignment = Buffer::create(system_resource(), 16, 3);
  ASSERT_FALSE(invalid_alignment.has_value());
  EXPECT_TRUE(invalid_alignment.error().IsInvalidArgument());

  Buffer buffer = Buffer::create(system_resource(), 16).value();
  auto outside = buffer.view().subview(17);
  ASSERT_FALSE(outside.has_value());
  EXPECT_TRUE(outside.error().IsOutOfRange());

  std::array<std::byte, 15> short_read{};
  Status status = buffer.view().read(short_read);
  EXPECT_TRUE(status.IsInvalidArgument());
}

TEST(Buffer, ReadOnlyHostMappingDoesNotExposeMutableBytes) {
  Buffer buffer = Buffer::create(system_resource(), 8).value();
  auto mapping = static_cast<const Buffer&>(buffer).view().map_host();
  ASSERT_TRUE(mapping.has_value());
  auto writable = mapping->writable_bytes();
  ASSERT_FALSE(writable.has_value());
  EXPECT_TRUE(writable.error().IsPermissionDenied());
}

TEST(Buffer, BaseProviderPathHonorsCoherenceAndCrossResourceCopy) {
  TestMappedResource source_resource;
  TestMappedResource destination_resource;
  {
    Buffer source =
        Buffer::create(MemoryResourceRef(source_resource), 32).value();
    Buffer destination =
        Buffer::create(MemoryResourceRef(destination_resource), 32).value();

    std::array<std::byte, 32> input{};
    input.fill(std::byte{0x41});
    ASSERT_TRUE(source.view().write(input).ok());
    EXPECT_EQ(source_resource.flushes(), 1);

    ASSERT_TRUE(destination.view().copy_from(source.view()).ok());
    EXPECT_GE(source_resource.invalidates(), 1);
    EXPECT_GE(destination_resource.flushes(), 1);

    std::array<std::byte, 32> output{};
    ASSERT_TRUE(destination.view().read(output).ok());
    EXPECT_EQ(output, input);
    EXPECT_GE(destination_resource.invalidates(), 1);
    EXPECT_GE(source_resource.maps(), 2);
    EXPECT_GE(destination_resource.maps(), 2);
  }
  EXPECT_EQ(source_resource.allocations(), 1);
  EXPECT_EQ(source_resource.deallocations(), 1);
  EXPECT_EQ(destination_resource.allocations(), 1);
  EXPECT_EQ(destination_resource.deallocations(), 1);
}

TEST(Buffer, ZeroLengthAllocationIsOwnedWithoutFabricatingAHandle) {
  Buffer empty = Buffer::create(system_resource(), 0).value();
  EXPECT_TRUE(empty.valid());
  EXPECT_EQ(empty.size(), 0u);
  EXPECT_EQ(empty.native_handle(), nullptr);
  EXPECT_TRUE(empty.view().fill(std::byte{0}).ok());
  auto mapping = empty.view().map_host();
  ASSERT_TRUE(mapping.has_value());
  EXPECT_TRUE(mapping->bytes().empty());
}

TEST(Buffer, FailedProviderMappingsAreStillUnmapped) {
  TestMappedResource source_resource;
  TestMappedResource destination_resource;
  Buffer source =
      Buffer::create(MemoryResourceRef(source_resource), 16).value();
  Buffer destination =
      Buffer::create(MemoryResourceRef(destination_resource), 16).value();

  source_resource.return_null_mapping(true);
  Status status = destination.view().copy_from(source.view());
  EXPECT_TRUE(status.IsInternal());
  EXPECT_EQ(source_resource.maps(), 1);
  EXPECT_EQ(source_resource.unmaps(), 1);
  EXPECT_EQ(destination_resource.maps(), 0);
}

}  // namespace
