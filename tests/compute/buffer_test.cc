#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/compute/device.h"
#include "hyle/compute/memory.h"
#include "hyle/mem/buffer.h"

using namespace hyle;
using namespace hyle::compute;

namespace {

Device gpu_or_host(bool* is_gpu) {
  Device devices[8];
  const std::size_t count = enumerate(devices, 8);
  for (std::size_t i = 0; i < count && i < 8; ++i) {
    if (devices[i].is_gpu()) {
      *is_gpu = true;
      return devices[i];
    }
  }
  *is_gpu = false;
  return default_device();
}

template <class T>
std::span<const std::byte> bytes_of(const std::vector<T>& values) {
  return std::as_bytes(std::span(values));
}

template <class T>
std::span<std::byte> writable_bytes_of(std::vector<T>& values) {
  return std::as_writable_bytes(std::span(values));
}

}  // namespace

TEST(ComputeMemory, ResourcePropertiesAndIdentity) {
  bool is_gpu = false;
  Device device = gpu_or_host(&is_gpu);
  auto device_resource = device_memory_resource(device);
  auto host_resource = host_memory_resource(device);
  ASSERT_TRUE(device_resource.has_value());
  ASSERT_TRUE(host_resource.has_value());

  const mem::MemoryProperties device_properties = device_resource->properties();
  const mem::MemoryProperties host_properties = host_resource->properties();
  EXPECT_TRUE(device_properties.has(mem::MemoryProperty::kDeviceAccessible));
  EXPECT_TRUE(host_properties.has(mem::MemoryProperty::kHostAccessible));
  EXPECT_TRUE(host_properties.has(mem::MemoryProperty::kDeviceAccessible));
  EXPECT_EQ(device_properties.has(mem::MemoryProperty::kHostAccessible), !is_gpu);
  EXPECT_EQ(host_properties.has(mem::MemoryProperty::kHostPinned),
            device.caps().host_buffer);
  EXPECT_EQ(device_properties.domain, host_properties.domain);
  EXPECT_TRUE(memory_resource_belongs_to(*device_resource, device));
  EXPECT_TRUE(memory_resource_belongs_to(*host_resource, device));
  EXPECT_FALSE(memory_resource_belongs_to(mem::system_resource(), device));
  EXPECT_FALSE(memory_resource_belongs_to(*device_resource, Device{}));
}

TEST(ComputeMemory, InvalidDeviceReturnsInvalidArgument) {
  EXPECT_TRUE(device_memory_resource(Device{}).error().IsInvalidArgument());
  EXPECT_TRUE(host_memory_resource(Device{}).error().IsInvalidArgument());
  EXPECT_TRUE(managed_memory_resource(Device{}).error().IsInvalidArgument());
}

TEST(ComputeMemory, TracksOwningBackendDevice) {
  Device owner = default_device();
  auto resource = device_memory_resource(owner).value();
  auto buffer = mem::Buffer::create(resource, 64);
  ASSERT_TRUE(buffer.has_value());

  Device devices[8];
  const std::size_t count = enumerate(devices, 8);
  for (std::size_t i = 0; i < count && i < 8; ++i) {
    EXPECT_EQ(memory_resource_belongs_to(buffer->resource(), devices[i]),
              devices[i].ordinal() == owner.ordinal())
        << devices[i].backend();
  }
}

TEST(ComputeMemory, DeviceReadWriteRoundTrip) {
  Device device = default_device();
  constexpr std::size_t kN = 4096;
  auto buffer = mem::Buffer::create(device_memory_resource(device).value(),
                                    kN * sizeof(std::uint32_t));
  ASSERT_TRUE(buffer.has_value()) << buffer.error().message();

  std::vector<std::uint32_t> input(kN), output(kN, 0);
  for (std::size_t i = 0; i < kN; ++i) {
    input[i] = static_cast<std::uint32_t>(i * 7 + 3);
  }
  ASSERT_TRUE(buffer->view().write(bytes_of(input)).ok());
  ASSERT_TRUE(buffer->view().read(writable_bytes_of(output)).ok());
  EXPECT_EQ(output, input);
}

TEST(ComputeMemory, HostMappingAndNativeCrossTierCopy) {
  Device device = default_device();
  constexpr std::size_t kBytes = 8192;
  auto host = mem::Buffer::create(host_memory_resource(device).value(), kBytes);
  auto local =
      mem::Buffer::create(device_memory_resource(device).value(), kBytes);
  ASSERT_TRUE(host.has_value() && local.has_value());

  auto mapping = host->view().map_host(mem::HostAccess::kReadWrite);
  ASSERT_TRUE(mapping.has_value()) << mapping.error().message();
  auto mapped = mapping->writable_bytes();
  ASSERT_TRUE(mapped.has_value());
  for (std::size_t i = 0; i < kBytes; ++i) {
    (*mapped)[i] = static_cast<std::byte>((i * 17) & 0xff);
  }
  ASSERT_TRUE(mapping->flush().ok());
  ASSERT_TRUE(local->view().copy_from(host->view()).ok());

  std::vector<std::byte> observed(kBytes);
  ASSERT_TRUE(local->view().read(observed).ok());
  EXPECT_TRUE(std::equal(observed.begin(), observed.end(), mapped->begin()));

  ASSERT_TRUE(local->view().fill(std::byte{0xa7}).ok());
  ASSERT_TRUE(host->view().copy_from(local->view()).ok());
  ASSERT_TRUE(mapping->invalidate().ok());
  for (std::byte value : *mapped) EXPECT_EQ(value, std::byte{0xa7});
}

TEST(ComputeMemory, ZeroSizeAllocationUsesTheGenericMemContract) {
  Device device = default_device();
  auto buffer = mem::Buffer::create(device_memory_resource(device).value(), 0);
  ASSERT_TRUE(buffer.has_value());
  EXPECT_TRUE(buffer->valid());
  EXPECT_EQ(buffer->native_handle(), nullptr);
  EXPECT_EQ(buffer->size(), 0u);
  EXPECT_TRUE(buffer->view().write({}).ok());
  EXPECT_TRUE(buffer->view().read({}).ok());
}

TEST(ComputeMemory, FillAtOffsetPreservesSurroundingBytes) {
  Device device = default_device();
  constexpr std::size_t kBytes = 1024;
  auto buffer =
      mem::Buffer::create(device_memory_resource(device).value(), kBytes);
  ASSERT_TRUE(buffer.has_value());
  std::vector<std::byte> initial(kBytes, std::byte{0x11});
  ASSERT_TRUE(buffer->view().write(initial).ok());

  auto middle = buffer->view().subview(256, 512);
  ASSERT_TRUE(middle.has_value());
  ASSERT_TRUE(middle->fill(std::byte{0xa5}).ok());

  std::vector<std::byte> observed(kBytes);
  ASSERT_TRUE(buffer->view().read(observed).ok());
  for (std::size_t i = 0; i < kBytes; ++i) {
    const std::byte expected =
        i >= 256 && i < 768 ? std::byte{0xa5} : std::byte{0x11};
    EXPECT_EQ(observed[i], expected) << i;
  }
}

TEST(ComputeMemory, MoveSemanticsPreserveProviderToken) {
  Device device = default_device();
  auto first = mem::Buffer::create(device_memory_resource(device).value(), 256);
  auto second = mem::Buffer::create(device_memory_resource(device).value(), 128);
  ASSERT_TRUE(first.has_value() && second.has_value());
  void* token = first->native_handle();
  *second = std::move(*first);
  EXPECT_FALSE(first->valid());
  EXPECT_TRUE(second->valid());
  EXPECT_EQ(second->native_handle(), token);
  EXPECT_EQ(second->size(), 256u);
}

TEST(ComputeMemory, KvCacheRepeatedOffsetWrites) {
  Device device = default_device();
  constexpr std::size_t kRowElements = 128;
  constexpr std::size_t kTokens = 512;
  constexpr std::size_t kRowBytes = kRowElements * sizeof(float);
  const std::size_t total = kRowBytes * kTokens;
  auto kv = mem::Buffer::create(device_memory_resource(device).value(), total);
  ASSERT_TRUE(kv.has_value()) << kv.error().message();
  ASSERT_TRUE(kv->view().fill(std::byte{0}).ok());

  std::vector<float> row(kRowElements);
  for (std::size_t token = 0; token < kTokens; ++token) {
    for (std::size_t column = 0; column < kRowElements; ++column) {
      row[column] = static_cast<float>(token * 1000 + column);
    }
    auto destination = kv->view().subview(token * kRowBytes, kRowBytes);
    ASSERT_TRUE(destination.has_value());
    ASSERT_TRUE(destination->write(bytes_of(row)).ok()) << token;
  }

  std::vector<float> observed(kRowElements);
  for (std::size_t token = 0; token < kTokens; ++token) {
    auto source = kv->view().subview(token * kRowBytes, kRowBytes);
    ASSERT_TRUE(source.has_value());
    ASSERT_TRUE(source->read(writable_bytes_of(observed)).ok()) << token;
    for (std::size_t column = 0; column < kRowElements; ++column) {
      EXPECT_FLOAT_EQ(observed[column],
                      static_cast<float>(token * 1000 + column));
    }
  }
}

TEST(ComputeMemory, ManagedProviderIsExplicitlyAvailableOrUnsupported) {
  Device device = default_device();
  auto resource = managed_memory_resource(device);
  if (!resource.has_value()) {
    EXPECT_TRUE(resource.error().IsUnimplemented());
    return;
  }
  const mem::MemoryProperties properties = resource->properties();
  EXPECT_TRUE(properties.has(mem::MemoryProperty::kManaged));
  EXPECT_TRUE(properties.has(mem::MemoryProperty::kHostAccessible));
  EXPECT_TRUE(properties.has(mem::MemoryProperty::kDeviceAccessible));
  auto buffer = mem::Buffer::create(*resource, 4096);
  ASSERT_TRUE(buffer.has_value()) << buffer.error().message();
  EXPECT_TRUE(buffer->view().fill(std::byte{0x2a}).ok());
}
