#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hyle/compute/device.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/memory.h"
#include "hyle/compute/stream.h"
#include "hyle/mem/buffer.h"

using namespace hyle;
using namespace hyle::compute;

#if defined(_WIN32)
#  define HYLE_THREADSAFE_DEATH() ((void)0)
#else
#  define HYLE_THREADSAFE_DEATH() GTEST_FLAG_SET(death_test_style, "threadsafe")
#endif

namespace {

bool find_two_backends(Device* first, Device* second) {
  Device devices[8];
  const std::size_t count = enumerate(devices, 8);
  for (std::size_t i = 0; i < count && i < 8; ++i) {
    for (std::size_t j = i + 1; j < count && j < 8; ++j) {
      if (std::strcmp(devices[i].backend(), devices[j].backend()) != 0) {
        *first = devices[i];
        *second = devices[j];
        return true;
      }
    }
  }
  return false;
}

Device host_device() {
  Device devices[8];
  const std::size_t count = enumerate(devices, 8);
  for (std::size_t i = 0; i < count && i < 8; ++i) {
    if (devices[i].type() == DeviceType::kCpu) return devices[i];
  }
  return default_device();
}

void noop_host_kernel(const LaunchConfig&, const KernelArgs&) noexcept {}

void launch_batch_over_cap() {
  auto stream = Stream::create(default_device());
  BatchItem items[1] = {};
  (void)launch_batch(*stream, items, 65);
}

void launch_batch_null_items() {
  auto stream = Stream::create(default_device());
  (void)launch_batch(*stream, nullptr, 1);
}

void record_cross_backend() {
  Device first;
  Device second;
  find_two_backends(&first, &second);
  auto stream = Stream::create(first);
  auto event = Event::create(second);
  stream->record(*event);
}

void launch_cross_backend() {
  Device host = host_device();
  Device first;
  Device second;
  find_two_backends(&first, &second);
  Device other = std::strcmp(first.backend(), host.backend()) == 0 ? second
                                                                   : first;
  auto kernel = Kernel::load_host(host, &noop_host_kernel);
  auto stream = Stream::create(other);
  KernelArgs empty{nullptr, 0, nullptr, 0, nullptr, 0};
  (void)launch(*stream, *kernel, LaunchConfig{}, empty);
}

}  // namespace

TEST(ComputeContract, MemViewsReportRangeErrorsInsteadOfAborting) {
  Device device = default_device();
  auto buffer =
      mem::Buffer::create(device_memory_resource(device).value(), 256).value();
  std::byte payload[64]{};
  EXPECT_TRUE(buffer.write(224, payload).IsOutOfRange());
  EXPECT_TRUE(buffer.read(224, payload).IsOutOfRange());
  EXPECT_TRUE(buffer.fill(224, 64, std::byte{0}).IsOutOfRange());
  EXPECT_TRUE(buffer.view().subview(224, 64).error().IsOutOfRange());
  EXPECT_TRUE(buffer.write(static_cast<std::size_t>(-3), payload)
                  .IsOutOfRange());
}

TEST(ComputeContract, ProviderRejectsUnsupportedOverAlignment) {
  Device device = default_device();
  const mem::MemoryResourceRef resource =
      device_memory_resource(device).value();
  const std::size_t alignment = resource.properties().minimum_alignment;
  if (alignment > SIZE_MAX / 2) GTEST_SKIP() << "alignment cannot be doubled";
  auto buffer = mem::Buffer::create(resource, 256, alignment * 2);
  EXPECT_FALSE(buffer.has_value());
  EXPECT_TRUE(buffer.error().IsInvalidArgument());
}

TEST(ComputeContract, CrossBackendDeviceCopyIsExplicitlyUnsupported) {
  Device first;
  Device second;
  if (!find_two_backends(&first, &second)) {
    GTEST_SKIP() << "only one backend enumerated";
  }
  Device host = first.type() == DeviceType::kCpu ? first : second;
  Device accelerator = first.type() == DeviceType::kCpu ? second : first;
  if (host.type() != DeviceType::kCpu || accelerator.type() == DeviceType::kCpu) {
    GTEST_SKIP() << "need host and accelerator backends";
  }
  auto source = mem::Buffer::create(
      device_memory_resource(accelerator).value(), 256).value();
  auto destination =
      mem::Buffer::create(device_memory_resource(host).value(), 256).value();
  Status status = destination.view().copy_from(source.view());
  EXPECT_TRUE(status.IsUnimplemented()) << status.message();
}

TEST(ComputeContract, TypedLaunchRejectsAnotherDevicesBuffer) {
  Device first;
  Device second;
  if (!find_two_backends(&first, &second)) {
    GTEST_SKIP() << "only one backend enumerated";
  }
  Device host = host_device();
  Device other = std::strcmp(first.backend(), host.backend()) == 0 ? second
                                                                   : first;
  auto kernel = Kernel::load_host(host, &noop_host_kernel);
  auto stream = Stream::create(host);
  auto wrong =
      mem::Buffer::create(device_memory_resource(other).value(), 64).value();
  Status status = launch(*stream, *kernel, LaunchConfig{}, wrong);
  EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
}

TEST(ComputeContractDeathTest, LaunchBatchCountExceedsCap) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(launch_batch_over_cap(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, LaunchBatchNullItems) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(launch_batch_null_items(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, RecordEventCrossBackend) {
  Device first;
  Device second;
  if (!find_two_backends(&first, &second)) {
    GTEST_SKIP() << "only one backend enumerated";
  }
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(record_cross_backend(), "HYLE fatal");
}

TEST(ComputeContractDeathTest, LaunchCrossBackend) {
  Device first;
  Device second;
  if (!find_two_backends(&first, &second)) {
    GTEST_SKIP() << "only one backend enumerated";
  }
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(launch_cross_backend(), "HYLE fatal");
}
