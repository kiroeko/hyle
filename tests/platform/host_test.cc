#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>

#include "hyle/platform/environment.h"
#include "hyle/platform/host_memory.h"
#include "hyle/platform/random.h"
#include "host_memory_internal.h"

namespace hyle::platform {
namespace {

TEST(HostMemory, SnapshotHasCoherentPhysicalBounds) {
  auto info = host_memory_info();
  ASSERT_TRUE(info.has_value()) << info.error().message();
  EXPECT_GT(info->physical_total_bytes, 0u);
  EXPECT_LE(info->physical_available_bytes, info->physical_total_bytes);
  EXPECT_LE(info->effective_available_bytes(),
            info->physical_available_bytes);
  if (info->process_available_bytes && info->process_limit_bytes) {
    EXPECT_LE(*info->process_available_bytes, *info->process_limit_bytes);
  }
  if (info->commit_available_bytes && info->commit_limit_bytes) {
    EXPECT_LE(*info->commit_available_bytes, *info->commit_limit_bytes);
  }
}

TEST(HostMemory, EffectiveBudgetUsesEveryAvailableConstraint) {
  HostMemoryInfo info;
  info.physical_available_bytes = 900;
  info.commit_available_bytes = 700;
  info.process_available_bytes = 800;
  EXPECT_EQ(info.effective_available_bytes(), 700u);
  EXPECT_TRUE(info.constrained());
  info.commit_available_bytes.reset();
  info.process_available_bytes.reset();
  EXPECT_EQ(info.effective_available_bytes(), 900u);
  EXPECT_FALSE(info.constrained());
}

TEST(HostMemory, LinuxMeminfoParserRejectsWrongUnitsAndOverflow) {
  std::uint64_t bytes = 0;
  EXPECT_TRUE(detail::parse_linux_meminfo_kib(
      "MemTotal:       123 kB\n", "MemTotal", bytes));
  EXPECT_EQ(bytes, 123u * 1024u);
  EXPECT_FALSE(detail::parse_linux_meminfo_kib(
      "MemFree: 123 kB\n", "MemTotal", bytes));
  EXPECT_FALSE(detail::parse_linux_meminfo_kib(
      "MemTotal: 123 MB\n", "MemTotal", bytes));
  EXPECT_FALSE(detail::parse_linux_meminfo_kib(
      "MemTotal: 18446744073709551615 kB\n", "MemTotal", bytes));
  EXPECT_FALSE(detail::parse_linux_meminfo_kib(
      "MemTotal: 12 kB trailing\n", "MemTotal", bytes));
}

TEST(SecureRandom, FillsCallerStorageAndAcceptsEmptySpans) {
  EXPECT_TRUE(secure_random({}).ok());
  std::array<std::byte, 64> first{};
  std::array<std::byte, 64> second{};
  ASSERT_TRUE(secure_random(first).ok());
  ASSERT_TRUE(secure_random(second).ok());
  EXPECT_NE(first, second);
}

TEST(Environment, DistinguishesEmptyPresentAndMissingValues) {
  constexpr const char* kName = "HYLE_PLATFORM_HOST_TEST_EMPTY";
  ASSERT_TRUE(unset_env_var(kName).ok());
  char buffer[8] = {'x'};
  auto missing = get_env_var(kName, buffer, sizeof(buffer));
  ASSERT_FALSE(missing.has_value());
  EXPECT_TRUE(missing.error().IsNotFound());
  EXPECT_EQ(buffer[0], '\0');

  ASSERT_TRUE(set_env_var(kName, "").ok());
  auto empty = get_env_var(kName, buffer, sizeof(buffer));
  ASSERT_TRUE(empty.has_value()) << empty.error().message();
  EXPECT_EQ(*empty, 0u);
  EXPECT_EQ(buffer[0], '\0');
  EXPECT_TRUE(unset_env_var(kName).ok());
}

TEST(Environment, ReportsFullLengthAndAlwaysTerminatesPartialOutput) {
  constexpr const char* kName = "HYLE_PLATFORM_HOST_TEST_VALUE";
  constexpr const char* kValue = "abcdef";
  ASSERT_TRUE(set_env_var(kName, kValue).ok());
  char tiny[4] = {};
  auto length = get_env_var(kName, tiny, sizeof(tiny));
  ASSERT_TRUE(length.has_value()) << length.error().message();
  EXPECT_EQ(*length, std::strlen(kValue));
  EXPECT_STREQ(tiny, "abc");
  EXPECT_TRUE(unset_env_var(kName).ok());
}

}  // namespace
}  // namespace hyle::platform
