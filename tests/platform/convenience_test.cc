#include "hyle/platform/convenience.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace hyle::platform {
namespace {

TEST(PlatformConvenience, ReturnsOwnedDirectoryAndEnvironmentSnapshots) {
  auto cwd = current_directory();
  auto temp = temp_directory();
  auto home = home_directory();
  ASSERT_TRUE(cwd.has_value()) << cwd.error().message();
  ASSERT_TRUE(temp.has_value()) << temp.error().message();
  ASSERT_TRUE(home.has_value()) << home.error().message();
  EXPECT_FALSE(cwd->empty());
  EXPECT_FALSE(temp->empty());
  EXPECT_FALSE(home->empty());

  constexpr const char* kName = "HYLE_PLATFORM_CONVENIENCE_TEST";
  ASSERT_TRUE(set_env_var(kName, "owned-value").ok());
  auto value = get_env_var(kName);
  ASSERT_TRUE(value.has_value()) << value.error().message();
  EXPECT_EQ(*value, "owned-value");
  ASSERT_TRUE(unset_env_var(kName).ok());
  auto missing = get_env_var(kName);
  ASSERT_FALSE(missing.has_value());
  EXPECT_TRUE(missing.error().IsNotFound());
}

TEST(PlatformConvenience, ReturnsCompleteOwnedCpuViews) {
  auto ids = available_cpu_ids();
  auto info = available_cpu_info();
  ASSERT_TRUE(ids.has_value()) << ids.error().message();
  ASSERT_TRUE(info.has_value()) << info.error().message();
  ASSERT_FALSE(ids->empty());
  ASSERT_EQ(info->size(), ids->size());
  for (std::size_t i = 0; i < ids->size(); ++i) {
    EXPECT_EQ((*info)[i].id, (*ids)[i]);
  }

  auto ordered = available_cpu_ids(CpuOrder::kPerformanceFirst);
  ASSERT_TRUE(ordered.has_value()) << ordered.error().message();
  auto natural_set = *ids;
  auto ordered_set = *ordered;
  std::sort(natural_set.begin(), natural_set.end());
  std::sort(ordered_set.begin(), ordered_set.end());
  EXPECT_EQ(ordered_set, natural_set);
}

TEST(PlatformConvenience, RetriesAStringThatGrowsBetweenCountAndFill) {
  unsigned calls = 0;
  auto value = convenience_detail::query_string(
      [&calls](char* dst, std::size_t cap) -> Result<std::size_t> {
        ++calls;
        const char* text = calls == 1 ? "hi" : "hello";
        const std::size_t size = std::strlen(text);
        if (dst != nullptr && cap != 0) {
          const std::size_t copied = std::min(size, cap - 1);
          std::memcpy(dst, text, copied);
          dst[copied] = '\0';
        }
        return size;
      });
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, "hello");
  EXPECT_EQ(calls, 3u);
}

TEST(PlatformConvenience, DoesNotReturnARepeatedlyChangingPartialSnapshot) {
  auto value = convenience_detail::query_string(
      [](char*, std::size_t cap) -> Result<std::size_t> {
        return cap == 0 ? 1u : cap + 1;
      });
  ASSERT_FALSE(value.has_value());
  EXPECT_TRUE(value.error().IsAborted());
}

TEST(PlatformConvenience, RetriesACpuViewThatGrowsAfterSizing) {
  unsigned calls = 0;
  auto values = convenience_detail::query_vector<unsigned>(
      [&calls](unsigned* dst, std::size_t cap) -> Result<std::size_t> {
        ++calls;
        const std::size_t count = calls == 1 ? 1 : 3;
        for (std::size_t i = 0; i < std::min(cap, count); ++i) {
          dst[i] = static_cast<unsigned>(i + 2);
        }
        return count;
      });
  ASSERT_TRUE(values.has_value());
  EXPECT_EQ(*values, (std::vector<unsigned>{2, 3, 4}));
  EXPECT_EQ(calls, 3u);
}

}  // namespace
}  // namespace hyle::platform
