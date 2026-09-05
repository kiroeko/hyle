#include "hyle/platform/cpu_convenience.h"
#include "hyle/platform/environment_convenience.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace hyle::platform {
namespace {

TEST(PlatformConvenience, ReturnsOwnedEnvironmentSnapshots) {
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
  auto cache_classes = available_cpu_cache_classes();
  ASSERT_TRUE(ids.has_value()) << ids.error().message();
  ASSERT_TRUE(info.has_value()) << info.error().message();
  ASSERT_TRUE(cache_classes.has_value()) << cache_classes.error().message();
  ASSERT_FALSE(ids->empty());
  ASSERT_FALSE(cache_classes->empty());
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

TEST(PlatformConvenience, RetriesACpuViewThatGrowsAfterSizing) {
  unsigned calls = 0;
  auto values = cpu_convenience_detail::query_vector<unsigned>(
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
