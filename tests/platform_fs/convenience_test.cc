#include "hyle/platform/fs_convenience.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

namespace hyle::platform {
namespace {

TEST(PlatformFsConvenience, ReturnsOwnedDirectorySnapshots) {
  auto cwd = current_directory();
  auto temp = temp_directory();
  auto home = home_directory();
  ASSERT_TRUE(cwd.has_value()) << cwd.error().message();
  ASSERT_TRUE(temp.has_value()) << temp.error().message();
  ASSERT_TRUE(home.has_value()) << home.error().message();
  EXPECT_FALSE(cwd->empty());
  EXPECT_FALSE(temp->empty());
  EXPECT_FALSE(home->empty());
}

TEST(PlatformFsConvenience, RetriesAStringThatGrowsBetweenCountAndFill) {
  unsigned calls = 0;
  auto value = fs_convenience_detail::query_string(
      [&calls](char* dst, std::size_t cap) -> Result<std::size_t> {
        ++calls;
        const char* text = calls == 1 ? "hi" : "hello";
        const std::size_t size = std::strlen(text);
        if (dst != nullptr && cap != 0) {
          const std::size_t copied = (std::min)(size, cap - 1);
          std::memcpy(dst, text, copied);
          dst[copied] = '\0';
        }
        return size;
      });
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, "hello");
  EXPECT_EQ(calls, 3u);
}

TEST(PlatformFsConvenience, RejectsARepeatedlyChangingPartialSnapshot) {
  auto value = fs_convenience_detail::query_string(
      [](char*, std::size_t cap) -> Result<std::size_t> {
        return cap == 0 ? 1u : cap + 1;
      });
  ASSERT_FALSE(value.has_value());
  EXPECT_TRUE(value.error().IsAborted());
}

}  // namespace
}  // namespace hyle::platform
