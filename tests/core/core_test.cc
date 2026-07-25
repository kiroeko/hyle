#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyle/core/cache.h"
#include "hyle/core/version.h"
#include "hyle/macros.h"

TEST(CoreVersion, CompileTimeAndRuntimeViewsAgree) {
  static_assert(hyle::kVersionMajor == HYLE_VERSION_MAJOR);
  static_assert(hyle::kVersionMinor == HYLE_VERSION_MINOR);
  static_assert(hyle::kVersionPatch == HYLE_VERSION_PATCH);
  static_assert(hyle::kVersion == HYLE_VERSION_PACKED);
  static_assert(std::string_view(hyle::kVersionString) == HYLE_VERSION_STRING);

  EXPECT_EQ(hyle::version(), hyle::kVersion);
  EXPECT_STREQ(hyle::version_string(), hyle::kVersionString);
}

TEST(CoreCache, StaticLayoutContractIsUsable) {
  static_assert(hyle::core::kCacheLineSize == HYLE_STATIC_CACHE_LINE_SIZE);
  static_assert(hyle::core::kConstructiveInterferenceSize == hyle::core::kCacheLineSize);
  static_assert(hyle::core::kDestructiveInterferenceSize == hyle::core::kCacheLineSize);
  static_assert(hyle::core::kCacheLineSize == 64 || hyle::core::kCacheLineSize == 128);

  alignas(hyle::core::kDestructiveInterferenceSize) std::byte storage[1]{};
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage) % hyle::core::kDestructiveInterferenceSize,
            0u);
  hyle::core::prefetch_read(storage);
}

TEST(CoreMacros, BranchHintsPreserveTheirExpression) {
  int value = 1;
  if (HYLE_LIKELY(value == 1)) value = 2;
  if (HYLE_UNLIKELY(value == 99)) value = 3;
  EXPECT_EQ(value, 2);
}

TEST(CoreMacrosDeath, FatalAssertAbortsOnFalse) {
  EXPECT_DEATH({ HYLE_FATAL_ASSERT(1 == 2); }, "HYLE fatal");
}
