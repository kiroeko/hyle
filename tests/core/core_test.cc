#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyle/c/compiler.h"
#include "hyle/core/cache.h"
#include "hyle/core/version.h"
#include "hyle/macros.h"

namespace {

HYLE_NOINLINE int restrict_sum(const int* HYLE_RESTRICT values,
                               std::size_t count) noexcept {
  HYLE_ASSUME(values != nullptr);
  int sum = 0;
  HYLE_UNROLL(4)
  for (std::size_t i = 0; i < count; ++i) sum += values[i];
  return sum;
}

}  // namespace

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
  hyle::core::prefetch_read_non_temporal(storage);
  hyle::core::prefetch_write(storage);
}

TEST(CoreMacros, BranchHintsPreserveTheirExpression) {
  int value = 1;
  if (HYLE_LIKELY(value == 1)) value = 2;
  if (HYLE_UNLIKELY(value == 99)) value = 3;
  EXPECT_EQ(value, 2);
}

TEST(CoreMacros, OptimizerContractsCompileAndPreserveValidInputs) {
  const int values[] = {1, 2, 3, 4};
  EXPECT_EQ(restrict_sum(values, 4), 10);
}

#if !defined(NDEBUG)
TEST(CoreMacrosDeath, AssumeDiagnosesFalseInDebugBuilds) {
  EXPECT_DEATH({ HYLE_ASSUME(false); }, "HYLE fatal");
}

TEST(CoreMacrosDeath, CAssumeDiagnosesFalseInDebugBuilds) {
  EXPECT_DEATH({ HYLE_C_ASSUME(0); }, "HYLE C fatal");
}
#endif

TEST(CoreMacrosDeath, FatalAssertAbortsOnFalse) {
  EXPECT_DEATH({ HYLE_FATAL_ASSERT(1 == 2); }, "HYLE fatal");
}
