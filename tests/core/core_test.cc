#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyle/core/assert.h"
#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/version.h"

namespace {

HYLE_NOINLINE int restrict_sum(const int* HYLE_RESTRICT values,
                               std::size_t count) noexcept {
  HYLE_ASSUME(values != nullptr);
  int sum = 0;
  HYLE_UNROLL(4)
  for (std::size_t i = 0; i < count; ++i) sum += values[i];
  return sum;
}

struct Empty {};
struct EmptyMemberLayout {
  HYLE_NO_UNIQUE_ADDRESS Empty empty;
  int value;
};

}  // namespace

TEST(CoreVersion, CompileTimeProductIdentityIsConsistent) {
  static_assert(hyle::kVersionMajor == HYLE_VERSION_MAJOR);
  static_assert(hyle::kVersionMinor == HYLE_VERSION_MINOR);
  static_assert(hyle::kVersionPatch == HYLE_VERSION_PATCH);
  static_assert(hyle::kVersionPacked == 0x000100u);
  static_assert(std::string_view(hyle::kVersionString) == HYLE_VERSION_STRING);
}

TEST(CoreCache, StaticLayoutContractIsUsable) {
  static_assert(hyle::core::kMaxDestructiveInterferenceSize == 128);
  static_assert(hyle::core::kConstructiveInterferenceSize <=
                hyle::core::kDestructiveInterferenceSize);
  static_assert(hyle::core::kDestructiveInterferenceSize <=
                hyle::core::kMaxDestructiveInterferenceSize);
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  static_assert(hyle::core::kConstructiveInterferenceSize == 128);
  static_assert(hyle::core::kDestructiveInterferenceSize == 128);
#elif defined(__x86_64__) || defined(_M_X64)
  static_assert(hyle::core::kConstructiveInterferenceSize == 64);
  static_assert(hyle::core::kDestructiveInterferenceSize == 64);
#else
  static_assert(hyle::core::kConstructiveInterferenceSize == 64);
  static_assert(hyle::core::kDestructiveInterferenceSize == 128);
#endif

  alignas(hyle::core::kDestructiveInterferenceSize) std::byte storage[1]{};
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(storage) %
                hyle::core::kDestructiveInterferenceSize,
            0u);
  hyle::core::prefetch_read(storage);
  hyle::core::prefetch_read_non_temporal(storage);
  hyle::core::prefetch_write(storage);
}

TEST(CoreCompiler, PortableContractsPreserveValidInputs) {
  static_assert(sizeof(EmptyMemberLayout) == sizeof(int));

  int value = 1;
  if (HYLE_LIKELY(value == 1)) value = 2;
  if (HYLE_UNLIKELY(value == 99)) value = 3;
  EXPECT_EQ(value, 2);

  const int values[] = {1, 2, 3, 4};
  EXPECT_EQ(restrict_sum(values, 4), 10);
}

#if !defined(NDEBUG)
TEST(CoreCompilerDeath, AssumeDiagnosesFalseInDebugBuilds) {
  EXPECT_DEATH({ HYLE_ASSUME(false); }, "HYLE fatal");
}

TEST(CoreCompilerDeath, UnreachableDiagnosesTripsInDebugBuilds) {
  EXPECT_DEATH({ HYLE_UNREACHABLE(); }, "HYLE fatal");
}
#endif

TEST(CoreAssertDeath, FatalAssertAbortsOnFalse) {
  EXPECT_DEATH({ HYLE_FATAL_ASSERT(1 == 2); }, "HYLE fatal");
}
