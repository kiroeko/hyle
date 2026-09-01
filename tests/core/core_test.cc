#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <thread>

#include "hyle/core/assert.h"
#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/core/environment.h"
#include "hyle/version.h"

#if !HYLE_HAS_CPP_ATTRIBUTE(nodiscard)
#  error "the supported Core toolchain must recognize [[nodiscard]]"
#endif

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

void observer_a(const hyle::core::FatalEvent&) noexcept {}
void observer_b(const hyle::core::FatalEvent&) noexcept {}

void reporting_observer(const hyle::core::FatalEvent& event) noexcept {
  std::fprintf(stderr, "observer fields=%zu kind=%u\n", event.fields.size(),
               static_cast<unsigned>(event.kind));
}

void recursive_observer(const hyle::core::FatalEvent&) noexcept {
  HYLE_FATAL_ASSERT_MSG(false, "recursive observer fallback");
}

void fail_with_context() {
  HYLE_FATAL_ASSERT_CONTEXT(
      false, "lookup invariant", HYLE_FATAL_NAMED_FIELD("index", 17u),
      HYLE_FATAL_NAMED_FIELD("name", "weights"));
}

void fail_after_installing_observer() {
  static_cast<void>(hyle::core::set_fatal_observer(&reporting_observer));
  HYLE_FATAL_ASSERT_CONTEXT(false, "observer event",
                            HYLE_FATAL_NAMED_FIELD("size", 3u));
}

void fail_recursively_from_observer() {
  static_cast<void>(hyle::core::set_fatal_observer(&recursive_observer));
  HYLE_FATAL_ASSERT(false);
}

}  // namespace

TEST(CoreVersion, CompileTimeProductIdentityIsConsistent) {
  static_assert(hyle::kVersionMajor == HYLE_VERSION_MAJOR);
  static_assert(hyle::kVersionMinor == HYLE_VERSION_MINOR);
  static_assert(hyle::kVersionPatch == HYLE_VERSION_PATCH);
  static_assert(hyle::kVersionPacked == 0x000100u);
  static_assert(std::string_view(hyle::kVersionString) == HYLE_VERSION_STRING);
}

TEST(CoreEnvironment, FactsAreNumericAndMutuallyConsistent) {
  static_assert(HYLE_OS_WINDOWS + HYLE_OS_LINUX + HYLE_OS_MACOS +
                    HYLE_OS_UNKNOWN ==
                1);
  static_assert(HYLE_ARCH_X64 + HYLE_ARCH_ARM64 + HYLE_ARCH_RISCV64 +
                    HYLE_ARCH_UNKNOWN ==
                1);
  static_assert(HYLE_COMPILER_MSVC + HYLE_COMPILER_CLANG +
                    HYLE_COMPILER_GCC + HYLE_COMPILER_UNKNOWN ==
                1);
  static_assert(HYLE_COMPILER_CLANG_CL <= HYLE_COMPILER_CLANG);
  static_assert(HYLE_POINTER_32 + HYLE_POINTER_64 == 1);
  static_assert(HYLE_BITNESS == sizeof(void*) * 8);
  static_assert(HYLE_BUILD_DEBUG + HYLE_BUILD_RELEASE == 1);
  static_assert(HYLE_ENDIAN_LITTLE + HYLE_ENDIAN_BIG +
                    HYLE_ENDIAN_UNKNOWN ==
                1);
  static_assert(HYLE_SANITIZER_ANY ==
                !!(HYLE_SANITIZER_ADDRESS || HYLE_SANITIZER_THREAD ||
                   HYLE_SANITIZER_MEMORY || HYLE_SANITIZER_UNDEFINED));
  static_assert(
      (std::endian::native == std::endian::little && HYLE_ENDIAN_LITTLE) ||
      (std::endian::native == std::endian::big && HYLE_ENDIAN_BIG) ||
      (std::endian::native != std::endian::little &&
       std::endian::native != std::endian::big && HYLE_ENDIAN_UNKNOWN));
}

TEST(CoreCache, StaticLayoutContractIsUsable) {
  static_assert(hyle::core::kMaxDestructiveInterferenceSize == 128);
  static_assert(hyle::core::kConstructiveInterferenceSize <=
                hyle::core::kDestructiveInterferenceSize);
  static_assert(hyle::core::kDestructiveInterferenceSize <=
                hyle::core::kMaxDestructiveInterferenceSize);
#if HYLE_OS_MACOS && HYLE_ARCH_ARM64
  static_assert(hyle::core::kConstructiveInterferenceSize == 128);
  static_assert(hyle::core::kDestructiveInterferenceSize == 128);
#elif HYLE_ARCH_ARM64
  static_assert(hyle::core::kConstructiveInterferenceSize == 64);
  static_assert(hyle::core::kDestructiveInterferenceSize == 64);
#elif HYLE_ARCH_X64
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

TEST(CoreAssert, ConditionsEvaluateOnceAndSuccessSkipsMessage) {
  int conditions = 0;
  int messages = 0;
  int fields = 0;
  HYLE_FATAL_ASSERT(++conditions == 1);
  HYLE_FATAL_ASSERT_MSG(++conditions == 2, (++messages, "unused"));
  HYLE_FATAL_ASSERT_CONTEXT(
      ++conditions == 3, (++messages, "unused"),
      hyle::core::fatal_field("field", ++fields));
  EXPECT_EQ(conditions, 3);
  EXPECT_EQ(messages, 0);
  EXPECT_EQ(fields, 0);
}

TEST(CoreAssert, FatalFieldsPreserveSupportedValueKinds) {
  int object = 0;
  const auto boolean = hyle::core::fatal_field("boolean", true);
  const auto signed_value = hyle::core::fatal_field("signed", -7);
  const auto unsigned_value = hyle::core::fatal_field("unsigned", 9u);
  const auto floating = hyle::core::fatal_field("floating", 1.25);
  const auto pointer = hyle::core::fatal_field("pointer", &object);
  const auto string = hyle::core::fatal_field("string", "value");

  EXPECT_EQ(boolean.value.kind, hyle::core::FatalValueKind::kBoolean);
  EXPECT_TRUE(boolean.value.boolean_value);
  EXPECT_EQ(signed_value.value.signed_value, -7);
  EXPECT_EQ(unsigned_value.value.unsigned_value, 9u);
  EXPECT_DOUBLE_EQ(floating.value.floating_value, 1.25);
  EXPECT_EQ(pointer.value.pointer_value, &object);
  EXPECT_EQ(string.value.string_value, "value");
}

TEST(CoreAssert, DefaultTextFormatterEnforcesItsDocumentedBound) {
#if HYLE_COMPILER_MSVC
  std::FILE* stream = nullptr;
  ASSERT_EQ(::tmpfile_s(&stream), 0);
#else
  std::FILE* stream = std::tmpfile();
#endif
  ASSERT_NE(stream, nullptr);
  const std::string text(hyle::core::kMaxFatalTextBytes + 17, 'x');
  hyle::core::detail::write_bounded(stream, text);
  ASSERT_EQ(std::fflush(stream), 0);
  ASSERT_EQ(std::fseek(stream, 0, SEEK_END), 0);
  EXPECT_EQ(std::ftell(stream),
            static_cast<long>(hyle::core::kMaxFatalTextBytes + 3));
  EXPECT_EQ(std::fclose(stream), 0);
}

TEST(CoreAssert, ObserverExchangeIsAtomicForConcurrentHosts) {
  EXPECT_EQ(hyle::core::set_fatal_observer(nullptr), nullptr);
  std::thread first([] {
    for (int i = 0; i != 1000; ++i) {
      static_cast<void>(hyle::core::set_fatal_observer(&observer_a));
    }
  });
  std::thread second([] {
    for (int i = 0; i != 1000; ++i) {
      static_cast<void>(hyle::core::set_fatal_observer(&observer_b));
    }
  });
  first.join();
  second.join();
  const auto previous = hyle::core::set_fatal_observer(nullptr);
  EXPECT_TRUE(previous == &observer_a || previous == &observer_b);
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

TEST(CoreAssertDeath, FatalAssertMessageIsReported) {
  EXPECT_DEATH({ HYLE_FATAL_ASSERT_MSG(false, "consumer context"); },
               "consumer context");
}

TEST(CoreAssertDeath, FatalContextReportsBoundedTypedFields) {
  EXPECT_DEATH({ fail_with_context(); }, "index = 17");
}

TEST(CoreAssertDeath, ObserverReceivesEventThenCoreStillTerminates) {
  EXPECT_DEATH({ fail_after_installing_observer(); }, "observer fields=1");
}

TEST(CoreAssertDeath, RecursiveFatalBypassesObserver) {
  EXPECT_DEATH({ fail_recursively_from_observer(); },
               "recursive observer fallback");
}

#if HYLE_DEBUG_ASSERT_ENABLED
TEST(CoreAssertDeath, DebugAssertionIsActiveInInstrumentedBuilds) {
  EXPECT_DEATH({ HYLE_DEBUG_ASSERT_MSG(false, "debug assertion"); },
               "debug assertion");
}
#endif
