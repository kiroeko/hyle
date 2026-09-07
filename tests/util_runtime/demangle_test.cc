#include "hyle/util/demangle.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

using hyle::util::demangle;

namespace {
struct DemangleWidget {};
}  // namespace

// typeid().name() is mangled on GCC/Clang ("_ZN...") and readable on MSVC; in
// both cases the demangled form must contain the simple type name.
TEST(Demangle, RoundTripsTypeName) {
  char buf[256];
  std::size_t n = demangle(typeid(DemangleWidget), buf);
  EXPECT_LT(n, sizeof(buf));  // not truncated
  EXPECT_NE(std::strstr(buf, "DemangleWidget"), nullptr);
}

TEST(Demangle, OwningStringRoundTripsTypeName) {
  const std::string text = demangle(typeid(DemangleWidget));
  EXPECT_NE(text.find("DemangleWidget"), std::string::npos);
}

TEST(Demangle, OwningStringPassesThroughPlainAndNullNames) {
  EXPECT_EQ(demangle("main"), "main");
  EXPECT_TRUE(demangle(static_cast<const char*>(nullptr)).empty());
}

TEST(Demangle, OwningStringDemanglesPlatformSymbol) {
#if defined(_WIN32)
  constexpr const char* kMangled = "?foo@@YAHH@Z";
#elif defined(__GNUC__) || defined(__clang__)
  constexpr const char* kMangled = "_Z3fooi";
#else
  constexpr const char* kMangled = "";
  GTEST_SKIP() << "toolchain has no supported runtime demangler";
#endif
  const std::string text = demangle(kMangled);
  EXPECT_NE(text, kMangled);
  EXPECT_NE(text.find("foo"), std::string::npos);
}

// A name outside the platform mangling scheme passes through unchanged.
TEST(Demangle, PassesThroughPlainName) {
  char buf[64];
  std::size_t n = demangle("main", buf);
  EXPECT_STREQ(buf, "main");
  EXPECT_EQ(n, 4u);
}

// snprintf semantics: too-small buffer truncates but stays NUL-terminated, and
// the return is the FULL length (so >= cap signals truncation).
TEST(Demangle, TruncatesButNulTerminates) {
  char buf[4];
  std::size_t n = demangle("abcdefgh", buf);
  EXPECT_EQ(n, 8u);                 // full length reported
  EXPECT_EQ(std::strlen(buf), 3u);  // 3 chars + NUL fit in 4
  EXPECT_STREQ(buf, "abc");
}

TEST(Demangle, NullInputYieldsEmpty) {
  char buf[16];
  std::size_t n = demangle(nullptr, buf, sizeof(buf));
  EXPECT_EQ(n, 0u);
  EXPECT_STREQ(buf, "");
}

// cap == 0: nothing is written (buf may be untouched), full length still
// returned.
TEST(Demangle, ZeroCapWritesNothing) {
  char sentinel[8];
  std::memset(sentinel, 'X', sizeof(sentinel));
  std::size_t n = demangle("hello", sentinel, 0);
  EXPECT_EQ(n, 5u);
  EXPECT_EQ(sentinel[0], 'X');  // untouched
}

TEST(DemangleDeathTest, NonZeroCapacityRequiresOutputBuffer) {
  EXPECT_DEATH((void)demangle("main", nullptr, 1), "");
}

TEST(Demangle, BufferMayExactlyOverlapInput) {
  char text[32] = "ordinary_symbol";
  const std::size_t size = demangle(text, text, sizeof(text));
  EXPECT_EQ(size, 15u);
  EXPECT_STREQ(text, "ordinary_symbol");
}

TEST(Demangle, BufferMayPartiallyOverlapInput) {
  char storage[32] = "xxordinary_symbol";
  const std::size_t size = demangle(storage + 2, storage, sizeof(storage));
  EXPECT_EQ(size, 15u);
  EXPECT_STREQ(storage, "ordinary_symbol");
}

// Distinct types demangle to distinct names.
TEST(Demangle, DistinctTypes) {
  char a[128], b[128];
  demangle(typeid(int).name(), a, sizeof(a));
  demangle(typeid(double).name(), b, sizeof(b));
  EXPECT_STRNE(a, b);
}

TEST(Demangle, ZeroAndOneCapacityAndReverseTruncatedOverlap) {
  EXPECT_EQ(demangle("hello", nullptr, 0), 5u);
  char one = 'X';
  EXPECT_EQ(demangle("hello", &one, 1), 5u);
  EXPECT_EQ(one, '\0');
  char text[32] = "ordinary_symbol";
  EXPECT_EQ(demangle(text, text + 2, 5), 15u);
  EXPECT_STREQ(text + 2, "ordi");
}

TEST(Demangle, PlatformSymbolMayOverlapAndTruncate) {
#if defined(_WIN32)
  constexpr const char* name = "?foo@@YAHH@Z";
#else
  constexpr const char* name = "_Z3fooi";
#endif
  const std::string expected = demangle(name);
  char storage[128];
  std::memcpy(storage, name, std::strlen(name) + 1);
  EXPECT_EQ(demangle(storage, storage, 4), expected.size());
  EXPECT_EQ(std::string(storage), expected.substr(0, 3));
}

TEST(Demangle, LongPlatformSymbolAndConcurrentCalls) {
  const std::string identifier(1500, 'x');
#if defined(_WIN32)
  const std::string name = "?" + identifier + "@@YAHH@Z";
#else
  const std::string name = "_Z1500" + identifier + "i";
#endif
  const std::string expected = demangle(name.c_str());
  // Native demanglers may reject a synthetic stress symbol. HYLE promises
  // either a successful spelling or an exact input fallback.
  if (expected != name) {
    ASSERT_NE(expected.find(identifier), std::string::npos);
  }
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&] {
      for (int j = 0; j < 32; ++j) {
        EXPECT_EQ(demangle(name.c_str()), expected);
      }
    });
  }
  for (auto& worker : workers) worker.join();
}
