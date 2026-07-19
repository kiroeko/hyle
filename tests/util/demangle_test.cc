#include <gtest/gtest.h>

#include <cstring>
#include <typeinfo>

#include "hyle/util/demangle.h"

using hyle::util::demangle;

namespace {
struct DemangleWidget {};
}  // namespace

// typeid().name() is mangled on GCC/Clang ("_ZN...") and readable on MSVC; in
// both cases the demangled form must contain the simple type name.
TEST(Demangle, RoundTripsTypeName) {
  char buf[256];
  std::size_t n = demangle(typeid(DemangleWidget).name(), buf, sizeof(buf));
  EXPECT_LT(n, sizeof(buf));  // not truncated
  EXPECT_NE(std::strstr(buf, "DemangleWidget"), nullptr);
}

// A name outside the platform mangling scheme passes through unchanged.
TEST(Demangle, PassesThroughPlainName) {
  char buf[64];
  std::size_t n = demangle("main", buf, sizeof(buf));
  EXPECT_STREQ(buf, "main");
  EXPECT_EQ(n, 4u);
}

// snprintf semantics: too-small buffer truncates but stays NUL-terminated, and
// the return is the FULL length (so >= cap signals truncation).
TEST(Demangle, TruncatesButNulTerminates) {
  char buf[4];
  std::size_t n = demangle("abcdefgh", buf, sizeof(buf));
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

// cap == 0: nothing is written (buf may be untouched), full length still returned.
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

// Distinct types demangle to distinct names.
TEST(Demangle, DistinctTypes) {
  char a[128], b[128];
  demangle(typeid(int).name(), a, sizeof(a));
  demangle(typeid(double).name(), b, sizeof(b));
  EXPECT_STRNE(a, b);
}
