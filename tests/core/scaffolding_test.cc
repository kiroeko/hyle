#include <gtest/gtest.h>

#include <expected>
#include <string>

#include "hyle/core/version.h"
#include "hyle/macros.h"
#include "hyle/platform.h"

// Validates the scaffolding: core links across the DLL boundary, the macro
// skeleton compiles, fail-fast aborts, and C++23 std::expected is usable.

TEST(Scaffolding, CoreLinksAndCallable) {
  EXPECT_TRUE(hyle::scaffolding_ok());
}

TEST(Scaffolding, VersionPacked) {
  EXPECT_EQ(hyle::version(), (0u << 16) | (1u << 8) | 0u);  // 0.1.0
}

TEST(Scaffolding, BranchPredictionMacros) {
  int x = 1;
  if (HYLE_LIKELY(x == 1)) x = 2;
  if (HYLE_UNLIKELY(x == 99)) x = 3;
  EXPECT_EQ(x, 2);
}

// HYLE_PAD must yield distinct member names when used twice in one struct
// (regression guard: the old `##__LINE__` paste produced identical names and
// failed to compile). Two pads of 4 bytes each must occupy at least 8 bytes.
namespace {
struct PadProbe {
  HYLE_PAD(4);
  HYLE_PAD(4);
};
}  // namespace
TEST(Scaffolding, PadMacroDistinctNames) {
  EXPECT_GE(sizeof(PadProbe), 8u);
}

// C++23 std::expected must be available (charter error model basis).
static std::expected<int, std::string> parse_pos(int v) {
  if (v < 0) return std::unexpected("negative");
  return v * 2;
}

TEST(Scaffolding, Cpp23ExpectedAvailable) {
  EXPECT_EQ(parse_pos(21).value(), 42);
  EXPECT_FALSE(parse_pos(-1).has_value());
  EXPECT_EQ(parse_pos(-1).error(), "negative");
}

// fail-fast: HYLE_FATAL_ASSERT must abort on false (death test).
TEST(ScaffoldingDeath, FatalAssertAbortsOnFalse) {
  EXPECT_DEATH({ HYLE_FATAL_ASSERT(1 == 2); }, "HYLE fatal");
}
