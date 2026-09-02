// A Windows SDK or another C-compatible header can define function-like
// min/max macros before HYLE is included. Time's public headers must compose
// without requiring every consumer to control include order or define NOMINMAX.
#define min(a, b) hyle_time_min_macro_must_not_expand(a, b)
#define max(a, b) hyle_time_max_macro_must_not_expand(a, b)

#include "hyle/time/clock.h"
#include "hyle/time/unix_timestamp.h"

#undef max
#undef min

#include <gtest/gtest.h>

TEST(TimeMacroComposition, AllPublicHeadersTolerateMinMaxMacros) {
  EXPECT_GT((hyle::time::Duration::max)().count(), 0);
  EXPECT_TRUE(hyle::time::unix_timestamp_valid({0, 0}));
}
