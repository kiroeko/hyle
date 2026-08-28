// Windows SDK consumers do not always define NOMINMAX before their first OS
// include. Public HYLE headers must therefore tolerate the function-like
// min/max macros regardless of include order.
#define min(a, b) hyle_windows_min_macro_must_not_expand(a, b)
#define max(a, b) hyle_windows_max_macro_must_not_expand(a, b)

#include "hyle/platform/convenience.h"
#include "hyle/time/clock.h"

#undef max
#undef min

#include <gtest/gtest.h>

TEST(WindowsMacroComposition, PublicPlatformAndTimeHeadersRemainUsable) {
  EXPECT_GT((hyle::time::Duration::max)().count(), 0);
}
