// A Windows SDK or another C-compatible header can define function-like
// min/max macros before HYLE is included. Timer headers must remain composable
// without requiring every consumer to control include order or define NOMINMAX.
#define min(a, b) hyle_timer_queue_min_macro_must_not_expand(a, b)
#define max(a, b) hyle_timer_queue_max_macro_must_not_expand(a, b)

#include "hyle/time/common_timeout_queue.h"
#include "hyle/time/timer_entry.h"
#include "hyle/time/timer_queue.h"

#undef max
#undef min

#include <gtest/gtest.h>

TEST(TimerQueueMacroComposition, AllPublicHeadersTolerateMinMaxMacros) {
  hyle::time::CommonTimeoutQueue queue(hyle::time::Duration(1));
  EXPECT_EQ(queue.duration(), hyle::time::Duration(1));
}
