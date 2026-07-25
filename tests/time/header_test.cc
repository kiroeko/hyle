#include "hyle/time/clock.h"
#include "hyle/time/timer_entry.h"
#include "hyle/time/timer_queue.h"
#include "hyle/time/common_timeout_queue.h"

#include <gtest/gtest.h>

#include <type_traits>

TEST(TimeHeaders, AreSelfContained) {
  static_assert(std::is_standard_layout_v<hyle::time::TimerEntry>);
  EXPECT_EQ(hyle::time::saturating_add_ns(UINT64_MAX, 1), UINT64_MAX);
}
