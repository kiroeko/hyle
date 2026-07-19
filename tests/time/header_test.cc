#include "hyle/time/clock.h"
#include "hyle/time/timer_source.h"
#include "hyle/time/timer_wheel.h"
#include "hyle/time/common_timeout_queue.h"

#include <gtest/gtest.h>

TEST(TimeHeaders, AreSelfContained) {
  static_assert(hyle::time::TimerSource<hyle::time::TimerWheel>);
  static_assert(hyle::time::TimerSource<hyle::time::CommonTimeoutQueue>);
  EXPECT_EQ(hyle::time::saturating_add_ns(UINT64_MAX, 1), UINT64_MAX);
}
