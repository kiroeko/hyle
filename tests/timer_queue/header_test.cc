#include "hyle/time/common_timeout_queue.h"
#include "hyle/time/timer_entry.h"
#include "hyle/time/timer_queue.h"

#include <gtest/gtest.h>

#include <type_traits>

TEST(TimerQueueHeaders, AreSelfContained) {
  static_assert(std::is_standard_layout_v<hyle::time::TimerEntry>);
  hyle::time::TimerQueue queue(0);
  EXPECT_TRUE(queue.empty());
}
