#include "hyle/time/clock.h"
#include "hyle/time/native_interop.h"
#include "hyle/time/unix_timestamp.h"

#include <gtest/gtest.h>

#include <type_traits>

TEST(TimeHeaders, AreSelfContained) {
  static_assert(std::is_standard_layout_v<hyle::time::UnixTimestamp>);
  EXPECT_EQ(hyle::time::saturating_add_ns(UINT64_MAX, 1), UINT64_MAX);
}
