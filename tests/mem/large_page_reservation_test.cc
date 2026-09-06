#include <gtest/gtest.h>

#include <limits>
#include <string_view>

#include "hyle/mem/large_page_reservation.h"
#include "hyle/platform/numa.h"

namespace hyle::mem {
namespace {

TEST(LargePageArenaReservation, ReportsZeroRequestWithoutSideEffects) {
  auto result = reserve_large_page_arena(0);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_TRUE(result->complete());
  EXPECT_EQ(result->requested_gib, 0u);
  EXPECT_EQ(result->reserved_gib, 0u);
  EXPECT_EQ(result->placement, LargePagePlacement::kUnconstrained);
  EXPECT_EQ(result->page_granularity, LargePageGranularity::kUnverified);
}

TEST(LargePageArenaReservation, ReportsAttemptInsteadOfHidingPartialState) {
  auto result = reserve_large_page_arena(1, platform::kNumaAny, 50);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_EQ(result->requested_gib, 1u);
  EXPECT_LE(result->reserved_gib, result->requested_gib);
  EXPECT_EQ(result->page_granularity, LargePageGranularity::kUnverified);
  EXPECT_EQ(result->complete(), result->reserved_gib == 1u &&
                                    result->outcome.ok());
  if (!result->complete()) {
    EXPECT_FALSE(result->outcome.ok());
  }
}

TEST(LargePageArenaReservation, ValidatesBeforeCallingProcessGlobalBackend) {
  auto invalid = reserve_large_page_arena(1, -2, 1);
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsOutOfRange());
  EXPECT_NE(invalid.error().message().find("NUMA node"),
            std::string_view::npos);

  auto overflow = reserve_large_page_arena(
      (std::numeric_limits<std::size_t>::max)() /
              (std::size_t{1} << 30) +
          1,
      platform::kNumaAny, 1);
  ASSERT_FALSE(overflow.has_value());
  EXPECT_TRUE(overflow.error().IsOutOfRange());
}

}  // namespace
}  // namespace hyle::mem
