#include <gtest/gtest.h>

#include <limits>
#include <string_view>

#include "hyle/mem/hugepage_reservation.h"
#include "hyle/platform/numa.h"

namespace hyle::mem {
namespace {

TEST(HugePageReservation, ReportsZeroRequestWithoutSideEffects) {
  auto result = reserve_huge_1g_pages(0);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_TRUE(result->complete());
  EXPECT_EQ(result->requested_pages, 0u);
  EXPECT_EQ(result->reserved_pages, 0u);
  EXPECT_EQ(result->placement, HugePagePlacement::kUnconstrained);
}

TEST(HugePageReservation, ReportsAttemptInsteadOfHidingPartialState) {
  auto result = reserve_huge_1g_pages(1, platform::kNumaAny, 50);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_EQ(result->requested_pages, 1u);
  EXPECT_LE(result->reserved_pages, result->requested_pages);
  EXPECT_EQ(result->complete(), result->reserved_pages == 1u &&
                                    result->outcome.ok());
  if (!result->complete()) EXPECT_FALSE(result->outcome.ok());
}

TEST(HugePageReservation, ValidatesBeforeCallingProcessGlobalBackend) {
  auto invalid = reserve_huge_1g_pages(1, -2, 1);
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsOutOfRange());
  EXPECT_NE(invalid.error().message().find("NUMA node"),
            std::string_view::npos);

  auto overflow = reserve_huge_1g_pages(
      (std::numeric_limits<std::size_t>::max)() /
              (std::size_t{1} << 30) +
          1,
      platform::kNumaAny, 1);
  ASSERT_FALSE(overflow.has_value());
  EXPECT_TRUE(overflow.error().IsOutOfRange());
}

}  // namespace
}  // namespace hyle::mem
