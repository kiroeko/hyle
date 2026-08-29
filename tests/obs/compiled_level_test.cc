#define HYLE_OBS_COMPILED_MIN_LEVEL 2
#include "hyle/obs/log.h"

#include <gtest/gtest.h>

#include <chrono>

TEST(LogCompiledLevel, RemovesLowerSeverityCallSitesAndArguments) {
  int evaluated = 0;
  HYLE_LOG_TRACE("{}", ++evaluated);
  HYLE_LOG_DEBUG("{}", ++evaluated);
  HYLE_LOG_TRACE_IF(++evaluated != 0, "{}", ++evaluated);
  HYLE_LOG_TRACE_EVERY_N(++evaluated, "{}", ++evaluated);
  HYLE_LOG_TRACE_EVERY_N_EXACT(++evaluated, "{}", ++evaluated);
  HYLE_LOG_TRACE_FIRST_N(++evaluated, "{}", ++evaluated);
  HYLE_LOG_TRACE_EVERY_POW_2("{}", ++evaluated);
  HYLE_LOG_TRACE_EVERY((++evaluated, std::chrono::nanoseconds(1)), "{}",
                       ++evaluated);
  EXPECT_EQ(evaluated, 0);
}
