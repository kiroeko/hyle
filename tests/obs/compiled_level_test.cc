#define HYLE_OBS_COMPILED_MIN_LEVEL 2
#include "hyle/obs/log.h"

#include <gtest/gtest.h>

TEST(LogCompiledLevel, RemovesLowerSeverityCallSitesAndArguments) {
  int evaluated = 0;
  HYLE_LOG_TRACE("{}", ++evaluated);
  HYLE_LOG_DEBUG("{}", ++evaluated);
  EXPECT_EQ(evaluated, 0);
}
