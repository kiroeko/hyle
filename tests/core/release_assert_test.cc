/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include <gtest/gtest.h>

#include "hyle/core/assert.h"

TEST(CoreReleaseAssert, DisabledChecksDoNotEvaluateAnyInput) {
#if HYLE_SANITIZER_ANY || HYLE_STATIC_ANALYZER
  SUCCEED() << "instrumented builds intentionally keep debug assertions";
#else
  static_assert(!HYLE_DEBUG_ASSERT_ENABLED);
  int conditions = 0;
  int messages = 0;
  int fields = 0;
  HYLE_DEBUG_ASSERT(++conditions == 1);
  HYLE_DEBUG_ASSERT_MSG(++conditions == 1, (++messages, "unused"));
  HYLE_DEBUG_ASSERT_CONTEXT(
      ++conditions == 1, (++messages, "unused"),
      hyle::core::fatal_field("field", ++fields));
  EXPECT_EQ(conditions, 0);
  EXPECT_EQ(messages, 0);
  EXPECT_EQ(fields, 0);
#endif
}
