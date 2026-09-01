/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include <gtest/gtest.h>

#include "hyle/core/assert.h"

namespace {

template <typename T>
void valid_dependent_debug_contract(T value, int& conditions, int& messages,
                                    int& fields) {
  HYLE_DEBUG_ASSERT((++conditions, value > T{}));
  HYLE_DEBUG_ASSERT_MSG((++conditions, value > T{}),
                        (++messages, "unused"));
  HYLE_DEBUG_ASSERT_CONTEXT(
      (++conditions, value > T{}), (++messages, "unused"),
      hyle::core::fatal_field("value", (++fields, value)));
}

}  // namespace

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

  valid_dependent_debug_contract(7, conditions, messages, fields);
  EXPECT_EQ(conditions, 0);
  EXPECT_EQ(messages, 0);
  EXPECT_EQ(fields, 0);
#endif
}
