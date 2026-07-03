#include "hyle/error/error_handler.h"

#include <gtest/gtest.h>

#include <cstring>

#include "hyle/error/result.h"
#include "hyle/error/status.h"

using hyle::ErrorRecord;
using hyle::Severity;
using hyle::Status;

// A recorder installed for the test routes diagnostics to counts + last message,
// then restores the previous handler so other tests see the stderr default.
TEST(ErrorHandler, ReportsRouteToHandler) {
  ErrorRecord rec;
  rec.install();
  HYLE_ERROR("boom %d", 42);
  rec.restore();

  EXPECT_EQ(rec.count(Severity::Error), 1u);
  EXPECT_EQ(rec.total(), 1u);
  EXPECT_STREQ(rec.last_message(), "boom 42");
}

TEST(ErrorHandler, SeverityCountsSeparate) {
  ErrorRecord rec;
  rec.install();
  HYLE_WARN("w");
  HYLE_ERROR("e1");
  HYLE_ERROR("e2");
  rec.restore();

  EXPECT_EQ(rec.count(Severity::Warning), 1u);
  EXPECT_EQ(rec.count(Severity::Error), 2u);
  EXPECT_EQ(rec.count(Severity::Fatal), 0u);
  EXPECT_EQ(rec.total(), 3u);
}

// HYLE_ASSERT is an EXPRESSION evaluating to the condition, and reports only on
// failure. It must survive release (no NDEBUG compile-out) -- tested by behavior.
TEST(ErrorHandler, AssertEvaluatesToBool) {
  ErrorRecord rec;
  rec.install();

  bool pass = HYLE_ASSERT(1 + 1 == 2);
  bool fail = HYLE_ASSERT(1 + 1 == 3);

  rec.restore();

  EXPECT_TRUE(pass);
  EXPECT_FALSE(fail);
  EXPECT_EQ(rec.count(Severity::Error), 1u);  // only the failing one reported
  EXPECT_NE(std::strstr(rec.last_message(), "assertion failed"), nullptr);
}

// HYLE_ASSERT must not evaluate its condition twice (no double side effects).
TEST(ErrorHandler, AssertEvaluatesConditionOnce) {
  ErrorRecord rec;
  rec.install();
  int calls = 0;
  auto bump = [&] { ++calls; return true; };
  bool ok = HYLE_ASSERT(bump());
  rec.restore();

  EXPECT_TRUE(ok);
  EXPECT_EQ(calls, 1);
}

// The intended use (charter §5): guard a precondition and early-return a Status
// on failure, without aborting. Exercises HYLE_ASSERT as a bool expression that
// composes with Result/Status propagation.
TEST(ErrorHandler, AssertUsableAsStatusGuard) {
  ErrorRecord rec;
  rec.install();
  auto guarded = [](int x) -> Status {
    if (!HYLE_ASSERT(x > 0)) return Status::InvalidArgument("x<=0");
    return Status();
  };
  Status bad = guarded(-1);
  Status good = guarded(1);
  rec.restore();

  EXPECT_TRUE(bad.IsInvalidArgument());
  EXPECT_TRUE(good.ok());
  EXPECT_EQ(rec.count(Severity::Error), 1u);  // only the failing guard reported
}

// Swapping handlers nests: install A, install B, restore B, A is active again.
TEST(ErrorHandler, HandlersNestAndRestore) {
  ErrorRecord a;
  a.install();
  {
    ErrorRecord b;
    b.install();
    HYLE_ERROR("to-b");
    b.restore();
    EXPECT_EQ(b.count(Severity::Error), 1u);
  }
  HYLE_ERROR("to-a");
  a.restore();
  EXPECT_EQ(a.count(Severity::Error), 1u);  // only the post-restore one
}

// A recorder can forward to a backing handler (count AND keep logging).
TEST(ErrorHandler, BackingForward) {
  ErrorRecord sink;       // plays the role of the "backing" target
  ErrorRecord front;
  // front forwards into sink's vtable. Build sink's vtable by installing it
  // first to capture its self_ binding, then point front->backing at it.
  sink.install();
  const hyle::error_handler_t* sink_h = hyle::error_default_handler();
  sink.restore();

  front.backing = sink_h;
  front.install();
  HYLE_ERROR("dual");
  front.restore();

  EXPECT_EQ(front.count(Severity::Error), 1u);
  EXPECT_EQ(sink.count(Severity::Error), 1u);  // forwarded
}
