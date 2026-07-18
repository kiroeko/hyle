#include "hyle/error/error_handler.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"

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

TEST(ErrorHandler, RecordInstallIsIdempotentAndDestructorRestores) {
  ErrorRecord outer;
  outer.install();
  {
    ErrorRecord inner;
    inner.install();
    inner.install();  // must not save itself as the previous handler
    HYLE_ERROR("inner");
    EXPECT_EQ(inner.count(Severity::Error), 1u);
  }  // restores outer even when restore() was not called explicitly
  HYLE_ERROR("outer");
  outer.restore();
  EXPECT_EQ(outer.count(Severity::Error), 1u);
}

TEST(ErrorHandler, NullFormatAndInvalidHandlerAreSafe) {
  ErrorRecord rec;
  rec.install();
  hyle::error_report(Severity::Error, nullptr, 0, nullptr);
  rec.restore();
  EXPECT_EQ(rec.count(Severity::Error), 1u);
  EXPECT_STREQ(rec.last_message(), "");

  hyle::error_handler_t invalid{};
  const auto* previous = hyle::error_set_default_handler(&invalid);
  EXPECT_NE(hyle::error_default_handler()->report, nullptr);
  hyle::error_set_default_handler(previous);
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

// ErrorRecord keeps only the last message and truncates at kMsgCap-1.
TEST(ErrorHandler, RecordTruncatesLongMessage) {
  ErrorRecord rec;
  rec.install();
  std::string long_msg(ErrorRecord::kMsgCap + 50, 'Z');
  HYLE_ERROR("%s", long_msg.c_str());
  rec.restore();

  EXPECT_EQ(rec.count(Severity::Error), 1u);
  EXPECT_EQ(std::strlen(rec.last_message()), ErrorRecord::kMsgCap - 1);
  EXPECT_EQ(rec.last_message()[0], 'Z');
}

// error_report formats into a 1024-byte stack buffer; oversize format truncates.
TEST(ErrorHandler, ReportFormatBufferTruncates) {
  ErrorRecord rec;
  rec.install();
  std::string long_msg(2000, 'A');
  HYLE_WARN("%s", long_msg.c_str());
  rec.restore();

  EXPECT_EQ(rec.count(Severity::Warning), 1u);
  // ErrorRecord further caps at 255; both layers truncate safely (no overrun).
  EXPECT_LT(std::strlen(rec.last_message()), ErrorRecord::kMsgCap);
  EXPECT_EQ(rec.last_message()[0], 'A');
}

TEST(ErrorHandler, SetNullResetsToDefault) {
  ErrorRecord rec;
  rec.install();
  const hyle::error_handler_t* prev = hyle::error_set_default_handler(nullptr);
  // prev was rec's handler; default is restored (stderr). Reinstall rec to check
  // we can still swap.
  EXPECT_NE(prev, nullptr);
  hyle::error_set_default_handler(prev);  // put rec back
  HYLE_ERROR("after-null-cycle");
  rec.restore();
  EXPECT_EQ(rec.count(Severity::Error), 1u);
}

namespace {
std::atomic<int> g_concurrent_hits{0};
void concurrent_report(hyle::error_handler_o*, Severity, const char*,
                       std::uint32_t, const char*) {
  g_concurrent_hits.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

namespace {
std::atomic<int> g_reentrant_hits{0};
void reentrant_report(hyle::error_handler_o*, Severity, const char*,
                      std::uint32_t, const char*) {
  const int hit = g_reentrant_hits.fetch_add(1, std::memory_order_relaxed);
  if (hit == 0) HYLE_WARN("nested diagnostic");
}
}  // namespace

TEST(ErrorHandler, HandlerMayReportRecursively) {
  g_reentrant_hits.store(0, std::memory_order_relaxed);
  hyle::error_handler_t custom{nullptr, &reentrant_report};
  const hyle::error_handler_t* prev =
      hyle::error_set_default_handler(&custom);
  HYLE_ERROR("outer diagnostic");
  hyle::error_set_default_handler(prev);
  EXPECT_EQ(g_reentrant_hits.load(std::memory_order_relaxed), 2);
}

TEST(ErrorHandler, ConcurrentReportsAreCounted) {
  // ErrorRecord counts are not atomic; use a custom handler with atomic hits.
  g_concurrent_hits.store(0);
  hyle::error_handler_t custom{};
  custom.inst = nullptr;
  custom.report = &concurrent_report;
  const hyle::error_handler_t* prev = hyle::error_set_default_handler(&custom);
  constexpr int kThreads = 4;
  constexpr int kPerThread = 50;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([t] {
      for (int i = 0; i < kPerThread; ++i) {
        HYLE_ERROR("t%d-%d", t, i);
      }
    });
  }
  for (auto& th : threads) th.join();
  hyle::error_set_default_handler(prev);
  EXPECT_EQ(g_concurrent_hits.load(), kThreads * kPerThread);
}

TEST(ErrorHandler, DeprecatedWarningReportsOnce) {
  ErrorRecord rec;
  rec.install();
  auto legacy = [] {
    HYLE_DEPRECATED_WARNING("use new_api() instead");
  };
  legacy();
  rec.restore();

  EXPECT_EQ(rec.count(Severity::Warning), 1u);
  EXPECT_NE(std::strstr(rec.last_message(), "deprecated"), nullptr);
  EXPECT_NE(std::strstr(rec.last_message(), "use new_api() instead"), nullptr);
}

// NOT_IMPLEMENTED reports before marking unreachable. On MSVC __assume(0) does
// not trap, so we can observe the diagnostic. On GCC/Clang the path is UB after
// the report; we still verify the message format via a direct error_report twin
// (same format string the macro uses) so coverage does not depend on death tests.
TEST(ErrorHandler, NotImplementedDiagnosticFormat) {
  ErrorRecord rec;
  rec.install();
  hyle::error_report(Severity::Error, "file.cc", 10, "'%s' not yet implemented",
                     "demo_fn");
  rec.restore();
  EXPECT_EQ(rec.count(Severity::Error), 1u);
  EXPECT_STREQ(rec.last_message(), "'demo_fn' not yet implemented");
}

// HYLE_NOT_IMPLEMENTED ends in HYLE_UNREACHABLE (MSVC may int3 / GCC UB after
// the report). Message format is covered above; do not call the full macro in
// a live test process.

// FATAL must trap after the handler runs. Death tests are the only safe way.
TEST(ErrorHandlerDeathTest, FatalTrapsAfterHandler) {
  // Install a no-op-ish path: default stderr is fine; FATAL always HYLE_TRAP().
  EXPECT_DEATH({ HYLE_FATAL("must-die %d", 1); }, ".*");
}
