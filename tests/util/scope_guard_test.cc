#include <gtest/gtest.h>

#include "hyle/util/scope_guard.h"

using hyle::util::ScopeGuard;

TEST(ScopeGuard, RunsOnScopeExit) {
  int ran = 0;
  {
    ScopeGuard g{[&] { ++ran; }};
    EXPECT_EQ(ran, 0);
  }
  EXPECT_EQ(ran, 1);
}

TEST(ScopeGuard, DismissCancels) {
  int ran = 0;
  {
    ScopeGuard g{[&] { ++ran; }};
    g.dismiss();
  }
  EXPECT_EQ(ran, 0);
}

TEST(ScopeGuard, MoveFiresExactlyOnce) {
  int ran = 0;
  {
    ScopeGuard g{[&] { ++ran; }};
    ScopeGuard moved = std::move(g);  // moved-from is dismissed
  }
  EXPECT_EQ(ran, 1);  // not 0 (lost) and not 2 (double-fire)
}

TEST(ScopeGuard, DeferMacro) {
  int order = 0;
  int a = 0, b = 0;
  {
    HYLE_DEFER(a = ++order);  // runs LAST (declared first, destroyed last)
    HYLE_DEFER(b = ++order);  // runs FIRST
  }
  EXPECT_EQ(b, 1);  // LIFO: last-declared defer runs first
  EXPECT_EQ(a, 2);
}

TEST(ScopeGuard, DeferFiresOnEarlyReturn) {
  // Models cleanup on any exit path: the guard runs when the lambda returns.
  int cleaned = 0;
  auto fn = [&](bool early) {
    HYLE_DEFER(++cleaned);
    if (early) return;
    (void)0;
  };
  fn(true);
  EXPECT_EQ(cleaned, 1);
  fn(false);
  EXPECT_EQ(cleaned, 2);
}
