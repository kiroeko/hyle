#include <gtest/gtest.h>

#include <type_traits>

#include "hyle/util/scope_guard.h"

using hyle::util::ScopeGuard;
using hyle::util::ScopeFail;
using hyle::util::ScopeSuccess;

using EmptyGuard = ScopeGuard<decltype([]() noexcept {})>;
static_assert(std::is_nothrow_destructible_v<EmptyGuard>);
static_assert(sizeof(EmptyGuard) == sizeof(bool),
              "an empty cleanup must overlap the active flag");

namespace {

struct CopyFallbackAction {
  int* runs;
  int* copies;
  bool* throw_on_copy;

  CopyFallbackAction(int* runs_in, int* copies_in,
                     bool* throw_on_copy_in) noexcept
      : runs(runs_in), copies(copies_in), throw_on_copy(throw_on_copy_in) {}
  CopyFallbackAction(const CopyFallbackAction& other)
      : runs(other.runs),
        copies(other.copies),
        throw_on_copy(other.throw_on_copy) {
    ++*copies;
    if (*throw_on_copy) throw 17;
  }
  CopyFallbackAction(CopyFallbackAction&&) noexcept(false) { throw 19; }

  void operator()() noexcept { ++*runs; }
};

}  // namespace

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

TEST(ScopeGuard, InvokeRunsAtMostOnce) {
  int ran = 0;
  {
    ScopeGuard guard{[&] noexcept { ++ran; }};
    EXPECT_TRUE(guard.active());
    guard.invoke();
    EXPECT_EQ(ran, 1);
    EXPECT_FALSE(guard.active());
    guard.invoke();
  }
  EXPECT_EQ(ran, 1);
}

TEST(ScopeGuard, ExplicitInvokeMayPropagate) {
  ScopeGuard guard{[] { throw 7; }};
  EXPECT_THROW(guard.invoke(), int);
  EXPECT_FALSE(guard.active());
  // The action was consumed before invocation, so unwinding and destruction
  // cannot run it a second time. Destructor-time invocation remains noexcept.
}

TEST(ScopeGuard, MoveFiresExactlyOnce) {
  int ran = 0;
  {
    ScopeGuard g{[&] { ++ran; }};
    ScopeGuard moved = std::move(g);  // moved-from is dismissed
  }
  EXPECT_EQ(ran, 1);  // not 0 (lost) and not 2 (double-fire)
}

TEST(ScopeGuard, ConstructionFailureRunsOriginalCleanup) {
  int runs = 0;
  int copies = 0;
  bool throw_on_copy = true;
  CopyFallbackAction action{&runs, &copies, &throw_on_copy};

  EXPECT_THROW(
      ([&] {
        ScopeGuard guard{action};
        (void)guard;
      }()),
      int);
  EXPECT_EQ(copies, 1);
  EXPECT_EQ(runs, 1);
}

TEST(ScopeGuard, ThrowingMoveUsesCopyFallback) {
  int runs = 0;
  int copies = 0;
  bool throw_on_copy = false;
  CopyFallbackAction action{&runs, &copies, &throw_on_copy};
  {
    ScopeGuard source{action};
    ScopeGuard moved{std::move(source)};
    EXPECT_FALSE(source.active());  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(moved.active());
    EXPECT_EQ(copies, 2);
  }
  EXPECT_EQ(runs, 1);
}

TEST(ScopeGuard, FailedTransferLeavesSourceOwningCleanup) {
  int runs = 0;
  int copies = 0;
  bool throw_on_copy = false;
  CopyFallbackAction action{&runs, &copies, &throw_on_copy};
  {
    ScopeGuard source{action};
    throw_on_copy = true;
    EXPECT_THROW(
        ([&] {
          ScopeGuard moved{std::move(source)};
          (void)moved;
        }()),
        int);
    EXPECT_TRUE(source.active());  // NOLINT(bugprone-use-after-move)
  }
  EXPECT_EQ(runs, 1);
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

TEST(ConditionalScopeGuard, SelectsNormalOrExceptionalExit) {
  int failures = 0;
  int successes = 0;
  const auto run = [&](bool fail) {
    ScopeFail on_fail{[&] noexcept { ++failures; }};
    ScopeSuccess on_success{[&] noexcept { ++successes; }};
    if (fail) throw 7;
  };

  EXPECT_NO_THROW(run(false));
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(successes, 1);
  EXPECT_THROW(run(true), int);
  EXPECT_EQ(failures, 1);
  EXPECT_EQ(successes, 1);
}

TEST(ConditionalScopeGuard, MoveAndDismissRemainOnceOnly) {
  int failures = 0;
  int successes = 0;
  EXPECT_THROW(
      ([&] {
        ScopeFail source{[&] noexcept { ++failures; }};
        auto moved = std::move(source);
        EXPECT_FALSE(source.active());  // NOLINT(bugprone-use-after-move)
        EXPECT_TRUE(moved.active());
        throw 1;
      }()),
      int);
  {
    ScopeSuccess success{[&] noexcept { ++successes; }};
    success.dismiss();
  }
  EXPECT_EQ(failures, 1);
  EXPECT_EQ(successes, 0);
}

TEST(ConditionalScopeGuard, NestedCaughtExceptionIsSuccessForOuterScope) {
  int failures = 0;
  int successes = 0;
  {
    ScopeFail on_fail{[&] noexcept { ++failures; }};
    ScopeSuccess on_success{[&] noexcept { ++successes; }};
    try {
      throw 3;
    } catch (int) {
    }
  }
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(successes, 1);
}

TEST(ConditionalScopeGuard, ConstructionFailureRunsOnlyFailureCleanup) {
  int fail_runs = 0;
  int success_runs = 0;
  int copies = 0;
  bool throw_on_copy = true;
  CopyFallbackAction fail_action{&fail_runs, &copies, &throw_on_copy};
  EXPECT_THROW(
      ([&] {
        ScopeFail guard{fail_action};
        (void)guard;
      }()),
      int);
  EXPECT_EQ(fail_runs, 1);

  copies = 0;
  CopyFallbackAction success_action{&success_runs, &copies, &throw_on_copy};
  EXPECT_THROW(
      ([&] {
        ScopeSuccess guard{success_action};
        (void)guard;
      }()),
      int);
  EXPECT_EQ(success_runs, 0);
}
