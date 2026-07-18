#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <cstring>

#include "hyle/error/result.h"
#include "hyle/error/status.h"

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;
using hyle::status_code_name;

// ToString now fills a caller buffer (no std::string across the seam). Tests may
// use std::string freely, so wrap it for convenient comparison.
static std::string to_str(const Status& s) {
  char buf[128];
  std::size_t need = s.ToString(buf, sizeof(buf));
  EXPECT_LT(need, sizeof(buf));  // these fixtures are short; no truncation
  return std::string(buf);
}

// ---- Status basics ----------------------------------------------------------

TEST(Status, DefaultIsOk) {
  Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(s.code(), StatusCode::Ok);
  EXPECT_FALSE(s.has_message());
  EXPECT_EQ(to_str(s), "OK");
}

TEST(Status, FactoryWithoutMessageDoesNotAllocate) {
  Status s = Status::NotFound();
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.IsNotFound());
  EXPECT_EQ(s.code(), StatusCode::NotFound);
  EXPECT_FALSE(s.has_message());  // message-less error -> zero alloc
  EXPECT_EQ(s.message(), "");
  EXPECT_EQ(to_str(s), "NotFound");
}

TEST(Status, FactoryWithMessage) {
  Status s = Status::Internal("boom");
  EXPECT_TRUE(s.IsInternal());
  EXPECT_TRUE(s.has_message());
  EXPECT_EQ(s.message(), "boom");
  EXPECT_EQ(to_str(s), "Internal: boom");
}

TEST(Status, OkFactoryIgnoresMessage) {
  Status s = Status::Ok("ignored");
  EXPECT_TRUE(s.ok());
  EXPECT_FALSE(s.has_message());
  EXPECT_EQ(to_str(s), "OK");
}

TEST(Status, CodeCtorHasNoMessage) {
  Status s(StatusCode::Aborted);
  EXPECT_TRUE(s.IsAborted());
  EXPECT_FALSE(s.has_message());
}

TEST(Status, Equality) {
  EXPECT_EQ(Status::NotFound("x"), Status::NotFound("x"));
  EXPECT_NE(Status::NotFound("x"), Status::NotFound("y"));
  EXPECT_NE(Status::NotFound("x"), Status::Internal("x"));
  EXPECT_EQ(Status(), Status::Ok());
}

TEST(Status, CopySharesMessage) {
  Status a = Status::Internal("shared");
  Status b = a;  // copy: ref-counts the message, no deep copy
  EXPECT_EQ(a, b);
  EXPECT_EQ(b.message(), "shared");
  EXPECT_EQ(a.message().data(), b.message().data());  // same backing buffer
}

TEST(Status, MoveLeavesUsableSource) {
  Status a = Status::Internal("m");
  Status b = std::move(a);
  EXPECT_TRUE(b.IsInternal());
  EXPECT_EQ(b.message(), "m");
}

// ---- exhaustive code-name table (every X-macro row) -------------------------

TEST(Status, EveryCodeNameRoundTrips) {
#define HYLE_CHECK_NAME(name, screaming, value) \
  EXPECT_EQ(status_code_name(StatusCode::name), #name);
  HYLE_FOREACH_STATUS_CODE(HYLE_CHECK_NAME)
#undef HYLE_CHECK_NAME
}

TEST(Status, EveryFactoryAndPredicateAgree) {
  // For each code, the factory must produce a status whose matching predicate
  // is true and whose code/name line up.
#define HYLE_CHECK_FACTORY(name, screaming, value)            \
  {                                                           \
    Status s = Status::name();                                \
    EXPECT_EQ(s.code(), StatusCode::name);                    \
    EXPECT_TRUE(s.Is##name());                                \
    EXPECT_EQ(status_code_name(s.code()), #name);             \
  }
  HYLE_FOREACH_STATUS_CODE(HYLE_CHECK_FACTORY)
#undef HYLE_CHECK_FACTORY
}

// ---- Status is a small, cheap carrier ---------------------------------------

TEST(Status, IsSmallObject) {
  // Tagged-pointer rep: exactly one word. This is the whole point of the
  // 8-byte design -- it is the error slot in every Result<T>.
  EXPECT_EQ(sizeof(Status), sizeof(void*));
}

TEST(Status, ToStringTruncates) {
  Status s = Status::Internal("boom");  // "Internal: boom" = 14 chars
  char small[5];
  std::size_t need = s.ToString(small, sizeof(small));
  EXPECT_EQ(need, 14u);                       // full length reported
  EXPECT_STREQ(small, "Inte");                // 4 bytes + NUL, truncated
  // zero-capacity must not write and must still report the needed length.
  EXPECT_EQ(s.ToString(nullptr, 0), 14u);
}

// ---- Result<T> --------------------------------------------------------------

TEST(Result, HoldsValue) {
  Result<int> r = 42;
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 42);
}

TEST(Result, HoldsError) {
  Result<int> r = std::unexpected(Status::OutOfRange("nope"));
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsOutOfRange());
  EXPECT_EQ(r.error().message(), "nope");
}

TEST(Result, MonadicTransformOnValue) {
  Result<int> r = 21;
  Result<int> doubled = r.transform([](int v) { return v * 2; });
  ASSERT_TRUE(doubled.has_value());
  EXPECT_EQ(*doubled, 42);
}

// ---- propagation macros: HYLE_RETURN_NOT_OK ---------------------------------

namespace {

Status inner_fails() { return Status::PermissionDenied("denied"); }
Status inner_ok() { return Status(); }

// returns Status, propagates Status
Status chain_status(bool fail) {
  HYLE_RETURN_NOT_OK(fail ? inner_fails() : inner_ok());
  return Status();
}

// returns Result<T>, propagates a bare Status (FailReturn -> unexpected)
Result<int> chain_result_from_status(bool fail) {
  HYLE_RETURN_NOT_OK(fail ? inner_fails() : inner_ok());
  return 7;
}

}  // namespace

TEST(ReturnNotOk, PropagatesInStatusFunction) {
  EXPECT_TRUE(chain_status(false).ok());
  Status s = chain_status(true);
  EXPECT_TRUE(s.IsPermissionDenied());
  EXPECT_EQ(s.message(), "denied");
}

TEST(ReturnNotOk, PropagatesIntoResultFunction) {
  EXPECT_EQ(chain_result_from_status(false).value(), 7);
  Result<int> r = chain_result_from_status(true);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsPermissionDenied());
}

// ---- propagation macros: HYLE_ASSIGN_OR_RAISE -------------------------------

namespace {

Result<int> parse_pos(int v) {
  if (v < 0) return std::unexpected(Status::InvalidArgument("negative"));
  return v;
}

// two assigns in one scope -> exercises __COUNTER__ uniqueness
Result<int> sum_two(int a, int b) {
  HYLE_ASSIGN_OR_RAISE(int x, parse_pos(a));
  HYLE_ASSIGN_OR_RAISE(int y, parse_pos(b));
  return x + y;
}

// 3-level nesting: leaf -> mid -> top, error must bubble unchanged
Result<int> leaf(int v) { return parse_pos(v); }
Result<int> mid(int v) {
  HYLE_ASSIGN_OR_RAISE(int x, leaf(v));
  return x + 1;
}
Result<int> top(int v) {
  HYLE_ASSIGN_OR_RAISE(int x, mid(v));
  return x + 1;
}

}  // namespace

TEST(AssignOrRaise, BindsValueOnOk) {
  Result<int> r = sum_two(20, 22);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 42);
}

TEST(AssignOrRaise, RaisesOnFirstError) {
  Result<int> r = sum_two(-1, 22);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsInvalidArgument());
  EXPECT_EQ(r.error().message(), "negative");
}

TEST(AssignOrRaise, ThreeLevelBubble) {
  EXPECT_EQ(top(10).value(), 12);  // +1 +1
  Result<int> r = top(-5);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsInvalidArgument());
  EXPECT_EQ(r.error().message(), "negative");
}

TEST(AssignOrRaise, RaisesOnSecondError) {
  // First assign ok, second fails -- exercises multi-assign + __COUNTER__.
  Result<int> r = sum_two(1, -2);
  ASSERT_FALSE(r.has_value());
  EXPECT_TRUE(r.error().IsInvalidArgument());
}

// ---- edge cases toward full coverage ----------------------------------------

TEST(Status, SelfAssignmentCopyAndMove) {
  Status a = Status::Internal("self");
  a = a;  // copy self-assign
  EXPECT_TRUE(a.IsInternal());
  EXPECT_EQ(a.message(), "self");
  a = std::move(a);  // guarded by this != &o — still Internal
  EXPECT_TRUE(a.IsInternal());
  EXPECT_EQ(a.message(), "self");
}

TEST(Status, LongMessageToStringAndView) {
  std::string long_msg(400, 'x');
  Status s = Status::DataLoss(long_msg);
  EXPECT_TRUE(s.has_message());
  EXPECT_EQ(s.message(), long_msg);
  char tiny[8];
  std::size_t need = s.ToString(tiny, sizeof(tiny));
  // "DataLoss: " (10) + 400
  EXPECT_EQ(need, 10u + 400u);
  EXPECT_EQ(std::strlen(tiny), 7u);  // truncated + NUL
  EXPECT_EQ(s.ToString(nullptr, 0), need);
}

TEST(Status, EmptyMessageStringViewIsEmpty) {
  Status s = Status::Cancelled();  // no message
  EXPECT_FALSE(s.has_message());
  EXPECT_TRUE(s.message().empty());
  EXPECT_EQ(to_str(s), "Cancelled");
}

TEST(Status, OomDegradesToInlineCodeWithoutMessage) {
  // Force status_make_heap to fail: code must survive, message dropped.
  hyle::detail::status_test_force_oom(true);
  Status s = Status::Internal("this message is dropped under forced OOM");
  hyle::detail::status_test_force_oom(false);

  EXPECT_TRUE(s.IsInternal());
  EXPECT_FALSE(s.has_message());
  EXPECT_TRUE(s.message().empty());
  EXPECT_EQ(to_str(s), "Internal");
}

TEST(Status, UnknownCodeNameFallback) {
  // Out-of-table code (not produced by factories) still names safely.
  auto bogus = static_cast<StatusCode>(999);
  EXPECT_EQ(status_code_name(bogus), "Unknown");
}

TEST(Result, TransformSkipsOnError) {
  Result<int> r = std::unexpected(Status::Aborted("no"));
  bool ran = false;
  Result<int> out = r.transform([&](int v) {
    ran = true;
    return v + 1;
  });
  EXPECT_FALSE(ran);
  ASSERT_FALSE(out.has_value());
  EXPECT_TRUE(out.error().IsAborted());
}

TEST(Result, VoidOkAndError) {
  Result<void> ok{};
  EXPECT_TRUE(ok.has_value());
  Result<void> err = std::unexpected(Status::Unavailable("down"));
  ASSERT_FALSE(err.has_value());
  EXPECT_TRUE(err.error().IsUnavailable());
}
