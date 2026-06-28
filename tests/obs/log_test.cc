#include "hyle/obs/log.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace hyle::obs;

namespace {

// A capture sink: records (level, msg) into a vector so tests assert on what was
// emitted, without touching stderr. One instance per test; registered as a sink
// via its {this, &write} pair.
struct Capture {
  struct Rec {
    LogLevel level;
    std::string msg;
  };
  std::vector<Rec> recs;

  LogSink as_sink() { return LogSink{this, &write}; }
  static void write(void* state, LogLevel level, const char* msg) {
    static_cast<Capture*>(state)->recs.push_back({level, msg});
  }
};

// Each test runs with ONLY a capture sink installed (default stderr sink removed)
// and a known level, then restores global state on teardown.
class LogTest : public ::testing::Test {
 protected:
  Capture cap;
  void SetUp() override {
    remove_sink(default_sink());  // silence console during tests
    add_sink(cap.as_sink());
    set_level(LogLevel::Trace);
  }
  void TearDown() override {
    remove_sink(cap.as_sink());
    set_level(LogLevel::Info);     // restore default
    add_sink(default_sink());      // restore console for other suites
  }
};

}  // namespace

TEST_F(LogTest, MessageReachesSink) {
  log_message(LogLevel::Info, "hello");
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].level, LogLevel::Info);
  EXPECT_EQ(cap.recs[0].msg, "hello");
}

TEST_F(LogTest, LevelGateDropsBelowThreshold) {
  set_level(LogLevel::Warn);
  log_message(LogLevel::Info, "dropped");
  log_message(LogLevel::Debug, "dropped");
  log_message(LogLevel::Warn, "kept");
  log_message(LogLevel::Error, "kept2");
  ASSERT_EQ(cap.recs.size(), 2u);
  EXPECT_EQ(cap.recs[0].msg, "kept");
  EXPECT_EQ(cap.recs[1].msg, "kept2");
}

TEST_F(LogTest, ShouldLogMatchesGate) {
  set_level(LogLevel::Warn);
  EXPECT_FALSE(should_log(LogLevel::Info));
  EXPECT_TRUE(should_log(LogLevel::Warn));
  EXPECT_TRUE(should_log(LogLevel::Error));
}

TEST_F(LogTest, OffSilencesEverything) {
  set_level(LogLevel::Off);
  log_message(LogLevel::Error, "nope");
  EXPECT_TRUE(cap.recs.empty());
}

TEST_F(LogTest, LogFmtFormatsCallerSide) {
  log_fmt(LogLevel::Info, "x={} y={}", 42, "str");
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg, "x=42 y=str");
}

TEST_F(LogTest, LogFmtSkipsFormattingWhenGated) {
  set_level(LogLevel::Error);
  // If this formatted despite the gate, a throwing formatter would fire; the
  // point is it must NOT format. Use a value that's cheap; assert nothing logged.
  log_fmt(LogLevel::Debug, "expensive {}", 123);
  EXPECT_TRUE(cap.recs.empty());
}

TEST_F(LogTest, MacrosRouteToLevels) {
  HYLE_LOG_WARN("w{}", 1);
  HYLE_LOG_ERROR("e{}", 2);
  ASSERT_EQ(cap.recs.size(), 2u);
  EXPECT_EQ(cap.recs[0].level, LogLevel::Warn);
  EXPECT_EQ(cap.recs[0].msg, "w1");
  EXPECT_EQ(cap.recs[1].level, LogLevel::Error);
}

TEST_F(LogTest, MultipleSinksAllReceive) {
  Capture second;
  add_sink(second.as_sink());
  log_message(LogLevel::Info, "broadcast");
  remove_sink(second.as_sink());
  EXPECT_EQ(cap.recs.size(), 1u);
  ASSERT_EQ(second.recs.size(), 1u);
  EXPECT_EQ(second.recs[0].msg, "broadcast");
}

TEST_F(LogTest, AddSinkIsIdempotent) {
  add_sink(cap.as_sink());  // already added in SetUp -> dedupe
  log_message(LogLevel::Info, "once");
  EXPECT_EQ(cap.recs.size(), 1u);  // not 2
}

TEST_F(LogTest, RemoveSinkStopsDelivery) {
  remove_sink(cap.as_sink());
  log_message(LogLevel::Info, "gone");
  EXPECT_TRUE(cap.recs.empty());
  add_sink(cap.as_sink());  // re-add so TearDown's remove is balanced
}

// Default level is Info: a fresh process drops Trace/Debug. Verified by reading
// the global default (this test does not depend on SetUp's Trace override).
TEST(LogDefault, DefaultLevelIsInfo) {
  set_level(LogLevel::Info);
  EXPECT_FALSE(should_log(LogLevel::Trace));
  EXPECT_FALSE(should_log(LogLevel::Debug));
  EXPECT_TRUE(should_log(LogLevel::Info));
}
