#include "hyle/obs/log.h"
#include "hyle/obs/trace.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct ThrowingFormatValue {};

template <> struct std::formatter<ThrowingFormatValue, char> {
  constexpr auto parse(std::format_parse_context &context) {
    return context.begin();
  }

  auto format(const ThrowingFormatValue &, std::format_context &) const
      -> std::format_context::iterator {
    throw std::runtime_error("formatter failure");
  }
};

using namespace hyle::obs;

namespace {

// Constructed before main, so it runs after ordinary function-local statics
// during process teardown. The test executable itself is the assertion: a
// process-lifetime registry remains valid for this late access.
struct LateRegistryAccess {
  ~LateRegistryAccess() { (void)sink_count(); }
} g_late_registry_access;

// A capture sink: records (level, msg) into a vector so tests assert on what
// was emitted, without touching stderr. One instance per test; registered as a
// sink via its {this, &write} pair.
struct Capture {
  struct Rec {
    LogLevel level;
    std::string msg;
  };
  std::vector<Rec> recs;

  LogSink as_sink() { return LogSink{this, &write}; }
  static void write(void *state, const LogRecord &record) noexcept {
    static_cast<Capture *>(state)->recs.push_back(
        {record.level, std::string(record.message)});
  }
};

// Each test runs with ONLY a capture sink installed (default stderr sink
// removed) and a known level, then restores global state on teardown.
class LogTest : public ::testing::Test {
protected:
  Capture cap;
  void SetUp() override {
    ASSERT_TRUE(remove_sink(default_sink()).ok()); // silence console
    ASSERT_TRUE(add_sink(cap.as_sink()).ok());
    set_level(LogLevel::Trace);
  }
  void TearDown() override {
    EXPECT_TRUE(remove_sink(cap.as_sink()).ok());
    set_level(LogLevel::Info);                  // restore default
    EXPECT_TRUE(add_sink(default_sink()).ok()); // restore console
  }
};

} // namespace

TEST_F(LogTest, MessageReachesSink) {
  log_message(LogLevel::Info, "hello");
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].level, LogLevel::Info);
  EXPECT_EQ(cap.recs[0].msg, "hello");
}

TEST_F(LogTest, MessageViewPreservesEmbeddedNullWithoutCopy) {
  constexpr char bytes[] = {'a', '\0', 'b'};
  EXPECT_TRUE(
      log_message(LogLevel::Info, std::string_view(bytes, sizeof(bytes))));
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg.size(), sizeof(bytes));
  EXPECT_EQ(cap.recs[0].msg[0], 'a');
  EXPECT_EQ(cap.recs[0].msg[1], '\0');
  EXPECT_EQ(cap.recs[0].msg[2], 'b');
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

TEST_F(LogTest, LogFmtTruncatesAtFixedBufferBoundary) {
  HYLE_LOG_INFO("{}", std::string(2048, 'x'));
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg.size(), 1023u);
  EXPECT_TRUE(cap.recs[0].msg.ends_with("...[truncated]"));
}

TEST_F(LogTest, MacroCapturesCallerSourceLocation) {
  struct SourceCapture {
    std::string file;
    std::string function;
    std::uint32_t line = 0;
    static void write(void *state, const LogRecord &record) noexcept {
      auto *capture = static_cast<SourceCapture *>(state);
      capture->file = record.file != nullptr ? record.file : "";
      capture->function = record.function != nullptr ? record.function : "";
      capture->line = record.line;
    }
  } source;
  LogSink sink{&source, &SourceCapture::write};
  ASSERT_TRUE(add_sink(sink).ok());
  const std::uint32_t expected_line = __LINE__ + 1;
  HYLE_LOG_INFO("source");
  EXPECT_TRUE(remove_sink(sink).ok());
  EXPECT_NE(source.file.find("log_test.cc"), std::string::npos);
  EXPECT_NE(source.function.find("MacroCapturesCallerSourceLocation"),
            std::string::npos);
  EXPECT_EQ(source.line, expected_line);
}

TEST_F(LogTest, StructuredRecordCarriesFieldsAndSource) {
  constexpr char exact_key[] = {'r', 'e', 'q', '\0', 'i', 'd'};
  constexpr char exact_value[] = {'a', 'b', 'c', '\0', 'x'};
  LogField fields[] = {
      {std::string_view(exact_key, sizeof(exact_key)),
       std::string_view(exact_value, sizeof(exact_value))},
      {"peer", "127.0.0.1"},
  };
  struct StructuredCapture {
    std::size_t fields = 0;
    std::string first_key;
    std::string first_value;
    std::string file;
    std::uint32_t line = 0;
    std::uint64_t timestamp = 0;
    static void write(void *state, const LogRecord &record) noexcept {
      auto *capture = static_cast<StructuredCapture *>(state);
      capture->fields = record.fields.size();
      capture->first_key.assign(record.fields.front().key);
      capture->first_value.assign(record.fields.front().value);
      capture->file = record.file != nullptr ? record.file : "";
      capture->line = record.line;
      capture->timestamp = record.timestamp_ns;
    }
  } structured;
  LogSink sink{&structured, &StructuredCapture::write};
  ASSERT_TRUE(add_sink(sink).ok());
  LogRecord record{
      .level = LogLevel::Warn,
      .message = "slow request",
      .fields = fields,
      .file = "server.cc",
      .function = "serve",
      .line = 42,
      .timestamp_ns = 123,
  };
  EXPECT_TRUE(log_record(record));
  EXPECT_TRUE(remove_sink(sink).ok());
  EXPECT_EQ(structured.fields, 2u);
  EXPECT_EQ(structured.first_key, std::string(exact_key, sizeof(exact_key)));
  EXPECT_EQ(structured.first_value,
            std::string(exact_value, sizeof(exact_value)));
  EXPECT_EQ(structured.file, "server.cc");
  EXPECT_EQ(structured.line, 42u);
  EXPECT_EQ(structured.timestamp, 123u);
}

TEST_F(LogTest, LogFmtSkipsFormattingWhenGated) {
  set_level(LogLevel::Error);
  // If this formatted despite the gate, a throwing formatter would fire; the
  // point is it must NOT format. Use a value that's cheap; assert nothing
  // logged.
  log_fmt(LogLevel::Debug, "expensive {}", 123);
  EXPECT_TRUE(cap.recs.empty());
}

TEST_F(LogTest, LogFmtContainsForeignFormatterExceptions) {
  EXPECT_NO_THROW(HYLE_LOG_INFO("value={}", ThrowingFormatValue{}));
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg, "[log formatting failed]");
}

TEST_F(LogTest, MacrosRouteToLevels) {
  HYLE_LOG_WARN("w{}", 1);
  HYLE_LOG_ERROR("e{}", 2);
  ASSERT_EQ(cap.recs.size(), 2u);
  EXPECT_EQ(cap.recs[0].level, LogLevel::Warn);
  EXPECT_EQ(cap.recs[0].msg, "w1");
  EXPECT_EQ(cap.recs[1].level, LogLevel::Error);
}

TEST_F(LogTest, ConditionalMacroShortCircuitsFormatting) {
  int conditions = 0;
  int formatted = 0;
  HYLE_LOG_INFO_IF(++conditions == 2, "value={}", ++formatted);
  HYLE_LOG_INFO_IF(++conditions == 2, "value={}", ++formatted);
  EXPECT_EQ(conditions, 2);
  EXPECT_EQ(formatted, 1);
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg, "value=1");
}

TEST_F(LogTest, RuntimeLevelPrecedesEveryControlArgument) {
  set_level(LogLevel::Error);
  int evaluated = 0;
  HYLE_LOG_INFO_IF(++evaluated != 0, "{}", ++evaluated);
  HYLE_LOG_INFO_EVERY_N(++evaluated, "{}", ++evaluated);
  HYLE_LOG_INFO_EVERY_N_EXACT(++evaluated, "{}", ++evaluated);
  HYLE_LOG_INFO_FIRST_N(++evaluated, "{}", ++evaluated);
  HYLE_LOG_INFO_EVERY_POW_2("{}", ++evaluated);
  HYLE_LOG_INFO_EVERY((++evaluated, std::chrono::nanoseconds(1)), "{}",
                      ++evaluated);
  EXPECT_EQ(evaluated, 0);
  EXPECT_TRUE(cap.recs.empty());
}

TEST_F(LogTest, CallsiteCountMacrosUseTheirDocumentedSequences) {
  int formatted = 0;
  for (int i = 0; i < 8; ++i)
    HYLE_LOG_INFO_EVERY_N(3, "approx={}", ++formatted);
  EXPECT_EQ(formatted, 3);
  EXPECT_EQ(cap.recs.size(), 3u);

  cap.recs.clear();
  formatted = 0;
  for (int i = 0; i < 8; ++i)
    HYLE_LOG_INFO_EVERY_N_EXACT(3, "exact={}", ++formatted);
  EXPECT_EQ(formatted, 3);
  EXPECT_EQ(cap.recs.size(), 3u);

  cap.recs.clear();
  formatted = 0;
  for (int i = 0; i < 8; ++i)
    HYLE_LOG_INFO_FIRST_N(3, "first={}", ++formatted);
  EXPECT_EQ(formatted, 3);
  EXPECT_EQ(cap.recs.size(), 3u);

  cap.recs.clear();
  formatted = 0;
  for (int i = 0; i < 8; ++i)
    HYLE_LOG_INFO_EVERY_POW_2("pow2={}", ++formatted);
  EXPECT_EQ(formatted, 4);
  EXPECT_EQ(cap.recs.size(), 4u);
}

TEST_F(LogTest, CallsiteIntervalMacroLogsFirstAndSuppressesTheBurst) {
  int formatted = 0;
  for (int i = 0; i < 8; ++i)
    HYLE_LOG_INFO_EVERY(std::chrono::hours(1), "interval={}", ++formatted);
  EXPECT_EQ(formatted, 1);
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg, "interval=1");
}

TEST(LogControlState, CountStatesHandleZeroResetMoveAndSequences) {
  LogEveryNState approximate;
  EXPECT_FALSE(approximate.should_log(0));
  EXPECT_FALSE(approximate.should_log(-1));
  EXPECT_EQ(approximate.count(), 0u);
  for (std::uint64_t i = 0; i < 8; ++i)
    EXPECT_EQ(approximate.should_log(3), i % 3 == 0);
  EXPECT_EQ(approximate.count(), 8u);

  LogEveryNState moved(std::move(approximate));
  EXPECT_EQ(moved.count(), 8u);
  EXPECT_EQ(approximate.count(), 0u);
  moved.reset();
  EXPECT_TRUE(moved.should_log(3));

  LogEveryNExactState exact;
  for (std::uint64_t i = 0; i < 8; ++i)
    EXPECT_EQ(exact.should_log(3), i % 3 == 0);

  LogFirstNState first;
  EXPECT_TRUE(first.should_log(3));
  EXPECT_TRUE(first.should_log(3));
  EXPECT_TRUE(first.should_log(3));
  EXPECT_FALSE(first.should_log(3));
  EXPECT_EQ(first.count(), 3u);

  LogEveryPow2State powers;
  for (std::uint64_t i = 1; i <= 8; ++i) {
    const bool expected = (i & (i - 1)) == 0;
    EXPECT_EQ(powers.should_log(), expected);
  }
}

TEST(LogControlState, ExactStatesEnforceConcurrentCardinality) {
  constexpr std::size_t kThreads = 8;
  constexpr std::size_t kChecks = 2000;
  constexpr std::uint64_t kPeriod = 17;
  LogEveryNExactState every;
  LogFirstNState first;
  LogIntervalLimiter interval;
  std::atomic<std::size_t> every_admitted{0};
  std::atomic<std::size_t> first_admitted{0};
  std::atomic<std::size_t> interval_admitted{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&] {
      for (std::size_t i = 0; i < kChecks; ++i) {
        if (every.should_log(kPeriod))
          every_admitted.fetch_add(1, std::memory_order_relaxed);
        if (first.should_log(113))
          first_admitted.fetch_add(1, std::memory_order_relaxed);
        if (interval.should_log_at(10'000, 1'000))
          interval_admitted.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();

  constexpr std::size_t kTotal = kThreads * kChecks;
  EXPECT_EQ(every_admitted.load(std::memory_order_relaxed),
            (kTotal - 1) / kPeriod + 1);
  EXPECT_EQ(first_admitted.load(std::memory_order_relaxed), 113u);
  EXPECT_EQ(interval_admitted.load(std::memory_order_relaxed), 1u);
}

TEST(LogControlState, IntervalSupportsInjectedTimeBoundariesAndReset) {
  LogIntervalLimiter limiter;
  EXPECT_TRUE(limiter.should_log_at(100, 10));
  EXPECT_FALSE(limiter.should_log_at(109, 10));
  EXPECT_TRUE(limiter.should_log_at(110, 10));
  EXPECT_FALSE(limiter.should_log_at(50, 10));
  EXPECT_TRUE(limiter.should_log_at(110, 0));

  limiter.reset();
  EXPECT_TRUE(limiter.should_log_at(0, 10));
  EXPECT_FALSE(limiter.should_log_at(9, 10));
  EXPECT_TRUE(limiter.should_log_at(10, 10));

  const auto origin = std::chrono::steady_clock::time_point{};
  LogIntervalLimiter typed;
  EXPECT_TRUE(typed.should_log_at(origin + std::chrono::nanoseconds(100),
                                  std::chrono::nanoseconds(10)));
  EXPECT_FALSE(typed.should_log_at(origin + std::chrono::nanoseconds(109),
                                   std::chrono::nanoseconds(10)));
  EXPECT_TRUE(typed.should_log_at(origin + std::chrono::nanoseconds(110),
                                  std::chrono::nanoseconds(10)));
}

TEST(LogControlState, IntervalDoesNotDuplicateAtUnsignedMaximum) {
  LogIntervalLimiter limiter;
  constexpr std::uint64_t kMax = (std::numeric_limits<std::uint64_t>::max)();
  EXPECT_TRUE(limiter.should_log_at(kMax, 1));
  EXPECT_FALSE(limiter.should_log_at(kMax, 1));
}

TEST_F(LogTest, MultipleSinksAllReceive) {
  Capture second;
  ASSERT_TRUE(add_sink(second.as_sink()).ok());
  log_message(LogLevel::Info, "broadcast");
  EXPECT_TRUE(remove_sink(second.as_sink()).ok());
  EXPECT_EQ(cap.recs.size(), 1u);
  ASSERT_EQ(second.recs.size(), 1u);
  EXPECT_EQ(second.recs[0].msg, "broadcast");
}

TEST_F(LogTest, SinkRegistrationOwnsMembershipAndMoves) {
  Capture scoped;
  {
    auto result = SinkRegistration::create(scoped.as_sink());
    ASSERT_TRUE(result.has_value());
    SinkRegistration registration = std::move(*result);
    EXPECT_FALSE(result->registered());
    EXPECT_TRUE(registration.registered());
    EXPECT_TRUE(log_message(LogLevel::Info, "scoped"));
    ASSERT_EQ(scoped.recs.size(), 1u);
  }

  EXPECT_EQ(remove_sink(scoped.as_sink()).code(), hyle::StatusCode::NotFound);
}

TEST_F(LogTest, SinkRegistrationResetIsIdempotent) {
  Capture scoped;
  auto result = SinkRegistration::create(scoped.as_sink());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->reset().ok());
  EXPECT_TRUE(result->reset().ok());
  EXPECT_FALSE(result->registered());
}

TEST_F(LogTest, TypedSinkAdapterWritesFlushesAndOwnsMembership) {
  struct Backend {
    std::size_t writes = 0;
    std::size_t flushes = 0;
    void write(const LogRecord &) noexcept { ++writes; }
    void flush() noexcept { ++flushes; }
  } backend;

  auto result = SinkRegistration::create(backend);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(log_message(LogLevel::Info, "typed"));
  EXPECT_EQ(flush_sinks(), 1u);
  EXPECT_EQ(backend.writes, 1u);
  EXPECT_EQ(backend.flushes, 1u);
}

TEST_F(LogTest, SinkRegistrationMoveAssignmentReleasesLiveTarget) {
  Capture first_capture;
  Capture second_capture;
  auto first = SinkRegistration::create(first_capture.as_sink());
  auto second = SinkRegistration::create(second_capture.as_sink());
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());

  *first = std::move(*second);
  EXPECT_TRUE(first->registered());
  EXPECT_FALSE(second->registered());
  EXPECT_EQ(remove_sink(first_capture.as_sink()).code(),
            hyle::StatusCode::NotFound);
  EXPECT_TRUE(log_message(LogLevel::Info, "after move assignment"));
  EXPECT_TRUE(first_capture.recs.empty());
  EXPECT_EQ(second_capture.recs.size(), 1u);
}

TEST_F(LogTest, SinkRegistrationCreateReportsDuplicate) {
  auto duplicate = SinkRegistration::create(cap.as_sink());
  ASSERT_FALSE(duplicate.has_value());
  EXPECT_EQ(duplicate.error().code(), hyle::StatusCode::AlreadyExists);
}

TEST_F(LogTest, DuplicateSinkIsRejectedWithoutDuplicateDelivery) {
  EXPECT_EQ(add_sink(cap.as_sink()).code(), hyle::StatusCode::AlreadyExists);
  log_message(LogLevel::Info, "once");
  EXPECT_EQ(cap.recs.size(), 1u); // not 2
}

TEST_F(LogTest, RemoveSinkStopsDelivery) {
  ASSERT_TRUE(remove_sink(cap.as_sink()).ok());
  log_message(LogLevel::Info, "gone");
  EXPECT_TRUE(cap.recs.empty());
  ASSERT_TRUE(add_sink(cap.as_sink()).ok()); // balance TearDown
}

// Default level is Info: a fresh process drops Trace/Debug. Verified by reading
// the global default (this test does not depend on SetUp's Trace override).
TEST(LogDefault, DefaultLevelIsInfo) {
  set_level(LogLevel::Info);
  EXPECT_FALSE(should_log(LogLevel::Trace));
  EXPECT_FALSE(should_log(LogLevel::Debug));
  EXPECT_TRUE(should_log(LogLevel::Info));
}

// Concurrency stress: many threads log_message in parallel while another churns
// add_sink/remove_sink. Validates immutable snapshots and quiescent retirement
// stay race-free -- the hard assertion is "no crash, no TSan race, no hang,
// every logger's message delivered".
TEST(LogConcurrency, ParallelLogVsChurn) {
  set_level(LogLevel::Trace);
  std::atomic<std::uint64_t> calls{0};
  struct Counter {
    std::atomic<std::uint64_t> *c;
  } counter{&calls};
  LogSink counting{&counter, [](void *s, const LogRecord &) noexcept {
                     static_cast<Counter *>(s)->c->fetch_add(
                         1, std::memory_order_relaxed);
                   }};
  ASSERT_TRUE(remove_sink(default_sink()).ok());
  ASSERT_TRUE(add_sink(counting).ok());

  constexpr int kLoggers = 6;
  constexpr int kPerThread = 5000;
  std::atomic<bool> stop{false};
  std::atomic<bool> registry_ok{true};
  std::vector<std::thread> ts;
  for (int t = 0; t < kLoggers; ++t)
    ts.emplace_back([&] {
      for (int i = 0; i < kPerThread; ++i)
        log_message(LogLevel::Info, "x");
    });
  // Churn thread: repeatedly add/remove a second sink to republish snapshots.
  Counter c2{&calls};
  LogSink churn{&c2, counting.write};
  std::thread churner([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      hyle::Status status = add_sink(churn);
      if (!status.ok() || !remove_sink(churn).ok()) {
        registry_ok.store(false, std::memory_order_relaxed);
        break;
      }
    }
  });

  for (auto &th : ts)
    th.join();
  stop.store(true, std::memory_order_relaxed);
  churner.join();

  // At least the counting sink saw every message from every logger (the churn
  // sink may have added more; we only assert a sane lower bound + no
  // crash/hang).
  EXPECT_GE(calls.load(), static_cast<std::uint64_t>(kLoggers) * kPerThread);
  EXPECT_TRUE(registry_ok.load(std::memory_order_relaxed));

  EXPECT_TRUE(remove_sink(counting).ok());
  set_level(LogLevel::Info);
  EXPECT_TRUE(add_sink(default_sink()).ok());
}

TEST(LogConcurrency, RemoveWaitsForConcurrentCallback) {
  ASSERT_TRUE(remove_sink(default_sink()).ok());
  set_level(LogLevel::Trace);
  struct Blocking {
    std::mutex mutex;
    std::condition_variable entered_cv;
    std::condition_variable release_cv;
    bool entered = false;
    bool release = false;
    static void write(void *state, const LogRecord &) noexcept {
      auto *blocking = static_cast<Blocking *>(state);
      std::unique_lock<std::mutex> lock(blocking->mutex);
      blocking->entered = true;
      blocking->entered_cv.notify_one();
      blocking->release_cv.wait(lock, [&] { return blocking->release; });
    }
  } blocking;
  LogSink sink{&blocking, &Blocking::write};
  ASSERT_TRUE(add_sink(sink).ok());

  std::thread logger([&] { log_message(LogLevel::Info, "blocked"); });
  {
    std::unique_lock<std::mutex> lock(blocking.mutex);
    blocking.entered_cv.wait(lock, [&] { return blocking.entered; });
  }
  std::atomic<bool> removed{false};
  std::thread remover([&] {
    EXPECT_TRUE(remove_sink(sink).ok());
    removed.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(removed.load(std::memory_order_acquire));
  {
    std::lock_guard<std::mutex> lock(blocking.mutex);
    blocking.release = true;
  }
  blocking.release_cv.notify_one();
  logger.join();
  remover.join();
  EXPECT_TRUE(removed.load(std::memory_order_acquire));
  EXPECT_TRUE(add_sink(default_sink()).ok());
  set_level(LogLevel::Info);
}

TEST(LogConcurrency, CallbackCanRemoveItselfWithoutDeadlock) {
  ASSERT_TRUE(remove_sink(default_sink()).ok());
  set_level(LogLevel::Trace);
  struct SelfRemoving {
    LogSink sink{};
    bool removed = false;
    static void write(void *state, const LogRecord &) noexcept {
      auto *self = static_cast<SelfRemoving *>(state);
      self->removed = remove_sink(self->sink).ok();
    }
  } state;
  state.sink = LogSink{&state, &SelfRemoving::write};
  ASSERT_TRUE(add_sink(state.sink).ok());
  EXPECT_TRUE(log_message(LogLevel::Info, "remove"));
  EXPECT_TRUE(state.removed);
  EXPECT_EQ(remove_sink(state.sink).code(), hyle::StatusCode::NotFound);
  EXPECT_TRUE(add_sink(default_sink()).ok());
  set_level(LogLevel::Info);
}

TEST(LogConcurrency, CallbackCannotCrossRemovePendingSink) {
  ASSERT_TRUE(remove_sink(default_sink()).ok());
  set_level(LogLevel::Trace);
  struct CrossRemoving {
    LogSink target{};
    bool result = true;
    static void write(void *state, const LogRecord &) noexcept {
      auto *self = static_cast<CrossRemoving *>(state);
      self->result = remove_sink(self->target).ok();
    }
  } remover;
  std::atomic<std::size_t> target_calls{0};
  LogSink first{&remover, &CrossRemoving::write};
  LogSink target{&target_calls, [](void *state, const LogRecord &) noexcept {
                   static_cast<std::atomic<std::size_t> *>(state)->fetch_add(
                       1, std::memory_order_relaxed);
                 }};
  remover.target = target;
  ASSERT_TRUE(add_sink(first).ok());
  ASSERT_TRUE(add_sink(target).ok());

  EXPECT_TRUE(log_message(LogLevel::Info, "cross-remove"));
  EXPECT_FALSE(remover.result);
  EXPECT_EQ(target_calls.load(std::memory_order_relaxed), 1u);
  EXPECT_TRUE(remove_sink(first).ok());
  EXPECT_TRUE(remove_sink(target).ok());
  EXPECT_TRUE(add_sink(default_sink()).ok());
  set_level(LogLevel::Info);
}

TEST_F(LogTest, NestedLogFromSinkDoesNotReenterSinkGraph) {
  struct Recursive {
    std::size_t calls = 0;
    bool nested_delivered = true;
    static void write(void *state, const LogRecord &) noexcept {
      auto *self = static_cast<Recursive *>(state);
      ++self->calls;
      if (self->calls < 32) {
        self->nested_delivered =
            log_message(LogLevel::Warn, "nested from sink");
      }
    }
  } recursive;
  LogSink sink{&recursive, &Recursive::write};
  ASSERT_TRUE(add_sink(sink).ok());

  EXPECT_TRUE(log_message(LogLevel::Info, "outer"));
  EXPECT_EQ(recursive.calls, 1u);
  EXPECT_FALSE(
      recursive.nested_delivered); // default sink is disabled in fixture
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg, "outer");
  EXPECT_TRUE(remove_sink(sink).ok());
}

TEST_F(LogTest, FlushInvokesOptionalCallbackWithoutNestedFanout) {
  struct Buffered {
    std::size_t writes = 0;
    std::size_t flushes = 0;
    static void write(void *state, const LogRecord &) noexcept {
      ++static_cast<Buffered *>(state)->writes;
    }
    static void flush(void *state) noexcept {
      auto *self = static_cast<Buffered *>(state);
      ++self->flushes;
      (void)log_message(LogLevel::Debug, "nested during flush");
    }
  } buffered;
  LogSink sink{&buffered, &Buffered::write, &Buffered::flush};
  ASSERT_TRUE(add_sink(sink).ok());

  EXPECT_EQ(flush_sinks(), 1u);
  EXPECT_EQ(buffered.flushes, 1u);
  EXPECT_EQ(buffered.writes, 0u);
  EXPECT_TRUE(cap.recs.empty());
  EXPECT_TRUE(remove_sink(sink).ok());
}

TEST_F(LogTest, FlushCallbackCanRemoveItselfAndNestedFlushIsSuppressed) {
  struct SelfRemoving {
    LogSink sink{};
    bool removed = false;
    std::size_t nested_flushes = 1;
    static void write(void *, const LogRecord &) noexcept {}
    static void flush(void *state) noexcept {
      auto *self = static_cast<SelfRemoving *>(state);
      self->nested_flushes = flush_sinks();
      self->removed = remove_sink(self->sink).ok();
    }
  } state;
  state.sink = LogSink{&state, &SelfRemoving::write, &SelfRemoving::flush};
  ASSERT_TRUE(add_sink(state.sink).ok());

  EXPECT_EQ(flush_sinks(), 1u);
  EXPECT_EQ(state.nested_flushes, 0u);
  EXPECT_TRUE(state.removed);
  EXPECT_EQ(remove_sink(state.sink).code(), hyle::StatusCode::NotFound);
}

TEST(LogConcurrency, RemoveWaitsForConcurrentFlushCallback) {
  ASSERT_TRUE(remove_sink(default_sink()).ok());
  struct BlockingFlush {
    std::mutex mutex;
    std::condition_variable entered_cv;
    std::condition_variable release_cv;
    bool entered = false;
    bool release = false;
    static void write(void *, const LogRecord &) noexcept {}
    static void flush(void *state) noexcept {
      auto *self = static_cast<BlockingFlush *>(state);
      std::unique_lock<std::mutex> lock(self->mutex);
      self->entered = true;
      self->entered_cv.notify_one();
      self->release_cv.wait(lock, [&] { return self->release; });
    }
  } blocking;
  LogSink sink{&blocking, &BlockingFlush::write, &BlockingFlush::flush};
  ASSERT_TRUE(add_sink(sink).ok());

  std::thread flusher([] { (void)flush_sinks(); });
  {
    std::unique_lock<std::mutex> lock(blocking.mutex);
    blocking.entered_cv.wait(lock, [&] { return blocking.entered; });
  }
  std::atomic<bool> removed{false};
  std::thread remover([&] {
    EXPECT_TRUE(remove_sink(sink).ok());
    removed.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(removed.load(std::memory_order_acquire));
  {
    std::lock_guard<std::mutex> lock(blocking.mutex);
    blocking.release = true;
  }
  blocking.release_cv.notify_one();
  flusher.join();
  remover.join();
  EXPECT_TRUE(removed.load(std::memory_order_acquire));
  EXPECT_TRUE(add_sink(default_sink()).ok());
}

TEST(LogRegistry, ReportsCapacityAndRejectsOverflow) {
  ASSERT_TRUE(remove_sink(default_sink()).ok());
  auto noop = +[](void *, const LogRecord &) noexcept {};
  std::vector<std::uintptr_t> identities(sink_capacity());
  for (std::size_t i = 0; i < identities.size(); ++i) {
    EXPECT_TRUE(add_sink(LogSink{&identities[i], noop}).ok());
  }
  EXPECT_EQ(sink_count(), sink_capacity());
  std::uintptr_t overflow = 0;
  EXPECT_EQ(add_sink(LogSink{&overflow, noop}).code(),
            hyle::StatusCode::ResourceExhausted);
  for (std::size_t i = 0; i < identities.size(); ++i) {
    EXPECT_TRUE(remove_sink(LogSink{&identities[i], noop}).ok());
  }
  EXPECT_TRUE(add_sink(default_sink()).ok());
}

TEST(LogLevelValidation, OffIsThresholdOnly) {
  set_level(LogLevel::Trace);
  EXPECT_FALSE(should_log(LogLevel::Off));
  EXPECT_FALSE(log_message(LogLevel::Off, "not a record"));
  set_level(LogLevel::Info);
}

TEST(LogContracts, InvalidNativeInputsFailFast) {
  EXPECT_DEATH(set_level(static_cast<LogLevel>(99)), "");
  EXPECT_DEATH(log_message(LogLevel::Info, static_cast<const char *>(nullptr)),
               "");
}

TEST(LogContracts, NullSinkCallbackIsARecoverableInvalidArgument) {
  EXPECT_EQ(add_sink(LogSink{}).code(), hyle::StatusCode::InvalidArgument);
  EXPECT_EQ(remove_sink(LogSink{}).code(), hyle::StatusCode::InvalidArgument);
}

TEST(LogDefaultSink, StateSetterIsIdempotentAndReportsSuccess) {
  EXPECT_TRUE(set_default_sink_enabled(false).ok());
  EXPECT_TRUE(set_default_sink_enabled(false).ok());
  EXPECT_FALSE(default_sink_enabled());
  EXPECT_TRUE(set_default_sink_enabled(true).ok());
  EXPECT_TRUE(set_default_sink_enabled(true).ok());
  EXPECT_TRUE(default_sink_enabled());
}

TEST(LogDefaultSink, EscapesControlsAndKeepsOnePhysicalRecord) {
  ASSERT_TRUE(set_default_sink_enabled(true).ok());
  set_level(LogLevel::Info);
  constexpr char message[] = {'l', 'i', 'n',  'e',    '1', '\n', 'l', 'i', 'n',
                              'e', '2', '\0', '\x1b', '[', '3',  '1', 'm'};
  LogField field{"key\t", "value\r"};
  LogRecord record{
      .level = LogLevel::Info,
      .message = std::string_view(message, sizeof(message)),
      .fields = std::span<const LogField>(&field, 1),
      .file = "source\nfile\x1b",
      .line = 9,
  };

  testing::internal::CaptureStderr();
  EXPECT_TRUE(log_record(record));
  const std::string output = testing::internal::GetCapturedStderr();
  EXPECT_NE(output.find("source\\nfile\\x1b:9"), std::string::npos);
  EXPECT_NE(output.find("line1\\nline2\\x00\\x1b[31m"), std::string::npos);
  EXPECT_NE(output.find("key\\t=value\\r"), std::string::npos);
  EXPECT_EQ(output.find('\x1b'), std::string::npos);
  EXPECT_EQ(output.find('\n'), output.size() - 1);
}

TEST(LogDefaultSink, ConcurrentSameDirectionSettersAreIdempotent) {
  ASSERT_TRUE(set_default_sink_enabled(true).ok());
  std::atomic<bool> all_ok{true};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      if (!set_default_sink_enabled(false).ok()) {
        all_ok.store(false, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_TRUE(all_ok.load(std::memory_order_relaxed));
  EXPECT_FALSE(default_sink_enabled());

  threads.clear();
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      if (!set_default_sink_enabled(true).ok()) {
        all_ok.store(false, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_TRUE(all_ok.load(std::memory_order_relaxed));
  EXPECT_TRUE(default_sink_enabled());
}

#if !defined(HYLE_TRACE_TRACY)
TEST(TraceGate, DisabledMacrosDoNotEvaluateArguments) {
  int evaluated = 0;
  HYLE_TRACE_ZONE();
  HYLE_TRACE_ZONE_NAMED((++evaluated, "zone"));
  HYLE_TRACE_ZONE_TEXT((++evaluated, "zone text"));
  HYLE_TRACE_ZONE_NAME((++evaluated, "zone name"));
  HYLE_TRACE_ZONE_VALUE(++evaluated);
  HYLE_TRACE_ZONE_COLOR(++evaluated);
  HYLE_TRACE_PLOT((++evaluated, "plot"), ++evaluated);
  HYLE_TRACE_COUNTER((++evaluated, "counter"), ++evaluated);
  HYLE_TRACE_MESSAGE((++evaluated, "message"));
  HYLE_TRACE_MESSAGE_LITERAL((++evaluated, "literal"));
  HYLE_TRACE_ALLOC((++evaluated, &evaluated), ++evaluated);
  HYLE_TRACE_FREE((++evaluated, &evaluated));
  HYLE_TRACE_ALLOC_NAMED((++evaluated, &evaluated), ++evaluated,
                         (++evaluated, "pool"));
  HYLE_TRACE_FREE_NAMED((++evaluated, &evaluated), (++evaluated, "pool"));
  HYLE_TRACE_THREAD_NAME((++evaluated, "thread"));
  HYLE_TRACE_FIBER_ENTER((++evaluated, "fiber"));
  HYLE_TRACE_FIBER_LEAVE();
  HYLE_TRACE_FRAME();
  HYLE_TRACE_FRAME_NAMED((++evaluated, "frame"));
  EXPECT_EQ(evaluated, 0);
}
#else
TEST(TraceBackend, CurrentZoneAnnotationsAndFullWidthCounterCompile) {
  HYLE_TRACE_ZONE_NAMED("hyle.obs.test");
  const std::string text = "operator=qwen attention";
  HYLE_TRACE_ZONE_TEXT(std::string_view(text));
  HYLE_TRACE_ZONE_NAME(text);
  HYLE_TRACE_ZONE_VALUE(std::numeric_limits<std::uint64_t>::max());
  HYLE_TRACE_ZONE_COLOR(0x22aaffu);
  HYLE_TRACE_COUNTER("hyle.obs.uint64",
                     std::numeric_limits<std::uint64_t>::max());
  HYLE_TRACE_MESSAGE(std::string_view(text));
}

TEST(TraceBackend, DynamicTextIsBoundedBeforeEnteringTracy) {
  std::string text(hyle::obs::detail::kTraceDynamicTextMax + 17, 'x');
  const std::string_view bounded_view =
      hyle::obs::detail::bounded_trace_text(std::string_view(text));
  EXPECT_EQ(bounded_view.data(), text.data());
  EXPECT_EQ(bounded_view.size(), hyle::obs::detail::kTraceDynamicTextMax);

  text.push_back('\0');
  const std::string_view bounded_c_string =
      hyle::obs::detail::bounded_trace_text(text.c_str());
  EXPECT_EQ(bounded_c_string.data(), text.data());
  EXPECT_EQ(bounded_c_string.size(), hyle::obs::detail::kTraceDynamicTextMax);

  HYLE_TRACE_ZONE_NAMED("hyle.obs.long-text-test");
  HYLE_TRACE_ZONE_TEXT(bounded_view);
  HYLE_TRACE_ZONE_NAME(bounded_view);
  HYLE_TRACE_MESSAGE(bounded_view);
}
#endif
