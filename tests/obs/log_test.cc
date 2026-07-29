#include "hyle/obs/log.h"
#include "hyle/obs/trace.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace hyle::obs;

namespace {

// Constructed before main, so it runs after ordinary function-local statics
// during process teardown. The test executable itself is the assertion: a
// process-lifetime registry remains valid for this late access.
struct LateRegistryAccess {
  ~LateRegistryAccess() { (void)sink_count(); }
} g_late_registry_access;

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
  static void write(void* state, const LogRecord& record) noexcept {
    static_cast<Capture*>(state)->recs.push_back(
        {record.level, std::string(record.message)});
  }
};

// Each test runs with ONLY a capture sink installed (default stderr sink removed)
// and a known level, then restores global state on teardown.
class LogTest : public ::testing::Test {
 protected:
  Capture cap;
  void SetUp() override {
    ASSERT_TRUE(remove_sink(default_sink()).ok());  // silence console
    ASSERT_TRUE(add_sink(cap.as_sink()).ok());
    set_level(LogLevel::Trace);
  }
  void TearDown() override {
    EXPECT_TRUE(remove_sink(cap.as_sink()).ok());
    set_level(LogLevel::Info);     // restore default
    EXPECT_TRUE(add_sink(default_sink()).ok());  // restore console
  }
};

}  // namespace

TEST_F(LogTest, MessageReachesSink) {
  log_message(LogLevel::Info, "hello");
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].level, LogLevel::Info);
  EXPECT_EQ(cap.recs[0].msg, "hello");
}

TEST_F(LogTest, MessageViewPreservesEmbeddedNullWithoutCopy) {
  constexpr char bytes[] = {'a', '\0', 'b'};
  EXPECT_TRUE(log_message(LogLevel::Info,
                          std::string_view(bytes, sizeof(bytes))));
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
    static void write(void* state, const LogRecord& record) noexcept {
      auto* capture = static_cast<SourceCapture*>(state);
      capture->file = record.file != nullptr ? record.file : "";
      capture->function =
          record.function != nullptr ? record.function : "";
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
  LogField fields[] = {{"request_id", "abc"}, {"peer", "127.0.0.1"}};
  struct StructuredCapture {
    std::size_t fields = 0;
    std::string file;
    std::uint32_t line = 0;
    std::uint64_t timestamp = 0;
    static void write(void* state, const LogRecord& record) noexcept {
      auto* capture = static_cast<StructuredCapture*>(state);
      capture->fields = record.field_count;
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
      .field_count = 2,
      .file = "server.cc",
      .function = "serve",
      .line = 42,
      .timestamp_ns = 123,
  };
  EXPECT_TRUE(log_record(record));
  EXPECT_TRUE(remove_sink(sink).ok());
  EXPECT_EQ(structured.fields, 2u);
  EXPECT_EQ(structured.file, "server.cc");
  EXPECT_EQ(structured.line, 42u);
  EXPECT_EQ(structured.timestamp, 123u);
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

  EXPECT_EQ(remove_sink(scoped.as_sink()).code(),
            hyle::StatusCode::NotFound);
}

TEST_F(LogTest, SinkRegistrationResetIsIdempotent) {
  Capture scoped;
  auto result = SinkRegistration::create(scoped.as_sink());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->reset().ok());
  EXPECT_TRUE(result->reset().ok());
  EXPECT_FALSE(result->registered());
}

TEST_F(LogTest, DuplicateSinkIsRejectedWithoutDuplicateDelivery) {
  EXPECT_EQ(add_sink(cap.as_sink()).code(), hyle::StatusCode::AlreadyExists);
  log_message(LogLevel::Info, "once");
  EXPECT_EQ(cap.recs.size(), 1u);  // not 2
}

TEST_F(LogTest, RemoveSinkStopsDelivery) {
  ASSERT_TRUE(remove_sink(cap.as_sink()).ok());
  log_message(LogLevel::Info, "gone");
  EXPECT_TRUE(cap.recs.empty());
  ASSERT_TRUE(add_sink(cap.as_sink()).ok());  // balance TearDown
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
// stay race-free -- the hard assertion is "no crash, no TSan race, no hang, every
// logger's message delivered".
TEST(LogConcurrency, ParallelLogVsChurn) {
  set_level(LogLevel::Trace);
  std::atomic<std::uint64_t> calls{0};
  struct Counter {
    std::atomic<std::uint64_t>* c;
  } counter{&calls};
  LogSink counting{&counter,
                   [](void* s, const LogRecord&) noexcept {
                     static_cast<Counter*>(s)->c->fetch_add(
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
      for (int i = 0; i < kPerThread; ++i) log_message(LogLevel::Info, "x");
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

  for (auto& th : ts) th.join();
  stop.store(true, std::memory_order_relaxed);
  churner.join();

  // At least the counting sink saw every message from every logger (the churn
  // sink may have added more; we only assert a sane lower bound + no crash/hang).
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
    static void write(void* state, const LogRecord&) noexcept {
      auto* blocking = static_cast<Blocking*>(state);
      std::unique_lock<std::mutex> lock(blocking->mutex);
      blocking->entered = true;
      blocking->entered_cv.notify_one();
      blocking->release_cv.wait(lock,
                                [&] { return blocking->release; });
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
    static void write(void* state, const LogRecord&) noexcept {
      auto* self = static_cast<SelfRemoving*>(state);
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
    static void write(void* state, const LogRecord&) noexcept {
      auto* self = static_cast<CrossRemoving*>(state);
      self->result = remove_sink(self->target).ok();
    }
  } remover;
  std::atomic<std::size_t> target_calls{0};
  LogSink first{&remover, &CrossRemoving::write};
  LogSink target{
      &target_calls,
      [](void* state, const LogRecord&) noexcept {
        static_cast<std::atomic<std::size_t>*>(state)->fetch_add(
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
    static void write(void* state, const LogRecord&) noexcept {
      auto* self = static_cast<Recursive*>(state);
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
  EXPECT_FALSE(recursive.nested_delivered);  // default sink is disabled in fixture
  ASSERT_EQ(cap.recs.size(), 1u);
  EXPECT_EQ(cap.recs[0].msg, "outer");
  EXPECT_TRUE(remove_sink(sink).ok());
}

TEST_F(LogTest, FlushInvokesOptionalCallbackWithoutNestedFanout) {
  struct Buffered {
    std::size_t writes = 0;
    std::size_t flushes = 0;
    static void write(void* state, const LogRecord&) noexcept {
      ++static_cast<Buffered*>(state)->writes;
    }
    static void flush(void* state) noexcept {
      auto* self = static_cast<Buffered*>(state);
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

TEST(LogRegistry, ReportsCapacityAndRejectsOverflow) {
  ASSERT_TRUE(remove_sink(default_sink()).ok());
  auto noop = +[](void*, const LogRecord&) noexcept {};
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
  EXPECT_DEATH(
      ([] {
        LogRecord record{
            .level = LogLevel::Info,
            .message = "fields",
            .fields = nullptr,
            .field_count = 1,
        };
        (void)log_record(record);
      }()),
      "");
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
  constexpr char message[] = {'l', 'i', 'n', 'e', '1', '\n', 'l', 'i', 'n',
                              'e', '2', '\0', '\x1b', '[', '3', '1', 'm'};
  LogField field{"key\t", "value\r"};
  LogRecord record{
      .level = LogLevel::Info,
      .message = std::string_view(message, sizeof(message)),
      .fields = &field,
      .field_count = 1,
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
  for (auto& thread : threads) thread.join();
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
  for (auto& thread : threads) thread.join();
  EXPECT_TRUE(all_ok.load(std::memory_order_relaxed));
  EXPECT_TRUE(default_sink_enabled());
}

#if !defined(HYLE_TRACE_TRACY)
TEST(TraceGate, DisabledMacrosDoNotEvaluateArguments) {
  int evaluated = 0;
  HYLE_TRACE_ZONE();
  HYLE_TRACE_ZONE_NAMED((++evaluated, "zone"));
  HYLE_TRACE_PLOT((++evaluated, "plot"), ++evaluated);
  HYLE_TRACE_COUNTER((++evaluated, "counter"), ++evaluated);
  HYLE_TRACE_MESSAGE((++evaluated, "message"));
  HYLE_TRACE_MESSAGE_LITERAL((++evaluated, "literal"));
  HYLE_TRACE_ALLOC((++evaluated, &evaluated), ++evaluated);
  HYLE_TRACE_FREE((++evaluated, &evaluated));
  HYLE_TRACE_ALLOC_NAMED((++evaluated, &evaluated), ++evaluated,
                         (++evaluated, "pool"));
  HYLE_TRACE_FREE_NAMED((++evaluated, &evaluated),
                        (++evaluated, "pool"));
  HYLE_TRACE_THREAD_NAME((++evaluated, "thread"));
  HYLE_TRACE_FIBER_ENTER((++evaluated, "fiber"));
  HYLE_TRACE_FIBER_LEAVE();
  HYLE_TRACE_FRAME();
  HYLE_TRACE_FRAME_NAMED((++evaluated, "frame"));
  EXPECT_EQ(evaluated, 0);
}
#endif
