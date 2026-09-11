#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <spawn.h>
#  include <sys/wait.h>
#  include <unistd.h>
extern char** environ;
#else
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include "hyle/io/io_context.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/process/process.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

namespace {

using hyle::process::ChildProcess;
using hyle::process::EnvironmentChange;
using hyle::process::EnvironmentMode;
using hyle::process::ExecutableSearch;
using hyle::process::ProcessExitKind;
using hyle::process::ProcessRuntime;
using hyle::process::ProcessStartOptions;
using hyle::process::ProcessStdio;
using hyle::process::ProcessWaitOptions;

constexpr std::string_view kHelper = HYLE_PROCESS_TEST_HELPER_PATH;

ProcessStartOptions helper_options(
    std::span<const std::string_view> arguments) {
  ProcessStartOptions options;
  options.executable = kHelper;
  options.arguments = arguments;
  options.stdin_stream = ProcessStdio::null();
  options.stdout_stream = ProcessStdio::null();
  options.stderr_stream = ProcessStdio::null();
  return options;
}

class TempDirectory {
 public:
  TempDirectory() {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    path_ = std::filesystem::temp_directory_path() /
            ("hyle-process-" + std::to_string(nonce) + "-" +
             std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    valid_ = std::filesystem::create_directory(path_);
  }

  ~TempDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  bool valid() const noexcept { return valid_; }
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
  bool valid_ = false;
};

struct FailAfterAllocator {
  std::size_t successful = 0;
  std::size_t fail_after = 0;
  std::size_t live = 0;

  static void* allocate(void* context, std::size_t size,
                        std::size_t alignment) noexcept {
    auto& self = *static_cast<FailAfterAllocator*>(context);
    if (self.successful >= self.fail_after) return nullptr;
    void* pointer =
        hyle::mem::system_allocator().allocate(size, alignment);
    if (pointer != nullptr) {
      ++self.successful;
      ++self.live;
    }
    return pointer;
  }

  static void deallocate(void* context, void* pointer, std::size_t size,
                         std::size_t alignment) noexcept {
    auto& self = *static_cast<FailAfterAllocator*>(context);
    if (pointer != nullptr) --self.live;
    hyle::mem::system_allocator().deallocate(pointer, size, alignment);
  }

  static void* reallocate(void* context, void* pointer,
                          std::size_t old_size, std::size_t new_size,
                          std::size_t alignment) noexcept {
    auto& self = *static_cast<FailAfterAllocator*>(context);
    if (pointer == nullptr) return allocate(context, new_size, alignment);
    if (new_size == 0) {
      deallocate(context, pointer, old_size, alignment);
      return nullptr;
    }
    if (self.successful >= self.fail_after) return nullptr;
    void* replacement = hyle::mem::system_allocator().reallocate(
        pointer, old_size, new_size, alignment);
    if (replacement != nullptr) ++self.successful;
    return replacement;
  }

  hyle::mem::AllocatorRef get() noexcept {
    return {&allocate, &deallocate, &reallocate, this};
  }
};

struct Drive {
  hyle::sched::Task<int> task;
};

void drive_entry(void* raw) noexcept { static_cast<Drive*>(raw)->task.resume(); }

struct PipeResult {
  std::atomic<bool> done{false};
  bool ok = false;
  bool initial_ok = false;
  bool kill_ok = false;
  bool exit_ok = false;
  bool eof_ok = false;
  bool eof_seen = false;
  hyle::StatusCode eof_status_code = hyle::StatusCode::Ok;
  std::int64_t eof_native_code = 0;
  std::string eof_message;
};

struct LargePipeResult {
  bool write_ok = false;
  bool read_ok = false;
  bool wait_ok = false;
  std::array<char, 12> error{};
  std::size_t written = 0;
  std::size_t read = 0;
  hyle::StatusCode write_code = hyle::StatusCode::Ok;
  hyle::StatusCode read_code = hyle::StatusCode::Ok;
  std::int64_t write_native = 0;
  std::int64_t read_native = 0;
  std::string write_message;
  std::string read_message;
};

hyle::sched::Task<int> pipe_round_trip(
    hyle::mem::AllocatorRef allocator, ChildProcess* child,
    PipeResult* result) {
  static constexpr char kPayload[] = "managed-process-pipe";
  char output[sizeof(kPayload) - 1]{};
  char error[12]{};
  auto written = co_await hyle::io::write_all(
      allocator, *child->stdin_pipe(),
      std::as_bytes(std::span{kPayload, sizeof(kPayload) - 1}));
  const hyle::Status closed = child->close_stdin();
  auto read = co_await hyle::io::read_exact(
      allocator, *child->stdout_pipe(),
      std::as_writable_bytes(std::span{output}));
  auto error_read = co_await hyle::io::read_exact(
      allocator, *child->stderr_pipe(),
      std::as_writable_bytes(std::span{error}));
  auto exit = child->wait();
  result->ok = written.ok() && closed.ok() && read.ok() && error_read.ok() &&
               exit.has_value() && exit->exit_code == 0 &&
               std::memcmp(output, kPayload, sizeof(output)) == 0 &&
               std::memcmp(error, "child-stderr", sizeof(error)) == 0;
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

hyle::sched::Task<int> read_environment(
    hyle::mem::AllocatorRef allocator, ChildProcess* child,
    PipeResult* result) {
  char output[11]{};
  auto read = co_await hyle::io::read_exact(
      allocator, *child->stdout_pipe(),
      std::as_writable_bytes(std::span{output}));
  auto exit = child->wait();
  result->ok = read.ok() && exit.has_value() && exit->exit_code == 0 &&
               std::memcmp(output, "value-12345", sizeof(output)) == 0;
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

hyle::sched::Task<int> kill_tree_and_observe_eof(
    hyle::mem::AllocatorRef allocator, ChildProcess* child,
    PipeResult* result) {
  char ready[5]{};
  auto initial = co_await hyle::io::read_exact(
      allocator, *child->stdout_pipe(),
      std::as_writable_bytes(std::span{ready}));
  const hyle::Status killed = child->force_kill_tree();
  auto exit = child->wait();
  std::byte byte{};
  hyle::io::OperationOptions bounded;
  bounded.deadline =
      hyle::time::deadline_after(std::chrono::seconds(2));
  auto eof = co_await child->stdout_pipe()->read_some(
      std::span<std::byte>{&byte, 1}, bounded);
  result->initial_ok = initial.ok() && std::memcmp(ready, "ready", 5) == 0;
  result->kill_ok = killed.ok();
  result->exit_ok =
      exit.has_value() && exit->kind == ProcessExitKind::kForced;
  result->eof_ok = eof.ok();
  result->eof_seen = eof.ok() && eof.eof;
  result->eof_status_code = eof.status.code();
  result->eof_native_code = eof.status.native_error().code;
  result->eof_message = eof.status.message();
  result->ok = result->initial_ok && result->kill_ok && result->exit_ok &&
               result->eof_ok && result->eof_seen;
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

hyle::sched::Task<int> write_large_pipe(
    hyle::mem::AllocatorRef allocator, ChildProcess* child,
    std::span<const std::byte> input, LargePipeResult* result) {
  auto written =
      co_await hyle::io::write_all(allocator, *child->stdin_pipe(), input);
  const hyle::Status closed = child->close_stdin();
  result->written = written.bytes;
  result->write_code = written.status.code();
  result->write_native = written.status.native_error().code;
  result->write_message = written.status.message();
  result->write_ok = written.ok() && written.bytes == input.size() &&
                     closed.ok();
  co_return 0;
}

hyle::sched::Task<int> read_large_pipe(
    hyle::mem::AllocatorRef allocator, ChildProcess* child,
    std::span<std::byte> output, LargePipeResult* result) {
  auto read =
      co_await hyle::io::read_exact(allocator, *child->stdout_pipe(), output);
  auto error_read = co_await hyle::io::read_exact(
      allocator, *child->stderr_pipe(),
      std::as_writable_bytes(std::span{result->error}));
  result->read = read.bytes;
  result->read_code = read.status.code();
  result->read_native = read.status.native_error().code;
  result->read_message = read.status.message();
  result->read_ok = read.ok() && read.bytes == output.size() &&
                    error_read.ok() &&
                    std::memcmp(result->error.data(), "child-stderr",
                                result->error.size()) == 0;
  co_return 0;
}

hyle::sched::Task<int> wait_in_fiber(hyle::mem::AllocatorRef,
                                     ChildProcess* child,
                                     LargePipeResult* result) {
  auto exit = child->wait();
  result->wait_ok = exit.has_value() &&
                    exit->kind == ProcessExitKind::kExited &&
                    exit->exit_code == 0;
  co_return 0;
}

TEST(Process, SpawnWaitAndRepeatedObservation) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value())
      << (runtime_result.has_value() ? "" : runtime_result.error().message());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"exit", "7"};
  auto child_result = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(child_result.has_value())
      << (child_result.has_value() ? "" : child_result.error().message());
  ChildProcess child = std::move(*child_result);
  EXPECT_NE(child.id(), 0u);
  auto exit = child.wait();
  ASSERT_TRUE(exit.has_value())
      << (exit.has_value() ? "" : exit.error().message());
  EXPECT_EQ(exit->kind, ProcessExitKind::kExited);
  EXPECT_EQ(exit->exit_code, 7u);
  auto repeated = child.try_wait();
  ASSERT_TRUE(repeated.has_value());
  ASSERT_TRUE(repeated->has_value());
  EXPECT_EQ((*repeated)->exit_code, 7u);
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, DeadlineCancelsOnlyTheWait) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"sleep", "200"};
  auto child_result = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(child_result.has_value());
  ChildProcess child = std::move(*child_result);
  ProcessWaitOptions wait;
  wait.deadline =
      hyle::time::deadline_after(std::chrono::milliseconds(5));
  auto timed = child.wait(wait);
  ASSERT_FALSE(timed.has_value());
  EXPECT_TRUE(timed.error().IsDeadlineExceeded());
  EXPECT_TRUE(child.force_kill_tree().ok());
  auto exit = child.wait();
  ASSERT_TRUE(exit.has_value());
  EXPECT_EQ(exit->kind, ProcessExitKind::kForced);
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, InvalidConfigurationFailsBeforeSpawn) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  ProcessStartOptions options;
  EXPECT_FALSE(runtime.spawn(options).has_value());
  options.executable = kHelper;
  options.stdin_stream = ProcessStdio::merge_stdout();
  EXPECT_FALSE(runtime.spawn(options).has_value());

  const char nul_argument[] = {'a', '\0', 'b'};
  const std::array<std::string_view, 1> invalid_arguments{
      std::string_view{nul_argument, sizeof(nul_argument)}};
  options = helper_options(invalid_arguments);
  EXPECT_FALSE(runtime.spawn(options).has_value());

  const std::array<EnvironmentChange, 2> duplicate_environment{{
      {"HYLE_DUPLICATE", std::string_view{"a"}},
      {"HYLE_DUPLICATE", std::string_view{"b"}},
  }};
  options = helper_options(std::span<const std::string_view>{});
  options.environment_changes = duplicate_environment;
  EXPECT_FALSE(runtime.spawn(options).has_value());

  const std::array<EnvironmentChange, 1> replacement_unset{{
      {"HYLE_UNSET", std::nullopt},
  }};
  options.environment_mode = EnvironmentMode::kReplace;
  options.environment_changes = replacement_unset;
  EXPECT_FALSE(runtime.spawn(options).has_value());

  options = helper_options(std::span<const std::string_view>{});
  options.stdout_stream = ProcessStdio::pipe();
  EXPECT_FALSE(runtime.spawn(options).has_value());

  runtime.begin_shutdown();
  EXPECT_FALSE(runtime.spawn(
      helper_options(std::span<const std::string_view>{})).has_value());
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, SpawnFailureDoesNotPoisonRuntime) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);

  ProcessStartOptions missing;
  missing.executable = "hyle-process-definitely-not-present-7f0f4f5c";
  missing.stdin_stream = ProcessStdio::null();
  missing.stdout_stream = ProcessStdio::null();
  missing.stderr_stream = ProcessStdio::null();
  EXPECT_FALSE(runtime.spawn(missing).has_value());
  EXPECT_EQ(runtime.active_processes(), 0u);

  const std::array<std::string_view, 2> arguments{"exit", "0"};
  auto child = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(child.has_value());
  auto exit = child->wait();
  ASSERT_TRUE(exit.has_value());
  EXPECT_EQ(exit->exit_code, 0u);
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, PathSearchAndArgumentBoundaries) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);

  const std::array<std::string_view, 5> quoted{
      "arguments", "", "a b", "\"", "trail\\"};
  auto quoted_child = runtime.spawn(helper_options(quoted));
  ASSERT_TRUE(quoted_child.has_value())
      << (quoted_child.has_value() ? "" : quoted_child.error().message());
  auto quoted_exit = quoted_child->wait();
  ASSERT_TRUE(quoted_exit.has_value());
  EXPECT_EQ(quoted_exit->exit_code, 0u);

#if defined(_WIN32)
  const std::array<std::string_view, 1> path_arguments{"cmd.exe"};
  ProcessStartOptions path = helper_options(path_arguments);
  path.executable = "where.exe";
#else
  const std::array<std::string_view, 2> path_arguments{"-c", "exit 9"};
  ProcessStartOptions path = helper_options(path_arguments);
  path.executable = "sh";
#endif
  path.executable_search = ExecutableSearch::kPath;
  auto path_child = runtime.spawn(path);
  ASSERT_TRUE(path_child.has_value())
      << (path_child.has_value() ? "" : path_child.error().message());
  auto path_exit = path_child->wait();
  ASSERT_TRUE(path_exit.has_value());
#if defined(_WIN32)
  EXPECT_EQ(path_exit->exit_code, 0u);
#else
  EXPECT_EQ(path_exit->exit_code, 9u);
#endif
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, AsyncPipesReuseHyleIo) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 128);
  hyle::io::IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);

  const std::array<std::string_view, 1> arguments{"echo"};
  ProcessStartOptions options = helper_options(arguments);
  options.stdin_stream = ProcessStdio::pipe();
  options.stdout_stream = ProcessStdio::pipe();
  options.stderr_stream = ProcessStdio::pipe();
  options.pipe_io = &io;
  auto child_result = runtime.spawn(options);
  ASSERT_TRUE(child_result.has_value())
      << (child_result.has_value() ? "" : child_result.error().message());
  ChildProcess child = std::move(*child_result);
  ASSERT_NE(child.stdin_pipe(), nullptr);
  ASSERT_NE(child.stdout_pipe(), nullptr);
  ASSERT_NE(child.stderr_pipe(), nullptr);

  PipeResult result;
  Drive drive{pipe_round_trip(hyle::mem::system_allocator(), &child, &result)};
  hyle::sched::Fiber fiber(hyle::sched::kDefaultStackSize, &drive_entry,
                           &drive);
  ASSERT_TRUE(scheduler.spawn_on(0, &fiber));
  scheduler.drain();
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.ok);
  EXPECT_TRUE(runtime.shutdown().ok());
  scheduler.shutdown();
}

TEST(Process, LargeBidirectionalPipeMakesBoundedProgress) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 256);
  hyle::io::IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);

  const std::array<std::string_view, 1> arguments{"echo"};
  ProcessStartOptions options = helper_options(arguments);
  options.stdin_stream = ProcessStdio::pipe();
  options.stdout_stream = ProcessStdio::pipe();
  options.stderr_stream = ProcessStdio::pipe();
  options.pipe_io = &io;
  auto child_result = runtime.spawn(options);
  ASSERT_TRUE(child_result.has_value());
  ChildProcess child = std::move(*child_result);

  constexpr std::size_t kBytes = 1024 * 1024;
  std::vector<std::byte> input(kBytes);
  std::vector<std::byte> output(kBytes);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::byte>((i * 131u + 17u) & 0xffu);

  LargePipeResult result;
  Drive writer{write_large_pipe(hyle::mem::system_allocator(), &child,
                               input, &result)};
  Drive reader{read_large_pipe(hyle::mem::system_allocator(), &child,
                              output, &result)};
  Drive waiter{
      wait_in_fiber(hyle::mem::system_allocator(), &child, &result)};
  hyle::sched::Fiber writer_fiber(hyle::sched::kDefaultStackSize,
                                  &drive_entry, &writer);
  hyle::sched::Fiber reader_fiber(hyle::sched::kDefaultStackSize,
                                  &drive_entry, &reader);
  hyle::sched::Fiber waiter_fiber(hyle::sched::kDefaultStackSize,
                                  &drive_entry, &waiter);
  ASSERT_TRUE(scheduler.spawn_on(0, &writer_fiber));
  ASSERT_TRUE(scheduler.spawn_on(0, &reader_fiber));
  ASSERT_TRUE(scheduler.spawn_on(0, &waiter_fiber));
  scheduler.drain();
  EXPECT_TRUE(result.write_ok)
      << "bytes=" << result.written
      << " status=" << static_cast<unsigned>(result.write_code)
      << " native=" << result.write_native
      << " message=" << result.write_message;
  EXPECT_TRUE(result.read_ok)
      << "bytes=" << result.read
      << " status=" << static_cast<unsigned>(result.read_code)
      << " native=" << result.read_native
      << " message=" << result.read_message;
  EXPECT_TRUE(result.wait_ok);
  EXPECT_EQ(output, input);
  EXPECT_TRUE(runtime.shutdown().ok());
  scheduler.shutdown();
}

TEST(Process, ReplacementEnvironmentIsCopiedForChild) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 128);
  hyle::io::IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);

  const std::array<std::string_view, 2> arguments{
      "environment", "HYLE_PROCESS_VALUE"};
  const std::array<EnvironmentChange, 1> changes{{
      {"HYLE_PROCESS_VALUE", std::string_view{"value-12345"}}}};
  ProcessStartOptions options = helper_options(arguments);
  options.environment_mode = EnvironmentMode::kReplace;
  options.environment_changes = changes;
  options.stdout_stream = ProcessStdio::pipe();
  options.pipe_io = &io;
  auto child_result = runtime.spawn(options);
  ASSERT_TRUE(child_result.has_value())
      << (child_result.has_value() ? "" : child_result.error().message());
  ChildProcess child = std::move(*child_result);
  PipeResult result;
  Drive drive{read_environment(hyle::mem::system_allocator(), &child, &result)};
  hyle::sched::Fiber fiber(hyle::sched::kDefaultStackSize, &drive_entry,
                           &drive);
  ASSERT_TRUE(scheduler.spawn_on(0, &fiber));
  scheduler.drain();
  EXPECT_TRUE(result.ok);
  EXPECT_TRUE(runtime.shutdown().ok());
  scheduler.shutdown();
}

TEST(Process, InheritedEnvironmentCanRemoveVariable) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{
      "environment_absent", "PATH"};
  const std::array<EnvironmentChange, 1> changes{{
      {"PATH", std::nullopt},
  }};
  ProcessStartOptions options = helper_options(arguments);
  options.environment_changes = changes;
  auto child = runtime.spawn(options);
  ASSERT_TRUE(child.has_value());
  auto exit = child->wait();
  ASSERT_TRUE(exit.has_value());
  EXPECT_EQ(exit->exit_code, 0u);
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, FileMergeAndWorkingDirectoryAreNaturalCompositions) {
  TempDirectory temporary;
  ASSERT_TRUE(temporary.valid());
  const std::filesystem::path output_path = temporary.path() / "streams.txt";
  const std::string output_name = output_path.string();
  auto output = hyle::platform::open_file(
      output_name.c_str(), hyle::platform::kOpenWrite |
                               hyle::platform::kOpenCreate |
                               hyle::platform::kOpenTruncate);
  ASSERT_TRUE(output.has_value())
      << (output.has_value() ? "" : output.error().message());

  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 1> stream_arguments{"streams"};
  ProcessStartOptions streams = helper_options(stream_arguments);
  streams.stdout_stream = ProcessStdio::file(*output);
  streams.stderr_stream = ProcessStdio::merge_stdout();
  auto stream_child = runtime.spawn(streams);
  ASSERT_TRUE(stream_child.has_value());
  EXPECT_TRUE(output->close().ok());
  auto stream_exit = stream_child->wait();
  ASSERT_TRUE(stream_exit.has_value());
  EXPECT_EQ(stream_exit->exit_code, 0u);

  auto input = hyle::platform::open_file(output_name.c_str(),
                                         hyle::platform::kOpenRead);
  ASSERT_TRUE(input.has_value());
  std::array<char, 6> content{};
  EXPECT_TRUE(input->read_exact(content.data(), content.size()).ok());
  EXPECT_EQ(std::string_view(content.data(), content.size()), "outerr");

  const std::array<std::string_view, 1> cwd_arguments{"cwd_marker"};
  ProcessStartOptions cwd = helper_options(cwd_arguments);
  const std::string directory = temporary.path().string();
  cwd.working_directory = directory;
  auto cwd_child = runtime.spawn(cwd);
  ASSERT_TRUE(cwd_child.has_value());
  auto cwd_exit = cwd_child->wait();
  ASSERT_TRUE(cwd_exit.has_value());
  EXPECT_EQ(cwd_exit->exit_code, 0u);
  EXPECT_TRUE(std::filesystem::exists(
      temporary.path() / "hyle-child-cwd-marker"));
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, CancellationCancelsWaitButNotChild) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"sleep", "30000"};
  auto child_result = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(child_result.has_value());
  ChildProcess child = std::move(*child_result);
  hyle::sched::CancelToken cancel =
      hyle::sched::CancelToken::root(hyle::mem::system_allocator());
  bool cancelled = false;
  std::thread waiter([&] {
    ProcessWaitOptions options;
    options.cancel = &cancel;
    auto result = child.wait(options);
    cancelled = !result.has_value() && result.error().IsCancelled();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  cancel.cancel();
  waiter.join();
  EXPECT_TRUE(cancelled);
  auto still_running = child.try_wait();
  ASSERT_TRUE(still_running.has_value());
  EXPECT_FALSE(still_running->has_value());
  EXPECT_TRUE(child.force_kill_tree().ok());
  EXPECT_TRUE(child.wait().has_value());
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, TerminationNamesDoNotHidePlatformSemantics) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"sleep", "30000"};
  auto child_result = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(child_result.has_value());
  ChildProcess child = std::move(*child_result);
  hyle::Status requested = child.request_termination();
#if defined(_WIN32)
  EXPECT_TRUE(requested.IsUnimplemented());
  EXPECT_TRUE(child.force_kill_tree().ok());
#else
  EXPECT_TRUE(requested.ok());
#endif
  auto exit = child.wait();
  ASSERT_TRUE(exit.has_value());
#if defined(_WIN32)
  EXPECT_EQ(exit->kind, ProcessExitKind::kForced);
#else
  EXPECT_EQ(exit->kind, ProcessExitKind::kSignaled);
  EXPECT_GT(exit->termination_signal, 0);
#endif
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, ReleaseAndRuntimeShutdownRetainReapOwnership) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"sleep", "30000"};
  auto released_result = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(released_result.has_value());
  ChildProcess released = std::move(*released_result);
  EXPECT_TRUE(released.release_to_reaper().ok());
  EXPECT_FALSE(released.valid());

  auto observed_result = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(observed_result.has_value());
  ChildProcess observed = std::move(*observed_result);
  EXPECT_TRUE(runtime.shutdown().ok());
  EXPECT_FALSE(runtime.accepting());
  EXPECT_EQ(runtime.active_processes(), 0u);
  auto exit = observed.wait();
  ASSERT_TRUE(exit.has_value());
  EXPECT_EQ(exit->kind, ProcessExitKind::kForced);
}

TEST(Process, DroppedOwnerTerminatesWithoutBlockingDestructor) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"sleep", "30000"};
  {
    auto child = runtime.spawn(helper_options(arguments));
    ASSERT_TRUE(child.has_value());
  }
  EXPECT_TRUE(runtime.shutdown().ok());
  EXPECT_EQ(runtime.active_processes(), 0u);
}

TEST(Process, ConcurrentSpawnAndExitShareOneRuntime) {
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"exit", "0"};
  constexpr std::size_t kThreads = 12;
  std::atomic<std::size_t> completed{0};
  std::array<std::thread, kThreads> threads;
  for (std::thread& thread : threads) {
    thread = std::thread([&] {
      auto child = runtime.spawn(helper_options(arguments));
      if (!child.has_value()) return;
      auto exit = child->wait();
      if (exit.has_value() && exit->kind == ProcessExitKind::kExited &&
          exit->exit_code == 0) {
        completed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  EXPECT_EQ(completed.load(std::memory_order_relaxed), kThreads);
  EXPECT_EQ(runtime.active_processes(), 0u);
  EXPECT_TRUE(runtime.shutdown().ok());
}

TEST(Process, AllocatorFailureIsTypedAndBalanced) {
  auto invalid = ProcessRuntime::create(hyle::mem::AllocatorRef{});
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());

  FailAfterAllocator failing;
  failing.fail_after = 1;
  {
    auto runtime_result = ProcessRuntime::create(failing.get());
    ASSERT_TRUE(runtime_result.has_value());
    ProcessRuntime runtime = std::move(*runtime_result);
    const std::array<std::string_view, 2> arguments{"exit", "0"};
    auto child = runtime.spawn(helper_options(arguments));
    ASSERT_FALSE(child.has_value());
    EXPECT_TRUE(child.error().IsResourceExhausted());
    EXPECT_TRUE(runtime.shutdown().ok());
  }
  EXPECT_EQ(failing.live, 0u);
}

TEST(Process, SpawnDoesNotLeakUnrelatedInheritableHandles) {
  std::uintptr_t raw = 0;
#if defined(_WIN32)
  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  HANDLE unrelated = ::CreateEventW(&security, TRUE, FALSE, nullptr);
  ASSERT_NE(unrelated, nullptr);
  raw = reinterpret_cast<std::uintptr_t>(unrelated);
#else
  int unrelated[2] = {-1, -1};
  ASSERT_EQ(::pipe(unrelated), 0);
  ASSERT_EQ(::fcntl(unrelated[0], F_SETFD, 0), 0);
  ASSERT_EQ(::fcntl(unrelated[1], F_SETFD, 0), 0);
  raw = static_cast<std::uintptr_t>(unrelated[0]);
#endif

  const std::string raw_text = std::to_string(raw);
  const std::array<std::string_view, 2> arguments{
      "unrelated_handle_absent", raw_text};
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  auto child = runtime.spawn(helper_options(arguments));
#if defined(_WIN32)
  EXPECT_NE(::CloseHandle(unrelated), FALSE);
#else
  EXPECT_EQ(::close(unrelated[0]), 0);
  EXPECT_EQ(::close(unrelated[1]), 0);
#endif
  ASSERT_TRUE(child.has_value())
      << (child.has_value() ? "" : child.error().message());
  auto exit = child->wait();
  ASSERT_TRUE(exit.has_value());
  EXPECT_EQ(exit->exit_code, 0u);
  EXPECT_TRUE(runtime.shutdown().ok());
}

#if !defined(_WIN32)
TEST(Process, ReaperNeverConsumesAnExternalChild) {
  char* external_arguments[] = {
      const_cast<char*>(kHelper.data()), const_cast<char*>("sleep"),
      const_cast<char*>("20"), nullptr};
  pid_t external = -1;
  ASSERT_EQ(::posix_spawn(&external, kHelper.data(), nullptr, nullptr,
                          external_arguments, environ),
            0);

  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);
  const std::array<std::string_view, 2> arguments{"exit", "0"};
  auto child = runtime.spawn(helper_options(arguments));
  ASSERT_TRUE(child.has_value());
  EXPECT_TRUE(child->wait().has_value());

  int external_status = 0;
  ASSERT_EQ(::waitpid(external, &external_status, 0), external);
  EXPECT_TRUE(WIFEXITED(external_status));
  EXPECT_EQ(WEXITSTATUS(external_status), 0);
  EXPECT_TRUE(runtime.shutdown().ok());
}
#endif

TEST(Process, ForceKillOwnsDescendantTreeAndPipeEof) {
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 128);
  hyle::io::IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  auto runtime_result = ProcessRuntime::create();
  ASSERT_TRUE(runtime_result.has_value());
  ProcessRuntime runtime = std::move(*runtime_result);

  const std::array<std::string_view, 1> arguments{"tree"};
  ProcessStartOptions options = helper_options(arguments);
  options.stdout_stream = ProcessStdio::pipe();
  options.pipe_io = &io;
  auto child_result = runtime.spawn(options);
  ASSERT_TRUE(child_result.has_value())
      << (child_result.has_value() ? "" : child_result.error().message());
  ChildProcess child = std::move(*child_result);
  PipeResult result;
  Drive drive{kill_tree_and_observe_eof(
      hyle::mem::system_allocator(), &child, &result)};
  hyle::sched::Fiber fiber(hyle::sched::kDefaultStackSize, &drive_entry,
                           &drive);
  ASSERT_TRUE(scheduler.spawn_on(0, &fiber));
  scheduler.drain();
  EXPECT_TRUE(result.initial_ok);
  EXPECT_TRUE(result.kill_ok);
  EXPECT_TRUE(result.exit_ok);
  EXPECT_TRUE(result.eof_ok)
      << "status=" << static_cast<unsigned>(result.eof_status_code)
      << " native=" << result.eof_native_code
      << " message=" << result.eof_message;
  EXPECT_TRUE(result.eof_seen)
      << "status=" << static_cast<unsigned>(result.eof_status_code)
      << " native=" << result.eof_native_code
      << " message=" << result.eof_message;
  EXPECT_TRUE(result.ok);
  EXPECT_TRUE(runtime.shutdown().ok());
  scheduler.shutdown();
}

}  // namespace
