#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <span>
#include <string_view>

#include "hyle/io/io_context.h"
#include "hyle/io/local.h"
#include "hyle/mem/allocator.h"
#include "hyle/process/process.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

namespace {

using namespace std::chrono_literals;

struct ProcessJourneyResult {
  hyle::Status status{};
  bool ready = false;
  bool wait_cancelled = false;
  bool child_survived_cancel = false;
  bool tree_forced = false;
  bool reaped = false;
  bool pipe_eof = false;
  bool pipe_closed = false;
  std::atomic<bool> done{false};
};

hyle::sched::Task<int> supervise_child(
    hyle::mem::AllocatorRef allocator, hyle::process::ChildProcess* child,
    ProcessJourneyResult* result) noexcept {
  std::array<char, 5> ready{};
  auto initial = co_await hyle::io::read_exact(
      allocator, *child->stdout_pipe(),
      std::as_writable_bytes(std::span{ready}),
      {.deadline = hyle::time::deadline_after(5s)});
  result->ready = initial.ok() && initial.bytes == ready.size() &&
                  std::memcmp(ready.data(), "ready", ready.size()) == 0;
  if (!result->ready) {
    result->status = std::move(initial.status);
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }

  auto cancel = hyle::sched::CancelToken::try_create_root(allocator);
  if (!cancel.has_value()) {
    result->status = std::move(cancel).error();
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  cancel->cancel();
  hyle::process::ProcessWaitOptions wait_options;
  wait_options.cancel = &*cancel;
  auto cancelled_wait = child->wait(wait_options);
  result->wait_cancelled =
      !cancelled_wait.has_value() && cancelled_wait.error().IsCancelled();
  auto still_running = child->try_wait();
  result->child_survived_cancel =
      still_running.has_value() && !still_running->has_value();

  const hyle::Status killed = child->force_kill_tree();
  result->tree_forced = killed.ok();
  auto exit = child->wait();
  result->reaped =
      exit.has_value() &&
      exit->kind == hyle::process::ProcessExitKind::kForced;

  std::byte trailing{};
  auto eof = co_await child->stdout_pipe()->read_some(
      std::span<std::byte>{&trailing, 1},
      {.deadline = hyle::time::deadline_after(5s)});
  result->pipe_eof = eof.ok() && eof.eof;
  result->pipe_closed = child->stdout_pipe()->close().ok();
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  hyle::sched::Task<int> task;
};

void drive_entry(void* raw) noexcept {
  static_cast<Drive*>(raw)->task.resume();
}

}  // namespace

TEST(CorrectnessJourney, WorkingProcessCancelsWaitThenKillsReapsAndClosesPipe) {
  hyle::sched::Scheduler scheduler(
      2, hyle::mem::system_allocator(), 128);
  hyle::io::IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();

  auto made_runtime = hyle::process::ProcessRuntime::create();
  ASSERT_TRUE(made_runtime.has_value()) << made_runtime.error().message();
  hyle::process::ProcessRuntime runtime = std::move(*made_runtime);
  const std::array<std::string_view, 1> arguments{"tree"};
  hyle::process::ProcessStartOptions options;
  options.executable = HYLE_PROCESS_TEST_HELPER_PATH;
  options.arguments = arguments;
  options.stdin_stream = hyle::process::ProcessStdio::null();
  options.stdout_stream = hyle::process::ProcessStdio::pipe();
  options.stderr_stream = hyle::process::ProcessStdio::null();
  options.pipe_io = &io;
  options.pipe_owner = 0;
  auto made_child = runtime.spawn(options);
  ASSERT_TRUE(made_child.has_value()) << made_child.error().message();
  hyle::process::ChildProcess child = std::move(*made_child);

  ProcessJourneyResult result;
  Drive drive{supervise_child(hyle::mem::system_allocator(), &child, &result)};
  ASSERT_TRUE(drive.task.valid());
  hyle::sched::Fiber fiber(hyle::sched::kDefaultStackSize,
                           &drive_entry, &drive);
  ASSERT_TRUE(scheduler.spawn_on(0, &fiber));
  scheduler.drain();

  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.status.ok()) << result.status.message();
  EXPECT_TRUE(result.ready);
  EXPECT_TRUE(result.wait_cancelled);
  EXPECT_TRUE(result.child_survived_cancel);
  EXPECT_TRUE(result.tree_forced);
  EXPECT_TRUE(result.reaped);
  EXPECT_TRUE(result.pipe_eof);
  EXPECT_TRUE(result.pipe_closed);
  EXPECT_EQ(runtime.active_processes(), 0u);
  EXPECT_TRUE(runtime.shutdown().ok());
  scheduler.shutdown();
}
