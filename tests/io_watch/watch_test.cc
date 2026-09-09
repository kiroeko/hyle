#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/io/watch.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

namespace {

using hyle::Status;
using hyle::io::FileWatchChange;
using hyle::io::FileWatcher;
using hyle::io::IoContext;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::Scheduler;
using hyle::sched::Task;

constexpr unsigned kOwner = 0;

class TempDirectory {
 public:
  TempDirectory() {
    static std::atomic<std::uint64_t> sequence{0};
    const auto suffix =
        static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        sequence.fetch_add(1, std::memory_order_relaxed);
    path_ = std::filesystem::temp_directory_path() /
            ("hyle_io_watch_" + std::to_string(suffix));
    std::filesystem::create_directory(path_);
  }

  ~TempDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

Status write_file(const std::string& path, std::string_view contents) noexcept {
  auto opened = hyle::platform::open_file(
      path.c_str(), hyle::platform::kOpenWrite |
                        hyle::platform::kOpenCreate |
                        hyle::platform::kOpenTruncate);
  if (!opened.has_value()) return std::move(opened).error();
  Status status = opened->write_all(contents.data(), contents.size());
  Status close = opened->close();
  status.Update(std::move(close));
  return status;
}

struct Drive {
  Task<int> task;
};

void drive_entry(void* context) noexcept {
  static_cast<Drive*>(context)->task.resume();
}

struct WatchOutcome {
  Status status{};
  bool saw_exact = false;
  bool saw_directory = false;
  bool saw_created_or_content = false;
};

Task<int> observe_created_file(AllocatorRef allocator, IoContext* io,
                               std::string root, std::string child,
                               WatchOutcome* outcome) noexcept {
  const std::array targets{
      hyle::io::FileWatchTarget::file(child),
      hyle::io::FileWatchTarget::directory(root)};
  auto made = FileWatcher::create(allocator, *io, kOwner, targets);
  if (!made.has_value()) {
    outcome->status = std::move(made).error();
    co_return 0;
  }
  FileWatcher watcher = std::move(*made);

  outcome->status = write_file(child, "new model");
  if (outcome->status.ok()) {
    auto batch = co_await watcher.next({.deadline =
        hyle::time::deadline_after(std::chrono::seconds(5))});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
    } else {
      for (const auto& event : batch->events()) {
        if (event.target_index == 0 && event.relative_path.empty())
          outcome->saw_exact = true;
        if (event.target_index == 1 && event.relative_path == "model.json")
          outcome->saw_directory = true;
        outcome->saw_created_or_content =
            outcome->saw_created_or_content ||
            hyle::io::contains(event.changes, FileWatchChange::kCreated) ||
            hyle::io::contains(event.changes, FileWatchChange::kContent);
      }
    }
  }
  outcome->status.Update(watcher.close());
  co_return 0;
}

struct CancelOutcome {
  Status status{};
  bool wait_cancelled = false;
};

Task<int> observe_precancel(AllocatorRef allocator, IoContext* io,
                            std::string root,
                            CancelOutcome* outcome) noexcept {
  const std::array targets{
      hyle::io::FileWatchTarget::directory(root)};
  auto made = FileWatcher::create(allocator, *io, kOwner, targets);
  if (!made.has_value()) {
    outcome->status = std::move(made).error();
    co_return 0;
  }
  FileWatcher watcher = std::move(*made);
  auto token = hyle::sched::CancelToken::try_create_root(allocator);
  if (!token.has_value()) {
    outcome->status = std::move(token).error();
  } else {
    token->cancel();
    auto batch = co_await watcher.next(
        {.deadline = std::nullopt, .cancel = &*token});
    outcome->wait_cancelled = !batch.has_value() &&
                              batch.error().IsCancelled();
    if (batch.has_value())
      outcome->status = Status::Internal(
          "pre-cancelled FileWatcher::next returned a batch");
    else if (!batch.error().IsCancelled())
      outcome->status = std::move(batch).error();
  }
  outcome->status.Update(watcher.close());
  co_return 0;
}

struct LifecycleOutcome {
  Status status{};
  bool deadline_reported = false;
  bool rescan_reported = false;
  bool atomic_replace_reported = false;
  bool concurrent_next_rejected = false;
  bool waiting_next_cancelled_by_close = false;
};

Task<void> wait_for_close(AllocatorRef allocator, FileWatcher* watcher,
                          LifecycleOutcome* outcome) noexcept {
  (void)allocator;
  auto batch = co_await watcher->next();
  outcome->waiting_next_cancelled_by_close =
      !batch.has_value() && batch.error().IsCancelled();
  co_return;
}

Task<int> exercise_lifecycle(AllocatorRef allocator, IoContext* io,
                             Scheduler* scheduler, std::string root,
                             LifecycleOutcome* outcome) noexcept {
  const std::array directory_target{
      hyle::io::FileWatchTarget::directory(root)};

  // A deadline ends only the current wait. The watcher remains reusable.
  auto deadline_made =
      FileWatcher::create(allocator, *io, kOwner, directory_target);
  if (!deadline_made.has_value()) {
    outcome->status = std::move(deadline_made).error();
    co_return 0;
  }
  FileWatcher deadline_watcher = std::move(*deadline_made);
  auto deadline_batch = co_await deadline_watcher.next(
      {.deadline =
           hyle::time::deadline_after(std::chrono::milliseconds(25))});
  outcome->deadline_reported =
      !deadline_batch.has_value() &&
      deadline_batch.error().IsDeadlineExceeded() &&
      deadline_watcher.valid();
  outcome->status.Update(deadline_watcher.close());
  if (!outcome->status.ok()) co_return 0;

  // A path that cannot fit the public bounded batch is not silently dropped.
  hyle::io::FileWatchOptions bounded;
  bounded.max_batch_path_bytes = 1;
  auto bounded_made = FileWatcher::create(
      allocator, *io, kOwner, directory_target, bounded);
  if (!bounded_made.has_value()) {
    outcome->status = std::move(bounded_made).error();
    co_return 0;
  }
  FileWatcher bounded_watcher = std::move(*bounded_made);
  const std::string oversized = root + "/long-name.json";
  outcome->status = write_file(oversized, "x");
  if (outcome->status.ok()) {
    auto bounded_batch = co_await bounded_watcher.next(
        {.deadline = hyle::time::deadline_after(std::chrono::seconds(5))});
    if (!bounded_batch.has_value())
      outcome->status = std::move(bounded_batch).error();
    else
      outcome->rescan_reported = bounded_batch->rescan_required();
  }
  outcome->status.Update(bounded_watcher.close());
  if (!outcome->status.ok()) co_return 0;

  // Atomic replacement is observed through the parent even though the watched
  // destination inode changes.
  const std::string destination = root + "/weights.bin";
  const std::string replacement = root + "/weights.new";
  outcome->status = write_file(destination, "old");
  if (!outcome->status.ok()) co_return 0;
  const std::array exact_target{
      hyle::io::FileWatchTarget::file(destination)};
  auto replace_made =
      FileWatcher::create(allocator, *io, kOwner, exact_target);
  if (!replace_made.has_value()) {
    outcome->status = std::move(replace_made).error();
    co_return 0;
  }
  FileWatcher replace_watcher = std::move(*replace_made);
  outcome->status = write_file(replacement, "new");
  if (outcome->status.ok())
    outcome->status =
        hyle::platform::rename_path(replacement.c_str(), destination.c_str());
  if (outcome->status.ok()) {
    auto replaced = co_await replace_watcher.next(
        {.deadline = hyle::time::deadline_after(std::chrono::seconds(5))});
    if (!replaced.has_value()) {
      outcome->status = std::move(replaced).error();
    } else {
      for (const auto& event : replaced->events()) {
        if (event.target_index == 0 && event.relative_path.empty())
          outcome->atomic_replace_reported = true;
      }
    }
  }
  outcome->status.Update(replace_watcher.close());
  if (!outcome->status.ok()) co_return 0;

  // A second next receives a typed contract failure. close concurrently wakes
  // and joins the first waiter and all native work.
  auto close_made =
      FileWatcher::create(allocator, *io, kOwner, directory_target);
  if (!close_made.has_value()) {
    outcome->status = std::move(close_made).error();
    co_return 0;
  }
  FileWatcher close_watcher = std::move(*close_made);
  auto waiter = hyle::sched::spawn_task_on(
      *scheduler, kOwner, allocator,
      wait_for_close(allocator, &close_watcher, outcome));
  if (waiter.is_finished()) {
    auto joined = waiter.join();
    if (joined.has_value())
      outcome->status = Status::Internal("watch waiter did not start");
    else
      outcome->status = std::move(joined).error();
    (void)close_watcher.close();
    co_return 0;
  }
  Fiber::yield();
  auto duplicate = co_await close_watcher.next(
      {.deadline =
           hyle::time::deadline_after(std::chrono::milliseconds(10))});
  outcome->concurrent_next_rejected =
      !duplicate.has_value() && duplicate.error().IsFailedPrecondition();
  outcome->status = close_watcher.close();
  auto joined = waiter.join();
  if (!joined.has_value()) outcome->status.Update(std::move(joined).error());
  co_return 0;
}

template <class Outcome, class Factory>
void run_watch_journey(Outcome* outcome, Factory&& factory) {
  Scheduler scheduler(1, system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  Drive drive{factory(&io, &scheduler)};
  ASSERT_TRUE(drive.task.valid());
  Fiber fiber(hyle::sched::kDefaultStackSize, &drive_entry, &drive);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  ASSERT_TRUE(drive.task.done());
  ASSERT_TRUE(outcome->status.ok()) << outcome->status.message();
}

TEST(FileWatchApi, TargetFactoriesAndChangeMaskAreTyped) {
  constexpr auto file = hyle::io::FileWatchTarget::file("model.json");
  constexpr auto directory =
      hyle::io::FileWatchTarget::directory("models");

  static_assert(file.kind() == hyle::io::FileWatchTargetKind::kFile);
  static_assert(directory.kind() ==
                hyle::io::FileWatchTargetKind::kDirectory);
  EXPECT_EQ(file.path(), std::string_view("model.json"));
  EXPECT_EQ(directory.path(), std::string_view("models"));

  constexpr auto changes = hyle::io::FileWatchChange::kContent |
                           hyle::io::FileWatchChange::kMetadata;
  static_assert(hyle::io::contains(changes,
                                   hyle::io::FileWatchChange::kContent));
  static_assert(!hyle::io::contains(changes,
                                    hyle::io::FileWatchChange::kRemoved));
}

TEST(FileWatchApi, EmptyOwnerIsInert) {
  hyle::io::FileWatcher watcher;
  EXPECT_FALSE(watcher.valid());
  EXPECT_EQ(watcher.owner_worker(), 0u);
  EXPECT_TRUE(watcher.close().ok());
}

TEST(FileWatchJourney, ExactAndDirectoryTargetsObserveCreation) {
  TempDirectory temporary;
  const std::string root = temporary.path().string();
  const std::string child = (temporary.path() / "model.json").string();
  WatchOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io, Scheduler*) {
    return observe_created_file(system_allocator(), io, root, child, &outcome);
  });
  EXPECT_TRUE(outcome.saw_exact);
  EXPECT_TRUE(outcome.saw_directory);
  EXPECT_TRUE(outcome.saw_created_or_content);
}

TEST(FileWatchJourney, PreCancelledWaitDoesNotCloseWatcher) {
  TempDirectory temporary;
  const std::string root = temporary.path().string();
  CancelOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io, Scheduler*) {
    return observe_precancel(system_allocator(), io, root, &outcome);
  });
  EXPECT_TRUE(outcome.wait_cancelled);
}

TEST(FileWatchJourney, CreateOffOwnerReturnsTypedFailureWithoutAllocatingOwner) {
  TempDirectory temporary;
  Scheduler scheduler(1, system_allocator(), 256);
  ASSERT_TRUE(scheduler.valid());
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  const std::string root = temporary.path().string();
  const std::array targets{
      hyle::io::FileWatchTarget::directory(root)};

  auto watcher = FileWatcher::create(system_allocator(), io, kOwner, targets);
  ASSERT_FALSE(watcher.has_value());
  EXPECT_TRUE(watcher.error().IsFailedPrecondition());
  scheduler.shutdown();
}

TEST(FileWatchJourney, LifecycleBoundsReplaceAndCloseAreExplicit) {
  TempDirectory temporary;
  const std::string root = temporary.path().string();
  LifecycleOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io, Scheduler* scheduler) {
    return exercise_lifecycle(system_allocator(), io, scheduler, root,
                              &outcome);
  });
  EXPECT_TRUE(outcome.deadline_reported);
  EXPECT_TRUE(outcome.rescan_reported);
  EXPECT_TRUE(outcome.atomic_replace_reported);
  EXPECT_TRUE(outcome.concurrent_next_rejected);
  EXPECT_TRUE(outcome.waiting_next_cancelled_by_close);
}

}  // namespace
