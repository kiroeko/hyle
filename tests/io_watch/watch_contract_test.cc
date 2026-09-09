#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/io/watch.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::io::FileWatchBatchView;
using hyle::io::FileWatchChange;
using hyle::io::FileWatchOptions;
using hyle::io::FileWatchTarget;
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
    const auto suffix = static_cast<std::uint64_t>(
                            std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count()) ^
                        sequence.fetch_add(1, std::memory_order_relaxed);
    path_ = std::filesystem::temp_directory_path() /
            ("hyle_io_watch_contract_" + std::to_string(suffix));
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
  status.Update(opened->close());
  return status;
}

struct Drive {
  Task<int> task;
};

void drive_entry(void* context) noexcept {
  static_cast<Drive*>(context)->task.resume();
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

struct FailAfterAllocator {
  AllocatorRef backing = system_allocator();
  std::size_t calls = 0;
  std::size_t fail_after = 0;
  std::size_t live = 0;

  static void* allocate(void* opaque, std::size_t size,
                        std::size_t alignment) noexcept {
    auto& self = *static_cast<FailAfterAllocator*>(opaque);
    if (self.calls++ >= self.fail_after) return nullptr;
    void* value = self.backing.allocate(size, alignment);
    if (value != nullptr) ++self.live;
    return value;
  }

  static void deallocate(void* opaque, void* value, std::size_t size,
                         std::size_t alignment) noexcept {
    auto& self = *static_cast<FailAfterAllocator*>(opaque);
    if (value != nullptr) --self.live;
    self.backing.deallocate(value, size, alignment);
  }

  static void* reallocate(void* opaque, void* value, std::size_t old_size,
                          std::size_t new_size,
                          std::size_t alignment) noexcept {
    auto& self = *static_cast<FailAfterAllocator*>(opaque);
    if (value == nullptr) return allocate(opaque, new_size, alignment);
    if (new_size == 0) {
      deallocate(opaque, value, old_size, alignment);
      return nullptr;
    }
    if (self.calls++ >= self.fail_after) return nullptr;
    return self.backing.reallocate(value, old_size, new_size, alignment);
  }

  AllocatorRef ref() noexcept {
    return AllocatorRef(&allocate, &deallocate, &reallocate, this);
  }
};

struct FactoryOutcome {
  Status status{};
  bool empty_rejected = false;
  bool empty_path_rejected = false;
  bool nul_rejected = false;
  bool utf8_rejected = false;
  bool missing_root_rejected = false;
  bool file_as_directory_rejected = false;
  bool option_bounds_rejected = false;
  bool missing_exact_accepted = false;
  bool oom_sweep_completed = false;
};

Task<int> exercise_factory_contract(AllocatorRef allocator, IoContext* io,
                                    std::string root,
                                    FactoryOutcome* outcome) noexcept {
  auto empty = FileWatcher::create(allocator, *io, kOwner, {});
  outcome->empty_rejected =
      !empty.has_value() && empty.error().IsInvalidArgument();

  const std::array empty_path{FileWatchTarget::file({})};
  auto empty_name = FileWatcher::create(allocator, *io, kOwner, empty_path);
  outcome->empty_path_rejected =
      !empty_name.has_value() && empty_name.error().IsInvalidArgument();

  std::string embedded_nul = root + "/bad";
  embedded_nul.push_back('\0');
  embedded_nul += "tail";
  const std::array nul_target{FileWatchTarget::file(embedded_nul)};
  auto nul = FileWatcher::create(allocator, *io, kOwner, nul_target);
  outcome->nul_rejected =
      !nul.has_value() && nul.error().IsInvalidArgument();

  const std::string invalid_utf8 = root + "/\xC3\x28";
  const std::array utf8_target{FileWatchTarget::file(invalid_utf8)};
  auto utf8 = FileWatcher::create(allocator, *io, kOwner, utf8_target);
  outcome->utf8_rejected =
      !utf8.has_value() && utf8.error().IsInvalidArgument();

  const std::string missing_root = root + "/missing/target.bin";
  const std::array missing_root_target{FileWatchTarget::file(missing_root)};
  auto missing =
      FileWatcher::create(allocator, *io, kOwner, missing_root_target);
  outcome->missing_root_rejected =
      !missing.has_value() && missing.error().IsNotFound();

  const std::string ordinary_file = root + "/ordinary.bin";
  outcome->status = write_file(ordinary_file, "x");
  if (!outcome->status.ok()) co_return 0;
  const std::array wrong_kind{FileWatchTarget::directory(ordinary_file)};
  auto wrong = FileWatcher::create(allocator, *io, kOwner, wrong_kind);
  outcome->file_as_directory_rejected =
      !wrong.has_value() && wrong.error().IsFailedPrecondition();

  const std::array directory{FileWatchTarget::directory(root)};
  FileWatchOptions too_small;
  too_small.native_buffer_bytes = 512;
  auto small = FileWatcher::create(allocator, *io, kOwner, directory, too_small);
  FileWatchOptions no_events;
  no_events.max_batch_events = 0;
  auto events = FileWatcher::create(allocator, *io, kOwner, directory, no_events);
  FileWatchOptions no_paths;
  no_paths.max_batch_path_bytes = 0;
  auto paths = FileWatcher::create(allocator, *io, kOwner, directory, no_paths);
  outcome->option_bounds_rejected =
      !small.has_value() && small.error().IsOutOfRange() &&
      !events.has_value() && events.error().IsInvalidArgument() &&
      !paths.has_value() && paths.error().IsInvalidArgument();

  const std::string absent_file = root + "/published-later.bin";
  const std::array absent{FileWatchTarget::file(absent_file)};
  auto absent_watcher = FileWatcher::create(allocator, *io, kOwner, absent);
  outcome->missing_exact_accepted = absent_watcher.has_value();
  if (absent_watcher.has_value())
    outcome->status.Update(absent_watcher->close());
  if (!outcome->status.ok()) co_return 0;

  // Sweep every allocation boundary until one complete construction succeeds.
  // Every failed partial native setup must roll back to zero live allocations.
  for (std::size_t fail_after = 0; fail_after != 128; ++fail_after) {
    FailAfterAllocator failing;
    failing.fail_after = fail_after;
    {
      auto made = FileWatcher::create(failing.ref(), *io, kOwner, directory);
      if (made.has_value()) {
        outcome->status.Update(made->close());
        if (!outcome->status.ok()) co_return 0;
        outcome->oom_sweep_completed = true;
      } else if (!made.error().IsResourceExhausted()) {
        outcome->status = std::move(made).error().WithContext(
            "OOM sweep returned a non-resource error");
        co_return 0;
      }
    }
    if (failing.live != 0) {
      outcome->status =
          Status::Internal("FileWatcher leaked an allocation during OOM rollback");
      co_return 0;
    }
    if (outcome->oom_sweep_completed) break;
  }
  if (!outcome->oom_sweep_completed)
    outcome->status = Status::Internal(
        "FileWatcher did not construct within the bounded OOM sweep");
  co_return 0;
}

struct EventOutcome {
  Status status{};
  bool exact_zero = false;
  bool exact_duplicate = false;
  bool directory_name = false;
  bool second_root_unicode = false;
  bool moved_from = false;
  bool moved_to = false;
  bool metadata = false;
  bool removed = false;
  bool nonrecursive = false;
  bool event_won_cancel = false;
  bool generation_advanced = false;
  bool old_view_stable = false;
};

Task<int> exercise_event_contract(AllocatorRef allocator, IoContext* io,
                                  std::string first_root,
                                  std::string second_root,
                                  EventOutcome* outcome) noexcept {
  const std::string exact = first_root + "/model.bin";
  const std::string old_name = first_root + "/old-name.bin";
  const std::string new_name = first_root + "/new-name.bin";
  const std::string metadata_name = first_root + "/metadata.bin";
  const std::string nested_name = first_root + "/nested/hidden.bin";
  const std::string unicode_basename =
      "\xE6\xA8\xA1\xE5\x9E\x8B-" + std::string(120, 'x') + ".json";
  const std::string unicode_name = second_root + "/" + unicode_basename;

  outcome->status = write_file(old_name, "old");
  outcome->status.Update(write_file(metadata_name, "metadata"));
  if (!outcome->status.ok()) co_return 0;

  const std::array targets{
      FileWatchTarget::file(exact), FileWatchTarget::file(exact),
      FileWatchTarget::directory(first_root),
      FileWatchTarget::directory(second_root)};
  auto made = FileWatcher::create(allocator, *io, kOwner, targets);
  if (!made.has_value()) {
    outcome->status = std::move(made).error();
    co_return 0;
  }
  FileWatcher watcher = std::move(*made);

  outcome->status = write_file(exact, "model");
  outcome->status.Update(write_file(unicode_name, "unicode"));
  if (!outcome->status.ok()) {
    outcome->status.Update(watcher.close());
    co_return 0;
  }

  const auto first_deadline =
      hyle::time::deadline_after(std::chrono::seconds(5));
  std::uint64_t first_generation = 0;
  std::string_view borrowed_path;
  for (unsigned attempt = 0; attempt != 16 &&
                             !(outcome->exact_zero &&
                               outcome->exact_duplicate &&
                               outcome->directory_name &&
                               outcome->second_root_unicode);
       ++attempt) {
    auto batch = co_await watcher.next({.deadline = first_deadline});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
      outcome->status.Update(watcher.close());
      co_return 0;
    }
    if (first_generation == 0) first_generation = batch->generation();
    for (const auto& event : batch->events()) {
      if (event.target_index == 0 && event.relative_path.empty())
        outcome->exact_zero = true;
      if (event.target_index == 1 && event.relative_path.empty())
        outcome->exact_duplicate = true;
      if (event.target_index == 2 && event.relative_path == "model.bin") {
        outcome->directory_name = true;
        borrowed_path = event.relative_path;
      }
      if (event.target_index == 3 &&
          event.relative_path == unicode_basename)
        outcome->second_root_unicode = true;
    }
    if (!borrowed_path.empty()) {
      outcome->old_view_stable = borrowed_path == "model.bin";
    }
  }

  outcome->status =
      hyle::platform::rename_path(old_name.c_str(), new_name.c_str());
  if (outcome->status.ok())
    outcome->status = hyle::platform::set_readonly(metadata_name.c_str(), true);
  if (outcome->status.ok())
    outcome->status = hyle::platform::remove_file(exact.c_str());
  if (!outcome->status.ok()) {
    outcome->status.Update(watcher.close());
    co_return 0;
  }

  auto cancelled = hyle::sched::CancelToken::try_create_root(allocator);
  if (!cancelled.has_value()) {
    outcome->status = std::move(cancelled).error();
    outcome->status.Update(watcher.close());
    co_return 0;
  }
  cancelled->cancel();
  const auto event_deadline =
      hyle::time::deadline_after(std::chrono::seconds(5));
  for (unsigned attempt = 0; attempt != 24 &&
                             !(outcome->moved_from && outcome->moved_to &&
                               outcome->metadata && outcome->removed);
       ++attempt) {
    auto batch = co_await watcher.next(
        {.deadline = event_deadline, .cancel = &*cancelled});
    if (!batch.has_value()) {
      // A cancelled token may win before the native batch reaches HYLE. The
      // watcher remains valid; continue without cancellation to collect it.
      if (!batch.error().IsCancelled()) {
        outcome->status = std::move(batch).error();
        outcome->status.Update(watcher.close());
        co_return 0;
      }
      break;
    }
    outcome->event_won_cancel = true;
    outcome->generation_advanced = batch->generation() > first_generation;
    for (const auto& event : batch->events()) {
      if (event.target_index != 2) continue;
      outcome->moved_from = outcome->moved_from ||
                            (event.relative_path == "old-name.bin" &&
                             hyle::io::contains(event.changes,
                                                FileWatchChange::kMovedFrom));
      outcome->moved_to = outcome->moved_to ||
                          (event.relative_path == "new-name.bin" &&
                           hyle::io::contains(event.changes,
                                              FileWatchChange::kMovedTo));
      outcome->metadata = outcome->metadata ||
                          (event.relative_path == "metadata.bin" &&
                           hyle::io::contains(event.changes,
                                              FileWatchChange::kMetadata));
      outcome->removed = outcome->removed ||
                         (event.relative_path == "model.bin" &&
                          hyle::io::contains(event.changes,
                                             FileWatchChange::kRemoved));
    }
  }

  // Finish collecting after the deliberate cancellation race.
  for (unsigned attempt = 0; attempt != 24 &&
                             !(outcome->moved_from && outcome->moved_to &&
                               outcome->metadata && outcome->removed);
       ++attempt) {
    auto batch = co_await watcher.next({.deadline = event_deadline});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
      outcome->status.Update(watcher.close());
      co_return 0;
    }
    outcome->generation_advanced =
        outcome->generation_advanced || batch->generation() > first_generation;
    for (const auto& event : batch->events()) {
      if (event.target_index != 2) continue;
      outcome->moved_from = outcome->moved_from ||
                            (event.relative_path == "old-name.bin" &&
                             hyle::io::contains(event.changes,
                                                FileWatchChange::kMovedFrom));
      outcome->moved_to = outcome->moved_to ||
                          (event.relative_path == "new-name.bin" &&
                           hyle::io::contains(event.changes,
                                              FileWatchChange::kMovedTo));
      outcome->metadata = outcome->metadata ||
                          (event.relative_path == "metadata.bin" &&
                           hyle::io::contains(event.changes,
                                              FileWatchChange::kMetadata));
      outcome->removed = outcome->removed ||
                         (event.relative_path == "model.bin" &&
                          hyle::io::contains(event.changes,
                                             FileWatchChange::kRemoved));
    }
  }

  // A file below a pre-existing child directory must not escape the portable
  // non-recursive contract, even though FSEvents is recursive internally.
  outcome->status = write_file(nested_name, "hidden");
  if (outcome->status.ok()) {
    auto nested = co_await watcher.next(
        {.deadline = hyle::time::deadline_after(
             std::chrono::milliseconds(250))});
    outcome->nonrecursive =
        !nested.has_value() && nested.error().IsDeadlineExceeded();
    if (!nested.has_value() && !nested.error().IsDeadlineExceeded())
      outcome->status = std::move(nested).error();
  }

  outcome->status.Update(watcher.close());
  co_return 0;
}

struct RootOutcome {
  Status status{};
  bool invalidated = false;
  bool rescan = false;
  bool terminal = false;
  bool moved_from_inert = false;
  bool close_idempotent = false;
};

Task<int> exercise_root_and_move(AllocatorRef allocator, IoContext* io,
                                 std::string root,
                                 RootOutcome* outcome) noexcept {
  const std::array target{FileWatchTarget::directory(root)};
  auto made = FileWatcher::create(allocator, *io, kOwner, target);
  if (!made.has_value()) {
    outcome->status = std::move(made).error();
    co_return 0;
  }
  FileWatcher source = std::move(*made);
  FileWatcher watcher = std::move(source);
  outcome->moved_from_inert = !source.valid() && source.close().ok() &&
                              watcher.valid();

  outcome->status = hyle::platform::remove_directory(root.c_str());
  if (!outcome->status.ok()) {
    outcome->status.Update(watcher.close());
    co_return 0;
  }
  const auto deadline = hyle::time::deadline_after(std::chrono::seconds(5));
  for (unsigned attempt = 0; attempt != 8 && !outcome->invalidated; ++attempt) {
    auto batch = co_await watcher.next({.deadline = deadline});
    if (!batch.has_value()) {
      outcome->terminal = batch.error().IsNotFound();
      if (!outcome->terminal) outcome->status = std::move(batch).error();
      break;
    }
    outcome->rescan = outcome->rescan || batch->rescan_required();
    for (const auto& event : batch->events()) {
      outcome->invalidated =
          outcome->invalidated ||
          hyle::io::contains(event.changes,
                             FileWatchChange::kRootInvalidated);
    }
  }
  if (outcome->invalidated && !outcome->terminal) {
    auto terminal = co_await watcher.next({.deadline = deadline});
    outcome->terminal =
        !terminal.has_value() && terminal.error().IsNotFound();
    if (!terminal.has_value() && !terminal.error().IsNotFound())
      outcome->status = std::move(terminal).error();
  }
  const Status first_close = watcher.close();
  const Status second_close = watcher.close();
  outcome->status.Update(first_close);
  outcome->status.Update(second_close);
  outcome->close_idempotent = first_close.ok() && second_close.ok();
  co_return 0;
}

TEST(FileWatchContract, FactoryValidationAndOomRollbackAreTyped) {
  TempDirectory temporary;
  FactoryOutcome outcome;
  const std::string root = temporary.path().string();
  run_watch_journey(&outcome, [&](IoContext* io, Scheduler*) {
    return exercise_factory_contract(system_allocator(), io, root, &outcome);
  });
  EXPECT_TRUE(outcome.empty_rejected);
  EXPECT_TRUE(outcome.empty_path_rejected);
  EXPECT_TRUE(outcome.nul_rejected);
  EXPECT_TRUE(outcome.utf8_rejected);
  EXPECT_TRUE(outcome.missing_root_rejected);
  EXPECT_TRUE(outcome.file_as_directory_rejected);
  EXPECT_TRUE(outcome.option_bounds_rejected);
  EXPECT_TRUE(outcome.missing_exact_accepted);
  EXPECT_TRUE(outcome.oom_sweep_completed);
}

TEST(FileWatchContract, MultipleTargetsKindsAndNonrecursiveFilterArePortable) {
  TempDirectory first;
  TempDirectory second;
  std::filesystem::create_directory(first.path() / "nested");
  EventOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io, Scheduler*) {
    return exercise_event_contract(system_allocator(), io,
                                   first.path().string(),
                                   second.path().string(), &outcome);
  });
  EXPECT_TRUE(outcome.exact_zero);
  EXPECT_TRUE(outcome.exact_duplicate);
  EXPECT_TRUE(outcome.directory_name);
  EXPECT_TRUE(outcome.second_root_unicode);
  EXPECT_TRUE(outcome.moved_from);
  EXPECT_TRUE(outcome.moved_to);
  EXPECT_TRUE(outcome.metadata);
  EXPECT_TRUE(outcome.removed);
  EXPECT_TRUE(outcome.nonrecursive);
  EXPECT_TRUE(outcome.generation_advanced);
  EXPECT_TRUE(outcome.old_view_stable);
  // Depending on native dispatch timing either the pending event or the
  // already-cancelled wait can win. Both are legal and the remaining loop
  // proves the watcher stays reusable.
  (void)outcome.event_won_cancel;
}

TEST(FileWatchContract, RootLossAndMoveOwnershipAreExplicit) {
  TempDirectory temporary;
  RootOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io, Scheduler*) {
    return exercise_root_and_move(system_allocator(), io,
                                  temporary.path().string(), &outcome);
  });
  EXPECT_TRUE(outcome.invalidated);
  EXPECT_TRUE(outcome.rescan);
  EXPECT_TRUE(outcome.terminal);
  EXPECT_TRUE(outcome.moved_from_inert);
  EXPECT_TRUE(outcome.close_idempotent);
}

TEST(FileWatchContract, InvalidAllocatorAndContextFailBeforeOwnerWork) {
  TempDirectory temporary;
  const std::array target{
      FileWatchTarget::directory(temporary.path().string())};
  Status scheduler_status;
  Scheduler scheduler(1, system_allocator(), 256, &scheduler_status);
  ASSERT_TRUE(scheduler_status.ok());
  hyle::io::IoContextOptions invalid_options;
  invalid_options.allocator = {};
  IoContext invalid_context(scheduler, invalid_options);
  ASSERT_FALSE(invalid_context.valid());
  auto no_allocator =
      FileWatcher::create({}, invalid_context, kOwner, target);
  EXPECT_FALSE(no_allocator.has_value());
  EXPECT_TRUE(no_allocator.error().IsInvalidArgument());

  auto no_context = FileWatcher::create(system_allocator(), invalid_context,
                                        kOwner, target);
  EXPECT_FALSE(no_context.has_value());
  EXPECT_TRUE(no_context.error().IsFailedPrecondition());
  scheduler.shutdown();
}

}  // namespace
