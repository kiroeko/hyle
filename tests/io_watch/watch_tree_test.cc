#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/core/environment.h"
#include "hyle/io/watch.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

namespace {

using hyle::Status;
using hyle::io::FileWatchChange;
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
            ("hyle_io_watch_tree_" + std::to_string(suffix));
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
  Drive drive{factory(&io)};
  ASSERT_TRUE(drive.task.valid());
  Fiber fiber(hyle::sched::kDefaultStackSize, &drive_entry, &drive);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  ASSERT_TRUE(drive.task.done());
  ASSERT_TRUE(outcome->status.ok()) << outcome->status.message();
}

struct TreeOutcome {
  Status status{};
  bool root_relative = false;
  bool overlap_relative = false;
  bool direct_filtered = true;
  bool topology_reported = false;
  bool new_descendant = false;
  bool renamed_descendant = false;
};

Task<int> exercise_tree(AllocatorRef allocator, IoContext* io,
                        std::string root, std::string nested,
                        TreeOutcome* outcome) noexcept {
  const std::array targets{
      FileWatchTarget::directory(root),
      FileWatchTarget::directory_tree(root),
      FileWatchTarget::directory_tree(nested)};
  auto made = FileWatcher::create(allocator, *io, kOwner, targets);
  if (!made.has_value()) {
    outcome->status = std::move(made).error();
    co_return 0;
  }
  FileWatcher watcher = std::move(*made);
  const auto deadline = hyle::time::deadline_after(std::chrono::seconds(5));

  const std::string deep = nested + "/deep.bin";
  outcome->status = write_file(deep, "deep");
  for (unsigned attempt = 0; outcome->status.ok() && attempt != 16 &&
                             !(outcome->root_relative &&
                               outcome->overlap_relative);
       ++attempt) {
    auto batch = co_await watcher.next({.deadline = deadline});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
      break;
    }
    for (const auto& event : batch->events()) {
      if (event.target_index == 0 &&
          event.relative_path.find('/') != std::string_view::npos)
        outcome->direct_filtered = false;
      if (event.target_index == 1 && event.relative_path == "nested/deep.bin")
        outcome->root_relative = true;
      if (event.target_index == 2 && event.relative_path == "deep.bin")
        outcome->overlap_relative = true;
    }
  }

  const std::string dynamic = root + "/dynamic";
  if (outcome->status.ok())
    outcome->status = hyle::platform::make_directory(dynamic.c_str());
  for (unsigned attempt = 0; outcome->status.ok() && attempt != 16 &&
                             !outcome->topology_reported;
       ++attempt) {
    auto batch = co_await watcher.next({.deadline = deadline});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
      break;
    }
    outcome->topology_reported = batch->rescan_required();
    for (const auto& event : batch->events()) {
      if (event.target_index == 1 && event.relative_path == "dynamic" &&
          hyle::io::contains(event.changes, FileWatchChange::kCreated))
        outcome->topology_reported = true;
    }
  }

  const std::string after_create = dynamic + "/after-create.bin";
  if (outcome->status.ok())
    outcome->status = write_file(after_create, "new child");
  for (unsigned attempt = 0; outcome->status.ok() && attempt != 16 &&
                             !outcome->new_descendant;
       ++attempt) {
    auto batch = co_await watcher.next({.deadline = deadline});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
      break;
    }
    for (const auto& event : batch->events()) {
      if (event.target_index == 1 &&
          event.relative_path == "dynamic/after-create.bin")
        outcome->new_descendant = true;
    }
  }

  const std::string renamed = root + "/renamed";
  if (outcome->status.ok())
    outcome->status =
        hyle::platform::rename_path(dynamic.c_str(), renamed.c_str());
  bool rename_observed = false;
  for (unsigned attempt = 0; outcome->status.ok() && attempt != 16 &&
                             !rename_observed;
       ++attempt) {
    auto batch = co_await watcher.next({.deadline = deadline});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
      break;
    }
    rename_observed = batch->rescan_required();
    for (const auto& event : batch->events()) {
      if (event.target_index == 1 &&
          (event.relative_path == "dynamic" ||
           event.relative_path == "renamed"))
        rename_observed = true;
    }
  }

  const std::string after_rename = renamed + "/after-rename.bin";
  if (outcome->status.ok())
    outcome->status = write_file(after_rename, "renamed child");
  for (unsigned attempt = 0; outcome->status.ok() && attempt != 16 &&
                             !outcome->renamed_descendant;
       ++attempt) {
    auto batch = co_await watcher.next({.deadline = deadline});
    if (!batch.has_value()) {
      outcome->status = std::move(batch).error();
      break;
    }
    for (const auto& event : batch->events()) {
      if (event.target_index == 1 &&
          event.relative_path == "renamed/after-rename.bin")
        outcome->renamed_descendant = true;
    }
  }

  outcome->status.Update(watcher.close());
  co_return 0;
}

struct CapacityOutcome {
  Status status{};
  bool platform_contract = false;
};

Task<int> exercise_capacity(AllocatorRef allocator, IoContext* io,
                            std::string root,
                            CapacityOutcome* outcome) noexcept {
  const std::array target{FileWatchTarget::directory_tree(root)};
  hyle::io::FileWatchOptions options;
  options.max_watched_directories = 1;
  auto made = FileWatcher::create(allocator, *io, kOwner, target, options);
#if HYLE_OS_LINUX
  outcome->platform_contract =
      !made.has_value() && made.error().IsResourceExhausted();
  if (made.has_value()) {
    outcome->status = Status::Internal(
        "recursive inotify tree ignored max_watched_directories");
    outcome->status.Update(made->close());
  } else if (!made.error().IsResourceExhausted()) {
    outcome->status = std::move(made).error();
  }
#else
  if (!made.has_value()) {
    outcome->status = std::move(made).error();
  } else {
    outcome->platform_contract = true;
    outcome->status = made->close();
  }
#endif
  co_return 0;
}

struct RejectionOutcome {
  Status status{};
  bool rejected = false;
};

Task<int> exercise_link_root(AllocatorRef allocator, IoContext* io,
                             std::string link,
                             RejectionOutcome* outcome) noexcept {
  const std::array target{FileWatchTarget::directory_tree(link)};
  auto made = FileWatcher::create(allocator, *io, kOwner, target);
  outcome->rejected =
      !made.has_value() && made.error().IsFailedPrecondition();
  if (made.has_value()) {
    outcome->status = Status::Internal(
        "recursive FileWatcher accepted a symlink or junction root");
    outcome->status.Update(made->close());
  } else if (!made.error().IsFailedPrecondition()) {
    outcome->status = std::move(made).error();
  }
  co_return 0;
}

TEST(FileWatchTree, DeepOverlapAndDynamicTopologyRemainObservable) {
  TempDirectory temporary;
  const std::filesystem::path nested = temporary.path() / "nested";
  std::filesystem::create_directory(nested);
  TreeOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io) {
    return exercise_tree(system_allocator(), io, temporary.path().string(),
                         nested.string(), &outcome);
  });
  EXPECT_TRUE(outcome.root_relative);
  EXPECT_TRUE(outcome.overlap_relative);
  EXPECT_TRUE(outcome.direct_filtered);
  EXPECT_TRUE(outcome.topology_reported);
  EXPECT_TRUE(outcome.new_descendant);
  EXPECT_TRUE(outcome.renamed_descendant);
}

TEST(FileWatchTree, NativeDirectoryCapacityIsExplicit) {
  TempDirectory temporary;
  std::filesystem::create_directory(temporary.path() / "nested");
  CapacityOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io) {
    return exercise_capacity(system_allocator(), io,
                             temporary.path().string(), &outcome);
  });
  EXPECT_TRUE(outcome.platform_contract);
}

TEST(FileWatchTree, RecursiveRootDoesNotFollowLink) {
  TempDirectory temporary;
  const std::string target = (temporary.path() / "target").string();
  const std::string link = (temporary.path() / "link").string();
  ASSERT_TRUE(hyle::platform::make_directory(target.c_str()).ok());
  const Status linked = hyle::platform::create_symlink(
      target.c_str(), link.c_str(), hyle::platform::SymlinkKind::kDirectory);
  if (!linked.ok()) GTEST_SKIP() << linked.message();
  RejectionOutcome outcome;
  run_watch_journey(&outcome, [&](IoContext* io) {
    return exercise_link_root(system_allocator(), io, link, &outcome);
  });
  EXPECT_TRUE(outcome.rejected);
}

}  // namespace
