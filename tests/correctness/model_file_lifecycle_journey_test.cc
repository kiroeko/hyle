#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/io/transfer.h"
#include "hyle/io/watch.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/mapping.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

namespace {

using namespace std::chrono_literals;

class TempDirectory {
 public:
  TempDirectory() {
    static std::atomic<std::uint64_t> sequence{0};
    const auto suffix =
        static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        sequence.fetch_add(1, std::memory_order_relaxed);
    path_ = std::filesystem::temp_directory_path() /
            ("hyle_correctness_model_" + std::to_string(suffix));
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

hyle::Status write_file(std::string_view path,
                        std::string_view contents) noexcept {
  const std::string owned_path(path);
  auto opened = hyle::platform::open_file(
      owned_path.c_str(), hyle::platform::kOpenWrite |
                              hyle::platform::kOpenCreate |
                              hyle::platform::kOpenTruncate);
  if (!opened.has_value()) return std::move(opened).error();
  hyle::Status status = opened->write_all(contents.data(), contents.size());
  if (status.ok()) status = opened->sync();
  status.Update(opened->close());
  return status;
}

struct ModelJourneyResult {
  hyle::Status status{};
  bool published = false;
  bool watch_observed = false;
  bool async_file_observed = false;
  bool mapping_observed = false;
  bool failure_preserved_destination = false;
  bool mapping_released = false;
  std::atomic<bool> done{false};
};

hyle::sched::Task<int> run_model_lifecycle(
    hyle::mem::AllocatorRef allocator, hyle::io::IoContext* io,
    const std::string* source, const std::string* destination,
    const std::string* missing_source, ModelJourneyResult* result) noexcept {
  constexpr unsigned kOwner = 0;
  constexpr std::string_view kPayload = "quantized-model-shard-v1";
  const std::array targets{
      hyle::io::FileWatchTarget::file(*destination)};
  auto watcher_result = hyle::io::FileWatcher::create(
      allocator, *io, kOwner, targets);
  if (!watcher_result.has_value()) {
    result->status = std::move(watcher_result).error();
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  hyle::io::FileWatcher watcher = std::move(*watcher_result);

  auto plan = hyle::io::TransferPlan::local(
      allocator, {*source}, {*destination});
  if (!plan.has_value()) {
    result->status = std::move(plan).error();
  } else {
    hyle::io::TransferExecution execution;
    execution.owner_worker = kOwner;
    hyle::io::TransferReport report = co_await hyle::io::transfer(
        allocator, *io, std::move(*plan), execution);
    result->published = report.published();
    if (!report.ok()) result->status = std::move(report.status);
  }

  if (result->status.ok()) {
    auto batch = co_await watcher.next(
        {.deadline = hyle::time::deadline_after(5s)});
    if (!batch.has_value()) {
      result->status = std::move(batch).error();
    } else {
      for (const auto& event : batch->events()) {
        if (event.target_index == 0 && event.relative_path.empty() &&
            (hyle::io::contains(event.changes,
                                hyle::io::FileWatchChange::kCreated) ||
             hyle::io::contains(event.changes,
                                hyle::io::FileWatchChange::kContent) ||
             hyle::io::contains(event.changes,
                                hyle::io::FileWatchChange::kMovedTo))) {
          result->watch_observed = true;
        }
      }
    }
  }

  std::array<std::byte, kPayload.size()> file_bytes{};
  if (result->status.ok()) {
    auto opened = co_await hyle::io::open_file(
        *io, kOwner, *destination,
        hyle::io::FileOpenOptions{.access = hyle::io::FileAccess::kRead});
    if (!opened.has_value()) {
      result->status = std::move(opened).error();
    } else {
      hyle::io::File file = std::move(*opened);
      hyle::io::OperationOptions read_options;
      // Windows IOCP and Linux io_uring can retire a bounded native file
      // operation. macOS deliberately offloads regular-file IO because kqueue
      // is a readiness API; asking that path for a cancellable deadline must
      // return Unimplemented. Select the strongest portable contract through
      // the public capability instead of guessing from the operating system.
      if (io->capabilities(kOwner).has(
              hyle::io::IoCapability::kAsyncFileData)) {
        read_options.deadline = hyle::time::deadline_after(5s);
      }
      auto read = co_await hyle::io::read_exact_at(
          allocator, file, 0, file_bytes, read_options);
      if (!read.ok()) {
        result->status = std::move(read.status);
      } else if (read.bytes != file_bytes.size() ||
                 std::memcmp(file_bytes.data(), kPayload.data(),
                             kPayload.size()) != 0) {
        result->status =
            hyle::Status::DataLoss("published model read did not match source");
      } else {
        result->async_file_observed = true;
      }
      result->status.Update(co_await file.close());
    }
  }

  if (result->status.ok()) {
    auto mapped = hyle::mem::FileMapping::open(destination->c_str());
    if (!mapped.has_value()) {
      result->status = std::move(mapped).error();
    } else {
      hyle::mem::FileMapping mapping = std::move(*mapped);
      auto bytes = mapping.bytes();
      result->mapping_observed =
          bytes.has_value() && bytes->size() == kPayload.size() &&
          std::memcmp(bytes->data(), kPayload.data(), kPayload.size()) == 0;
      result->status.Update(mapping.reset());
      result->mapping_released = !mapping.valid();
    }
  }

  // A failed replacement must not damage the published model.  This is the
  // composition-level transactional guarantee, not another Transfer unit case.
  if (result->status.ok()) {
    hyle::io::TransferOptions options;
    options.partial_on_failure = hyle::io::TransferPartialPolicy::kDiscard;
    auto missing = hyle::io::TransferPlan::local(
        allocator, {*missing_source}, {*destination}, options);
    if (!missing.has_value()) {
      result->status = std::move(missing).error();
    } else {
      hyle::io::TransferExecution execution;
      execution.owner_worker = kOwner;
      auto failed = co_await hyle::io::transfer(
          allocator, *io, std::move(*missing), execution);
      auto preserved = hyle::mem::FileMapping::open(destination->c_str());
      if (!failed.ok() && preserved.has_value()) {
        auto bytes = preserved->bytes();
        result->failure_preserved_destination =
            bytes.has_value() && bytes->size() == kPayload.size() &&
            std::memcmp(bytes->data(), kPayload.data(), kPayload.size()) == 0;
        result->status.Update(preserved->reset());
      } else if (failed.ok()) {
        result->status = hyle::Status::Internal(
            "missing model source unexpectedly published");
      } else {
        result->status = std::move(preserved).error();
      }
    }
  }

  result->status.Update(watcher.close());
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

TEST(CorrectnessJourney,
     ModelFileLifecyclePublishesWatchesReadsMapsAndPreservesOldData) {
  constexpr std::string_view kPayload = "quantized-model-shard-v1";
  TempDirectory temporary;
  ASSERT_TRUE(temporary.valid());
  const std::string source =
      (temporary.path() / "weights.source").string();
  const std::string destination =
      (temporary.path() / "weights.bin").string();
  const std::string missing_source =
      (temporary.path() / "missing.source").string();
  ASSERT_TRUE(write_file(source, kPayload).ok());

  hyle::sched::Scheduler scheduler(
      2, hyle::mem::system_allocator(), 256);
  hyle::io::IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  ModelJourneyResult result;
  Drive drive{run_model_lifecycle(
      hyle::mem::system_allocator(), &io, &source, &destination,
      &missing_source, &result)};
  ASSERT_TRUE(drive.task.valid());
  hyle::sched::Fiber fiber(
      hyle::sched::kDefaultStackSize, &drive_entry, &drive);
  ASSERT_TRUE(scheduler.spawn_on(0, &fiber));
  scheduler.drain();

  EXPECT_TRUE(drive.task.done());
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.status.ok()) << result.status.message();
  EXPECT_TRUE(result.published);
  EXPECT_TRUE(result.watch_observed);
  EXPECT_TRUE(result.async_file_observed);
  EXPECT_TRUE(result.mapping_observed);
  EXPECT_TRUE(result.failure_preserved_destination);
  EXPECT_TRUE(result.mapping_released);
  scheduler.shutdown();
}
