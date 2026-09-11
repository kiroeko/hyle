#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

using hyle::io::File;
using hyle::io::ConstBuffer;
using hyle::io::FileAccess;
using hyle::io::FileCacheMode;
using hyle::io::FileOpenOptions;
using hyle::io::FileSyncMode;
using hyle::io::IoContext;
using hyle::io::MutableBuffer;
using hyle::io::open_file;
using hyle::io::read_exact_at;
using hyle::io::write_all_at;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;

struct FileResult {
  std::atomic<bool> done{false};
  std::atomic<bool> opened{false};
  std::atomic<bool> write_ok{false};
  std::atomic<bool> read_ok{false};
  std::atomic<bool> info_ok{false};
  std::atomic<bool> sync_ok{false};
  std::atomic<bool> close_ok{false};
  std::atomic<bool> expected_open_error{false};
  std::atomic<bool> direct_contract{false};
  std::atomic<std::uint64_t> size{0};
  char got[64]{};
};

std::string temp_path() {
  static std::atomic<unsigned> counter{0};
  return "hyle_file_test_" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) +
         ".tmp";
}

Task<int> file_coro(AllocatorRef allocator, IoContext *io, const char *path,
                    FileResult *result) {
  static constexpr char kMessage[] = "hyle-file-io-roundtrip";
  auto requested = open_file(
      *io, kOwner, path,
      FileOpenOptions{
          .access = FileAccess::kReadWrite, .create = true, .truncate = true});
  auto moved = std::move(requested);
  auto operation = open_file(
      *io, kOwner, "hyle-file-replaced-awaitable.tmp",
      FileOpenOptions{.access = FileAccess::kRead});
  operation = std::move(moved);
  auto opened = co_await operation;
  if (!opened.has_value()) {
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  File file = std::move(*opened);
  result->opened.store(true, std::memory_order_release);

  const ConstBuffer output[] = {
      {kMessage, 4}, {kMessage + 4, sizeof(kMessage) - 4}};
  auto written = co_await write_all_at(
      allocator, file, 0, std::span<const ConstBuffer>{output});
  result->write_ok.store(
      written.ok() && written.progress.bytes == sizeof(kMessage),
                         std::memory_order_release);

  hyle::Status synced = co_await file.sync(FileSyncMode::kDataOnly);
  result->sync_ok.store(synced.ok(), std::memory_order_release);

  const MutableBuffer input[] = {
      {result->got, 7}, {result->got + 7, sizeof(kMessage) - 7}};
  auto read = co_await read_exact_at(
      allocator, file, 0, std::span<const MutableBuffer>{input});
  result->read_ok.store(
      read.ok() && read.progress.bytes == sizeof(kMessage),
                        std::memory_order_release);

  auto info = co_await file.info();
  result->info_ok.store(info.has_value(), std::memory_order_release);
  if (info.has_value())
    result->size.store(info->size, std::memory_order_release);

  hyle::Status closed = co_await file.close();
  result->close_ok.store(closed.ok(), std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> missing_coro(AllocatorRef, IoContext *io, FileResult *result) {
  auto opened =
      co_await open_file(*io, kOwner, "hyle_no_such_file_9d3f.tmp",
                         FileOpenOptions{.access = FileAccess::kRead});
  result->expected_open_error.store(!opened.has_value() &&
                                        opened.error().IsNotFound(),
                                    std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> preallocate_coro(AllocatorRef, IoContext *io, const char *path,
                           FileResult *result) {
  auto opened = co_await open_file(
      *io, kOwner, path,
      FileOpenOptions{
          .access = FileAccess::kReadWrite, .create = true, .truncate = true});
  if (!opened.has_value()) {
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  File file = std::move(*opened);
  result->opened.store(true, std::memory_order_release);
  hyle::Status allocated = co_await file.preallocate(0, 4096);
  result->write_ok.store(allocated.ok(), std::memory_order_release);
  auto info = co_await file.info();
  result->info_ok.store(info.has_value(), std::memory_order_release);
  if (info.has_value())
    result->size.store(info->size, std::memory_order_release);
  (void)(co_await file.close());
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> adopt_coro(AllocatorRef allocator, IoContext *io,
                     hyle::platform::FileHandle *owner, FileResult *result) {
  auto adopted = File::adopt(*io, kOwner, std::move(*owner));
  if (!adopted.has_value()) {
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  File file = std::move(*adopted);
  result->opened.store(true, std::memory_order_release);
  auto read = co_await read_exact_at(
      allocator, file, 0,
      std::as_writable_bytes(std::span{result->got, std::size_t{5}}));
  result->read_ok.store(read.ok() && read.bytes == 5,
                        std::memory_order_release);
  *owner = file.release();
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> direct_coro(AllocatorRef, IoContext *io, const char *path,
                      FileResult *result) {
  auto opened =
      co_await open_file(*io, kOwner, path,
                         FileOpenOptions{.access = FileAccess::kReadWrite,
                                         .create = true,
                                         .truncate = true,
                                         .cache = FileCacheMode::kDirect});
  if (!opened.has_value()) {
    result->expected_open_error.store(true, std::memory_order_release);
    result->done.store(true, std::memory_order_release);
    co_return 0;
  }
  File file = std::move(*opened);
  const auto alignment = file.direct_alignment();
  std::byte deliberately_misaligned[2]{};
  auto transfer = co_await file.read_some_at(
      1, std::span<std::byte>(deliberately_misaligned + 1, 1));
  result->direct_contract.store(file.cache_mode() == FileCacheMode::kDirect &&
                                    alignment.memory_alignment > 0 &&
                                    transfer.status.IsInvalidArgument(),
                                std::memory_order_release);
  (void)(co_await file.close());
  result->done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

template <class MakeTask>
void run_one(Scheduler &scheduler, FileResult &result, MakeTask &&make_task) {
  Drive driver{make_task()};
  Fiber fiber(kDefaultStackSize, &drive_entry, &driver);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(result.done.load(std::memory_order_acquire));
}

} // namespace

TEST(IoFile, TypedOpenWriteReadInfoSyncClose) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  const std::string path = temp_path();
  FileResult result;
  run_one(scheduler, result, [&] {
    return file_coro(system_allocator(), &io, path.c_str(), &result);
  });
  std::remove(path.c_str());

  static constexpr char kMessage[] = "hyle-file-io-roundtrip";
  EXPECT_TRUE(result.opened.load());
  EXPECT_TRUE(result.write_ok.load());
  EXPECT_TRUE(result.read_ok.load());
  EXPECT_TRUE(result.info_ok.load());
  EXPECT_TRUE(result.sync_ok.load());
  EXPECT_TRUE(result.close_ok.load());
  EXPECT_EQ(result.size.load(), sizeof(kMessage));
  EXPECT_STREQ(result.got, kMessage);
}

TEST(IoFile, MissingPathPreservesTypedFailure) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  FileResult result;
  run_one(scheduler, result,
          [&] { return missing_coro(system_allocator(), &io, &result); });
  EXPECT_TRUE(result.expected_open_error.load());
}

TEST(IoFile, PreallocateExtendsFile) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  const std::string path = temp_path();
  FileResult result;
  run_one(scheduler, result, [&] {
    return preallocate_coro(system_allocator(), &io, path.c_str(), &result);
  });
  std::remove(path.c_str());
  EXPECT_TRUE(result.opened.load());
  EXPECT_TRUE(result.write_ok.load());
  EXPECT_TRUE(result.info_ok.load());
  EXPECT_GE(result.size.load(), 4096u);
}

TEST(IoFile, PlatformOwnerCanBeAdoptedAndReleased) {
  const std::string path = temp_path();
  auto opened = hyle::platform::open_file(
      path.c_str(), hyle::platform::kOpenRead | hyle::platform::kOpenWrite |
                        hyle::platform::kOpenCreate |
                        hyle::platform::kOpenTruncate);
  ASSERT_TRUE(opened.has_value());
  hyle::platform::FileHandle owner = std::move(*opened);
  static constexpr char kData[] = "model";
  ASSERT_TRUE(owner.write_all(kData, 5).ok());

  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  FileResult result;
  run_one(scheduler, result,
          [&] { return adopt_coro(system_allocator(), &io, &owner, &result); });
  EXPECT_TRUE(owner.valid());
  EXPECT_TRUE(result.opened.load());
  EXPECT_TRUE(result.read_ok.load());
  EXPECT_EQ(std::string_view(result.got, 5), "model");
  EXPECT_TRUE(owner.close().ok());
  std::remove(path.c_str());
}

TEST(IoFile, DirectOpenNeverSilentlyBecomesBuffered) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  const std::string path = temp_path();
  FileResult result;
  run_one(scheduler, result, [&] {
    return direct_coro(system_allocator(), &io, path.c_str(), &result);
  });
  std::remove(path.c_str());
  EXPECT_TRUE(result.expected_open_error.load() ||
              result.direct_contract.load());
}
