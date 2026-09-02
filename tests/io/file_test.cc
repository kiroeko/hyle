#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// io-05 file io test: a real async open/write/read/stat/fsync/close round-trip
// over the proactor, using the RAII File wrapper + co_await. On io_uring these are
// native async ops; on IOCP read/write are OVERLAPPED + metadata synchronous; on
// epoll everything is synchronous + synthesized. The SAME co_await code exercises
// whichever backend is active (forced-epoll on Linux via HYLE_IO_FORCE_EPOLL).

using hyle::io::File;
using hyle::io::IoContext;
using hyle::io::open_file;
using hyle::io::Proactor;
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
  std::atomic<std::int64_t> wrote{0};
  std::atomic<std::int64_t> read_back{0};
  std::atomic<std::int64_t> stat_rc{0};
  std::atomic<std::uint64_t> stat_size{0};
  std::atomic<std::int64_t> stat_seconds{0};
  std::atomic<std::uint32_t> stat_nanoseconds{0};
  std::atomic<std::int64_t> sync_rc{0};
  std::atomic<std::int64_t> close_rc{0};
  char got[64] = {};
};

// Build a unique temp path (kept simple + cross-platform: cwd-relative file).
std::string temp_path() {
  static std::atomic<int> ctr{0};
  return "hyle_file_test_" + std::to_string(ctr.fetch_add(1)) + ".tmp";
}

// open (create+trunc+rw) -> write -> fsync -> read back at offset 0 -> stat ->
// close. The message must round-trip and stat must report its size.
Task<int> file_coro(AllocatorRef, IoContext* io, const char* path, FileResult* r) {
  static const char kMsg[] = "hyle-file-io-roundtrip";
  File f = co_await open_file(*io, kOwner, path,
                              Proactor::kRead | Proactor::kWrite |
                                  Proactor::kCreate | Proactor::kTruncate);
  if (!f.valid()) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->opened.store(true, std::memory_order_release);

  std::int64_t w = co_await f.write(kMsg, sizeof(kMsg), 0);
  r->wrote.store(w, std::memory_order_release);

  r->sync_rc.store(co_await f.sync(/*datasync=*/true), std::memory_order_release);

  std::int64_t rd = co_await f.read(r->got, sizeof(kMsg), 0);
  r->read_back.store(rd, std::memory_order_release);

  Proactor::FileStat st;
  r->stat_rc.store(co_await f.stat(&st), std::memory_order_release);
  r->stat_size.store(st.size, std::memory_order_release);
  r->stat_seconds.store(st.modified_time.seconds, std::memory_order_release);
  r->stat_nanoseconds.store(st.modified_time.nanoseconds,
                            std::memory_order_release);

  r->close_rc.store(co_await f.close(), std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// Open a missing file WITHOUT create -> must fail (invalid File), not crash/hang.
Task<int> open_missing_coro(AllocatorRef, IoContext* io, FileResult* r) {
  File f = co_await open_file(*io, kOwner, "hyle_no_such_file_9d3f.tmp",
                              Proactor::kRead);
  r->opened.store(f.valid(), std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

// Preallocate 4096 bytes, then stat must report the file is at least that big.
Task<int> fallocate_coro(AllocatorRef, IoContext* io, const char* path,
                         FileResult* r, std::uint32_t mode) {
  File f = co_await open_file(*io, kOwner, path,
                              Proactor::kRead | Proactor::kWrite |
                                  Proactor::kCreate | Proactor::kTruncate);
  if (!f.valid()) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->opened.store(true, std::memory_order_release);
  std::int64_t alloc_rc = co_await f.allocate(0, 4096, mode);
  r->wrote.store(alloc_rc, std::memory_order_release);  // reuse: alloc rc
  Proactor::FileStat st;
  r->stat_rc.store(co_await f.stat(&st), std::memory_order_release);
  r->stat_size.store(st.size, std::memory_order_release);
  co_await f.close();
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

}  // namespace

TEST(IoFile, OpenWriteReadStatSyncClose) {
  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  std::string path = temp_path();
  auto r = std::make_unique<FileResult>();
  auto d = std::make_unique<Drive>();
  d->task = file_coro(system_allocator(), &io, path.c_str(), r.get());
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  std::remove(path.c_str());  // cleanup the temp file

  static const char kMsg[] = "hyle-file-io-roundtrip";
  EXPECT_TRUE(r->done.load());
  EXPECT_TRUE(r->opened.load());
  EXPECT_EQ(r->wrote.load(), static_cast<std::int64_t>(sizeof(kMsg)));
  EXPECT_EQ(r->read_back.load(), static_cast<std::int64_t>(sizeof(kMsg)));
  EXPECT_STREQ(r->got, "hyle-file-io-roundtrip");
  EXPECT_EQ(r->sync_rc.load(), 0);
  EXPECT_EQ(r->stat_rc.load(), 0);
  EXPECT_EQ(r->stat_size.load(), static_cast<std::uint64_t>(sizeof(kMsg)));
  EXPECT_GT(r->stat_seconds.load(), 1'577'836'800);  // after 2020-01-01
  EXPECT_LT(r->stat_nanoseconds.load(), 1'000'000'000u);
  EXPECT_EQ(r->close_rc.load(), 0);
}

// A missing file opened WITHOUT create must fail (negative result -> invalid File),
// not crash or hang. Proves the error path (open failure -> synthesized negative
// completion) resumes the fiber cleanly on every backend.
TEST(IoFile, OpenMissingFails) {
  constexpr unsigned kWorkers = 1;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  auto r = std::make_unique<FileResult>();
  auto d = std::make_unique<Drive>();
  d->task = open_missing_coro(system_allocator(), &io, r.get());
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_FALSE(r->opened.load());  // open of a missing file (no create) -> invalid
}

// fallocate then stat: preallocating N bytes must make the file at least N bytes.
TEST(IoFile, FallocateExtends) {
  constexpr unsigned kWorkers = 1;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());

  std::string path = temp_path();
  auto r = std::make_unique<FileResult>();
  auto d = std::make_unique<Drive>();
  d->task =
      fallocate_coro(system_allocator(), &io, path.c_str(), r.get(), 0);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  std::remove(path.c_str());

  EXPECT_TRUE(r->done.load());
  EXPECT_TRUE(r->opened.load());
  EXPECT_EQ(r->wrote.load(), 0);  // fallocate succeeded
  EXPECT_EQ(r->stat_rc.load(), 0);
  EXPECT_GE(r->stat_size.load(), static_cast<std::uint64_t>(4096));
}

// Linux-specific fallocate flags are not part of the portable facade. Native
// io_uring may expose them through its extension leg, but IOCP/epoll/kqueue must
// reject rather than silently execute mode=0 and report success.
TEST(IoFile, NonzeroAllocationModeIsNotSilentlyIgnored) {
#if defined(__linux__)
  ::setenv("HYLE_IO_FORCE_EPOLL", "1", 1);
#endif
  constexpr unsigned kWorkers = 1;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
#if defined(__linux__)
  ::unsetenv("HYLE_IO_FORCE_EPOLL");
#endif
  ASSERT_TRUE(io.valid());

  std::string path = temp_path();
  auto r = std::make_unique<FileResult>();
  auto d = std::make_unique<Drive>();
  d->task =
      fallocate_coro(system_allocator(), &io, path.c_str(), r.get(), 1);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  std::remove(path.c_str());
  EXPECT_TRUE(r->done.load());
  EXPECT_TRUE(r->opened.load());
  EXPECT_LT(r->wrote.load(), 0);
}
