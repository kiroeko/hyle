#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__linux__)
#  include <sys/socket.h>
#  include <unistd.h>

#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#endif

#include "hyle/io/proactor.h"

using hyle::io::Completion;
using hyle::io::Proactor;

// io-01 is the completion-reaping seam; these tests prove the mechanism the real
// backends (io-02 io_uring / IOCP) drive: attach a 64-bit token to an operation,
// get it back verbatim on completion, reap in batch. On Windows this exercises
// the real IOCP port (PostQueuedCompletionStatus/GetQueuedCompletionStatusEx);
// on Linux the backend is io-02-pending so init() returns false and we skip.

TEST(Proactor, InitAndPostReapRoundTrip) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";
  ASSERT_TRUE(p.valid());

  // Post 3 completions carrying distinct tokens. post() is the backend-neutral
  // "inject a wake token" hook -- it guarantees the user_data (Waker token)
  // round-trips, NOT a result payload: IOCP can carry bytes, but io_uring's NOP
  // res is always 0. A real op's byte count is an io-04 concern; here we assert
  // only what post() promises on BOTH backends -- the token.
  EXPECT_TRUE(p.post(0xA1));
  EXPECT_TRUE(p.post(0xB2));
  EXPECT_TRUE(p.post(0xC3));

  // Reap them back (block up to 1s so a slow CI still catches all three).
  Completion out[8];
  std::size_t got = 0;
  while (got < 3) {
    std::size_t n = p.reap(out + got, 8 - got, 1'000'000'000ull);
    if (n == 0) break;  // timeout -> stop (test will fail the count check)
    got += n;
  }
  ASSERT_EQ(got, 3u);

  // Verify every token came back verbatim (order not guaranteed).
  std::uint64_t seen_tokens = 0;
  for (std::size_t i = 0; i < got; ++i) seen_tokens |= out[i].user_data;
  EXPECT_EQ(seen_tokens, 0xA1u | 0xB2u | 0xC3u);
}

TEST(Proactor, ReapEmptyIsNonBlockingWithZeroTimeout) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";
  Completion out[4];
  // Nothing posted -> a 0-timeout poll returns immediately with 0.
  auto t0 = std::chrono::steady_clock::now();
  std::size_t n = p.reap(out, 4, 0);
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_EQ(n, 0u);
  EXPECT_LT(elapsed, std::chrono::milliseconds(100));  // did not block
}

// The cross-thread wake path the real backend relies on: a consumer blocks in
// reap(); another thread posts a completion; the consumer wakes and gets the
// token. This is io->Waker's physical basis (post == "an op finished elsewhere").
TEST(Proactor, BlockedReapWokenByPostFromAnotherThread) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";

  std::atomic<std::uint64_t> got_token{0};
  std::thread consumer([&] {
    Completion out[4];
    std::size_t n = p.reap(out, 4, 5'000'000'000ull);  // block up to 5s
    if (n > 0) got_token.store(out[0].user_data, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));  // ensure it parked
  EXPECT_TRUE(p.post(0xDEAD, 42));  // wake it from here
  consumer.join();
  EXPECT_EQ(got_token.load(std::memory_order_acquire), 0xDEADu);
}

// Batch reap: post many, confirm reap pulls them in batches (io_uring
// for_each_cqe analogue -- amortized, not one syscall per completion).
TEST(Proactor, BatchReapManyCompletions) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";
  constexpr int kN = 100;
  for (int i = 0; i < kN; ++i) {
    ASSERT_TRUE(p.post(static_cast<std::uint64_t>(i + 1), i));
  }
  Completion out[64];
  int total = 0;
  std::uint64_t token_sum = 0;
  while (total < kN) {
    std::size_t n = p.reap(out, 64, 1'000'000'000ull);
    if (n == 0) break;
    for (std::size_t i = 0; i < n; ++i) token_sum += out[i].user_data;
    total += static_cast<int>(n);
  }
  EXPECT_EQ(total, kN);
  EXPECT_EQ(token_sum, static_cast<std::uint64_t>(kN) * (kN + 1) / 2);  // 1..100
}

// ---- io-02 real IO ops (io_uring on Linux) ----------------------------------
// A TRUE read/write closed loop (not NOP): write bytes to a temp file via
// submit_write, reap the completion (result == bytes written), then submit_read
// them back and verify the data + result. This proves the SQE fd/addr/len/off
// path and the CQE res mapping -- the actual point of io-02. IOCP real ops are
// still pending (submit_* returns false there), so this test is Linux-only.
#if defined(__linux__)
TEST(Proactor, RealReadWriteRoundTrip) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";

  // A temp file fd to do real IO against.
  char path[] = "/tmp/hyle_io_XXXXXX";
  int fd = ::mkstemp(path);
  ASSERT_GE(fd, 0);
  ::unlink(path);  // unlink now; fd stays valid until closed

  const char msg[] = "hyle-io-uring-real-op";
  const std::uint32_t len = sizeof(msg);  // includes NUL

  // Write at offset 0.
  ASSERT_TRUE(p.submit_write(fd, msg, len, 0, /*user_data=*/0x111));
  Completion c{};
  ASSERT_EQ(p.reap(&c, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c.user_data, 0x111u);
  EXPECT_EQ(c.result, static_cast<std::int64_t>(len));  // bytes written

  // Read it back at offset 0.
  char buf[64] = {};
  ASSERT_TRUE(p.submit_read(fd, buf, len, 0, /*user_data=*/0x222));
  Completion c2{};
  ASSERT_EQ(p.reap(&c2, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c2.user_data, 0x222u);
  EXPECT_EQ(c2.result, static_cast<std::int64_t>(len));  // bytes read
  EXPECT_STREQ(buf, msg);  // the actual bytes round-tripped through the kernel

  ::close(fd);
}

// An IO error surfaces as a NEGATIVE result (io_uring CQE res = -errno), not a
// submit failure: read from an invalid fd -> the submit succeeds, the completion
// carries -EBADF.
TEST(Proactor, RealReadBadFdReturnsNegativeResult) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";
  char buf[8] = {};
  ASSERT_TRUE(p.submit_read(/*fd=*/-1, buf, sizeof(buf), 0, 0x333));
  Completion c{};
  ASSERT_EQ(p.reap(&c, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c.user_data, 0x333u);
  EXPECT_LT(c.result, 0);  // negative errno (-EBADF)
}

// io-03 epoll fallback: force the epoll (reactor) backend via the env knob and
// drive a REAL read/write over a pipe (epoll rejects regular files, so a pipe
// is the honest fixture). Proves the reactor->completion emulation: submit_write
// arms EPOLLOUT + does the write when ready; submit_read arms EPOLLIN + reads.
// This is the "io_uring disabled" production path (hardened containers), run
// here by forcing it even though io_uring works on this box.
TEST(Proactor, EpollFallbackRealReadWrite) {
  ::setenv("HYLE_IO_FORCE_EPOLL", "1", /*overwrite=*/1);
  Proactor p;
  bool ok = p.init();
  ::unsetenv("HYLE_IO_FORCE_EPOLL");
  ASSERT_TRUE(ok);  // epoll backend must always stand up on Linux

  int fds[2];
  ASSERT_EQ(::pipe(fds), 0);
  int rd = fds[0], wr = fds[1];

  const char msg[] = "epoll-fallback";
  const std::uint32_t len = sizeof(msg);

  // Write to the pipe (EPOLLOUT ready immediately since the pipe has space).
  ASSERT_TRUE(p.submit_write(wr, msg, len, 0, 0x777));
  Completion c{};
  ASSERT_EQ(p.reap(&c, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c.user_data, 0x777u);
  EXPECT_EQ(c.result, static_cast<std::int64_t>(len));

  // Read it back (EPOLLIN ready now that data is in the pipe).
  char buf[64] = {};
  ASSERT_TRUE(p.submit_read(rd, buf, len, 0, 0x888));
  Completion c2{};
  ASSERT_EQ(p.reap(&c2, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c2.user_data, 0x888u);
  EXPECT_EQ(c2.result, static_cast<std::int64_t>(len));
  EXPECT_STREQ(buf, msg);

  ::close(rd);
  ::close(wr);
}

// The epoll backend also honors post() (software wake queue, no fd).
TEST(Proactor, EpollFallbackPostWake) {
  ::setenv("HYLE_IO_FORCE_EPOLL", "1", 1);
  Proactor p;
  bool ok = p.init();
  ::unsetenv("HYLE_IO_FORCE_EPOLL");
  ASSERT_TRUE(ok);
  ASSERT_TRUE(p.post(0x999));
  Completion c{};
  ASSERT_EQ(p.reap(&c, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c.user_data, 0x999u);
}

// io-04 socket ops: a real send/recv over a connected socketpair via io_uring
// IORING_OP_SEND/RECV. Proves the socket submit path (distinct opcodes from
// read/write) before the TcpListener/Stream layer is built on top.
TEST(Proactor, SocketSendRecvRoundTrip) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";

  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);

  const char msg[] = "io-uring-socket";
  const std::uint32_t len = sizeof(msg);

  // send on sv[0].
  ASSERT_TRUE(p.submit_send(sv[0], msg, len, 0, 0x5E4D));
  Completion c{};
  ASSERT_EQ(p.reap(&c, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c.user_data, 0x5E4Du);
  EXPECT_EQ(c.result, static_cast<std::int64_t>(len));  // bytes sent

  // recv on sv[1].
  char buf[64] = {};
  ASSERT_TRUE(p.submit_recv(sv[1], buf, len, 0, 0x8EC0));
  Completion c2{};
  ASSERT_EQ(p.reap(&c2, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c2.user_data, 0x8EC0u);
  EXPECT_EQ(c2.result, static_cast<std::int64_t>(len));  // bytes received
  EXPECT_STREQ(buf, msg);

  ::close(sv[0]);
  ::close(sv[1]);
}
#endif  // __linux__
