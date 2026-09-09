#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>

#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>

#  include <cstring>
#endif

#include "hyle/io/detail/proactor.h"

#include "timeout_internal.h"

using hyle::io::detail::BackendKind;
using hyle::io::detail::Completion;
using hyle::io::detail::Proactor;
using hyle::io::detail::ProactorCapability;

static_assert(hyle::io::detail::finite_timeout_milliseconds(0, 100) == 0);
static_assert(hyle::io::detail::finite_timeout_milliseconds(1, 100) == 1);
static_assert(
    hyle::io::detail::finite_timeout_milliseconds(999'999, 100) == 1);
static_assert(
    hyle::io::detail::finite_timeout_milliseconds(1'000'000, 100) == 1);
static_assert(
    hyle::io::detail::finite_timeout_milliseconds(1'000'001, 100) == 2);
static_assert(hyle::io::detail::finite_timeout_milliseconds(
                  (std::numeric_limits<std::uint64_t>::max)() - 1, 100) ==
              100);

// io-01 is the completion-reaping seam; these tests prove the mechanism the real
// backends (io_uring / IOCP) drive: attach a 64-bit token to an operation, get it
// back verbatim on completion, reap in batch. On Windows this exercises the real
// IOCP port (PostQueuedCompletionStatus/GetQueuedCompletionStatusEx); on Linux
// io_uring (or the epoll fallback when forced). BOTH platforms now carry real IO
// ops (file read/write + socket accept/connect/send/recv), tested below.

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

TEST(Proactor, CapabilitiesMatchSelectedBackend) {
  Proactor p;
  ASSERT_TRUE(p.init());
  const auto caps = p.capabilities();
  EXPECT_EQ(caps.backend, p.backend_kind());
  if (caps.backend == BackendKind::kIoUring) {
    EXPECT_TRUE(caps.has(ProactorCapability::kNativeMultishotAccept));
    EXPECT_TRUE(caps.has(ProactorCapability::kProvidedBufferRing));
    EXPECT_TRUE(caps.has(ProactorCapability::kFixedFiles));
    EXPECT_TRUE(caps.has(ProactorCapability::kAsyncFileMetadata));
    EXPECT_TRUE(p.has_async_file_metadata());
  } else {
    EXPECT_TRUE(caps.has(ProactorCapability::kEmulatedMultishotAccept));
    EXPECT_FALSE(caps.has(ProactorCapability::kProvidedBufferRing));
    EXPECT_FALSE(caps.has(ProactorCapability::kFixedFiles));
    EXPECT_FALSE(caps.has(ProactorCapability::kAsyncFileMetadata));
    EXPECT_FALSE(p.has_async_file_metadata());
  }
}

TEST(Proactor, QueueDepthBackpressuresAndReleasesUserOperations) {
#if defined(_WIN32)
  const std::wstring stem =
      L"\\\\.\\pipe\\hyle-proactor-depth-" +
      std::to_wstring(static_cast<unsigned long>(::GetCurrentProcessId()));
  const std::wstring first_name = stem + L"-a";
  const std::wstring second_name = stem + L"-b";
  HANDLE first = ::CreateNamedPipeW(
      first_name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0,
      nullptr);
  HANDLE second = ::CreateNamedPipeW(
      second_name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0,
      nullptr);
  ASSERT_NE(first, INVALID_HANDLE_VALUE);
  ASSERT_NE(second, INVALID_HANDLE_VALUE);

  {
    Proactor p;
    ASSERT_TRUE(p.init(1));
    ASSERT_TRUE(p.register_handle(reinterpret_cast<std::intptr_t>(first)));
    ASSERT_TRUE(p.register_handle(reinterpret_cast<std::intptr_t>(second)));
    ASSERT_TRUE(p.submit_pipe_accept(reinterpret_cast<std::intptr_t>(first),
                                     0xD001));
    EXPECT_FALSE(p.submit_pipe_accept(reinterpret_cast<std::intptr_t>(second),
                                      0xD002));

    HANDLE client = ::CreateFileW(first_name.c_str(), GENERIC_READ | GENERIC_WRITE,
                                  0, nullptr, OPEN_EXISTING, 0, nullptr);
    ASSERT_NE(client, INVALID_HANDLE_VALUE);
    Completion completion{};
    ASSERT_EQ(p.reap(&completion, 1, 1'000'000'000ull), 1u);
    EXPECT_EQ(completion.user_data, 0xD001u);
    ::CloseHandle(client);

    // Reaping returned the fixed slab slot, so the next operation can enter.
    ASSERT_TRUE(p.submit_pipe_accept(reinterpret_cast<std::intptr_t>(second),
                                     0xD002));
    client = ::CreateFileW(second_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                           nullptr, OPEN_EXISTING, 0, nullptr);
    ASSERT_NE(client, INVALID_HANDLE_VALUE);
    ASSERT_EQ(p.reap(&completion, 1, 1'000'000'000ull), 1u);
    EXPECT_EQ(completion.user_data, 0xD002u);
    ::CloseHandle(client);
  }
  ::CloseHandle(first);
  ::CloseHandle(second);
#else
  int first[2] = {-1, -1};
  int second[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, first), 0);
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, second), 0);
  for (int fd : {first[0], first[1], second[0], second[1]}) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ASSERT_GE(flags, 0);
    ASSERT_EQ(::fcntl(fd, F_SETFL, flags | O_NONBLOCK), 0);
  }
  char first_byte = 0;
  char second_byte = 0;
  {
    Proactor p;
    ASSERT_TRUE(p.init(1));
    ASSERT_TRUE(p.submit_recv(first[0], &first_byte, 1, 0, 0xD001));
    EXPECT_FALSE(p.submit_recv(second[0], &second_byte, 1, 0, 0xD002));
    ASSERT_EQ(::write(first[1], "a", 1), 1);
    Completion completion{};
    ASSERT_EQ(p.reap(&completion, 1, 1'000'000'000ull), 1u);
    EXPECT_EQ(completion.user_data, 0xD001u);
    EXPECT_EQ(first_byte, 'a');

    ASSERT_TRUE(p.submit_recv(second[0], &second_byte, 1, 0, 0xD002));
    ASSERT_EQ(::write(second[1], "b", 1), 1);
    ASSERT_EQ(p.reap(&completion, 1, 1'000'000'000ull), 1u);
    EXPECT_EQ(completion.user_data, 0xD002u);
    EXPECT_EQ(second_byte, 'b');
  }
  for (int fd : {first[0], first[1], second[0], second[1]}) ::close(fd);
#endif
}

TEST(Proactor, FullQueueStillAcceptsCancellationControlWork) {
  Completion completions[4] = {};
  bool saw_target = false;
#if defined(_WIN32)
  const std::wstring name =
      L"\\\\.\\pipe\\hyle-proactor-cancel-" +
      std::to_wstring(static_cast<unsigned long>(::GetCurrentProcessId()));
  HANDLE pipe = ::CreateNamedPipeW(
      name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0,
      nullptr);
  ASSERT_NE(pipe, INVALID_HANDLE_VALUE);
  {
    Proactor p;
    ASSERT_TRUE(p.init(1));
    ASSERT_TRUE(p.register_handle(reinterpret_cast<std::intptr_t>(pipe)));
    ASSERT_TRUE(p.submit_pipe_accept(reinterpret_cast<std::intptr_t>(pipe),
                                     0xCA01));
    ASSERT_TRUE(p.submit_cancel(0xCA01, 0xCA02));
    for (unsigned guard = 0; guard < 4 && p.in_flight(); ++guard) {
      const std::size_t n = p.reap(completions, 4, 1'000'000'000ull);
      ASSERT_GT(n, 0u);
      for (std::size_t i = 0; i < n; ++i)
        saw_target |= completions[i].user_data == 0xCA01u;
    }
  }
  ::CloseHandle(pipe);
#else
  int pair[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
  for (int fd : pair) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ASSERT_GE(flags, 0);
    ASSERT_EQ(::fcntl(fd, F_SETFL, flags | O_NONBLOCK), 0);
  }
  char byte = 0;
  {
    Proactor p;
    ASSERT_TRUE(p.init(1));
    ASSERT_TRUE(p.submit_recv(pair[0], &byte, 1, 0, 0xCA01));
    ASSERT_TRUE(p.submit_cancel(0xCA01, 0xCA02));
    for (unsigned guard = 0; guard < 4 && p.in_flight(); ++guard) {
      const std::size_t n = p.reap(completions, 4, 1'000'000'000ull);
      ASSERT_GT(n, 0u);
      for (std::size_t i = 0; i < n; ++i)
        saw_target |= completions[i].user_data == 0xCA01u;
    }
  }
  ::close(pair[0]);
  ::close(pair[1]);
#endif
  EXPECT_TRUE(saw_target);
}

#if defined(__linux__) || defined(__APPLE__)
TEST(Proactor, ReadinessBackendRejectsSecondSameDirectionOp) {
#  if defined(__linux__)
  ::setenv("HYLE_IO_FORCE_EPOLL", "1", 1);
#  endif
  Proactor p;
  const bool initialized = p.init();
#  if defined(__linux__)
  ::unsetenv("HYLE_IO_FORCE_EPOLL");
#  endif
  ASSERT_TRUE(initialized);

  int sv[2] = {-1, -1};
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  for (int fd : sv) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ASSERT_GE(flags, 0);
    ASSERT_EQ(::fcntl(fd, F_SETFL, flags | O_NONBLOCK), 0);
  }

  char first[8] = {};
  char second[8] = {};
  ASSERT_TRUE(p.submit_recv(sv[0], first, sizeof(first), 0, 0x101));
  // epoll/kqueue have one pending slot per fd+direction. Replacing the first
  // operation would orphan its caller, so the second submit must fail cleanly.
  EXPECT_FALSE(p.submit_recv(sv[0], second, sizeof(second), 0, 0x102));

  ASSERT_EQ(::write(sv[1], "x", 1), 1);
  Completion out[2];
  ASSERT_EQ(p.reap(out, 2, 1'000'000'000ull), 1u);
  EXPECT_EQ(out[0].user_data, 0x101u);
  EXPECT_EQ(out[0].result, 1);
  EXPECT_EQ(first[0], 'x');

  ::close(sv[0]);
  ::close(sv[1]);
}

#if defined(__linux__)
TEST(Proactor, NativeStatxCompletesAsynchronously) {
  Proactor p;
  ASSERT_TRUE(p.init());
  if (p.backend_kind() != BackendKind::kIoUring)
    GTEST_SKIP() << "native STATX requires io_uring";

  char path[] = "/tmp/hyle-statx-XXXXXX";
  int fd = ::mkstemp(path);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(::unlink(path), 0);
  constexpr char kBytes[] = "statx-native";
  ASSERT_EQ(::write(fd, kBytes, sizeof(kBytes)),
            static_cast<ssize_t>(sizeof(kBytes)));

  Proactor::FileStat stat{};
  stat.size = UINT64_MAX;
  ASSERT_TRUE(p.submit_statx(fd, &stat, 0x57A7));
  EXPECT_EQ(stat.size, UINT64_MAX)
      << "native statx must not execute synchronously in submit";
  Completion completion{};
  ASSERT_EQ(p.reap(&completion, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(completion.user_data, 0x57A7u);
  EXPECT_EQ(completion.result, 0);
  EXPECT_EQ(stat.size, sizeof(kBytes));
  EXPECT_FALSE(stat.is_directory);
  EXPECT_GT(stat.modified_time.seconds, 1'577'836'800);  // after 2020-01-01
  EXPECT_LT(stat.modified_time.nanoseconds, 1'000'000'000u);
  ::close(fd);
}
#endif
#endif

// post() injects a completion carrying a token that reap() returns. On io_uring
// post fills a NOP SQE on the SUBMITTING thread -- the submission queue is
// single-producer, so post is a SAME-THREAD hook (the header documents this).
// This drives it same-thread: post three tokens, then reap them back. (Cross-
// thread wake of a blocked reap is a separate concern; the real scheduler wakes
// a parked worker via its futex Parker, not a cross-thread post into the ring.)
TEST(Proactor, PostInjectsTokenReapReturnsIt) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";

  EXPECT_TRUE(p.post(0xDEAD, 42));
  EXPECT_TRUE(p.post(0xBEEF));

  std::uint64_t seen = 0;
  Completion out[8];
  std::size_t got = 0;
  while (got < 2) {
    std::size_t n = p.reap(out + got, 8 - got, 1'000'000'000ull);
    if (n == 0) break;
    got += n;
  }
  ASSERT_EQ(got, 2u);
  for (std::size_t i = 0; i < got; ++i) seen |= out[i].user_data;
  EXPECT_EQ(seen, 0xDEADu | 0xBEEFu);
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
  // io_uring-specific: exercises the SQE fd/addr/len/off path + CQE res mapping.
  // Under forced-epoll the same submit_read/write go through the reactor's
  // regular-file synchronous fallback (a different path, covered by the io-05 file
  // tests), so skip here to keep this test about the io_uring op path.
  if (::getenv("HYLE_IO_FORCE_EPOLL") != nullptr)
    GTEST_SKIP() << "io_uring-specific (forced-epoll uses the sync file fallback)";
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
  // io_uring-specific: a bad fd's error surfaces as a NEGATIVE completion result
  // (submit succeeds, CQE res = -EBADF). The epoll reactor rejects a bad fd at
  // submit time instead (a different, backend-specific contract), so skip when
  // forced-epoll.
  if (::getenv("HYLE_IO_FORCE_EPOLL") != nullptr)
    GTEST_SKIP() << "io_uring-specific negative-result contract";
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

// A full TCP accept+connect handshake over loopback, then a send/recv on the
// accepted connection. Driven against WHATEVER backend `p` is (io_uring by
// default; the epoll fallback in the forced test below) -- so the SAME closed
// loop proves both submit_accept/submit_connect paths, not just io_uring's.
static void RunTcpAcceptConnectRoundTrip(Proactor& p) {
  // Listener bound to an ephemeral loopback port.
  int ls = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(ls, 0);
  int one = 1;
  ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ASSERT_EQ(::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);
  ASSERT_EQ(::listen(ls, 8), 0);
  socklen_t slen = sizeof(sa);
  ASSERT_EQ(::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen), 0);

  // Client socket, non-blocking so the epoll path sees EINPROGRESS (io_uring is
  // async regardless). Arm accept on the listener, then kick off the connect.
  int cl = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(cl, 0);
  int fl = ::fcntl(cl, F_GETFL, 0);
  ::fcntl(cl, F_SETFL, fl | O_NONBLOCK);

  sockaddr_storage peer{};
  socklen_t peer_len = sizeof(peer);
  ASSERT_TRUE(p.submit_accept(ls, &peer, &peer_len, /*user_data=*/0xACCE));
  ASSERT_TRUE(
      p.submit_connect(cl, &sa, sizeof(sa), /*user_data=*/0xC0DE));

  // Reap both completions (order/batching not guaranteed): the accept carries a
  // fresh fd (>= 0) in its result, the connect carries 0 (connected).
  int accepted = -1;
  bool got_accept = false, got_connect = false;
  for (int spins = 0; spins < 100 && !(got_accept && got_connect); ++spins) {
    Completion c[4];
    std::size_t n = p.reap(c, 4, 1'000'000'000ull);
    for (std::size_t i = 0; i < n; ++i) {
      if (c[i].user_data == 0xACCE) {
        got_accept = true;
        ASSERT_GE(c[i].result, 0) << "accept result must be a valid fd";
        accepted = static_cast<int>(c[i].result);
      } else if (c[i].user_data == 0xC0DE) {
        got_connect = true;
        EXPECT_EQ(c[i].result, 0) << "connect result must be 0 (connected)";
      }
    }
  }
  ASSERT_TRUE(got_accept);
  ASSERT_TRUE(got_connect);
  ASSERT_GE(accepted, 0);
  ASSERT_EQ(peer.ss_family, AF_INET);
  ASSERT_GE(peer_len, sizeof(sockaddr_in));
  sockaddr_in client_local{};
  socklen_t client_local_len = sizeof(client_local);
  ASSERT_EQ(::getsockname(cl, reinterpret_cast<sockaddr*>(&client_local),
                         &client_local_len),
            0);
  EXPECT_EQ(reinterpret_cast<const sockaddr_in*>(&peer)->sin_port,
            client_local.sin_port);

  // The accepted fd is a real, usable connection: send on the client, recv on
  // the accepted end (via the proactor, exercising submit_send/submit_recv too).
  const char msg[] = "tcp-accept-connect";
  const std::uint32_t len = sizeof(msg);
  ASSERT_TRUE(p.submit_send(cl, msg, len, 0, 0x5E4D));
  Completion cs{};
  ASSERT_EQ(p.reap(&cs, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(cs.user_data, 0x5E4Du);
  EXPECT_EQ(cs.result, static_cast<std::int64_t>(len));

  char buf[64] = {};
  ASSERT_TRUE(p.submit_recv(accepted, buf, len, 0, 0x8EC0));
  Completion cr{};
  ASSERT_EQ(p.reap(&cr, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(cr.user_data, 0x8EC0u);
  EXPECT_EQ(cr.result, static_cast<std::int64_t>(len));
  EXPECT_STREQ(buf, msg);

  // Non-zero socket flags are part of the portable contract. Prove send and
  // recv independently with TCP urgent data; a readiness backend that silently
  // substituted read/write(flags=0) would fail one of these checks.
  const char urgent_out = '!';
  ASSERT_TRUE(p.submit_send(cl, &urgent_out, 1, MSG_OOB, 0x50B));
  Completion urgent_send{};
  ASSERT_EQ(p.reap(&urgent_send, 1, 1'000'000'000ull), 1u);
  ASSERT_EQ(urgent_send.result, 1);
  char urgent_seen = 0;
  ASSERT_EQ(::recv(accepted, &urgent_seen, 1, MSG_OOB), 1);
  EXPECT_EQ(urgent_seen, urgent_out);

  const char urgent_in = '?';
  ASSERT_EQ(::send(cl, &urgent_in, 1, MSG_OOB), 1);
  char urgent_recv = 0;
  ASSERT_TRUE(p.submit_recv(accepted, &urgent_recv, 1, MSG_OOB, 0xB05));
  Completion urgent_completion{};
  ASSERT_EQ(p.reap(&urgent_completion, 1, 1'000'000'000ull), 1u);
  ASSERT_EQ(urgent_completion.result, 1);
  EXPECT_EQ(urgent_recv, urgent_in);

  ::close(accepted);
  ::close(cl);
  ::close(ls);
}

// io_uring path: real IORING_OP_ACCEPT + IORING_OP_CONNECT over loopback TCP.
TEST(Proactor, TcpAcceptConnectRoundTripUring) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";
  RunTcpAcceptConnectRoundTrip(p);
}

// epoll fallback path: submit_accept arms EPOLLIN + does ::accept when ready;
// submit_connect kicks off the non-blocking connect + arms EPOLLOUT, reads
// SO_ERROR when writable. Same loopback closed loop, forced onto the reactor.
TEST(Proactor, TcpAcceptConnectRoundTripEpoll) {
  ::setenv("HYLE_IO_FORCE_EPOLL", "1", 1);
  Proactor p;
  bool ok = p.init();
  ::unsetenv("HYLE_IO_FORCE_EPOLL");
  ASSERT_TRUE(ok);
  RunTcpAcceptConnectRoundTrip(p);
}

// io-07 multishot accept: ONE submit_accept_multishot yields a completion per
// connection (kMore SET while armed). Connect kMulti clients; each must surface a
// distinct accepted fd from the SAME user_data with kMore set. Exercised on both
// io_uring (native IORING_ACCEPT_MULTISHOT) and the epoll fallback (re-arm
// emulation) via the shared helper.
static void RunMultishotAccept(Proactor& p) {
  constexpr int kMulti = 4;
  int ls = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(ls, 0);
  int one = 1;
  ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ASSERT_EQ(::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);
  ASSERT_EQ(::listen(ls, 16), 0);
  socklen_t slen = sizeof(sa);
  ASSERT_EQ(::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen), 0);

  // ONE multishot accept arm -- must serve all kMulti connections.
  ASSERT_TRUE(p.submit_accept_multishot(ls, /*user_data=*/0xACCE));

  int clients[kMulti];
  int accepted_count = 0;
  int more_set = 0;
  for (int i = 0; i < kMulti; ++i) {
    clients[i] = ::socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(clients[i], 0);
    // Blocking connect to loopback; the listener backlog holds it until accepted.
    ASSERT_EQ(::connect(clients[i], reinterpret_cast<sockaddr*>(&sa), sizeof(sa)),
              0);
  }

  // Reap until all kMulti accepts surface. Every one must carry user_data 0xACCE
  // (verbatim, same token) and a valid fd; while armed each has kMore set.
  for (int spins = 0; spins < 500 && accepted_count < kMulti; ++spins) {
    Completion c[8];
    std::size_t n = p.reap(c, 8, 1'000'000'000ull);
    for (std::size_t i = 0; i < n; ++i) {
      if (c[i].user_data != 0xACCE) continue;
      ASSERT_GE(c[i].result, 0) << "multishot accept must carry a valid fd";
      if (c[i].flags & Completion::kMore) ++more_set;
      ::close(static_cast<int>(c[i].result));
      ++accepted_count;
    }
  }
  EXPECT_EQ(accepted_count, kMulti) << "one multishot submit served all conns";
  EXPECT_GE(more_set, kMulti - 1) << "armed completions carry kMore";

  for (int i = 0; i < kMulti; ++i) ::close(clients[i]);
  ::close(ls);
}

TEST(Proactor, MultishotAcceptUring) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";
  RunMultishotAccept(p);
}

TEST(Proactor, MultishotAcceptEpoll) {
  ::setenv("HYLE_IO_FORCE_EPOLL", "1", 1);
  Proactor p;
  bool ok = p.init();
  ::unsetenv("HYLE_IO_FORCE_EPOLL");
  ASSERT_TRUE(ok);
  RunMultishotAccept(p);
}

// io-07b provided buffer ring + multishot recv (io_uring only). Register a small
// buffer ring, arm ONE multishot recv on a socketpair end, send several messages
// from the other end. Each must surface as a completion carrying kBuffer + a
// buffer_id whose backing memory holds the sent bytes; returning the buffer lets
// the ring keep serving. Proves the kBuffer half of the reserved Completion flags.
TEST(Proactor, ProvidedBufRingMultishotRecv) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";
  if (p.backend_kind() != BackendKind::kIoUring)
    GTEST_SKIP() << "io_uring backend unavailable";

  constexpr int kBgid = 7;
  constexpr unsigned kEntries = 8;
  constexpr unsigned kBufSize = 128;
  if (!p.register_buf_ring(kBgid, kEntries, kBufSize))
    GTEST_SKIP() << "provided buffer ring unsupported on this kernel";

  // DGRAM pair: message boundaries preserved, so N sends == N recv completions
  // (a STREAM socket would coalesce the sends into fewer buffers).
  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_DGRAM, 0, sv), 0);

  // Send 3 distinct messages BEFORE arming, so they're queued and delivered
  // promptly (avoids blocking reaps waiting on data that isn't coming yet).
  const char* msgs[] = {"ring-buf-0", "ring-buf-1", "ring-buf-2"};
  constexpr int kN = 3;
  for (int i = 0; i < kN; ++i)
    ASSERT_GT(::write(sv[1], msgs[i], std::strlen(msgs[i]) + 1), 0);

  // ONE multishot recv arm on sv[0], drawing from the ring.
  ASSERT_TRUE(
      p.submit_recv_multishot(sv[0], kBgid, /*user_data=*/0xBEEF));

  int got = 0;
  int kbuf_set = 0;
  for (int spins = 0; spins < 100 && got < kN; ++spins) {
    Completion c[8];
    // Short timeout: once the 3 queued datagrams are drained the arm stays live
    // (kMore) with no more data, so a long blocking reap would stall the test.
    std::size_t n = p.reap(c, 8, 50'000'000ull);  // 50ms
    for (std::size_t i = 0; i < n; ++i) {
      if (c[i].user_data != 0xBEEF) continue;
      if (c[i].result <= 0) continue;  // 0/neg = terminal; skip
      ASSERT_TRUE(c[i].flags & Completion::kBuffer)
          << "multishot recv completion must carry a kernel-selected buffer";
      ++kbuf_set;
      void* addr = p.buf_ring_addr(kBgid, c[i].buffer_id);
      ASSERT_NE(addr, nullptr);
      // The bytes the kernel wrote into the selected buffer must match a message.
      EXPECT_STREQ(static_cast<const char*>(addr), msgs[got]);
      // Return the buffer so the ring doesn't run dry.
      EXPECT_TRUE(p.buf_ring_return(kBgid, c[i].buffer_id));
      ++got;
    }
  }
  EXPECT_EQ(got, kN) << "one multishot recv arm delivered all messages";
  EXPECT_EQ(kbuf_set, kN) << "every delivery carried kBuffer";

  ::close(sv[0]);
  ::close(sv[1]);
}

// io-07c registered (fixed) files: register a sparse table, place a socket in a slot,
// and drive a recv+send by SLOT INDEX (IOSQE_FIXED_FILE) instead of the raw fd. Proves
// the fixed-file path resolves the fd from the kernel table and delivers correctly.
TEST(Proactor, RegisteredFixedFiles) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";
  if (p.backend_kind() != BackendKind::kIoUring)
    GTEST_SKIP() << "io_uring backend unavailable";
  if (!p.register_file_table(8))
    GTEST_SKIP() << "registered files unsupported on this kernel";

  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  // Place the RECEIVE end (sv[0]) into a fixed-file slot; sv[1] is the raw sender.
  int slot = p.assign_fixed_slot(sv[0]);
  ASSERT_GE(slot, 0) << "assign_fixed_slot must return a valid slot index";

  // Send from the raw peer, then recv via the FIXED slot (not sv[0]).
  const char* msg = "fixed-file-recv";
  ASSERT_GT(::write(sv[1], msg, std::strlen(msg) + 1), 0);
  char buf[64] = {};
  ASSERT_TRUE(
      p.submit_recv_fixed(slot, buf, sizeof(buf), 0, /*user_data=*/0xF1));

  bool got = false;
  for (int spins = 0; spins < 100 && !got; ++spins) {
    Completion c[4];
    std::size_t n = p.reap(c, 4, 50'000'000ull);  // 50ms
    for (std::size_t i = 0; i < n; ++i) {
      if (c[i].user_data != 0xF1) continue;
      ASSERT_GT(c[i].result, 0) << "fixed recv must transfer bytes";
      EXPECT_STREQ(buf, msg) << "fixed-file recv delivered the peer's bytes";
      got = true;
    }
  }
  EXPECT_TRUE(got) << "the fixed-file recv never completed";

  // Also exercise submit_send_fixed: send via the slot back to the peer, verify it.
  const char* reply = "fixed-file-send";
  ASSERT_TRUE(p.submit_send_fixed(slot, reply, std::strlen(reply) + 1, 0,
                                  /*user_data=*/0xF2));
  bool sent = false;
  for (int spins = 0; spins < 100 && !sent; ++spins) {
    Completion c[4];
    std::size_t n = p.reap(c, 4, 50'000'000ull);
    for (std::size_t i = 0; i < n; ++i)
      if (c[i].user_data == 0xF2 && c[i].result > 0) sent = true;
  }
  EXPECT_TRUE(sent) << "the fixed-file send never completed";
  char rbuf[64] = {};
  EXPECT_GT(::read(sv[1], rbuf, sizeof(rbuf)), 0);
  EXPECT_STREQ(rbuf, reply) << "peer received the fixed-file send bytes";

  EXPECT_TRUE(p.release_fixed_slot(slot));
  ::close(sv[0]);
  ::close(sv[1]);
}

TEST(Proactor, FixedFileSubmissionsHonorQueueDepth) {
  // The fixed-file SQEs have fields submit_one cannot express. They still must
  // share its bounded admission invariant: depth operations enter, the next is
  // backpressured, and reap makes those slots reusable.
  Proactor p;
  if (!p.init(8)) GTEST_SKIP() << "no io_uring backend";
  if (p.backend_kind() != BackendKind::kIoUring)
    GTEST_SKIP() << "io_uring backend unavailable";
  if (!p.register_file_table(4))
    GTEST_SKIP() << "registered files unsupported on this kernel";

  int sv[2];
  ASSERT_EQ(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  const int slot = p.assign_fixed_slot(sv[0]);
  ASSERT_GE(slot, 0);

  constexpr std::size_t kOps = 32;
  char sent[kOps];
  char received[kOps] = {};
  for (std::size_t i = 0; i < kOps; ++i)
    sent[i] = static_cast<char>('A' + (i % 26));
  ASSERT_EQ(::write(sv[1], sent, sizeof(sent)),
            static_cast<ssize_t>(sizeof(sent)));
  std::size_t completed = 0;
  while (completed < kOps) {
    constexpr std::size_t kDepth = 8;
    const std::size_t batch_begin = completed;
    for (std::size_t i = 0; i < kDepth; ++i) {
      const std::size_t index = batch_begin + i;
      ASSERT_TRUE(p.submit_recv_fixed(
          slot, &received[index], 1, 0, 0xF100u + index));
    }
    if (batch_begin + kDepth < kOps) {
      const std::size_t blocked = batch_begin + kDepth;
      EXPECT_FALSE(p.submit_recv_fixed(
          slot, &received[blocked], 1, 0, 0xF100u + blocked));
    }

    std::size_t batch_completed = 0;
    while (batch_completed < kDepth) {
      Completion completions[kDepth];
      const std::size_t n = p.reap(completions, kDepth - batch_completed,
                                   1'000'000'000ull);
      ASSERT_GT(n, 0u);
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_GE(completions[i].user_data, 0xF100u + batch_begin);
        ASSERT_LT(completions[i].user_data,
                  0xF100u + batch_begin + kDepth);
        EXPECT_EQ(completions[i].result, 1);
      }
      batch_completed += n;
    }
    completed += kDepth;
  }
  EXPECT_EQ(std::memcmp(sent, received, sizeof(sent)), 0);

  EXPECT_TRUE(p.release_fixed_slot(slot));
  ::close(sv[0]);
  ::close(sv[1]);
}

// io-07c step 2 direct accept: accept a connection STRAIGHT into a registered-file
// slot (no process fd ever created), then recv on that direct descriptor by slot, and
// close_direct it. Proves the full direct path: submit_accept_direct -> CQE res = slot
// -> submit_recv_fixed(slot) -> submit_close_direct(slot).
TEST(Proactor, DirectAccept) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no io_uring backend";
  if (p.backend_kind() != BackendKind::kIoUring)
    GTEST_SKIP() << "io_uring backend unavailable";
  if (!p.register_file_table(8))
    GTEST_SKIP() << "registered files unsupported on this kernel";

  int ls = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(ls, 0);
  int one = 1;
  ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ASSERT_EQ(::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);
  ASSERT_EQ(::listen(ls, 8), 0);
  socklen_t slen = sizeof(sa);
  ASSERT_EQ(::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen), 0);

  // Direct accept armed BEFORE the client connects (accept waits for it).
  ASSERT_TRUE(p.submit_accept_direct(ls, /*user_data=*/0xDA));

  // Client connects + sends on a separate thread (blocking connect to loopback).
  sockaddr_in ca = sa;
  std::thread cli([&]{
    int c = ::socket(AF_INET, SOCK_STREAM, 0);
    if (::connect(c, reinterpret_cast<sockaddr*>(&ca), sizeof(ca)) == 0) {
      const char* m = "direct-accept-hi";
      (void)!::write(c, m, std::strlen(m) + 1);
      ::usleep(50'000);
    }
    ::close(c);
  });

  // Reap the accept -> res is the ALLOCATED SLOT INDEX (>=0).
  int slot = -1;
  int direct_error = 0;
  for (int spins = 0; spins < 500 && slot < 0; ++spins) {
    Completion c[8];
    std::size_t n = p.reap(c, 8, 1'000'000'000ull);
    for (std::size_t i = 0; i < n; ++i)
      if (c[i].user_data == 0xDA) {
        if (c[i].result < 0)
          direct_error = static_cast<int>(c[i].result);
        else
          slot = static_cast<int>(c[i].result);
      }
    if (direct_error != 0) break;
  }
  cli.join();
  if (direct_error == -EINVAL || direct_error == -ENOSYS ||
      direct_error == -EOPNOTSUPP) {
    ::close(ls);
    GTEST_SKIP() << "kernel accepts registered files but not accept_direct: "
                 << direct_error;
  }
  ASSERT_EQ(direct_error, 0) << "direct accept failed";
  ASSERT_GE(slot, 0) << "direct accept never produced a slot";

  // recv on the DIRECT descriptor by slot (IOSQE_FIXED_FILE) -- it was never a raw fd.
  char buf[64] = {};
  ASSERT_TRUE(
      p.submit_recv_fixed(slot, buf, sizeof(buf), 0, /*user_data=*/0xDB));
  bool got = false;
  for (int spins = 0; spins < 200 && !got; ++spins) {
    Completion c[4];
    std::size_t n = p.reap(c, 4, 1'000'000'000ull);
    for (std::size_t i = 0; i < n; ++i)
      if (c[i].user_data == 0xDB && c[i].result > 0) {
        EXPECT_STREQ(buf, "direct-accept-hi");
        got = true;
      }
  }
  EXPECT_TRUE(got) << "recv on the direct-accepted descriptor never delivered";

  // close_direct the slot (IORING_OP_CLOSE with file_index).
  ASSERT_TRUE(p.submit_close_direct(slot, /*user_data=*/0xDC));
  bool closed = false;
  for (int spins = 0; spins < 100 && !closed; ++spins) {
    Completion c[4];
    std::size_t n = p.reap(c, 4, 500'000'000ull);
    for (std::size_t i = 0; i < n; ++i)
      if (c[i].user_data == 0xDC) { EXPECT_EQ(c[i].result, 0); closed = true; }
  }
  EXPECT_TRUE(closed) << "close_direct never completed";
  ::close(ls);
}
#endif  // __linux__

// ---- Windows IOCP real IO ops ----------------------------------------------
// The IOCP counterparts of the io_uring tests above: a real file write/read
// closed loop (ReadFile/WriteFile + per-op OVERLAPPED), a bad-handle negative
// result, and a full TCP accept/connect/send/recv over loopback via
// AcceptEx/ConnectEx + WSASend/WSARecv. Proves the per-op OVERLAPPED lifetime
// and the completion->result mapping the proactor now carries on Windows.
#if defined(_WIN32)
TEST(Proactor, RealFileReadWriteRoundTrip) {
  Proactor p;
  ASSERT_TRUE(p.init());

  // A temp file opened OVERLAPPED (async IO) + associated with the port.
  char dir[MAX_PATH];
  char path[MAX_PATH];
  ::GetTempPathA(MAX_PATH, dir);
  ::GetTempFileNameA(dir, "hy", 0, path);
  HANDLE fh = ::CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                            CREATE_ALWAYS, FILE_FLAG_OVERLAPPED, nullptr);
  ASSERT_NE(fh, INVALID_HANDLE_VALUE);
  ASSERT_TRUE(p.register_handle(reinterpret_cast<std::intptr_t>(fh)));

  const char msg[] = "hyle-iocp-real-op";
  const std::uint32_t len = sizeof(msg);
  auto fd = reinterpret_cast<std::intptr_t>(fh);

  ASSERT_TRUE(p.submit_write(fd, msg, len, 0, /*user_data=*/0x111));
  Completion c{};
  ASSERT_EQ(p.reap(&c, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c.user_data, 0x111u);
  EXPECT_EQ(c.result, static_cast<std::int64_t>(len));  // bytes written

  char buf[64] = {};
  ASSERT_TRUE(p.submit_read(fd, buf, len, 0, /*user_data=*/0x222));
  Completion c2{};
  ASSERT_EQ(p.reap(&c2, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(c2.user_data, 0x222u);
  EXPECT_EQ(c2.result, static_cast<std::int64_t>(len));  // bytes read
  EXPECT_STREQ(buf, msg);  // round-tripped through the kernel

  ::CloseHandle(fh);
  ::DeleteFileA(path);
}

// A bad handle fails the submit synchronously (no completion is ever queued),
// so submit_read returns false -- distinct from an IO error, which would submit
// fine and surface a negative result at reap.
TEST(Proactor, RealFileReadBadHandleSubmitFails) {
  Proactor p;
  ASSERT_TRUE(p.init());
  char buf[8] = {};
  EXPECT_FALSE(p.submit_read(reinterpret_cast<std::intptr_t>(INVALID_HANDLE_VALUE),
                             buf, sizeof(buf), 0, 0x333));
}

TEST(Proactor, DirectoryChangesCompleteAndRearmIocp) {
  Proactor p;
  ASSERT_TRUE(p.init());

  wchar_t temp_path[MAX_PATH];
  wchar_t unique_path[MAX_PATH];
  ASSERT_NE(::GetTempPathW(MAX_PATH, temp_path), 0u);
  ASSERT_NE(::GetTempFileNameW(temp_path, L"hyw", 0, unique_path), 0u);
  ASSERT_TRUE(::DeleteFileW(unique_path));
  ASSERT_TRUE(::CreateDirectoryW(unique_path, nullptr));

  HANDLE directory = ::CreateFileW(
      unique_path, FILE_LIST_DIRECTORY,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
  ASSERT_NE(directory, INVALID_HANDLE_VALUE);
  ASSERT_TRUE(p.register_handle(reinterpret_cast<std::intptr_t>(directory)));

  alignas(DWORD) unsigned char buffer[4096]{};
  ASSERT_TRUE(p.submit_directory_changes(
      reinterpret_cast<std::intptr_t>(directory), buffer, sizeof(buffer),
      0xD1u));

  std::wstring child = unique_path;
  child += L"\\changed.txt";
  HANDLE file = ::CreateFileW(child.c_str(), GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE |
                                  FILE_SHARE_DELETE,
                              nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
  ASSERT_NE(file, INVALID_HANDLE_VALUE);
  ASSERT_TRUE(::CloseHandle(file));

  Completion completion{};
  ASSERT_EQ(p.reap(&completion, 1, 5'000'000'000ull), 1u);
  EXPECT_EQ(completion.user_data, 0xD1u);
  ASSERT_GT(completion.result, 0);
  const auto* info = reinterpret_cast<const FILE_NOTIFY_INFORMATION*>(buffer);
  EXPECT_EQ(info->Action, FILE_ACTION_ADDED);
  EXPECT_EQ(std::wstring(info->FileName,
                         info->FileNameLength / sizeof(wchar_t)),
            L"changed.txt");

  // The native operation is one-shot: a second arm reports a later mutation.
  ASSERT_TRUE(p.submit_directory_changes(
      reinterpret_cast<std::intptr_t>(directory), buffer, sizeof(buffer),
      0xD2u));
  ASSERT_TRUE(::DeleteFileW(child.c_str()));
  ASSERT_EQ(p.reap(&completion, 1, 5'000'000'000ull), 1u);
  EXPECT_EQ(completion.user_data, 0xD2u);
  EXPECT_GT(completion.result, 0);

  ASSERT_TRUE(::CloseHandle(directory));
  ASSERT_TRUE(::RemoveDirectoryW(unique_path));
}

TEST(Proactor, DirectoryChangesHonorAdmissionAndCancellationIocp) {
  Proactor p;
  ASSERT_TRUE(p.init(1));

  wchar_t temp_path[MAX_PATH];
  wchar_t unique_path[MAX_PATH];
  ASSERT_NE(::GetTempPathW(MAX_PATH, temp_path), 0u);
  ASSERT_NE(::GetTempFileNameW(temp_path, L"hyw", 0, unique_path), 0u);
  ASSERT_TRUE(::DeleteFileW(unique_path));
  ASSERT_TRUE(::CreateDirectoryW(unique_path, nullptr));
  HANDLE directory = ::CreateFileW(
      unique_path, FILE_LIST_DIRECTORY,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
  ASSERT_NE(directory, INVALID_HANDLE_VALUE);
  ASSERT_TRUE(p.register_handle(reinterpret_cast<std::intptr_t>(directory)));

  alignas(DWORD) unsigned char first[1024]{};
  alignas(DWORD) unsigned char second[1024]{};
  ASSERT_TRUE(p.submit_directory_changes(
      reinterpret_cast<std::intptr_t>(directory), first, sizeof(first), 0xD3u));
  EXPECT_FALSE(p.submit_directory_changes(
      reinterpret_cast<std::intptr_t>(directory), second, sizeof(second),
      0xD4u));
  EXPECT_FALSE(p.submit_directory_changes(
      reinterpret_cast<std::intptr_t>(INVALID_HANDLE_VALUE), second,
      sizeof(second), 0xD5u));
  ASSERT_TRUE(p.submit_cancel(0xD3u, 0xD6u));
  Completion completion{};
  ASSERT_EQ(p.reap(&completion, 1, 5'000'000'000ull), 1u);
  EXPECT_EQ(completion.user_data, 0xD3u);
  EXPECT_EQ(completion.result,
            -static_cast<std::int64_t>(ERROR_OPERATION_ABORTED));

  ASSERT_TRUE(::CloseHandle(directory));
  ASSERT_TRUE(::RemoveDirectoryW(unique_path));
}

TEST(Proactor, TcpAcceptConnectRoundTripIocp) {
  Proactor p;
  ASSERT_TRUE(p.init());

  // Listener on an ephemeral loopback port (overlapped socket, port-associated).
  SOCKET ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0,
                           WSA_FLAG_OVERLAPPED);
  ASSERT_NE(ls, INVALID_SOCKET);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ASSERT_EQ(::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);
  ASSERT_EQ(::listen(ls, 8), 0);
  int slen = sizeof(sa);
  ASSERT_EQ(::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen), 0);
  ASSERT_TRUE(p.register_handle(static_cast<std::intptr_t>(ls)));

  // Client socket, also overlapped + port-associated (ConnectEx completes here).
  SOCKET cl = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0,
                           WSA_FLAG_OVERLAPPED);
  ASSERT_NE(cl, INVALID_SOCKET);
  ASSERT_TRUE(p.register_handle(static_cast<std::intptr_t>(cl)));

  sockaddr_storage peer{};
  int peer_len = sizeof(peer);
  ASSERT_TRUE(p.submit_accept(static_cast<std::intptr_t>(ls), &peer, &peer_len,
                              0xACCE));
  ASSERT_TRUE(p.submit_connect(static_cast<std::intptr_t>(cl), &sa, sizeof(sa),
                               0xC0DE));

  SOCKET accepted = INVALID_SOCKET;
  bool got_accept = false, got_connect = false;
  for (int spins = 0; spins < 100 && !(got_accept && got_connect); ++spins) {
    Completion c[4];
    std::size_t n = p.reap(c, 4, 1'000'000'000ull);
    for (std::size_t i = 0; i < n; ++i) {
      if (c[i].user_data == 0xACCE) {
        got_accept = true;
        ASSERT_GE(c[i].result, 0);
        accepted = static_cast<SOCKET>(c[i].result);
      } else if (c[i].user_data == 0xC0DE) {
        got_connect = true;
        EXPECT_EQ(c[i].result, 0);
      }
    }
  }
  ASSERT_TRUE(got_accept);
  ASSERT_TRUE(got_connect);
  ASSERT_NE(accepted, INVALID_SOCKET);
  ASSERT_EQ(peer.ss_family, AF_INET);
  ASSERT_GE(peer_len, static_cast<int>(sizeof(sockaddr_in)));
  sockaddr_in client_local{};
  int client_local_len = sizeof(client_local);
  ASSERT_EQ(::getsockname(cl, reinterpret_cast<sockaddr*>(&client_local),
                         &client_local_len),
            0);
  EXPECT_EQ(reinterpret_cast<const sockaddr_in*>(&peer)->sin_port,
            client_local.sin_port);
  ASSERT_TRUE(p.register_handle(static_cast<std::intptr_t>(accepted)));

  const char msg[] = "iocp-accept-connect";
  const std::uint32_t len = sizeof(msg);
  ASSERT_TRUE(p.submit_send(static_cast<std::intptr_t>(cl), msg, len, 0, 0x5E4D));
  Completion cs{};
  ASSERT_EQ(p.reap(&cs, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(cs.user_data, 0x5E4Du);
  EXPECT_EQ(cs.result, static_cast<std::int64_t>(len));

  char buf[64] = {};
  ASSERT_TRUE(
      p.submit_recv(static_cast<std::intptr_t>(accepted), buf, len, 0, 0x8EC0));
  Completion cr{};
  ASSERT_EQ(p.reap(&cr, 1, 1'000'000'000ull), 1u);
  EXPECT_EQ(cr.user_data, 0x8EC0u);
  EXPECT_EQ(cr.result, static_cast<std::int64_t>(len));
  EXPECT_STREQ(buf, msg);

  const char urgent_out = '!';
  ASSERT_TRUE(p.submit_send(static_cast<std::intptr_t>(cl), &urgent_out, 1,
                            MSG_OOB, 0x50B));
  Completion urgent_send{};
  ASSERT_EQ(p.reap(&urgent_send, 1, 1'000'000'000ull), 1u);
  ASSERT_EQ(urgent_send.result, 1);
  char urgent_seen = 0;
  ASSERT_EQ(::recv(accepted, &urgent_seen, 1, MSG_OOB), 1);
  EXPECT_EQ(urgent_seen, urgent_out);

  const char urgent_in = '?';
  ASSERT_EQ(::send(cl, &urgent_in, 1, MSG_OOB), 1);
  char urgent_recv = 0;
  ASSERT_TRUE(p.submit_recv(static_cast<std::intptr_t>(accepted), &urgent_recv,
                            1, MSG_OOB, 0xB05));
  Completion urgent_completion{};
  ASSERT_EQ(p.reap(&urgent_completion, 1, 1'000'000'000ull), 1u);
  ASSERT_EQ(urgent_completion.result, 1);
  EXPECT_EQ(urgent_recv, urgent_in);

  ::closesocket(accepted);
  ::closesocket(cl);
  ::closesocket(ls);
}

// io-07 multishot accept on IOCP: no native multishot, so submit_accept_multishot
// arms one AcceptEx and reap RE-ARMS after each completion + stamps kMore. ONE
// submit must serve kMulti connections, each carrying the same user_data + a valid
// accepted socket, with kMore set while armed.
TEST(Proactor, MultishotAcceptIocp) {
  constexpr int kMulti = 4;
  Proactor p;
  ASSERT_TRUE(p.init());

  SOCKET ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0,
                           WSA_FLAG_OVERLAPPED);
  ASSERT_NE(ls, INVALID_SOCKET);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ASSERT_EQ(::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);
  ASSERT_EQ(::listen(ls, 16), 0);
  int slen = sizeof(sa);
  ASSERT_EQ(::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen), 0);
  ASSERT_TRUE(p.register_handle(static_cast<std::intptr_t>(ls)));

  ASSERT_TRUE(p.submit_accept_multishot(static_cast<std::intptr_t>(ls), 0xACCE));

  SOCKET clients[kMulti];
  for (int i = 0; i < kMulti; ++i) {
    clients[i] = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0,
                              WSA_FLAG_OVERLAPPED);
    ASSERT_NE(clients[i], INVALID_SOCKET);
    // Blocking connect to loopback; backlog holds it until AcceptEx serves it.
    ASSERT_EQ(::connect(clients[i], reinterpret_cast<sockaddr*>(&sa), sizeof(sa)),
              0);
  }

  int accepted_count = 0, more_set = 0;
  for (int spins = 0; spins < 500 && accepted_count < kMulti; ++spins) {
    Completion c[8];
    std::size_t n = p.reap(c, 8, 1'000'000'000ull);
    for (std::size_t i = 0; i < n; ++i) {
      if (c[i].user_data != 0xACCE) continue;
      ASSERT_GE(c[i].result, 0);
      if (c[i].flags & Completion::kMore) ++more_set;
      ::closesocket(static_cast<SOCKET>(c[i].result));
      ++accepted_count;
    }
  }
  EXPECT_EQ(accepted_count, kMulti) << "one multishot submit served all conns";
  EXPECT_GE(more_set, kMulti - 1) << "armed completions carry kMore";

  for (int i = 0; i < kMulti; ++i) ::closesocket(clients[i]);
  ::closesocket(ls);
}

// Destroying the proactor with an op STILL in flight must not leak or UAF: the
// dtor cancels the pending op and waits for the cancellation to land before
// freeing its OVERLAPPED. Here a pending accept never gets a connection, so it
// is live at teardown -- the dtor's CancelIoEx + GetOverlappedResult(wait) path
// is what keeps this from freeing an OVERLAPPED the kernel still owns.
TEST(Proactor, DestructWithInflightAcceptCancelsCleanly) {
  // The listener is created inside the proactor's lifetime so WSAStartup (done by
  // init) is active -- WSASocketW fails otherwise. It outlives p so the dtor runs
  // with the accept still pending.
  SOCKET ls = INVALID_SOCKET;
  {
    Proactor p;
    ASSERT_TRUE(p.init());
    ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
    ASSERT_NE(ls, INVALID_SOCKET);
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sa.sin_port = 0;
    ASSERT_EQ(::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)), 0);
    ASSERT_EQ(::listen(ls, 8), 0);
    ASSERT_TRUE(p.register_handle(static_cast<std::intptr_t>(ls)));
    ASSERT_TRUE(p.submit_accept(static_cast<std::intptr_t>(ls), nullptr, nullptr,
                                0xACCE));
    // No client connects -> the accept stays in flight; p goes out of scope here
    // with a live op. A clean run (no crash / no ASan complaint) is the pass.
  }
  if (ls != INVALID_SOCKET) ::closesocket(ls);
}
#endif  // _WIN32
