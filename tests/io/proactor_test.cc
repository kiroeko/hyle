#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__linux__)
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

#include "hyle/io/proactor.h"

using hyle::io::Completion;
using hyle::io::Proactor;

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
  sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
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

  ASSERT_TRUE(p.submit_accept(ls, nullptr, nullptr, /*user_data=*/0xACCE));
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
  sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
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

TEST(Proactor, TcpAcceptConnectRoundTripIocp) {
  Proactor p;
  ASSERT_TRUE(p.init());

  // Listener on an ephemeral loopback port (overlapped socket, port-associated).
  SOCKET ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0,
                           WSA_FLAG_OVERLAPPED);
  ASSERT_NE(ls, INVALID_SOCKET);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
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

  ASSERT_TRUE(p.submit_accept(static_cast<std::intptr_t>(ls), nullptr, nullptr,
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
  sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
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
    sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
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
