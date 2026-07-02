#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(__linux__)
#  include <sys/socket.h>
#  include <unistd.h>
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#endif

// io-04 integration test (plan B): the co_await socket layer over a real fd,
// driven by a Scheduler worker whose loop reaps the proactor via IoContext. This
// is the payoff -- a fiber that `co_await`s a socket send/recv actually yields the
// fiber and resumes with the byte count when the kernel completes the op. Plan B
// uses a SINGLE worker (the shared proactor is reaped only from that worker's
// loop; multi-worker per-ring is plan A).

using hyle::io::IoContext;
using hyle::io::RecvOp;
using hyle::io::SendOp;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

// A connected pair of fds for a loopback round-trip, plus a place to register
// them with the proactor (IOCP needs the association; io_uring/epoll don't).
struct SocketPair {
  std::intptr_t a = -1;  // one end
  std::intptr_t b = -1;  // the other end
  bool ok = false;
};

#if defined(__linux__)
SocketPair make_pair(hyle::io::Proactor&) {
  int sv[2];
  SocketPair sp;
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return sp;
  sp.a = sv[0];
  sp.b = sv[1];
  sp.ok = true;
  return sp;  // io_uring/epoll take the fd in each op; no registration needed
}
void close_pair(SocketPair& sp) {
  if (sp.a >= 0) ::close(static_cast<int>(sp.a));
  if (sp.b >= 0) ::close(static_cast<int>(sp.b));
}
#endif

#if defined(_WIN32)
// Windows has no socketpair; build a connected TCP loopback pair synchronously,
// then register both ends with the proactor's port (IOCP requirement).
SocketPair make_pair(hyle::io::Proactor& p) {
  SocketPair sp;
  SOCKET ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (ls == INVALID_SOCKET) return sp;
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  ::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  ::listen(ls, 1);
  int slen = sizeof(sa);
  ::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen);
  // Client connects (blocking, synchronous) while we accept.
  SOCKET c = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  ::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  SOCKET s = ::accept(ls, nullptr, nullptr);
  ::closesocket(ls);
  if (c == INVALID_SOCKET || s == INVALID_SOCKET) return sp;
  sp.a = static_cast<std::intptr_t>(c);
  sp.b = static_cast<std::intptr_t>(s);
  sp.ok = p.register_handle(sp.a) && p.register_handle(sp.b);
  return sp;
}
void close_pair(SocketPair& sp) {
  if (sp.a != -1) ::closesocket(static_cast<SOCKET>(sp.a));
  if (sp.b != -1) ::closesocket(static_cast<SOCKET>(sp.b));
}
#endif

// The coroutine under test: co_await send on one fd, co_await recv on the other,
// verify the bytes round-tripped through the kernel. Returns bytes received.
struct EchoState {
  hyle::io::Proactor* p;
  std::intptr_t send_fd;
  std::intptr_t recv_fd;
  char recv_buf[64] = {};
  std::int64_t sent = 0;
  std::int64_t received = 0;
};

Task<int> echo_coro(Allocator, EchoState* st) {
  static const char kMsg[] = "co_await-socket-echo";
  const std::uint32_t len = sizeof(kMsg);
  // Send on fd a. co_await suspends THIS fiber until the send completes; the
  // worker loop reaps the completion and wakes us with the byte count.
  st->sent = co_await SendOp{st->p, st->send_fd, kMsg, len};
  // Now receive it on fd b (the send already delivered it into the socket buf).
  st->received = co_await RecvOp{st->p, st->recv_fd, st->recv_buf, len};
  co_return 0;
}

// Fiber body: drive the coroutine. One resume() runs it to completion because
// each co_await yields the FIBER (not just the coroutine) and SuspendFiber
// resumes the coroutine on wake -- so resume() returns only once done.
struct Drive {
  Task<int> task;
  std::atomic<bool> finished{false};
};
void drive_entry(void* p) {
  auto* d = static_cast<Drive*>(p);
  d->task.resume();
  d->finished.store(true, std::memory_order_release);
}

}  // namespace

TEST(IoNet, CoAwaitSendRecvRoundTrip) {
  Scheduler sched(1, system_allocator(), 256);  // plan B: single worker
  IoContext io(sched);
  ASSERT_TRUE(io.valid()) << "proactor backend must come up";

  SocketPair sp = make_pair(io.proactor());
  ASSERT_TRUE(sp.ok) << "failed to build a connected socket pair";

  EchoState st{};
  st.p = &io.proactor();
  st.send_fd = sp.a;
  st.recv_fd = sp.b;

  auto d = std::make_unique<Drive>();
  d->task = echo_coro(system_allocator(), &st);
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());

  sched.spawn(&f);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(d->finished.load());
  EXPECT_EQ(st.sent, static_cast<std::int64_t>(sizeof("co_await-socket-echo")));
  EXPECT_EQ(st.received, static_cast<std::int64_t>(sizeof("co_await-socket-echo")));
  EXPECT_STREQ(st.recv_buf, "co_await-socket-echo");

  close_pair(sp);
}
