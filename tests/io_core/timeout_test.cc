#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>

#include "hyle/io/detail/io_context_access.h"
#include "hyle/io/detail/net.h"
#include "hyle/io/detail/net_timeout.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/udp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/timer.h"
#include "hyle/time/clock.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#endif

// io-06 timed io test: a recv WITH a deadline over loopback TCP. Two cases prove
// both race outcomes on every backend (Win IOCP / Linux io_uring / forced-epoll):
//   1. TIMEOUT wins: the server accepts but never sends -> the client's timed recv
//      hits its deadline and returns -ETIMEDOUT (-110); the in-flight recv is
//      cancelled; the public await resumes only after the original completion
//      retires, so neither the slot nor the borrowed byte buffers can be touched
//      after return.
//   2. IO wins: the server sends promptly -> the timed recv returns the bytes
//      before the (generous) deadline, and the timer is cancelled cleanly.

using hyle::io::IoContext;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::detail::RecvOp;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr std::int64_t kDeadlineExceededResult = -110;

// A counting allocator to catch TimedSlot leaks: every allocate bumps `live`, every
// deallocate drops it. After a timed op fully retires (including the late cancel
// completion that frees the orphan slot), live must return to 0. This is what
// catches the "timeout path leaks kernel-ref" class of bug that functional asserts
// and TSan both miss.
struct CountingAlloc {
  static std::atomic<long> live;
  static void* a(void*, std::size_t n, std::size_t al) noexcept {
    live.fetch_add(1, std::memory_order_relaxed);
#if defined(_WIN32)
    return _aligned_malloc(n, al < sizeof(void*) ? sizeof(void*) : al);
#else
    if (al < sizeof(void*)) al = sizeof(void*);
    return std::aligned_alloc(al, (n + al - 1) & ~(al - 1));
#endif
  }
  static void f(void*, void* p, std::size_t, std::size_t) noexcept {
    if (p == nullptr) return;
    live.fetch_sub(1, std::memory_order_relaxed);
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
  static void* r(void*, void* p, std::size_t, std::size_t nn, std::size_t al) noexcept {
#if defined(_WIN32)
    return _aligned_realloc(p, nn, al < sizeof(void*) ? sizeof(void*) : al);
#else
    (void)al;
    return std::realloc(p, nn);
#endif
  }
  static AllocatorRef make() noexcept { return AllocatorRef(&a, &f, &r, nullptr); }
};
std::atomic<long> CountingAlloc::live{0};

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) noexcept { static_cast<Drive*>(p)->task.resume(); }

// ---- case 1: timeout wins (server accepts, never sends) ---------------------
struct TimeoutResult {
  std::atomic<bool> done{false};
  std::atomic<bool> connected{false};
  std::atomic<std::int64_t> recv_rc{999};  // expect -110 (ETIMEDOUT)
  std::atomic<bool> native_retired{false};
  TcpStream held;  // keep the accepted server conn alive (so it doesn't RST)
};

Task<int> timeout_server(AllocatorRef, IoContext*, TcpListener* ls,
                         TimeoutResult* r) {
  auto accepted = co_await ls->accept(kOwner);
  if (!accepted.has_value()) co_return 0;
  TcpStream conn = std::move(*accepted);
  r->held = std::move(conn);  // hold it open but send NOTHING -> client recv stalls
  co_return 0;
}

Task<int> timeout_client(AllocatorRef alloc, IoContext* io, TimeoutResult* r,
                         std::uint16_t port) {
  (void)alloc;
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value()) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  TcpStream s = std::move(*opened);
  if (!(co_await s.connect(Endpoint::loopback(AddressFamily::kIpv4, port))).ok()) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->connected.store(true, std::memory_order_release);
  char first[17] = {};
  char second[47] = {};
  const hyle::io::MutableBuffer buffers[] = {
      {first, sizeof(first)}, {second, sizeof(second)}};
  // Deadline 50ms out; server never sends, so the timer must win.
  auto transfer = co_await s.read_some(
      std::span<const hyle::io::MutableBuffer>{buffers},
      hyle::io::OperationOptions{
          .deadline = hyle::time::deadline_after(std::chrono::milliseconds(50))});
  const std::int64_t rc =
      transfer.ok() ? static_cast<std::int64_t>(transfer.progress.bytes)
                    : (transfer.status.code() ==
                               hyle::StatusCode::DeadlineExceeded
                           ? kDeadlineExceededResult
                           : -999);
  r->recv_rc.store(rc, std::memory_order_release);
  r->native_retired.store(
      !hyle::io::detail::IoContextAccess::proactor_for_current(*io).in_flight(),
                          std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

// ---- case 2: io wins (server sends promptly) --------------------------------
struct IoWinResult {
  std::atomic<bool> done{false};
  std::atomic<std::int64_t> recv_rc{999};  // expect > 0 (bytes)
  char got[64] = {};
};

Task<int> iowin_server(AllocatorRef, IoContext*, TcpListener* ls, IoWinResult*) {
  auto accepted = co_await ls->accept(kOwner);
  if (!accepted.has_value()) co_return 0;
  TcpStream conn = std::move(*accepted);
  static const char kMsg[] = "prompt";
  (void)(co_await conn.write_some(std::as_bytes(std::span{kMsg})));
  co_return 0;
}

Task<int> iowin_client(AllocatorRef alloc, IoContext* io, IoWinResult* r,
                       std::uint16_t port) {
  (void)alloc;
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value()) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  TcpStream s = std::move(*opened);
  if (!(co_await s.connect(Endpoint::loopback(AddressFamily::kIpv4, port))).ok()) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  // Generous 5s deadline; the server sends at once, so io must win well within it.
  auto transfer = co_await s.read_some(
      std::as_writable_bytes(std::span{r->got}),
      hyle::io::OperationOptions{
          .deadline = hyle::time::deadline_after(std::chrono::seconds(5))});
  const std::int64_t rc = transfer.ok()
                              ? static_cast<std::int64_t>(transfer.bytes)
                              : -999;
  r->recv_rc.store(rc, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

}  // namespace

TEST(IoTimeout, TimeoutWinsReturnsETimedout) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54390;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());
  auto listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener.has_value()) << "listen failed (port in use?)";
  TcpListener ls = std::move(*listener);

  auto r = std::make_unique<TimeoutResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = timeout_server(system_allocator(), &io, &ls, r.get());
  dc->task = timeout_client(system_allocator(), &io, r.get(), kPort);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_TRUE(r->connected.load());
  EXPECT_EQ(r->recv_rc.load(), -110);  // -ETIMEDOUT: the timer won the race
  EXPECT_TRUE(r->native_retired.load());
}

TEST(IoTimeout, IoWinsReturnsBytes) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54391;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());
  auto listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener.has_value()) << "listen failed (port in use?)";
  TcpListener ls = std::move(*listener);

  auto r = std::make_unique<IoWinResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = iowin_server(system_allocator(), &io, &ls, r.get());
  dc->task = iowin_client(system_allocator(), &io, r.get(), kPort);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_EQ(r->recv_rc.load(), static_cast<std::int64_t>(sizeof("prompt")));
  EXPECT_STREQ(r->got, "prompt");
}

// Accept with a deadline: no client ever connects -> the timer must win.
struct AcceptResult {
  std::atomic<bool> done{false};
  std::atomic<std::int64_t> rc{999};
};

Task<int> accept_timeout_client(AllocatorRef alloc, IoContext* io, TcpListener* ls,
                                AcceptResult* r) {
  (void)alloc;
  (void)io;
  auto accepted = co_await ls->accept(
      kOwner, hyle::io::OperationOptions{
                  .deadline = hyle::time::deadline_after(
                      std::chrono::milliseconds(50))});
  const std::int64_t rc =
      !accepted.has_value() &&
              accepted.error().code() == hyle::StatusCode::DeadlineExceeded
          ? kDeadlineExceededResult
          : (accepted.has_value() ? 0 : -999);
  r->rc.store(rc, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoTimeout, AcceptTimesOut) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54393;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());
  auto listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener.has_value()) << "listen failed (port in use?)";
  TcpListener ls = std::move(*listener);

  auto r = std::make_unique<AcceptResult>();
  auto d = std::make_unique<Drive>();
  d->task = accept_timeout_client(system_allocator(), &io, &ls, r.get());
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_EQ(r->rc.load(), -110);  // accept timed out
}

namespace {
Task<int> udp_timeout_recv(AllocatorRef alloc, hyle::io::UdpSocket* socket,
                           AcceptResult* result) {
  (void)alloc;
  char buffer[32] = {};
  auto received = co_await socket->receive_from(
      std::as_writable_bytes(std::span{buffer}),
      hyle::io::OperationOptions{
          .deadline = hyle::time::deadline_after(
              std::chrono::milliseconds(50))});
  result->rc.store(received.status.code() ==
                           hyle::StatusCode::DeadlineExceeded
                       ? kDeadlineExceededResult
                       : static_cast<std::int64_t>(received.bytes),
                   std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(IoTimeout, UdpReceiveTimesOut) {
  constexpr unsigned kWorkers = 1;
  constexpr std::uint16_t kPort = 54394;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());
  auto opened = hyle::io::UdpSocket::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(opened.has_value());
  auto socket = std::move(*opened);

  auto result = std::make_unique<AcceptResult>();
  auto drive = std::make_unique<Drive>();
  drive->task =
      udp_timeout_recv(system_allocator(), &socket, result.get());
  Fiber fiber(kDefaultStackSize, &drive_entry, drive.get());
  sched.spawn_on(kOwner, &fiber);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(result->done.load());
  EXPECT_EQ(result->rc.load(), -110);
}

TEST(IoTimeout, TimeoutPathDoesNotLeakSlot) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54392;
  CountingAlloc::live.store(0, std::memory_order_relaxed);
  {
    Scheduler sched(kWorkers, system_allocator(), 256);
    IoContext io(sched, hyle::io::IoContextOptions{
                            .allocator = CountingAlloc::make()});
    ASSERT_TRUE(io.valid());
    auto listener = TcpListener::bind(
        io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
    ASSERT_TRUE(listener.has_value()) << "listen failed (port in use?)";
    TcpListener ls = std::move(*listener);

    auto r = std::make_unique<TimeoutResult>();
    auto ds = std::make_unique<Drive>();
    auto dc = std::make_unique<Drive>();
    ds->task = timeout_server(system_allocator(), &io, &ls, r.get());
    // The timed op's slot comes from CountingAlloc -> its alloc/free is counted.
    dc->task = timeout_client(CountingAlloc::make(), &io, r.get(), kPort);
    Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
    Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
    sched.spawn_on(kOwner, &fs);
    sched.spawn_on(kOwner, &fc);
    sched.drain();
    sched.shutdown();
    EXPECT_EQ(r->recv_rc.load(), -110);  // timeout won
    // IoContext destructor (end of scope) drains any orphaned late completion.
  }
  // After IoContext teardown, the orphan slot must be freed: no net allocation.
  EXPECT_EQ(CountingAlloc::live.load(), 0) << "TimedSlot leaked on timeout path";
}

// ---- L5: base-op completion routing intact after the teardown-drop split -------
// The L5 fix changed ~IoContext's orphan drain to DROP base-op (untimed) late
// completions instead of routing them to complete() (which would wake a possibly-
// abandoned fiber on a freed stack slot = UAF). This test guards that the change
// did NOT break the NORMAL base-op path: a base recv over loopback still completes
// correctly (routed via the runtime poll, waking the live fiber) and its bytes
// arrive. (The teardown DROP line itself is a defensive backstop for the app-abuse
// case "abandon a fiber with a base op in flight then tear down" -- not
// deterministically triggerable because ~Scheduler's unconditional drain() blocks
// on any still-counted fiber; so we verify the routing split's live path here, and
// rely on TSan across the whole io suite to catch any teardown regression.)
namespace {
struct BaseRecvState {
  IoContext* io;
  TcpStream s;
  char buf[64] = {};
  std::atomic<std::int64_t> rc{999};
};
Task<int> base_recv_server(AllocatorRef, IoContext*, TcpListener* ls,
                           TcpStream* held) {
  auto accepted = co_await ls->accept(kOwner);
  if (!accepted.has_value()) co_return 0;
  TcpStream conn = std::move(*accepted);
  static const char kMsg[] = "base-recv-ok";
  (void)(co_await conn.write_some(std::as_bytes(std::span{kMsg})));
  *held = std::move(conn);  // keep open
  co_return 0;
}
Task<int> base_recv_client(AllocatorRef, IoContext* io, BaseRecvState* s,
                           std::uint16_t port) {
  if (!(co_await s->s.connect(
            Endpoint::loopback(AddressFamily::kIpv4, port))).ok()) co_return 0;
  s->rc.store(co_await RecvOp{
                  &hyle::io::detail::IoContextAccess::proactor_for_current(*io),
                              s->s.native_handle(), s->buf,
                              sizeof(s->buf)},
              std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(IoTimeout, BaseOpCompletionRoutingIntact) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54396;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());
  auto listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener.has_value());
  TcpListener ls = std::move(*listener);

  auto st = std::make_unique<BaseRecvState>();
  st->io = &io;
  auto opened = TcpStream::open(io, kOwner);
  ASSERT_TRUE(opened.has_value());
  st->s = std::move(*opened);
  TcpStream held;
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = base_recv_server(system_allocator(), &io, &ls, &held);
  dc->task = base_recv_client(system_allocator(), &io, st.get(), kPort);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_EQ(st->rc.load(), static_cast<std::int64_t>(sizeof("base-recv-ok")));
  EXPECT_STREQ(st->buf, "base-recv-ok");
}
