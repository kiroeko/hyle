#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "hyle/io/io_context.h"
#include "hyle/io/net_timeout.h"
#include "hyle/io/tcp.h"
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
//      cancelled (its late -ECANCELED/ABORTED completion must NOT UAF the freed
//      slot -- the refcount orphan keeps it alive; TSan + a clean shutdown verify).
//   2. IO wins: the server sends promptly -> the timed recv returns the bytes
//      before the (generous) deadline, and the timer is cancelled cleanly.

using hyle::io::IoContext;
using hyle::io::resolve_addr;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::tcp_listen;
using hyle::io::tcp_socket;
using hyle::io::TimedRecvOp;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;

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
  static Allocator make() noexcept { return Allocator(&a, &f, &r, nullptr); }
};
std::atomic<long> CountingAlloc::live{0};

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// ---- case 1: timeout wins (server accepts, never sends) ---------------------
struct TimeoutResult {
  std::atomic<bool> done{false};
  std::atomic<bool> connected{false};
  std::atomic<std::int64_t> recv_rc{999};  // expect -110 (ETIMEDOUT)
  TcpStream held;  // keep the accepted server conn alive (so it doesn't RST)
};

Task<int> timeout_server(Allocator, IoContext* io, TcpListener* ls,
                         TimeoutResult* r) {
  TcpStream conn = co_await ls->accept(kOwner);
  r->held = std::move(conn);  // hold it open but send NOTHING -> client recv stalls
  co_return 0;
}

Task<int> timeout_client(Allocator alloc, IoContext* io, TimeoutResult* r,
                         std::uint16_t port) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", port, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  r->connected.store(true, std::memory_order_release);
  char buf[64] = {};
  // Deadline 50ms out; server never sends, so the timer must win.
  std::uint64_t deadline = hyle::time::monotonic_ns() + 50'000'000ull;
  std::int64_t rc = co_await TimedRecvOp{&io->proactor_for_current(), s.fd(),
                                         buf, sizeof(buf), alloc, deadline};
  r->recv_rc.store(rc, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

// ---- case 2: io wins (server sends promptly) --------------------------------
struct IoWinResult {
  std::atomic<bool> done{false};
  std::atomic<std::int64_t> recv_rc{999};  // expect > 0 (bytes)
  char got[64] = {};
};

Task<int> iowin_server(Allocator, IoContext* io, TcpListener* ls, IoWinResult*) {
  TcpStream conn = co_await ls->accept(kOwner);
  static const char kMsg[] = "prompt";
  co_await conn.write(kMsg, sizeof(kMsg));
  co_return 0;
}

Task<int> iowin_client(Allocator alloc, IoContext* io, IoWinResult* r,
                       std::uint16_t port) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", port, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) {
    r->done.store(true, std::memory_order_release);
    co_return 0;
  }
  // Generous 5s deadline; the server sends at once, so io must win well within it.
  std::uint64_t deadline = hyle::time::monotonic_ns() + 5'000'000'000ull;
  std::int64_t rc = co_await TimedRecvOp{&io->proactor_for_current(), s.fd(),
                                         r->got, sizeof(r->got), alloc, deadline};
  r->recv_rc.store(rc, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

}  // namespace

TEST(IoTimeout, TimeoutWinsReturnsETimedout) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54390;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(ls.valid()) << "listen failed (port in use?)";

  auto r = std::make_unique<TimeoutResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = timeout_server(system_allocator(), &io, &ls, r.get());
  dc->task = timeout_client(system_allocator(), &io, r.get(), kPort);
  Fiber fs(system_allocator(), kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(system_allocator(), kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_TRUE(r->connected.load());
  EXPECT_EQ(r->recv_rc.load(), -110);  // -ETIMEDOUT: the timer won the race
}

TEST(IoTimeout, IoWinsReturnsBytes) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54391;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(ls.valid()) << "listen failed (port in use?)";

  auto r = std::make_unique<IoWinResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = iowin_server(system_allocator(), &io, &ls, r.get());
  dc->task = iowin_client(system_allocator(), &io, r.get(), kPort);
  Fiber fs(system_allocator(), kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(system_allocator(), kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_EQ(r->recv_rc.load(), static_cast<std::int64_t>(sizeof("prompt")));
  EXPECT_STREQ(r->got, "prompt");
}

// Leak check: run the timeout-wins path with a COUNTING allocator and assert the
// heap TimedSlot is fully reclaimed after teardown. This is the guard for the
// "timeout path never releases the kernel-ref" bug class (epoll synth-cancel +
// shutdown orphan drain) that functional asserts + TSan both miss.
TEST(IoTimeout, TimeoutPathDoesNotLeakSlot) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54392;
  CountingAlloc::live.store(0, std::memory_order_relaxed);
  {
    Scheduler sched(kWorkers, system_allocator(), 256);
    IoContext io(sched, kWorkers);
    ASSERT_TRUE(io.valid());
    TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
    ASSERT_TRUE(ls.valid()) << "listen failed (port in use?)";

    auto r = std::make_unique<TimeoutResult>();
    auto ds = std::make_unique<Drive>();
    auto dc = std::make_unique<Drive>();
    ds->task = timeout_server(system_allocator(), &io, &ls, r.get());
    // The timed op's slot comes from CountingAlloc -> its alloc/free is counted.
    dc->task = timeout_client(CountingAlloc::make(), &io, r.get(), kPort);
    Fiber fs(system_allocator(), kDefaultStackSize, &drive_entry, ds.get());
    Fiber fc(system_allocator(), kDefaultStackSize, &drive_entry, dc.get());
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

