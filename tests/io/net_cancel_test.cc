#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

#include "hyle/io/io_context.h"
#include "hyle/io/net_cancel.h"
#include "hyle/io/tcp.h"
#include "hyle/io/udp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/timer.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#endif

// sched-07 / io end-to-end: a CancelToken-bound recv over loopback TCP. When the
// token is cancelled while the recv is in flight (server connected but sends
// nothing), the in-flight kernel op is actively cancelled (submit_cancel) and the
// recv returns -ECANCELED (-125) -- the structured-cancellation counterpart to the
// timeout path in timeout_test.cc. This is the io side of the HTTP-request fan-out
// consumer: cancelling a request cancels the recv its handler is blocked on.
//
// Also the fan-out shape: one request-root token derives N child tokens; cancelling
// the root wakes every child's blocked recv with -ECANCELED (connected-cancel).

using hyle::io::CancelRecvOp;
using hyle::io::IoContext;
using hyle::io::resolve_addr;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::tcp_listen;
using hyle::io::tcp_socket;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::CancelToken;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// ---- case 1: token cancel wins (server connects, never sends) ---------------
struct CancelResult {
  std::atomic<bool> done{false};
  std::atomic<bool> connected{false};
  std::atomic<std::int64_t> recv_rc{999};  // expect -125 (ECANCELED)
  TcpStream held;  // hold the server conn open (don't RST) so the recv truly stalls
  CancelToken* tok = nullptr;
};

Task<int> cancel_server(Allocator, IoContext*, TcpListener* ls, CancelResult* r) {
  TcpStream conn = co_await ls->accept(kOwner);
  r->held = std::move(conn);  // hold open, send NOTHING -> client recv stalls
  co_return 0;
}

Task<int> cancel_client(Allocator, IoContext* io, CancelResult* r,
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
  // A cancellable recv bound to the request token. The server never sends, so only
  // a token cancel can unblock it (-ECANCELED).
  std::int64_t rc =
      co_await s.read_cancel(buf, sizeof(buf), r->tok);
  r->recv_rc.store(rc, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}

// A fiber that, once the client has connected + submitted its recv, cancels the
// token. It runs on the SAME owner worker (shard-per-core), so the cancel callback
// (submit_cancel) and the recv's reap are single-threaded. It sleeps briefly (a real
// timer wake, not a park-forever spin) to let the client fiber reach suspend on the
// recv before cancelling.
Task<int> canceller(Allocator, CancelResult* r) {
  while (!r->connected.load(std::memory_order_acquire)) {
    hyle::sched::sleep_for_ns(1'000'000ull);  // 1ms: yields + timer-wakes
  }
  hyle::sched::sleep_for_ns(20'000'000ull);  // 20ms: let the recv get in flight
  r->tok->cancel();
  co_return 0;
}

}  // namespace

// The canceller needs to be woken to run its yields; simplest is to not park it.
// Instead we drive cancellation from the main thread after connected + a sleep.
TEST(NetCancel, TokenCancelUnblocksRecv) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54480;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(ls.valid()) << "listen failed (port in use?)";

  CancelToken tok = CancelToken::root(system_allocator());
  auto r = std::make_unique<CancelResult>();
  r->tok = &tok;
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = cancel_server(system_allocator(), &io, &ls, r.get());
  dc->task = cancel_client(system_allocator(), &io, r.get(), kPort);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);

  // A canceller fiber pinned to the SAME owner worker cancels the token once the
  // client has connected (so submit_cancel + reap stay single-threaded, shard-per-
  // core). It is spawned up front and internally waits on connected via sleep.
  auto dk = std::make_unique<Drive>();
  dk->task = canceller(system_allocator(), r.get());
  Fiber fk(kDefaultStackSize, &drive_entry, dk.get());
  sched.spawn_on(kOwner, &fk);

  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_TRUE(r->connected.load());
  EXPECT_EQ(r->recv_rc.load(), hyle::io::kCancelled);  // normalized: token cancel won
}

// A recv whose token is ALREADY cancelled before it starts returns -ECANCELED
// without ever touching the ring (await_ready short-circuit).
namespace {
struct PrecancelResult {
  std::atomic<bool> done{false};
  std::atomic<std::int64_t> rc{999};
};
Task<int> precancel_client(Allocator, IoContext* io, PrecancelResult* r,
                           CancelToken* tok, std::intptr_t fd) {
  char buf[16] = {};
  std::int64_t rc =
      co_await CancelRecvOp{&io->proactor_for_current(), fd, buf, sizeof(buf), tok};
  r->rc.store(rc, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(NetCancel, AlreadyCancelledRecvReturnsCancelledImmediately) {
  constexpr unsigned kWorkers = 2;
  constexpr std::uint16_t kPort = 54481;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(ls.valid());

  // Pre-cancel the token; a client socket bound to the ring never needs to connect
  // because await_ready short-circuits before submitting.
  CancelToken tok = CancelToken::root(system_allocator());
  tok.cancel();
  TcpStream s = tcp_socket(io, kOwner);
  ASSERT_TRUE(s.valid());

  auto r = std::make_unique<PrecancelResult>();
  auto d = std::make_unique<Drive>();
  d->task = precancel_client(system_allocator(), &io, r.get(), &tok, s.fd());
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_EQ(r->rc.load(), hyle::io::kCancelled);  // short-circuited, no ring op
}

namespace {
Task<int> precancel_udp_send(Allocator, hyle::io::UdpSocket* socket,
                             PrecancelResult* result, CancelToken* token) {
  alignas(16) unsigned char address[128]{};
  std::uint32_t address_length = 0;
  resolve_addr("127.0.0.1", 54482, address, &address_length);
  static const char message[] = "cancelled";
  result->rc.store(
      co_await socket->send_to_cancel(message, sizeof(message), address,
                                      address_length, token),
      std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(NetCancel, AlreadyCancelledUdpSendReturnsCancelledImmediately) {
  Scheduler sched(1, system_allocator(), 256);
  IoContext io(sched, 1);
  ASSERT_TRUE(io.valid());
  auto socket = hyle::io::udp_socket(io, kOwner);
  ASSERT_TRUE(socket.valid());
  CancelToken token = CancelToken::root(system_allocator());
  token.cancel();

  auto result = std::make_unique<PrecancelResult>();
  auto drive = std::make_unique<Drive>();
  drive->task = precancel_udp_send(system_allocator(), &socket, result.get(),
                                   &token);
  Fiber fiber(kDefaultStackSize, &drive_entry, drive.get());
  sched.spawn_on(kOwner, &fiber);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(result->done.load());
  EXPECT_EQ(result->rc.load(), hyle::io::kCancelled);
}
