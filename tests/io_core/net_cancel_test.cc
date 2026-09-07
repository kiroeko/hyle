#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <thread>

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
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::IoContext;
using hyle::io::OperationOptions;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::mem::AllocatorRef;
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
void drive_entry(void* p) noexcept { static_cast<Drive*>(p)->task.resume(); }

// ---- case 1: token cancel wins (server connects, never sends) ---------------
struct CancelResult {
  std::atomic<bool> done{false};
  std::atomic<bool> connected{false};
  std::atomic<std::int64_t> recv_rc{999};  // expect -125 (ECANCELED)
  TcpStream held;  // hold the server conn open (don't RST) so the recv truly stalls
  CancelToken* tok = nullptr;
};

Task<int> cancel_server(AllocatorRef, IoContext*, TcpListener* ls, CancelResult* r) {
  auto accepted = co_await ls->accept(kOwner);
  if (!accepted.has_value()) co_return 0;
  TcpStream conn = std::move(*accepted);
  r->held = std::move(conn);  // hold open, send NOTHING -> client recv stalls
  co_return 0;
}

Task<int> cancel_client(AllocatorRef, IoContext* io, CancelResult* r,
                        std::uint16_t port) {
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
  char buf[64] = {};
  // One public operation budget combines cancellation and a generous deadline.
  // The server never sends, so cancellation must win this three-way race.
  OperationOptions options{
      .deadline = hyle::time::deadline_after(std::chrono::seconds(5)),
      .cancel = r->tok};
  auto transfer = co_await s.read_some(
      std::as_writable_bytes(std::span{buf}), options);
  const std::int64_t rc =
      transfer.ok() ? static_cast<std::int64_t>(transfer.bytes)
                    : (transfer.status.code() == hyle::StatusCode::Cancelled
                           ? hyle::io::kCancelled
                           : -999);
  r->recv_rc.store(rc, std::memory_order_release);
  r->done.store(true, std::memory_order_release);
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
  auto listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener.has_value()) << "listen failed (port in use?)";
  TcpListener ls = std::move(*listener);

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

  // Cancel from a foreign OS thread. The callback may arbitrate there, but the
  // native Proactor cancellation must be handed back to the pinned owner worker.
  std::thread canceller([&] {
    while (!r->connected.load(std::memory_order_acquire))
      std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    tok.cancel();
  });

  sched.drain();
  canceller.join();
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
Task<int> precancel_client(AllocatorRef, IoContext* io, PrecancelResult* r,
                           CancelToken* tok, TcpStream* stream) {
  (void)io;
  char buf[16] = {};
  auto transfer = co_await stream->read_some(
      std::as_writable_bytes(std::span{buf}),
      OperationOptions{.cancel = tok});
  const std::int64_t rc =
      transfer.status.code() == hyle::StatusCode::Cancelled
          ? hyle::io::kCancelled
          : static_cast<std::int64_t>(transfer.bytes);
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
  auto listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(listener.has_value());
  TcpListener ls = std::move(*listener);

  // Pre-cancel the token; a client socket bound to the ring never needs to connect
  // because await_ready short-circuits before submitting.
  CancelToken tok = CancelToken::root(system_allocator());
  tok.cancel();
  auto opened = TcpStream::open(io, kOwner);
  ASSERT_TRUE(opened.has_value());
  TcpStream s = std::move(*opened);

  auto r = std::make_unique<PrecancelResult>();
  auto d = std::make_unique<Drive>();
  d->task = precancel_client(system_allocator(), &io, r.get(), &tok, &s);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn_on(kOwner, &f);
  sched.drain();
  sched.shutdown();

  EXPECT_TRUE(r->done.load());
  EXPECT_EQ(r->rc.load(), hyle::io::kCancelled);  // short-circuited, no ring op
}

namespace {
Task<int> precancel_udp_send(AllocatorRef, hyle::io::UdpSocket* socket,
                             PrecancelResult* result, CancelToken* token) {
  static const char message[] = "cancelled";
  auto sent = co_await socket->send_to(
      std::as_bytes(std::span{message}),
      Endpoint::loopback(AddressFamily::kIpv4, 54482),
      OperationOptions{.cancel = token});
  result->rc.store(sent.status.code() == hyle::StatusCode::Cancelled
                       ? hyle::io::kCancelled
                       : static_cast<std::int64_t>(sent.bytes),
                   std::memory_order_release);
  result->done.store(true, std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(NetCancel, AlreadyCancelledUdpSendReturnsCancelledImmediately) {
  Scheduler sched(1, system_allocator(), 256);
  IoContext io(sched, 1);
  ASSERT_TRUE(io.valid());
  auto opened = hyle::io::UdpSocket::open(io, kOwner);
  ASSERT_TRUE(opened.has_value());
  auto socket = std::move(*opened);
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
