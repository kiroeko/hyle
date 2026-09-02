// io echo benchmark (io-04): the honest io perf comparison the io-01/02/03 notes
// deferred until there was a real concurrent-connection load. Isolated single
// read/write measures the kernel, not HYLE -- so this drives a realistic echo
// workload (N connections each doing R round-trips) through HYLE's shard-per-core
// stack and reports round-trips/sec.
//
// Three honest measurements:
//   1. HYLE echo throughput -- HYLE's own number (co_await socket layer +
//      shard-per-core scheduler driving per-core proactor rings).
//   2. Shard scaling -- 1/2/4/8 workers on the SAME connection count: does
//      throughput scale with cores (the whole point of shard-per-core)?
//   3. vs raw epoll (Linux) -- a hand-written single-thread readiness-loop echo
//      over the same socketpairs, no HYLE. Proves HYLE's proactor stack isn't
//      slower than the plain kernel primitive every server dev reaches for.
//
// Run under scripts/bench-stat.py -n 15 for the statistical (tie/win) verdict;
// a one-shot ratio is a coin flip (see bench-stat.py rationale).

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif
#if defined(__linux__)
#  include <sys/epoll.h>
#endif
#if defined(__linux__) && defined(HYLE_BENCH_LIBURING)
#  include <liburing.h>
#endif
#if defined(__linux__) && defined(HYLE_BENCH_ASIO)
#  include <asio.hpp>
#endif
#if defined(HYLE_BENCH_LIBUV)
#  include <uv.h>
#  if defined(__linux__)
#    include <csignal>  // SIGPIPE ignore for the libuv teardown write race
#  endif
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#endif

using hyle::io::IoContext;
using hyle::io::RecvOp;
using hyle::io::SendOp;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr int kMsgLen = 64;      // echo payload size
constexpr int kRoundTrips = 64;  // round-trips per connection per epoch
constexpr int kConns = 64;       // concurrent connections

// Pipelined workload knobs: kPipeConns independent echo streams sharing a worker,
// each doing kPipeRoundTrips round-trips. Because MANY streams live on one core,
// a single reap wakes MANY fibers that each immediately submit -- so N SQEs pile
// up before the next reap. THIS is the many-in-flight load that fills the SQ and
// lets submit-batching matter (the plain 64-conn ping-pong above keeps ~1 op in
// flight per stream at a time, so the SQ never fills -- batching can't help it).
#if defined(__linux__)
constexpr int kPipeConns = 512;       // concurrent streams (high fan-out, >> ring depth 256 -> forces the worker's blocking GETEVENTS reap path)
constexpr int kPipeRoundTrips = 16;   // round-trips per stream per epoch
#endif

// A connected fd pair.
struct Pair {
  std::intptr_t a = -1;
  std::intptr_t b = -1;
};

#if !defined(_WIN32)
Pair make_pair() {
  int sv[2];
  Pair p;
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0) {
    for (int fd : sv) {
      const int flags = ::fcntl(fd, F_GETFL, 0);
      if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        ::close(sv[0]);
        ::close(sv[1]);
        return p;
      }
      ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    }
    p.a = sv[0];
    p.b = sv[1];
  }
  return p;
}
void close_pair(const Pair& p) {
  if (p.a >= 0) ::close(static_cast<int>(p.a));
  if (p.b >= 0) ::close(static_cast<int>(p.b));
}
#endif
#if defined(_WIN32)
Pair make_pair() {
  Pair p;
  SOCKET ls = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (ls == INVALID_SOCKET) return p;
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
  sa.sin_port = 0;
  if (::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) == SOCKET_ERROR ||
      ::listen(ls, 1) == SOCKET_ERROR) {
    ::closesocket(ls);
    return p;
  }
  int slen = sizeof(sa);
  if (::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen) ==
      SOCKET_ERROR) {
    ::closesocket(ls);
    return p;
  }
  SOCKET c = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  if (c == INVALID_SOCKET ||
      ::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) ==
          SOCKET_ERROR) {
    if (c != INVALID_SOCKET) ::closesocket(c);
    ::closesocket(ls);
    return p;
  }
  SOCKET s = ::accept(ls, nullptr, nullptr);
  ::closesocket(ls);
  if (s == INVALID_SOCKET) {
    ::closesocket(c);
    return p;
  }
  p.a = static_cast<std::intptr_t>(c);
  p.b = static_cast<std::intptr_t>(s);
  return p;
}
void close_pair(const Pair& p) {
  // These are socketpair emulation endpoints, not a connection-lifecycle
  // workload. Abortive close keeps repeated process-level sampling from
  // exhausting Windows' ~16K ephemeral-port range with TIME_WAIT sockets.
  // Teardown is outside the measured echo work conceptually and both compared
  // variants use the identical helper.
  linger reset{1, 0};
  for (std::intptr_t fd : {p.a, p.b}) {
    if (fd == -1) continue;
    ::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_LINGER,
                 reinterpret_cast<const char*>(&reset), sizeof(reset));
    ::closesocket(static_cast<SOCKET>(fd));
  }
}
#endif

// One connection's two fibers. The client sends then reads the echo, R times;
// the server reads then echoes, R times. Both pinned to the connection's owner
// worker (shard-per-core): all io stays on that core.
struct Conn {
  IoContext* io;
  std::intptr_t client_fd;
  std::intptr_t server_fd;
  int round_trips;
  std::atomic<int>* completed;
  char cbuf[kMsgLen];
  char sbuf[kMsgLen];
};

Task<int> client_task(AllocatorRef, Conn* c) {
  static const char kMsg[kMsgLen] = "hyle-io-echo-bench-payload";
  for (int i = 0; i < c->round_trips; ++i) {
    std::int64_t w = co_await SendOp{&c->io->proactor_for_current(), c->client_fd,
                                     kMsg, kMsgLen};
    if (w != kMsgLen) break;
    std::int64_t r = co_await RecvOp{&c->io->proactor_for_current(), c->client_fd,
                                     c->cbuf, kMsgLen};
    if (r != kMsgLen) break;
    c->completed->fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

Task<int> server_task(AllocatorRef, Conn* c) {
  for (int i = 0; i < c->round_trips; ++i) {
    std::int64_t r = co_await RecvOp{&c->io->proactor_for_current(), c->server_fd,
                                     c->sbuf, kMsgLen};
    if (r <= 0) break;
    std::int64_t w = co_await SendOp{&c->io->proactor_for_current(), c->server_fd,
                                     c->sbuf, static_cast<std::uint32_t>(r)};
    if (w != r) break;
  }
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// Run kConns connections x kRoundTrips echo round-trips on `workers` cores,
// connections sharded round-robin across the cores. Returns after all complete.
int run_hyle_echo(unsigned workers) {
  Scheduler sched(workers, system_allocator(), 4096);
  IoContext io(sched, workers);

  std::vector<Pair> pairs(kConns);
  std::vector<std::unique_ptr<Conn>> conns;
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  std::atomic<int> completed{0};
  for (int i = 0; i < kConns; ++i) {
    pairs[i] = make_pair();
    unsigned owner = static_cast<unsigned>(i) % workers;
    io.bind_fd(owner, pairs[i].a);
    io.bind_fd(owner, pairs[i].b);
    auto c = std::make_unique<Conn>();
    c->io = &io;
    c->client_fd = pairs[i].a;
    c->server_fd = pairs[i].b;
    c->round_trips = kRoundTrips;
    c->completed = &completed;
    auto dc = std::make_unique<Drive>();
    auto ds = std::make_unique<Drive>();
    dc->task = client_task(system_allocator(), c.get());
    ds->task = server_task(system_allocator(), c.get());
    auto fc =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, dc.get());
    auto fs =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, ds.get());
    sched.spawn_on(owner, fs.get());
    sched.spawn_on(owner, fc.get());
    conns.push_back(std::move(c));
    drives.push_back(std::move(dc));
    drives.push_back(std::move(ds));
    fibers.push_back(std::move(fc));
    fibers.push_back(std::move(fs));
  }
  sched.drain();
  sched.shutdown();
  for (auto& p : pairs) close_pair(p);
  return completed.load(std::memory_order_relaxed);
}

// Pipelined / high-fan-out variant: kPipeConns streams x kPipeRoundTrips on
// `workers` cores. With hundreds of streams per core, one reap wakes dozens of
// fibers that each submit -- so many SQEs accumulate between reaps (the SQ can
// fill). This is the load where submit-batching (one io_uring_enter for many
// SQEs) can pay off; the plain ping-pong bench cannot exercise it.
#if defined(__linux__)
int run_hyle_pipelined(unsigned workers) {
  Scheduler sched(workers, system_allocator(), 8192);
  IoContext io(sched, workers);

  std::vector<Pair> pairs(kPipeConns);
  std::vector<std::unique_ptr<Conn>> conns;
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  std::atomic<int> completed{0};
  for (int i = 0; i < kPipeConns; ++i) {
    pairs[i] = make_pair();
    unsigned owner = static_cast<unsigned>(i) % workers;
    io.bind_fd(owner, pairs[i].a);
    io.bind_fd(owner, pairs[i].b);
    auto c = std::make_unique<Conn>();
    c->io = &io;
    c->client_fd = pairs[i].a;
    c->server_fd = pairs[i].b;
    c->round_trips = kPipeRoundTrips;
    c->completed = &completed;
    auto dc = std::make_unique<Drive>();
    auto ds = std::make_unique<Drive>();
    dc->task = client_task(system_allocator(), c.get());
    ds->task = server_task(system_allocator(), c.get());
    auto fc =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, dc.get());
    auto fs =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, ds.get());
    sched.spawn_on(owner, fs.get());
    sched.spawn_on(owner, fc.get());
    conns.push_back(std::move(c));
    drives.push_back(std::move(dc));
    drives.push_back(std::move(ds));
    fibers.push_back(std::move(fc));
    fibers.push_back(std::move(fs));
  }
  sched.drain();
  sched.shutdown();
  for (auto& p : pairs) close_pair(p);
  return completed.load(std::memory_order_relaxed);
}
#endif

#if defined(__linux__)
// Raw epoll echo baseline: a single thread runs a classic readiness loop over
// all connections. Each connection is a socketpair; we drive both ends. This is
// the "plain epoll server" every dev knows -- no HYLE, no fibers, no io_uring.
// Same total work (kConns x kRoundTrips echoes) so the ratio is apples-to-apples
// against the 1-worker HYLE row.
void run_raw_epoll_echo() {
  static const char kMsg[kMsgLen] = "hyle-io-echo-bench-payload";
  std::vector<Pair> pairs(kConns);
  for (int i = 0; i < kConns; ++i) pairs[i] = make_pair();

  // Prime: client sends the first message on every connection.
  for (int i = 0; i < kConns; ++i)
    (void)!::write(static_cast<int>(pairs[i].a), kMsg, kMsgLen);

  int ep = ::epoll_create1(0);
  // Watch every server + client fd for readability. data.u32 encodes conn*2 +
  // (0=server,1=client) so we know which side fired.
  for (int i = 0; i < kConns; ++i) {
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u32 = static_cast<std::uint32_t>(i * 2);  // server end
    ::epoll_ctl(ep, EPOLL_CTL_ADD, static_cast<int>(pairs[i].b), &ev);
    ev.data.u32 = static_cast<std::uint32_t>(i * 2 + 1);  // client end
    ::epoll_ctl(ep, EPOLL_CTL_ADD, static_cast<int>(pairs[i].a), &ev);
  }

  // Count remaining round-trips per connection. A server read+echo advances
  // nothing; a client read completes one round-trip and (if more remain) sends
  // again. Done when every connection hit kRoundTrips.
  std::vector<int> client_done(kConns, 0);
  int live = kConns;
  char buf[kMsgLen];
  epoll_event evs[256];
  while (live > 0) {
    int n = ::epoll_wait(ep, evs, 256, -1);
    for (int e = 0; e < n; ++e) {
      std::uint32_t tag = evs[e].data.u32;
      int conn = static_cast<int>(tag / 2);
      bool is_client = (tag & 1) != 0;
      if (!is_client) {
        // server side readable: read the request, echo it back.
        ssize_t r = ::read(static_cast<int>(pairs[conn].b), buf, kMsgLen);
        if (r > 0) (void)!::write(static_cast<int>(pairs[conn].b), buf, r);
      } else {
        // client side readable: the echo arrived = one round-trip done.
        ssize_t r = ::read(static_cast<int>(pairs[conn].a), buf, kMsgLen);
        if (r > 0) {
          if (++client_done[conn] >= kRoundTrips) {
            --live;
          } else {
            (void)!::write(static_cast<int>(pairs[conn].a), kMsg, kMsgLen);  // next
          }
        }
      }
    }
  }
  ::close(ep);
  for (auto& p : pairs) close_pair(p);
}
#endif

#if defined(__linux__) && defined(HYLE_BENCH_LIBURING)
// Raw liburing echo baseline: the SAME 64-conn x 64-round-trip x 64B ping-pong load
// as run_hyle_echo(1 worker), but driven by a hand-written liburing state machine on
// ONE ring -- no fibers, no coroutines, no scheduler. This is the honest "does HYLE's
// co_await + shard-per-core stack add overhead over a bare io_uring loop written with
// the community-standard wrapper?" comparison (echo_bench's other baseline, raw epoll,
// answers "vs the OLD readiness model"; this answers "vs the SAME io_uring, hand-rolled").
//
// State machine per connection (encoded in user_data = conn_idx<<2 | stage):
//   - prime: every client sends kMsg.
//   - SERVER_RECV completes -> server sends the echo back (SERVER side).
//   - CLIENT_RECV completes -> one round-trip done; if <kRoundTrips, client sends again.
// The server's recv is (re)armed whenever we expect a request; the client's recv after
// each of its sends. Loop: submit_and_wait(1) + io_uring_for_each_cqe, until every
// connection hits kRoundTrips. Ring depth is generous (4096) so the SQ never starves.
enum LuStage { STAGE_CLIENT_RECV = 0, STAGE_SERVER_RECV = 1 };

void run_raw_liburing_echo() {
  static const char kMsg[kMsgLen] = "hyle-io-echo-bench-payload";
  std::vector<Pair> pairs(kConns);
  for (int i = 0; i < kConns; ++i) pairs[i] = make_pair();

  io_uring ring;
  // 4096 SQ entries: 64 conns each with at most a couple ops in flight -> never starves.
  if (io_uring_queue_init(4096, &ring, 0) < 0) {
    for (auto& p : pairs) close_pair(p);
    return;
  }

  std::vector<char> cbuf(static_cast<std::size_t>(kConns) * kMsgLen);
  std::vector<char> sbuf(static_cast<std::size_t>(kConns) * kMsgLen);
  std::vector<int> rt_done(kConns, 0);

  auto ud = [](int conn, LuStage st) -> std::uint64_t {
    return (static_cast<std::uint64_t>(conn) << 2) | static_cast<std::uint64_t>(st);
  };

  // Prime: arm each server recv, then each client send its first request.
  for (int i = 0; i < kConns; ++i) {
    io_uring_sqe* s = io_uring_get_sqe(&ring);
    io_uring_prep_recv(s, static_cast<int>(pairs[i].b), &sbuf[i * kMsgLen], kMsgLen, 0);
    io_uring_sqe_set_data64(s, ud(i, STAGE_SERVER_RECV));
    io_uring_sqe* c = io_uring_get_sqe(&ring);
    io_uring_prep_send(c, static_cast<int>(pairs[i].a), kMsg, kMsgLen, 0);
    io_uring_sqe_set_data64(c, ~0ull);  // client-send completion: ignored
  }

  int live = kConns;
  while (live > 0) {
    io_uring_submit_and_wait(&ring, 1);
    unsigned head;
    io_uring_cqe* cqe;
    unsigned count = 0;
    io_uring_for_each_cqe(&ring, head, cqe) {
      ++count;
      std::uint64_t data = cqe->user_data;
      std::int32_t res = cqe->res;
      if (data == ~0ull) continue;  // a client-send completion; nothing to do
      int conn = static_cast<int>(data >> 2);
      LuStage st = static_cast<LuStage>(data & 3);
      if (res <= 0) { --live; continue; }
      if (st == STAGE_SERVER_RECV) {
        // Server got the request -> echo it back (server send).
        io_uring_sqe* s = io_uring_get_sqe(&ring);
        io_uring_prep_send(s, static_cast<int>(pairs[conn].b), &sbuf[conn * kMsgLen],
                           static_cast<unsigned>(res), 0);
        io_uring_sqe_set_data64(s, ~0ull);  // server-send completion: ignored
        // Arm the client's recv for the echo.
        io_uring_sqe* c = io_uring_get_sqe(&ring);
        io_uring_prep_recv(c, static_cast<int>(pairs[conn].a), &cbuf[conn * kMsgLen],
                           kMsgLen, 0);
        io_uring_sqe_set_data64(c, ud(conn, STAGE_CLIENT_RECV));
      } else {  // STAGE_CLIENT_RECV: the echo arrived -> one round-trip complete.
        if (++rt_done[conn] >= kRoundTrips) {
          --live;
        } else {
          // Next round-trip: client sends again + re-arm server recv.
          io_uring_sqe* c = io_uring_get_sqe(&ring);
          io_uring_prep_send(c, static_cast<int>(pairs[conn].a), kMsg, kMsgLen, 0);
          io_uring_sqe_set_data64(c, ~0ull);
          io_uring_sqe* s = io_uring_get_sqe(&ring);
          io_uring_prep_recv(s, static_cast<int>(pairs[conn].b), &sbuf[conn * kMsgLen],
                             kMsgLen, 0);
          io_uring_sqe_set_data64(s, ud(conn, STAGE_SERVER_RECV));
        }
      }
    }
    io_uring_cq_advance(&ring, count);
  }

  io_uring_queue_exit(&ring);
  for (auto& p : pairs) close_pair(p);
}
#endif

#if defined(__linux__) && defined(HYLE_BENCH_ASIO)
// asio echo baseline: the SAME 64-conn x 64-round-trip x 64B ping-pong as
// run_hyle_echo(1 worker), driven by asio's async_read/async_write over ONE
// io_context on ONE thread (io_uring backend -- ASIO_HAS_IO_URING). This is the
// "vs a mature async framework built on the SAME io_uring" comparison: asio is THE
// canonical C++ async-io library, so the ratio answers "does HYLE's shard-per-core
// stack keep up with the framework everyone reaches for?". asio's completion handlers
// are its abstraction seam (vs HYLE's co_await); both run on io_uring. Single thread
// = the honest single-core comparison (HYLE-1w row); HYLE's multi-worker rows then
// show what shard-per-core adds that asio's single io_context does not give for free.
void run_asio_echo() {
  static const char kMsg[kMsgLen] = "hyle-io-echo-bench-payload";
  asio::io_context ctx(1);
  std::vector<Pair> pairs(kConns);
  for (int i = 0; i < kConns; ++i) pairs[i] = make_pair();

  // Per-connection state: two asio sockets (client + server end of the socketpair),
  // the buffers, and a remaining round-trip counter. The chain is:
  //   client async_write(req) -> server async_read -> server async_write(echo)
  //   -> client async_read -> (--rt) client async_write ... until rt == 0.
  struct AConn {
    asio::posix::stream_descriptor client;
    asio::posix::stream_descriptor server;
    char cbuf[kMsgLen];
    char sbuf[kMsgLen];
    int rt;
  };
  std::vector<std::unique_ptr<AConn>> conns;
  conns.reserve(kConns);
  for (int i = 0; i < kConns; ++i) {
    conns.push_back(std::make_unique<AConn>(AConn{
        asio::posix::stream_descriptor(ctx, static_cast<int>(pairs[i].a)),
        asio::posix::stream_descriptor(ctx, static_cast<int>(pairs[i].b)),
        {}, {}, kRoundTrips}));
  }

  // Recursive handler chains, value-semantic (each continuation captures the raw
  // AConn* -- stable for the whole ctx.run(), since `conns` owns them). No heap /
  // shared_ptr: the ONLY state is in *p. Termination: after its last round-trip the
  // client CLOSES its descriptor, so the server's outstanding async_read completes
  // with EOF (n==0) -> no ops left in flight -> ctx.run() returns (same "client close
  // -> server EOF" teardown as the HYLE + liburing rows).
  for (auto& c : conns) {
    AConn* p = c.get();
    // Server: read a request, echo it back, re-arm (until the client closes -> EOF).
    struct Srv {
      AConn* p;
      void operator()(const std::error_code& ec, std::size_t n) const {
        if (ec || n == 0) return;  // EOF/err: client closed -> stop (op completes)
        asio::async_write(p->server, asio::buffer(p->sbuf, n),
                          [p = p](const std::error_code& e, std::size_t) {
                            if (e) return;
                            asio::async_read(p->server, asio::buffer(p->sbuf, kMsgLen), Srv{p});
                          });
      }
    };
    asio::async_read(p->server, asio::buffer(p->sbuf, kMsgLen), Srv{p});
    // Client: write a request, read the echo, --rt; at 0 close (EOF-signals server).
    struct Cli {
      AConn* p;
      void operator()() const {
        asio::async_write(p->client, asio::buffer(kMsg, kMsgLen),
                          [p = p](const std::error_code& e, std::size_t) {
                            if (e) return;
                            asio::async_read(p->client, asio::buffer(p->cbuf, kMsgLen),
                                             [p = p](const std::error_code& ec, std::size_t) {
                                               if (ec) return;
                                               if (--p->rt > 0) Cli{p}();
                                               else { std::error_code ig; p->client.close(ig); }
                                             });
                          });
      }
    };
    Cli{p}();
  }
  ctx.run();
  for (auto& p : pairs) close_pair(p);
}
#endif  // linux && HYLE_BENCH_ASIO

#if defined(HYLE_BENCH_LIBUV)
// libuv echo baseline: the SAME 64-conn x 64-round-trip x 64B ping-pong, driven by
// libuv's uv_read_start / uv_write over ONE uv_loop on ONE thread (epoll backend --
// libuv does not use io_uring by default). libuv is THE C async-io runtime (Node.js's
// engine), so this answers "vs the most battle-tested async C runtime?". Its callback
// model is the abstraction seam. NOTE: libuv is epoll-based, so the fairest HYLE
// comparison for this row is the raw-epoll baseline / HYLE forced-epoll, not the
// io_uring rows -- read it as "HYLE's proactor vs libuv's reactor", cross-model.
void run_libuv_echo() {
  static const char kMsg[kMsgLen] = "hyle-io-echo-bench-payload";
  uv_loop_t loop;
  uv_loop_init(&loop);

  struct UConn {
    uv_tcp_t client;    // client end (uv_tcp over a uv_socketpair socket)
    uv_tcp_t server;    // server end
    uv_write_t cwreq;
    uv_write_t swreq;
    char cbuf[kMsgLen];
    char sbuf[kMsgLen];
    int rt;
  };
  std::vector<std::unique_ptr<UConn>> conns;
  conns.reserve(kConns);

  auto alloc_cb = [](uv_handle_t* h, size_t, uv_buf_t* buf) {
    auto* u = static_cast<UConn*>(h->data);
    if (reinterpret_cast<uv_tcp_t*>(h) == &u->server)
      *buf = uv_buf_init(u->sbuf, kMsgLen);
    else
      *buf = uv_buf_init(u->cbuf, kMsgLen);
  };

  // Chain on the WRITE-COMPLETE callback (a uv_write_t can't be reused while in
  // flight). Reads are continuous (uv_read_start) so on_*_read re-fires per message.
  // CONTRACT: libuv delivers nread==0 for "no data available right now" (EAGAIN-like),
  // which is NOT end-of-stream -- only nread<0 (UV_EOF) is. Closing on n==0 (the
  // `<= 0` trap) wrongly tears the connection down after the first read; guard n<0.
  struct Cbs {
    static void on_srv_write(uv_write_t*, int) {}
    static void on_cli_write(uv_write_t*, int) {}
    static void on_srv_read(uv_stream_t* s, ssize_t nread, const uv_buf_t*) {
      auto* u = static_cast<UConn*>(s->data);
      if (nread == 0) return;  // EAGAIN, not EOF -- keep the read armed
      if (nread < 0) { uv_close(reinterpret_cast<uv_handle_t*>(&u->server), nullptr); return; }
      uv_buf_t wb = uv_buf_init(u->sbuf, static_cast<size_t>(nread));
      uv_write(&u->swreq, reinterpret_cast<uv_stream_t*>(&u->server), &wb, 1, on_srv_write);
    }
    static void on_cli_read(uv_stream_t* s, ssize_t nread, const uv_buf_t*) {
      auto* u = static_cast<UConn*>(s->data);
      if (nread == 0) return;  // EAGAIN, not EOF
      if (nread < 0) { uv_close(reinterpret_cast<uv_handle_t*>(&u->client), nullptr); return; }
      if (--u->rt > 0) {
        uv_buf_t wb = uv_buf_init(const_cast<char*>(kMsg), kMsgLen);
        uv_write(&u->cwreq, reinterpret_cast<uv_stream_t*>(&u->client), &wb, 1, on_cli_write);
      } else {
        uv_close(reinterpret_cast<uv_handle_t*>(&u->client), nullptr);  // -> server EOF
      }
    }
  };

  for (int i = 0; i < kConns; ++i) {
    // libuv-owned socketpair (uv_tcp needs libuv's non-blocking setup, not a raw
    // ::socketpair fd opened as uv_pipe -- that stalled after one read on this kernel).
    uv_os_sock_t sv[2];
    if (uv_socketpair(SOCK_STREAM, 0, sv, UV_NONBLOCK_PIPE, UV_NONBLOCK_PIPE) != 0) break;
    auto u = std::make_unique<UConn>();
    u->rt = kRoundTrips;
    uv_tcp_init(&loop, &u->client);
    uv_tcp_init(&loop, &u->server);
    uv_tcp_open(&u->client, sv[0]);
    uv_tcp_open(&u->server, sv[1]);
    u->client.data = u.get();
    u->server.data = u.get();
    uv_read_start(reinterpret_cast<uv_stream_t*>(&u->server), alloc_cb, Cbs::on_srv_read);
    uv_read_start(reinterpret_cast<uv_stream_t*>(&u->client), alloc_cb, Cbs::on_cli_read);
    uv_buf_t wb = uv_buf_init(const_cast<char*>(kMsg), kMsgLen);  // prime req#1
    uv_write(&u->cwreq, reinterpret_cast<uv_stream_t*>(&u->client), &wb, 1, Cbs::on_cli_write);
    conns.push_back(std::move(u));
  }
  uv_run(&loop, UV_RUN_DEFAULT);
  uv_loop_close(&loop);
}
#endif  // HYLE_BENCH_LIBUV

}  // namespace

int main() {
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
#if defined(__linux__) && defined(HYLE_BENCH_LIBUV)
  // libuv's uv_write to a peer whose read end just closed can raise SIGPIPE (fatal by
  // default) during the connection teardown race at the end of each epoch. Ignore it
  // so the baseline's close ordering doesn't kill the bench (the write just fails,
  // which libuv's completion reports and we ignore). Same guard as tls_bench.
  std::signal(SIGPIPE, SIG_IGN);
#endif
  const std::uint64_t batch =
      static_cast<std::uint64_t>(kConns) * kRoundTrips;  // echoes per epoch

  // ---- shard scaling: HYLE echo on 1/2/4/8 workers ----
  {
    ankerl::nanobench::Bench b;
    b.title("io echo: 64 conns x 64 round-trips")
        .relative(true)
        .batch(batch)
        .unit("echo")
        // One invocation already completes 4096 real round-trips and creates a
        // fresh scheduler/socket set. Repeating it 20x per epoch made the
        // Windows harness take minutes before printing a row; five one-or-more
        // invocation epochs keep the sample bounded while cross-process
        // bench-stat supplies the release-level repetition.
        .minEpochIterations(1)
        .epochs(5);
    for (unsigned w : {1u, 2u, 4u, 8u}) {
      char name[64];
      std::snprintf(name, sizeof(name), "HYLE shard-per-core (%u workers)", w);
      b.run(name, [&] {
        if (run_hyle_echo(w) != kConns * kRoundTrips) std::abort();
      });
    }
#if defined(__linux__)
    b.run("raw epoll (1 thread, readiness loop)", [&] { run_raw_epoll_echo(); });
#endif
#if defined(__linux__) && defined(HYLE_BENCH_LIBURING)
    // The SAME io_uring, hand-rolled with the community-standard wrapper (no fibers,
    // no coroutines). Compared head-to-head with the HYLE 1-worker row above: the
    // ratio is the cost of HYLE's co_await + shard-per-core abstraction over a bare
    // liburing loop. Gated -DHYLE_BENCH_LIBURING (links liburing.a).
    b.run("raw liburing (1 thread, hand-rolled)", [&] { run_raw_liburing_echo(); });
#endif
#if defined(__linux__) && defined(HYLE_BENCH_ASIO)
    // asio over io_uring (single io_context, single thread) -- THE canonical C++
    // async framework on the SAME kernel primitive. Head-to-head with HYLE-1w: does
    // our shard-per-core co_await stack keep up with the framework everyone uses?
    // Gated -DHYLE_BENCH_ASIO (header-only asio + links liburing).
    b.run("asio (1 io_context, io_uring)", [&] { run_asio_echo(); });
#endif
#if defined(HYLE_BENCH_LIBUV)
    // libuv (single loop, single thread, epoll backend) -- the most battle-tested
    // async C runtime (Node.js's engine). Cross-model (its reactor vs HYLE's
    // proactor), so read it against the raw-epoll / HYLE forced-epoll rows, not the
    // io_uring ones. Gated -DHYLE_BENCH_LIBUV (links libuv.a).
    b.run("libuv (1 loop, epoll)", [&] { run_libuv_echo(); });
#endif
  }

  // ---- pipelined / high-fan-out: many streams per core (SQ can fill) ----
  // This is the load that can exercise submit-batching -- hundreds of streams on
  // one core means a reap wakes many fibers that each submit before the next
  // reap. It is specifically an io_uring SQ/GETEVENTS probe; IOCP has no submit
  // queue and kqueue is a readiness backend, so running 512 loopback TCP pairs
  // there measures pair construction rather than the mechanism under test.
#if defined(__linux__)
  {
    const std::uint64_t pbatch =
        static_cast<std::uint64_t>(kPipeConns) * kPipeRoundTrips;
    ankerl::nanobench::Bench b;
    b.title("io pipelined: 512 streams x 16 round-trips")
        .relative(true)
        .batch(pbatch)
        .unit("echo")
        .minEpochIterations(1)
        .epochs(5);
    for (unsigned w : {1u, 2u, 4u, 8u}) {
      char name[64];
      std::snprintf(name, sizeof(name), "HYLE pipelined (%u workers)", w);
      b.run(name, [&] {
        if (run_hyle_pipelined(w) != kPipeConns * kPipeRoundTrips)
          std::abort();
      });
    }
  }
#endif

#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
