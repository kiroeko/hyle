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
#include <cstring>
#include <memory>
#include <vector>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(__linux__)
#  include <sys/epoll.h>
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

constexpr int kMsgLen = 64;      // echo payload size
constexpr int kRoundTrips = 64;  // round-trips per connection per epoch
constexpr int kConns = 64;       // concurrent connections

// Pipelined workload knobs: kPipeConns independent echo streams sharing a worker,
// each doing kPipeRoundTrips round-trips. Because MANY streams live on one core,
// a single reap wakes MANY fibers that each immediately submit -- so N SQEs pile
// up before the next reap. THIS is the many-in-flight load that fills the SQ and
// lets submit-batching matter (the plain 64-conn ping-pong above keeps ~1 op in
// flight per stream at a time, so the SQ never fills -- batching can't help it).
constexpr int kPipeConns = 300;       // concurrent streams (high fan-out, > ring depth 256 -> exercises CQ-overflow recovery)
constexpr int kPipeRoundTrips = 16;   // round-trips per stream per epoch

// A connected fd pair.
struct Pair {
  std::intptr_t a = -1;
  std::intptr_t b = -1;
};

#if defined(__linux__)
Pair make_pair() {
  int sv[2];
  Pair p;
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0) {
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
  ::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  ::listen(ls, 1);
  int slen = sizeof(sa);
  ::getsockname(ls, reinterpret_cast<sockaddr*>(&sa), &slen);
  SOCKET c = ::WSASocketW(AF_INET, SOCK_STREAM, 0, nullptr, 0, WSA_FLAG_OVERLAPPED);
  ::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  SOCKET s = ::accept(ls, nullptr, nullptr);
  ::closesocket(ls);
  p.a = static_cast<std::intptr_t>(c);
  p.b = static_cast<std::intptr_t>(s);
  return p;
}
void close_pair(const Pair& p) {
  if (p.a != -1) ::closesocket(static_cast<SOCKET>(p.a));
  if (p.b != -1) ::closesocket(static_cast<SOCKET>(p.b));
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
  char cbuf[kMsgLen];
  char sbuf[kMsgLen];
};

Task<int> client_task(Allocator, Conn* c) {
  static const char kMsg[kMsgLen] = "hyle-io-echo-bench-payload";
  for (int i = 0; i < c->round_trips; ++i) {
    std::int64_t w = co_await SendOp{&c->io->proactor_for_current(), c->client_fd,
                                     kMsg, kMsgLen};
    if (w != kMsgLen) break;
    std::int64_t r = co_await RecvOp{&c->io->proactor_for_current(), c->client_fd,
                                     c->cbuf, kMsgLen};
    if (r != kMsgLen) break;
  }
  co_return 0;
}

Task<int> server_task(Allocator, Conn* c) {
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
void run_hyle_echo(unsigned workers) {
  Scheduler sched(workers, system_allocator(), 4096);
  IoContext io(sched, workers);

  std::vector<Pair> pairs(kConns);
  std::vector<std::unique_ptr<Conn>> conns;
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
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
    auto dc = std::make_unique<Drive>();
    auto ds = std::make_unique<Drive>();
    dc->task = client_task(system_allocator(), c.get());
    ds->task = server_task(system_allocator(), c.get());
    auto fc = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                      &drive_entry, dc.get());
    auto fs = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                      &drive_entry, ds.get());
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
}

// Pipelined / high-fan-out variant: kPipeConns streams x kPipeRoundTrips on
// `workers` cores. With hundreds of streams per core, one reap wakes dozens of
// fibers that each submit -- so many SQEs accumulate between reaps (the SQ can
// fill). This is the load where submit-batching (one io_uring_enter for many
// SQEs) can pay off; the plain ping-pong bench cannot exercise it.
void run_hyle_pipelined(unsigned workers) {
  Scheduler sched(workers, system_allocator(), 8192);
  IoContext io(sched, workers);

  std::vector<Pair> pairs(kPipeConns);
  std::vector<std::unique_ptr<Conn>> conns;
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
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
    auto dc = std::make_unique<Drive>();
    auto ds = std::make_unique<Drive>();
    dc->task = client_task(system_allocator(), c.get());
    ds->task = server_task(system_allocator(), c.get());
    auto fc = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                      &drive_entry, dc.get());
    auto fs = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                      &drive_entry, ds.get());
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
}

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

}  // namespace

int main() {
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
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
        .minEpochIterations(20);
    for (unsigned w : {1u, 2u, 4u, 8u}) {
      char name[64];
      std::snprintf(name, sizeof(name), "HYLE shard-per-core (%u workers)", w);
      b.run(name, [&] { run_hyle_echo(w); });
    }
#if defined(__linux__)
    b.run("raw epoll (1 thread, readiness loop)", [&] { run_raw_epoll_echo(); });
#endif
  }

  // ---- pipelined / high-fan-out: many streams per core (SQ can fill) ----
  // This is the load that can exercise submit-batching -- hundreds of streams on
  // one core means a reap wakes many fibers that each submit before the next
  // reap. Reported separately from the ping-pong bench above.
  {
    const std::uint64_t pbatch =
        static_cast<std::uint64_t>(kPipeConns) * kPipeRoundTrips;
    ankerl::nanobench::Bench b;
    b.title("io pipelined: 512 streams x 16 round-trips")
        .relative(true)
        .batch(pbatch)
        .unit("echo")
        .minEpochIterations(10);
    for (unsigned w : {1u, 2u, 4u, 8u}) {
      char name[64];
      std::snprintf(name, sizeof(name), "HYLE pipelined (%u workers)", w);
      b.run(name, [&] { run_hyle_pipelined(w); });
    }
  }

#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
