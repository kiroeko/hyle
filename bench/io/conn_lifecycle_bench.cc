// io CONNECTION-LIFECYCLE bench: the REALISTIC full-connection cost that the other
// io benches deliberately factor OUT. echo_bench pre-builds socketpairs (never accepts
// or closes -- pure steady-state transport); accept_bench accepts then DROPS instantly
// (pure setup, no service). Neither is how a real server spends a connection. A real
// short-lived connection is: accept -> serve R request/response round-trips -> close.
// That is HTTP/1.0, health checks, RPC-without-pooling, and any connection-churn
// workload -- the regime where the ACCEPT + TEARDOWN path (not steady-state echo) is a
// real fraction of the cost.
//
// LOAD MODEL (the shard-per-core server pattern -- IOCP-safe): ONE TcpListener on
// worker 0 accepts; for each accepted connection the accept loop SPAWNS a handler fiber
// on the connection's assigned core (round-robin), then immediately loops to accept the
// next. The handler serves R echo round-trips of a small message, then its TcpStream
// (scoped in an inner block) drops -> fd closes at block exit. Service thus runs in
// PARALLEL across all cores (accept stays on the listener's core 0, as IOCP requires;
// the accepted fd is bound to its owner core by AcceptAwaitable, so the handler's
// recv/send are valid on that core's ring/port). N client fibers (round-robin across
// workers) each connect, do R round-trips, close. Real loopback TCP (real 3-way
// handshake + real close/TIME_WAIT): the WHOLE lifecycle, not one phase.
//
// SWEEP R (round-trips per connection) = 1 / 4 / 16 -> walks the ephemeral->persistent
// spectrum. R=1 is a one-shot request (setup+teardown DOMINATES -- the accept regime);
// R=16 amortizes setup over more service (approaches steady state). Reading conns/s
// across R separates per-connection fixed cost (accept+handshake+close) from per-request
// cost. Batch = N so nanobench divides out per-conn.
//
// Real TCP over loopback accrues TIME_WAIT on close; Linux recycles far faster than
// Windows, so the honest lifecycle numbers come from the io_uring run (Windows verifies
// it builds + the low-N/low-R rows). Run under scripts/bench-stat.py -n 15.

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#else
#  include <sys/resource.h>
#endif

using hyle::io::IoContext;
using hyle::io::resolve_addr;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::tcp_listen;
using hyle::io::tcp_socket;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {
namespace nb = ankerl::nanobench;

constexpr int kMsgLen = 64;  // request/response payload size

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// Per-connection handler: serve `rounds` echo round-trips, then close. The TcpStream is
// SCOPED in an inner block so its destructor (fd close) runs at block exit -- BEFORE
// co_return -- giving a DETERMINISTIC per-connection close. (A handler frame lingers
// after co_return because final_suspend is suspend_always; but the fd is already closed
// by then.) `conn` is the moved-in stream; moving it to the block-local `s` leaves the
// parameter inert (fd_ = -1) so its later frame-teardown destruction closes nothing.
Task<int> serve_conn(Allocator, TcpStream conn, int rounds,
                     std::atomic<int>* served) {
  bool complete = true;
  {
    TcpStream s = std::move(conn);
    char buf[kMsgLen];
    for (int r = 0; r < rounds; ++r) {
      std::int64_t got = co_await s.read(buf, kMsgLen);
      if (got <= 0) {
        complete = false;
        break;
      }
      std::int64_t put = co_await s.write(buf, static_cast<std::uint32_t>(got));
      if (put != got) {
        complete = false;
        break;
      }
    }
    // s destroyed here (block exit) -> connection fd closed deterministically.
  }
  if (complete) served->fetch_add(1, std::memory_order_relaxed);
  co_return 0;
}

// The accept loop (on the listener's core 0): accept N connections, and for EACH spawn a
// handler fiber on its assigned core (round-robin), then immediately accept the next --
// so service parallelizes across cores while accepts stay on core 0 (IOCP requirement).
// Handler Drives/Fibers are stored in caller-owned vectors (reserved to N) that outlive
// drain(); only this loop (core 0) mutates them, so no race.
Task<int> accept_dispatch_loop(Allocator, Scheduler* sched, TcpListener* ls, int n,
                               int rounds, unsigned workers,
                               std::atomic<int>* served,
                               std::vector<std::unique_ptr<Drive>>* hdrives,
                               std::vector<std::unique_ptr<Fiber>>* hfibers) {
  for (int i = 0; i < n; ++i) {
    unsigned owner = static_cast<unsigned>(i) % workers;
    TcpStream conn = co_await ls->accept(owner);
    if (!conn.valid()) continue;
    auto dh = std::make_unique<Drive>();
    dh->task = serve_conn(system_allocator(), std::move(conn), rounds, served);
    auto fh =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, dh.get());
    sched->spawn_on(owner, fh.get());  // hand off -> parallel service on `owner`
    hdrives->push_back(std::move(dh));
    hfibers->push_back(std::move(fh));
  }
  co_return 0;
}

// One client: connect, do `rounds` request/response round-trips, then drop (close).
Task<int> client_lifecycle(Allocator, IoContext* io, unsigned owner,
                           std::uint16_t port, int rounds) {
  TcpStream s = tcp_socket(*io, owner);
  if (!s.valid()) co_return 0;
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  if (!resolve_addr("127.0.0.1", port, &ss, &len)) co_return 0;
  if (co_await s.connect(&ss, len) != 0) co_return 0;
  char buf[kMsgLen];
  std::memset(buf, 'x', sizeof(buf));
  for (int r = 0; r < rounds; ++r) {
    std::int64_t put = co_await s.write(buf, kMsgLen);
    if (put != kMsgLen) break;
    std::int64_t got = co_await s.read(buf, kMsgLen);
    if (got != kMsgLen) break;
  }
  co_return 0;  // s drops -> client fd closed
}

// Run N full connection lifecycles (accept + serve R rounds + close) on `workers` cores:
// the accept-dispatch loop on worker 0 hands each connection to a per-conn handler fiber
// on its core; N client fibers round-robin across workers. Fresh port per call dodges
// the previous epoch's TIME_WAIT sockets.
int run_lifecycle(unsigned workers, int conns, int rounds, std::uint16_t port) {
  Scheduler sched(workers, system_allocator(), 8192);
  IoContext io(sched, workers);

  // backlog MUST cover the whole burst (all N client fibers spawn at once): a starved
  // backlog drops overflow SYNs -> TCP SYN-retransmit (1s backoff) -> a fabricated
  // super-linear blowup that looks like a broken accept path (see accept_bench).
  TcpListener ls = tcp_listen(io, 0, "127.0.0.1", port, conns);
  if (!ls.valid()) {  // port still in TIME_WAIT etc -> skip this iteration cleanly
    sched.shutdown();
    return -1;
  }

  std::atomic<int> served{0};
  std::vector<std::unique_ptr<Drive>> hdrives;
  std::vector<std::unique_ptr<Fiber>> hfibers;
  hdrives.reserve(static_cast<std::size_t>(conns));  // no realloc during the loop
  hfibers.reserve(static_cast<std::size_t>(conns));

  auto da = std::make_unique<Drive>();
  da->task = accept_dispatch_loop(system_allocator(), &sched, &ls, conns, rounds,
                                  workers, &served, &hdrives, &hfibers);
  auto fa = std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, da.get());
  sched.spawn_on(0, fa.get());

  std::vector<std::unique_ptr<Drive>> cdrives;
  std::vector<std::unique_ptr<Fiber>> cfibers;
  cdrives.reserve(static_cast<std::size_t>(conns));
  cfibers.reserve(static_cast<std::size_t>(conns));
  for (int i = 0; i < conns; ++i) {
    unsigned owner = static_cast<unsigned>(i) % workers;
    auto dc = std::make_unique<Drive>();
    dc->task = client_lifecycle(system_allocator(), &io, owner, port, rounds);
    auto fc =
        std::make_unique<Fiber>(kDefaultStackSize, &drive_entry, dc.get());
    sched.spawn_on(owner, fc.get());
    cdrives.push_back(std::move(dc));
    cfibers.push_back(std::move(fc));
  }

  sched.drain();  // returns when accept loop + all handlers + all clients are done
  sched.shutdown();
  return served.load(std::memory_order_relaxed);
}

}  // namespace

int main() {
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#else
  rlimit rl;
  if (::getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < rl.rlim_max) {
    rl.rlim_cur = rl.rlim_max;
    ::setrlimit(RLIMIT_NOFILE, &rl);
  }
#endif

  // ---- connection lifecycle: conns/s vs round-trips-per-connection ----
  // Fixed N connections on 4 workers (accept on core 0, service sharded across cores),
  // sweeping R = rounds-per-connection. R=1 => setup/teardown dominates (accept regime);
  // R=16 => amortized toward steady state. A fresh port per epoch dodges TIME_WAIT.
  {
    constexpr unsigned kWorkers = 4;
    constexpr int kConns = 256;
    nb::Bench b;
    b.title("io connection lifecycle: conns/s vs round-trips/conn (accept+serve+close)")
        .relative(true)
        .unit("conn");
    std::uint16_t port = 25000;
    for (int rounds : {1, 4, 16}) {
      char name[64];
      std::snprintf(name, sizeof(name), "conns=%d rounds=%d", kConns, rounds);
      b.batch(static_cast<std::uint64_t>(kConns))
          .minEpochIterations(1)
          .epochs(5)
          .run(name, [&] {
            if (run_lifecycle(kWorkers, kConns, rounds, port++) != kConns)
              std::abort();
          });
    }
  }

#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
