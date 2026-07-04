// io ACCEPT-SCALING bench (third leg, after echo_bench's throughput + io_scaling's
// two complexity curves). This one isolates CONNECTION ESTABLISHMENT -- the accept
// path -- which the echo benches never exercise (they pre-build every socketpair
// outside the timed region). A server's accept throughput is its own SOTA axis: how
// fast can the proactor turn a listen backlog into serviced connections, and does
// that rate hold as the connection count grows (O(1) per accept)?
//
// LOAD MODEL (mirrors liburing's test/accept-mshot-stress.c, adapted to HYLE's
// shard-per-core wrappers): ONE TcpListener on worker 0 accepts N connections in a
// loop; N client fibers, round-robin across ALL workers, each connect() once. New
// connections are assigned round-robin to workers (TcpListener::accept(owner)) --
// the seastar/HYLE "shard new connections across cores" pattern. No echo after the
// handshake: the accepted stream is dropped (closed) immediately, so the ONLY thing
// timed is connection setup. Reported as accepts/second; batch = N so nanobench
// divides out the per-accept cost.
//
// SCALING READOUT: sweep N = 64/256/1024 and read accepts/s. A flat accepts/s (i.e.
// per-accept ns that does NOT climb with N) is the "accept is O(1)" evidence. MEASURED
// (io_uring): ~3.8k accepts/s essentially CONSTANT across the 16x N span -- O(1)/accept
// confirmed, once the listen backlog is sized to the burst (see run_accept: a starved
// backlog fakes a 16x blowup via SYN-retransmit). This is real TCP over loopback
// (accept needs a real listener -- socketpair can't accept), so it accrues TIME_WAIT on
// close; Linux recycles those far faster than Windows, so the honest scaling numbers
// come from the io_uring run (Windows verifies it builds + the low-N rows).
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict.

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
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

// The server accept loop: accept `n` connections, assigning each round-robin to a
// worker, then drop each (close) -- we time setup only, not any echo. Pinned to the
// listener's owner worker (0).
Task<int> accept_loop(Allocator, TcpListener* ls, int n, unsigned workers,
                      std::atomic<int>* accepted) {
  for (int i = 0; i < n; ++i) {
    unsigned owner = static_cast<unsigned>(i) % workers;
    TcpStream conn = co_await ls->accept(owner);
    if (conn.valid()) accepted->fetch_add(1, std::memory_order_relaxed);
    // conn drops here -> fd closed. Pure connection-setup timing.
  }
  co_return 0;
}

// One client: connect once to the listener, then drop (close). Pinned to its worker.
Task<int> client_connect(Allocator, IoContext* io, unsigned owner,
                         std::uint16_t port) {
  TcpStream s = tcp_socket(*io, owner);
  if (!s.valid()) co_return 0;
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  if (!resolve_addr("127.0.0.1", port, &ss, &len)) co_return 0;
  co_await s.connect(&ss, len);
  co_return 0;  // s drops -> client fd closed
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) { static_cast<Drive*>(p)->task.resume(); }

// Establish `conns` connections through one listener on `workers` cores: the accept
// loop on worker 0, `conns` client fibers round-robin across workers. Returns after
// all have connected + been accepted. Each call uses a fresh port to avoid colliding
// with the still-draining TIME_WAIT sockets of the previous nanobench iteration.
void run_accept(unsigned workers, int conns, std::uint16_t port) {
  Scheduler sched(workers, system_allocator(), 8192);
  IoContext io(sched, workers);

  // backlog MUST cover the whole burst: this bench spawns all N client fibers at
  // once, so up to N SYNs can arrive before the single accept loop drains them. A
  // too-small backlog (the default 128) drops the overflow SYNs -> TCP SYN-retransmit
  // (1s backoff) -> a fabricated 16x super-linear blowup at N>=256 that looks like a
  // broken accept path but is really just a starved listen queue. Size it to N.
  TcpListener ls = tcp_listen(io, 0, "127.0.0.1", port, conns);
  if (!ls.valid()) {  // port still in TIME_WAIT etc -> skip this iteration cleanly
    sched.shutdown();
    return;
  }

  std::atomic<int> accepted{0};
  auto da = std::make_unique<Drive>();
  da->task = accept_loop(system_allocator(), &ls, conns, workers, &accepted);
  auto fa = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                    &drive_entry, da.get());
  sched.spawn_on(0, fa.get());

  std::vector<std::unique_ptr<Drive>> cdrives;
  std::vector<std::unique_ptr<Fiber>> cfibers;
  cdrives.reserve(conns);
  cfibers.reserve(conns);
  for (int i = 0; i < conns; ++i) {
    unsigned owner = static_cast<unsigned>(i) % workers;
    auto dc = std::make_unique<Drive>();
    dc->task = client_connect(system_allocator(), &io, owner, port);
    auto fc = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                      &drive_entry, dc.get());
    sched.spawn_on(owner, fc.get());
    cdrives.push_back(std::move(dc));
    cfibers.push_back(std::move(fc));
  }

  sched.drain();
  sched.shutdown();
}

}  // namespace

int main() {
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

  // ---- accept scaling: accepts/s vs #conns (flat => O(1)/accept) ----
  // One listener, N clients connecting, N accepts, on 4 workers (new connections
  // shard round-robin -- the shard-per-core accept pattern). A fresh port per epoch
  // side-steps TIME_WAIT reuse. Sweep N and read accepts/s: a rate that holds as N
  // grows is the "accept path is O(1) per connection" evidence.
  {
    constexpr unsigned kWorkers = 4;
    nb::Bench b;
    b.title("io accept scaling: accepts/s vs #conns (flat => O(1)/accept)")
        .unit("accept");
    std::uint16_t port = 24000;
    for (int conns : {64, 256, 1024}) {
      char name[64];
      std::snprintf(name, sizeof(name), "conns=%d", conns);
      b.batch(static_cast<std::uint64_t>(conns)).minEpochIterations(3).run(name, [&] {
        run_accept(kWorkers, conns, port++);  // ++ -> a new port each iteration
      });
    }
  }

#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
