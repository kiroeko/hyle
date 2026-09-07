// io SCALING bench (complements echo_bench.cc). echo_bench proves throughput +
// worker scaling at a FIXED 64-conn/64B load; this file proves the two COMPLEXITY
// curves a "SOTA proactor" claim rests on -- what a single-point ratio can't show:
//
//  1. CONNECTION-COUNT SCALING (proactor completion path is O(1) per op). FIXED
//     round-trips PER connection, spread over N = 64/256/1024 concurrent connections
//     on one worker. Reported as ns-per-round-trip. Because R is per-connection (not a
//     fixed total), the batch (N*R) grows with N in lockstep with the O(N) setup cost
//     (build N socketpairs + 2N fibers), so that setup lands as a CONSTANT ns/round-trip
//     offset instead of ballooning as N rises. The proactor-side proof is in the STATS
//     (HYLE_IO_BENCH_STATS): submit batch stays flat (~55 SQE/enter) and sqes grow
//     EXACTLY linearly with N -- the io_uring/IOCP completion model has no per-connection
//     cost (no fd-set rescan the way select/poll does). The measured ns/round-trip still
//     rises modestly with N; that residual is a user-space working-set effect (more live
//     fibers = more stack + Conn cache footprint -- a smaller-stack A/B collapses the
//     64->256 step), NOT proactor algorithmic cost. The point of this row is that
//     separation, documented honestly, not a single "flat == O(1)" claim.
//
//  2. PAYLOAD-SIZE SCALING (no hidden copy / per-byte walk). Fixed 64 conns, message
//     size 64B/1KB/16KB/64KB, reported as BYTES/s: small sizes are syscall-bound, large
//     sizes approach memory bandwidth. The curve must have NO super-linear knee -- a
//     knee would mean an extra memcpy or per-byte pass in the co_await path. HYLE's
//     awaitable hands the kernel the caller's buffer directly (zero-copy into the user
//     buffer), so bytes/s should climb then plateau cleanly.
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict; the SHAPE across
// rows (flat ns/op, clean bytes/s plateau) is the scaling readout, not any one number.

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if !defined(_WIN32)
#  include <sys/resource.h>
#endif
#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
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
namespace nb = ankerl::nanobench;

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
      ::listen(ls, 128) == SOCKET_ERROR) {
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
  linger reset{1, 0};
  for (std::intptr_t fd : {p.a, p.b}) {
    if (fd == -1) continue;
    ::setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_LINGER,
                 reinterpret_cast<const char*>(&reset), sizeof(reset));
    ::closesocket(static_cast<SOCKET>(fd));
  }
}
#endif

// A connection with a runtime-sized payload (unlike echo_bench's fixed kMsgLen), so
// the payload sweep can vary message size. Buffers come from the heap (std::vector).
struct Conn {
  IoContext* io;
  std::intptr_t client_fd;
  std::intptr_t server_fd;
  int round_trips;
  std::uint32_t msg_len;
  std::vector<char> cbuf;
  std::vector<char> sbuf;
  std::atomic<std::uint64_t>* completed;
};

Task<int> client_task(AllocatorRef, Conn* c) {
  for (int i = 0; i < c->round_trips; ++i) {
    std::uint32_t sent = 0;
    while (sent < c->msg_len) {
      std::int64_t w = co_await SendOp{
          &c->io->proactor_for_current(), c->client_fd,
          c->cbuf.data() + sent, c->msg_len - sent};
      if (w <= 0) co_return 0;
      sent += static_cast<std::uint32_t>(w);
    }
    std::uint32_t got = 0;
    while (got < c->msg_len) {  // a large payload may arrive over several reads
      std::int64_t r = co_await RecvOp{&c->io->proactor_for_current(), c->client_fd,
                                       c->cbuf.data() + got, c->msg_len - got};
      if (r <= 0) co_return 0;
      got += static_cast<std::uint32_t>(r);
    }
    c->completed->fetch_add(1, std::memory_order_relaxed);
  }
  co_return 0;
}

Task<int> server_task(AllocatorRef, Conn* c) {
  for (int i = 0; i < c->round_trips; ++i) {
    std::uint32_t got = 0;
    while (got < c->msg_len) {
      std::int64_t r = co_await RecvOp{&c->io->proactor_for_current(), c->server_fd,
                                       c->sbuf.data() + got, c->msg_len - got};
      if (r <= 0) co_return 0;
      got += static_cast<std::uint32_t>(r);
    }
    std::uint32_t sent = 0;
    while (sent < c->msg_len) {
      std::int64_t w = co_await SendOp{&c->io->proactor_for_current(), c->server_fd,
                                       c->sbuf.data() + sent, c->msg_len - sent};
      if (w <= 0) co_return 0;
      sent += static_cast<std::uint32_t>(w);
    }
  }
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void* p) noexcept { static_cast<Drive*>(p)->task.resume(); }

// Run `conns` connections x `round_trips` echoes of `msg_len` bytes on `workers`
// cores. Connections shard round-robin across workers. Returns after all complete.
std::uint64_t run_echo(unsigned workers, int conns, int round_trips,
                       std::uint32_t msg_len) {
  Scheduler sched(workers, system_allocator(), 8192);
  IoContext io(sched);

  std::atomic<std::uint64_t> completed{0};
  std::vector<Pair> pairs(conns);
  std::vector<std::unique_ptr<Conn>> cs;
  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  for (int i = 0; i < conns; ++i) {
    pairs[i] = make_pair();
    unsigned owner = static_cast<unsigned>(i) % workers;
    io.bind_fd(owner, pairs[i].a);
    io.bind_fd(owner, pairs[i].b);
    auto c = std::make_unique<Conn>();
    c->io = &io;
    c->client_fd = pairs[i].a;
    c->server_fd = pairs[i].b;
    c->round_trips = round_trips;
    c->msg_len = msg_len;
    c->cbuf.assign(msg_len, 'x');
    c->sbuf.assign(msg_len, 0);
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
    cs.push_back(std::move(c));
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

}  // namespace

int main() {
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
#if !defined(_WIN32)
  // The conn sweep needs up to 1024 conns = 2048 fds; the default soft RLIMIT_NOFILE
  // (often 1024) would silently make make_pair() fail past ~500 conns, and the failed
  // fds (-1) would run zero round-trips while still counting in the batch -> fabricated
  // numbers. Raise the soft limit toward the hard limit so the high-N rows are real.
  {
    rlimit rl;
    if (::getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < rl.rlim_max) {
      rl.rlim_cur = rl.rlim_max;
      ::setrlimit(RLIMIT_NOFILE, &rl);
    }
  }
#endif

  // ---- 1. connection-count scaling: ns/round-trip vs #conns ----
  // FIXED round-trips PER connection (kRt), swept over N concurrent connections on ONE
  // worker (isolate the proactor cost from work-stealing). Because R is per-connection,
  // batch (N*R) tracks the O(N) build cost, so setup is a CONSTANT ns/round-trip offset
  // rather than a term that grows with N. What this ISOLATES is the proactor completion
  // path: under HYLE_IO_BENCH_STATS the submit batch stays flat (avg ~55) and sqes grow
  // exactly linearly with N -- i.e. the kernel/proactor side is clean O(1) per op, no
  // fd-set rescan (the "proactor beats select/poll" argument). The MEASURED ns/round-trip
  // still rises modestly with N (a user-space working-set effect: more live fibers = more
  // stack + Conn cache footprint, confirmed by a smaller-stack A/B collapsing the 64->256
  // step), NOT a per-op algorithmic cost. See docs/performance/benchmarks-io.md.
  {
    constexpr int kRt = 64;  // round-trips PER connection, held constant across N
    nb::Bench b;
    b.title("io conn scaling: ns/round-trip vs #conns (proactor side O(1)/op)")
        .unit("round-trip")
        .minEpochIterations(1)
        .epochs(5);
    for (int conns : {64, 256, 1024}) {
      char name[64];
      std::snprintf(name, sizeof(name), "conns=%d (rt/conn=%d)", conns, kRt);
      b.batch(static_cast<std::uint64_t>(conns) * kRt).run(name, [&] {
        if (run_echo(1, conns, kRt, 64) !=
            static_cast<std::uint64_t>(conns) * kRt)
          std::abort();
      });
    }
  }

  // ---- 2. payload-size scaling: bytes/s vs message size (clean plateau) ----
  // Fixed 64 conns x 32 round-trips, message size sweeps 64B..64KB. Reported as
  // bytes/s: should rise (syscall amortized over more bytes) then plateau at memory
  // bandwidth, with NO super-linear knee (a knee = an extra copy / per-byte pass in the
  // co_await path). Fixed conn count => the socketpair build cost is a constant across
  // the sweep (on Linux it is cheap next to 2048 round-trips of io). The sweep stops at
  // 64KB: by 16KB ns/byte is already at the memory-bandwidth floor (see the doc), and a
  // 256KB row moves 512MB/iteration -- minutes of runtime for no extra shape.
  {
    constexpr int kConns = 64;
    constexpr int kRt = 32;
    nb::Bench b;
    b.title("io payload scaling: bytes/s vs msg size (clean plateau => no extra copy)")
        .unit("byte");
    for (std::uint32_t sz : {64u, 1024u, 16384u, 65536u}) {
      char name[64];
      std::snprintf(name, sizeof(name), "payload=%uB", sz);
      // batch = total bytes moved (conns * rt * size) so nanobench reports bytes/s.
      b.batch(static_cast<std::uint64_t>(kConns) * kRt * sz)
          .minEpochIterations(1)
          .epochs(5)
          .run(name, [&] {
            if (run_echo(1, kConns, kRt, sz) !=
                static_cast<std::uint64_t>(kConns) * kRt)
              std::abort();
          });
    }
  }

#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}
