// io-07b multishot ZERO-COPY ECHO/proxy perf: the workload multishot is actually FOR.
// The recv-drain bench (multishot_bench.cc) showed a TIE because lockstep keeps ~1 op
// in flight -- multishot can't amortize its saved submit. This bench is the consumer
// whose data SHAPE matches multishot's model (per-chunk, no contiguity needed): an
// ECHO server that, for each received chunk, sends it straight back FROM THE RING
// BUFFER (zero-copy: no copy into a caller buffer) then returns the buffer -- exactly
// liburing proxy.c's recv-multishot -> send -> replenish pattern.
//
// Two echo servers, SAME load (a flooding client sends kMsgs datagrams; server echoes
// each), SAME same-worker fibers + DGRAM (no cross-thread floor, equal completion
// counts). The ONLY difference:
//   1. multishot zero-copy: ONE recv arm; per chunk, submit_send FROM buf_ring_addr
//      (kernel-picked buffer, no copy), await send, then RecvBuf dtor returns it.
//   2. one-shot copy: recv into a stack buffer (a submit per datagram), send from it.
// Reported as ns/echo. If multishot wins HERE, it proves the awaitable earns its keep
// on the workload it was built for (vs the recv-drain tie).
//
// Run under scripts/bench-stat.py -n 15.
//
// MEASURED (Linux io_uring): ANOTHER TIE -- multishot 349,226ns vs one-shot 350,758ns
// per echo (0.4%, noise). Both echo exactly 2000 (equal work). This is the THIRD
// multishot bench (after recv-drain's tie) to tie, and it pins down WHY: this echo is
// still LOCKSTEP (client waits for each echo before sending the next), so ~1 op is in
// flight and the ~349us/echo is DOMINATED by the fiber round-trip (client send ->
// server recv -> server send -> client recv, each a suspend/resume). The zero-copy
// send-from-ring genuinely saves a 64-byte copy, but that copy is nothing next to the
// us-scale fiber scheduling. HONEST CONCLUSION: on a SINGLE worker with lockstep
// round-trips, NO io-mechanic optimization (multishot, zero-copy, fixed-file) shows --
// fiber scheduling dominates. Multishot's win needs genuine concurrency/burst (data
// arriving faster than one consumer drains), which a single-worker lockstep bench
// structurally cannot produce. So multishot is proven CORRECT + NOT-A-REGRESSION here;
// showing its throughput edge needs a multi-consumer or bursty harness (future work).

#include <nanobench.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/io/io_context.h"
#include "hyle/io/net.h"
#include "hyle/io/net_multishot.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if defined(__linux__)
#  include <sys/socket.h>
#  include <unistd.h>
#endif

using hyle::io::IoContext;
using hyle::io::MultishotRecvStream;
using hyle::io::RecvBuf;
using hyle::io::RecvOp;
using hyle::io::SendOp;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr int kMsgLen = 64;
constexpr int kMsgs = 2000;   // datagrams echoed per epoch
constexpr int kBgid = 30;

#if defined(__linux__)
struct Pair { std::intptr_t a = -1, b = -1; };
Pair make_pair() {
  int sv[2]; Pair p;
  if (::socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0) { p.a = sv[0]; p.b = sv[1]; }
  return p;
}
void close_pair(Pair& p) {
  if (p.a >= 0) ::close(static_cast<int>(p.a));
  if (p.b >= 0) ::close(static_cast<int>(p.b));
}

struct Drive { Task<int> task; };
void drive_entry(void* v) { static_cast<Drive*>(v)->task.resume(); }

struct Shared {
  IoContext* io;
  std::intptr_t cli_fd;   // client end: sends requests, reads echoes (pair.a)
  std::intptr_t srv_fd;   // server end: echoes (pair.b)
  int echoed = 0;         // echoes the server sent (correctness)
};

// Client: send a datagram, read its echo, kMsgs times (drives the round-trips). Same
// for both rows -- it just feeds the server.
Task<int> client(Allocator, Shared* s) {
  static const char msg[kMsgLen] = "hyle-multishot-echo-bench-payload";
  char buf[kMsgLen];
  for (int i = 0; i < kMsgs; ++i) {
    if (co_await SendOp{&s->io->proactor_for_current(), s->cli_fd, msg, kMsgLen, 0}
        != kMsgLen) break;
    if (co_await RecvOp{&s->io->proactor_for_current(), s->cli_fd, buf, kMsgLen, 0}
        <= 0) break;
  }
  co_return 0;
}

// Server A: multishot zero-copy echo. One recv arm; for each chunk, send it back
// STRAIGHT FROM THE RING BUFFER (b.data() points into the provided-buffer ring -- no
// copy), await the send, THEN let b go out of scope (its dtor buf_ring_returns the
// buffer -- correct ordering: the send has completed, so the kernel is done reading
// the buffer before we hand it back).
Task<int> srv_multishot(Allocator alloc, Shared* s) {
  auto stream = MultishotRecvStream{&s->io->proactor_for_current(), s->srv_fd,
                                    kBgid, alloc};
  while (s->echoed < kMsgs) {
    RecvBuf b = co_await stream.next();
    if (b.size() == 0 || !b.more()) { if (!b.more()) break; else continue; }
    // Zero-copy echo: send directly from the ring buffer, await completion, then the
    // RecvBuf dtor (end of this scope) returns the buffer.
    co_await SendOp{&s->io->proactor_for_current(), s->srv_fd, b.data(), b.size(), 0};
    ++s->echoed;
  }
  co_return 0;
}

// Server B: one-shot copy echo. recv into a stack buffer (a submit per datagram), send
// from that buffer. The extra vs A: a submit per recv + the recv lands in a caller
// buffer (a copy at the kernel boundary that multishot's ring-select avoids).
Task<int> srv_oneshot(Allocator, Shared* s) {
  char buf[kMsgLen];
  while (s->echoed < kMsgs) {
    std::int64_t n = co_await RecvOp{&s->io->proactor_for_current(), s->srv_fd,
                                     buf, kMsgLen, 0};
    if (n <= 0) break;
    co_await SendOp{&s->io->proactor_for_current(), s->srv_fd, buf,
                    static_cast<std::uint32_t>(n), 0};
    ++s->echoed;
  }
  co_return 0;
}

int run(bool use_multishot) {
  Scheduler sched(1, system_allocator(), 4096);
  IoContext io(sched, 1);
  Pair p = make_pair();
  io.bind_fd(0, p.a);
  io.bind_fd(0, p.b);
  if (use_multishot) {
    if (!io.proactor(0).register_buf_ring(kBgid, 1024, kMsgLen)) {
      close_pair(p); sched.shutdown(); return -1;  // non-uring: skip
    }
  }
  Shared sh{}; sh.io = &io; sh.cli_fd = p.a; sh.srv_fd = p.b;
  auto dsrv = std::make_unique<Drive>();
  auto dcli = std::make_unique<Drive>();
  dsrv->task = use_multishot ? srv_multishot(system_allocator(), &sh)
                             : srv_oneshot(system_allocator(), &sh);
  dcli->task = client(system_allocator(), &sh);
  Fiber fsrv(kDefaultStackSize, &drive_entry, dsrv.get());
  Fiber fcli(kDefaultStackSize, &drive_entry, dcli.get());
  sched.spawn_on(0, &fsrv);  // server armed first
  sched.spawn_on(0, &fcli);
  sched.drain();
  sched.shutdown();
  close_pair(p);
  return sh.echoed;
}

}  // namespace

int main() {
  int ms = run(true), os = run(false);
  std::fprintf(stderr, "[smoke] multishot echoed=%d  one-shot echoed=%d (expect %d; -1=skip)\n",
               ms, os, kMsgs);
  if (ms == -1) {
    std::fprintf(stderr,
                 "[skip] active backend lacks provided-buffer multishot recv\n");
    return 0;
  }
  if (ms != kMsgs || os != kMsgs) {
    std::fprintf(stderr, "[fail] smoke completed unequal work\n");
    return 1;
  }
  const std::uint64_t batch = static_cast<std::uint64_t>(kMsgs);
  ankerl::nanobench::Bench b;
  b.title("io zero-copy echo: 2000 datagrams x 64B (multishot ring-send vs one-shot copy)")
      .relative(true)
      .batch(batch)
      .unit("echo")
      .minEpochIterations(1)
      .epochs(5);
  b.run("HYLE multishot zero-copy (send from ring buffer)",
        [&] { if (run(true) != kMsgs) std::abort(); });
  b.run("HYLE one-shot copy (recv into buffer, send from it)",
        [&] { if (run(false) != kMsgs) std::abort(); });
  return 0;
}
#else
}  // namespace

int main() {
  std::fprintf(stderr, "multishot echo bench: Linux/io_uring only -- skipped\n");
  return 0;
}
#endif
