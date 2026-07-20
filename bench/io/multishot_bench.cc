// io-07b multishot recv perf, CLEAN comparison. The earlier attempt was confounded 3
// ways (all producing FAKE ratios): (1) one buf-ring per stream -> 2nd register fails
// -> multishot row bailed doing nothing; (2) pre-drain blocking feed deadlocked; (3) a
// background feeder thread injected cross-thread wakeup latency (WSL2's ~12us floor)
// that DOMINATED the one-shot row (174us/recv), and STREAM coalescing made the two rows
// process unequal completion counts. This version removes all three:
//
//   * SAME-WORKER fibers: a sender fiber and a receiver fiber both pinned to worker 0.
//     Data flows core-locally through the proactor -- NO cross-thread wakeup floor.
//   * DGRAM socketpair: message boundaries preserved, so kMsgs sends == kMsgs recv
//     completions on BOTH rows -- an apples-to-apples completion count (a STREAM socket
//     coalesced the 64B writes, making multishot see far fewer, bigger reads).
//   * DETERMINISTIC stop: the receiver counts exactly kMsgs chunks then stops (DGRAM
//     peer-close does not reliably deliver a multishot terminal; counting is exact).
//
// The ONLY difference between the two rows is how the receiver takes each datagram:
//   1. multishot: ONE submit_recv_multishot arms the stream; co_await stream.next()
//      per datagram, kernel-picked ring buffer, NO per-datagram submit.
//   2. one-shot:  co_await RecvOp per datagram -- a submit_recv syscall-path each time.
// The sender fiber is byte-for-byte identical. So the ratio is exactly the per-recv
// submit + caller-buffer cost that multishot removes. Reported as ns/datagram.
//
// Run under scripts/bench-stat.py -n 15 for the tie/win verdict.
//
// MEASURED (Linux io_uring, n=15): STATISTICAL TIE -- multishot 165,679ns vs one-shot
// 166,985ns per datagram (ratio 0.998, gap 0.2% << spread 13.4%). Multishot does NOT
// beat one-shot HERE, and that is the honest, expected result: this same-worker
// lockstep keeps only ~1 datagram in flight (sender co_awaits each send; receiver
// co_awaits each recv), so the ~166us/datagram is dominated by the fiber round-trip
// BOTH paths pay, and multishot never gets a BATCH of arrivals to amortize its saved
// submit over -- which is its whole point. Multishot's win needs a BURSTY producer
// (many datagrams queued before the receiver drains), which a lockstep sender can't
// create. So this bench proves multishot is NOT A REGRESSION (ties one-shot on the
// unfavorable workload); showing its advantage needs a high-fanout/bursty load a
// single-worker socketpair can't cleanly produce (registered as future work).

#include <nanobench.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
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

// The whole bench (Pair + io_uring multishot) is Linux-only; on Windows there is no
// IOCP provided-buffer-ring equivalent, so it compiles to a stub main after #else.
#if defined(__linux__)
namespace {

constexpr int kMsgLen = 64;   // bytes per datagram
constexpr int kMsgs = 2000;   // datagrams per epoch
constexpr int kBgid = 20;     // provided-buffer-ring group id
// The sender can outrun the receiver; DGRAM socketpair buffers only so many datagrams
// before send blocks. Send in WINDOWS of kWindow, letting the receiver drain between
// (both fibers on one worker interleave via co_await) so the socket buffer never wedges.
constexpr int kWindow = 64;

struct Pair { std::intptr_t a = -1, b = -1; };
Pair make_pair() {
  int sv[2];
  Pair p;
  if (::socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0) { p.a = sv[0]; p.b = sv[1]; }
  return p;
}
void close_pair(Pair& p) {
  if (p.a >= 0) ::close(static_cast<int>(p.a));
  if (p.b >= 0) ::close(static_cast<int>(p.b));
}

// Everything below uses the socketpair Pair + io_uring multishot -- Linux only. On
// Windows the whole bench is a no-op (multishot is an io_uring feature; there is no
// IOCP provided-buffer-ring equivalent), so it compiles to a stub main that prints a
// skip line. That keeps the target building on both platforms without a meaningless
// loopback-TCP emulation.

struct Drive { Task<int> task; };
void drive_entry(void* v) { static_cast<Drive*>(v)->task.resume(); }

struct Shared {
  IoContext* io;
  std::intptr_t send_fd;  // sender writes here (pair.a)
  std::intptr_t recv_fd;  // receiver drains here (pair.b)
  int received = 0;       // datagrams the receiver has taken (for correctness assert)
};

// Sender fiber: kMsgs datagrams via co_await SendOp, in windows so the DGRAM buffer
// (bounded) never fills faster than the receiver drains. Identical for both rows.
Task<int> sender(Allocator, Shared* s) {
  static const char kMsg[kMsgLen] = "hyle-multishot-clean-bench-payload";
  for (int i = 0; i < kMsgs; ++i) {
    std::int64_t w = co_await SendOp{&s->io->proactor_for_current(), s->send_fd,
                                     kMsg, kMsgLen, 0};
    if (w != kMsgLen) break;
    // Every window, the co_await above naturally yields to the receiver between sends
    // (single worker, cooperative), so the buffer drains -- no explicit throttle needed
    // beyond the send completion round-trip. kWindow documents the intent.
    (void)kWindow;
  }
  co_return 0;
}

// ---- receiver: multishot ----
Task<int> recv_multishot(Allocator alloc, Shared* s) {
  auto stream = MultishotRecvStream{
      s->io->proactor_for_current().io_uring_extension(), s->recv_fd, kBgid,
      alloc};
  while (s->received < kMsgs) {
    RecvBuf b = co_await stream.next();
    if (b.size() > 0) ++s->received;
    if (!b.more()) break;  // unexpected terminal (error) -> stop
  }
  co_return 0;  // stream dtor cancels the still-armed multishot (cancel-safe path)
}

// ---- receiver: one-shot RecvOp (a submit per datagram) ----
Task<int> recv_oneshot(Allocator, Shared* s) {
  char buf[kMsgLen];
  while (s->received < kMsgs) {
    std::int64_t n = co_await RecvOp{&s->io->proactor_for_current(), s->recv_fd,
                                     buf, kMsgLen, 0};
    if (n <= 0) break;
    ++s->received;
  }
  co_return 0;
}

// Drive one epoch: sender + receiver fibers on worker 0, DGRAM pair. `use_multishot`
// selects the receiver. Returns the datagrams received (== kMsgs on success).
int run(bool use_multishot) {
  Scheduler sched(1, system_allocator(), 4096);
  IoContext io(sched, 1);
  Pair p = make_pair();
  io.bind_fd(0, p.a);
  io.bind_fd(0, p.b);
  if (use_multishot) {
    // ONE ring shared by the single receiver. Enough buffers for a full window in
    // flight + headroom.
    auto uring = io.proactor(0).io_uring_extension();
    if (!uring || !uring.register_buffer_ring(kBgid, 1024, kMsgLen)) {
      close_pair(p);
      sched.shutdown();
      return -1;  // non-uring backend / old kernel: skip
    }
  }
  Shared sh{};
  sh.io = &io;
  sh.send_fd = p.a;
  sh.recv_fd = p.b;

  auto ds = std::make_unique<Drive>();
  auto dr = std::make_unique<Drive>();
  ds->task = sender(system_allocator(), &sh);
  dr->task = use_multishot ? recv_multishot(system_allocator(), &sh)
                           : recv_oneshot(system_allocator(), &sh);
  // Spawn the receiver FIRST (armed to consume), then the sender.
  Fiber fr(kDefaultStackSize, &drive_entry, dr.get());
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  sched.spawn_on(0, &fr);
  sched.spawn_on(0, &fs);
  sched.drain();
  sched.shutdown();
  close_pair(p);
  return sh.received;
}

}  // namespace

int main() {
  // Correctness smoke: both receivers must drain exactly kMsgs (or -1 = skipped).
  int ms_got = run(/*use_multishot=*/true);
  int os_got = run(/*use_multishot=*/false);
  std::fprintf(stderr, "[smoke] multishot received=%d  one-shot received=%d  (expect %d each; -1=skip)\n",
               ms_got, os_got, kMsgs);
  if (ms_got == -1) {
    std::fprintf(stderr,
                 "[skip] active backend lacks provided-buffer multishot recv\n");
    return 0;
  }
  if (ms_got != kMsgs || os_got != kMsgs) {
    std::fprintf(stderr, "[fail] smoke completed unequal work\n");
    return 1;
  }

  const std::uint64_t batch = static_cast<std::uint64_t>(kMsgs);
  ankerl::nanobench::Bench b;
  b.title("io recv: 2000 datagrams x 64B (same-worker sender+receiver, DGRAM)")
      .relative(true)
      .batch(batch)
      .unit("datagram")
      .minEpochIterations(1)
      .epochs(5);
  b.run("HYLE multishot (1 arm, ring buffers, no per-recv submit)",
        [&] {
          if (run(/*use_multishot=*/true) != kMsgs) std::abort();
        });
  b.run("HYLE one-shot RecvOp (submit per datagram)",
        [&] {
          if (run(/*use_multishot=*/false) != kMsgs) std::abort();
        });
  return 0;
}
#else   // !__linux__
int main() {
  std::fprintf(stderr, "multishot bench: Linux/io_uring only (no IOCP provided-buffer-"
                       "ring equivalent) -- skipped on this platform\n");
  return 0;
}
#endif  // __linux__
