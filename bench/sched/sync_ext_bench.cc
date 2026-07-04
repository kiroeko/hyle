// sched-09 concurrency-primitive microbenchmarks: Semaphore / RwLock / oneshot /
// watch / broadcast. sched-05's sync_bench covers Mutex/Channel/WaitGroup/Latch/CV;
// this file covers the sched-09 additions (tokio-semantics primitives built on the
// same fiber-aware CV floor). Each row isolates ONE primitive's hot path; a std
// baseline is raced where a fair same-shape one exists (RwLock vs std::shared_mutex),
// and where none does (oneshot/watch/broadcast are tokio-only shapes, no std analogue)
// we report HYLE's own honest cost + the scaling shape (fan-out over N receivers).
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict.

#include <nanobench.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::make_broadcast;
using hyle::sched::make_oneshot;
using hyle::sched::make_watch;
using hyle::sched::RwLock;
using hyle::sched::Scheduler;
using hyle::sched::Semaphore;

namespace {
namespace nb = ankerl::nanobench;

// ---- (2) Semaphore contended: N fibers each acquire(1)+release(1) `iters` times.
// permits=1 so they serialize through the semaphore's FIFO grant path (the blocking
// acquire -> release-wakes-front-waiter round-trip, not the try_acquire fast path).
struct SemHammer {
  Semaphore sem{1};
  int iters;
};
void sem_hammer_body(void* p) {
  auto* a = static_cast<SemHammer*>(p);
  for (int i = 0; i < a->iters; ++i) {
    a->sem.acquire(1);
    a->sem.release(1);
  }
}

// ---- (3) RwLock read-side: N reader fibers each take a read guard `iters` times.
// Readers don't exclude each other (permits share), so this is the read-lock
// acquire/release cost under concurrent readers -- the RwLock's common case.
struct RwHammer {
  RwLock rw;
  int iters;
};
void rw_read_body(void* p) {
  auto* a = static_cast<RwHammer*>(p);
  std::uint64_t sink = 0;
  for (int i = 0; i < a->iters; ++i) {
    auto g = a->rw.read();
    ++sink;
  }
  nb::doNotOptimizeAway(sink);
}

// ---- (4) oneshot send+recv round-trip: a sender fiber sends one int, a receiver
// fiber recvs it. Each epoch builds a fresh channel (oneshot is single-shot), so this
// measures the full make+send+recv+teardown of one request/response slot.
struct OneshotPair {
  Scheduler* sched;
};

// ---- (5) watch: one sender bumps the value `rounds` times; a receiver fiber recvs
// each new version. Measures the version-bump + wake-latest-observer path.
struct WatchState {
  Scheduler* sched;
  int rounds;
};

// ---- (6) broadcast fan-out: one sender pushes `msgs` values; N receiver fibers each
// recv all of them. Measures the ring-buffer fan-out (one send -> N receivers wake).

struct Drive {
  void (*fn)(void*);
  void* arg;
};

}  // namespace

int main() {
  // ---- 1. Semaphore try_acquire/release fast path (uncontended, single thread) ----
  // try_acquire is a non-blocking spinlock-guarded permit decrement; the limiter hot
  // path (acquire a slot, do work, release). No scheduler: raw primitive cost.
  {
    constexpr int kIters = 5000;
    nb::Bench b;
    b.title("Semaphore try_acquire+release (uncontended)").batch(kIters).minEpochIterations(200);
    b.run("HYLE Semaphore(1) try_acquire/release", [&] {
      Semaphore sem(1);
      for (int i = 0; i < kIters; ++i) {
        bool ok = sem.try_acquire(1);
        nb::doNotOptimizeAway(ok);
        if (ok) sem.release(1);
      }
    });
  }

  // ---- 2. Semaphore contended acquire/release: 8 fibers/4 workers, permits=1 ----
  {
    constexpr int kFibers = 8;
    constexpr int kIters = 2000;
    nb::Bench b;
    b.title("Semaphore contended acquire/release (8 fibers/4 workers)")
        .batch(static_cast<std::uint64_t>(kFibers) * kIters).minEpochIterations(20);
    b.run("HYLE Semaphore(1) acquire/release", [&] {
      SemHammer h;
      h.iters = kIters;
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kFibers);
      for (int i = 0; i < kFibers; ++i)
        fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                                 &sem_hammer_body, &h));
      Scheduler sched(4, system_allocator(), 1024);
      for (auto& f : fibers) sched.spawn(f.get());
      sched.drain();
    });
  }

  // ---- 3. RwLock read-lock vs std::shared_mutex lock_shared ----
  // HYLE RwLock is fiber-aware (built on Semaphore); std::shared_mutex is thread-
  // level. Not apples-to-apples (like the Mutex-vs-std::mutex caveat in sync_bench),
  // but it anchors the read-guard cost against the primitive every dev reaches for.
  // Uncontended single-thread read/unlock -- the raw read-side cost.
  {
    constexpr int kIters = 5000;
    nb::Bench b;
    b.title("RwLock read-lock/unlock (uncontended)").relative(true).batch(kIters).minEpochIterations(200);
    b.run("std::shared_mutex lock_shared/unlock_shared", [&] {
      std::shared_mutex sm;
      for (int i = 0; i < kIters; ++i) {
        sm.lock_shared();
        sm.unlock_shared();
      }
    });
    b.run("HYLE RwLock read()", [&] {
      RwLock rw;
      for (int i = 0; i < kIters; ++i) {
        auto g = rw.read();
      }
    });
  }

  // ---- 3b. RwLock concurrent readers: 8 reader fibers/4 workers ----
  {
    constexpr int kFibers = 8;
    constexpr int kIters = 2000;
    nb::Bench b;
    b.title("RwLock concurrent readers (8 fibers/4 workers)")
        .batch(static_cast<std::uint64_t>(kFibers) * kIters).minEpochIterations(20);
    b.run("HYLE RwLock read() x8", [&] {
      RwHammer h;
      h.iters = kIters;
      std::vector<std::unique_ptr<Fiber>> fibers;
      fibers.reserve(kFibers);
      for (int i = 0; i < kFibers; ++i)
        fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                                 &rw_read_body, &h));
      Scheduler sched(4, system_allocator(), 1024);
      for (auto& f : fibers) sched.spawn(f.get());
      sched.drain();
    });
  }

  // ---- 4. oneshot send+recv round-trip (request/response slot) ----
  // Fresh channel each iteration (single-shot); a sender fiber sends, a receiver
  // fiber recvs. batch=1: reports ns per full make+send+recv round-trip.
  {
    constexpr int kRounds = 2000;
    nb::Bench b;
    b.title("oneshot send+recv round-trip").batch(kRounds).minEpochIterations(20);
    b.run("HYLE oneshot<int> make+send+recv", [&] {
      struct S { Scheduler* sched; int rounds; } s;
      Scheduler sched(2, system_allocator(), 256);
      s.sched = &sched;
      s.rounds = kRounds;
      // One driver fiber runs the whole round loop (send + recv on the same fiber is
      // fine: send doesn't block, recv gets the already-set value without yielding).
      auto body = +[](void* p) {
        auto* st = static_cast<S*>(p);
        std::uint64_t sink = 0;
        for (int i = 0; i < st->rounds; ++i) {
          auto [tx, rx] = make_oneshot<int>(system_allocator());
          (void)std::move(tx).send(i);
          auto r = std::move(rx).recv();
          if (r.has_value()) sink += static_cast<std::uint64_t>(*r);
        }
        nb::doNotOptimizeAway(sink);
      };
      auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize, body, &s);
      sched.spawn(f.get());
      sched.drain();
    });
  }

  // ---- 5. watch send+borrow: sender bumps value, one receiver borrows latest ----
  // watch keeps only the latest value; a bump + borrow is the config-reload hot path.
  {
    constexpr int kRounds = 5000;
    nb::Bench b;
    b.title("watch send+borrow (latest-value)").batch(kRounds).minEpochIterations(50);
    b.run("HYLE watch<int> send+borrow", [&] {
      struct S { int rounds; } s{kRounds};
      Scheduler sched(1, system_allocator(), 256);
      auto body = +[](void* p) {
        auto* st = static_cast<S*>(p);
        auto [tx, rx] = make_watch<int>(system_allocator(), 0);
        std::uint64_t sink = 0;
        for (int i = 0; i < st->rounds; ++i) {
          (void)tx.send(i);
          sink += static_cast<std::uint64_t>(rx.borrow());
        }
        nb::doNotOptimizeAway(sink);
      };
      auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize, body, &s);
      sched.spawn(f.get());
      sched.drain();
    });
  }

  // ---- 6. broadcast fan-out: 1 send -> N receivers each recv it ----
  // Sweep N receivers to show the fan-out cost scales linearly (one send wakes N
  // cursors). cap sized > msgs so no receiver lags. Reported per (send x receiver).
  {
    nb::Bench b;
    b.title("broadcast fan-out: 1 send -> N recv (linear in N)").unit("delivery");
    for (int nrecv : {2, 8, 32}) {
      constexpr int kMsgs = 200;
      char name[64];
      std::snprintf(name, sizeof(name), "broadcast<int> %d receivers", nrecv);
      b.batch(static_cast<std::uint64_t>(kMsgs) * nrecv).minEpochIterations(20).run(name, [&] {
        struct S { int msgs; int nrecv; } s{kMsgs, nrecv};
        Scheduler sched(4, system_allocator(), 1024);
        // Sender + N receivers as fibers. Receivers subscribe up front, then all run.
        auto [tx, rx0] = make_broadcast<int>(system_allocator(), 256);
        // Hold the sender + receivers in structures the fibers can reach.
        struct Ctx {
          decltype(tx) sender;
          std::vector<decltype(rx0)> receivers;
          int msgs;
        };
        auto ctx = std::make_unique<Ctx>(Ctx{std::move(tx), {}, kMsgs});
        ctx->receivers.reserve(nrecv);
        ctx->receivers.push_back(std::move(rx0));
        for (int i = 1; i < nrecv; ++i) ctx->receivers.push_back(ctx->sender.subscribe());

        // Receiver fiber body: recv msgs values.
        struct RxArg { Ctx* c; int idx; };
        std::vector<std::unique_ptr<RxArg>> rxargs;
        std::vector<std::unique_ptr<Fiber>> fibers;
        auto rx_body = +[](void* p) {
          auto* ra = static_cast<RxArg*>(p);
          std::uint64_t sink = 0;
          for (int i = 0; i < ra->c->msgs; ++i) {
            auto r = ra->c->receivers[ra->idx].recv();
            if (r.has_value()) sink += static_cast<std::uint64_t>(*r);
          }
          nb::doNotOptimizeAway(sink);
        };
        auto tx_body = +[](void* p) {
          auto* c = static_cast<Ctx*>(p);
          for (int i = 0; i < c->msgs; ++i) (void)c->sender.send(i);
        };
        for (int i = 0; i < nrecv; ++i) {
          rxargs.push_back(std::make_unique<RxArg>(RxArg{ctx.get(), i}));
          fibers.push_back(std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                                   rx_body, rxargs.back().get()));
        }
        auto txf = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                           tx_body, ctx.get());
        for (auto& f : fibers) sched.spawn(f.get());
        sched.spawn(txf.get());
        sched.drain();
      });
    }
  }

  return 0;
}
