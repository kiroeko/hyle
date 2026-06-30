// platform primitive bench (red-team gap-fill #55/#57): Parker park/unpark and
// Thread spawn/join -- the cross-platform blocking + thread primitives that the
// scheduler rides on but that had ZERO benchmark (only mentioned in comments).
//
//  1. Parker fast path: unpark() THEN park() (token already available) -> park
//     consumes the token and returns without ever sleeping. The hot path the CV/
//     scheduler hits when a wake races ahead of the wait. Rival: std::atomic flag
//     spin (the lower bound) -- Parker should be close, since the fast path is
//     just a fetch_sub + branch, no syscall.
//  2. Parker round-trip: two threads ping-pong via park/unpark (REAL block+wake,
//     a futex/WaitOnAddress syscall each way). Rival: std::condition_variable
//     ping-pong (the std equivalent of "block until told"). This is the honest
//     cost of the blocking path.
//  3. Thread spawn+join: HYLE platform::Thread (native CreateThread/pthread, can
//     set stack/affinity) vs std::thread. Round-trip of create + run trivial + join.
//
// NOTE: thread create/join is dominated by the OS; the SOTA here IS std::thread
// (both wrap the same syscalls). The number proves HYLE's handle adds no overhead.

#include <nanobench.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <thread>

#include "hyle/platform/fs.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/wait.h"

using hyle::platform::Parker;
using hyle::platform::Thread;
using hyle::platform::ThreadOpts;

namespace {
namespace nb = ankerl::nanobench;

void noop_entry(void*) {}
}  // namespace

int main() {
  // ---- 1. Parker fast path: unpark-then-park (token present, no sleep) -------
  {
    constexpr int kIters = 100000;
    Parker parker;
    nb::Bench b;
    b.title("Parker unpark+park fast path (no block)")
        .relative(true)
        .batch(kIters)
        .minEpochIterations(20);
    // Lower bound: a plain atomic store+load+branch (what a no-syscall wake costs).
    b.run("atomic flag set+consume (lower bound)", [&] {
      std::atomic<int> flag{0};
      for (int i = 0; i < kIters; ++i) {
        flag.store(1, std::memory_order_release);
        while (flag.exchange(0, std::memory_order_acquire) == 0) {}  // consume
      }
      nb::doNotOptimizeAway(flag.load());
    });
    b.run("HYLE Parker unpark()+park()", [&] {
      for (int i = 0; i < kIters; ++i) {
        parker.unpark();  // make a token available
        parker.park();    // consume it, no sleep (token present)
      }
      nb::doNotOptimizeAway(&parker);
    });
  }

  // ---- 2. Parker round-trip: 2-thread ping-pong (REAL block+wake) ------------
  // Thread A parks waiting for B; B unparks A then parks waiting for A; repeat.
  // Each leg is a real futex/WaitOnAddress sleep+wake. Rival: the same ping-pong
  // on std::condition_variable (the std "block until told" primitive).
  {
    constexpr int kRounds = 20000;
    {
      nb::Bench b;
      b.title("blocking ping-pong round-trip (2 threads)")
          .relative(true)
          .batch(kRounds)
          .minEpochIterations(3);

      b.run("std::condition_variable ping-pong", [&] {
        std::mutex m;
        std::condition_variable cv;
        int turn = 0;  // 0 = A's turn, 1 = B's turn
        std::thread bthread([&] {
          for (int i = 0; i < kRounds; ++i) {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return turn == 1; });
            turn = 0;
            cv.notify_one();
          }
        });
        for (int i = 0; i < kRounds; ++i) {
          std::unique_lock<std::mutex> lk(m);
          turn = 1;
          cv.notify_one();
          cv.wait(lk, [&] { return turn == 0; });
        }
        bthread.join();
      });

      b.run("HYLE Parker ping-pong", [&] {
        Parker pa;  // A parks here
        Parker pb;  // B parks here
        std::atomic<bool> go{true};
        std::thread bthread([&] {
          for (int i = 0; i < kRounds; ++i) {
            pb.park();      // wait for A's signal
            pa.unpark();    // signal A
          }
        });
        for (int i = 0; i < kRounds; ++i) {
          pb.unpark();      // signal B
          pa.park();        // wait for B's signal
        }
        nb::doNotOptimizeAway(go.load());
        bthread.join();
      });
    }
  }

  // ---- 3. Thread spawn+join round-trip: HYLE Thread vs std::thread -----------
  // OS-dominated; std::thread IS the SOTA (same syscalls). Proves HYLE's ABI-clean
  // handle (native CreateThread/pthread + stack/affinity options) costs no more.
  {
    constexpr int kSpawns = 2000;
    nb::Bench b;
    b.title("thread spawn+join round-trip")
        .relative(true)
        .batch(kSpawns)
        .minEpochIterations(5);
    b.run("std::thread spawn+join", [&] {
      for (int i = 0; i < kSpawns; ++i) {
        std::thread t([] {});
        t.join();
      }
    });
    b.run("HYLE platform::Thread spawn+join", [&] {
      for (int i = 0; i < kSpawns; ++i) {
        auto r = Thread::spawn(&noop_entry, nullptr);
        if (r) r.value().join();
      }
    });
  }

  // ---- 4. directory_iterate: visitor scan of a directory (vs std::filesystem) -
  // HYLE directory_iterate is a type-erased VISITOR (no allocation, no per-entry
  // object). std::filesystem::directory_iterator constructs a directory_entry per
  // step (stats + stores a path). Syscall-dominated (readdir/FindNextFile), so the
  // win is bounded -- but the no-allocation visitor should still edge out the
  // entry-materializing iterator. Scans the current directory (read-only).
  {
    nb::Bench b;
    b.title("directory_iterate (visitor vs std::filesystem)")
        .relative(true)
        .minEpochIterations(200);
    b.run("std::filesystem::directory_iterator", [&] {
      std::uint64_t n = 0;
      std::error_code ec;
      for (auto it = std::filesystem::directory_iterator(".", ec);
           it != std::filesystem::directory_iterator(); it.increment(ec)) {
        n += it->is_directory(ec) ? 2 : 1;
      }
      nb::doNotOptimizeAway(n);
    });
    b.run("HYLE directory_iterate (visitor)", [&] {
      std::uint64_t n = 0;
      hyle::platform::for_each_entry(".", [&](const char*, bool is_dir) {
        n += is_dir ? 2 : 1;
      });
      nb::doNotOptimizeAway(n);
    });
  }

  return 0;
}
