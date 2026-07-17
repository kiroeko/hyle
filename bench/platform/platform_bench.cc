// platform primitive bench (red-team gap-fill #55/#57): Parker park/unpark and
// Thread spawn/join -- the cross-platform blocking + thread primitives that the
// scheduler rides on but that had ZERO benchmark (only mentioned in comments).
//
//  1. Parker fast path: unpark() THEN park() (token already available) -> park
//     consumes the token and returns without ever sleeping. The hot path the CV/
//     scheduler hits when a wake races ahead of the wait. The atomic-flag row is
//     a NO-SYSCALL FLOOR REFERENCE, *not* a fair rival: a bare flag gives no
//     unpark-before-park guarantee (a wake delivered before the wait is lost ->
//     deadlock), so it cannot be Parker's apples-to-apples opponent. It only
//     shows the fast path is in the ~10ns / no-syscall regime; the gap to it is
//     the fixed cost of the 3-state machine's wake safety, NOT a "loss". (No
//     standard primitive is both token-safe AND syscall-free on the fast path:
//     std::binary_semaphore is token-safe but futexes every acquire. So this
//     row reports an absolute floor, with no honest win/loss denominator -- see
//     performance.md platform section.)
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
#include "hyle/platform/virtual_memory.h"
#include "hyle/platform/wait.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

using hyle::platform::Parker;
using hyle::platform::Thread;
using hyle::platform::ThreadOpts;
using hyle::platform::vm_commit;
using hyle::platform::vm_release;
using hyle::platform::vm_reserve;

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
    // No-syscall FLOOR REFERENCE (not a fair rival): a plain atomic store+load+
    // branch -- what a wake with ZERO wake-safety costs. Parker's gap to this is
    // the fixed price of unpark-before-park safety, not a loss (see file header).
    b.run("atomic flag set+consume (floor ref, NOT fair rival)", [&] {
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

  // ---- 5. VM reserve+commit+release cycle: HYLE vm_* vs the raw OS syscalls ---
  // vm_reserve/commit/release are thin wrappers over mmap/mprotect (POSIX) or
  // VirtualAlloc (Windows). The SOTA floor IS the raw syscall (same kernel work);
  // this proves the wrapper adds no measurable overhead. One page per iteration
  // (reserve address space, commit RW, release) -- syscall-dominated cold path.
  {
    constexpr int kIters = 20000;
    const std::size_t pg = hyle::platform::page_size();
    nb::Bench b;
    b.title("VM reserve+commit+release (1 page)")
        .relative(true)
        .batch(kIters)
        .minEpochIterations(50);
    b.run("raw OS (mmap+munmap / VirtualAlloc+Free)", [&] {
      for (int i = 0; i < kIters; ++i) {
#if defined(_WIN32)
        void* p = ::VirtualAlloc(nullptr, pg, MEM_RESERVE | MEM_COMMIT,
                                 PAGE_READWRITE);
        if (p) ::VirtualFree(p, 0, MEM_RELEASE);
#else
        void* p = ::mmap(nullptr, pg, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) ::munmap(p, pg);
#endif
        nb::doNotOptimizeAway(p);
      }
    });
    b.run("HYLE vm_reserve+commit+release", [&] {
      for (int i = 0; i < kIters; ++i) {
        void* p = vm_reserve(pg);
        if (p) {
          vm_commit(p, pg);
          vm_release(p, pg);
        }
        nb::doNotOptimizeAway(p);
      }
    });
  }

  // ---- 6. Large-region commit throughput: reserve once, commit a big span ----
  // The mem/fiber-stack pattern: reserve a large arena, then commit it. Measures
  // the page-fault / mprotect cost over a multi-MiB span (not per-page syscall).
  {
    constexpr std::size_t kSpan = 16u * 1024u * 1024u;  // 16 MiB
    constexpr int kIters = 200;
    nb::Bench b;
    b.title("VM commit 16MiB region (reserve+commit+release)")
        .relative(true)
        .batch(kIters)
        .minEpochIterations(3);
    b.run("HYLE vm_reserve+commit(16MiB)+release", [&] {
      for (int i = 0; i < kIters; ++i) {
        void* p = vm_reserve(kSpan);
        if (p) {
          vm_commit(p, kSpan);
          vm_release(p, kSpan);
        }
        nb::doNotOptimizeAway(p);
      }
    });
  }

  // ---- 7. Parker round-trip vs the RAW futex/WaitOnAddress floor -------------
  // Section 2 raced std::condition_variable (the std "block until told"). This
  // adds the true SOTA FLOOR: a ping-pong on the raw OS wait primitive HYLE wraps
  // (futex / WaitOnAddress via wait_on_address + wake_one). Parker layers a
  // token-safe 3-state machine on top; the gap here is exactly that safety's cost,
  // measured against the bare syscall (an honest floor, like section 1's flag).
  {
    constexpr int kRounds = 20000;
    nb::Bench b;
    b.title("blocking ping-pong vs raw wait primitive (2 threads)")
        .relative(true)
        .batch(kRounds)
        .minEpochIterations(30);
    b.run("raw wait_on_address+wake floor", [&] {
      // One word: turn==0 => A runs, ==1 => B runs. Each side waits while the
      // value is NOT its turn and flips + wakes the other. This is the bare
      // primitive with NO token safety (a wake before the wait is re-checked via
      // the value). detail::wait_on_address is the OS futex/WaitOnAddress wrapper.
      namespace pd = hyle::platform::detail;
      std::atomic<uint32_t> turn{0};
      std::thread bthread([&] {
        for (int i = 0; i < kRounds; ++i) {
          while (turn.load(std::memory_order_acquire) != 1)
            pd::wait_on_address(&turn, 0);  // sleep while value==0 (not our turn)
          turn.store(0, std::memory_order_release);
          pd::wake_one(&turn);
        }
      });
      for (int i = 0; i < kRounds; ++i) {
        turn.store(1, std::memory_order_release);
        pd::wake_one(&turn);
        while (turn.load(std::memory_order_acquire) != 0)
          pd::wait_on_address(&turn, 1);  // sleep while value==1 (not our turn)
      }
      bthread.join();
    });
    b.run("HYLE Parker ping-pong", [&] {
      Parker pa, pb;
      std::thread bthread([&] {
        for (int i = 0; i < kRounds; ++i) {
          pb.park();
          pa.unpark();
        }
      });
      for (int i = 0; i < kRounds; ++i) {
        pb.unpark();
        pa.park();
      }
      bthread.join();
    });
  }

  return 0;
}
