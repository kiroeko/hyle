#if defined(_MSC_VER)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

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
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hyle/platform/fs.h"
#include "hyle/platform/mapped_file.h"
#include "hyle/platform/thread.h"
#include "hyle/platform/virtual_memory.h"
#include "hyle/platform/wait.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

using hyle::platform::MappedFile;
using hyle::platform::Parker;
using hyle::platform::page_size;
using hyle::platform::Protect;
using hyle::platform::Thread;
using hyle::platform::ThreadOpts;
using hyle::platform::vm_allocate_rw;
using hyle::platform::vm_commit;
using hyle::platform::vm_protect;
using hyle::platform::vm_release;
using hyle::platform::vm_reserve;

namespace {
namespace nb = ankerl::nanobench;

void noop_entry(void*) {}
}  // namespace

int main() {
  const char* only = std::getenv("HYLE_PLATFORM_BENCH_ONLY");
  const auto enabled = [only](const char* tag) noexcept {
    return only == nullptr || only[0] == '\0' || std::strstr(only, tag) != nullptr;
  };
  // ---- 1. Parker fast path: unpark-then-park (token present, no sleep) -------
  if (enabled("parker-fast")) {
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
  if (enabled("parker-block")) {
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
  if (enabled("thread")) {
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
  if (enabled("directory")) {
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
      auto completed = hyle::platform::for_each_entry(
          ".", [&](const char*, bool is_dir) {
            n += is_dir ? 2 : 1;
          });
      nb::doNotOptimizeAway(completed.has_value());
      nb::doNotOptimizeAway(n);
    });
  }

  // ---- 5. VM reserve+commit+release cycle: HYLE vm_* vs the raw OS syscalls ---
  // vm_reserve/commit/release are thin wrappers over mmap/mprotect (POSIX) or
  // VirtualAlloc (Windows). The SOTA floor IS the raw syscall (same kernel work);
  // this proves the wrapper adds no measurable overhead. One page per iteration
  // (reserve address space, commit RW, release) -- syscall-dominated cold path.
  if (enabled("vm-cycle")) {
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
    b.run("HYLE vm_allocate_rw+release", [&] {
      for (int i = 0; i < kIters; ++i) {
        auto allocated = vm_allocate_rw(pg);
        void* p = allocated ? *allocated : nullptr;
        if (p != nullptr) (void)vm_release(p, pg);
        nb::doNotOptimizeAway(p);
      }
    });
    b.run("HYLE vm_reserve+commit+release", [&] {
      for (int i = 0; i < kIters; ++i) {
        auto reserved = vm_reserve(pg);
        void* p = reserved ? *reserved : nullptr;
        if (p != nullptr) {
          (void)vm_commit(p, pg);
          (void)vm_release(p, pg);
        }
        nb::doNotOptimizeAway(p);
      }
    });
  }

  // ---- 6. Large-region commit throughput: reserve once, commit a big span ----
  // The mem/fiber-stack pattern: reserve a large arena, then commit it. Measures
  // the page-fault / mprotect cost over a multi-MiB span (not per-page syscall).
  if (enabled("vm-large")) {
    constexpr std::size_t kSpan = 16u * 1024u * 1024u;  // 16 MiB
    constexpr int kIters = 200;
    nb::Bench b;
    b.title("VM commit 16MiB region (reserve+commit+release)")
        .relative(true)
        .batch(kIters)
        .minEpochIterations(3);
    b.run("HYLE vm_reserve+commit(16MiB)+release", [&] {
      for (int i = 0; i < kIters; ++i) {
        auto reserved = vm_reserve(kSpan);
        void* p = reserved ? *reserved : nullptr;
        if (p != nullptr) {
          (void)vm_commit(p, kSpan);
          (void)vm_release(p, kSpan);
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
  if (enabled("parker-raw")) {
    constexpr int kRounds = 20000;
    nb::Bench b;
    // minEpochIterations kept modest: each epoch is 2*kRounds real futex wakes.
    // WSL/Linux ~20µs/round makes minEpoch=30 alone multi-minute; multi-run
    // (bench-stat -n15) then burns hours without tighter error bars.
    b.title("blocking ping-pong vs raw wait primitive (2 threads)")
        .relative(true)
        .batch(kRounds)
        .minEpochIterations(5);
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

  // ---- 8. park_timeout: fast path (token present) vs zero-timeout poll -------
  // park_timeout has two hot regimes. FAST PATH: a token is already available
  // (unpark ran first), so the fetch_sub sees NOTIFIED and returns true with NO
  // syscall -- the same ~10ns/no-syscall floor as park()'s fast path (section 1).
  // ZERO-TIMEOUT POLL: no token + timeout_ns==0, so it takes the timed wait and
  // returns false at once; that still ENTERS the OS wait (WaitOnAddress ms=0 /
  // FUTEX_WAIT {0,0}), so this row is the syscall-entry cost of the timed path
  // with no real sleep (a genuine expiry just adds the deadline's wall time on
  // top). The gap between the two rows is exactly that syscall's price.
  //
  // kIters intentionally tiny for the zero-timeout arm: each call is a real
  // kernel entry (Win ~µs, Linux futex ETIMEDOUT ~tens of µs). Old 100k made
  // bench-stat -n15 multi-hour on WSL. Ratio vs fast path only needs order-of-
  // magnitude, not 1% precision.
  if (enabled("parker-timeout")) {
    constexpr int kIters = 500;
    Parker parker;
    nb::Bench b;
    b.title("Parker park_timeout: fast path vs zero-timeout poll")
        .relative(true)
        .batch(kIters)
        .minEpochIterations(30);
    b.run("park_timeout fast path (token present, no syscall)", [&] {
      for (int i = 0; i < kIters; ++i) {
        parker.unpark();  // make a token available
        nb::doNotOptimizeAway(
            parker.park_timeout(1'000'000'000ull));  // consumes it, no sleep
      }
    });
    b.run("park_timeout zero-timeout poll (no token, syscall)", [&] {
      for (int i = 0; i < kIters; ++i) {
        nb::doNotOptimizeAway(parker.park_timeout(0));  // no token -> false at once
      }
    });
  }

  // ---- 9. vm_protect toggle: HYLE vs raw mprotect / VirtualProtect -----------
  // Flipping page protection is the guard-page / W^X hot op (fiber stacks, JIT).
  // vm_protect is a thin wrapper over mprotect (POSIX) / VirtualProtect (Windows),
  // so the raw syscall IS the floor (same kernel work). Reserve+commit one page
  // once, then time a RW->RO->RW toggle PAIR per iteration (leaves the page RW so
  // the run repeats cleanly). Proves the wrapper + enum mapping cost nothing.
  if (enabled("vm-protect")) {
    constexpr int kIters = 50000;
    const std::size_t pg = page_size();
    auto reserved = vm_reserve(pg);
    void* base = reserved ? *reserved : nullptr;
    if (base != nullptr) (void)vm_commit(base, pg);
    nb::Bench b;
    b.title("VM protect toggle (1 page RW<->RO pair)")
        .relative(true)
        .batch(kIters)
        .minEpochIterations(20);
    b.run("raw mprotect / VirtualProtect", [&] {
      for (int i = 0; i < kIters; ++i) {
#if defined(_WIN32)
        DWORD old_prot;
        ::VirtualProtect(base, pg, PAGE_READONLY, &old_prot);
        ::VirtualProtect(base, pg, PAGE_READWRITE, &old_prot);
#else
        ::mprotect(base, pg, PROT_READ);
        ::mprotect(base, pg, PROT_READ | PROT_WRITE);
#endif
      }
      nb::doNotOptimizeAway(base);
    });
    b.run("HYLE vm_protect", [&] {
      for (int i = 0; i < kIters; ++i) {
        (void)vm_protect(base, pg, Protect::kRead);
        (void)vm_protect(base, pg, Protect::kReadWrite);
      }
      nb::doNotOptimizeAway(base);
    });
    if (base != nullptr) (void)vm_release(base, pg);
  }

  // ---- 10. mapped_file open+read: HYLE vs std::ifstream vs raw mmap ----------
  // Zero-copy file load (weights/datasets): MappedFile maps the bytes and the OS
  // pages them in on touch -- no user-space copy. std::ifstream is the std "read a
  // file" rival: it COPIES every byte into a buffer. The raw mmap / MapViewOfFile
  // path is the zero-copy FLOOR (the same syscalls HYLE wraps). Each row opens,
  // sums every byte (forces the reads / page faults), and closes. 64 KiB scratch.
  if (enabled("mapped-file")) {
    namespace sfs = std::filesystem;
    constexpr std::size_t kBytes = 64u * 1024u;
    const std::string path =
        (sfs::temp_directory_path() / "hyle_platform_bench_map.bin").string();
    {
      std::ofstream f(path, std::ios::binary | std::ios::trunc);
      std::vector<char> seed(kBytes);
      for (std::size_t i = 0; i < kBytes; ++i)
        seed[i] = static_cast<char>(i & 0x7F);
      f.write(seed.data(), static_cast<std::streamsize>(kBytes));
    }
    nb::Bench b;
    b.title("mapped_file open+read 64KiB (vs ifstream / raw mmap)")
        .relative(true)
        .minEpochIterations(50);
    b.run("std::ifstream read+sum (copies)", [&] {
      std::ifstream f(path, std::ios::binary);
      std::vector<char> buf(kBytes);
      f.read(buf.data(), static_cast<std::streamsize>(kBytes));
      std::uint64_t s = 0;
      for (std::size_t i = 0; i < kBytes; ++i)
        s += static_cast<unsigned char>(buf[i]);
      nb::doNotOptimizeAway(s);
    });
    b.run("raw mmap+sum floor", [&] {
      std::uint64_t s = 0;
#if defined(_WIN32)
      HANDLE fh = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                                nullptr);
      if (fh != INVALID_HANDLE_VALUE) {
        HANDLE mh =
            ::CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mh) {
          const unsigned char* d = static_cast<const unsigned char*>(
              ::MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0));
          if (d) {
            for (std::size_t i = 0; i < kBytes; ++i) s += d[i];
            ::UnmapViewOfFile(d);
          }
          ::CloseHandle(mh);
        }
        ::CloseHandle(fh);
      }
#else
      int fd = ::open(path.c_str(), O_RDONLY);
      if (fd >= 0) {
        void* pv = ::mmap(nullptr, kBytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (pv != MAP_FAILED) {
          const unsigned char* d = static_cast<const unsigned char*>(pv);
          for (std::size_t i = 0; i < kBytes; ++i) s += d[i];
          ::munmap(pv, kBytes);
        }
        ::close(fd);
      }
#endif
      nb::doNotOptimizeAway(s);
    });
    b.run("HYLE MappedFile open+sum (zero-copy)", [&] {
      std::uint64_t s = 0;
      auto mapped = MappedFile::open(path.c_str());
      if (mapped) {
        const unsigned char* d =
            static_cast<const unsigned char*>(mapped->data());
        for (std::size_t i = 0; i < mapped->size(); ++i) s += d[i];
      }
      nb::doNotOptimizeAway(s);
    });
    std::error_code ec;
    sfs::remove(path, ec);
  }

  return 0;
}
