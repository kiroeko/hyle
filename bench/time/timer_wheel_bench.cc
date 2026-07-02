// time-02 timer-wheel bench: is the hierarchical wheel actually the right data
// structure for the "arm >> cancel >> expire" agent workload, or is a heap fine?
//
// SOTA rival = a min-heap of (deadline, id), the structure Go/libev/libuv pick
// for timers. It is the honest dependency-free comparison for the DATA STRUCTURE
// claim in time-02-timer-wheel.md. (folly HHWheelTimer would be the same-family
// rival, but it is welded to an EventBase/libevent loop + glog + boost -- linking
// that into a microbench measures the event loop, not the wheel. tokio is Rust.
// So we race the alternative ALGORITHM, which is what the design doc argues over.)
//
// The heap must be modelled honestly on both counts the doc calls out:
//   - cancel: a binary heap has no O(1) keyed removal, so real timer code
//     (libev, Go pre-4-ary-tweaks) uses LAZY cancel = mark a tombstone, skip it
//     at pop. That is what we model. It is the heap's genuine behaviour, not a
//     strawman -- and it is exactly why the doc says the heap "bloats under
//     add-then-cancel churn".
//   - expire: pop while top <= now, skipping tombstones.
//
// Decision rule (perf doc): run under scripts/bench-stat.py -n>=15; a gap smaller
// than the summed run-to-run spread is a STATISTICAL TIE, not a win.
#include <nanobench.h>

#include <cstdint>
#include <queue>
#include <vector>

#include "hyle/time/timer_wheel.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::time::TimerEntry;
using hyle::time::TimerWheel;

constexpr std::uint64_t kMs = TimerWheel::kTickNs;

// ---- honest min-heap timer (Go/libev style, lazy cancel) --------------------
struct HeapTimer {
  struct Node {
    std::uint64_t deadline;
    std::uint64_t id;
    bool operator>(const Node& o) const { return deadline > o.deadline; }
  };
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::vector<bool> cancelled;  // id -> tombstone (lazy cancel, no keyed remove)
  std::uint64_t live = 0;

  explicit HeapTimer(std::size_t cap) : cancelled(cap, false) {}
  void schedule(std::uint64_t id, std::uint64_t deadline) {
    pq.push({deadline, id});
    ++live;
  }
  void cancel(std::uint64_t id) {
    if (!cancelled[id]) {
      cancelled[id] = true;  // O(1) mark; the node stays in the heap (bloat)
      --live;
    }
  }
  // Drain everything due; skip tombstones (the lazy-cancel cost surfaces here).
  std::uint64_t expire(std::uint64_t now, std::uint64_t& fired) {
    while (!pq.empty() && pq.top().deadline <= now) {
      Node n = pq.top();
      pq.pop();
      if (!cancelled[n.id]) {
        ++fired;
      }
    }
    return fired;
  }
};

}  // namespace

int main() {
  // Workload shape (the target reality): arm N timeouts, cancel most of them
  // before they fire (a response arrived), expire the survivors. N per op.
  constexpr std::uint64_t kN = 4096;

  // ---- 1. arm-then-cancel (the dominant hot path) ----
  // Every timer armed then immediately cancelled -- the "response beat the
  // timeout" case that the doc says dominates. Measures add+cancel cost.
  {
    nb::Bench b;
    b.title("timer arm+cancel (N=4096, all cancelled)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    b.run("min-heap (Go/libev, lazy cancel)", [&] {
      HeapTimer h(kN);
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      for (std::uint64_t i = 0; i < kN; ++i) h.cancel(i);
      nb::doNotOptimizeAway(h.live);
    });
    b.run("HYLE timing wheel", [&] {
      TimerWheel w(0);
      std::vector<TimerEntry> e(kN);
      for (std::uint64_t i = 0; i < kN; ++i) {
        e[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
        w.schedule(&e[i], (i % 1000 + 1) * kMs);
      }
      for (std::uint64_t i = 0; i < kN; ++i) w.cancel(&e[i]);
      nb::doNotOptimizeAway(w.size());
    });
  }

  // ---- 2. arm-then-expire (all fire) ----
  // No cancels: arm N, advance past all deadlines, drain. Measures add + the
  // cascade/expiry path vs heap pop-all.
  {
    nb::Bench b;
    b.title("timer arm+expire (N=4096, all fire)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    b.run("min-heap (Go/libev)", [&] {
      HeapTimer h(kN);
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      std::uint64_t fired = 0;
      h.expire(1001 * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE timing wheel", [&] {
      TimerWheel w(0);
      std::vector<TimerEntry> e(kN);
      for (std::uint64_t i = 0; i < kN; ++i) {
        e[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
        w.schedule(&e[i], (i % 1000 + 1) * kMs);
      }
      TimerWheel::EntryList out;
      w.advance(1001 * kMs, out);
      std::uint64_t fired = 0;
      while (out.pop_front()) ++fired;
      nb::doNotOptimizeAway(fired);
    });
  }

  // ---- 3. realistic churn: arm N, cancel 70%, expire the rest ----
  // The full agent picture. The heap carries 70% tombstones through expiry
  // (the bloat the doc predicts); the wheel unlinked them in O(1) at cancel.
  {
    nb::Bench b;
    b.title("timer churn (N=4096, 70% cancelled then expire)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    b.run("min-heap (Go/libev, lazy cancel)", [&] {
      HeapTimer h(kN);
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      for (std::uint64_t i = 0; i < kN; ++i)
        if (i % 10 < 7) h.cancel(i);
      std::uint64_t fired = 0;
      h.expire(1001 * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE timing wheel", [&] {
      TimerWheel w(0);
      std::vector<TimerEntry> e(kN);
      for (std::uint64_t i = 0; i < kN; ++i) {
        e[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
        w.schedule(&e[i], (i % 1000 + 1) * kMs);
      }
      for (std::uint64_t i = 0; i < kN; ++i)
        if (i % 10 < 7) w.cancel(&e[i]);
      TimerWheel::EntryList out;
      w.advance(1001 * kMs, out);
      std::uint64_t fired = 0;
      while (out.pop_front()) ++fired;
      nb::doNotOptimizeAway(fired);
    });
  }

  return 0;
}
