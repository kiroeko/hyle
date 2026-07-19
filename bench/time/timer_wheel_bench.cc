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
// The heap baseline stores each node's heap index and erases it eagerly in
// O(log n), matching libevent min_heap_erase_ and libuv heap_remove. The old
// benchmark used lazy tombstones and therefore did not model either reference.
//
// Decision rule (perf doc): run under scripts/bench-stat.py -n>=15; a gap smaller
// than the summed run-to-run spread is a STATISTICAL TIE, not a win.
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hyle/time/common_timeout_queue.h"
#include "hyle/time/timer_wheel.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::time::CommonTimeoutQueue;
using hyle::time::TimerEntry;
using hyle::time::TimerWheel;

constexpr std::uint64_t kMs = TimerWheel::kTickNs;

// ---- SOTA same-algorithm baseline for CommonTimeoutQueue --------------------
// libevent embeds TAILQ_ENTRY directly in struct event and removes through the
// event pointer. Mirror that exact zero-allocation/direct-node shape; std::list +
// unordered_map was not a libevent baseline and artificially charged allocation
// plus hashing to the rival.
struct IntrusiveTimeoutQueue {
  struct Node {
    std::uint64_t deadline = 0;
    std::uint64_t source_state = 0;
    void* data = nullptr;
    void (*on_fire)(Node*) noexcept = nullptr;
    Node* prev = nullptr;
    Node* next = nullptr;
    const void* owner = nullptr;
  };
  std::vector<Node> nodes;
  Node* head = nullptr;
  Node* tail = nullptr;
  std::size_t count = 0;

  explicit IntrusiveTimeoutQueue(std::size_t cap) : nodes(cap) {}
  void schedule(std::uint64_t id, std::uint64_t deadline) {
    Node& n = nodes[static_cast<std::size_t>(id)];
    n.deadline = deadline;
    n.data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(id));
    n.prev = tail;
    n.next = nullptr;
    n.owner = this;
    if (tail != nullptr) tail->next = &n;
    else head = &n;
    tail = &n;
    ++count;
  }
  void cancel(std::uint64_t id) {
    Node& n = nodes[static_cast<std::size_t>(id)];
    if (n.owner == nullptr) return;
    if (n.prev != nullptr) n.prev->next = n.next;
    else head = n.next;
    if (n.next != nullptr) n.next->prev = n.prev;
    else tail = n.prev;
    n.prev = n.next = nullptr;
    n.owner = nullptr;
    --count;
  }
  std::uint64_t expire(std::uint64_t now, std::uint64_t& fired) {
    Node* ready_head = nullptr;
    Node* ready_tail = nullptr;
    while (head != nullptr && head->deadline <= now) {
      Node* n = head;
      head = n->next;
      if (head != nullptr) head->prev = nullptr;
      else tail = nullptr;
      n->prev = ready_tail;
      n->next = nullptr;
      if (ready_tail != nullptr) ready_tail->next = n;
      else ready_head = n;
      ready_tail = n;
      n->owner = nullptr;
      --count;
    }
    nb::doNotOptimizeAway(ready_head);
    // Same contract as HYLE advance: expired nodes first move to a caller-owned
    // intrusive output list, then the caller drains that list.
    while (ready_head != nullptr) {
      Node* n = ready_head;
      ready_head = n->next;
      if (ready_head != nullptr) ready_head->prev = nullptr;
      n->prev = n->next = nullptr;
      ++fired;
    }
    return fired;
  }
};

// ---- indexed binary min-heap (libevent/libuv eager erase shape) --------------
struct IndexedHeapTimer {
  struct Node {
    std::uint64_t deadline = 0;
    std::uint64_t id = 0;
    std::size_t index = SIZE_MAX;
    Node* ready_prev = nullptr;
    Node* ready_next = nullptr;
  };
  std::vector<Node> nodes;
  std::vector<Node*> heap;
  std::uint64_t live = 0;

  explicit IndexedHeapTimer(std::size_t cap) : nodes(cap) {
    heap.reserve(cap);
  }
  void schedule(std::uint64_t id, std::uint64_t deadline) {
    Node& n = nodes[static_cast<std::size_t>(id)];
    n.deadline = deadline;
    n.id = id;
    n.index = heap.size();
    heap.push_back(&n);
    shift_up(n.index);
    ++live;
  }
  void cancel(std::uint64_t id) {
    Node& n = nodes[static_cast<std::size_t>(id)];
    if (n.index == SIZE_MAX) return;
    erase_at(n.index);
    --live;
  }
  std::uint64_t expire(std::uint64_t now, std::uint64_t& fired) {
    Node* ready_head = nullptr;
    Node* ready_tail = nullptr;
    while (!heap.empty() && heap.front()->deadline <= now) {
      Node* n = heap.front();
      erase_at(0);
      --live;
      n->ready_prev = ready_tail;
      n->ready_next = nullptr;
      if (ready_tail != nullptr) ready_tail->ready_next = n;
      else ready_head = n;
      ready_tail = n;
    }
    nb::doNotOptimizeAway(ready_head);
    while (ready_head != nullptr) {
      Node* n = ready_head;
      ready_head = n->ready_next;
      if (ready_head != nullptr) ready_head->ready_prev = nullptr;
      n->ready_prev = n->ready_next = nullptr;
      ++fired;
    }
    return fired;
  }

 private:
  void swap_nodes(std::size_t a, std::size_t b) {
    Node* tmp = heap[a];
    heap[a] = heap[b];
    heap[b] = tmp;
    heap[a]->index = a;
    heap[b]->index = b;
  }
  void shift_up(std::size_t i) {
    while (i != 0) {
      const std::size_t parent = (i - 1) / 2;
      if (heap[parent]->deadline <= heap[i]->deadline) break;
      swap_nodes(parent, i);
      i = parent;
    }
  }
  void shift_down(std::size_t i) {
    for (;;) {
      const std::size_t left = i * 2 + 1;
      if (left >= heap.size()) return;
      const std::size_t right = left + 1;
      std::size_t child =
          right < heap.size() &&
                  heap[right]->deadline < heap[left]->deadline
              ? right
              : left;
      if (heap[i]->deadline <= heap[child]->deadline) return;
      swap_nodes(i, child);
      i = child;
    }
  }
  void erase_at(std::size_t i) {
    Node* removed = heap[i];
    Node* last = heap.back();
    heap.pop_back();
    removed->index = SIZE_MAX;
    if (i == heap.size()) return;
    heap[i] = last;
    last->index = i;
    if (i != 0 &&
        heap[i]->deadline < heap[(i - 1) / 2]->deadline) {
      shift_up(i);
    } else {
      shift_down(i);
    }
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
    IndexedHeapTimer h(kN);
    TimerWheel w(0);
    std::vector<TimerEntry> e(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      e[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    }
    b.run("indexed min-heap (libevent/libuv)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      nb::doNotOptimizeAway(h.live);
      for (std::uint64_t i = 0; i < kN; ++i) h.cancel(i);
      nb::doNotOptimizeAway(h.live);
    });
    b.run("HYLE timing wheel", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&e[i], (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(w.size());
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
    IndexedHeapTimer h(kN);
    TimerWheel w(0);
    std::vector<TimerEntry> e(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      e[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    }
    std::uint64_t wheel_base = 0;
    b.run("indexed min-heap (libevent/libuv)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      nb::doNotOptimizeAway(h.live);
      std::uint64_t fired = 0;
      h.expire(1001 * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE timing wheel", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&e[i], wheel_base + (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(w.size());
      TimerWheel::EntryList out;
      w.advance(wheel_base + 1001 * kMs, out);
      wheel_base += 1001 * kMs;
      std::uint64_t fired = 0;
      while (out.pop_front()) ++fired;
      nb::doNotOptimizeAway(fired);
    });
  }

  // ---- 3. realistic churn: arm N, cancel 70%, expire the rest ----
  // The full agent picture. Both structures eagerly remove cancellations; the
  // comparison is O(1) wheel unlink vs O(log n) indexed-heap erase.
  {
    nb::Bench b;
    b.title("timer churn (N=4096, 70% cancelled then expire)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    IndexedHeapTimer h(kN);
    TimerWheel w(0);
    std::vector<TimerEntry> e(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      e[i].data = reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    }
    std::uint64_t wheel_base = 0;
    b.run("indexed min-heap (libevent/libuv)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      nb::doNotOptimizeAway(h.live);
      for (std::uint64_t i = 0; i < kN; ++i)
        if (i % 10 < 7) h.cancel(i);
      std::uint64_t fired = 0;
      h.expire(1001 * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE timing wheel", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&e[i], wheel_base + (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(w.size());
      for (std::uint64_t i = 0; i < kN; ++i)
        if (i % 10 < 7) w.cancel(&e[i]);
      TimerWheel::EntryList out;
      w.advance(wheel_base + 1001 * kMs, out);
      wheel_base += 1001 * kMs;
      std::uint64_t fired = 0;
      while (out.pop_front()) ++fired;
      nb::doNotOptimizeAway(fired);
    });
  }

  // ---- 4. FIXED-DURATION serving load: wheel vs CommonTimeoutQueue ----
  // HYLE's expiry path bulk-splices the due prefix to the ready list, then
  // clears ownership in one traversal.
  // The io-serving case: N connections all armed with the SAME timeout duration,
  // so deadlines are monotonic. This is CommonTimeoutQueue's home turf (one
  // sorted list, O(1) tail-append). We race it against the general wheel on the
  // SAME load to show WHEN to pick the queue over the wheel. Both arm N in
  // ascending-deadline order, then expire all. (Not a "wheel is bad" bench --
  // the wheel handles arbitrary durations the queue can't; this is the narrow
  // slice where the queue's specialization pays.)
  {
    nb::Bench b;
    b.title("fixed-duration arm+expire (N=4096, same timeout)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    TimerWheel w(0);
    std::vector<TimerEntry> wheel_entries(kN);
    CommonTimeoutQueue q(0, kMs);
    std::vector<TimerEntry> common_entries(kN);
    IntrusiveTimeoutQueue tailq(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      wheel_entries[i].data =
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
      common_entries[i].data =
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    }
    std::uint64_t wheel_base = 0;
    b.run("HYLE timing wheel", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&wheel_entries[i],
                   wheel_base + (i + 1) * kMs);  // monotonic
      }
      nb::doNotOptimizeAway(w.size());
      TimerWheel::EntryList out;
      w.advance(wheel_base + (kN + 1) * kMs, out);
      wheel_base += (kN + 1) * kMs;
      std::uint64_t fired = 0;
      while (out.pop_front()) ++fired;
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE CommonTimeoutQueue", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        q.schedule(&common_entries[i],
                   (i + 1) * kMs);  // monotonic tail-append
      }
      nb::doNotOptimizeAway(q.size());
      CommonTimeoutQueue::EntryList out;
      q.advance((kN + 1) * kMs, out);
      std::uint64_t fired = 0;
      while (out.pop_front()) ++fired;
      nb::doNotOptimizeAway(fired);
    });
    // SOTA same-algorithm baseline: libevent embeds the intrusive TAILQ hook.
    b.run("intrusive TAILQ (libevent common_timeout shape)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) tailq.schedule(i, (i + 1) * kMs);
      nb::doNotOptimizeAway(tailq.count);
      std::uint64_t fired = 0;
      tailq.expire((kN + 1) * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
  }

  // ---- 5. FIXED-DURATION arm+cancel (the serving hot path): all cancelled ----
  // Connections that respond before timeout -> arm then cancel. Same fixed
  // duration. Queue cancel is O(1) unlink; wheel cancel is O(1) unlink + slot
  // math. Shows the per-op arm/cancel cost on the queue's target load.
  {
    nb::Bench b;
    b.title("fixed-duration arm+cancel (N=4096, same timeout, all cancelled)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    TimerWheel w(0);
    std::vector<TimerEntry> wheel_entries(kN);
    CommonTimeoutQueue q(0, kMs);
    std::vector<TimerEntry> common_entries(kN);
    IntrusiveTimeoutQueue tailq(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      wheel_entries[i].data =
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
      common_entries[i].data =
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    }
    b.run("HYLE timing wheel", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&wheel_entries[i], (i + 1) * kMs);
      }
      nb::doNotOptimizeAway(w.size());
      for (std::uint64_t i = 0; i < kN; ++i) w.cancel(&wheel_entries[i]);
      nb::doNotOptimizeAway(w.size());
    });
    b.run("HYLE CommonTimeoutQueue", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        q.schedule(&common_entries[i], (i + 1) * kMs);
      }
      nb::doNotOptimizeAway(q.size());
      for (std::uint64_t i = 0; i < kN; ++i) q.cancel(&common_entries[i]);
      nb::doNotOptimizeAway(q.size());
    });
    // SOTA same-algorithm baseline (direct-node O(1) intrusive unlink).
    b.run("intrusive TAILQ (libevent common_timeout shape)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) tailq.schedule(i, (i + 1) * kMs);
      nb::doNotOptimizeAway(tailq.count);
      for (std::uint64_t i = 0; i < kN; ++i) tailq.cancel(i);
      nb::doNotOptimizeAway(tailq.count);
    });
  }

  return 0;
}
