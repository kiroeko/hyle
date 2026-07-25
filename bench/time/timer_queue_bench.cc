// General/fixed-duration timer-queue bench for the real
// "arm >> cancel >> expire" server/runtime workload.
//
// SOTA rival = a min-heap of (deadline, id), the structure Go/libev/libuv pick
// for timers. It is the honest dependency-free comparison for the DATA STRUCTURE
// claim. The Seastar-shaped radix timer set below is the same-complexity rival;
// the indexed libevent/libuv heap is the alternative algorithm.
//
// The heap baseline stores each node's heap index and erases it eagerly in
// O(log n), matching libevent min_heap_erase_ and libuv heap_remove. The old
// benchmark used lazy tombstones and therefore did not model either reference.
//
// Decision rule (perf doc): run under scripts/bench-stat.py -n>=15; a gap smaller
// than the summed run-to-run spread is a STATISTICAL TIE, not a win.
#include <nanobench.h>

#include <array>
#include <bit>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "hyle/time/common_timeout_queue.h"
#include "hyle/time/timer_queue.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::time::CommonTimeoutQueue;
using hyle::time::TimerEntry;
using hyle::time::TimerQueue;

constexpr std::uint64_t kMs = 100'000;

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

  explicit IntrusiveTimeoutQueue(std::size_t cap) : nodes(cap) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].data =
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    }
  }
  void schedule(std::uint64_t id, std::uint64_t deadline) {
    Node& n = nodes[static_cast<std::size_t>(id)];
    n.deadline = deadline;
    n.prev = tail;
    n.next = nullptr;
    n.owner = this;
    if (tail != nullptr) tail->next = &n;
    else head = &n;
    tail = &n;
    ++count;
  }
  // Contract-matched diagnostic: mirror the production queue's reschedule,
  // exact-owner, unlinked-node and monotonic-deadline checks before using the
  // same raw TAILQ link operation. The unchecked schedule() remains the
  // libevent-shaped lower bound.
  void schedule_checked(std::uint64_t id, std::uint64_t deadline) {
    Node& n = nodes[static_cast<std::size_t>(id)];
    if (n.owner != nullptr) {
      HYLE_FATAL_ASSERT(n.owner == this);
      cancel(id);
    }
    HYLE_FATAL_ASSERT(n.prev == nullptr && n.next == nullptr);
    HYLE_FATAL_ASSERT(tail == nullptr || tail->deadline <= deadline);
    schedule(id, deadline);
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

// ---- same-complexity mature rival: Seastar timer_set radix buckets ---------
// Faithful dependency-free port of seastar/core/timer-set.hh's core algorithm:
// 64 leading-zero buckets plus one already-due bucket, intrusive O(1) removal,
// minimum tracking, and sorting deferred until expire. The node carries HYLE's
// same persistent user-data/callback and exact-owner metadata. Callbacks are not
// invoked on either side because the benchmark ends at delivery to a
// caller-owned ready list, but omitting the fields would shrink the rival node
// from the real 48-byte public contract to an artificial 40-byte benchmark node.
struct RadixTimerSet {
  static constexpr std::size_t kBuckets = 65;
  static constexpr std::uint64_t kNever =
      std::numeric_limits<std::uint64_t>::max();

  struct Node {
    std::uint64_t deadline = 0;
    void* data = nullptr;
    void (*fire_callback)(Node*) noexcept = nullptr;
    Node* prev = nullptr;
    Node* next = nullptr;
    const void* owner = nullptr;
  };

  struct Bucket {
    Node* head = nullptr;
    Node* tail = nullptr;

    bool empty() const noexcept { return head == nullptr; }
    void push_back(Node* node) noexcept {
      node->prev = tail;
      node->next = nullptr;
      if (tail != nullptr) tail->next = node;
      else head = node;
      tail = node;
    }
    Node* pop_front() noexcept {
      Node* node = head;
      if (node == nullptr) return nullptr;
      head = node->next;
      if (head != nullptr) head->prev = nullptr;
      else tail = nullptr;
      node->prev = node->next = nullptr;
      return node;
    }
    void remove(Node* node) noexcept {
      if (node->prev != nullptr) node->prev->next = node->next;
      else head = node->next;
      if (node->next != nullptr) node->next->prev = node->prev;
      else tail = node->prev;
      node->prev = node->next = nullptr;
    }
    void append(Bucket& source) noexcept {
      if (source.empty()) return;
      if (tail != nullptr) {
        tail->next = source.head;
        source.head->prev = tail;
      } else {
        head = source.head;
      }
      tail = source.tail;
      source.head = source.tail = nullptr;
    }
    template <class Fn>
    void for_each(Fn&& fn) noexcept {
      for (Node* node = head; node != nullptr; node = node->next) fn(node);
    }
  };

  std::vector<Node> nodes;
  std::array<Bucket, kBuckets> buckets{};
  std::bitset<kBuckets> nonempty{};
  std::uint64_t last = 0;
  std::uint64_t next = kNever;
  std::uint64_t live = 0;

  explicit RadixTimerSet(std::size_t capacity) : nodes(capacity) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].data =
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i));
    }
  }

  std::size_t index_for(std::uint64_t deadline) const noexcept {
    if (deadline <= last) return kBuckets - 1;
    return static_cast<std::size_t>(std::countl_zero(deadline ^ last));
  }

  void insert_node(Node* node) noexcept {
    const std::size_t index = index_for(node->deadline);
    buckets[index].push_back(node);
    nonempty.set(index);
    if (node->deadline < next) next = node->deadline;
  }

  void schedule(std::uint64_t id, std::uint64_t deadline) noexcept {
    Node& node = nodes[static_cast<std::size_t>(id)];
    if (node.owner != nullptr) {
      HYLE_FATAL_ASSERT(node.owner == this);
      cancel(id);
    }
    HYLE_FATAL_ASSERT(node.prev == nullptr && node.next == nullptr);
    node.deadline = deadline;
    node.owner = this;
    insert_node(&node);
    ++live;
  }

  void cancel(std::uint64_t id) noexcept {
    Node& node = nodes[static_cast<std::size_t>(id)];
    if (node.owner == nullptr) return;
    HYLE_FATAL_ASSERT(node.owner == this);
    const std::size_t index = index_for(node.deadline);
    buckets[index].remove(&node);
    if (buckets[index].empty()) nonempty.reset(index);
    node.owner = nullptr;
    --live;
  }

  std::uint64_t expire(std::uint64_t now, std::uint64_t& fired) noexcept {
    HYLE_FATAL_ASSERT(now >= last);
    const std::size_t index = index_for(now);
    Bucket ready;

    // Exactly Seastar's splice of buckets known to be no later than `now`.
    for (std::size_t i = index + 1; i < kBuckets; ++i) {
      if (!nonempty.test(i)) continue;
      ready.append(buckets[i]);
      nonempty.reset(i);
    }

    Bucket inspect;
    inspect.append(buckets[index]);
    nonempty.reset(index);
    last = now;
    next = kNever;
    while (Node* node = inspect.pop_front()) {
      if (node->deadline <= now) ready.push_back(node);
      else insert_node(node);
    }

    // If no reinsert found the exact minimum, Seastar scans the nearest
    // non-empty radix bucket and computes its minimum lazily.
    if (next == kNever && nonempty.any()) {
      std::size_t nearest = kBuckets;
      while (nearest != 0) {
        --nearest;
        if (nonempty.test(nearest)) break;
      }
      for (Node* node = buckets[nearest].head; node != nullptr;
           node = node->next) {
        if (node->deadline < next) next = node->deadline;
      }
    }

    nb::doNotOptimizeAway(ready.head);
    // Match HYLE/Seastar's ready-list contract: expiry first disarms every node
    // while it remains linked in the returned list; caller delivery/drain is a
    // second pass. Folding disarm into drain would give the rival a weaker
    // lifecycle contract and hide the cost users actually pay.
    ready.for_each([](Node* node) noexcept { node->owner = nullptr; });
    while (Node* node = ready.pop_front()) {
      --live;
      ++fired;
    }
    return fired;
  }
};

static_assert(sizeof(RadixTimerSet::Node) == sizeof(TimerEntry),
              "same-class radix baseline must retain public node metadata");

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
    RadixTimerSet r(kN);
    TimerQueue w(0);
    std::vector<TimerEntry> e(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      e[i].set_data(reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
    }
    b.run("indexed min-heap (libevent/libuv)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      nb::doNotOptimizeAway(h.live);
      for (std::uint64_t i = 0; i < kN; ++i) h.cancel(i);
      nb::doNotOptimizeAway(h.live);
    });
    b.run("radix timer_set (Seastar shape)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        r.schedule(i, (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(r.live);
      for (std::uint64_t i = 0; i < kN; ++i) r.cancel(i);
      nb::doNotOptimizeAway(r.live);
    });
    b.run("HYLE TimerQueue (radix)", [&] {
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
    RadixTimerSet r(kN);
    TimerQueue w(0);
    std::vector<TimerEntry> e(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      e[i].set_data(reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
    }
    std::uint64_t queue_base = 0;
    std::uint64_t radix_base = 0;
    b.run("indexed min-heap (libevent/libuv)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      nb::doNotOptimizeAway(h.live);
      std::uint64_t fired = 0;
      h.expire(1001 * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
    b.run("radix timer_set (Seastar shape)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        r.schedule(i, radix_base + (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(r.live);
      std::uint64_t fired = 0;
      r.expire(radix_base + 1001 * kMs, fired);
      radix_base += 1001 * kMs;
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE TimerQueue (radix)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&e[i], queue_base + (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(w.size());
      TimerQueue::EntryList out;
      w.advance(queue_base + 1001 * kMs, out);
      queue_base += 1001 * kMs;
      std::uint64_t fired = 0;
      out.consume_all([&](TimerEntry*) noexcept { ++fired; });
      nb::doNotOptimizeAway(fired);
    });
  }

  // ---- 3. realistic churn: arm N, cancel 70%, expire the rest ----
  // The full agent picture. Both structures eagerly remove cancellations; the
  // comparison is O(1) radix unlink vs O(log n) indexed-heap erase.
  {
    nb::Bench b;
    b.title("timer churn (N=4096, 70% cancelled then expire)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    IndexedHeapTimer h(kN);
    RadixTimerSet r(kN);
    TimerQueue w(0);
    std::vector<TimerEntry> e(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      e[i].set_data(reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
    }
    std::uint64_t queue_base = 0;
    std::uint64_t radix_base = 0;
    b.run("indexed min-heap (libevent/libuv)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) h.schedule(i, (i % 1000 + 1) * kMs);
      nb::doNotOptimizeAway(h.live);
      for (std::uint64_t i = 0; i < kN; ++i)
        if (i % 10 < 7) h.cancel(i);
      std::uint64_t fired = 0;
      h.expire(1001 * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
    b.run("radix timer_set (Seastar shape)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        r.schedule(i, radix_base + (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(r.live);
      for (std::uint64_t i = 0; i < kN; ++i) {
        if (i % 10 < 7) r.cancel(i);
      }
      std::uint64_t fired = 0;
      r.expire(radix_base + 1001 * kMs, fired);
      radix_base += 1001 * kMs;
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE TimerQueue (radix)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&e[i], queue_base + (i % 1000 + 1) * kMs);
      }
      nb::doNotOptimizeAway(w.size());
      for (std::uint64_t i = 0; i < kN; ++i)
        if (i % 10 < 7) w.cancel(&e[i]);
      TimerQueue::EntryList out;
      w.advance(queue_base + 1001 * kMs, out);
      queue_base += 1001 * kMs;
      std::uint64_t fired = 0;
      out.consume_all([&](TimerEntry*) noexcept { ++fired; });
      nb::doNotOptimizeAway(fired);
    });
  }

  // ---- 4. FIXED-DURATION serving load: general vs common queue ----
  // HYLE's expiry path bulk-splices the due prefix to the ready list, then
  // clears ownership in one traversal.
  // The io-serving case: N connections all armed with the SAME timeout duration,
  // so deadlines are monotonic. This is CommonTimeoutQueue's home turf (one
  // sorted list, O(1) tail-append). We race it against the general queue on the
  // SAME load to show WHEN to pick the specialization. Both arm N in
  // ascending-deadline order, then expire all. The general queue handles
  // arbitrary durations the common queue cannot; this is the narrow
  // slice where the queue's specialization pays.
  {
    nb::Bench b;
    b.title("fixed-duration arm+expire (N=4096, same timeout)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    TimerQueue w(0);
    std::vector<TimerEntry> queue_entries(kN);
    CommonTimeoutQueue q(kMs);
    std::vector<TimerEntry> common_entries(kN);
    IntrusiveTimeoutQueue tailq(kN);
    IntrusiveTimeoutQueue checked_tailq(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      queue_entries[i].set_data(
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
      common_entries[i].set_data(
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
    }
    std::uint64_t queue_base = 0;
    b.run("HYLE TimerQueue (radix)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&queue_entries[i],
                   queue_base + (i + 1) * kMs);  // monotonic
      }
      nb::doNotOptimizeAway(w.size());
      TimerQueue::EntryList out;
      w.advance(queue_base + (kN + 1) * kMs, out);
      queue_base += (kN + 1) * kMs;
      std::uint64_t fired = 0;
      out.consume_all([&](TimerEntry*) noexcept { ++fired; });
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
      out.consume_all([&](TimerEntry*) noexcept { ++fired; });
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
    b.run("checked intrusive TAILQ (HYLE contracts)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        checked_tailq.schedule_checked(i, (i + 1) * kMs);
      }
      nb::doNotOptimizeAway(checked_tailq.count);
      std::uint64_t fired = 0;
      checked_tailq.expire((kN + 1) * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
  }

  // Same two implementations in the opposite registration order. Apple
  // silicon changes frequency/QoS aggressively; a fixed "HYLE then rival"
  // sequence can otherwise reward the later variant. Keep this as a permanent
  // instrument guard, not a one-off investigation.
  {
    nb::Bench b;
    b.title("fixed-duration arm+expire ORDER-SWAP (N=4096)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    CommonTimeoutQueue q(kMs);
    std::vector<TimerEntry> common_entries(kN);
    IntrusiveTimeoutQueue tailq(kN);

    b.run("intrusive TAILQ first", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) tailq.schedule(i, (i + 1) * kMs);
      nb::doNotOptimizeAway(tailq.count);
      std::uint64_t fired = 0;
      tailq.expire((kN + 1) * kMs, fired);
      nb::doNotOptimizeAway(fired);
    });
    b.run("HYLE CommonTimeoutQueue second", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        q.schedule(&common_entries[i], (i + 1) * kMs);
      }
      nb::doNotOptimizeAway(q.size());
      CommonTimeoutQueue::EntryList out;
      q.advance((kN + 1) * kMs, out);
      std::uint64_t fired = 0;
      out.consume_all([&](TimerEntry*) noexcept { ++fired; });
      nb::doNotOptimizeAway(fired);
    });
  }

  // ---- 5. FIXED-DURATION arm+cancel (the serving hot path): all cancelled ----
  // Connections that respond before timeout -> arm then cancel. Same fixed
  // duration. Both cancel through O(1) intrusive unlink; this shows the cost
  // saved when the fixed-duration invariant removes radix bucket work.
  {
    nb::Bench b;
    b.title("fixed-duration arm+cancel (N=4096, same timeout, all cancelled)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(200);
    TimerQueue w(0);
    std::vector<TimerEntry> queue_entries(kN);
    CommonTimeoutQueue q(kMs);
    std::vector<TimerEntry> common_entries(kN);
    IntrusiveTimeoutQueue tailq(kN);
    for (std::uint64_t i = 0; i < kN; ++i) {
      queue_entries[i].set_data(
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
      common_entries[i].set_data(
          reinterpret_cast<void*>(static_cast<std::uintptr_t>(i)));
    }
    b.run("HYLE TimerQueue (radix)", [&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        w.schedule(&queue_entries[i], (i + 1) * kMs);
      }
      nb::doNotOptimizeAway(w.size());
      for (std::uint64_t i = 0; i < kN; ++i) w.cancel(&queue_entries[i]);
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
