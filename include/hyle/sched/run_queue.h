#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/container/bounded_queue.h"
#include "hyle/mem/allocator.h"

namespace hyle::sched {

class Fiber;  // only pointers are stored; no need for the full definition here

// Per-core run-queue: a bounded FIFO ring of runnable fibers plus a single LIFO
// slot for the "run this next" fiber (e.g. one just woken by a completion --
// resuming it immediately keeps its stack/buffers hot). Structure mirrors tokio:
// the ring itself is FIFO; the LIFO behavior lives ONLY in the slot, never in
// the ring (sched-design §2.2). sched-02 builds the structure + budget; the
// completion wiring that feeds the LIFO slot arrives in sched-04.
//
// Single-owner: one worker thread drives one RunQueue. Not yet steal-safe
// (sched-03 adds the concurrent steal protocol); today push/pop are owner-only.
class RunQueue {
 public:
  // Cap on consecutive LIFO-slot polls before the slot's fiber is demoted to the
  // FIFO tail, so a hot ping-pong pair can't starve the ring (tokio
  // MAX_LIFO_POLLS_PER_TICK).
  static constexpr int kMaxLifoPolls = 3;

  RunQueue(mem::Allocator alloc, std::size_t capacity)
      : fifo_(alloc, capacity) {}

  // Enqueue onto the FIFO tail. Returns false if the local ring is full (the
  // caller will later spill to the global queue -- sched-03). Used for newly
  // spawned fibers and for fibers that voluntarily yielded.
  bool schedule_fifo(Fiber* f) noexcept { return fifo_.try_push(f); }

  // Put `f` in the LIFO slot (run-next). If the slot was occupied, the previous
  // occupant must move to the FIFO tail first; if the ring is full we cannot do
  // that, so we leave state UNCHANGED and reject (return false) -- the caller
  // keeps ownership of `f` (and will spill to the global queue in sched-03).
  // Never drops the evicted fiber. Used by completion wakeups (sched-04).
  bool schedule_lifo(Fiber* f) noexcept {
    if (lifo_ != nullptr) {
      if (!fifo_.try_push(lifo_)) return false;  // no room to evict; intact
      lifo_ = nullptr;
    }
    lifo_ = f;
    return true;
  }

  // Pick the next fiber to run: LIFO slot (while under its poll budget) else the
  // FIFO head. Returns nullptr when both are empty. Enforces the LIFO budget so
  // the slot cannot monopolize the worker.
  Fiber* next() noexcept {
    if (lifo_ != nullptr && lifo_polls_ < kMaxLifoPolls) {
      ++lifo_polls_;
      Fiber* f = lifo_;
      lifo_ = nullptr;
      return f;
    }
    // Budget spent (or slot empty): demote any slot occupant to the ring so it
    // stays reachable, then serve the FIFO. Drawing from FIFO resets the budget.
    if (lifo_ != nullptr) {
      // Best-effort demote; if the ring is full it simply stays in the slot and
      // will be taken next call (still progresses).
      if (fifo_.try_push(lifo_)) lifo_ = nullptr;
    }
    Fiber* f = nullptr;
    if (fifo_.try_pop(f)) {
      lifo_polls_ = 0;
      return f;
    }
    // FIFO empty: if a fiber is still stuck in the slot, run it now.
    if (lifo_ != nullptr) {
      Fiber* s = lifo_;
      lifo_ = nullptr;
      lifo_polls_ = 0;
      return s;
    }
    return nullptr;
  }

  // Note: no empty() -- the worker loop terminates on next() == nullptr, which
  // already means "LIFO slot empty AND FIFO drained". BoundedQueue exposes no
  // non-consuming size/empty, so a separate predicate would either lie or
  // consume; next()'s nullptr is the single honest signal.

 private:
  // Opt-in runtime allocator (the scheduler injects one); BoundedQueue's default is the
  // compile-time SystemAllocator, so name mem::Allocator explicitly here.
  container::BoundedQueue<Fiber*, mem::Allocator> fifo_;
  Fiber* lifo_ = nullptr;
  int lifo_polls_ = 0;
};

}  // namespace hyle::sched
