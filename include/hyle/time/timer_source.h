#pragma once

#include <cstdint>

#include "hyle/container/intrusive_list.h"

// Timer data-structure abstraction (time-02 / io serving). HYLE ships MORE than
// one timer structure because the SOTA references do: libevent keeps a min-heap
// for general timeouts AND a separate per-duration sorted-queue ("common
// timeout") for the many-timers-of-the-same-duration case, because a heap is the
// wrong tool there (its own words: "more efficient than the minheap if we happen
// to get several thousands of timeout events all with the same timeout value").
//
// So the timer structure is chosen by WORKLOAD, behind one static-polymorphism
// concept (NOT a vtable -- arming a timer is a per-agent hot path; a virtual call
// per arm would block inlining and cost an indirect branch every time). Each
// structure is a plain type satisfying `TimerSource`; the consumer (sched worker,
// io reactor) is templated on it, so the choice is compile-time and zero-overhead.
//
//   - TimerWheel (timer_wheel.h): hierarchical timing wheel, ANY deadline, O(1)
//     amortised arm/cancel, no tombstones. The general-purpose default; what a
//     fiber sleep of arbitrary duration uses. (tokio/Linux-kernel lineage.)
//   - CommonTimeoutQueue (common_timeout_queue.h): one sorted intrusive list per
//     FIXED duration. arm = O(1) tail-append (same-duration deadlines arrive in
//     order), expire = O(1) head-pop, cancel = O(1) unlink. For the "millions of
//     connections all with the SAME 30s timeout" serving load (libevent's case).
//
// Both hand fired entries back (never run callbacks); the consumer re-injects the
// woken fibers -- the same completion-spine wake path as io/gpu (sched-design §3).

namespace hyle::time {

// One armed timer, shared by every TimerSource. Embed it in (or point it at)
// whatever the caller wants woken; the source treats `data` as opaque and never
// dereferences it. The intrusive hook means arming/cancelling allocates nothing.
// A given TimerEntry belongs to at most one TimerSource at a time.
struct TimerEntry {
  container::IntrusiveListHook hook;
  // Absolute expiry in monotonic ns (from time::monotonic_ns()). The canonical
  // field every source keys on; set by schedule(), read by cancel().
  std::uint64_t deadline_ns = 0;
  // Scratch owned by the CURRENT source while armed (opaque to everyone else).
  // A source may cache a derived key here to make cancel() O(1) without
  // recomputing under a since-changed clock: the wheel stashes the armed tick
  // (its slot math must match arm-time rounding, which depends on elapsed_ then);
  // the common-timeout queue leaves it unused. Not valid once !scheduled.
  std::uint64_t source_state = 0;
  // Opaque payload -- typically the Waker/fiber pointer the caller wants woken.
  void* data = nullptr;
  // Optional fire callback (io-06 timed io). When null (the default, e.g. plain
  // sleep) the consumer wakes `data` as a Fiber* directly. When set, the consumer
  // (fire_due_timers) invokes on_fire(this) INSTEAD of the direct wake, letting a
  // timed op arbitrate against a racing io completion before deciding to wake --
  // the timer side of the io-vs-timer race. The wheel itself never calls this (it
  // stays a pure data structure that hands back expired entries); the callback runs
  // in the consumer layer, consistent with "sources don't run callbacks".
  void (*on_fire)(TimerEntry*) noexcept = nullptr;
  // True iff currently held by a source (armed, not yet fired or cancelled).
  // Distinguishes "in the source" from "handed back on the expired list" so the
  // caller may reuse `hook` on its own list after firing.
  bool scheduled = false;
};

// The shared expired/fired list type (FIFO of TimerEntry via its hook).
using TimerList = container::IntrusiveList<TimerEntry, &TimerEntry::hook>;

// A timer data structure. Any type modelling this can back a sched worker's
// timers or an io reactor's timeouts; the consumer templates on it so the call
// sites inline and pay no indirection. `start_ns` (monotonic origin) is a ctor
// arg by convention, not part of the concept (ctors aren't constrainable here).
//
// Semantics each model MUST honour:
//   schedule(e, deadline_ns): arm `e` to fire at absolute monotonic deadline_ns.
//     If `e` is already scheduled, reschedule it. A past/at-now deadline fires on
//     the next advance() (never dropped, never fired early relative to advance).
//   cancel(e): O(1) remove if scheduled; no-op otherwise. Never a tombstone.
//   advance(now_ns, out): move every entry whose deadline <= now_ns onto `out`
//     (FIFO), clearing their scheduled flag. Runs nothing itself.
//   next_deadline_ns(): absolute ns of the earliest armed entry, or UINT64_MAX if
//     empty. Lets a worker park exactly until the next timer (no busy tick).
//   empty()/size(): live entry count bookkeeping (empty() gates the hot path).
template <class T>
concept TimerSource = requires(T t, const T ct, TimerEntry* e,
                               std::uint64_t ns, TimerList& out) {
  { t.schedule(e, ns) } noexcept -> std::same_as<void>;
  { t.cancel(e) } noexcept -> std::same_as<void>;
  { t.advance(ns, out) } noexcept -> std::same_as<void>;
  { ct.next_deadline_ns() } noexcept -> std::same_as<std::uint64_t>;
  { ct.empty() } noexcept -> std::same_as<bool>;
  { ct.size() } noexcept -> std::same_as<std::size_t>;
};

}  // namespace hyle::time
