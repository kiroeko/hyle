#include "hyle/graph/data_parallel.h"

#include <atomic>
#include <cstddef>
#include <new>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/waker.h"
#include "hyle/util/spin_lock.h"

// graph-02 step-1 impl: parallel_for + parallel_reduce (fast) as recursive binary-split
// fork-join over the sched work-stealing pool. Compiled only where fcontext (sched) exists.
// See data_parallel.h for the model + the 5 design decisions.

namespace hyle::graph {

namespace {

// Auto grainsize (decision ①): aim for ~ workers * kOversample leaf tasks -- enough for
// work-stealing to balance, few enough that split/spawn overhead stays negligible. Ceil-div
// so we never under-shoot to 0. Clamp to >=1.
constexpr std::size_t kOversample = 4;
// Floor for AUTO grainsize. WHY: each recursive split forks a task, and spawn_task eagerly
// allocates a full fiber stack (kDefaultStackSize) + JoinState per fork. auto-grain that
// bisected all the way to 1 on a huge range would churn ~n such 64KB fibers -> memory blowup.
// The floor keeps auto tasks coarse enough that fork overhead stays negligible; a caller who
// truly wants finer blocks passes an explicit small grainsize (and owns the cost). Explicit
// grainsize is NEVER floored -- only kAutoGrain.
constexpr std::size_t kAutoGrainFloor = 1024;
std::size_t resolve_grain(sched::Scheduler& sched, BlockedRange r) noexcept {
  if (r.grainsize != kAutoGrain) return r.grainsize;
  const std::size_t n = r.size();
  const std::size_t w = sched.num_workers();
  const std::size_t targets = (w == 0 ? 1 : w) * kOversample;  // targets >= 1
  std::size_t g = n / targets;  // floor-div; no overflow (targets>=1, n<=SIZE_MAX)
  if (g < kAutoGrainFloor) g = kAutoGrainFloor;
  return g;  // >= kAutoGrainFloor >= 1
}

// Shared state for ONE parallel_for: the body + ctx (immutable), plus the first error. Lives
// on the calling fiber's stack (it blocks in join until the whole tree is done, so the frame
// outlives every task). first_err is written once via a CAS gate; err_mu guards the Status
// write (Status is refcounted, not trivially assignable under a bare atomic).
struct ForState {
  ForBodyFn body;
  void* ctx;
  std::size_t grain;
  std::atomic<bool> failed{false};
  util::SpinLock err_mu;
  Status first_err;

  void record(Status s) noexcept {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      err_mu.lock();
      first_err = std::move(s);
      err_mu.unlock();
    }
  }
};

// Recursive fork-join over [b,e). Returns trivial bool (ok) -- the Status travels through
// ForState::record, NOT through the Task's JoinState value slot (graph-01 lesson: a refcounted
// Status in JoinState double-handles its refcount -> UAF). LEAF (size <= grain): run the body.
// SPLIT: spawn the RIGHT half as a task (lands on this worker's local queue), recurse the LEFT
// inline (co_await), then join the right. The join is the fork-join barrier + the lifetime
// edge that keeps `st` (on the root fiber stack) alive across the whole subtree.
sched::Task<bool> pf_rec(mem::AllocatorRef alloc, sched::Scheduler* sched, ForState* st,
                         std::size_t b, std::size_t e) {
  // Stop splitting once at/under grainsize OR once a sibling already failed (prune the tree).
  if (e - b <= st->grain || st->failed.load(std::memory_order_acquire)) {
    if (!st->failed.load(std::memory_order_acquire)) {
      Status s = st->body(st->ctx, b, e);
      if (!s.ok()) st->record(std::move(s));
    }
    co_return true;
  }
  const std::size_t mid = b + (e - b) / 2;
  auto right = spawn_task(*sched, alloc, pf_rec(alloc, sched, st, mid, e));  // fork right
  auto left = pf_rec(alloc, sched, st, b, mid);
  if (left.valid()) {
    co_await left;  // recurse left inline (same fiber)
  } else {
    st->record(Status::ResourceExhausted(
        "parallel_for left coroutine frame allocation failed"));
  }
  auto right_result = right.join();            // join right (fork-join barrier)
  if (!right_result) {
    // A worker-side recursive spawn can be rejected when every bounded runnable
    // queue is full.  Dropping that error used to report success after silently
    // skipping the right range.
    st->record(std::move(right_result).error().WithContext(
        "parallel_for right subtree"));
  }
  co_return true;
}

// Driver coroutine: run the whole recursion on ONE spawned fiber, then wake the caller. Same
// trivial-return + caller-slot pattern; here the "slot" is ForState itself, on the caller stack.
sched::Task<bool> pf_driver(mem::AllocatorRef alloc, sched::Scheduler* sched, ForState* st,
                            std::size_t b, std::size_t e) {
  auto root = pf_rec(alloc, sched, st, b, e);
  if (root.valid()) {
    co_await root;
  } else {
    st->record(Status::ResourceExhausted(
        "parallel_for recursive coroutine frame allocation failed"));
  }
  co_return true;
}

}  // namespace

Status parallel_for(sched::Scheduler& sched, mem::AllocatorRef alloc, BlockedRange range,
                    ForBodyFn body, void* ctx) noexcept {
  HYLE_FATAL_ASSERT(body != nullptr && range.begin <= range.end);
  if (range.begin == range.end) return Status();  // empty range -> nothing to do

  ForState st;
  st.body = body;
  st.ctx = ctx;
  st.grain = resolve_grain(sched, range);

  // Run the recursion under a spawned root task and join it: join() is the fiber-aware barrier
  // that blocks THIS fiber until the whole tree finishes, keeping `st` (this frame) alive
  // throughout. The root task returns trivial bool; the real result is st.first_err.
  auto root = spawn_task(sched, alloc, pf_driver(alloc, &sched, &st, range.begin, range.end));
  auto root_result = root.join();
  if (!root_result) {
    return std::move(root_result).error().WithContext(
        "parallel_for root task");
  }

  if (st.failed.load(std::memory_order_acquire)) {
    st.err_mu.lock();
    Status s = st.first_err;
    st.err_mu.unlock();
    return s;
  }
  return Status();
}

// ---- parallel_reduce (fast) --------------------------------------------------
namespace {

// Reduce shared state: body/combine/identity + elem_size (all immutable) + first error. Unlike
// parallel_for, each subtree PRODUCES a value; the value is combined up the tree, so it travels
// as a by-value byte blob through the recursion's out-param (not JoinState). first_err as above.
struct RedState {
  ReduceBodyFn body;
  CombineFn combine;
  IdentityFn identity;
  void* ctx;
  std::size_t elem_size;
  std::size_t grain;
  std::atomic<bool> failed{false};
  util::SpinLock err_mu;
  Status first_err;

  void record(Status s) noexcept {
    bool expected = false;
    if (failed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      err_mu.lock();
      first_err = std::move(s);
      err_mu.unlock();
    }
  }
};

// Recurse [b,e) into `out` (an elem_size slot the CALLER provides, on the recursion's stack).
// LEAF: body reduces the block into out. SPLIT: fork right into a SEPARATE slot, recurse left
// into `out`, join right, then combine(out, right_slot). The right slot is a stack buffer in
// THIS frame (alive across the join). Returns trivial bool; Status via record.
sched::Task<bool> pr_rec(mem::AllocatorRef alloc, sched::Scheduler* sched, RedState* st,
                         std::size_t b, std::size_t e, void* out) {
  if (e - b <= st->grain || st->failed.load(std::memory_order_acquire)) {
    if (!st->failed.load(std::memory_order_acquire)) {
      Status s = st->body(st->ctx, b, e, out);
      if (!s.ok()) st->record(std::move(s));
    } else {
      st->identity(st->ctx, out);  // failed path: leave a well-formed identity in out
    }
    co_return true;
  }
  const std::size_t mid = b + (e - b) / 2;
  // Right partial goes in a stack slot; allocate elem_size bytes with max alignment.
  alignas(alignof(std::max_align_t)) unsigned char rbuf[256];
  void* rslot = rbuf;  // elem_size <= 256 asserted at entry (see parallel_reduce)
  auto right = spawn_task(*sched, alloc, pr_rec(alloc, sched, st, mid, e, rslot));
  auto left = pr_rec(alloc, sched, st, b, mid, out);
  if (left.valid()) {
    co_await left;  // left into out
  } else {
    st->record(Status::ResourceExhausted(
        "parallel_reduce left coroutine frame allocation failed"));
  }
  auto right_result = right.join();
  if (!right_result) {
    // On admission failure the right coroutine never ran, so rslot is
    // uninitialized.  Record the structured scheduler error before the combine
    // guard; reading rslot here was the intermittent ReduceSumMatchesSerial UB.
    st->record(std::move(right_result).error().WithContext(
        "parallel_reduce right subtree"));
  }
  if (!st->failed.load(std::memory_order_acquire))
    st->combine(st->ctx, out, rslot);  // out := combine(left, right)
  co_return true;
}

}  // namespace

Status parallel_reduce(sched::Scheduler& sched, mem::AllocatorRef alloc, BlockedRange range,
                       std::size_t elem_size, ReduceBodyFn body, CombineFn combine,
                       IdentityFn identity, void* ctx, void* result) noexcept {
  HYLE_FATAL_ASSERT(body && combine && identity && result && elem_size > 0);
  HYLE_FATAL_ASSERT(elem_size <= 256 && range.begin <= range.end);
  identity(ctx, result);  // seed result with identity (also the empty-range answer)
  if (range.begin == range.end) return Status();

  RedState st;
  st.body = body;
  st.combine = combine;
  st.identity = identity;
  st.ctx = ctx;
  st.elem_size = elem_size;
  st.grain = resolve_grain(sched, range);

  auto root = spawn_task(sched, alloc, pr_rec(alloc, &sched, &st, range.begin, range.end, result));
  auto root_result = root.join();
  if (!root_result) {
    return std::move(root_result).error().WithContext(
        "parallel_reduce root task");
  }

  if (st.failed.load(std::memory_order_acquire)) {
    st.err_mu.lock();
    Status s = st.first_err;
    st.err_mu.unlock();
    return s;
  }
  return Status();
}

}  // namespace hyle::graph
