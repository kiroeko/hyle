#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::CancelCallback;
using hyle::sched::CancelScope;
using hyle::sched::CancelToken;
using hyle::sched::Fiber;
using hyle::sched::is_cancel_requested;
using hyle::sched::kDefaultStackSize;
using hyle::sched::schedule_task;
using hyle::sched::Scheduler;
using hyle::sched::spawn_task;
using hyle::sched::SuspendFiber;
using hyle::sched::CaptureWaker;
using hyle::sched::Task;
using hyle::sched::Waker;

namespace {
void* fail_alloc(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void*, void*, std::size_t, std::size_t) noexcept {}
void* fail_realloc(void*, void*, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}

struct FailNthAllocation {
  std::size_t attempts = 0;
  std::size_t allocations = 0;
  std::size_t frees = 0;
  std::size_t fail_on = 0;
};

void* fail_nth_alloc(void* raw, std::size_t size,
                     std::size_t alignment) noexcept {
  auto* state = static_cast<FailNthAllocation*>(raw);
  if (state->attempts++ == state->fail_on) return nullptr;
  void* pointer = system_allocator().allocate(size, alignment);
  if (pointer != nullptr) ++state->allocations;
  return pointer;
}

void fail_nth_free(void* raw, void* pointer, std::size_t size,
                   std::size_t alignment) noexcept {
  auto* state = static_cast<FailNthAllocation*>(raw);
  if (pointer != nullptr) ++state->frees;
  system_allocator().deallocate(pointer, size, alignment);
}

void* fail_nth_realloc(void*, void* pointer, std::size_t old_size,
                       std::size_t new_size,
                       std::size_t alignment) noexcept {
  return system_allocator().reallocate(pointer, old_size, new_size, alignment);
}
}  // namespace

// sched-07 cancellation-tree tests. The tree ops (derive/cancel/propagate/re-parent/
// callback) are pure control-path logic -- most can be exercised directly on the
// test thread. CancelScope migration and cancellable IO still need a real Fiber;
// the same execution-local chain also serves worker-stack tasks.

// ---- tree propagation (pure logic, no scheduler) ---------------------------

// A lone root token: cancel sets its own flag.
TEST(Cancel, RootCancelSetsFlag) {
  CancelToken root = CancelToken::root(system_allocator());
  ASSERT_TRUE(root.valid());
  EXPECT_FALSE(root.is_cancelled());
  root.cancel();
  EXPECT_TRUE(root.is_cancelled());
}

TEST(Cancel, InvalidAndExhaustedAllocatorsAreRecoverable) {
  auto invalid = CancelToken::try_create_root(AllocatorRef{});
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
  AllocatorRef failing(&fail_alloc, &fail_free, &fail_realloc, nullptr);
  auto exhausted = CancelToken::try_create_root(failing);
  ASSERT_FALSE(exhausted.has_value());
  EXPECT_TRUE(exhausted.error().IsResourceExhausted());
  CancelScope no_op{CancelToken{}};
  EXPECT_FALSE(is_cancel_requested());
}

TEST(Cancel, InvalidDeriveAndEveryAllocationStageAreRecoverable) {
  auto invalid = CancelToken{}.try_derive();
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());

  for (std::size_t fail_on : {1u, 2u}) {
    FailNthAllocation state;
    state.fail_on = fail_on;
    AllocatorRef allocator(&fail_nth_alloc, &fail_nth_free,
                           &fail_nth_realloc, &state);
    {
      auto root_result = CancelToken::try_create_root(allocator);
      if (fail_on == 1) {
        ASSERT_FALSE(root_result.has_value());
        EXPECT_TRUE(root_result.error().IsResourceExhausted());
      } else {
        ASSERT_TRUE(root_result.has_value());
        CancelToken root = *std::move(root_result);
        auto child = root.try_derive();
        ASSERT_FALSE(child.has_value());
        EXPECT_TRUE(child.error().IsResourceExhausted());
        root.cancel();
        EXPECT_TRUE(root.is_cancelled());
      }
    }
    EXPECT_EQ(state.allocations, state.frees) << fail_on;
  }
}

TEST(Cancel, ScopePublishesOnWorkerStackWithoutAFiber) {
  CancelToken token = CancelToken::root(system_allocator());
  token.cancel();
  EXPECT_FALSE(is_cancel_requested());
  Scheduler scheduler(1, system_allocator(), 64);
  auto handle = schedule_task(
      scheduler, system_allocator(), [token]() noexcept {
        CancelScope scope{token};
        return is_cancel_requested();
      });
  auto result = handle.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(*result);
  scheduler.shutdown();
  EXPECT_FALSE(is_cancel_requested());
}

TEST(Cancel, CopiesShareOneThreadSafeCancellationState) {
  CancelToken root = CancelToken::root(system_allocator());
  CancelToken copy = root;
  std::thread canceller([copy]() { copy.cancel(); });
  canceller.join();
  EXPECT_TRUE(root.is_cancelled());
  EXPECT_TRUE(copy.is_cancelled());
}

TEST(Cancel, CopyAssignmentReleasesOldTreeAndSupportsSelfAssignment) {
  CancelToken first = CancelToken::root(system_allocator());
  CancelToken second = CancelToken::root(system_allocator());
  CancelToken assigned = first.derive();

  assigned = second;
  first.cancel();
  EXPECT_FALSE(assigned.is_cancelled());

  // Exercise runtime self-assignment through an alias. Spelling
  // `assigned = assigned` is rejected by current AppleClang under WERROR,
  // before the ownership contract can be tested.
  CancelToken* const self = &assigned;
  assigned = *self;
  second.cancel();
  EXPECT_TRUE(assigned.is_cancelled());

  CancelToken empty;
  assigned = empty;
  EXPECT_FALSE(assigned.valid());
}

// Cancelling a parent cascades to a derived child.
TEST(Cancel, ParentCancelPropagatesToChild) {
  CancelToken root = CancelToken::root(system_allocator());
  CancelToken child = root.derive();
  ASSERT_TRUE(child.valid());
  EXPECT_FALSE(child.is_cancelled());
  root.cancel();
  EXPECT_TRUE(root.is_cancelled());
  EXPECT_TRUE(child.is_cancelled());  // cascaded
}

// A grandchild (multi-level tree) is cancelled by the root -- the fan-out shape an
// HTTP request produces (request -> upstream A -> sub-call).
TEST(Cancel, DeepTreePropagates) {
  CancelToken root = CancelToken::root(system_allocator());
  CancelToken a = root.derive();
  CancelToken b = root.derive();
  CancelToken a1 = a.derive();
  CancelToken a2 = a.derive();
  CancelToken b1 = b.derive();
  root.cancel();
  EXPECT_TRUE(a.is_cancelled());
  EXPECT_TRUE(b.is_cancelled());
  EXPECT_TRUE(a1.is_cancelled());
  EXPECT_TRUE(a2.is_cancelled());
  EXPECT_TRUE(b1.is_cancelled());
}

// Cancelling a SUBTREE root cancels only that branch, not siblings or the root.
TEST(Cancel, SubtreeCancelIsScoped) {
  CancelToken root = CancelToken::root(system_allocator());
  CancelToken a = root.derive();
  CancelToken b = root.derive();
  CancelToken a1 = a.derive();
  a.cancel();
  EXPECT_TRUE(a.is_cancelled());
  EXPECT_TRUE(a1.is_cancelled());  // a's subtree
  EXPECT_FALSE(b.is_cancelled());  // sibling untouched
  EXPECT_FALSE(root.is_cancelled());  // parent untouched
}

// Deriving from an ALREADY-cancelled token yields a born-cancelled child (the
// parent-cancels-then-new-child race, resolved without a lock since same shard).
TEST(Cancel, DeriveFromCancelledIsBornCancelled) {
  CancelToken root = CancelToken::root(system_allocator());
  root.cancel();
  CancelToken child = root.derive();
  ASSERT_TRUE(child.valid());
  EXPECT_TRUE(child.is_cancelled());
}

// Destructor re-parenting: a middle token dropped while its child lives on must not
// orphan the grandchild from the root's cancel (tokio move_children_to_parent). We
// drop `mid` (its node re-parents `leaf` to root), then cancel root -> leaf must see it.
TEST(Cancel, DropMiddleReparentsGrandchild) {
  CancelToken root = CancelToken::root(system_allocator());
  CancelToken leaf;
  {
    CancelToken mid = root.derive();
    leaf = mid.derive();
    // mid drops here (end of scope): leaf is re-parented onto root.
  }
  EXPECT_FALSE(leaf.is_cancelled());
  root.cancel();
  EXPECT_TRUE(leaf.is_cancelled());  // still reachable from root after re-parent
}

// After a middle token is dropped and its child re-parented, deriving a FURTHER
// grandchild off the re-parented child stays connected to root (the re-parent
// rewired the links correctly, not just for the one existing child).
TEST(Cancel, DeriveAfterReparentStaysConnected) {
  CancelToken root = CancelToken::root(system_allocator());
  CancelToken leaf;
  {
    CancelToken mid = root.derive();
    leaf = mid.derive();  // mid drops -> leaf re-parented to root
  }
  CancelToken deeper = leaf.derive();  // derive off the re-parented node
  ASSERT_TRUE(deeper.valid());
  root.cancel();
  EXPECT_TRUE(leaf.is_cancelled());
  EXPECT_TRUE(deeper.is_cancelled());  // reachable root -> leaf -> deeper
}

// Many siblings under one parent, plus a moderately deep chain, to exercise the
// iterative (non-recursive) walk without relying on stack depth.
TEST(Cancel, WideAndDeepTreeAllCancelled) {
  CancelToken root = CancelToken::root(system_allocator());
  std::vector<CancelToken> wide;
  for (int i = 0; i < 64; ++i) wide.push_back(root.derive());
  // A deep chain hanging off the first wide child.
  std::vector<CancelToken> deep;
  CancelToken cur = wide[0].derive();
  for (int i = 0; i < 256; ++i) {
    CancelToken next = cur.derive();
    deep.push_back(std::move(cur));
    cur = std::move(next);
  }
  deep.push_back(std::move(cur));
  root.cancel();
  for (auto& t : wide) EXPECT_TRUE(t.is_cancelled());
  for (auto& t : deep) EXPECT_TRUE(t.is_cancelled());
}

// ---- callbacks -------------------------------------------------------------

// A CancelCallback fires exactly once when its token is cancelled.
TEST(Cancel, CallbackFiresOnCancel) {
  CancelToken root = CancelToken::root(system_allocator());
  int fired = 0;
  auto fn = [](void* c) noexcept { ++*static_cast<int*>(c); };
  {
    CancelCallback cb{root, fn, &fired};
    EXPECT_EQ(fired, 0);
    root.cancel();
    EXPECT_EQ(fired, 1);
  }
}

// Registering a callback on an ALREADY-cancelled token runs it immediately (else a
// cancel that already happened would be missed forever).
TEST(Cancel, CallbackOnAlreadyCancelledFiresNow) {
  CancelToken root = CancelToken::root(system_allocator());
  root.cancel();
  int fired = 0;
  auto fn = [](void* c) noexcept { ++*static_cast<int*>(c); };
  CancelCallback cb{root, fn, &fired};
  EXPECT_EQ(fired, 1);  // fired on registration
}

// A callback unregistered (its scope exits) before cancel does NOT fire -- the io-op
// "completed normally, tear down the cancel hook" path.
TEST(Cancel, UnregisteredCallbackDoesNotFire) {
  CancelToken root = CancelToken::root(system_allocator());
  int fired = 0;
  auto fn = [](void* c) noexcept { ++*static_cast<int*>(c); };
  { CancelCallback cb{root, fn, &fired}; }  // unregisters at scope end
  root.cancel();
  EXPECT_EQ(fired, 0);
}

// A callback that fires on a CHILD when the PARENT is cancelled (propagation reaches
// registered hooks down the tree, not just the flag).
TEST(Cancel, ParentCancelFiresChildCallback) {
  CancelToken root = CancelToken::root(system_allocator());
  CancelToken child = root.derive();
  int fired = 0;
  auto fn = [](void* c) noexcept { ++*static_cast<int*>(c); };
  CancelCallback cb{child, fn, &fired};
  root.cancel();
  EXPECT_EQ(fired, 1);
}

// A callback body that unregisters ANOTHER callback / registers work must not corrupt
// the walk. Here the body deregisters itself via its own RAII after firing; we assert
// no double-fire and no crash when several are attached.
TEST(Cancel, MultipleCallbacksAllFireOnce) {
  CancelToken root = CancelToken::root(system_allocator());
  int counts[3] = {0, 0, 0};
  auto fn = [](void* c) noexcept { ++*static_cast<int*>(c); };
  CancelCallback c0{root, fn, &counts[0]};
  CancelCallback c1{root, fn, &counts[1]};
  CancelCallback c2{root, fn, &counts[2]};
  root.cancel();
  EXPECT_EQ(counts[0], 1);
  EXPECT_EQ(counts[1], 1);
  EXPECT_EQ(counts[2], 1);
  root.cancel();  // idempotent: no second fire
  EXPECT_EQ(counts[0], 1);
  EXPECT_EQ(counts[1], 1);
  EXPECT_EQ(counts[2], 1);
}

// Robustness: a callback body that unregisters a STILL-PENDING sibling callback must
// not corrupt the fire walk (pop-then-invoke re-reads the live head each step). Here
// c0's body destroys c1 (the adjacent pending callback); c1 must then NOT fire, and
// c2 must still fire. Exercises the fire_callbacks sibling-teardown guard.
namespace {
struct SiblingTeardown {
  std::unique_ptr<CancelCallback> victim;  // the callback c0's body will destroy
  int c0 = 0, c2 = 0;
};
}  // namespace
TEST(Cancel, CallbackBodyUnregisteringSiblingIsSafe) {
  CancelToken root = CancelToken::root(system_allocator());
  SiblingTeardown st;
  int victim_fired = 0;
  auto victim_fn = [](void* c) noexcept { ++*static_cast<int*>(c); };
  // Register the victim (c1) FIRST so it sits deeper in the list; c0 is pushed after
  // it (front insert), so when c0 fires it tears down the still-pending c1.
  st.victim = std::make_unique<CancelCallback>(root, victim_fn, &victim_fired);
  auto c0_fn = [](void* c) noexcept {
    auto* s = static_cast<SiblingTeardown*>(c);
    ++s->c0;
    s->victim.reset();  // destroy the adjacent still-pending callback mid-walk
  };
  CancelCallback c0{root, c0_fn, &st};
  auto c2_fn = [](void* c) noexcept { ++static_cast<SiblingTeardown*>(c)->c2; };
  CancelCallback c2{root, c2_fn, &st};
  root.cancel();
  EXPECT_EQ(st.c0, 1);        // c0 fired
  EXPECT_EQ(victim_fired, 0); // c1 torn down by c0's body -> never fired
  EXPECT_EQ(st.c2, 1);        // c2 still fired (walk survived the teardown)
}

// ---- CancelScope <-> is_cancel_requested() integration (needs a fiber) ------

namespace {

// A task that installs a CancelScope for `tok`, parks until woken, then reports
// whether cancellation is observed via is_cancel_requested() (the sched-10 seam).
struct ScopeBox {
  CancelToken tok;
  std::atomic<bool> ready{false};
  Waker waker;
  std::atomic<int> observed{-1};  // 1 = saw cancel, 0 = did not
};

Task<int> scoped_waiter(AllocatorRef, ScopeBox* box) {
  CancelScope scope{box->tok};  // publish tok's flag to this fiber
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};
  box->observed.store(is_cancel_requested() ? 1 : 0, std::memory_order_release);
  co_return 0;
}

}  // namespace

// A CancelScope makes a token's cancellation visible to is_cancel_requested() inside
// the running fiber -- so a task opts into a token with one RAII line and the existing
// cooperative-cancel checkpoint works unchanged.
TEST(Cancel, ScopePublishesTokenToIsCancelRequested) {
  Scheduler sched(2, system_allocator(), 256);
  auto box = std::make_unique<ScopeBox>();
  box->tok = CancelToken::root(system_allocator());
  auto h = spawn_task(sched, system_allocator(), scoped_waiter(system_allocator(), box.get()));
  while (!box->ready.load(std::memory_order_acquire)) {}
  box->tok.cancel();      // cancel via the TOKEN (not JoinHandle task request)
  box->waker.wake();      // wake the parked fiber so it reaches the check
  auto r = h.join();
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(box->observed.load(), 1);  // the fiber saw the token's cancellation
  sched.shutdown();
}

TEST(Cancel, TokenScopeComposesWithJoinHandleCancelRequest) {
  Scheduler sched(2, system_allocator(), 256);
  auto box = std::make_unique<ScopeBox>();
  box->tok = CancelToken::root(system_allocator());
  auto handle = spawn_task(
      sched, system_allocator(),
      scoped_waiter(system_allocator(), box.get()));
  while (!box->ready.load(std::memory_order_acquire)) {}
  handle.request_cancel();  // request token remains live and not cancelled
  box->waker.wake();
  auto result = handle.join();
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(box->observed.load(std::memory_order_acquire), 1);
  EXPECT_FALSE(box->tok.is_cancelled());
  sched.shutdown();
}

// Fan-out: a parent token drives several child-scoped tasks; cancelling the parent
// is observed by every child (the HTTP request -> N sub-tasks connected-cancel shape).
namespace {
struct FanBox {
  CancelToken child;
  std::atomic<bool> ready{false};
  Waker waker;
  std::atomic<int> observed{-1};
};
Task<int> fan_child(AllocatorRef, FanBox* box) {
  CancelScope scope{box->child};
  box->waker = co_await CaptureWaker{};
  box->ready.store(true, std::memory_order_release);
  co_await SuspendFiber{};
  box->observed.store(is_cancel_requested() ? 1 : 0, std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(Cancel, ParentCancelObservedByAllFannedOutChildren) {
  constexpr int kChildren = 4;
  Scheduler sched(2, system_allocator(), 256);
  CancelToken parent = CancelToken::root(system_allocator());
  std::unique_ptr<FanBox> boxes[kChildren];
  hyle::sched::JoinHandle<int> handles[kChildren];
  for (int i = 0; i < kChildren; ++i) {
    boxes[i] = std::make_unique<FanBox>();
    boxes[i]->child = parent.derive();  // each child token derives from the parent
    handles[i] = spawn_task(sched, system_allocator(),
                            fan_child(system_allocator(), boxes[i].get()));
  }
  for (int i = 0; i < kChildren; ++i)
    while (!boxes[i]->ready.load(std::memory_order_acquire)) {}
  parent.cancel();  // one cancel -> all children's tokens cancelled
  for (int i = 0; i < kChildren; ++i) {
    boxes[i]->waker.wake();
    auto r = handles[i].join();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(boxes[i]->observed.load(), 1) << "child " << i << " missed cancel";
  }
  sched.shutdown();
}
