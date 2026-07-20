#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/waker.h"

#if defined(HYLE_STATIC)
#  define HYLE_SCHED_API
#elif defined(HYLE_SCHED_BUILD)
#  define HYLE_SCHED_API HYLE_EXPORT
#else
#  define HYLE_SCHED_API HYLE_IMPORT
#endif

// sched-07: structured cancellation -- CancelToken tree + await cancel points
// (lifecycle-design §1, capability C3). The value over the flat cooperative flag
// already shipped in sched-10 (JoinHandle::abort + is_cancel_requested) is TREE
// PROPAGATION: cancelling one token cascades to every token derived from it. The
// real consumer is an HTTP request fan-out (io-10): one request handler spawns
// several child tasks / upstream calls; cancelling the request (deadline, client
// hang-up) must connectedly cancel the whole subtree it spawned.
//
// SHARD-PER-CORE => NO LOCKS ON THE TREE. A request's fibers are pinned to one
// owner worker (Scheduler::spawn_on), so a token and every token derived under it
// live on, and are only ever touched by, that ONE worker thread. Tree links
// (parent*/child list), the callback list, and derive/cancel/drop all run
// single-threaded and non-atomically -- we deliberately DON'T pay tokio's
// Mutex<Inner> + try-lock-in-creation-order cost (which exists purely for its
// multi-threaded token sharing). Reference: tokio-util tree_node.rs (topology +
// destructor re-parenting + iterative cancel, all borrowed WITHOUT the locking)
// and folly CancellationCallback (intrusive callback list + register-checks-
// already-cancelled + unlink-before-invoke, borrowed WITHOUT the atomic state
// word / cross-thread deregister handshake). See docs sched-07 milestone.
//
// The ONE atomic: `cancelled` is std::atomic<bool>, but only so it plugs straight
// into the existing is_cancel_requested() seam (Fiber::cancel_flag_ is an
// atomic<bool>*; a task fiber may migrate workers between awaits, so the READ of
// the flag must be visible cross-thread -- that's the sched-10 root-cause fix).
// On x86 a relaxed/acquire bool load is a plain mov: zero cost. The WRITE only
// ever happens on the owner worker.

namespace hyle::sched {

class CancelToken;

namespace detail {

// A node in the cancellation tree. Heap-allocated through a mem::Allocator (no
// hidden new; charter). One node per CancelToken; a root has parent == nullptr.
// Owner-worker-only: every field except `cancelled` is plain (non-atomic) because
// a whole tree lives on one shard. Intrusive sibling list keeps derive/drop O(1).
struct CancelNode {
  mem::Allocator alloc;
  CancelNode* parent = nullptr;   // up-link (null for a root); used for re-parenting
  CancelNode* first_child = nullptr;
  CancelNode* next_sibling = nullptr;  // intrusive list of `parent`'s children
  CancelNode* prev_sibling = nullptr;  // doubly-linked for O(1) unlink

  // Callback list: an io op / awaiter registers here so that a cancel() reaches an
  // ALREADY-BLOCKED consumer (e.g. submit_cancel on an in-flight kernel op). folly
  // CancellationCallback shape, minus the atomics.
  struct Callback;
  Callback* first_cb = nullptr;

  // The one cross-thread-visible field (see file header). Set once, never cleared.
  std::atomic<bool> cancelled{false};

  // Re-entrancy guard: true while THIS node's cancel() is walking its callback
  // list, so a callback that unregisters itself (the common "op completed, tear
  // down its cancel hook" path) doesn't corrupt the walk. Single-threaded, so a
  // plain bool suffices (folly needs a thread-id + spin for the cross-thread case;
  // we can't have one -- same shard).
  bool invoking_cb = false;
};

// A cancellation callback: invoked (once) when the node it is attached to is
// cancelled. Intrusive, non-owning. The classic use is an in-flight io op that
// must be actively cancelled (submit_cancel) rather than merely observed at the
// next await. Lives on the registrant's frame (fiber stack / awaitable).
struct CancelNode::Callback {
  using Fn = void (*)(void* ctx) noexcept;
  CancelNode* node = nullptr;   // the node we're attached to (null once detached)
  Fn fn = nullptr;
  void* ctx = nullptr;
  Callback* next = nullptr;     // singly-linked; prev_next for O(1) unlink
  Callback** prev_next = nullptr;  // points at the pointer that points at us
};

// Cascade cancellation to `root` and its entire subtree. Iterative (an explicit
// walk down first_child / next_sibling), NOT recursive: a deep fan-out tree must
// not blow the fiber stack. Marks each node cancelled, then fires its callbacks.
// Owner-worker-only. Idempotent per node (the cancelled flag gates re-entry).
HYLE_SCHED_API void cancel_subtree(CancelNode* root) noexcept;

// Fire (and detach) every callback on one node. Sets invoking_cb across the walk.
HYLE_SCHED_API void fire_callbacks(CancelNode* n) noexcept;

// Detach `n` from its parent's child list AND re-parent its children to the
// grandparent (tokio move_children_to_parent): if a middle token is dropped while
// its children live on, the grandchildren must still receive an ancestor's cancel.
// Owner-worker-only.
HYLE_SCHED_API void unlink_node(CancelNode* n) noexcept;

}  // namespace detail

// ---- CancelToken ------------------------------------------------------------
// A handle to one node of the cancellation tree. Move-only, single-owner (no
// refcount: each derive() makes a NEW node; there is no token cloning/sharing --
// that is what lets the whole tree be lock-free on one shard). Dropping a token
// destroys its node (re-parenting any live children first).
//
// Usage (HTTP fan-out):
//   CancelToken req = CancelToken::root(alloc);        // per-request root
//   CancelToken a = req.derive();                      // child for upstream A
//   CancelToken b = req.derive();                      // child for upstream B
//   ... spawn task bound to a / b, each installs a CancelScope ...
//   req.cancel();   // deadline/hangup -> a and b (and their subtrees) all cancel
class CancelToken {
 public:
  CancelToken() noexcept = default;

  // A root token owns a fresh tree. Returns an invalid token on OOM (valid()==false).
  static CancelToken root(mem::Allocator alloc) noexcept {
    if (!alloc.valid()) return CancelToken{};
    auto* n = static_cast<detail::CancelNode*>(
        alloc.allocate(sizeof(detail::CancelNode), alignof(detail::CancelNode)));
    if (n == nullptr) return CancelToken{};
    new (n) detail::CancelNode{};
    n->alloc = alloc;
    return CancelToken{n};
  }

  // Derive a child token. Cancelling `*this` (or any ancestor) cancels the child.
  // If `*this` is ALREADY cancelled the child is born cancelled (parent-cancels-
  // then-new-child race, handled without a lock since same shard). Uses this
  // node's allocator. Returns an invalid token on OOM.
  CancelToken derive() noexcept {
    if (node_ == nullptr) return CancelToken{};
    auto* c = static_cast<detail::CancelNode*>(node_->alloc.allocate(
        sizeof(detail::CancelNode), alignof(detail::CancelNode)));
    if (c == nullptr) return CancelToken{};
    new (c) detail::CancelNode{};
    c->alloc = node_->alloc;
    c->parent = node_;
    // Push onto parent's child list (front insert, O(1)).
    c->next_sibling = node_->first_child;
    if (node_->first_child != nullptr) node_->first_child->prev_sibling = c;
    node_->first_child = c;
    // Born-cancelled if the parent is already cancelled (must be observed at once).
    if (node_->cancelled.load(std::memory_order_acquire)) {
      c->cancelled.store(true, std::memory_order_release);
    }
    return CancelToken{c};
  }

  CancelToken(CancelToken&& o) noexcept : node_(std::exchange(o.node_, nullptr)) {}
  CancelToken& operator=(CancelToken&& o) noexcept {
    if (this != &o) {
      reset();
      node_ = std::exchange(o.node_, nullptr);
    }
    return *this;
  }
  HYLE_DISABLE_COPY(CancelToken);
  ~CancelToken() { reset(); }

  bool valid() const noexcept { return node_ != nullptr; }

  // Request cancellation of this token's whole subtree. Cooperative: it marks
  // nodes + fires registered callbacks (which actively cancel in-flight io); a
  // running fiber observes it at its next await / is_cancel_requested() check.
  // Owner-worker-only. Idempotent.
  void cancel() noexcept {
    if (node_ != nullptr) detail::cancel_subtree(node_);
  }

  // Has this token been cancelled (by itself or an ancestor)? A plain acquire load
  // (a mov on x86); safe to read from any worker the fiber migrated to.
  bool is_cancelled() const noexcept {
    return node_ != nullptr && node_->cancelled.load(std::memory_order_acquire);
  }

  // The atomic flag this token publishes to is_cancel_requested() via a CancelScope
  // (below). Internal seam -- the flag lives on the node so it survives fiber
  // migration (same contract as sched-10's cancel_req).
  std::atomic<bool>* flag() const noexcept {
    return node_ != nullptr ? &node_->cancelled : nullptr;
  }

  detail::CancelNode* node_for_scope_() const noexcept { return node_; }

 private:
  explicit CancelToken(detail::CancelNode* n) noexcept : node_(n) {}

  // Destroy this token's node: re-parent live children to the grandparent, unlink
  // from the parent's list, then free. (A dropped middle node must not orphan its
  // subtree from ancestor cancellation -- tokio destructor re-parenting.)
  void reset() noexcept {
    if (node_ == nullptr) return;
    detail::unlink_node(node_);  // re-parents children + unlinks from parent
    mem::Allocator a = node_->alloc;
    node_->~CancelNode();
    a.deallocate(node_, sizeof(detail::CancelNode), alignof(detail::CancelNode));
    node_ = nullptr;
  }

  detail::CancelNode* node_ = nullptr;
};

// ---- CancelScope ------------------------------------------------------------
// Publishes a token's cancel flag to the CURRENT fiber so is_cancel_requested()
// (sched-10) sees it, then restores the previous flag on scope exit. This is how
// a task body opts into a token: construct a CancelScope at the top of the task.
// RAII + nested-safe (saves/restores the prior flag, so a child scope inside a
// parent scope works). Reuses the Fiber::cancel_flag_ slot -- no new plumbing, and
// is_cancel_requested() / abort() keep working unchanged.
class CancelScope {
 public:
  explicit CancelScope(const CancelToken& tok) noexcept {
    std::atomic<bool>* flag = tok.flag();
    if (Fiber* f = Fiber::current(); f != nullptr && flag != nullptr) {
      prev_ = f->cancel_flag();
      f->set_cancel_flag(flag);
      active_ = true;
    }
  }
  ~CancelScope() {
    if (!active_) return;
    Fiber* f = Fiber::current();
    HYLE_FATAL_ASSERT(f != nullptr);
    f->set_cancel_flag(prev_);
  }
  HYLE_DISABLE_COPY_AND_MOVE(CancelScope);

 private:
  void* prev_ = nullptr;
  bool active_ = false;
};

// ---- CancelCallback ---------------------------------------------------------
// Register `fn(ctx)` to run when `tok` is cancelled. RAII: the callback is
// unregistered on destruction (so an io op that COMPLETES normally tears down its
// cancel hook). If the token is ALREADY cancelled at construction, `fn` runs
// immediately (register-checks-already-cancelled; else a cancel that already
// happened would be missed forever). Owner-worker-only. The callback body is the
// active-cancel action -- e.g. "submit_cancel this in-flight kernel op".
//
// Non-owning + intrusive: the Callback record lives INSIDE this object (on the
// registrant's frame), so no allocation. Must not outlive `tok`'s node.
class CancelCallback {
 public:
  using Fn = detail::CancelNode::Callback::Fn;

  CancelCallback(const CancelToken& tok, Fn fn, void* ctx) noexcept {
    HYLE_FATAL_ASSERT(fn != nullptr);
    detail::CancelNode* n = tok.node_for_scope_();
    if (n == nullptr) return;
    if (n->cancelled.load(std::memory_order_acquire)) {
      fn(ctx);  // already cancelled: run now, don't register (nothing would fire it)
      return;
    }
    cb_.node = n;
    cb_.fn = fn;
    cb_.ctx = ctx;
    cb_.next = n->first_cb;
    cb_.prev_next = &n->first_cb;
    if (n->first_cb != nullptr) n->first_cb->prev_next = &cb_.next;
    n->first_cb = &cb_;
  }
  ~CancelCallback() { unregister(); }
  HYLE_DISABLE_COPY_AND_MOVE(CancelCallback);

  // Detach without firing (idempotent). Called by the destructor; safe to call
  // during the node's own cancel walk (fire_callbacks detaches before invoking, so
  // by the time a callback body runs it is already off the list).
  void unregister() noexcept {
    if (cb_.node == nullptr) return;
    if (cb_.prev_next != nullptr) {
      *cb_.prev_next = cb_.next;
      if (cb_.next != nullptr) cb_.next->prev_next = cb_.prev_next;
    }
    cb_.node = nullptr;
    cb_.prev_next = nullptr;
    cb_.next = nullptr;
  }

 private:
  detail::CancelNode::Callback cb_{};
};

// ---- await cancel point -----------------------------------------------------
// A running fiber checks this at its OWN checkpoints inside a long CPU loop that
// has no await to piggy-back on (deep pure-CPU work is un-cancellable otherwise --
// the cooperative cost). At an await point cancellation is observed for free (the
// awaitable returns Cancelled); this macro is for the gaps between awaits.
//
//   HYLE_CHECK_CANCEL();  // returns Status::Cancelled from the enclosing fn if set
//
// Requires the enclosing function to return Status / Result<T> (so the early
// return type-checks). Mirrors HYLE_ASSIGN_OR_RAISE's early-return style.
#define HYLE_CHECK_CANCEL()                              \
  do {                                                   \
    if (::hyle::sched::is_cancel_requested())            \
      return ::hyle::Status::Cancelled("cancelled");     \
  } while (0)

}  // namespace hyle::sched
