#include "hyle/sched/cancel.h"

#include <atomic>

// sched-07 cancellation-tree operations. Owner-worker-only: a whole tree lives on
// one shard (shard-per-core), so every walk here is single-threaded and touches
// the plain (non-atomic) tree/callback links directly. The only atomic is the
// per-node `cancelled` flag (so is_cancel_requested() reads it correctly after a
// task fiber migrates workers between awaits -- see cancel.h header).
//
// INVARIANT (relied on by the walk): cancelling a node cancels every descendant.
// A node's `cancelled` becomes true only via cancel_subtree (which cascades to the
// whole subtree) or born-cancelled at derive() (a leaf with no children yet). So a
// node already cancelled implies its entire current subtree is cancelled -- the
// walk can prune it.

namespace hyle::sched {
namespace detail {

void fire_callbacks(CancelNode* n) noexcept {
  // Pop-then-invoke: detach the HEAD callback, then run it -- repeat until the list
  // is empty. Popping one at a time (rather than caching next before invoking) keeps
  // the walk correct even if a callback body unregisters ANOTHER still-pending
  // callback: that unregister splices the live list via prev_next, and we always
  // re-read the current head, so no stale/dangling pointer is ever dereferenced (the
  // folly unlink-before-invoke discipline, made robust to sibling teardown). A body
  // that registers a NEW callback pushes onto the head and it, too, will fire.
  // Single-threaded (owner worker), so no atomics; invoking_cb documents re-entry.
  n->invoking_cb = true;
  while (n->first_cb != nullptr) {
    CancelNode::Callback* cb = n->first_cb;
    // Unlink the head from the live list BEFORE invoking (O(1) via prev_next).
    n->first_cb = cb->next;
    if (cb->next != nullptr) cb->next->prev_next = &n->first_cb;
    cb->node = nullptr;  // mark detached so its owner's unregister() is a no-op
    cb->prev_next = nullptr;
    cb->next = nullptr;
    cb->fn(cb->ctx);  // active-cancel action (e.g. submit_cancel the in-flight op)
  }
  n->invoking_cb = false;
}

void cancel_subtree(CancelNode* root) noexcept {
  if (root == nullptr) return;
  // Mark the root; if it was already cancelled, its subtree is too (invariant) -> done.
  if (root->cancelled.exchange(true, std::memory_order_acq_rel)) return;
  fire_callbacks(root);

  // Iterative depth-first walk over the tree's own first_child/next_sibling links
  // (NO recursion, NO allocation: a deep fan-out tree must not blow the fiber
  // stack). Descend only into NEWLY-cancelled nodes; an already-cancelled node's
  // whole subtree is already cancelled (invariant), so prune it. Climb back via
  // parent, stopping at `root` (never touch root's own siblings/parent).
  CancelNode* cur = root->first_child;
  while (cur != nullptr && cur != root) {
    bool was = cur->cancelled.exchange(true, std::memory_order_acq_rel);
    if (!was) fire_callbacks(cur);
    if (!was && cur->first_child != nullptr) {
      cur = cur->first_child;  // descend into the newly-cancelled node
    } else if (cur->next_sibling != nullptr) {
      cur = cur->next_sibling;  // across to the next sibling
    } else {
      // Climb to the nearest ancestor (below root) that has an unvisited sibling.
      while (cur != nullptr && cur != root && cur->next_sibling == nullptr) {
        cur = cur->parent;
      }
      if (cur == nullptr || cur == root) break;  // whole subtree walked
      cur = cur->next_sibling;
    }
  }
}

void unlink_node(CancelNode* n) noexcept {
  // Re-parent this node's children to its parent BEFORE removing it, so a dropped
  // middle node never orphans its subtree from an ancestor's future cancel (tokio
  // move_children_to_parent). A root drop (parent == nullptr) leaves its children
  // parentless -- they simply become roots (their own drops free them).
  CancelNode* p = n->parent;
  CancelNode* child = n->first_child;
  if (child != nullptr) {
    // Point every direct child at p; remember the tail to splice the span.
    CancelNode* tail = child;
    for (;;) {
      tail->parent = p;
      if (tail->next_sibling == nullptr) break;
      tail = tail->next_sibling;
    }
    if (p != nullptr) {
      // Splice n's children onto the FRONT of p's child list.
      tail->next_sibling = p->first_child;
      if (p->first_child != nullptr) p->first_child->prev_sibling = tail;
      p->first_child = child;
      child->prev_sibling = nullptr;
      // If p is already cancelled, the freshly re-parented children missed the
      // original cascade -- cancel each re-parented subtree now. Walk the original
      // child span [child .. tail] (tail->next_sibling now points into p's old list,
      // so stop AT tail).
      if (p->cancelled.load(std::memory_order_acquire)) {
        for (CancelNode* c = child;;) {
          CancelNode* nxt = (c == tail) ? nullptr : c->next_sibling;
          cancel_subtree(c);
          if (nxt == nullptr) break;
          c = nxt;
        }
      }
    } else {
      child->prev_sibling = nullptr;  // no grandparent: children become roots
    }
  }
  // Unlink n from p's child list (O(1) via the sibling doubly-linked list).
  if (n->prev_sibling != nullptr) {
    n->prev_sibling->next_sibling = n->next_sibling;
  } else if (p != nullptr && p->first_child == n) {
    p->first_child = n->next_sibling;
  }
  if (n->next_sibling != nullptr) n->next_sibling->prev_sibling = n->prev_sibling;
  n->parent = nullptr;
  n->first_child = nullptr;
  n->next_sibling = nullptr;
  n->prev_sibling = nullptr;
}

}  // namespace detail
}  // namespace hyle::sched
