#pragma once

#include <cstddef>
#include <type_traits>

#include "hyle/macros.h"

namespace hyle::core {

// Lowest-level intrusive linking primitive shared by time/ and container/.
// Keeping it in core avoids the false time -> container module dependency that
// existed when timer_entry.h included hyle/container/intrusive_list.h.
//
// Embed one hook in a stable-address node, then list it via
// IntrusiveList<Node, &Node::hook>. The list owns no nodes and allocates nothing.
struct IntrusiveListHook {
  bool linked() const noexcept { return prev_ != nullptr; }

 private:
  template <class T, IntrusiveListHook T::* Hook>
  friend class IntrusiveList;

  IntrusiveListHook* prev_ = nullptr;
  IntrusiveListHook* next_ = nullptr;
};

static_assert(sizeof(IntrusiveListHook) == 2 * sizeof(void*));

template <class T, IntrusiveListHook T::* Hook>
class IntrusiveList {
 public:
  static_assert(std::is_standard_layout_v<T>,
                "intrusive container_of requires a standard-layout node");

  IntrusiveList() noexcept {
    sentinel_.prev_ = &sentinel_;
    sentinel_.next_ = &sentinel_;
  }
  IntrusiveList(const IntrusiveList&) = delete;
  IntrusiveList& operator=(const IntrusiveList&) = delete;

  bool empty() const noexcept { return sentinel_.next_ == &sentinel_; }

  void push_back(T* node) noexcept {
    HYLE_FATAL_ASSERT(node != nullptr);
    HYLE_FATAL_ASSERT(!hook_of(node)->linked());
    push_back_unchecked(node);
  }
  void push_front(T* node) noexcept {
    HYLE_FATAL_ASSERT(node != nullptr);
    HYLE_FATAL_ASSERT(!hook_of(node)->linked());
    insert_before(sentinel_.next_, hook_of(node));
  }

  // Low-level companion for an owner that has already checked both preconditions
  // as part of a larger invariant. Keeping the link-only core separate avoids
  // repeating branches on production hot paths without weakening their checks.
  void push_back_unchecked(T* node) noexcept {
    insert_before(&sentinel_, hook_of(node));
  }

  T* front() noexcept { return empty() ? nullptr : node_of(sentinel_.next_); }
  const T* front() const noexcept {
    return empty() ? nullptr : node_of(sentinel_.next_);
  }
  T* back() noexcept { return empty() ? nullptr : node_of(sentinel_.prev_); }
  const T* back() const noexcept {
    return empty() ? nullptr : node_of(sentinel_.prev_);
  }

  T* pop_front() noexcept {
    if (empty()) return nullptr;
    IntrusiveListHook* h = sentinel_.next_;
    // The predecessor is known to be sentinel_. Avoid the generic unlink()
    // load of h->prev_; this is the dominant drain path for timer ready lists.
    IntrusiveListHook* next = h->next_;
    sentinel_.next_ = next;
    next->prev_ = &sentinel_;
    h->prev_ = nullptr;
    h->next_ = nullptr;
    return node_of(h);
  }
  T* pop_back() noexcept {
    if (empty()) return nullptr;
    IntrusiveListHook* h = sentinel_.prev_;
    IntrusiveListHook* prev = h->prev_;
    sentinel_.prev_ = prev;
    prev->next_ = &sentinel_;
    h->prev_ = nullptr;
    h->next_ = nullptr;
    return node_of(h);
  }

  // Detach and consume the complete list without updating the sentinel for
  // every node. The current node is fully unlinked before fn runs, so fn may
  // destroy or relink it. This is the preferred ready-list drain path.
  template <class Fn>
  std::size_t consume_all(Fn&& fn) noexcept {
    IntrusiveListHook* current = sentinel_.next_;
    if (current == &sentinel_) return 0;

    sentinel_.prev_ = &sentinel_;
    sentinel_.next_ = &sentinel_;
    std::size_t count = 0;
    while (current != &sentinel_) {
      IntrusiveListHook* next = current->next_;
      current->prev_ = nullptr;
      current->next_ = nullptr;
      fn(node_of(current));
      current = next;
      ++count;
    }
    return count;
  }

  // Transfer one node without clearing and rebuilding its links.
  T* move_front_to(IntrusiveList& dst) noexcept {
    HYLE_FATAL_ASSERT(&dst != this);
    if (empty()) return nullptr;
    IntrusiveListHook* h = sentinel_.next_;
    h->next_->prev_ = &sentinel_;
    sentinel_.next_ = h->next_;
    insert_before(&dst.sentinel_, h);
    return node_of(h);
  }

  // Concatenate `source` onto this list in O(1). Nodes stay linked throughout;
  // neither list owns them. This is the radix/timing-wheel bucket-splice
  // primitive and is also useful for batching completion queues.
  void append(IntrusiveList& source) noexcept {
    HYLE_FATAL_ASSERT(&source != this);
    if (source.empty()) return;

    IntrusiveListHook* first = source.sentinel_.next_;
    IntrusiveListHook* last = source.sentinel_.prev_;
    source.sentinel_.prev_ = &source.sentinel_;
    source.sentinel_.next_ = &source.sentinel_;

    IntrusiveListHook* dst_last = sentinel_.prev_;
    dst_last->next_ = first;
    first->prev_ = dst_last;
    last->next_ = &sentinel_;
    sentinel_.prev_ = last;
  }

  // Visit linked nodes without changing membership. `fn` must not unlink,
  // destroy or relink the current node.
  template <class Fn>
  void for_each(Fn&& fn) noexcept {
    IntrusiveListHook* current = sentinel_.next_;
    while (current != &sentinel_) {
      fn(node_of(current));
      current = current->next_;
    }
  }
  template <class Fn>
  void for_each(Fn&& fn) const noexcept {
    const IntrusiveListHook* current = sentinel_.next_;
    while (current != &sentinel_) {
      fn(node_of(current));
      current = current->next_;
    }
  }

  // Move the longest front prefix satisfying pred to dst's tail. Scanning and
  // on_move are O(moved); relinking the complete range is O(1).
  template <class Pred, class OnMove>
  std::size_t move_prefix_to(IntrusiveList& dst, Pred&& pred,
                             OnMove&& on_move) noexcept {
    HYLE_FATAL_ASSERT(&dst != this);
    IntrusiveListHook* first = sentinel_.next_;
    IntrusiveListHook* after = first;
    std::size_t count = 0;
    while (after != &sentinel_) {
      T* node = node_of(after);
      if (!pred(node)) break;
      on_move(node);
      after = after->next_;
      ++count;
    }
    if (count == 0) return 0;

    IntrusiveListHook* last = after->prev_;
    sentinel_.next_ = after;
    after->prev_ = &sentinel_;

    IntrusiveListHook* dst_last = dst.sentinel_.prev_;
    dst_last->next_ = first;
    first->prev_ = dst_last;
    last->next_ = &dst.sentinel_;
    dst.sentinel_.prev_ = last;
    return count;
  }

  void remove(T* node) noexcept {
    HYLE_FATAL_ASSERT(node != nullptr);
    HYLE_FATAL_ASSERT(hook_of(node)->linked());
    remove_unchecked(node);
  }

  // Low-level companion for owners whose membership invariant already proves
  // the node is linked in this list. This keeps redundant contract branches out
  // of timer cancel hot paths; public data structures still validate ownership
  // before calling it.
  void remove_unchecked(T* node) noexcept {
    unlink(hook_of(node));
  }

 private:
  static IntrusiveListHook* hook_of(T* node) noexcept { return &(node->*Hook); }

  static T* node_of(IntrusiveListHook* h) noexcept {
    const auto off = reinterpret_cast<std::size_t>(
        &(reinterpret_cast<T*>(0)->*Hook));
    return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(h) - off);
  }
  static const T* node_of(const IntrusiveListHook* h) noexcept {
    return node_of(const_cast<IntrusiveListHook*>(h));
  }

  static void insert_before(IntrusiveListHook* pos,
                            IntrusiveListHook* node) noexcept {
    node->prev_ = pos->prev_;
    node->next_ = pos;
    pos->prev_->next_ = node;
    pos->prev_ = node;
  }

  static void unlink(IntrusiveListHook* h) noexcept {
    h->prev_->next_ = h->next_;
    h->next_->prev_ = h->prev_;
    h->prev_ = nullptr;
    h->next_ = nullptr;
  }

  IntrusiveListHook sentinel_;
};

}  // namespace hyle::core
