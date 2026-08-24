#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "hyle/macros.h"

namespace hyle::core {

// Lowest-level intrusive linking primitive shared by time/ and container/.
// Keeping it in core avoids the false time -> container module dependency that
// existed when timer_entry.h included hyle/container/intrusive_list.h.
//
// Embed one hook in a stable-address node, then list it via
// IntrusiveList<Node, &Node::hook>. The list owns no nodes and allocates
// nothing. Like folly::IntrusiveList's auto-unlink hook, destroying a node
// detaches it. Construction by copy/move always creates an unlinked hook;
// assignment deliberately preserves the destination hook's membership and
// never imports the source hook's membership.
struct IntrusiveListHook {
  IntrusiveListHook() noexcept = default;
  IntrusiveListHook(const IntrusiveListHook &) noexcept {}
  IntrusiveListHook &operator=(const IntrusiveListHook &) noexcept {
    return *this;
  }
  IntrusiveListHook(IntrusiveListHook &&) noexcept {}
  IntrusiveListHook &operator=(IntrusiveListHook &&) noexcept { return *this; }
  ~IntrusiveListHook() { unlink(); }

  bool linked() const noexcept { return prev_ != nullptr; }
  void unlink() noexcept {
    if (!linked())
      return;
    IntrusiveListHook *previous = prev_;
    IntrusiveListHook *next = next_;
    previous->next_ = next;
    next->prev_ = previous;
    next_ = nullptr;
    prev_ = nullptr;
  }

private:
  template <class T, IntrusiveListHook T::*Hook> friend class IntrusiveList;

  IntrusiveListHook *prev_ = nullptr;
  IntrusiveListHook *next_ = nullptr;
};

static_assert(sizeof(IntrusiveListHook) == 2 * sizeof(void *));

template <class T, IntrusiveListHook T::*Hook> class IntrusiveList {
public:
  static_assert(std::is_standard_layout_v<T>,
                "intrusive container_of requires a standard-layout node");

  IntrusiveList() noexcept {
    sentinel_.next_ = &sentinel_;
    sentinel_.prev_ = &sentinel_;
  }
  IntrusiveList(const IntrusiveList &) = delete;
  IntrusiveList &operator=(const IntrusiveList &) = delete;
  IntrusiveList(IntrusiveList &&other) noexcept : IntrusiveList() {
    append(other);
  }
  IntrusiveList &operator=(IntrusiveList &&other) noexcept {
    if (this != &other) {
      clear();
      append(other);
    }
    return *this;
  }
  ~IntrusiveList() { clear(); }

  bool empty() const noexcept { return sentinel_.next_ == &sentinel_; }

  // Preconditions: node is non-null and its selected hook is not linked.
  void push_back(T *node) noexcept { insert_before(&sentinel_, hook_of(node)); }
  void push_front(T *node) noexcept {
    insert_before(sentinel_.next_, hook_of(node));
  }

  bool try_push_back(T *node) noexcept {
    if (node == nullptr || hook_of(node)->linked())
      return false;
    push_back(node);
    return true;
  }
  bool try_push_front(T *node) noexcept {
    if (node == nullptr || hook_of(node)->linked())
      return false;
    push_front(node);
    return true;
  }

  T *front() noexcept { return empty() ? nullptr : node_of(sentinel_.next_); }
  const T *front() const noexcept {
    return empty() ? nullptr : node_of(sentinel_.next_);
  }
  T *back() noexcept { return empty() ? nullptr : node_of(sentinel_.prev_); }
  const T *back() const noexcept {
    return empty() ? nullptr : node_of(sentinel_.prev_);
  }

  T *pop_front() noexcept {
    if (empty())
      return nullptr;
    IntrusiveListHook *h = sentinel_.next_;
    // The predecessor is known to be sentinel_. Avoid the generic unlink()
    // load of h->prev_; this is the dominant drain path for timer ready lists.
    IntrusiveListHook *next = h->next_;
    sentinel_.next_ = next;
    next->prev_ = &sentinel_;
    h->prev_ = nullptr;
    h->next_ = nullptr;
    return node_of(h);
  }
  T *pop_back() noexcept {
    if (empty())
      return nullptr;
    IntrusiveListHook *h = sentinel_.prev_;
    IntrusiveListHook *prev = h->prev_;
    sentinel_.prev_ = prev;
    prev->next_ = &sentinel_;
    h->prev_ = nullptr;
    h->next_ = nullptr;
    return node_of(h);
  }

  // Detach all nodes. The list never destroys them because it owns no storage.
  void clear() noexcept {
    IntrusiveListHook *current = sentinel_.next_;
    while (current != &sentinel_) {
      IntrusiveListHook *next = current->next_;
      current->next_ = nullptr;
      current->prev_ = nullptr;
      current = next;
    }
    sentinel_.next_ = &sentinel_;
    sentinel_.prev_ = &sentinel_;
  }

  // Detach and consume the complete list without updating the sentinel for
  // every node. The current node is fully unlinked before fn runs, so fn may
  // destroy or relink it. Because the sentinel is detached up front, an
  // exception could strand the unvisited chain; require a non-throwing
  // callback. This is the preferred ready-list drain path.
  template <class Fn>
    requires std::is_nothrow_invocable_v<Fn &, T *>
  std::size_t consume_all(Fn &&fn) noexcept {
    IntrusiveListHook *current = sentinel_.next_;
    if (current == &sentinel_)
      return 0;

    sentinel_.prev_ = &sentinel_;
    sentinel_.next_ = &sentinel_;
    std::size_t count = 0;
    while (current != &sentinel_) {
      IntrusiveListHook *next = current->next_;
      current->prev_ = nullptr;
      current->next_ = nullptr;
      fn(node_of(current));
      current = next;
      ++count;
    }
    return count;
  }

  // Transfer one node without clearing and rebuilding its links.
  T *move_front_to(IntrusiveList &dst) noexcept {
    HYLE_FATAL_ASSERT(&dst != this);
    if (empty())
      return nullptr;
    IntrusiveListHook *h = sentinel_.next_;
    h->next_->prev_ = &sentinel_;
    sentinel_.next_ = h->next_;
    insert_before(&dst.sentinel_, h);
    return node_of(h);
  }

  // Concatenate `source` onto this list in O(1). Nodes stay linked throughout;
  // neither list owns them. This is the radix/timing-wheel bucket-splice
  // primitive and is also useful for batching completion queues.
  void append(IntrusiveList &source) noexcept {
    HYLE_FATAL_ASSERT(&source != this);
    if (source.empty())
      return;

    IntrusiveListHook *first = source.sentinel_.next_;
    IntrusiveListHook *last = source.sentinel_.prev_;
    source.sentinel_.prev_ = &source.sentinel_;
    source.sentinel_.next_ = &source.sentinel_;

    IntrusiveListHook *dst_last = sentinel_.prev_;
    dst_last->next_ = first;
    first->prev_ = dst_last;
    last->next_ = &sentinel_;
    sentinel_.prev_ = last;
  }

  // Visit linked nodes without changing membership or lifetime. If `fn`
  // throws, traversal stops and the unchanged list remains valid.
  template <class Fn>
  void for_each(Fn &&fn) noexcept(std::is_nothrow_invocable_v<Fn &, T *>) {
    IntrusiveListHook *current = sentinel_.next_;
    while (current != &sentinel_) {
      fn(node_of(current));
      current = current->next_;
    }
  }
  template <class Fn>
  void for_each(Fn &&fn) const
      noexcept(std::is_nothrow_invocable_v<Fn &, const T *>) {
    const IntrusiveListHook *current = sentinel_.next_;
    while (current != &sentinel_) {
      fn(node_of(current));
      current = current->next_;
    }
  }

  // Move the longest front prefix satisfying pred to dst's tail. During the
  // scan both callbacks observe nodes still linked in this list; neither may
  // change any involved node's membership or lifetime. Requiring non-throwing
  // callbacks makes the range splice an all-or-nothing structural step.
  // Scanning and on_move are O(moved); relinking the complete range is O(1).
  template <class Pred, class OnMove>
    requires(std::is_nothrow_invocable_r_v<bool, Pred &, T *> &&
             std::is_nothrow_invocable_v<OnMove &, T *>)
  std::size_t move_prefix_to(IntrusiveList &dst, Pred &&pred,
                             OnMove &&on_move) noexcept {
    HYLE_FATAL_ASSERT(&dst != this);
    IntrusiveListHook *first = sentinel_.next_;
    IntrusiveListHook *after = first;
    std::size_t count = 0;
    while (after != &sentinel_) {
      T *node = node_of(after);
      if (!pred(node))
        break;
      on_move(node);
      after = after->next_;
      ++count;
    }
    if (count == 0)
      return 0;

    IntrusiveListHook *last = after->prev_;
    sentinel_.next_ = after;
    after->prev_ = &sentinel_;

    IntrusiveListHook *dst_last = dst.sentinel_.prev_;
    dst_last->next_ = first;
    first->prev_ = dst_last;
    last->next_ = &dst.sentinel_;
    dst.sentinel_.prev_ = last;
    return count;
  }

  // Preconditions: node is non-null and belongs to this list. Matching mature
  // intrusive containers, erase/remove does not store or check an owner on the
  // hot path. Use node's idempotent hook.unlink() when membership is optional.
  void remove(T *node) noexcept { unlink_unchecked(hook_of(node)); }

private:
  static IntrusiveListHook *hook_of(T *node) noexcept { return &(node->*Hook); }

  // C++ has no standard inverse for an arbitrary pointer-to-data-member.
  // HYLE's supported compilers use one of the two documented data-member
  // pointer ABIs below for standard-layout classes. Decode the representation
  // without the traditional null-pointer member expression, which is UB even
  // for a standard-layout T. The size assertions fail closed if a compiler
  // selects a richer inheritance representation.
  static std::ptrdiff_t hook_offset() noexcept {
#if defined(_MSC_VER)
    static_assert(sizeof(Hook) == sizeof(int),
                  "unsupported MSVC data-member pointer representation");
    return static_cast<std::ptrdiff_t>(std::bit_cast<int>(Hook));
#elif defined(__GNUC__) || defined(__clang__)
    static_assert(sizeof(Hook) == sizeof(std::ptrdiff_t),
                  "unsupported Itanium data-member pointer representation");
    return std::bit_cast<std::ptrdiff_t>(Hook);
#else
#error "IntrusiveList requires a supported data-member pointer ABI"
#endif
  }

  static T *node_of(IntrusiveListHook *h) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(h) -
                         static_cast<std::uintptr_t>(hook_offset());
    return std::launder(reinterpret_cast<T *>(address));
  }
  static const T *node_of(const IntrusiveListHook *h) noexcept {
    return node_of(const_cast<IntrusiveListHook *>(h));
  }

  static void insert_before(IntrusiveListHook *pos,
                            IntrusiveListHook *node) noexcept {
    node->prev_ = pos->prev_;
    node->next_ = pos;
    pos->prev_->next_ = node;
    pos->prev_ = node;
  }

  static void unlink_unchecked(IntrusiveListHook *hook) noexcept {
    IntrusiveListHook *previous = hook->prev_;
    IntrusiveListHook *next = hook->next_;
    previous->next_ = next;
    next->prev_ = previous;
    hook->next_ = nullptr;
    hook->prev_ = nullptr;
  }

  IntrusiveListHook sentinel_;
};

} // namespace hyle::core
