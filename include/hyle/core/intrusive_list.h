#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
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

private:
  template <bool IsConst> class BasicIterator {
    friend class IntrusiveList;
    template <bool> friend class BasicIterator;

    using HookPointer =
        std::conditional_t<IsConst, const IntrusiveListHook *,
                           IntrusiveListHook *>;

    explicit BasicIterator(HookPointer hook) noexcept : hook_(hook) {}

  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = std::bidirectional_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const T *, T *>;
    using reference = std::conditional_t<IsConst, const T &, T &>;

    BasicIterator() noexcept = default;
    template <bool OtherConst>
      requires(IsConst && !OtherConst)
    BasicIterator(const BasicIterator<OtherConst> &other) noexcept
        : hook_(other.hook_) {}

    reference operator*() const noexcept {
      return *IntrusiveList::node_of(hook_);
    }
    pointer operator->() const noexcept {
      return IntrusiveList::node_of(hook_);
    }

    BasicIterator &operator++() noexcept {
      hook_ = hook_->next_;
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }
    BasicIterator &operator--() noexcept {
      hook_ = hook_->prev_;
      return *this;
    }
    BasicIterator operator--(int) noexcept {
      BasicIterator copy = *this;
      --*this;
      return copy;
    }

    template <bool Other>
    bool operator==(const BasicIterator<Other> &right) const noexcept {
      return hook_ == right.hook_;
    }

  private:
    HookPointer hook_ = nullptr;
  };

public:
  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

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

  iterator begin() noexcept { return iterator(sentinel_.next_); }
  const_iterator begin() const noexcept {
    return const_iterator(sentinel_.next_);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  iterator end() noexcept { return iterator(&sentinel_); }
  const_iterator end() const noexcept { return const_iterator(&sentinel_); }
  const_iterator cend() const noexcept { return end(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const noexcept { return rbegin(); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const noexcept { return rend(); }

  // Preconditions: node is non-null and its selected hook is not linked.
  void push_back(T *node) noexcept { insert_before(&sentinel_, hook_of(node)); }
  void push_front(T *node) noexcept {
    insert_before(sentinel_.next_, hook_of(node));
  }

  HYLE_NODISCARD bool try_push_back(T *node) noexcept {
    if (node == nullptr || hook_of(node)->linked())
      return false;
    push_back(node);
    return true;
  }
  HYLE_NODISCARD bool try_push_front(T *node) noexcept {
    if (node == nullptr || hook_of(node)->linked())
      return false;
    push_front(node);
    return true;
  }

  // Insert before pos. Preconditions: pos belongs to this list, node is
  // non-null and its selected hook is not linked.
  iterator insert(const_iterator pos, T *node) noexcept {
    IntrusiveListHook *hook = hook_of(node);
    insert_before(mutable_hook(pos), hook);
    return iterator(hook);
  }

  HYLE_NODISCARD bool try_insert(const_iterator pos, T *node) noexcept {
    if (node == nullptr || hook_of(node)->linked())
      return false;
    (void)insert(pos, node);
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

  HYLE_NODISCARD T *pop_front() noexcept {
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
  HYLE_NODISCARD T *pop_back() noexcept {
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

  // Erase pos and return its successor. Preconditions: pos is dereferenceable
  // and belongs to this list.
  iterator erase(const_iterator pos) noexcept {
    const_iterator next = pos;
    ++next;
    unlink_unchecked(mutable_hook(pos));
    return iterator(mutable_hook(next));
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

  // Transfer all of source before pos in O(1). Nodes stay linked throughout;
  // neither list owns them. Preconditions: pos belongs to this list.
  void splice(const_iterator pos, IntrusiveList &source) noexcept {
    HYLE_ASSUME(&source != this);
    if (source.empty())
      return;

    IntrusiveListHook *first = source.sentinel_.next_;
    IntrusiveListHook *last = source.sentinel_.prev_;
    source.sentinel_.prev_ = &source.sentinel_;
    source.sentinel_.next_ = &source.sentinel_;

    IntrusiveListHook *position = mutable_hook(pos);
    IntrusiveListHook *before = position->prev_;
    before->next_ = first;
    first->prev_ = before;
    last->next_ = position;
    position->prev_ = last;
  }

  // Transfer item from source before pos in O(1), matching the conventional
  // intrusive-list splice shape. Iterators to the node remain valid. The same
  // list is allowed; moving immediately before itself/its successor is a no-op.
  // Preconditions: pos belongs to this list; item is dereferenceable and
  // belongs to source.
  iterator splice(const_iterator pos, IntrusiveList &source,
                  const_iterator item) noexcept {
    HYLE_ASSUME(item.hook_ != &source.sentinel_);
    if (pos.hook_ == item.hook_ || pos.hook_ == item.hook_->next_)
      return iterator(mutable_hook(item));
    IntrusiveListHook *hook = mutable_hook(item);
    unlink_unchecked(hook);
    insert_before(mutable_hook(pos), hook);
    return iterator(hook);
  }

  // Concatenate source onto this list in O(1). This is the radix/timing-wheel
  // bucket-splice primitive and is also useful for completion batches.
  void append(IntrusiveList &source) noexcept {
    HYLE_ASSUME(&source != this);
    if (source.empty())
      return;

    IntrusiveListHook *first = source.sentinel_.next_;
    IntrusiveListHook *last = source.sentinel_.prev_;
    source.sentinel_.prev_ = &source.sentinel_;
    source.sentinel_.next_ = &source.sentinel_;

    IntrusiveListHook *destination_last = sentinel_.prev_;
    destination_last->next_ = first;
    first->prev_ = destination_last;
    last->next_ = &sentinel_;
    sentinel_.prev_ = last;
  }

  void swap(IntrusiveList &other) noexcept {
    if (this == &other)
      return;
    IntrusiveList temporary;
    temporary.append(*this);
    append(other);
    other.append(temporary);
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
    HYLE_ASSUME(&dst != this);
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
  static IntrusiveListHook *mutable_hook(const_iterator position) noexcept {
    return const_cast<IntrusiveListHook *>(position.hook_);
  }

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
