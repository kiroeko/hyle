#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"

namespace hyle::container {

namespace detail {

// The sentinel is structural state, not a user-visible hook. Keeping the raw
// link separate prevents a safe-link sentinel from looking permanently linked
// to itself at destruction. Both public hooks contain exactly one raw link.
struct IntrusiveListLink {
  IntrusiveListLink *prev = nullptr;
  IntrusiveListLink *next = nullptr;
};

template <class T, auto Hook, class PublicList> class IntrusiveListImpl;

} // namespace detail

// Allocation-free intrusive linking primitive shared by time and the full
// container module.  It is its own lightweight component so low-level users do
// not acquire the owning container/mem/platform dependency closure.
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

  bool linked() const noexcept { return link_.prev != nullptr; }
  void unlink() noexcept {
    if (!linked())
      return;
    detail::IntrusiveListLink *previous = link_.prev;
    detail::IntrusiveListLink *next = link_.next;
    previous->next = next;
    next->prev = previous;
    link_.next = nullptr;
    link_.prev = nullptr;
  }

private:
  template <class, auto, class> friend class detail::IntrusiveListImpl;

  detail::IntrusiveListLink link_;
};

static_assert(sizeof(IntrusiveListHook) == 2 * sizeof(void *));
static_assert(std::is_standard_layout_v<IntrusiveListHook>);

// Safe-link policy for lifetimes that must be explicitly coordinated with a
// list. Destroying a linked hook is diagnosed in Debug and is a violated
// optimizer precondition in Release; it never mutates neighboring nodes from
// a node destructor. Deliberately no arbitrary unlink() is exposed: removal
// must go through SafeIntrusiveList so future list invariants remain local.
struct SafeIntrusiveListHook {
  SafeIntrusiveListHook() noexcept = default;
  SafeIntrusiveListHook(const SafeIntrusiveListHook &) noexcept {}
  SafeIntrusiveListHook &operator=(const SafeIntrusiveListHook &) noexcept {
    return *this;
  }
  SafeIntrusiveListHook(SafeIntrusiveListHook &&) noexcept {}
  SafeIntrusiveListHook &operator=(SafeIntrusiveListHook &&) noexcept {
    return *this;
  }
  ~SafeIntrusiveListHook() {
    HYLE_ASSUME(link_.prev == nullptr && link_.next == nullptr);
  }

  bool linked() const noexcept { return link_.prev != nullptr; }

private:
  template <class, auto, class> friend class detail::IntrusiveListImpl;

  detail::IntrusiveListLink link_;
};

static_assert(sizeof(SafeIntrusiveListHook) == 2 * sizeof(void *));
static_assert(std::is_standard_layout_v<SafeIntrusiveListHook>);

template <class T, IntrusiveListHook T::*Hook> class IntrusiveList;
template <class T, SafeIntrusiveListHook T::*Hook> class SafeIntrusiveList;

namespace detail {

template <class T, auto Hook, class PublicList> class IntrusiveListImpl {
public:
  using HookType =
      std::remove_reference_t<decltype(std::declval<T &>().*Hook)>;
  static_assert(std::is_standard_layout_v<T>,
                "intrusive container_of requires a standard-layout node");
  static_assert(std::is_same_v<HookType, IntrusiveListHook> ||
                    std::is_same_v<HookType, SafeIntrusiveListHook>,
                "unsupported intrusive-list hook policy");
  // clang-cl with MSVC's standard library does not consistently expose
  // is_pointer_interconvertible_with_class. Both approved hooks are
  // standard-layout, so an explicit first-member offset is the portable
  // fail-closed condition needed by node_of().
  static_assert(offsetof(HookType, link_) == 0,
                "intrusive raw link must remain the hook's first member");

private:
  template <bool IsConst> class BasicIterator {
    friend class IntrusiveListImpl;
    template <bool> friend class BasicIterator;

    using HookPointer = std::conditional_t<IsConst, const IntrusiveListLink *,
                                           IntrusiveListLink *>;

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
      return *IntrusiveListImpl::node_of(hook_);
    }
    pointer operator->() const noexcept {
      return IntrusiveListImpl::node_of(hook_);
    }

    BasicIterator &operator++() noexcept {
      hook_ = hook_->next;
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }
    BasicIterator &operator--() noexcept {
      hook_ = hook_->prev;
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
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  bool empty() const noexcept { return sentinel_.next == &sentinel_; }

  iterator begin() noexcept { return iterator(sentinel_.next); }
  const_iterator begin() const noexcept {
    return const_iterator(sentinel_.next);
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

  // Recover an iterator directly from a known node in O(1). Preconditions:
  // node is non-null, its selected hook is linked, and it belongs to this list.
  // No owner metadata is stored, so this is intentionally an unchecked hot-path
  // operation just like erase/remove.
  iterator iterator_to(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    IntrusiveListLink *hook = hook_of(node);
    HYLE_ASSUME(hook->prev != nullptr);
    return iterator(hook);
  }
  const_iterator iterator_to(const T *node) const noexcept {
    HYLE_ASSUME(node != nullptr);
    const IntrusiveListLink *hook = hook_of(node);
    HYLE_ASSUME(hook->prev != nullptr);
    return const_iterator(hook);
  }

  // Preconditions: node is non-null and its selected hook is not linked.
  void push_back(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    IntrusiveListLink *hook = hook_of(node);
    HYLE_ASSUME(hook->prev == nullptr);
    insert_before(&sentinel_, hook);
  }
  void push_front(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    IntrusiveListLink *hook = hook_of(node);
    HYLE_ASSUME(hook->prev == nullptr);
    insert_before(sentinel_.next, hook);
  }

  [[nodiscard]] bool try_push_back(T *node) noexcept {
    if (node == nullptr || hook_of(node)->prev != nullptr)
      return false;
    push_back(node);
    return true;
  }
  [[nodiscard]] bool try_push_front(T *node) noexcept {
    if (node == nullptr || hook_of(node)->prev != nullptr)
      return false;
    push_front(node);
    return true;
  }

  // Insert before pos. Preconditions: pos belongs to this list, node is
  // non-null and its selected hook is not linked.
  iterator insert(const_iterator pos, T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    IntrusiveListLink *hook = hook_of(node);
    HYLE_ASSUME(hook->prev == nullptr);
    insert_before(mutable_hook(pos), hook);
    return iterator(hook);
  }

  [[nodiscard]] bool try_insert(const_iterator pos, T *node) noexcept {
    if (node == nullptr || hook_of(node)->prev != nullptr)
      return false;
    (void)insert(pos, node);
    return true;
  }

  T *front() noexcept { return empty() ? nullptr : node_of(sentinel_.next); }
  const T *front() const noexcept {
    return empty() ? nullptr : node_of(sentinel_.next);
  }
  T *back() noexcept { return empty() ? nullptr : node_of(sentinel_.prev); }
  const T *back() const noexcept {
    return empty() ? nullptr : node_of(sentinel_.prev);
  }

  [[nodiscard]] T *pop_front() noexcept {
    IntrusiveListLink *h = sentinel_.next;
    if (h == &sentinel_)
      return nullptr;
    // The predecessor is known to be sentinel_. Avoid the generic unlink()
    // load of h->prev; this is the dominant drain path for timer ready lists.
    IntrusiveListLink *next = h->next;
    sentinel_.next = next;
    next->prev = &sentinel_;
    h->prev = nullptr;
    h->next = nullptr;
    return node_of(h);
  }
  [[nodiscard]] T *pop_back() noexcept {
    IntrusiveListLink *h = sentinel_.prev;
    if (h == &sentinel_)
      return nullptr;
    IntrusiveListLink *prev = h->prev;
    sentinel_.prev = prev;
    prev->next = &sentinel_;
    h->prev = nullptr;
    h->next = nullptr;
    return node_of(h);
  }

  // Erase pos and return its successor. Preconditions: pos is dereferenceable
  // and belongs to this list.
  iterator erase(const_iterator pos) noexcept {
    HYLE_ASSUME(pos.hook_ != &sentinel_);
    const_iterator next = pos;
    ++next;
    unlink_unchecked(mutable_hook(pos));
    return iterator(mutable_hook(next));
  }

  // Detach all nodes. The list never destroys them because it owns no storage.
  void clear() noexcept {
    IntrusiveListLink *current = sentinel_.next;
    while (current != &sentinel_) {
      IntrusiveListLink *next = current->next;
      current->next = nullptr;
      current->prev = nullptr;
      current = next;
    }
    sentinel_.next = &sentinel_;
    sentinel_.prev = &sentinel_;
  }

  // Detach and consume the complete list without updating the sentinel for
  // every node. The current node is fully unlinked before fn runs, so fn may
  // destroy or relink it. Because the sentinel is detached up front, an
  // exception could strand the unvisited chain; require a non-throwing
  // callback. This is the preferred ready-list drain path.
  template <class Fn>
    requires std::is_nothrow_invocable_v<Fn &, T *>
  std::size_t consume_all(Fn &&fn) noexcept {
    IntrusiveListLink *current = sentinel_.next;
    if (current == &sentinel_)
      return 0;

    sentinel_.prev = &sentinel_;
    sentinel_.next = &sentinel_;
    std::size_t count = 0;
    while (current != &sentinel_) {
      IntrusiveListLink *next = current->next;
      current->prev = nullptr;
      current->next = nullptr;
      std::invoke(fn, node_of(current));
      current = next;
      ++count;
    }
    return count;
  }

  // Transfer all of source before pos in O(1). Nodes stay linked throughout;
  // neither list owns them. Preconditions: pos belongs to this list.
  void splice(const_iterator pos, PublicList &source) noexcept {
    IntrusiveListImpl &source_impl = source;
    HYLE_ASSUME(&source_impl != this);
    if (source.empty())
      return;

    IntrusiveListLink *first = source_impl.sentinel_.next;
    IntrusiveListLink *last = source_impl.sentinel_.prev;
    source_impl.sentinel_.prev = &source_impl.sentinel_;
    source_impl.sentinel_.next = &source_impl.sentinel_;

    IntrusiveListLink *position = mutable_hook(pos);
    IntrusiveListLink *before = position->prev;
    before->next = first;
    first->prev = before;
    last->next = position;
    position->prev = last;
  }

  // Transfer item from source before pos in O(1), matching the conventional
  // intrusive-list splice shape. Iterators to the node remain valid. The same
  // list is allowed; moving immediately before itself/its successor is a no-op.
  // Preconditions: pos belongs to this list; item is dereferenceable and
  // belongs to source.
  iterator splice(const_iterator pos, PublicList &source,
                  const_iterator item) noexcept {
    IntrusiveListImpl &source_impl = source;
    HYLE_ASSUME(item.hook_ != &source_impl.sentinel_);
    if (pos.hook_ == item.hook_ || pos.hook_ == item.hook_->next)
      return iterator(mutable_hook(item));
    IntrusiveListLink *hook = mutable_hook(item);
    IntrusiveListLink *position = mutable_hook(pos);
    IntrusiveListLink *before_item = hook->prev;
    IntrusiveListLink *after_item = hook->next;
    IntrusiveListLink *before_position = position->prev;

    // Transfer directly instead of clearing the hook and immediately writing
    // it again. This is the six-store circular-list primitive used by mature
    // intrusive containers.
    before_item->next = after_item;
    after_item->prev = before_item;
    before_position->next = hook;
    hook->prev = before_position;
    hook->next = position;
    position->prev = hook;
    return iterator(hook);
  }

  // Transfer [first, last) from source before pos. The range relink is O(1):
  // no node is visited and iterators to transferred nodes stay valid. Empty
  // ranges and a range already immediately before pos are no-ops. For a
  // same-list transfer, pos must not lie strictly inside [first, last).
  // Preconditions: pos belongs to this list and [first, last) is a valid range
  // in source. Returns the first transferred iterator, or pos for a no-op.
  iterator splice(const_iterator pos, PublicList &source,
                  const_iterator first, const_iterator last) noexcept {
    IntrusiveListLink *position = mutable_hook(pos);
    IntrusiveListLink *range_first = mutable_hook(first);
    IntrusiveListLink *after_range = mutable_hook(last);
    if (range_first == after_range || position == range_first ||
        position == after_range)
      return iterator(position);

    IntrusiveListLink *before_range = range_first->prev;
    IntrusiveListLink *range_last = after_range->prev;
    IntrusiveListLink *before_position = position->prev;

    before_range->next = after_range;
    after_range->prev = before_range;
    before_position->next = range_first;
    range_first->prev = before_position;
    range_last->next = position;
    position->prev = range_last;
    (void)source;
    return iterator(range_first);
  }

  // Concatenate source onto this list in O(1). This is the radix/timing-wheel
  // bucket-splice primitive and is also useful for completion batches.
  void append(PublicList &source) noexcept {
    IntrusiveListImpl &source_impl = source;
    HYLE_ASSUME(&source_impl != this);
    if (source.empty())
      return;

    IntrusiveListLink *first = source_impl.sentinel_.next;
    IntrusiveListLink *last = source_impl.sentinel_.prev;
    source_impl.sentinel_.prev = &source_impl.sentinel_;
    source_impl.sentinel_.next = &source_impl.sentinel_;

    IntrusiveListLink *destination_last = sentinel_.prev;
    destination_last->next = first;
    first->prev = destination_last;
    last->next = &sentinel_;
    sentinel_.prev = last;
  }

  void swap(PublicList &other) noexcept {
    IntrusiveListImpl &other_impl = other;
    if (this == &other_impl)
      return;

    IntrusiveListLink *this_first = sentinel_.next;
    IntrusiveListLink *this_last = sentinel_.prev;
    IntrusiveListLink *other_first = other_impl.sentinel_.next;
    IntrusiveListLink *other_last = other_impl.sentinel_.prev;

    sentinel_.next = other_first;
    sentinel_.prev = other_last;
    other_impl.sentinel_.next = this_first;
    other_impl.sentinel_.prev = this_last;

    if (other_first == &other_impl.sentinel_) {
      sentinel_.next = &sentinel_;
      sentinel_.prev = &sentinel_;
    } else {
      other_first->prev = &sentinel_;
      other_last->next = &sentinel_;
    }
    if (this_first == &sentinel_) {
      other_impl.sentinel_.next = &other_impl.sentinel_;
      other_impl.sentinel_.prev = &other_impl.sentinel_;
    } else {
      this_first->prev = &other_impl.sentinel_;
      this_last->next = &other_impl.sentinel_;
    }
  }

  // Visit linked nodes without changing membership or lifetime. If `fn`
  // throws, traversal stops and the unchanged list remains valid.
  template <class Fn>
  void for_each(Fn &&fn) noexcept(std::is_nothrow_invocable_v<Fn &, T *>) {
    IntrusiveListLink *current = sentinel_.next;
    while (current != &sentinel_) {
      std::invoke(fn, node_of(current));
      current = current->next;
    }
  }
  template <class Fn>
  void for_each(Fn &&fn) const
      noexcept(std::is_nothrow_invocable_v<Fn &, const T *>) {
    const IntrusiveListLink *current = sentinel_.next;
    while (current != &sentinel_) {
      std::invoke(fn, node_of(current));
      current = current->next;
    }
  }

  // Preconditions: node is non-null and belongs to this list. Matching mature
  // intrusive containers, erase/remove does not store or check an owner on the
  // hot path. Auto-unlink users may instead use the node hook's idempotent
  // unlink(); safe-link membership is deliberately changed only through list
  // operations.
  void remove(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    IntrusiveListLink *hook = hook_of(node);
    HYLE_ASSUME(hook->prev != nullptr);
    unlink_unchecked(hook);
  }

private:
  static IntrusiveListLink *mutable_hook(const_iterator position) noexcept {
    return const_cast<IntrusiveListLink *>(position.hook_);
  }

  static IntrusiveListLink *hook_of(T *node) noexcept {
    return &(node->*Hook).link_;
  }
  static const IntrusiveListLink *hook_of(const T *node) noexcept {
    return &(node->*Hook).link_;
  }

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

  static T *node_of(IntrusiveListLink *h) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(h) -
                         static_cast<std::uintptr_t>(hook_offset());
    return std::launder(reinterpret_cast<T *>(address));
  }
  static const T *node_of(const IntrusiveListLink *h) noexcept {
    return node_of(const_cast<IntrusiveListLink *>(h));
  }

  static void insert_before(IntrusiveListLink *pos,
                            IntrusiveListLink *node) noexcept {
    IntrusiveListLink *before = pos->prev;
    node->prev = before;
    node->next = pos;
    before->next = node;
    pos->prev = node;
  }

  static void unlink_unchecked(IntrusiveListLink *hook) noexcept {
    IntrusiveListLink *previous = hook->prev;
    IntrusiveListLink *next = hook->next;
    previous->next = next;
    next->prev = previous;
    hook->next = nullptr;
    hook->prev = nullptr;
  }

protected:
  IntrusiveListImpl() noexcept {
    sentinel_.next = &sentinel_;
    sentinel_.prev = &sentinel_;
  }
  IntrusiveListImpl(const IntrusiveListImpl &) = delete;
  IntrusiveListImpl &operator=(const IntrusiveListImpl &) = delete;
  ~IntrusiveListImpl() { clear(); }

private:
  IntrusiveListLink sentinel_;
};

} // namespace detail

// Auto-unlink list: nodes may detach themselves through hook.unlink() and a
// linked node detaches during destruction. This preserves the original HYLE
// API, layout and hot-path code generation.
template <class T, IntrusiveListHook T::*Hook>
class IntrusiveList
    : public detail::IntrusiveListImpl<T, Hook, IntrusiveList<T, Hook>> {
  using Base = detail::IntrusiveListImpl<T, Hook, IntrusiveList<T, Hook>>;

public:
  IntrusiveList() noexcept = default;
  IntrusiveList(const IntrusiveList &) = delete;
  IntrusiveList &operator=(const IntrusiveList &) = delete;
  IntrusiveList(IntrusiveList &&other) noexcept { this->append(other); }
  IntrusiveList &operator=(IntrusiveList &&other) noexcept {
    if (this != &other)
      this->swap(other);
    return *this;
  }
  ~IntrusiveList() = default;

  friend void swap(IntrusiveList &left, IntrusiveList &right) noexcept {
    left.swap(right);
  }
};

// Safe-link list: identical list operations and footprint, but membership may
// only be changed by the list. A linked node must not be destroyed.
template <class T, SafeIntrusiveListHook T::*Hook>
class SafeIntrusiveList
    : public detail::IntrusiveListImpl<T, Hook, SafeIntrusiveList<T, Hook>> {
  using Base =
      detail::IntrusiveListImpl<T, Hook, SafeIntrusiveList<T, Hook>>;

public:
  SafeIntrusiveList() noexcept = default;
  SafeIntrusiveList(const SafeIntrusiveList &) = delete;
  SafeIntrusiveList &operator=(const SafeIntrusiveList &) = delete;
  SafeIntrusiveList(SafeIntrusiveList &&other) noexcept {
    this->append(other);
  }
  SafeIntrusiveList &operator=(SafeIntrusiveList &&other) noexcept {
    if (this != &other)
      this->swap(other);
    return *this;
  }
  ~SafeIntrusiveList() = default;

  friend void swap(SafeIntrusiveList &left,
                   SafeIntrusiveList &right) noexcept {
    left.swap(right);
  }
};

} // namespace hyle::container
