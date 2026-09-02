#pragma once

#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "hyle/container/detail/intrusive_member.h"
#include "hyle/core/compiler.h"

namespace hyle::container {

namespace detail {

// Auto-unlink needs enough information in an untyped hook destructor to repair
// its successor, so its chain stores hook-subobject addresses.
struct IntrusiveHListLink {
  IntrusiveHListLink *next = nullptr;
  IntrusiveHListLink **previous_next = nullptr;
};

// Safe-link retirement always goes through the typed list. Store enclosing
// node addresses as void* instead: this matches a hand-written T* hlist and
// avoids container_of work on callback iteration/pop without adding metadata.
struct SafeIntrusiveHListLink {
  void *next = nullptr;
  void **previous_next = nullptr;
};

template <class T, auto Hook, class PublicList> class IntrusiveHListImpl;

} // namespace detail

// Head-compact intrusive membership. Each node stores next + a pointer to the
// predecessor's next slot, while a list stores only its head pointer. Hooks
// and lists are thread-compatible only; callers must externally synchronize
// mutation, traversal and overlapping hook/list lifetime transitions.
struct IntrusiveHListHook {
  IntrusiveHListHook() noexcept = default;
  IntrusiveHListHook(const IntrusiveHListHook &) noexcept {}
  IntrusiveHListHook &operator=(const IntrusiveHListHook &) noexcept {
    return *this;
  }
  IntrusiveHListHook(IntrusiveHListHook &&) noexcept {}
  IntrusiveHListHook &operator=(IntrusiveHListHook &&) noexcept {
    return *this;
  }
  ~IntrusiveHListHook() { unlink(); }

  bool linked() const noexcept { return link_.previous_next != nullptr; }
  void unlink() noexcept {
    if (!linked()) return;
    *link_.previous_next = link_.next;
    if (link_.next != nullptr) link_.next->previous_next = link_.previous_next;
    link_.next = nullptr;
    link_.previous_next = nullptr;
  }

private:
  template <class, auto, class> friend class detail::IntrusiveHListImpl;
  detail::IntrusiveHListLink link_;
};

struct SafeIntrusiveHListHook {
  SafeIntrusiveHListHook() noexcept = default;
  SafeIntrusiveHListHook(const SafeIntrusiveHListHook &) noexcept {}
  SafeIntrusiveHListHook &operator=(const SafeIntrusiveHListHook &) noexcept {
    return *this;
  }
  SafeIntrusiveHListHook(SafeIntrusiveHListHook &&) noexcept {}
  SafeIntrusiveHListHook &operator=(SafeIntrusiveHListHook &&) noexcept {
    return *this;
  }
  ~SafeIntrusiveHListHook() {
    HYLE_ASSUME(link_.next == nullptr && link_.previous_next == nullptr);
  }

  bool linked() const noexcept { return link_.previous_next != nullptr; }

private:
  template <class, auto, class> friend class detail::IntrusiveHListImpl;
  detail::SafeIntrusiveHListLink link_;
};

static_assert(sizeof(IntrusiveHListHook) == 2 * sizeof(void *));
static_assert(sizeof(SafeIntrusiveHListHook) == 2 * sizeof(void *));
static_assert(std::is_standard_layout_v<IntrusiveHListHook>);
static_assert(std::is_standard_layout_v<SafeIntrusiveHListHook>);

template <class T, IntrusiveHListHook T::*Hook> class IntrusiveHList;
template <class T, SafeIntrusiveHListHook T::*Hook> class SafeIntrusiveHList;

namespace detail {

template <class T, auto Hook, class PublicList> class IntrusiveHListImpl {
public:
  using HookType = std::remove_reference_t<decltype(std::declval<T &>().*Hook)>;
  static constexpr bool kSafe =
      std::is_same_v<HookType, SafeIntrusiveHListHook>;
  static_assert(kSafe || std::is_same_v<HookType, IntrusiveHListHook>);
  using LinkType =
      std::conditional_t<kSafe, SafeIntrusiveHListLink, IntrusiveHListLink>;
  using ChainPointer = std::conditional_t<kSafe, void *, IntrusiveHListLink *>;
  using ChainSlot = ChainPointer *;
  static_assert(offsetof(HookType, link_) == 0);

private:
  template <bool IsConst> class BasicIterator {
    friend class IntrusiveHListImpl;
    template <bool> friend class BasicIterator;
    explicit BasicIterator(ChainPointer pointer) noexcept : pointer_(pointer) {}

  public:
    using iterator_category = std::forward_iterator_tag;
    using iterator_concept = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const T *, T *>;
    using reference = std::conditional_t<IsConst, const T &, T &>;

    BasicIterator() noexcept = default;
    template <bool OtherConst>
      requires(IsConst && !OtherConst)
    BasicIterator(const BasicIterator<OtherConst> &other) noexcept
        : pointer_(other.pointer_) {}
    reference operator*() const noexcept {
      return *IntrusiveHListImpl::node_of(pointer_);
    }
    pointer operator->() const noexcept {
      return IntrusiveHListImpl::node_of(pointer_);
    }
    BasicIterator &operator++() noexcept {
      pointer_ = IntrusiveHListImpl::next_of(pointer_);
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }
    template <bool Other>
    bool operator==(const BasicIterator<Other> &other) const noexcept {
      return pointer_ == other.pointer_;
    }

  private:
    ChainPointer pointer_ = nullptr;
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

  bool empty() const noexcept { return head_ == nullptr; }
  iterator begin() noexcept { return iterator(head_); }
  const_iterator begin() const noexcept { return const_iterator(head_); }
  const_iterator cbegin() const noexcept { return begin(); }
  iterator end() noexcept { return iterator(nullptr); }
  const_iterator end() const noexcept { return const_iterator(nullptr); }
  const_iterator cend() const noexcept { return end(); }
  T *front() noexcept { return empty() ? nullptr : node_of(head_); }
  const T *front() const noexcept { return empty() ? nullptr : node_of(head_); }

  iterator iterator_to(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    const ChainPointer pointer = chain_of(node);
    const ChainSlot slot = previous_slot(pointer);
    HYLE_ASSUME(slot != nullptr);
    return iterator(pointer);
  }
  const_iterator iterator_to(const T *node) const noexcept {
    HYLE_ASSUME(node != nullptr);
    const ChainPointer pointer = chain_of(const_cast<T *>(node));
    const ChainSlot slot = previous_slot(pointer);
    HYLE_ASSUME(slot != nullptr);
    return const_iterator(pointer);
  }

  void push_front(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    const ChainPointer pointer = chain_of(node);
    const ChainSlot old_slot = previous_slot(pointer);
    HYLE_ASSUME(old_slot == nullptr);
    set_next(pointer, head_);
    set_previous_slot(pointer, &head_);
    if (head_ != nullptr) set_previous_slot(head_, next_slot(pointer));
    head_ = pointer;
  }
  [[nodiscard]] bool try_push_front(T *node) noexcept {
    if (node == nullptr || previous_slot(chain_of(node)) != nullptr)
      return false;
    push_front(node);
    return true;
  }

  // Insert before a known node in O(1). end() is not an insertion position:
  // a one-pointer head deliberately has no tail.
  iterator insert_before(const_iterator position, T *node) noexcept {
    HYLE_ASSUME(position.pointer_ != nullptr && node != nullptr);
    const ChainPointer next = position.pointer_;
    const ChainPointer pointer = chain_of(node);
    const ChainSlot old_slot = previous_slot(pointer);
    HYLE_ASSUME(old_slot == nullptr);
    const ChainSlot slot = previous_slot(next);
    set_previous_slot(pointer, slot);
    set_next(pointer, next);
    *slot = pointer;
    set_previous_slot(next, next_slot(pointer));
    return iterator(pointer);
  }
  iterator insert_after(const_iterator position, T *node) noexcept {
    HYLE_ASSUME(position.pointer_ != nullptr && node != nullptr);
    const ChainPointer previous = position.pointer_;
    const ChainPointer pointer = chain_of(node);
    const ChainSlot old_slot = previous_slot(pointer);
    HYLE_ASSUME(old_slot == nullptr);
    const ChainPointer next = next_of(previous);
    set_next(pointer, next);
    set_previous_slot(pointer, next_slot(previous));
    set_next(previous, pointer);
    if (next != nullptr) set_previous_slot(next, next_slot(pointer));
    return iterator(pointer);
  }

  iterator erase(const_iterator position) noexcept {
    HYLE_ASSUME(position.pointer_ != nullptr);
    const ChainPointer pointer = position.pointer_;
    const ChainPointer next = next_of(pointer);
    unlink_unchecked(pointer);
    return iterator(next);
  }
  void remove(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    const ChainPointer pointer = chain_of(node);
    const ChainSlot slot = previous_slot(pointer);
    HYLE_ASSUME(slot != nullptr);
    unlink_unchecked(pointer);
  }
  [[nodiscard]] T *pop_front() noexcept {
    if (empty()) return nullptr;
    const ChainPointer pointer = head_;
    unlink_unchecked(pointer);
    return node_of(pointer);
  }

  iterator replace(const_iterator position, T *replacement) noexcept {
    HYLE_ASSUME(position.pointer_ != nullptr && replacement != nullptr);
    const ChainPointer old_pointer = position.pointer_;
    const ChainPointer new_pointer = chain_of(replacement);
    const ChainSlot old_replacement_slot = previous_slot(new_pointer);
    HYLE_ASSUME(old_replacement_slot == nullptr);
    const ChainPointer next = next_of(old_pointer);
    const ChainSlot slot = previous_slot(old_pointer);
    set_next(new_pointer, next);
    set_previous_slot(new_pointer, slot);
    *slot = new_pointer;
    if (next != nullptr) set_previous_slot(next, next_slot(new_pointer));
    reset(old_pointer);
    return iterator(new_pointer);
  }

  void clear() noexcept {
    ChainPointer current = head_;
    head_ = nullptr;
    while (current != nullptr) {
      const ChainPointer next = next_of(current);
      reset(current);
      current = next;
    }
  }
  template <class Fn>
    requires std::is_nothrow_invocable_v<Fn &, T *>
  std::size_t consume_all(Fn &&fn) noexcept {
    ChainPointer current = head_;
    head_ = nullptr;
    std::size_t count = 0;
    while (current != nullptr) {
      const ChainPointer next = next_of(current);
      T *node = node_of(current);
      reset(current);
      std::invoke(fn, node);
      current = next;
      ++count;
    }
    return count;
  }
  void swap(PublicList &other) noexcept {
    IntrusiveHListImpl &other_impl = other;
    if (this == &other_impl) return;
    std::swap(head_, other_impl.head_);
    if (head_ != nullptr) set_previous_slot(head_, &head_);
    if (other_impl.head_ != nullptr)
      set_previous_slot(other_impl.head_, &other_impl.head_);
  }

  template <class Fn>
  void for_each(Fn &&fn) noexcept(std::is_nothrow_invocable_v<Fn &, T *>) {
    for (ChainPointer current = head_; current != nullptr;
         current = next_of(current))
      std::invoke(fn, node_of(current));
  }
  template <class Fn>
  void for_each(Fn &&fn) const
      noexcept(std::is_nothrow_invocable_v<Fn &, const T *>) {
    for (ChainPointer current = head_; current != nullptr;
         current = next_of(current))
      std::invoke(fn, static_cast<const T *>(node_of(current)));
  }

protected:
  IntrusiveHListImpl() noexcept = default;
  IntrusiveHListImpl(const IntrusiveHListImpl &) = delete;
  IntrusiveHListImpl &operator=(const IntrusiveHListImpl &) = delete;
  ~IntrusiveHListImpl() { clear(); }

private:
  static LinkType *link_of(T *node) noexcept {
    return &IntrusiveMemberAccess<T, Hook>::member_of(node)->link_;
  }
  static const LinkType *link_of(const T *node) noexcept {
    return &IntrusiveMemberAccess<T, Hook>::member_of(node)->link_;
  }
  static ChainPointer chain_of(T *node) noexcept {
    if constexpr (kSafe) return static_cast<void *>(node);
    else return link_of(node);
  }
  static T *node_of(ChainPointer pointer) noexcept {
    if constexpr (kSafe) {
      return static_cast<T *>(pointer);
    } else {
      return IntrusiveMemberAccess<T, Hook>::owner_of(
          reinterpret_cast<HookType *>(pointer));
    }
  }
  static ChainPointer next_of(ChainPointer pointer) noexcept {
    if constexpr (kSafe) return link_of(static_cast<T *>(pointer))->next;
    else return pointer->next;
  }
  static void set_next(ChainPointer pointer, ChainPointer next) noexcept {
    if constexpr (kSafe) link_of(static_cast<T *>(pointer))->next = next;
    else pointer->next = next;
  }
  static ChainSlot next_slot(ChainPointer pointer) noexcept {
    if constexpr (kSafe) return &link_of(static_cast<T *>(pointer))->next;
    else return &pointer->next;
  }
  static ChainSlot previous_slot(ChainPointer pointer) noexcept {
    if constexpr (kSafe)
      return link_of(static_cast<T *>(pointer))->previous_next;
    else return pointer->previous_next;
  }
  static void set_previous_slot(ChainPointer pointer, ChainSlot slot) noexcept {
    if constexpr (kSafe)
      link_of(static_cast<T *>(pointer))->previous_next = slot;
    else pointer->previous_next = slot;
  }
  static void reset(ChainPointer pointer) noexcept {
    set_next(pointer, nullptr);
    set_previous_slot(pointer, nullptr);
  }
  static void unlink_unchecked(ChainPointer pointer) noexcept {
    const ChainSlot slot = previous_slot(pointer);
    const ChainPointer next = next_of(pointer);
    *slot = next;
    if (next != nullptr) set_previous_slot(next, slot);
    reset(pointer);
  }

  ChainPointer head_ = nullptr;
};

} // namespace detail

template <class T, IntrusiveHListHook T::*Hook>
class IntrusiveHList
    : public detail::IntrusiveHListImpl<T, Hook, IntrusiveHList<T, Hook>> {
public:
  IntrusiveHList() noexcept = default;
  IntrusiveHList(const IntrusiveHList &) = delete;
  IntrusiveHList &operator=(const IntrusiveHList &) = delete;
  IntrusiveHList(IntrusiveHList &&other) noexcept { this->swap(other); }
  IntrusiveHList &operator=(IntrusiveHList &&other) noexcept {
    if (this != &other) this->swap(other);
    return *this;
  }
  ~IntrusiveHList() = default;

  friend void swap(IntrusiveHList &left, IntrusiveHList &right) noexcept {
    left.swap(right);
  }
};

template <class T, SafeIntrusiveHListHook T::*Hook>
class SafeIntrusiveHList
    : public detail::IntrusiveHListImpl<T, Hook, SafeIntrusiveHList<T, Hook>> {
public:
  SafeIntrusiveHList() noexcept = default;
  SafeIntrusiveHList(const SafeIntrusiveHList &) = delete;
  SafeIntrusiveHList &operator=(const SafeIntrusiveHList &) = delete;
  SafeIntrusiveHList(SafeIntrusiveHList &&other) noexcept { this->swap(other); }
  SafeIntrusiveHList &operator=(SafeIntrusiveHList &&other) noexcept {
    if (this != &other) this->swap(other);
    return *this;
  }
  ~SafeIntrusiveHList() = default;

  friend void swap(SafeIntrusiveHList &left,
                   SafeIntrusiveHList &right) noexcept {
    left.swap(right);
  }
};

} // namespace hyle::container
