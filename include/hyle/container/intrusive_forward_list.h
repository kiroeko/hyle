#pragma once

#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "hyle/container/detail/intrusive_member.h"
#include "hyle/core/compiler.h"

namespace hyle::container {

struct IntrusiveForwardListHook;

namespace detail {
struct IntrusiveForwardListLink {
  IntrusiveForwardListLink *next = nullptr;
};

template <class T, IntrusiveForwardListHook T::*Hook>
class IntrusiveForwardListImpl;
} // namespace detail

// One-pointer explicit-lifetime hook for caller-owned forward sequences. A
// self link denotes an unlinked hook; a linked tail uses nullptr, so linked()
// remains unambiguous without owner metadata. The hook and list are
// thread-compatible only; callers must externally synchronize every mutation,
// traversal and lifetime transition that can overlap another access.
struct IntrusiveForwardListHook {
  IntrusiveForwardListHook() noexcept { link_.next = &link_; }
  IntrusiveForwardListHook(const IntrusiveForwardListHook &) noexcept
      : IntrusiveForwardListHook() {}
  IntrusiveForwardListHook &
  operator=(const IntrusiveForwardListHook &) noexcept {
    return *this;
  }
  IntrusiveForwardListHook(IntrusiveForwardListHook &&) noexcept
      : IntrusiveForwardListHook() {}
  IntrusiveForwardListHook &operator=(IntrusiveForwardListHook &&) noexcept {
    return *this;
  }
  ~IntrusiveForwardListHook() { HYLE_ASSUME(!linked()); }

  bool linked() const noexcept { return link_.next != &link_; }

private:
  template <class T, IntrusiveForwardListHook T::*Hook>
  friend class detail::IntrusiveForwardListImpl;

  detail::IntrusiveForwardListLink link_;
};

static_assert(sizeof(IntrusiveForwardListHook) == sizeof(void *));
static_assert(std::is_standard_layout_v<IntrusiveForwardListHook>);

template <class T, IntrusiveForwardListHook T::*Hook>
class IntrusiveForwardList;

namespace detail {

template <class T, IntrusiveForwardListHook T::*Hook>
class IntrusiveForwardListImpl {
  static_assert(offsetof(IntrusiveForwardListHook, link_) == 0);

  template <bool IsConst> class BasicIterator {
    friend class IntrusiveForwardListImpl;
    template <bool> friend class BasicIterator;
    using LinkPointer =
        std::conditional_t<IsConst, const IntrusiveForwardListLink *,
                           IntrusiveForwardListLink *>;

    explicit BasicIterator(LinkPointer link) noexcept : link_(link) {}

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
        : link_(other.link_) {}

    reference operator*() const noexcept {
      return *IntrusiveForwardListImpl::node_of(link_);
    }
    pointer operator->() const noexcept {
      return IntrusiveForwardListImpl::node_of(link_);
    }
    BasicIterator &operator++() noexcept {
      link_ = link_->next;
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }
    template <bool Other>
    bool operator==(const BasicIterator<Other> &other) const noexcept {
      return link_ == other.link_;
    }

  private:
    LinkPointer link_ = nullptr;
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

  bool empty() const noexcept { return before_.next == nullptr; }
  iterator before_begin() noexcept { return iterator(&before_); }
  const_iterator before_begin() const noexcept {
    return const_iterator(&before_);
  }
  const_iterator cbefore_begin() const noexcept { return before_begin(); }
  iterator begin() noexcept { return iterator(before_.next); }
  const_iterator begin() const noexcept { return const_iterator(before_.next); }
  const_iterator cbegin() const noexcept { return begin(); }
  iterator end() noexcept { return iterator(nullptr); }
  const_iterator end() const noexcept { return const_iterator(nullptr); }
  const_iterator cend() const noexcept { return end(); }

  T *front() noexcept { return empty() ? nullptr : node_of(before_.next); }
  const T *front() const noexcept {
    return empty() ? nullptr : node_of(before_.next);
  }
  T *back() noexcept { return empty() ? nullptr : node_of(tail_); }
  const T *back() const noexcept { return empty() ? nullptr : node_of(tail_); }

  iterator iterator_to(T *node) noexcept {
    HYLE_ASSUME(node != nullptr);
    auto *link = hook_of(node);
    HYLE_ASSUME(link->next != link);
    return iterator(link);
  }
  const_iterator iterator_to(const T *node) const noexcept {
    HYLE_ASSUME(node != nullptr);
    const auto *link = hook_of(node);
    HYLE_ASSUME(link->next != link);
    return const_iterator(link);
  }

  iterator insert_after(const_iterator position, T *node) noexcept {
    HYLE_ASSUME(position.link_ != nullptr);
    HYLE_ASSUME(node != nullptr);
    auto *link = hook_of(node);
    HYLE_ASSUME(link->next == link);
    auto *previous = mutable_link(position);
    link->next = previous->next;
    previous->next = link;
    if (tail_ == previous) tail_ = link;
    return iterator(link);
  }
  [[nodiscard]] bool try_insert_after(const_iterator position,
                                      T *node) noexcept {
    if (position.link_ == nullptr || node == nullptr ||
        hook_of(node)->next != hook_of(node))
      return false;
    (void)insert_after(position, node);
    return true;
  }
  void push_front(T *node) noexcept {
    (void)insert_after(before_begin(), node);
  }
  void push_back(T *node) noexcept {
    (void)insert_after(iterator(tail_), node);
  }
  [[nodiscard]] bool try_push_front(T *node) noexcept {
    return try_insert_after(before_begin(), node);
  }
  [[nodiscard]] bool try_push_back(T *node) noexcept {
    return try_insert_after(iterator(tail_), node);
  }

  iterator erase_after(const_iterator position) noexcept {
    HYLE_ASSUME(position.link_ != nullptr);
    auto *previous = mutable_link(position);
    auto *removed = previous->next;
    HYLE_ASSUME(removed != nullptr);
    previous->next = removed->next;
    if (tail_ == removed) tail_ = previous;
    removed->next = removed;
    return iterator(previous->next);
  }
  [[nodiscard]] T *pop_front() noexcept {
    if (empty()) return nullptr;
    auto *removed = before_.next;
    (void)erase_after(before_begin());
    return node_of(removed);
  }

  void clear() noexcept {
    auto *current = before_.next;
    before_.next = nullptr;
    tail_ = &before_;
    while (current != nullptr) {
      auto *next = current->next;
      current->next = current;
      current = next;
    }
  }

  template <class Fn>
    requires std::is_nothrow_invocable_v<Fn &, T *>
  std::size_t consume_all(Fn &&fn) noexcept {
    auto *current = before_.next;
    before_.next = nullptr;
    tail_ = &before_;
    std::size_t count = 0;
    while (current != nullptr) {
      auto *next = current->next;
      current->next = current;
      std::invoke(fn, node_of(current));
      current = next;
      ++count;
    }
    return count;
  }

  void splice_after(const_iterator position,
                    IntrusiveForwardList<T, Hook> &source) noexcept {
    HYLE_ASSUME(
        (&source != static_cast<IntrusiveForwardList<T, Hook> *>(this)));
    if (source.empty()) return;
    auto *previous = mutable_link(position);
    HYLE_ASSUME(previous != nullptr);
    source.tail_->next = previous->next;
    previous->next = source.before_.next;
    if (tail_ == previous) tail_ = source.tail_;
    source.before_.next = nullptr;
    source.tail_ = &source.before_;
  }

  iterator splice_after(const_iterator position,
                        IntrusiveForwardList<T, Hook> &source,
                        const_iterator before_item) noexcept {
    auto *destination_previous = mutable_link(position);
    auto *source_previous = mutable_link(before_item);
    HYLE_ASSUME(destination_previous != nullptr && source_previous != nullptr);
    auto *moving = source_previous->next;
    HYLE_ASSUME(moving != nullptr);
    if (destination_previous == source_previous ||
        destination_previous == moving)
      return iterator(moving);

    source_previous->next = moving->next;
    if (source.tail_ == moving) source.tail_ = source_previous;
    moving->next = destination_previous->next;
    destination_previous->next = moving;
    if (tail_ == destination_previous) tail_ = moving;
    return iterator(moving);
  }

  // Move (before_first,last) after position. Locating the range tail is O(k);
  // relinking and all known-predecessor single/whole transfers are O(1).
  iterator splice_after(const_iterator position,
                        IntrusiveForwardList<T, Hook> &source,
                        const_iterator before_first,
                        const_iterator last) noexcept {
    auto *destination_previous = mutable_link(position);
    auto *source_previous = mutable_link(before_first);
    auto *after_range = mutable_link(last);
    HYLE_ASSUME(destination_previous != nullptr && source_previous != nullptr);
    auto *range_first = source_previous->next;
    if (range_first == after_range) return iterator(destination_previous->next);

    auto *range_last = range_first;
    while (range_last->next != after_range)
      range_last = range_last->next;
    HYLE_ASSUME(destination_previous != range_first &&
                destination_previous != range_last);

    source_previous->next = after_range;
    if (source.tail_ == range_last) source.tail_ = source_previous;
    range_last->next = destination_previous->next;
    destination_previous->next = range_first;
    if (tail_ == destination_previous) tail_ = range_last;
    return iterator(range_first);
  }

  void append(IntrusiveForwardList<T, Hook> &source) noexcept {
    splice_after(iterator(tail_), source);
  }

  void swap(IntrusiveForwardList<T, Hook> &other) noexcept {
    if (&other == static_cast<IntrusiveForwardList<T, Hook> *>(this)) return;
    auto *this_head = before_.next;
    auto *this_tail = tail_;
    before_.next = other.before_.next;
    tail_ = other.empty() ? &before_ : other.tail_;
    other.before_.next = this_head;
    other.tail_ = this_head == nullptr ? &other.before_ : this_tail;
  }

  template <class Fn>
  void for_each(Fn &&fn) noexcept(std::is_nothrow_invocable_v<Fn &, T *>) {
    for (auto *current = before_.next; current != nullptr;
         current = current->next)
      std::invoke(fn, node_of(current));
  }
  template <class Fn>
  void for_each(Fn &&fn) const
      noexcept(std::is_nothrow_invocable_v<Fn &, const T *>) {
    for (const auto *current = before_.next; current != nullptr;
         current = current->next)
      std::invoke(fn, node_of(current));
  }

  template <class Predicate> std::size_t remove_if(Predicate predicate) {
    std::size_t removed = 0;
    auto previous = before_begin();
    while (mutable_link(previous)->next != nullptr) {
      auto current = previous;
      ++current;
      if (std::invoke(predicate, *current)) {
        erase_after(previous);
        ++removed;
      } else {
        ++previous;
      }
    }
    return removed;
  }
  std::size_t remove(const T &value) {
    return remove_if([&](const T &candidate) { return candidate == value; });
  }

  template <class BinaryPredicate = std::equal_to<>>
  std::size_t unique(BinaryPredicate equivalent = {}) {
    std::size_t removed = 0;
    auto previous = begin();
    if (previous == end()) return 0;
    while (mutable_link(previous)->next != nullptr) {
      auto current = previous;
      ++current;
      if (std::invoke(equivalent, *previous, *current)) {
        erase_after(previous);
        ++removed;
      } else {
        ++previous;
      }
    }
    return removed;
  }

  void reverse() noexcept {
    if (empty() || before_.next->next == nullptr) return;
    auto *old_head = before_.next;
    auto *previous = static_cast<IntrusiveForwardListLink *>(nullptr);
    auto *current = before_.next;
    while (current != nullptr) {
      auto *next = current->next;
      current->next = previous;
      previous = current;
      current = next;
    }
    before_.next = previous;
    tail_ = old_head;
  }

  template <class Compare = std::less<>>
  void merge(IntrusiveForwardList<T, Hook> &source, Compare compare = {}) {
    HYLE_ASSUME(
        (&source != static_cast<IntrusiveForwardList<T, Hook> *>(this)));
    auto before_destination = before_begin();
    auto destination = begin();
    while (destination != end() && !source.empty()) {
      if (std::invoke(compare, *source.begin(), *destination)) {
        before_destination =
            splice_after(before_destination, source, source.before_begin());
      } else {
        before_destination = destination;
        ++destination;
      }
    }
    if (!source.empty()) splice_after(before_destination, source);
  }

  template <class Compare = std::less<>> void sort(Compare compare = {}) {
    if (empty() || before_.next->next == nullptr) return;
    IntrusiveForwardList<T, Hook> carry;
    IntrusiveForwardList<T, Hook> bins[sizeof(std::size_t) * 8];
    std::size_t fill = 0;
    struct GatherGuard {
      IntrusiveForwardList<T, Hook> *destination;
      IntrusiveForwardList<T, Hook> *carry;
      IntrusiveForwardList<T, Hook> *bins;
      bool active = true;

      ~GatherGuard() {
        if (!active) return;
        destination->append(*carry);
        for (std::size_t index = 0; index < sizeof(std::size_t) * 8; ++index)
          destination->append(bins[index]);
      }
    };
    GatherGuard guard{&static_cast<IntrusiveForwardList<T, Hook> &>(*this),
                      &carry, bins};

    while (!empty()) {
      carry.splice_after(carry.before_begin(),
                         static_cast<IntrusiveForwardList<T, Hook> &>(*this),
                         before_begin());
      std::size_t index = 0;
      while (index < fill && !bins[index].empty()) {
        bins[index].merge(carry, compare);
        bins[index].swap(carry);
        ++index;
      }
      carry.swap(bins[index]);
      if (index == fill) ++fill;
    }
    for (std::size_t index = 1; index < fill; ++index)
      bins[index].merge(bins[index - 1], compare);
    bins[fill - 1].swap(static_cast<IntrusiveForwardList<T, Hook> &>(*this));
    guard.active = false;
  }

protected:
  IntrusiveForwardListImpl() noexcept { tail_ = &before_; }
  IntrusiveForwardListImpl(const IntrusiveForwardListImpl &) = delete;
  IntrusiveForwardListImpl &
  operator=(const IntrusiveForwardListImpl &) = delete;
  ~IntrusiveForwardListImpl() { clear(); }

private:
  static IntrusiveForwardListLink *
  mutable_link(const_iterator position) noexcept {
    return const_cast<IntrusiveForwardListLink *>(position.link_);
  }
  static IntrusiveForwardListLink *hook_of(T *node) noexcept {
    return &IntrusiveMemberAccess<T, Hook>::member_of(node)->link_;
  }
  static const IntrusiveForwardListLink *hook_of(const T *node) noexcept {
    return &IntrusiveMemberAccess<T, Hook>::member_of(node)->link_;
  }
  static T *node_of(IntrusiveForwardListLink *link) noexcept {
    return IntrusiveMemberAccess<T, Hook>::owner_of(
        reinterpret_cast<IntrusiveForwardListHook *>(link));
  }
  static const T *node_of(const IntrusiveForwardListLink *link) noexcept {
    return node_of(const_cast<IntrusiveForwardListLink *>(link));
  }

  IntrusiveForwardListLink before_;
  IntrusiveForwardListLink *tail_ = nullptr;
};

} // namespace detail

template <class T, IntrusiveForwardListHook T::*Hook>
class IntrusiveForwardList : public detail::IntrusiveForwardListImpl<T, Hook> {
public:
  IntrusiveForwardList() noexcept = default;
  IntrusiveForwardList(const IntrusiveForwardList &) = delete;
  IntrusiveForwardList &operator=(const IntrusiveForwardList &) = delete;
  IntrusiveForwardList(IntrusiveForwardList &&other) noexcept {
    this->append(other);
  }
  IntrusiveForwardList &operator=(IntrusiveForwardList &&other) noexcept {
    if (this != &other) this->swap(other);
    return *this;
  }
  ~IntrusiveForwardList() = default;

  friend void swap(IntrusiveForwardList &left,
                   IntrusiveForwardList &right) noexcept {
    left.swap(right);
  }
};

} // namespace hyle::container
