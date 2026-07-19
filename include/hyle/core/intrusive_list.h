#pragma once

#include <cstddef>
#include <type_traits>

#include "hyle/macros.h"

namespace hyle::core {

// Lowest-level intrusive linking primitive shared by time/ and container/.
// Keeping it in core avoids the false time -> container module dependency that
// existed when timer_source.h included hyle/container/intrusive_list.h.
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
    insert_before(&sentinel_, hook_of(node));
  }
  void push_front(T* node) noexcept {
    HYLE_FATAL_ASSERT(node != nullptr);
    HYLE_FATAL_ASSERT(!hook_of(node)->linked());
    insert_before(sentinel_.next_, hook_of(node));
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
    unlink(h);
    return node_of(h);
  }
  T* pop_back() noexcept {
    if (empty()) return nullptr;
    IntrusiveListHook* h = sentinel_.prev_;
    unlink(h);
    return node_of(h);
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
