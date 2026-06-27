#pragma once

#include <cstddef>
#include <cstdint>

namespace hyle::container {

// Embed one IntrusiveListHook in your node type, then list it via
// IntrusiveList<Node, &Node::hook>. O(1) push/pop/remove with ZERO allocation
// (the links live in the node). For the scheduler's waiter chains / run-queue
// overflow. Design follows folly IntrusiveList (boost-free), not the code.
struct IntrusiveListHook {
  IntrusiveListHook* prev = nullptr;
  IntrusiveListHook* next = nullptr;
  bool linked() const noexcept { return prev != nullptr || next != nullptr; }
};

template <class T, IntrusiveListHook T::* Hook>
class IntrusiveList {
 public:
  IntrusiveList() noexcept {
    sentinel_.prev = &sentinel_;
    sentinel_.next = &sentinel_;
  }
  IntrusiveList(const IntrusiveList&) = delete;
  IntrusiveList& operator=(const IntrusiveList&) = delete;

  bool empty() const noexcept { return sentinel_.next == &sentinel_; }

  void push_back(T* node) noexcept { insert_before(&sentinel_, hook_of(node)); }
  void push_front(T* node) noexcept {
    insert_before(sentinel_.next, hook_of(node));
  }

  T* front() noexcept { return empty() ? nullptr : node_of(sentinel_.next); }
  T* back() noexcept { return empty() ? nullptr : node_of(sentinel_.prev); }

  T* pop_front() noexcept {
    if (empty()) return nullptr;
    IntrusiveListHook* h = sentinel_.next;
    unlink(h);
    return node_of(h);
  }
  T* pop_back() noexcept {
    if (empty()) return nullptr;
    IntrusiveListHook* h = sentinel_.prev;
    unlink(h);
    return node_of(h);
  }

  // O(1) removal of any node (no traversal) -- the point of intrusive lists.
  void remove(T* node) noexcept { unlink(hook_of(node)); }

 private:
  static IntrusiveListHook* hook_of(T* node) noexcept { return &(node->*Hook); }

  // Recover the enclosing T* from a hook pointer via the member offset (the
  // classic container_of; boost.intrusive does the same).
  static T* node_of(IntrusiveListHook* h) noexcept {
    auto off = reinterpret_cast<std::size_t>(
        &(reinterpret_cast<T*>(0)->*Hook));
    return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(h) - off);
  }

  static void insert_before(IntrusiveListHook* pos,
                            IntrusiveListHook* node) noexcept {
    node->prev = pos->prev;
    node->next = pos;
    pos->prev->next = node;
    pos->prev = node;
  }
  static void unlink(IntrusiveListHook* h) noexcept {
    h->prev->next = h->next;
    h->next->prev = h->prev;
    h->prev = nullptr;
    h->next = nullptr;
  }

  IntrusiveListHook sentinel_;  // circular sentinel: empty == next points to self
};

}  // namespace hyle::container
