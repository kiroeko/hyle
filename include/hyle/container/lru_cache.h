#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/flat_hash_set.h"
#include "hyle/container/intrusive_list.h"
#include "hyle/core/compiler.h"
#include "hyle/core/environment.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Zero means a deliberately disabled limit; kUnlimited removes that axis.
struct LruCacheLimits {
  static constexpr std::size_t kUnlimited =
      std::numeric_limits<std::size_t>::max();

  std::size_t max_entries = kUnlimited;
  std::size_t max_weight = kUnlimited;
};

enum class LruInsertStatus : std::uint8_t {
  kInserted,
  kAssigned,
  kOutOfMemory,
  kRejected,
};

template <class V> struct LruInsertResult {
  V* value = nullptr;
  LruInsertStatus status = LruInsertStatus::kRejected;
  std::size_t evicted = 0;

  [[nodiscard]] bool stored() const noexcept {
    return status == LruInsertStatus::kInserted ||
           status == LruInsertStatus::kAssigned;
  }
};

namespace detail {
struct IgnoreLruEviction {
  template <class K, class V>
  void operator()(K&&, V&&, std::size_t) const noexcept {}
};
}  // namespace detail

// Single-threaded hash-indexed LRU storage. The cache owns K/V and accepts an
// explicit per-entry weight; model/device lifetime, pinning and admission
// policy remain with the caller. Hash lookup, touch, erase and one eviction are
// expected O(1). A hit does not allocate. New entries allocate one stable node
// plus amortized FlatHashSet index storage.
//
// get() promotes; peek() does not. Returned pointers remain valid until an
// operation can erase/replace/evict that entry. Eviction callbacks are per-call
// noexcept templates, so the ordinary object stores no function pointer and
// the callback must not re-enter the same cache.
template <class K, class V, class Hasher = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
class LruCache {
  static_assert(std::is_nothrow_destructible_v<K> &&
                    std::is_nothrow_destructible_v<V>,
                "LruCache requires non-throwing destruction");
  static_assert(
      std::is_nothrow_invocable_r_v<std::uint64_t, const Hasher&, const K&> &&
          std::is_nothrow_invocable_r_v<bool, const Eq&, const K&, const K&>,
      "LruCache requires noexcept hash and equality");

  struct Node {
    Node(K node_key, V node_value, std::size_t node_weight)
        : key(std::move(node_key)), value(std::move(node_value)),
          weight(node_weight) {}

    IntrusiveListHook hook;
    K key;
    V value;
    std::size_t weight;
  };

  struct IndexHash {
    HYLE_NO_UNIQUE_ADDRESS Hasher hash;

    std::uint64_t operator()(const Node* node) const noexcept {
      return hash(node->key);
    }
    template <class Q>
      requires(!std::is_convertible_v<const Q&, const Node*>)
    std::uint64_t operator()(const Q& key) const
        noexcept(noexcept(hash(key))) {
      return hash(key);
    }
  };

  struct IndexEq {
    HYLE_NO_UNIQUE_ADDRESS Eq eq;

    bool operator()(const Node* left, const Node* right) const noexcept {
      return eq(left->key, right->key);
    }
    template <class Q>
      requires(!std::is_convertible_v<const Q&, const Node*>)
    bool operator()(const Node* left, const Q& right) const
        noexcept(noexcept(eq(left->key, right))) {
      return eq(left->key, right);
    }
  };

  using Index = FlatHashSet<Node*, IndexHash, IndexEq, Alloc>;
  using Recency = IntrusiveList<Node, &Node::hook>;

 public:
  explicit LruCache(LruCacheLimits limits,
                    Alloc alloc = Alloc{}, Hasher hash = Hasher{},
                    Eq eq = Eq{})
      : alloc_(std::move(alloc)),
        index_(alloc_, IndexHash{std::move(hash)}, IndexEq{std::move(eq)}),
        limits_(limits) {}

  ~LruCache() { clear(); }
  LruCache(const LruCache&) = delete;
  LruCache& operator=(const LruCache&) = delete;
  LruCache(LruCache&&) = delete;
  LruCache& operator=(LruCache&&) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::size_t total_weight() const noexcept {
    return total_weight_;
  }
  [[nodiscard]] LruCacheLimits limits() const noexcept { return limits_; }
  [[nodiscard]] const Alloc& allocator() const noexcept { return alloc_; }

  template <class Q> [[nodiscard]] bool contains(const Q& key) const noexcept {
    return find_node(key) != nullptr;
  }

  template <class Q> [[nodiscard]] V* peek(const Q& key) noexcept {
    Node* node = find_node(key);
    return node == nullptr ? nullptr : &node->value;
  }
  template <class Q> [[nodiscard]] const V* peek(const Q& key) const noexcept {
    const Node* node = find_node(key);
    return node == nullptr ? nullptr : &node->value;
  }

  template <class Q> [[nodiscard]] V* get(const Q& key) noexcept {
    Node* node = find_node(key);
    if (node == nullptr) return nullptr;
    promote(node);
    return &node->value;
  }

  template <class Q> bool touch(const Q& key) noexcept {
    Node* node = find_node(key);
    if (node == nullptr) return false;
    promote(node);
    return true;
  }

  template <class OnEvict>
    requires std::is_nothrow_invocable_v<OnEvict&, K&&, V&&, std::size_t>
  LruInsertResult<V> try_insert_or_assign(
      K key, V value, std::size_t weight, OnEvict&& on_evict) {
    if (!can_store(weight))
      return {nullptr, LruInsertStatus::kRejected, 0};

    if (Node* existing = find_node(key)) {
      const std::size_t weight_without_existing =
          total_weight_ - existing->weight;
      if (weight > std::numeric_limits<std::size_t>::max() -
                       weight_without_existing)
        return {nullptr, LruInsertStatus::kRejected, 0};
      existing->value = std::move(value);
      total_weight_ = weight_without_existing;
      existing->weight = weight;
      total_weight_ += weight;
      promote(existing);
      const std::size_t evicted =
          trim_to_limits(std::forward<OnEvict>(on_evict));
      return {&existing->value, LruInsertStatus::kAssigned, evicted};
    }

    if (weight > std::numeric_limits<std::size_t>::max() - total_weight_)
      return {nullptr, LruInsertStatus::kRejected, 0};

    void* storage = alloc_.allocate(sizeof(Node), alignof(Node));
    if (storage == nullptr)
      return {nullptr, LruInsertStatus::kOutOfMemory, 0};
    Node* node = nullptr;
#if HYLE_HAS_EXCEPTIONS
    try {
      node = ::new (storage)
          Node(std::move(key), std::move(value), weight);
    } catch (...) {
      alloc_.deallocate(storage, sizeof(Node), alignof(Node));
      throw;
    }
#else
    node = ::new (storage) Node(std::move(key), std::move(value), weight);
#endif

    const auto inserted = index_.try_insert(node);
    if (inserted != Index::InsertResult::kInserted) {
      destroy_node(node);
      HYLE_FATAL_ASSERT(inserted != Index::InsertResult::kAlreadyPresent);
      return {nullptr, LruInsertStatus::kOutOfMemory, 0};
    }

    recency_.push_front(node);
    ++size_;
    total_weight_ += weight;
    const std::size_t evicted =
        trim_to_limits(std::forward<OnEvict>(on_evict));
    return {&node->value, LruInsertStatus::kInserted, evicted};
  }

  LruInsertResult<V> try_insert_or_assign(K key, V value,
                                          std::size_t weight = 1) {
    return try_insert_or_assign(std::move(key), std::move(value), weight,
                                detail::IgnoreLruEviction{});
  }

  template <class OnEvict>
    requires std::is_nothrow_invocable_v<OnEvict&, K&&, V&&, std::size_t>
  V& insert_or_assign(K key, V value, std::size_t weight,
                      OnEvict&& on_evict) {
    LruInsertResult<V> result = try_insert_or_assign(
        std::move(key), std::move(value), weight,
        std::forward<OnEvict>(on_evict));
    HYLE_FATAL_ASSERT(result.stored());
    return *result.value;
  }

  V& insert_or_assign(K key, V value, std::size_t weight = 1) {
    return insert_or_assign(std::move(key), std::move(value), weight,
                            detail::IgnoreLruEviction{});
  }

  template <class Q> bool erase(const Q& key) noexcept {
    Node* node = find_node(key);
    if (node == nullptr) return false;
    erase_node(node);
    return true;
  }

  void clear() noexcept {
    while (Node* node = recency_.pop_back()) {
      const bool erased = index_.erase(node);
      HYLE_FATAL_ASSERT(erased);
      destroy_node(node);
    }
    size_ = 0;
    total_weight_ = 0;
  }

  template <class OnEvict>
    requires std::is_nothrow_invocable_v<OnEvict&, K&&, V&&, std::size_t>
  std::size_t set_limits(LruCacheLimits limits, OnEvict&& on_evict) noexcept {
    limits_ = limits;
    return trim_to_limits(std::forward<OnEvict>(on_evict));
  }

  std::size_t set_limits(LruCacheLimits limits) noexcept {
    return set_limits(limits, detail::IgnoreLruEviction{});
  }

  template <class Fn> void for_each_mru(Fn&& fn) const {
    recency_.for_each([&](const Node* node) {
      std::invoke(fn, static_cast<const K&>(node->key),
                  static_cast<const V&>(node->value), node->weight);
    });
  }

 private:
  [[nodiscard]] bool can_store(std::size_t weight) const noexcept {
    return limits_.max_entries != 0 && weight <= limits_.max_weight;
  }

  template <class Q> Node* find_node(const Q& key) const noexcept {
    Node* const* slot = index_.find(key);
    return slot == nullptr ? nullptr : *slot;
  }

  void promote(Node* node) noexcept {
    if (recency_.front() == node) return;
    recency_.remove(node);
    recency_.push_front(node);
  }

  void erase_node(Node* node) noexcept {
    recency_.remove(node);
    const bool erased = index_.erase(node);
    HYLE_FATAL_ASSERT(erased);
    --size_;
    total_weight_ -= node->weight;
    destroy_node(node);
  }

  void destroy_node(Node* node) noexcept {
    node->~Node();
    alloc_.deallocate(node, sizeof(Node), alignof(Node));
  }

  template <class OnEvict>
  std::size_t trim_to_limits(OnEvict&& on_evict) noexcept {
    std::size_t evicted = 0;
    while (size_ > limits_.max_entries ||
           total_weight_ > limits_.max_weight) {
      Node* node = recency_.pop_back();
      HYLE_FATAL_ASSERT(node != nullptr);
      const bool erased = index_.erase(node);
      HYLE_FATAL_ASSERT(erased);
      --size_;
      total_weight_ -= node->weight;
      std::invoke(on_evict, std::move(node->key), std::move(node->value),
                  node->weight);
      destroy_node(node);
      ++evicted;
    }
    return evicted;
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_;
  Index index_;
  Recency recency_;
  LruCacheLimits limits_;
  std::size_t size_ = 0;
  std::size_t total_weight_ = 0;
};

}  // namespace hyle::container
