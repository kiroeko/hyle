#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Value exposed by BTreeMap iterators. The key is deliberately read-only so a
// caller cannot invalidate the ordering invariant through an iterator.
template <class K, class V> class BTreeMapEntry {
public:
  template <class... Args>
  BTreeMapEntry(K key, std::in_place_t, Args &&...args) noexcept(
      std::is_nothrow_move_constructible_v<K> &&
      std::is_nothrow_constructible_v<V, Args...>)
      : key_(std::move(key)), value_(std::forward<Args>(args)...) {}

  BTreeMapEntry(const BTreeMapEntry &) = default;
  BTreeMapEntry &operator=(const BTreeMapEntry &) = delete;
  BTreeMapEntry(BTreeMapEntry &&) noexcept(
      std::is_nothrow_move_constructible_v<K> &&
      std::is_nothrow_move_constructible_v<V>) = default;
  BTreeMapEntry &operator=(BTreeMapEntry &&) = delete;

  const K &key() const noexcept { return key_; }
  V &value() noexcept { return value_; }
  const V &value() const noexcept { return value_; }

private:
  K key_;
  HYLE_NO_UNIQUE_ADDRESS V value_;
};

namespace detail {

template <class K, class V> struct BTreeMapPolicy {
  static constexpr bool kIsSet = false;
  using key_type = K;
  using entry_type = BTreeMapEntry<K, V>;
  using value_type = entry_type;
  using reference = entry_type &;
  using const_reference = const entry_type &;
  using pointer = entry_type *;
  using const_pointer = const entry_type *;

  static const K &key(const entry_type &entry) noexcept { return entry.key(); }
  static reference element(entry_type &entry) noexcept { return entry; }
  static const_reference element(const entry_type &entry) noexcept {
    return entry;
  }
};

template <class K> struct BTreeSetEntry {
  explicit BTreeSetEntry(K value) noexcept(
      std::is_nothrow_move_constructible_v<K>)
      : key(std::move(value)) {}
  BTreeSetEntry(const BTreeSetEntry &) = default;
  BTreeSetEntry &operator=(const BTreeSetEntry &) = delete;
  BTreeSetEntry(BTreeSetEntry &&) noexcept(
      std::is_nothrow_move_constructible_v<K>) = default;
  BTreeSetEntry &operator=(BTreeSetEntry &&) = delete;
  K key;
};

template <class K> struct BTreeSetPolicy {
  static constexpr bool kIsSet = true;
  using key_type = K;
  using entry_type = BTreeSetEntry<K>;
  using value_type = K;
  using reference = const K &;
  using const_reference = const K &;
  using pointer = const K *;
  using const_pointer = const K *;

  static const K &key(const entry_type &entry) noexcept { return entry.key; }
  static reference element(entry_type &entry) noexcept { return entry.key; }
  static const_reference element(const entry_type &entry) noexcept {
    return entry.key;
  }
};

// Cache-oriented mutable B-tree core. Values live in both leaf and internal
// nodes (the Abseil family shape), avoiding duplicate separator keys and making
// move-only keys possible. Leaf storage targets 256 bytes; internal nodes add
// their child-pointer array. Capacity is forced to 2*t-1 so ordinary CLRS
// borrow/merge deletion keeps O(log N) height. As in Abseil's mature B-tree
// shape, repeated deletion at an outer node boundary may leave a non-empty
// node temporarily below half full: eagerly borrowing values that the same
// expiry/window walk will immediately delete creates avoidable page traffic.
template <class Policy, class Compare, class Alloc> class BTreeCore {
protected:
  using Entry = typename Policy::entry_type;
  using Key = typename Policy::key_type;

private:
  struct Node {
    Node *parent;
    std::uint16_t count;
    std::uint16_t position;
    bool leaf;
  };

  struct InsertLocation {
    Node *node;
    std::uint16_t index;
    bool found;
  };

  struct NodeSearchResult {
    std::uint16_t index;
    bool found;
  };

  static constexpr std::size_t kTargetLeafBytes = 256;
  static constexpr std::size_t
      kRawCapacity = kTargetLeafBytes > sizeof(Node)
                         ? (kTargetLeafBytes - sizeof(Node)) / sizeof(Entry)
                         : 0;
  static constexpr std::size_t kAtLeastFive =
      kRawCapacity < 5 ? 5 : kRawCapacity;
  static constexpr std::size_t kCapacity =
      (kAtLeastFive & 1) != 0 ? kAtLeastFive : kAtLeastFive - 1;
  static constexpr std::size_t kMinKeys = kCapacity / 2;
  static constexpr std::size_t kNodeAlignment =
      std::max({alignof(Node), alignof(Entry), alignof(Node *)});

  static constexpr std::size_t align_up(std::size_t value,
                                        std::size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
  }
  static constexpr std::size_t kEntriesOffset =
      align_up(sizeof(Node), alignof(Entry));
  static constexpr std::size_t kChildrenOffset =
      align_up(kEntriesOffset + kCapacity * sizeof(Entry), alignof(Node *));
  static constexpr std::size_t kLeafBytes =
      kEntriesOffset + kCapacity * sizeof(Entry);
  static constexpr std::size_t kInternalBytes =
      kChildrenOffset + (kCapacity + 1) * sizeof(Node *);

  static_assert(kCapacity >= 5 && (kCapacity & 1) != 0);
  static_assert(kCapacity <= std::numeric_limits<std::uint16_t>::max());
  static_assert(std::is_nothrow_move_constructible_v<Entry> &&
                    std::is_nothrow_destructible_v<Entry>,
                "BTree entries require nothrow move/destruction");
  static_assert(std::is_nothrow_invocable_r_v<bool, const Compare &,
                                              const Key &, const Key &>,
                "BTree comparator must be noexcept");
  static_assert(std::is_nothrow_move_constructible_v<Compare> &&
                    std::is_nothrow_move_assignable_v<Compare> &&
                    std::is_nothrow_swappable_v<Compare>,
                "BTree comparator movement/swap must be noexcept");
  static_assert(std::is_nothrow_move_constructible_v<Alloc> &&
                    std::is_nothrow_move_assignable_v<Alloc> &&
                    std::is_nothrow_swappable_v<Alloc>,
                "BTree allocator movement/swap must be noexcept");

  template <class Q>
  static constexpr bool comparable_with =
      std::is_nothrow_invocable_r_v<bool, const Compare &, const Key &,
                                    const Q &> &&
      std::is_nothrow_invocable_r_v<bool, const Compare &, const Q &,
                                    const Key &>;

public:
  using key_type = Key;
  using value_type = typename Policy::value_type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = typename Policy::reference;
  using const_reference = typename Policy::const_reference;
  using pointer = typename Policy::pointer;
  using const_pointer = typename Policy::const_pointer;

  enum class InsertResult : std::uint8_t {
    kInserted,
    kAlreadyPresent,
    kAssigned,
    kOutOfMemory,
  };

  template <bool IsConst> class BasicIterator {
    friend class BTreeCore;
    template <bool> friend class BasicIterator;
    using Owner = std::conditional_t<IsConst, const BTreeCore, BTreeCore>;
    using IteratorNode = std::conditional_t<IsConst, const Node, Node>;

    BasicIterator(Owner *owner, IteratorNode *node,
                  std::uint16_t index) noexcept
        : owner_(owner), node_(node), index_(index) {}

  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = std::bidirectional_iterator_tag;
    using value_type = typename Policy::value_type;
    using difference_type = std::ptrdiff_t;
    using reference =
        std::conditional_t<IsConst, typename Policy::const_reference,
                           typename Policy::reference>;
    using pointer = std::conditional_t<IsConst, typename Policy::const_pointer,
                                       typename Policy::pointer>;

    BasicIterator() noexcept = default;
    template <bool OtherConst>
      requires(IsConst && !OtherConst)
    BasicIterator(const BasicIterator<OtherConst> &other) noexcept
        : owner_(other.owner_), node_(other.node_), index_(other.index_) {}

    reference operator*() const noexcept {
      HYLE_FATAL_ASSERT(owner_ != nullptr && node_ != nullptr);
      return Policy::element(*owner_->entry(node_, index_));
    }
    pointer operator->() const noexcept { return std::addressof(**this); }

    BasicIterator &operator++() noexcept {
      HYLE_FATAL_ASSERT(owner_ != nullptr && node_ != nullptr);
      if (node_->leaf && index_ + 1 < node_->count) {
        ++index_;
        return *this;
      }
      owner_->advance(node_, index_);
      return *this;
    }
    BasicIterator operator++(int) noexcept {
      BasicIterator copy = *this;
      ++*this;
      return copy;
    }
    BasicIterator &operator--() noexcept {
      HYLE_FATAL_ASSERT(owner_ != nullptr);
      if (node_ != nullptr && node_->leaf && index_ != 0) {
        --index_;
        return *this;
      }
      owner_->retreat(node_, index_);
      return *this;
    }
    BasicIterator operator--(int) noexcept {
      BasicIterator copy = *this;
      --*this;
      return copy;
    }

    friend bool operator==(const BasicIterator &left,
                           const BasicIterator &right) noexcept {
      return left.owner_ == right.owner_ && left.node_ == right.node_ &&
             left.index_ == right.index_;
    }

  private:
    Owner *owner_ = nullptr;
    IteratorNode *node_ = nullptr;
    std::uint16_t index_ = 0;
  };

  using iterator = BasicIterator<false>;
  using const_iterator = BasicIterator<true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  BTreeCore()
    requires(std::is_default_constructible_v<Compare> &&
             std::is_default_constructible_v<Alloc>)
  = default;

  explicit BTreeCore(Compare compare, Alloc alloc = Alloc{})
      : compare_(std::move(compare)), alloc_(std::move(alloc)) {}

  BTreeCore(const BTreeCore &other)
    requires std::is_copy_constructible_v<Entry>
      : compare_(other.compare_), alloc_(other.alloc_) {
    try {
      copy_from(other);
    } catch (...) {
      clear();
      throw;
    }
  }

  BTreeCore &operator=(const BTreeCore &other)
    requires std::is_copy_constructible_v<Entry>
  {
    if (this != &other) {
      BTreeCore replacement(other);
      swap(replacement);
    }
    return *this;
  }

  BTreeCore(BTreeCore &&other) noexcept
      : root_(std::exchange(other.root_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        compare_(std::move(other.compare_)), alloc_(std::move(other.alloc_)) {}

  BTreeCore &operator=(BTreeCore &&other) noexcept {
    if (this != &other) {
      clear();
      root_ = std::exchange(other.root_, nullptr);
      size_ = std::exchange(other.size_, 0);
      compare_ = std::move(other.compare_);
      alloc_ = std::move(other.alloc_);
    }
    return *this;
  }

  ~BTreeCore() { clear(); }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  const Compare &key_comp() const noexcept { return compare_; }
  const Alloc &allocator() const noexcept { return alloc_; }

  iterator begin() noexcept {
    auto [node, index] = first_location();
    return iterator(this, node, index);
  }
  const_iterator begin() const noexcept {
    auto [node, index] = first_location();
    return const_iterator(this, node, index);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  iterator end() noexcept { return iterator(this, nullptr, 0); }
  const_iterator end() const noexcept {
    return const_iterator(this, nullptr, 0);
  }
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

  template <class Q>
    requires(comparable_with<Q>)
  iterator find(const Q &key) noexcept {
    auto [node, index] = find_location(key);
    return iterator(this, node, index);
  }
  template <class Q>
    requires(comparable_with<Q>)
  const_iterator find(const Q &key) const noexcept {
    auto [node, index] = find_location(key);
    return const_iterator(this, node, index);
  }
  template <class Q>
    requires(comparable_with<Q>)
  bool contains(const Q &key) const noexcept {
    return find(key) != end();
  }
  template <class Q>
    requires(comparable_with<Q>)
  std::size_t count(const Q &key) const noexcept {
    return contains(key) ? 1 : 0;
  }

  template <class Q>
    requires(comparable_with<Q>)
  iterator lower_bound(const Q &key) noexcept {
    auto [node, index] = bound_location(key, false);
    return iterator(this, node, index);
  }
  template <class Q>
    requires(comparable_with<Q>)
  const_iterator lower_bound(const Q &key) const noexcept {
    auto [node, index] = bound_location(key, false);
    return const_iterator(this, node, index);
  }
  template <class Q>
    requires(comparable_with<Q>)
  iterator upper_bound(const Q &key) noexcept {
    auto [node, index] = bound_location(key, true);
    return iterator(this, node, index);
  }
  template <class Q>
    requires(comparable_with<Q>)
  const_iterator upper_bound(const Q &key) const noexcept {
    auto [node, index] = bound_location(key, true);
    return const_iterator(this, node, index);
  }
  template <class Q>
    requires(comparable_with<Q>)
  std::pair<iterator, iterator> equal_range(const Q &key) noexcept {
    return {lower_bound(key), upper_bound(key)};
  }
  template <class Q>
    requires(comparable_with<Q>)
  std::pair<const_iterator, const_iterator>
  equal_range(const Q &key) const noexcept {
    return {lower_bound(key), upper_bound(key)};
  }

  // Lookup finishes before rebalance starts, so an exact-key reference may
  // safely alias an entry in this tree. Avoiding an owning snapshot keeps the
  // common erase(key) path allocation-free and supports move-only keys.
  bool erase(const Key &key) noexcept {
    auto [node, index] = find_location(key);
    if (node == nullptr)
      return false;
    erase_location(node, index);
    return true;
  }

  template <class Q>
    requires(!std::is_same_v<std::remove_cvref_t<Q>, Key> && comparable_with<Q>)
  bool erase(const Q &key) noexcept {
    auto [node, index] = find_location(key);
    if (node == nullptr)
      return false;
    erase_location(node, index);
    return true;
  }

  iterator erase(const_iterator position) {
    HYLE_FATAL_ASSERT(position.owner_ == this && position.node_ != nullptr);
    return erase_and_get_next(const_cast<Node *>(position.node_),
                              position.index_);
  }

  iterator erase(const_iterator first, const_iterator last) {
    HYLE_FATAL_ASSERT(first.owner_ == this && last.owner_ == this);
    if (first == last)
      return iterator(this, const_cast<Node *>(last.node_), last.index_);
    if constexpr (std::is_copy_constructible_v<Key>)
      return erase_range_to_key(first, last);

    // Count before mutating because every successful mutation invalidates the
    // supplied iterators. Range deletion then removes a leaf run with one
    // suffix move and one rebalance, rather than performing a fresh tree
    // lookup for every element in the common expiry/prefix-eviction shape.
    std::size_t count = 0;
    for (auto cursor = first; cursor != last;) {
      if (!cursor.node_->leaf) {
        ++count;
        ++cursor;
        continue;
      }
      if (cursor.node_ == last.node_) {
        HYLE_FATAL_ASSERT(cursor.index_ <= last.index_);
        count += last.index_ - cursor.index_;
        break;
      }
      count += cursor.node_->count - cursor.index_;
      cursor.index_ = static_cast<std::uint16_t>(cursor.node_->count - 1);
      ++cursor;
    }
    if (count == size_) {
      clear();
      return end();
    }

    const std::size_t target_size = size_ - count;
    while (size_ != target_size) {
      HYLE_FATAL_ASSERT(first != end());
      if (!first.node_->leaf) {
        first = erase(first);
        continue;
      }

      const std::size_t remaining = size_ - target_size;
      const std::uint16_t available =
          static_cast<std::uint16_t>(first.node_->count - first.index_);
      const std::uint16_t to_erase = static_cast<std::uint16_t>(
          std::min<std::size_t>(remaining, available));
      Node *node = const_cast<Node *>(first.node_);
      Node *next_node = node;
      std::uint16_t next_index = first.index_;
      remove_entries(node, first.index_, to_erase);
      size_ -= to_erase;
      // When the range also consumes the separator after an emptied leaf,
      // unlink that leaf directly. A conventional merge would first move the
      // entire right sibling into the empty node only for the next loop to
      // destroy those entries again. Direct unlinking is the B-tree analogue
      // of dropping a fully expired page and is useful for any long range, not
      // just the prefix-expiry benchmark.
      if (node->count == 0 && node != root_ && size_ != target_size &&
          node->position < node->parent->count) {
        Node *parent = node->parent;
        const std::uint16_t position = node->position;
        remove_entry(parent, position);
        --size_; // the in-order separator is also inside [first, last)
        for (std::size_t child_index = position; child_index <= parent->count;
             ++child_index)
          child(parent, child_index) = child(parent, child_index + 1);
        update_child_positions(parent, position);
        next_node = child(parent, position);
        next_index = 0;
        deallocate_empty_node(node);
        rebalance_after_delete(parent, &next_node, &next_index);
      } else {
        rebalance_after_delete(node, &next_node, &next_index);
      }
      if (next_node != nullptr && next_index == next_node->count) {
        HYLE_FATAL_ASSERT(next_node->count != 0);
        next_index = static_cast<std::uint16_t>(next_node->count - 1);
        advance(next_node, next_index);
      }
      first = const_iterator(this, next_node, next_index);
    }
    return iterator(this, const_cast<Node *>(first.node_), first.index_);
  }

  void clear() noexcept {
    destroy_subtree(root_);
    root_ = nullptr;
    size_ = 0;
  }

  void swap(BTreeCore &other) noexcept {
    using std::swap;
    swap(root_, other.root_);
    swap(size_, other.size_);
    swap(compare_, other.compare_);
    swap(alloc_, other.alloc_);
  }

protected:
  template <class OnExisting>
  std::pair<iterator, InsertResult> try_insert_entry(Entry candidate,
                                                     OnExisting &&on_existing) {
    const Key &key = Policy::key(candidate);
    InsertLocation location = locate_for_insert(key);
    if (location.found) {
      const InsertResult result = std::forward<OnExisting>(on_existing)(
          *entry(location.node, location.index), candidate);
      return {iterator(this, location.node, location.index), result};
    }
    if (location.node == nullptr) {
      Node *root = allocate_node(true, nullptr);
      if (root == nullptr)
        return {end(), InsertResult::kOutOfMemory};
      construct_entry(root, 0, std::move(candidate));
      root->count = 1;
      root_ = root;
      size_ = 1;
      return {iterator(this, root, 0), InsertResult::kInserted};
    }
    if (!make_room_for_insert(location.node, location.index))
      return {end(), InsertResult::kOutOfMemory};
    insert_entry(location.node, location.index, std::move(candidate));
    ++size_;
    return {iterator(this, location.node, location.index),
            InsertResult::kInserted};
  }

  std::pair<iterator, InsertResult> try_insert_entry(Entry candidate) {
    return try_insert_entry(std::move(candidate),
                            [](Entry &, Entry &) noexcept {
                              return InsertResult::kAlreadyPresent;
                            });
  }

  template <class Factory>
  std::pair<iterator, InsertResult> try_emplace_entry(const Key &key,
                                                      Factory &&factory) {
    InsertLocation location = locate_for_insert(key);
    if (location.found)
      return {iterator(this, location.node, location.index),
              InsertResult::kAlreadyPresent};

    // Construction happens only after the unique-key lookup, but before any
    // structural mutation. Throwing mapped constructors therefore leave the
    // index and all of its iterators untouched.
    Entry candidate = std::forward<Factory>(factory)();
    if (location.node == nullptr) {
      Node *root = allocate_node(true, nullptr);
      if (root == nullptr)
        return {end(), InsertResult::kOutOfMemory};
      construct_entry(root, 0, std::move(candidate));
      root->count = 1;
      root_ = root;
      size_ = 1;
      return {iterator(this, root, 0), InsertResult::kInserted};
    }
    if (!make_room_for_insert(location.node, location.index))
      return {end(), InsertResult::kOutOfMemory};
    insert_entry(location.node, location.index, std::move(candidate));
    ++size_;
    return {iterator(this, location.node, location.index),
            InsertResult::kInserted};
  }

private:
  static Entry *entry(Node *node, std::size_t index) noexcept {
    auto *bytes = reinterpret_cast<std::byte *>(node);
    return std::launder(reinterpret_cast<Entry *>(bytes + kEntriesOffset) +
                        index);
  }
  static const Entry *entry(const Node *node, std::size_t index) noexcept {
    auto *bytes = reinterpret_cast<const std::byte *>(node);
    return std::launder(
        reinterpret_cast<const Entry *>(bytes + kEntriesOffset) + index);
  }
  static Node **children(Node *node) noexcept {
    auto *bytes = reinterpret_cast<std::byte *>(node);
    return reinterpret_cast<Node **>(bytes + kChildrenOffset);
  }
  static Node *const *children(const Node *node) noexcept {
    auto *bytes = reinterpret_cast<const std::byte *>(node);
    return reinterpret_cast<Node *const *>(bytes + kChildrenOffset);
  }
  static Node *&child(Node *node, std::size_t index) noexcept {
    return children(node)[index];
  }
  static Node *child(const Node *node, std::size_t index) noexcept {
    return children(node)[index];
  }

  template <class... Args>
  static void construct_entry(Node *node, std::size_t index,
                              Args &&...args) noexcept {
    ::new (static_cast<void *>(entry(node, index)))
        Entry(std::forward<Args>(args)...);
  }
  static void destroy_entry(Node *node, std::size_t index) noexcept {
    entry(node, index)->~Entry();
  }

  Node *allocate_node(bool leaf, Node *parent) noexcept {
    const std::size_t bytes = leaf ? kLeafBytes : kInternalBytes;
    void *storage = alloc_.allocate(bytes, kNodeAlignment);
    if (storage == nullptr)
      return nullptr;
    return ::new (storage) Node{parent, 0, 0, leaf};
  }
  void deallocate_empty_node(Node *node) noexcept {
    if (node == nullptr)
      return;
    HYLE_FATAL_ASSERT(node->count == 0);
    const std::size_t bytes = node->leaf ? kLeafBytes : kInternalBytes;
    node->~Node();
    alloc_.deallocate(node, bytes, kNodeAlignment);
  }
  void deallocate_node(Node *node) noexcept {
    HYLE_FATAL_ASSERT(node != nullptr);
    for (std::size_t i = 0; i < node->count; ++i)
      destroy_entry(node, i);
    node->count = 0;
    deallocate_empty_node(node);
  }

  static constexpr bool kLinearNodeSearch =
      std::is_arithmetic_v<Key> &&
      (std::is_same_v<Compare, std::less<Key>> ||
       std::is_same_v<Compare, std::less<>> ||
       std::is_same_v<Compare, std::greater<Key>> ||
       std::is_same_v<Compare, std::greater<>>);

  template <class Q>
  std::uint16_t node_lower_bound(const Node *node,
                                 const Q &key) const noexcept {
    const Entry *values = entry(node, 0);
    if constexpr (kLinearNodeSearch) {
      std::uint16_t index = 0;
      while (index < node->count &&
             compare_(Policy::key(values[index]), key))
        ++index;
      return index;
    } else {
      std::uint16_t first = 0;
      std::uint16_t last = node->count;
      while (first != last) {
        const std::uint16_t middle =
            static_cast<std::uint16_t>((first + last) / 2);
        if (compare_(Policy::key(values[middle]), key))
          first = static_cast<std::uint16_t>(middle + 1);
        else
          last = middle;
      }
      return first;
    }
  }

  template <class Q>
  std::uint16_t node_upper_bound(const Node *node,
                                 const Q &key) const noexcept {
    const Entry *values = entry(node, 0);
    if constexpr (kLinearNodeSearch) {
      std::uint16_t index = 0;
      while (index < node->count &&
             !compare_(key, Policy::key(values[index])))
        ++index;
      return index;
    } else {
      std::uint16_t first = 0;
      std::uint16_t last = node->count;
      while (first != last) {
        const std::uint16_t middle =
            static_cast<std::uint16_t>((first + last) / 2);
        if (!compare_(key, Policy::key(values[middle])))
          first = static_cast<std::uint16_t>(middle + 1);
        else
          last = middle;
      }
      return first;
    }
  }

  template <class Q>
  NodeSearchResult node_search(const Node *node, const Q &key) const noexcept {
    // std::less/std::greater on integral keys can obtain ordering and equality
    // from one comparison. This is the compare-to shape used by mature B-tree
    // implementations; it saves the reverse comparator call on every level
    // without imposing a three-way-comparison contract on user comparators.
    if constexpr (std::is_integral_v<Key> &&
                  std::is_same_v<std::remove_cvref_t<Q>, Key> &&
                  (std::is_same_v<Compare, std::less<Key>> ||
                   std::is_same_v<Compare, std::less<>>)) {
      std::uint16_t index = 0;
      const Entry *values = entry(node, 0);
      if constexpr (Policy::kIsSet) {
        while (index < node->count) {
          const Key &candidate = Policy::key(values[index]);
          if (candidate >= key)
            return {index, candidate == key};
          ++index;
        }
      } else {
        while (index < node->count) {
          const Key &candidate = Policy::key(values[index]);
          if (candidate >= key)
            return {index, candidate == key};
          ++index;
        }
      }
      return {index, false};
    } else if constexpr (std::is_integral_v<Key> &&
                         std::is_same_v<std::remove_cvref_t<Q>, Key> &&
                         (std::is_same_v<Compare, std::greater<Key>> ||
                          std::is_same_v<Compare, std::greater<>>)) {
      std::uint16_t index = 0;
      const Entry *values = entry(node, 0);
      if constexpr (Policy::kIsSet) {
        while (index < node->count) {
          const Key &candidate = Policy::key(values[index]);
          if (candidate <= key)
            return {index, candidate == key};
          ++index;
        }
      } else {
        while (index < node->count) {
          const Key &candidate = Policy::key(values[index]);
          if (candidate <= key)
            return {index, candidate == key};
          ++index;
        }
      }
      return {index, false};
    } else {
      const std::uint16_t index = node_lower_bound(node, key);
      return {index, index < node->count &&
                         !compare_(key, Policy::key(*entry(node, index)))};
    }
  }

  template <class Q>
  std::pair<Node *, std::uint16_t> find_location(const Q &key) const noexcept {
    Node *node = root_;
    while (node != nullptr) {
      const NodeSearchResult result = node_search(node, key);
      if (result.found)
        return {node, result.index};
      if (node->leaf)
        break;
      node = child(node, result.index);
    }
    return {nullptr, 0};
  }

  template <class Q>
  InsertLocation locate_for_insert(const Q &key) const noexcept {
    Node *node = root_;
    if (node == nullptr)
      return {nullptr, 0, false};
    for (;;) {
      const NodeSearchResult result = node_search(node, key);
      if (result.found)
        return {node, result.index, true};
      if (node->leaf)
        return {node, result.index, false};
      node = child(node, result.index);
    }
  }

  template <class Q>
  std::pair<Node *, std::uint16_t> bound_location(const Q &key,
                                                  bool upper) const noexcept {
    Node *node = root_;
    Node *candidate = nullptr;
    std::uint16_t candidate_index = 0;
    while (node != nullptr) {
      const std::uint16_t index =
          upper ? node_upper_bound(node, key) : node_lower_bound(node, key);
      if (index < node->count) {
        candidate = node;
        candidate_index = index;
      }
      if (node->leaf)
        break;
      node = child(node, index);
    }
    return {candidate, candidate_index};
  }

  std::pair<Node *, std::uint16_t> first_location() const noexcept {
    Node *node = root_;
    if (node == nullptr)
      return {nullptr, 0};
    while (!node->leaf)
      node = child(node, 0);
    return {node, 0};
  }
  std::pair<Node *, std::uint16_t> last_location() const noexcept {
    Node *node = root_;
    if (node == nullptr)
      return {nullptr, 0};
    while (!node->leaf)
      node = child(node, node->count);
    return {node, static_cast<std::uint16_t>(node->count - 1)};
  }

  template <class IteratorNode>
  void advance(IteratorNode *&node, std::uint16_t &index) const noexcept {
    if (!node->leaf) {
      node = child(node, index + 1);
      while (!node->leaf)
        node = child(node, 0);
      index = 0;
      return;
    }
    if (index + 1 < node->count) {
      ++index;
      return;
    }
    while (node->parent != nullptr) {
      const std::uint16_t position = node->position;
      node = node->parent;
      if (position < node->count) {
        index = position;
        return;
      }
    }
    node = nullptr;
    index = 0;
  }

  template <class IteratorNode>
  void retreat(IteratorNode *&node, std::uint16_t &index) const noexcept {
    if (node == nullptr) {
      auto last = last_location();
      node = last.first;
      index = last.second;
      HYLE_FATAL_ASSERT(node != nullptr);
      return;
    }
    if (!node->leaf) {
      node = child(node, index);
      while (!node->leaf)
        node = child(node, node->count);
      index = static_cast<std::uint16_t>(node->count - 1);
      return;
    }
    if (index != 0) {
      --index;
      return;
    }
    while (node->parent != nullptr) {
      const std::uint16_t position = node->position;
      node = node->parent;
      if (position != 0) {
        index = static_cast<std::uint16_t>(position - 1);
        return;
      }
    }
    HYLE_FATAL_ASSERT(false && "decrementing begin() is invalid");
  }

  static void insert_entry(Node *node, std::uint16_t index,
                           Entry candidate) noexcept {
    HYLE_FATAL_ASSERT(node->count < kCapacity && index <= node->count);
    if constexpr (std::is_trivially_copyable_v<Entry>) {
      std::memmove(static_cast<void *>(entry(node, index + 1)),
                   static_cast<const void *>(entry(node, index)),
                   (node->count - index) * sizeof(Entry));
      std::memcpy(static_cast<void *>(entry(node, index)),
                  static_cast<const void *>(std::addressof(candidate)),
                  sizeof(Entry));
    } else {
      for (std::size_t i = node->count; i > index; --i) {
        construct_entry(node, i, std::move(*entry(node, i - 1)));
        destroy_entry(node, i - 1);
      }
      construct_entry(node, index, std::move(candidate));
    }
    ++node->count;
  }

  static void remove_entry(Node *node, std::uint16_t index) noexcept {
    HYLE_FATAL_ASSERT(index < node->count);
    if constexpr (std::is_trivially_copyable_v<Entry>) {
      std::memmove(static_cast<void *>(entry(node, index)),
                   static_cast<const void *>(entry(node, index + 1)),
                   (node->count - index - 1) * sizeof(Entry));
    } else {
      destroy_entry(node, index);
      for (std::size_t i = index; i + 1 < node->count; ++i) {
        construct_entry(node, i, std::move(*entry(node, i + 1)));
        destroy_entry(node, i + 1);
      }
    }
    --node->count;
  }

  static void remove_entries(Node *node, std::uint16_t index,
                             std::uint16_t count) noexcept {
    HYLE_FATAL_ASSERT(count != 0 && index + count <= node->count);
    const std::uint16_t old_count = node->count;
    if constexpr (std::is_trivially_copyable_v<Entry>) {
      std::memmove(static_cast<void *>(entry(node, index)),
                   static_cast<const void *>(entry(node, index + count)),
                   (old_count - index - count) * sizeof(Entry));
    } else {
      for (std::size_t i = index; i < static_cast<std::size_t>(index) + count;
           ++i)
        destroy_entry(node, i);
      for (std::size_t i = static_cast<std::size_t>(index) + count;
           i < old_count; ++i) {
        construct_entry(node, i - count, std::move(*entry(node, i)));
        destroy_entry(node, i);
      }
    }
    node->count = static_cast<std::uint16_t>(old_count - count);
  }

  static void replace_entry(Node *node, std::uint16_t index,
                            Entry replacement) noexcept {
    if constexpr (std::is_trivially_copyable_v<Entry>) {
      std::memcpy(static_cast<void *>(entry(node, index)),
                  static_cast<const void *>(std::addressof(replacement)),
                  sizeof(Entry));
    } else {
      destroy_entry(node, index);
      construct_entry(node, index, std::move(replacement));
    }
  }

  static void update_child_positions(Node *node, std::size_t first) noexcept {
    if (node->leaf)
      return;
    for (std::size_t i = first; i <= node->count; ++i) {
      child(node, i)->parent = node;
      child(node, i)->position = static_cast<std::uint16_t>(i);
    }
  }

  static void split_child(Node *parent, std::uint16_t child_index,
                          Node *sibling, std::uint16_t insert_index) noexcept {
    Node *source = child(parent, child_index);
    HYLE_FATAL_ASSERT(parent->count < kCapacity && source->count == kCapacity &&
                      sibling->leaf == source->leaf &&
                      insert_index <= kCapacity);
    const std::uint16_t old_parent_count = parent->count;

    for (std::size_t i = old_parent_count + 1;
         i > static_cast<std::size_t>(child_index) + 1; --i)
      child(parent, i) = child(parent, i - 1);
    child(parent, child_index + 1) = sibling;

    // Bias an edge split toward the incoming key. Monotonic event streams then
    // leave the old page full and the new page empty instead of allocating a
    // half-empty page after every kCapacity/2 appends. Interior insertion keeps
    // the conventional even split.
    const std::uint16_t right_count =
        insert_index == 0
            ? static_cast<std::uint16_t>(kCapacity - 1)
            : (insert_index == kCapacity ? 0
                                         : static_cast<std::uint16_t>(kMinKeys));
    const std::uint16_t left_count = static_cast<std::uint16_t>(
        kCapacity - static_cast<std::size_t>(right_count) - 1);
    for (std::size_t i = 0; i < right_count; ++i) {
      construct_entry(sibling, i,
                      std::move(*entry(source, left_count + 1 + i)));
      destroy_entry(source, left_count + 1 + i);
    }
    sibling->count = right_count;

    Entry median(std::move(*entry(source, left_count)));
    destroy_entry(source, left_count);
    source->count = left_count;
    insert_entry(parent, child_index, std::move(median));

    if (!source->leaf) {
      for (std::size_t i = 0; i <= right_count; ++i)
        child(sibling, i) = child(source, left_count + 1 + i);
      update_child_positions(sibling, 0);
    }
    update_child_positions(parent, child_index);
  }

  bool make_room_for_insert(Node *&node, std::uint16_t &index) noexcept {
    if (node->count < kCapacity)
      return true;
    if (node != root_) {
      Node *parent = node->parent;
      const std::uint16_t position = node->position;
      Node *left = position == 0 ? nullptr : child(parent, position - 1);
      Node *right =
          position == parent->count ? nullptr : child(parent, position + 1);
      const bool can_use_left =
          left != nullptr && left->count < kCapacity &&
          (index != 0 || static_cast<std::size_t>(left->count) + 1 < kCapacity);
      const bool can_use_right =
          right != nullptr && right->count < kCapacity &&
          (index != node->count ||
           static_cast<std::size_t>(right->count) + 1 < kCapacity);

      // Prefer a neighboring node over allocation/split. This is the mature
      // B-tree shape: mutable indexes stay densely packed under churn instead
      // of leaving every freshly split node permanently half full.
      const auto rebalance_into_left = [&]() noexcept {
        const std::uint16_t old_left_count = left->count;
        const std::uint16_t moves = static_cast<std::uint16_t>(std::max(
            std::size_t{1},
            (kCapacity - left->count) /
                (std::size_t{1} + (index < kCapacity ? 1u : 0u))));
        borrow_from_right_n(parent, static_cast<std::uint16_t>(position - 1),
                            moves);
        if (moves > index) {
          node = left;
          index = static_cast<std::uint16_t>(old_left_count + 1 + index);
        } else {
          index = static_cast<std::uint16_t>(index - moves);
        }
      };
      const auto rebalance_into_right = [&]() noexcept {
        const std::uint16_t old_count = node->count;
        const std::uint16_t moves = static_cast<std::uint16_t>(std::max(
            std::size_t{1},
            (kCapacity - right->count) /
                (std::size_t{1} + (index > 0 ? 1u : 0u))));
        borrow_from_left_n(parent, static_cast<std::uint16_t>(position + 1),
                           moves);
        const std::uint16_t remaining =
            static_cast<std::uint16_t>(old_count - moves);
        if (index > remaining) {
          node = right;
          index = static_cast<std::uint16_t>(index - remaining - 1);
        }
      };
      if (can_use_left && (index <= kMinKeys || !can_use_right)) {
        rebalance_into_left();
        return true;
      }
      if (can_use_right) {
        rebalance_into_right();
        return true;
      }
      if (can_use_left) {
        rebalance_into_left();
        return true;
      }
    }
    if (node == root_) {
      Node *new_root = allocate_node(false, nullptr);
      Node *sibling = allocate_node(node->leaf, new_root);
      if (new_root == nullptr || sibling == nullptr) {
        deallocate_empty_node(new_root);
        deallocate_empty_node(sibling);
        return false;
      }
      child(new_root, 0) = node;
      node->parent = new_root;
      node->position = 0;
      split_child(new_root, 0, sibling, index);
      root_ = new_root;
      const std::uint16_t left_count = node->count;
      if (index > left_count) {
        node = sibling;
        index = static_cast<std::uint16_t>(index - left_count - 1);
      }
      return true;
    }

    // Reserve the node needed at this level before a recursive parent split.
    // Thus an allocation failure cannot occur after an ancestor was already
    // changed: failed insertion is a true structural no-op.
    Node *sibling = allocate_node(node->leaf, node->parent);
    if (sibling == nullptr)
      return false;
    Node *parent = node->parent;
    std::uint16_t parent_index = node->position;
    if (!make_room_for_insert(parent, parent_index)) {
      deallocate_empty_node(sibling);
      return false;
    }

    parent = node->parent;
    split_child(parent, node->position, sibling, index);
    const std::uint16_t left_count = node->count;
    if (index > left_count) {
      node = sibling;
      index = static_cast<std::uint16_t>(index - left_count - 1);
    }
    return true;
  }

  static void borrow_from_left(Node *parent,
                               std::uint16_t child_index) noexcept {
    Node *left = child(parent, child_index - 1);
    Node *target = child(parent, child_index);
    const std::uint16_t left_old_count = left->count;
    const std::uint16_t target_old_count = target->count;

    Entry separator(std::move(*entry(parent, child_index - 1)));
    destroy_entry(parent, child_index - 1);
    construct_entry(parent, child_index - 1,
                    std::move(*entry(left, left_old_count - 1)));
    destroy_entry(left, left_old_count - 1);
    --left->count;
    insert_entry(target, 0, std::move(separator));

    if (!target->leaf) {
      for (std::size_t i = target_old_count + 1; i != 0; --i)
        child(target, i) = child(target, i - 1);
      child(target, 0) = child(left, left_old_count);
      update_child_positions(target, 0);
    }
  }

  static void borrow_from_right(Node *parent,
                                std::uint16_t child_index) noexcept {
    Node *target = child(parent, child_index);
    Node *right = child(parent, child_index + 1);
    const std::uint16_t target_old_count = target->count;
    const std::uint16_t right_old_count = right->count;

    Entry separator(std::move(*entry(parent, child_index)));
    destroy_entry(parent, child_index);
    construct_entry(parent, child_index, std::move(*entry(right, 0)));
    insert_entry(target, target->count, std::move(separator));
    remove_entry(right, 0);

    if (!target->leaf) {
      child(target, target_old_count + 1) = child(right, 0);
      child(target, target_old_count + 1)->parent = target;
      child(target, target_old_count + 1)->position =
          static_cast<std::uint16_t>(target_old_count + 1);
      for (std::size_t i = 0; i < right_old_count; ++i)
        child(right, i) = child(right, i + 1);
      update_child_positions(right, 0);
    }
  }

  static void borrow_from_left_n(Node *parent, std::uint16_t child_index,
                                 std::uint16_t moves) noexcept {
    Node *left = child(parent, child_index - 1);
    Node *target = child(parent, child_index);
    const std::uint16_t left_old_count = left->count;
    const std::uint16_t target_old_count = target->count;
    HYLE_FATAL_ASSERT(moves != 0 && moves <= left_old_count &&
                      static_cast<std::size_t>(target_old_count) + moves <=
                          kCapacity);

    for (std::size_t i = target_old_count; i != 0; --i) {
      construct_entry(target, i - 1 + moves, std::move(*entry(target, i - 1)));
      destroy_entry(target, i - 1);
    }

    Entry separator(std::move(*entry(parent, child_index - 1)));
    destroy_entry(parent, child_index - 1);
    construct_entry(target, moves - 1, std::move(separator));
    for (std::size_t i = 0; i + 1 < moves; ++i) {
      const std::size_t source = left_old_count - moves + 1 + i;
      construct_entry(target, i, std::move(*entry(left, source)));
      destroy_entry(left, source);
    }
    const std::size_t new_separator = left_old_count - moves;
    construct_entry(parent, child_index - 1,
                    std::move(*entry(left, new_separator)));
    destroy_entry(left, new_separator);

    left->count = static_cast<std::uint16_t>(left_old_count - moves);
    target->count = static_cast<std::uint16_t>(target_old_count + moves);
    if (!target->leaf) {
      for (std::size_t i = target_old_count + 1; i != 0; --i)
        child(target, i - 1 + moves) = child(target, i - 1);
      for (std::size_t i = 0; i < moves; ++i)
        child(target, i) = child(left, left->count + 1 + i);
      update_child_positions(target, 0);
    }
  }

  static void borrow_from_right_n(Node *parent, std::uint16_t child_index,
                                  std::uint16_t moves) noexcept {
    Node *target = child(parent, child_index);
    Node *right = child(parent, child_index + 1);
    const std::uint16_t target_old_count = target->count;
    const std::uint16_t right_old_count = right->count;
    HYLE_FATAL_ASSERT(moves != 0 && moves <= right_old_count &&
                      static_cast<std::size_t>(target_old_count) + moves <=
                          kCapacity);

    Entry separator(std::move(*entry(parent, child_index)));
    destroy_entry(parent, child_index);
    construct_entry(target, target_old_count, std::move(separator));
    for (std::size_t i = 0; i + 1 < moves; ++i) {
      construct_entry(target, target_old_count + 1 + i,
                      std::move(*entry(right, i)));
      destroy_entry(right, i);
    }
    construct_entry(parent, child_index, std::move(*entry(right, moves - 1)));
    destroy_entry(right, moves - 1);
    for (std::size_t i = moves; i < right_old_count; ++i) {
      construct_entry(right, i - moves, std::move(*entry(right, i)));
      destroy_entry(right, i);
    }

    target->count = static_cast<std::uint16_t>(target_old_count + moves);
    right->count = static_cast<std::uint16_t>(right_old_count - moves);
    if (!target->leaf) {
      for (std::size_t i = 0; i < moves; ++i)
        child(target, target_old_count + 1 + i) = child(right, i);
      for (std::size_t i = moves; i <= right_old_count; ++i)
        child(right, i - moves) = child(right, i);
      update_child_positions(target, target_old_count + 1);
      update_child_positions(right, 0);
    }
  }

  Node *merge_children(Node *parent, std::uint16_t index) noexcept {
    Node *left = child(parent, index);
    Node *right = child(parent, index + 1);
    const std::uint16_t left_old_count = left->count;
    const std::uint16_t right_count = right->count;
    const std::uint16_t parent_old_count = parent->count;
    HYLE_FATAL_ASSERT(static_cast<std::size_t>(left_old_count) + 1 +
                          right_count <=
                      kCapacity);

    construct_entry(left, left_old_count, std::move(*entry(parent, index)));
    destroy_entry(parent, index);
    for (std::size_t i = 0; i < right_count; ++i) {
      construct_entry(left, left_old_count + 1 + i,
                      std::move(*entry(right, i)));
      destroy_entry(right, i);
    }
    left->count = static_cast<std::uint16_t>(left_old_count + 1 + right_count);
    right->count = 0;

    if (!left->leaf) {
      for (std::size_t i = 0; i <= right_count; ++i)
        child(left, left_old_count + 1 + i) = child(right, i);
      update_child_positions(left, left_old_count + 1);
    }

    for (std::size_t i = index; i + 1 < parent_old_count; ++i) {
      construct_entry(parent, i, std::move(*entry(parent, i + 1)));
      destroy_entry(parent, i + 1);
    }
    --parent->count;
    for (std::size_t i = index + 1; i < parent_old_count; ++i)
      child(parent, i) = child(parent, i + 1);
    update_child_positions(parent, index);
    deallocate_empty_node(right);
    return left;
  }

  void rebalance_after_delete(Node *node, Node **cursor_node = nullptr,
                              std::uint16_t *cursor_index = nullptr) noexcept {
    while (node != root_ && node->count < kMinKeys) {
      Node *parent = node->parent;
      const std::uint16_t position = node->position;
      Node *left = position == 0 ? nullptr : child(parent, position - 1);
      Node *right =
          position == parent->count ? nullptr : child(parent, position + 1);

      // Prefer removing a node before moving values between siblings. Apart
      // from doing less work, this preserves density under long-lived churn.
      if (left != nullptr &&
          static_cast<std::size_t>(left->count) + 1 + node->count <=
              kCapacity) {
        const std::uint16_t left_old_count = left->count;
        if (cursor_node != nullptr && *cursor_node == node) {
          *cursor_node = left;
          *cursor_index =
              static_cast<std::uint16_t>(left_old_count + 1 + *cursor_index);
        }
        merge_children(parent, static_cast<std::uint16_t>(position - 1));
        node = parent;
        continue;
      }
      if (right != nullptr &&
          static_cast<std::size_t>(node->count) + 1 + right->count <=
              kCapacity) {
        merge_children(parent, position);
        node = parent;
        continue;
      }

      const bool cursor_is_here =
          cursor_node != nullptr && cursor_index != nullptr &&
          *cursor_node == node;
      const bool deleted_at_front =
          cursor_is_here && node->count != 0 && *cursor_index == 0;
      const bool deleted_at_back =
          cursor_is_here && node->count != 0 && *cursor_index == node->count;

      // Repeated erase(begin())/front-expiry should not pull values from the
      // right only to erase them next. Symmetrically preserve a sparse right
      // edge while deleting from the back. A later interior deletion, insert,
      // or empty-node removal restores ordinary density.
      if (right != nullptr && right->count > kMinKeys &&
          !deleted_at_front) {
        std::uint16_t moves = static_cast<std::uint16_t>(
            (right->count - node->count) / 2);
        moves = std::min<std::uint16_t>(
            moves, static_cast<std::uint16_t>(right->count - 1));
        HYLE_FATAL_ASSERT(moves != 0);
        borrow_from_right_n(parent, position, moves);
        break;
      }
      if (left != nullptr && left->count > kMinKeys && !deleted_at_back) {
        std::uint16_t moves = static_cast<std::uint16_t>(
            (left->count - node->count) / 2);
        moves = std::min<std::uint16_t>(
            moves, static_cast<std::uint16_t>(left->count - 1));
        HYLE_FATAL_ASSERT(moves != 0);
        borrow_from_left_n(parent, position, moves);
        if (cursor_is_here)
          *cursor_index = static_cast<std::uint16_t>(*cursor_index + moves);
      }
      break;
    }
    collapse_empty_root(cursor_node, cursor_index);
  }

  iterator erase_range_to_key(const_iterator first, const_iterator last)
    requires std::is_copy_constructible_v<Key>
  {
    if (first == begin() && last == end()) {
      clear();
      return end();
    }

    // Preserve only the boundary key, not every element count. Successful
    // mutations invalidate `last`, while the owning boundary lets each leaf
    // remove its complete local run in one move. Move-only keys use the
    // allocation-free counted fallback in the public overload above.
    std::optional<Key> stop;
    if (last != end())
      stop.emplace(Policy::key(*entry(last.node_, last.index_)));

    const auto before_stop = [&](const Node *node,
                                 std::uint16_t index) noexcept {
      return !stop || compare_(Policy::key(*entry(node, index)), *stop);
    };
    while (first != end() && before_stop(first.node_, first.index_)) {
      if (!first.node_->leaf) {
        first = erase(first);
        continue;
      }

      Node *node = const_cast<Node *>(first.node_);
      const std::uint16_t boundary =
          stop ? node_lower_bound(node, *stop) : node->count;
      HYLE_FATAL_ASSERT(boundary > first.index_);
      const std::uint16_t to_erase =
          static_cast<std::uint16_t>(boundary - first.index_);
      Node *next_node = node;
      std::uint16_t next_index = first.index_;
      remove_entries(node, first.index_, to_erase);
      size_ -= to_erase;

      const bool separator_is_in_range =
          node->count == 0 && node != root_ &&
          node->position < node->parent->count &&
          (!stop ||
           compare_(Policy::key(*entry(node->parent, node->position)), *stop));
      if (separator_is_in_range) {
        Node *parent = node->parent;
        const std::uint16_t position = node->position;
        remove_entry(parent, position);
        --size_;
        for (std::size_t child_index = position; child_index <= parent->count;
             ++child_index)
          child(parent, child_index) = child(parent, child_index + 1);
        update_child_positions(parent, position);
        next_node = child(parent, position);
        next_index = 0;
        deallocate_empty_node(node);
        rebalance_after_delete(parent, &next_node, &next_index);
      } else {
        rebalance_after_delete(node, &next_node, &next_index);
      }
      if (next_node != nullptr && next_index == next_node->count) {
        HYLE_FATAL_ASSERT(next_node->count != 0);
        next_index = static_cast<std::uint16_t>(next_node->count - 1);
        advance(next_node, next_index);
      }
      first = const_iterator(this, next_node, next_index);
    }
    return iterator(this, const_cast<Node *>(first.node_), first.index_);
  }

  iterator erase_and_get_next(Node *node, std::uint16_t index) noexcept {
    const bool internal_delete = !node->leaf;
    if (internal_delete) {
      Node *predecessor = child(node, index);
      while (!predecessor->leaf)
        predecessor = child(predecessor, predecessor->count);
      const std::uint16_t predecessor_index =
          static_cast<std::uint16_t>(predecessor->count - 1);
      Entry replacement(std::move(*entry(predecessor, predecessor_index)));
      destroy_entry(predecessor, predecessor_index);
      --predecessor->count;
      replace_entry(node, index, std::move(replacement));
      node = predecessor;
      index = predecessor_index;
    } else {
      remove_entry(node, index);
    }
    --size_;
    Node *next_node = node;
    std::uint16_t next_index = index;
    rebalance_after_delete(node, &next_node, &next_index);
    if (next_node == nullptr)
      return end();
    if (next_index == next_node->count) {
      HYLE_FATAL_ASSERT(next_node->count != 0);
      next_index = static_cast<std::uint16_t>(next_node->count - 1);
      advance(next_node, next_index);
    }
    iterator next(this, next_node, next_index);
    if (internal_delete && next != end())
      ++next;
    return next;
  }

  void erase_location(Node *node, std::uint16_t index) noexcept {
    (void)erase_and_get_next(node, index);
  }

  void collapse_empty_root(Node **cursor_node = nullptr,
                           std::uint16_t *cursor_index = nullptr) noexcept {
    if (root_ == nullptr || root_->count != 0)
      return;
    Node *old_root = root_;
    if (old_root->leaf) {
      root_ = nullptr;
    } else {
      root_ = child(old_root, 0);
      root_->parent = nullptr;
      root_->position = 0;
    }
    if (cursor_node != nullptr && *cursor_node == old_root) {
      *cursor_node = root_;
      *cursor_index = 0;
    }
    deallocate_empty_node(old_root);
  }

  void destroy_subtree(Node *node) noexcept {
    if (node == nullptr)
      return;
    if (!node->leaf)
      for (std::size_t i = 0; i <= node->count; ++i)
        destroy_subtree(child(node, i));
    deallocate_node(node);
  }

  void copy_from(const BTreeCore &other)
    requires std::is_copy_constructible_v<Entry>
  {
    root_ = clone_subtree(other.root_, nullptr);
    size_ = other.size_;
  }

  Node *clone_subtree(const Node *source, Node *parent)
    requires std::is_copy_constructible_v<Entry>
  {
    if (source == nullptr)
      return nullptr;
    Node *copy = allocate_node(source->leaf, parent);
    HYLE_FATAL_ASSERT(copy != nullptr);
    copy->position = source->position;
    std::size_t copied_children = 0;
    try {
      for (std::size_t index = 0; index < source->count; ++index) {
        ::new (static_cast<void *>(entry(copy, index)))
            Entry(*entry(source, index));
        ++copy->count;
      }
      if (!source->leaf) {
        for (; copied_children <= source->count; ++copied_children)
          child(copy, copied_children) =
              clone_subtree(child(source, copied_children), copy);
      }
    } catch (...) {
      if (!source->leaf)
        for (std::size_t index = 0; index < copied_children; ++index)
          destroy_subtree(child(copy, index));
      deallocate_node(copy);
      throw;
    }
    return copy;
  }

  Node *root_ = nullptr;
  std::size_t size_ = 0;
  HYLE_NO_UNIQUE_ADDRESS Compare compare_{};
  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
};

} // namespace detail

// Ordered unique-key map for mutable range indexes and deterministic iteration.
// It follows the cache-oriented Abseil B-tree family rather than std::map's
// one-allocation-per-key red-black tree. Any successful mutation may move slots
// and invalidates all iterators/references; use ConcurrentNodeHashMap instead
// when stable identity is the primary contract. This type is not thread-safe.
template <class K, class V, class Compare = std::less<>,
          class Alloc = mem::SystemAllocator>
class BTreeMap
    : public detail::BTreeCore<detail::BTreeMapPolicy<K, V>, Compare, Alloc> {
  using Base = detail::BTreeCore<detail::BTreeMapPolicy<K, V>, Compare, Alloc>;
  using Entry = BTreeMapEntry<K, V>;

public:
  using typename Base::const_iterator;
  using typename Base::InsertResult;
  using typename Base::iterator;
  using typename Base::value_type;

  BTreeMap() = default;
  explicit BTreeMap(Compare compare, Alloc alloc = Alloc{})
      : Base(std::move(compare), std::move(alloc)) {}

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  BTreeMap(It first, Sentinel last) {
    insert(first, last);
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  BTreeMap(It first, Sentinel last, Compare compare, Alloc alloc = Alloc{})
      : Base(std::move(compare), std::move(alloc)) {
    insert(first, last);
  }

  BTreeMap(std::initializer_list<std::pair<K, V>> values) {
    insert(values.begin(), values.end());
  }

  BTreeMap(std::initializer_list<std::pair<K, V>> values, Compare compare,
           Alloc alloc = Alloc{})
      : Base(std::move(compare), std::move(alloc)) {
    insert(values.begin(), values.end());
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(K key, Args &&...args) {
    auto [position, result] = Base::try_emplace_entry(key, [&]() -> Entry {
      return Entry(std::move(key), std::in_place, std::forward<Args>(args)...);
    });
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return {position, result == InsertResult::kInserted};
  }

  InsertResult try_insert(K key, V value) {
    Entry candidate(std::move(key), std::in_place, std::move(value));
    return Base::try_insert_entry(std::move(candidate)).second;
  }

  std::pair<iterator, bool> insert(K key, V value) {
    Entry candidate(std::move(key), std::in_place, std::move(value));
    auto [position, result] = Base::try_insert_entry(std::move(candidate));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return {position, result == InsertResult::kInserted};
  }

  InsertResult try_insert_or_assign(K key, V value) {
    Entry candidate(std::move(key), std::in_place, std::move(value));
    return Base::try_insert_entry(
               std::move(candidate),
               [](Entry &existing, Entry &incoming) -> InsertResult {
                 existing.value() = std::move(incoming.value());
                 return InsertResult::kAssigned;
               })
        .second;
  }

  bool insert_or_assign(K key, V value) {
    const InsertResult result =
        try_insert_or_assign(std::move(key), std::move(value));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return result == InsertResult::kInserted;
  }

  V &operator[](K key) { return try_emplace(std::move(key)).first->value(); }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  void insert(It first, Sentinel last) {
    for (; first != last; ++first)
      insert(first->first, first->second);
  }

  void insert(std::initializer_list<std::pair<K, V>> values) {
    insert(values.begin(), values.end());
  }

  friend void swap(BTreeMap &left, BTreeMap &right) noexcept {
    left.swap(right);
  }
};

// Ordered unique-key set. It shares the B-tree algorithms with BTreeMap but
// stores only K in each slot; there is no dummy mapped value.
template <class K, class Compare = std::less<>,
          class Alloc = mem::SystemAllocator>
class BTreeSet
    : public detail::BTreeCore<detail::BTreeSetPolicy<K>, Compare, Alloc> {
  using Base = detail::BTreeCore<detail::BTreeSetPolicy<K>, Compare, Alloc>;
  using Entry = detail::BTreeSetEntry<K>;

public:
  using typename Base::InsertResult;
  using typename Base::iterator;

  BTreeSet() = default;
  explicit BTreeSet(Compare compare, Alloc alloc = Alloc{})
      : Base(std::move(compare), std::move(alloc)) {}

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  BTreeSet(It first, Sentinel last) {
    insert(first, last);
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  BTreeSet(It first, Sentinel last, Compare compare, Alloc alloc = Alloc{})
      : Base(std::move(compare), std::move(alloc)) {
    insert(first, last);
  }

  BTreeSet(std::initializer_list<K> values) {
    insert(values.begin(), values.end());
  }

  BTreeSet(std::initializer_list<K> values, Compare compare,
           Alloc alloc = Alloc{})
      : Base(std::move(compare), std::move(alloc)) {
    insert(values.begin(), values.end());
  }

  InsertResult try_insert(K key) {
    return Base::try_insert_entry(Entry(std::move(key))).second;
  }

  std::pair<iterator, bool> insert(K key) {
    auto [position, result] = Base::try_insert_entry(Entry(std::move(key)));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return {position, result == InsertResult::kInserted};
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  void insert(It first, Sentinel last) {
    for (; first != last; ++first)
      insert(*first);
  }

  void insert(std::initializer_list<K> values) {
    insert(values.begin(), values.end());
  }

  friend void swap(BTreeSet &left, BTreeSet &right) noexcept {
    left.swap(right);
  }
};

} // namespace hyle::container
