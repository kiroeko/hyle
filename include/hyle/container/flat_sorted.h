#pragma once

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Promise that an input range is strictly ordered by the destination
// comparator and contains no equivalent keys. The promise buys linear bulk
// construction; violating it is a contract failure, not a request to sort.
struct sorted_unique_t {
  explicit constexpr sorted_unique_t() noexcept = default;
};
inline constexpr sorted_unique_t sorted_unique{};

// Public map element. The key has no mutable accessor, so contiguous iterators
// cannot silently break ordering; the mapped value remains normally mutable.
template <class K, class V> class FlatMapEntry {
public:
  template <class KeyArg, class... Args>
    requires std::constructible_from<K, KeyArg>
  FlatMapEntry(KeyArg &&key, std::in_place_t, Args &&...args) noexcept(
      std::is_nothrow_constructible_v<K, KeyArg> &&
      std::is_nothrow_constructible_v<V, Args...>)
      : key_(std::forward<KeyArg>(key)),
        value_(std::forward<Args>(args)...) {}

  FlatMapEntry(const FlatMapEntry &) = default;
  FlatMapEntry &operator=(const FlatMapEntry &) = delete;
  FlatMapEntry(FlatMapEntry &&) noexcept(
      std::is_nothrow_move_constructible_v<K> &&
      std::is_nothrow_move_constructible_v<V>) = default;
  FlatMapEntry &operator=(FlatMapEntry &&) = delete;

  const K &key() const noexcept { return key_; }
  V &value() noexcept { return value_; }
  const V &value() const noexcept { return value_; }

  friend bool operator==(const FlatMapEntry &, const FlatMapEntry &)
    requires(std::equality_comparable<K> && std::equality_comparable<V>)
  = default;

private:
  K key_;
  HYLE_NO_UNIQUE_ADDRESS V value_;
};

enum class FlatInsertResult : std::uint8_t {
  kInserted,
  kAlreadyPresent,
  kAssigned,
  kOutOfMemory,
};

namespace detail {

template <class Compare, class K, class Q>
concept FlatComparable =
    std::predicate<const Compare &, const K &, const Q &> &&
    std::predicate<const Compare &, const Q &, const K &>;

template <class Pair>
decltype(auto) flat_range_key(Pair &&pair) {
  if constexpr (requires { std::forward<Pair>(pair).key(); })
    return std::forward<Pair>(pair).key();
  else
    return (std::forward<Pair>(pair).first);
}

template <class Pair>
decltype(auto) flat_range_value(Pair &&pair) {
  if constexpr (requires { std::forward<Pair>(pair).value(); })
    return std::forward<Pair>(pair).value();
  else
    return (std::forward<Pair>(pair).second);
}

template <class Pair>
concept FlatMapRangeValue =
    (requires(Pair &&pair) {
      std::forward<Pair>(pair).key();
      std::forward<Pair>(pair).value();
    }) ||
    (requires(Pair &&pair) {
      std::forward<Pair>(pair).first;
      std::forward<Pair>(pair).second;
    });

template <class Entry, class K, class KeyOf, class Compare, class Alloc>
class FlatSortedStorage {
  // Reuse the established non-trivial sequence engine. One inline entry keeps
  // the empty/singleton registry allocation-free without inventing a second
  // vector or relying on a zero-capacity raw-pointer sentinel.
  using Sequence = SmallVector<Entry, 1, Alloc>;

  static_assert(std::is_nothrow_move_constructible_v<Entry> &&
                    std::is_nothrow_destructible_v<Entry>,
                "flat ordered entries require nothrow move/destruction");
  static_assert(std::is_nothrow_move_constructible_v<Compare> &&
                    std::is_nothrow_move_assignable_v<Compare> &&
                    std::is_nothrow_swappable_v<Compare>,
                "flat ordered comparator movement/swap must be noexcept");

public:
  using size_type = std::size_t;

  FlatSortedStorage() = default;
  explicit FlatSortedStorage(Compare compare, Alloc alloc = Alloc{})
      : entries_(std::move(alloc)), compare_(std::move(compare)) {}

  FlatSortedStorage(const FlatSortedStorage &) = default;
  FlatSortedStorage &operator=(const FlatSortedStorage &) = default;
  FlatSortedStorage(FlatSortedStorage &&) noexcept = default;
  FlatSortedStorage &operator=(FlatSortedStorage &&) noexcept = default;

  size_type size() const noexcept { return entries_.size(); }
  size_type capacity() const noexcept { return entries_.capacity(); }
  bool empty() const noexcept { return entries_.empty(); }
  Entry *data() noexcept { return entries_.data(); }
  const Entry *data() const noexcept { return entries_.data(); }
  Entry *iterator_at(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    return index == 0 ? data() : data() + index;
  }
  const Entry *iterator_at(size_type index) const noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    return index == 0 ? data() : data() + index;
  }
  size_type iterator_index(const Entry *position,
                           bool allow_end) const noexcept {
    HYLE_FATAL_ASSERT(position >= data() &&
                      (allow_end ? position <= data() + size()
                                 : position < data() + size()));
    return static_cast<size_type>(position - data());
  }
  const Compare &key_comp() const noexcept { return compare_; }
  const Alloc &allocator() const noexcept { return entries_.allocator(); }

  bool try_reserve(size_type count) noexcept {
    return entries_.try_reserve(count);
  }
  void reserve(size_type count) { entries_.reserve(count); }
  bool try_shrink_to_fit() noexcept { return entries_.try_shrink_to_fit(); }
  void shrink_to_fit() { entries_.shrink_to_fit(); }
  void clear() noexcept { entries_.clear(); }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  size_type lower_bound_index(const Q &key) const {
    if constexpr (std::is_same_v<Entry, K>) {
      // FlatSet stores K directly, so keep the projection-free index loop.
      // Map entries use the standard algorithm below: its comparator/projector
      // shape generates more balanced code across MSVC and GCC.
      size_type first = 0;
      size_type count = entries_.size();
      while (count != 0) {
        const size_type step = count / 2;
        const size_type middle = first + step;
        if (compare_(entries_[middle], key)) {
          first = middle + 1;
          count -= step + 1;
        } else {
          count = step;
        }
      }
      return first;
    } else {
      const auto position = std::lower_bound(
          entries_.begin(), entries_.end(), key,
          [this](const Entry &entry, const Q &candidate) {
            return compare_(KeyOf{}(entry), candidate);
          });
      return static_cast<size_type>(position - entries_.begin());
    }
  }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  size_type upper_bound_index(const Q &key) const {
    if constexpr (std::is_same_v<Entry, K>) {
      size_type first = 0;
      size_type count = entries_.size();
      while (count != 0) {
        const size_type step = count / 2;
        const size_type middle = first + step;
        if (!compare_(key, entries_[middle])) {
          first = middle + 1;
          count -= step + 1;
        } else {
          count = step;
        }
      }
      return first;
    } else {
      const auto position = std::upper_bound(
          entries_.begin(), entries_.end(), key,
          [this](const Q &candidate, const Entry &entry) {
            return compare_(candidate, KeyOf{}(entry));
          });
      return static_cast<size_type>(position - entries_.begin());
    }
  }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  bool equivalent_at(size_type index, const Q &key) const {
    return index != size() && !compare_(key, KeyOf{}(entries_[index]));
  }

  std::pair<size_type, FlatInsertResult> try_insert_unique(Entry candidate) {
    const size_type index = lower_bound_index(KeyOf{}(candidate));
    if (equivalent_at(index, KeyOf{}(candidate)))
      return {index, FlatInsertResult::kAlreadyPresent};
    if (!try_insert_at(index, std::move(candidate)))
      return {size(), FlatInsertResult::kOutOfMemory};
    return {index, FlatInsertResult::kInserted};
  }

  bool try_insert_at(size_type index, Entry candidate) {
    HYLE_FATAL_ASSERT(index <= size());
    if (size() == std::numeric_limits<size_type>::max() ||
        !entries_.try_reserve(size() + 1))
      return false;
    auto position = index == 0 ? entries_.begin() : entries_.begin() + index;
    entries_.emplace(position, std::move(candidate));
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel,
            class Construct>
  bool try_append_sorted(It first, Sentinel last, Construct construct) {
    // This storage is a fresh replacement. SmallVector may therefore reserve
    // once and directly construct the projected entries without alias checks
    // or one capacity branch per element.
    return SmallVectorAccess::try_append_construct_unaliased(
        entries_, std::move(first), std::move(last), std::move(construct));
  }

  void erase_index(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index < size());
    auto position = index == 0 ? entries_.begin() : entries_.begin() + index;
    entries_.erase(position);
  }
  void erase_indices(size_type first, size_type last) noexcept {
    HYLE_FATAL_ASSERT(first <= last && last <= size());
    auto first_position =
        first == 0 ? entries_.begin() : entries_.begin() + first;
    auto last_position =
        last == 0 ? entries_.begin() : entries_.begin() + last;
    entries_.erase(first_position, last_position);
  }

  void swap(FlatSortedStorage &other) noexcept {
    using std::swap;
    swap(compare_, other.compare_);
    entries_.swap(other.entries_);
  }

private:
  Sequence entries_{};
  HYLE_NO_UNIQUE_ADDRESS Compare compare_{};
};

template <class K, class V> struct FlatMapKeyOf {
  const K &operator()(const FlatMapEntry<K, V> &entry) const noexcept {
    return entry.key();
  }
};

template <class K> struct FlatSetKeyOf {
  const K &operator()(const K &key) const noexcept { return key; }
};

} // namespace detail

// Sorted contiguous unique-key map for lookup-heavy, mutation-light data:
// immutable registries, compact configuration tables, frozen indexes and
// batch-built routing tables. Lookup is O(log N), ordered iteration is O(N),
// sorted_unique construction/assign is O(N), and positional insert/erase is
// O(N). Use BTreeMap when steady-state insertion/expiry is common and
// FlatHashMap when ordering/range queries are unnecessary.
//
// Successful insertion/erase invalidates iterators and references at or after
// the mutation point; a reallocation invalidates all of them. reserve and
// shrink_to_fit may invalidate all. Lookup and mapped-value assignment do not.
template <class K, class V, class Compare = std::less<>,
          class Alloc = mem::SystemAllocator>
class FlatMap {
  using Entry = FlatMapEntry<K, V>;
  using Core = detail::FlatSortedStorage<Entry, K,
                                         detail::FlatMapKeyOf<K, V>, Compare,
                                         Alloc>;

public:
  using key_type = K;
  using mapped_type = V;
  using value_type = Entry;
  using allocator_type = Alloc;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = Entry &;
  using const_reference = const Entry &;
  using pointer = Entry *;
  using const_pointer = const Entry *;
  using iterator = Entry *;
  using const_iterator = const Entry *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using InsertResult = FlatInsertResult;

  FlatMap() = default;
  explicit FlatMap(Compare compare, Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {}

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires detail::FlatMapRangeValue<std::iter_reference_t<It>>
  FlatMap(It first, Sentinel last) {
    insert(first, last);
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires detail::FlatMapRangeValue<std::iter_reference_t<It>>
  FlatMap(It first, Sentinel last, Compare compare, Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {
    insert(first, last);
  }

  FlatMap(std::initializer_list<std::pair<K, V>> values) {
    insert(values.begin(), values.end());
  }

  FlatMap(std::initializer_list<std::pair<K, V>> values, Compare compare,
          Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {
    insert(values.begin(), values.end());
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires detail::FlatMapRangeValue<std::iter_reference_t<It>>
  FlatMap(sorted_unique_t, It first, Sentinel last,
          Compare compare = Compare{}, Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {
    HYLE_FATAL_ASSERT(try_assign(sorted_unique, first, last));
  }

  FlatMap(sorted_unique_t, std::initializer_list<std::pair<K, V>> values,
          Compare compare = Compare{}, Alloc alloc = Alloc{})
      : FlatMap(sorted_unique, values.begin(), values.end(),
                std::move(compare), std::move(alloc)) {}

  FlatMap(const FlatMap &) = default;
  FlatMap &operator=(const FlatMap &) = default;
  FlatMap(FlatMap &&) noexcept = default;
  FlatMap &operator=(FlatMap &&) noexcept = default;

  size_type size() const noexcept { return core_.size(); }
  size_type capacity() const noexcept { return core_.capacity(); }
  bool empty() const noexcept { return core_.empty(); }
  pointer data() noexcept { return core_.data(); }
  const_pointer data() const noexcept { return core_.data(); }
  const Compare &key_comp() const noexcept { return core_.key_comp(); }
  const Alloc &allocator() const noexcept { return core_.allocator(); }

  iterator begin() noexcept { return core_.iterator_at(0); }
  const_iterator begin() const noexcept { return core_.iterator_at(0); }
  const_iterator cbegin() const noexcept { return core_.iterator_at(0); }
  iterator end() noexcept { return core_.iterator_at(size()); }
  const_iterator end() const noexcept { return core_.iterator_at(size()); }
  const_iterator cend() const noexcept { return core_.iterator_at(size()); }
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

  bool try_reserve(size_type count) noexcept { return core_.try_reserve(count); }
  void reserve(size_type count) { core_.reserve(count); }
  bool try_shrink_to_fit() noexcept { return core_.try_shrink_to_fit(); }
  void shrink_to_fit() { core_.shrink_to_fit(); }
  void clear() noexcept { core_.clear(); }

  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator lower_bound(const Q &key) {
    return core_.iterator_at(core_.lower_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  const_iterator lower_bound(const Q &key) const {
    return core_.iterator_at(core_.lower_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator upper_bound(const Q &key) {
    return core_.iterator_at(core_.upper_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  const_iterator upper_bound(const Q &key) const {
    return core_.iterator_at(core_.upper_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator find(const Q &key) {
    const size_type index = core_.lower_bound_index(key);
    return core_.equivalent_at(index, key) ? core_.iterator_at(index) : end();
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  const_iterator find(const Q &key) const {
    const size_type index = core_.lower_bound_index(key);
    return core_.equivalent_at(index, key) ? core_.iterator_at(index) : end();
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  bool contains(const Q &key) const { return find(key) != end(); }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  size_type count(const Q &key) const { return contains(key) ? 1 : 0; }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  std::pair<iterator, iterator> equal_range(const Q &key) {
    return {lower_bound(key), upper_bound(key)};
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  std::pair<const_iterator, const_iterator>
  equal_range(const Q &key) const {
    return {lower_bound(key), upper_bound(key)};
  }

  InsertResult try_insert(K key, V value) {
    return core_
        .try_insert_unique(
            Entry(std::move(key), std::in_place, std::move(value)))
        .second;
  }

  std::pair<iterator, bool> insert(K key, V value) {
    auto [index, result] = core_.try_insert_unique(
        Entry(std::move(key), std::in_place, std::move(value)));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return {core_.iterator_at(index), result == InsertResult::kInserted};
  }

  template <class... Args>
  std::pair<iterator, bool> try_emplace(K key, Args &&...args) {
    const size_type index = core_.lower_bound_index(key);
    if (core_.equivalent_at(index, key))
      return {core_.iterator_at(index), false};
    Entry candidate(std::move(key), std::in_place,
                    std::forward<Args>(args)...);
    HYLE_FATAL_ASSERT(core_.try_insert_at(index, std::move(candidate)));
    return {core_.iterator_at(index), true};
  }

  InsertResult try_insert_or_assign(K key, V value) {
    const size_type index = core_.lower_bound_index(key);
    if (core_.equivalent_at(index, key)) {
      data()[index].value() = std::move(value);
      return InsertResult::kAssigned;
    }
    Entry candidate(std::move(key), std::in_place, std::move(value));
    return core_.try_insert_at(index, std::move(candidate))
               ? InsertResult::kInserted
               : InsertResult::kOutOfMemory;
  }

  bool insert_or_assign(K key, V value) {
    const InsertResult result =
        try_insert_or_assign(std::move(key), std::move(value));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return result == InsertResult::kInserted;
  }

  V &operator[](K key)
    requires std::default_initializable<V>
  {
    return try_emplace(std::move(key)).first->value();
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires detail::FlatMapRangeValue<std::iter_reference_t<It>>
  void insert(It first, Sentinel last) {
    for (; first != last; ++first) {
      auto &&item = *first;
      insert(detail::flat_range_key(
                 std::forward<decltype(item)>(item)),
             detail::flat_range_value(
                 std::forward<decltype(item)>(item)));
    }
  }
  void insert(std::initializer_list<std::pair<K, V>> values) {
    insert(values.begin(), values.end());
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires detail::FlatMapRangeValue<std::iter_reference_t<It>>
  bool try_assign(sorted_unique_t, It first, Sentinel last) {
    FlatMap replacement(core_.key_comp(), core_.allocator());
    if (!replacement.core_.try_append_sorted(
            std::move(first), std::move(last),
            [](void *storage, auto &&item) noexcept(noexcept(Entry(
                detail::flat_range_key(std::forward<decltype(item)>(item)),
                std::in_place,
                detail::flat_range_value(
                    std::forward<decltype(item)>(item))))) {
              ::new (storage) Entry(
                  detail::flat_range_key(
                      std::forward<decltype(item)>(item)),
                  std::in_place,
                  detail::flat_range_value(
                      std::forward<decltype(item)>(item)));
            }))
      return false;
    swap(replacement);
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires detail::FlatMapRangeValue<std::iter_reference_t<It>>
  void assign(sorted_unique_t tag, It first, Sentinel last) {
    HYLE_FATAL_ASSERT(try_assign(tag, first, last));
  }

  void assign(sorted_unique_t tag,
              std::initializer_list<std::pair<K, V>> values) {
    assign(tag, values.begin(), values.end());
  }

  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  bool erase(const Q &key) {
    const size_type index = core_.lower_bound_index(key);
    if (!core_.equivalent_at(index, key))
      return false;
    core_.erase_index(index);
    return true;
  }
  iterator erase(const_iterator position) noexcept {
    const size_type index = core_.iterator_index(position, false);
    core_.erase_index(index);
    return core_.iterator_at(index);
  }
  iterator erase(const_iterator first, const_iterator last) noexcept {
    const size_type first_index = core_.iterator_index(first, true);
    const size_type last_index = core_.iterator_index(last, true);
    HYLE_FATAL_ASSERT(first_index <= last_index);
    core_.erase_indices(first_index, last_index);
    return core_.iterator_at(first_index);
  }

  void swap(FlatMap &other) noexcept { core_.swap(other.core_); }
  friend void swap(FlatMap &left, FlatMap &right) noexcept { left.swap(right); }

  friend bool operator==(const FlatMap &left, const FlatMap &right)
    requires std::equality_comparable<Entry>
  {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
  }

private:
  Core core_{};
};

// Set counterpart sharing the same storage/search/mutation engine. Iterators
// are const even on a mutable set, preserving the sorted invariant.
template <class K, class Compare = std::less<>,
          class Alloc = mem::SystemAllocator>
class FlatSet {
  using Core = detail::FlatSortedStorage<K, K, detail::FlatSetKeyOf<K>,
                                         Compare, Alloc>;

public:
  using key_type = K;
  using value_type = K;
  using allocator_type = Alloc;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = const K &;
  using const_reference = const K &;
  using pointer = const K *;
  using const_pointer = const K *;
  using iterator = const K *;
  using const_iterator = const K *;
  using reverse_iterator = std::reverse_iterator<const_iterator>;
  using const_reverse_iterator = reverse_iterator;
  using InsertResult = FlatInsertResult;

  FlatSet() = default;
  explicit FlatSet(Compare compare, Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {}

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  FlatSet(It first, Sentinel last) { insert(first, last); }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  FlatSet(It first, Sentinel last, Compare compare, Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {
    insert(first, last);
  }

  FlatSet(std::initializer_list<K> values) { insert(values.begin(), values.end()); }
  FlatSet(std::initializer_list<K> values, Compare compare,
          Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {
    insert(values.begin(), values.end());
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  FlatSet(sorted_unique_t, It first, Sentinel last,
          Compare compare = Compare{}, Alloc alloc = Alloc{})
      : core_(std::move(compare), std::move(alloc)) {
    HYLE_FATAL_ASSERT(try_assign(sorted_unique, first, last));
  }

  FlatSet(sorted_unique_t, std::initializer_list<K> values,
          Compare compare = Compare{}, Alloc alloc = Alloc{})
      : FlatSet(sorted_unique, values.begin(), values.end(),
                std::move(compare), std::move(alloc)) {}

  FlatSet(const FlatSet &) = default;
  FlatSet &operator=(const FlatSet &) = default;
  FlatSet(FlatSet &&) noexcept = default;
  FlatSet &operator=(FlatSet &&) noexcept = default;

  size_type size() const noexcept { return core_.size(); }
  size_type capacity() const noexcept { return core_.capacity(); }
  bool empty() const noexcept { return core_.empty(); }
  const_pointer data() const noexcept { return core_.data(); }
  const Compare &key_comp() const noexcept { return core_.key_comp(); }
  const Alloc &allocator() const noexcept { return core_.allocator(); }
  iterator begin() noexcept { return core_.iterator_at(0); }
  const_iterator begin() const noexcept { return core_.iterator_at(0); }
  const_iterator cbegin() const noexcept { return core_.iterator_at(0); }
  iterator end() noexcept { return core_.iterator_at(size()); }
  const_iterator end() const noexcept { return core_.iterator_at(size()); }
  const_iterator cend() const noexcept { return core_.iterator_at(size()); }
  reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
  reverse_iterator crbegin() const noexcept { return rbegin(); }
  reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
  reverse_iterator crend() const noexcept { return rend(); }

  bool try_reserve(size_type count) noexcept { return core_.try_reserve(count); }
  void reserve(size_type count) { core_.reserve(count); }
  bool try_shrink_to_fit() noexcept { return core_.try_shrink_to_fit(); }
  void shrink_to_fit() { core_.shrink_to_fit(); }
  void clear() noexcept { core_.clear(); }

  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator lower_bound(const Q &key) const {
    return core_.iterator_at(core_.lower_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator upper_bound(const Q &key) const {
    return core_.iterator_at(core_.upper_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator find(const Q &key) const {
    const size_type index = core_.lower_bound_index(key);
    return core_.equivalent_at(index, key) ? core_.iterator_at(index) : end();
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  bool contains(const Q &key) const { return find(key) != end(); }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  size_type count(const Q &key) const { return contains(key) ? 1 : 0; }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  std::pair<iterator, iterator> equal_range(const Q &key) const {
    return {lower_bound(key), upper_bound(key)};
  }

  InsertResult try_insert(K key) {
    return core_.try_insert_unique(std::move(key)).second;
  }
  std::pair<iterator, bool> insert(K key) {
    auto [index, result] = core_.try_insert_unique(std::move(key));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return {core_.iterator_at(index), result == InsertResult::kInserted};
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  void insert(It first, Sentinel last) {
    for (; first != last; ++first)
      insert(*first);
  }
  void insert(std::initializer_list<K> values) {
    insert(values.begin(), values.end());
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  bool try_assign(sorted_unique_t, It first, Sentinel last) {
    FlatSet replacement(core_.key_comp(), core_.allocator());
    if (!replacement.core_.try_append_sorted(
            std::move(first), std::move(last),
            [](void *storage, auto &&key) noexcept(
                std::is_nothrow_constructible_v<K, decltype(key)>) {
              ::new (storage) K(std::forward<decltype(key)>(key));
            }))
      return false;
    swap(replacement);
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  void assign(sorted_unique_t tag, It first, Sentinel last) {
    HYLE_FATAL_ASSERT(try_assign(tag, first, last));
  }
  void assign(sorted_unique_t tag, std::initializer_list<K> values) {
    assign(tag, values.begin(), values.end());
  }

  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  bool erase(const Q &key) {
    const size_type index = core_.lower_bound_index(key);
    if (!core_.equivalent_at(index, key))
      return false;
    core_.erase_index(index);
    return true;
  }
  iterator erase(const_iterator position) noexcept {
    const size_type index = core_.iterator_index(position, false);
    core_.erase_index(index);
    return core_.iterator_at(index);
  }
  iterator erase(const_iterator first, const_iterator last) noexcept {
    const size_type first_index = core_.iterator_index(first, true);
    const size_type last_index = core_.iterator_index(last, true);
    HYLE_FATAL_ASSERT(first_index <= last_index);
    core_.erase_indices(first_index, last_index);
    return core_.iterator_at(first_index);
  }

  void swap(FlatSet &other) noexcept { core_.swap(other.core_); }
  friend void swap(FlatSet &left, FlatSet &right) noexcept { left.swap(right); }

  friend bool operator==(const FlatSet &left, const FlatSet &right)
    requires std::equality_comparable<K>
  {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
  }

private:
  Core core_{};
};

} // namespace hyle::container
