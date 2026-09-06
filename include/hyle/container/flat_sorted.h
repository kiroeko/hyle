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
#include <span>
#include <type_traits>
#include <utility>

#include "hyle/core/compiler.h"
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

enum class FlatInsertResult : std::uint8_t {
  kInserted,
  kAlreadyPresent,
  kAssigned,
  kOutOfMemory,
};

namespace detail {

// A flat map stores keys and mapped values in separate contiguous arrays, as
// C++23 flat_map and Boost.Container do. Iteration therefore needs a lightweight
// pair-of-references proxy rather than pretending that an Entry object exists.
// The standard .first/.second spelling is primary; key()/value() remain useful
// named accessors when generic pair syntax would be less readable.
template <class K, class V, bool Const>
struct FlatMapReference {
  using mapped_reference =
      std::conditional_t<Const, const V &, V &>;

  const K &first;
  mapped_reference second;

  const K &key() const noexcept { return first; }
  mapped_reference value() const noexcept { return second; }

  operator std::pair<K, V>() const
    requires(std::copy_constructible<K> && std::copy_constructible<V>)
  {
    return {first, second};
  }

  template <bool OtherConst>
  bool operator==(const FlatMapReference<K, V, OtherConst> &other) const
    requires(std::equality_comparable<K> && std::equality_comparable<V>)
  {
    return first == other.first && second == other.second;
  }
};

template <class Reference>
struct FlatMapArrow {
  Reference reference;
  const Reference *operator->() const noexcept { return &reference; }
};

template <class K, class V, bool Const>
class FlatMapIterator {
  template <class, class, bool> friend class FlatMapIterator;

  using mapped_pointer = std::conditional_t<Const, const V *, V *>;

public:
  using iterator_concept = std::random_access_iterator_tag;
  using iterator_category = std::random_access_iterator_tag;
  using value_type = std::pair<K, V>;
  using difference_type = std::ptrdiff_t;
  using reference = FlatMapReference<K, V, Const>;
  using pointer = FlatMapArrow<reference>;

  FlatMapIterator() = default;
  FlatMapIterator(const K *key, mapped_pointer value) noexcept
      : key_(key), value_(value) {}

  template <bool OtherConst>
    requires(Const && !OtherConst)
  FlatMapIterator(const FlatMapIterator<K, V, OtherConst> &other) noexcept
      : key_(other.key_), value_(other.value_) {}

  reference operator*() const noexcept { return {*key_, *value_}; }
  pointer operator->() const noexcept { return pointer{**this}; }
  reference operator[](difference_type offset) const noexcept {
    return {key_[offset], value_[offset]};
  }

  FlatMapIterator &operator++() noexcept {
    ++key_;
    ++value_;
    return *this;
  }
  FlatMapIterator operator++(int) noexcept {
    FlatMapIterator copy = *this;
    ++*this;
    return copy;
  }
  FlatMapIterator &operator--() noexcept {
    --key_;
    --value_;
    return *this;
  }
  FlatMapIterator operator--(int) noexcept {
    FlatMapIterator copy = *this;
    --*this;
    return copy;
  }
  FlatMapIterator &operator+=(difference_type offset) noexcept {
    if (offset != 0) {
      key_ += offset;
      value_ += offset;
    }
    return *this;
  }
  FlatMapIterator &operator-=(difference_type offset) noexcept {
    return *this += -offset;
  }

  friend FlatMapIterator operator+(FlatMapIterator iterator,
                                   difference_type offset) noexcept {
    iterator += offset;
    return iterator;
  }
  friend FlatMapIterator operator+(difference_type offset,
                                   FlatMapIterator iterator) noexcept {
    iterator += offset;
    return iterator;
  }
  friend FlatMapIterator operator-(FlatMapIterator iterator,
                                   difference_type offset) noexcept {
    iterator -= offset;
    return iterator;
  }

  template <bool OtherConst>
  difference_type
  operator-(const FlatMapIterator<K, V, OtherConst> &other) const noexcept {
    return key_ == other.key_ ? 0 : key_ - other.key_;
  }
  template <bool OtherConst>
  bool operator==(
      const FlatMapIterator<K, V, OtherConst> &other) const noexcept {
    return key_ == other.key_;
  }
  template <bool OtherConst>
  auto operator<=>(
      const FlatMapIterator<K, V, OtherConst> &other) const noexcept {
    return key_ == other.key_ ? std::strong_ordering::equal
                              : key_ <=> other.key_;
  }

private:
  const K *key_ = nullptr;
  mapped_pointer value_ = nullptr;
};

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

template <class K, class Compare, class Alloc>
class FlatSetStorage {
  // Reuse the established non-trivial sequence engine without embedding an
  // arbitrary K in every empty set. Flat ordered containers target compact,
  // read-mostly indexes across the full supported size range.
  using Sequence = SmallVector<K, 0, Alloc>;

  static_assert(std::is_nothrow_move_constructible_v<K> &&
                    std::is_nothrow_destructible_v<K>,
                "flat ordered keys require nothrow move/destruction");
  static_assert(std::is_nothrow_move_constructible_v<Compare> &&
                    std::is_nothrow_move_assignable_v<Compare> &&
                    std::is_nothrow_swappable_v<Compare>,
                "flat ordered comparator movement/swap must be noexcept");

public:
  using size_type = std::size_t;

  FlatSetStorage() = default;
  explicit FlatSetStorage(Compare compare, Alloc alloc = Alloc{})
      : entries_(std::move(alloc)), compare_(std::move(compare)) {}

  FlatSetStorage(const FlatSetStorage &) = default;
  FlatSetStorage &operator=(const FlatSetStorage &) = default;
  FlatSetStorage(FlatSetStorage &&) noexcept = default;
  FlatSetStorage &operator=(FlatSetStorage &&) noexcept = default;

  size_type size() const noexcept { return entries_.size(); }
  size_type capacity() const noexcept { return entries_.capacity(); }
  bool empty() const noexcept { return entries_.empty(); }
  K *data() noexcept { return entries_.data(); }
  const K *data() const noexcept { return entries_.data(); }
  K *iterator_at(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    return index == 0 ? data() : data() + index;
  }
  const K *iterator_at(size_type index) const noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    return index == 0 ? data() : data() + index;
  }
  size_type iterator_index(const K *position,
                           bool allow_end) const noexcept {
    if (data() == nullptr) {
      HYLE_FATAL_ASSERT(position == nullptr && allow_end);
      return 0;
    }
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
  }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  size_type upper_bound_index(const Q &key) const {
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
  }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  bool equivalent_at(size_type index, const Q &key) const {
    return index != size() && !compare_(key, entries_[index]);
  }

  std::pair<size_type, FlatInsertResult> try_insert_unique(K candidate) {
    const size_type index = lower_bound_index(candidate);
    if (equivalent_at(index, candidate))
      return {index, FlatInsertResult::kAlreadyPresent};
    if (!try_insert_at(index, std::move(candidate)))
      return {size(), FlatInsertResult::kOutOfMemory};
    return {index, FlatInsertResult::kInserted};
  }

  bool try_insert_at(size_type index, K candidate) {
    HYLE_FATAL_ASSERT(index <= size());
    if (size() == std::numeric_limits<size_type>::max() ||
        !SmallVectorAccess::try_reserve_growth(entries_, size() + 1))
      return false;
    auto position = index == 0 ? entries_.begin() : entries_.begin() + index;
    entries_.emplace(position, std::move(candidate));
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  bool try_append_sorted(It first, Sentinel last) {
    // This storage is a fresh replacement. SmallVector may therefore reserve
    // once and directly construct the projected entries without alias checks
    // or one capacity branch per element.
    return SmallVectorAccess::try_append_construct_unaliased(
        entries_, std::move(first), std::move(last),
        [](void *storage, auto &&key) noexcept(
            std::is_nothrow_constructible_v<K, decltype(key)>) {
          ::new (storage) K(std::forward<decltype(key)>(key));
        });
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

  void swap(FlatSetStorage &other) noexcept {
    using std::swap;
    swap(compare_, other.compare_);
    entries_.swap(other.entries_);
  }

private:
  Sequence entries_{};
  HYLE_NO_UNIQUE_ADDRESS Compare compare_{};
};

template <class K, class V, class Compare, class Alloc>
class FlatMapStorage {
  using Keys = SmallVector<K, 0, Alloc>;
  using Values = SmallVector<V, 0, Alloc>;

  static_assert(std::is_nothrow_move_constructible_v<K> &&
                    std::is_nothrow_destructible_v<K>,
                "flat ordered keys require nothrow move/destruction");
  static_assert(std::is_nothrow_move_constructible_v<V> &&
                    std::is_nothrow_destructible_v<V>,
                "flat ordered mapped values require nothrow move/destruction");
  static_assert(std::copy_constructible<Alloc>,
                "flat map needs one allocator copy for each SoA array");
  static_assert(std::is_nothrow_move_constructible_v<Compare> &&
                    std::is_nothrow_move_assignable_v<Compare> &&
                    std::is_nothrow_swappable_v<Compare>,
                "flat ordered comparator movement/swap must be noexcept");

public:
  using size_type = std::size_t;

  FlatMapStorage() = default;
  explicit FlatMapStorage(Compare compare, Alloc alloc = Alloc{})
      : keys_(alloc), values_(std::move(alloc)),
        compare_(std::move(compare)) {}

  FlatMapStorage(const FlatMapStorage &) = default;
  FlatMapStorage &operator=(const FlatMapStorage &other) {
    if (this == &other)
      return *this;
    FlatMapStorage replacement(other);
    swap(replacement);
    return *this;
  }
  FlatMapStorage(FlatMapStorage &&) noexcept = default;
  FlatMapStorage &operator=(FlatMapStorage &&) noexcept = default;

  size_type size() const noexcept {
    HYLE_FATAL_ASSERT(keys_.size() == values_.size());
    return keys_.size();
  }
  size_type capacity() const noexcept {
    return std::min(keys_.capacity(), values_.capacity());
  }
  bool empty() const noexcept { return size() == 0; }
  K *keys_data() noexcept { return keys_.data(); }
  const K *keys_data() const noexcept { return keys_.data(); }
  V *values_data() noexcept { return values_.data(); }
  const V *values_data() const noexcept { return values_.data(); }
  V &mapped_at(size_type index) noexcept { return values_[index]; }
  const V &mapped_at(size_type index) const noexcept { return values_[index]; }
  const Compare &key_comp() const noexcept { return compare_; }
  const Alloc &allocator() const noexcept { return keys_.allocator(); }

  bool try_reserve(size_type count) noexcept {
    if (count > std::numeric_limits<size_type>::max() / sizeof(K) ||
        count > std::numeric_limits<size_type>::max() / sizeof(V))
      return false;
    // One successful reserve followed by one failure merely leaves spare
    // capacity in one array; the paired logical size and contents stay intact.
    return keys_.try_reserve(count) && values_.try_reserve(count);
  }
  void reserve(size_type count) { HYLE_FATAL_ASSERT(try_reserve(count)); }
  bool try_shrink_to_fit() noexcept {
    const bool keys_ok = keys_.try_shrink_to_fit();
    const bool values_ok = values_.try_shrink_to_fit();
    return keys_ok && values_ok;
  }
  void shrink_to_fit() { HYLE_FATAL_ASSERT(try_shrink_to_fit()); }
  void clear() noexcept {
    keys_.clear();
    values_.clear();
  }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  size_type lower_bound_index(const Q &key) const {
    size_type first = 0;
    size_type count = keys_.size();
    while (count != 0) {
      const size_type step = count / 2;
      const size_type middle = first + step;
      if (compare_(keys_[middle], key)) {
        first = middle + 1;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  size_type upper_bound_index(const Q &key) const {
    size_type first = 0;
    size_type count = keys_.size();
    while (count != 0) {
      const size_type step = count / 2;
      const size_type middle = first + step;
      if (!compare_(key, keys_[middle])) {
        first = middle + 1;
        count -= step + 1;
      } else {
        count = step;
      }
    }
    return first;
  }

  template <class Q>
    requires FlatComparable<Compare, K, Q>
  bool equivalent_at(size_type index, const Q &key) const {
    return index != size() && !compare_(key, keys_[index]);
  }

  bool try_insert_at(size_type index, K key, V value) {
    HYLE_FATAL_ASSERT(index <= size());
    if (size() == std::numeric_limits<size_type>::max() ||
        !SmallVectorAccess::try_reserve_growth(keys_, size() + 1) ||
        !SmallVectorAccess::try_reserve_growth(values_, size() + 1))
      return false;
    auto *key_position = index == 0 ? keys_.begin() : keys_.begin() + index;
    auto *value_position =
        index == 0 ? values_.begin() : values_.begin() + index;
    keys_.emplace(key_position, std::move(key));
    values_.emplace(value_position, std::move(value));
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
  bool try_append_sorted(It first, Sentinel last) {
    if constexpr (std::forward_iterator<It>) {
      // Replaying a forward range lets each contiguous array use the established
      // one-reserve/uninitialized bulk seam. A move iterator moves the key on
      // pass one and the mapped value on pass two; the pair remains dereferenceable.
      const It values_first = first;
      if (!SmallVectorAccess::try_append_construct_unaliased(
              keys_, first, last,
              [](void *storage, auto &&item) noexcept(noexcept(K(
                  flat_range_key(std::forward<decltype(item)>(item))))) {
                ::new (storage) K(flat_range_key(
                    std::forward<decltype(item)>(item)));
              }))
        return false;
      if (!SmallVectorAccess::try_append_construct_unaliased(
              values_, values_first, last,
              [](void *storage, auto &&item) noexcept(noexcept(V(
                  flat_range_value(std::forward<decltype(item)>(item))))) {
                ::new (storage) V(flat_range_value(
                    std::forward<decltype(item)>(item)));
              }))
        return false;
      return true;
    } else {
      for (; first != last; ++first) {
        auto &&item = *first;
        K key(flat_range_key(std::forward<decltype(item)>(item)));
        V value(flat_range_value(std::forward<decltype(item)>(item)));
        if (!try_insert_at(size(), std::move(key), std::move(value)))
          return false;
      }
      return true;
    }
  }

  void erase_index(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index < size());
    keys_.erase(index == 0 ? keys_.begin() : keys_.begin() + index);
    values_.erase(index == 0 ? values_.begin() : values_.begin() + index);
  }
  void erase_indices(size_type first, size_type last) noexcept {
    HYLE_FATAL_ASSERT(first <= last && last <= size());
    auto *keys_first = first == 0 ? keys_.begin() : keys_.begin() + first;
    auto *keys_last = last == 0 ? keys_.begin() : keys_.begin() + last;
    auto *values_first =
        first == 0 ? values_.begin() : values_.begin() + first;
    auto *values_last =
        last == 0 ? values_.begin() : values_.begin() + last;
    keys_.erase(keys_first, keys_last);
    values_.erase(values_first, values_last);
  }

  void swap(FlatMapStorage &other) noexcept {
    using std::swap;
    swap(compare_, other.compare_);
    keys_.swap(other.keys_);
    values_.swap(other.values_);
  }

  friend bool operator==(const FlatMapStorage &left,
                         const FlatMapStorage &right)
    requires(std::equality_comparable<K> && std::equality_comparable<V>)
  {
    return left.keys_ == right.keys_ && left.values_ == right.values_;
  }

private:
  Keys keys_{};
  Values values_{};
  HYLE_NO_UNIQUE_ADDRESS Compare compare_{};
};

} // namespace detail

// Sorted unique-key map for lookup-heavy, mutation-light data: immutable
// registries, compact configuration tables, frozen indexes and batch-built
// routing tables. Keys and mapped values occupy separate contiguous arrays;
// lookup touches only keys, while keys()/values() expose explicit bulk views.
// The familiar random-access iterator returns a pair-of-references proxy with
// const .first and mutable .second. Lookup is O(log N), ordered iteration and
// sorted_unique construction/assign are O(N), and positional insert/erase is
// O(N). Use BTreeMap when steady-state insertion/expiry is common and
// FlatHashMap when ordering/range queries are unnecessary.
//
// Successful insertion/erase invalidates iterators and references at or after
// the mutation point; a reallocation invalidates all of them. reserve and
// shrink_to_fit may invalidate all. Lookup and mapped-value assignment do not.
template <class K, class V, class Compare = std::less<>,
          class Alloc = mem::SystemAllocator>
class FlatMap {
  using Core = detail::FlatMapStorage<K, V, Compare, Alloc>;

public:
  using key_type = K;
  using mapped_type = V;
  using value_type = std::pair<K, V>;
  using allocator_type = Alloc;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = detail::FlatMapReference<K, V, false>;
  using const_reference = detail::FlatMapReference<K, V, true>;
  using iterator = detail::FlatMapIterator<K, V, false>;
  using const_iterator = detail::FlatMapIterator<K, V, true>;
  using pointer = typename iterator::pointer;
  using const_pointer = typename const_iterator::pointer;
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
  std::span<const K> keys() const noexcept {
    return {core_.keys_data(), size()};
  }
  std::span<V> values() noexcept { return {core_.values_data(), size()}; }
  std::span<const V> values() const noexcept {
    return {core_.values_data(), size()};
  }
  const Compare &key_comp() const noexcept { return core_.key_comp(); }
  const Alloc &allocator() const noexcept { return core_.allocator(); }

  iterator begin() noexcept { return iterator_at(0); }
  const_iterator begin() const noexcept { return iterator_at(0); }
  const_iterator cbegin() const noexcept { return iterator_at(0); }
  iterator end() noexcept { return iterator_at(size()); }
  const_iterator end() const noexcept { return iterator_at(size()); }
  const_iterator cend() const noexcept { return iterator_at(size()); }
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
    return iterator_at(core_.lower_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  const_iterator lower_bound(const Q &key) const {
    return iterator_at(core_.lower_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator upper_bound(const Q &key) {
    return iterator_at(core_.upper_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  const_iterator upper_bound(const Q &key) const {
    return iterator_at(core_.upper_bound_index(key));
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  iterator find(const Q &key) {
    const size_type index = core_.lower_bound_index(key);
    return core_.equivalent_at(index, key) ? iterator_at(index) : end();
  }
  template <class Q>
    requires detail::FlatComparable<Compare, K, Q>
  const_iterator find(const Q &key) const {
    const size_type index = core_.lower_bound_index(key);
    return core_.equivalent_at(index, key) ? iterator_at(index) : end();
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
    const size_type index = core_.lower_bound_index(key);
    if (core_.equivalent_at(index, key))
      return InsertResult::kAlreadyPresent;
    return core_.try_insert_at(index, std::move(key), std::move(value))
               ? InsertResult::kInserted
               : InsertResult::kOutOfMemory;
  }

  InsertResult try_insert(value_type value) {
    return try_insert(std::move(value.first), std::move(value.second));
  }

  std::pair<iterator, bool> insert(K key, V value) {
    const size_type index = core_.lower_bound_index(key);
    if (core_.equivalent_at(index, key))
      return {iterator_at(index), false};
    HYLE_FATAL_ASSERT(
        core_.try_insert_at(index, std::move(key), std::move(value)));
    return {iterator_at(index), true};
  }

  std::pair<iterator, bool> insert(value_type value) {
    return insert(std::move(value.first), std::move(value.second));
  }

  template <class... Args>
  std::pair<iterator, InsertResult> try_emplace(K key, Args &&...args) {
    const size_type index = core_.lower_bound_index(key);
    if (core_.equivalent_at(index, key))
      return {iterator_at(index), InsertResult::kAlreadyPresent};
    V candidate(std::forward<Args>(args)...);
    if (!core_.try_insert_at(index, std::move(key), std::move(candidate)))
      return {end(), InsertResult::kOutOfMemory};
    return {iterator_at(index), InsertResult::kInserted};
  }

  template <class... Args>
  std::pair<iterator, bool> emplace(K key, Args &&...args) {
    auto [position, result] =
        try_emplace(std::move(key), std::forward<Args>(args)...);
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return {position, result == InsertResult::kInserted};
  }

  InsertResult try_insert_or_assign(K key, V value) {
    const size_type index = core_.lower_bound_index(key);
    if (core_.equivalent_at(index, key)) {
      core_.mapped_at(index) = std::move(value);
      return InsertResult::kAssigned;
    }
    return core_.try_insert_at(index, std::move(key), std::move(value))
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
    return emplace(std::move(key)).first->second;
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
    if (!replacement.core_.try_append_sorted(std::move(first),
                                              std::move(last)))
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
    const difference_type distance = position - cbegin();
    HYLE_FATAL_ASSERT(distance >= 0 &&
                      static_cast<size_type>(distance) < size());
    const size_type index = static_cast<size_type>(distance);
    core_.erase_index(index);
    return iterator_at(index);
  }
  iterator erase(const_iterator first, const_iterator last) noexcept {
    const difference_type first_distance = first - cbegin();
    const difference_type last_distance = last - cbegin();
    HYLE_FATAL_ASSERT(first_distance >= 0 &&
                      first_distance <= last_distance &&
                      static_cast<size_type>(last_distance) <= size());
    const size_type first_index = static_cast<size_type>(first_distance);
    const size_type last_index = static_cast<size_type>(last_distance);
    core_.erase_indices(first_index, last_index);
    return iterator_at(first_index);
  }

  void swap(FlatMap &other) noexcept { core_.swap(other.core_); }
  friend void swap(FlatMap &left, FlatMap &right) noexcept { left.swap(right); }

  friend bool operator==(const FlatMap &left, const FlatMap &right)
    requires(std::equality_comparable<K> && std::equality_comparable<V>)
  {
    return left.core_ == right.core_;
  }

private:
  iterator iterator_at(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    auto *keys = core_.keys_data();
    auto *values = core_.values_data();
    return iterator(index == 0 ? keys : keys + index,
                    index == 0 ? values : values + index);
  }
  const_iterator iterator_at(size_type index) const noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    const auto *keys = core_.keys_data();
    const auto *values = core_.values_data();
    return const_iterator(index == 0 ? keys : keys + index,
                          index == 0 ? values : values + index);
  }

  Core core_{};
};

// Set counterpart. Iterators are const even on a mutable set, preserving the
// sorted invariant; unlike FlatMap, the single key array is genuinely contiguous.
template <class K, class Compare = std::less<>,
          class Alloc = mem::SystemAllocator>
class FlatSet {
  using Core = detail::FlatSetStorage<K, Compare, Alloc>;

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
    if (!replacement.core_.try_append_sorted(std::move(first),
                                              std::move(last)))
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
           (left.empty() ||
            std::equal(left.begin(), left.end(), right.begin()));
  }

private:
  Core core_{};
};

} // namespace hyle::container
