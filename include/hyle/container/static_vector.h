#pragma once

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/macros.h"

namespace hyle::container {

namespace detail {

template <std::size_t Capacity>
using static_vector_size_t = std::conditional_t<
    (Capacity <= std::numeric_limits<std::uint8_t>::max()), std::uint8_t,
    std::conditional_t<
        (Capacity <= std::numeric_limits<std::uint16_t>::max()), std::uint16_t,
        std::conditional_t<(Capacity <=
                            std::numeric_limits<std::uint32_t>::max()),
                           std::uint32_t, std::size_t>>>;

// The inactive union retains a real T[Capacity] array boundary without
// constructing any T. StaticVector starts/ends individual element lifetimes
// explicitly. data() can therefore form the array member address while empty;
// it must not launder a raw-byte pointer when no live T exists yet.
template <class T, std::size_t Capacity> union StaticVectorElements {
  constexpr StaticVectorElements() noexcept {}
  ~StaticVectorElements() noexcept {}

  T values[Capacity];
};

template <class T, std::size_t Capacity, bool = (Capacity == 0)>
struct StaticVectorStorage {
  StaticVectorElements<T, Capacity> elements;
  static_vector_size_t<Capacity> size = 0;
};

// A zero-capacity vector needs neither dummy T-aligned bytes nor size state.
// no_unique_address lets the complete container remain a one-byte empty type.
template <class T, std::size_t Capacity>
struct StaticVectorStorage<T, Capacity, true> {};

} // namespace detail

// A fixed-capacity owning sequence. All element storage is embedded in the
// object: StaticVector never allocates, and data() never changes for the
// lifetime of the vector. Use SmallVector when spilling past N is allowed.
//
// The ordinary std-shaped operations fail fast on capacity overflow. Their
// try_* counterparts return false before changing the vector. Construction of
// T may still throw; nothrow move construction/destruction makes relocation
// during positional mutation recoverable without move assignment.
template <class T, std::size_t N> class StaticVector {
  static_assert(std::is_nothrow_destructible_v<T>,
                "StaticVector requires nothrow destruction");
  static_assert(N <= std::numeric_limits<std::size_t>::max() / sizeof(T),
                "StaticVector capacity overflows object storage");

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  StaticVector() noexcept = default;

  explicit StaticVector(size_type count)
    requires std::is_default_constructible_v<T>
  {
    construct_with_cleanup([&] { resize(count); });
  }

  StaticVector(size_type count, const T &value)
    requires std::is_copy_constructible_v<T>
  {
    construct_with_cleanup([&] { resize(count, value); });
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires std::constructible_from<T, std::iter_reference_t<It>>
  StaticVector(It first, Sentinel last) {
    construct_with_cleanup([&] { append_range(first, last); });
  }

  StaticVector(std::initializer_list<T> init)
    requires std::is_copy_constructible_v<T>
  {
    construct_with_cleanup([&] { append_range(init.begin(), init.end()); });
  }

  StaticVector(const StaticVector &other)
    requires std::is_copy_constructible_v<T>
  {
    construct_with_cleanup([&] { append_range(other.begin(), other.end()); });
  }

  StaticVector(StaticVector &&other) noexcept(
      std::is_nothrow_move_constructible_v<T>)
    requires std::is_move_constructible_v<T>
  {
    if constexpr (std::is_nothrow_move_constructible_v<T>) {
      move_from(other);
    } else {
      construct_with_cleanup([&] { move_from(other); });
    }
  }

  StaticVector &operator=(const StaticVector &other)
    requires std::is_copy_constructible_v<T>
  {
    if (this == &other)
      return *this;
    StaticVector replacement(other);
    replace_with(replacement);
    return *this;
  }

  StaticVector &operator=(StaticVector &&other) noexcept(
      std::is_nothrow_move_constructible_v<T>)
    requires std::is_move_constructible_v<T>
  {
    if (this == &other)
      return *this;
    clear();
    move_from(other);
    return *this;
  }

  StaticVector &operator=(std::initializer_list<T> init)
    requires std::is_copy_constructible_v<T>
  {
    assign(init);
    return *this;
  }

  ~StaticVector() { clear(); }

  [[nodiscard]] size_type size() const noexcept {
    if constexpr (N == 0) {
      return 0;
    } else {
      const size_type value = static_cast<size_type>(storage_.size);
      // Every mutator funnels through set_size(). Preserve that invariant in
      // the optimizer model too; otherwise a byte-sized counter for N=4 can
      // be speculated as 5..255 and GCC invents an out-of-bounds destructor.
      HYLE_ASSUME(value <= N);
      return value;
    }
  }
  [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }
  [[nodiscard]] static constexpr size_type max_size() noexcept { return N; }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] bool full() const noexcept { return size() == N; }

  pointer data() noexcept {
    if constexpr (N == 0) {
      return nullptr;
    } else {
      return storage_.elements.values;
    }
  }
  const_pointer data() const noexcept {
    if constexpr (N == 0) {
      return nullptr;
    } else {
      return storage_.elements.values;
    }
  }

  reference operator[](size_type index) noexcept { return data()[index]; }
  const_reference operator[](size_type index) const noexcept {
    return data()[index];
  }
  reference front() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return *data();
  }
  const_reference front() const noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return *data();
  }
  reference back() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return data()[size() - 1];
  }
  const_reference back() const noexcept {
    HYLE_FATAL_ASSERT(!empty());
    return data()[size() - 1];
  }

  iterator begin() noexcept { return data(); }
  const_iterator begin() const noexcept { return data(); }
  const_iterator cbegin() const noexcept { return data(); }
  iterator end() noexcept {
    if constexpr (N == 0) {
      return nullptr;
    } else {
      return data() + size();
    }
  }
  const_iterator end() const noexcept {
    if constexpr (N == 0) {
      return nullptr;
    } else {
      return data() + size();
    }
  }
  const_iterator cend() const noexcept { return end(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

#if defined(_MSC_VER)
#pragma warning(push)
// A deliberately always-throwing test/fault-injection constructor can make
// MSVC report the continuation of this generic function as C4702. Other T
// reach it normally, so suppress the template-instantiation false positive.
#pragma warning(disable : 4702)
#endif
  template <class... Args> bool try_emplace_back(Args &&...args) {
    if (full())
      return false;
    ::new (static_cast<void *>(data() + size())) T(std::forward<Args>(args)...);
    increment_size();
    return true;
  }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

  template <class... Args> reference emplace_back(Args &&...args) {
    HYLE_FATAL_ASSERT(try_emplace_back(std::forward<Args>(args)...));
    return back();
  }

  bool try_push_back(const T &value)
    requires std::is_copy_constructible_v<T>
  {
    return try_emplace_back(value);
  }
  bool try_push_back(T &&value) { return try_emplace_back(std::move(value)); }
  void push_back(const T &value)
    requires std::is_copy_constructible_v<T>
  {
    HYLE_FATAL_ASSERT(try_push_back(value));
  }
  void push_back(T &&value) {
    HYLE_FATAL_ASSERT(try_push_back(std::move(value)));
  }

  template <class... Args>
    requires std::is_nothrow_move_constructible_v<T>
  bool try_emplace(const_iterator position, Args &&...args) {
    const size_type index = position_index(position, true);
    if (full())
      return false;
    // Build first so aliased arguments survive the suffix relocation and a
    // throwing user constructor leaves the vector unchanged.
    T saved(std::forward<Args>(args)...);
    open_gap(index, 1);
    ::new (static_cast<void *>(data() + index)) T(std::move(saved));
    increment_size();
    return true;
  }

  template <class... Args>
    requires std::is_nothrow_move_constructible_v<T>
  iterator emplace(const_iterator position, Args &&...args) {
    const size_type index = position_index(position, true);
    HYLE_FATAL_ASSERT(
        try_emplace(pointer_at(index), std::forward<Args>(args)...));
    return pointer_at(index);
  }

  bool try_insert(const_iterator position, const T &value)
    requires std::is_copy_constructible_v<T> &&
             std::is_nothrow_move_constructible_v<T>
  {
    return try_emplace(position, value);
  }
  bool try_insert(const_iterator position, T &&value)
    requires std::is_nothrow_move_constructible_v<T>
  {
    return try_emplace(position, std::move(value));
  }
  iterator insert(const_iterator position, const T &value)
    requires std::is_copy_constructible_v<T> &&
             std::is_nothrow_move_constructible_v<T>
  {
    return emplace(position, value);
  }
  iterator insert(const_iterator position, T &&value)
    requires std::is_nothrow_move_constructible_v<T>
  {
    return emplace(position, std::move(value));
  }

  bool try_insert(const_iterator position, size_type count, const T &value)
    requires std::is_copy_constructible_v<T> &&
             std::is_nothrow_move_constructible_v<T>
  {
    const size_type index = position_index(position, true);
    if (count > N - size())
      return false;
    StaticVector inserted(count, value);
    insert_prepared(index, inserted);
    return true;
  }

  iterator insert(const_iterator position, size_type count, const T &value)
    requires std::is_copy_constructible_v<T> &&
             std::is_nothrow_move_constructible_v<T>
  {
    const size_type index = position_index(position, true);
    HYLE_FATAL_ASSERT(try_insert(pointer_at(index), count, value));
    return pointer_at(index);
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires std::constructible_from<T, std::iter_reference_t<It>> &&
             std::is_nothrow_move_constructible_v<T>
  bool try_insert(const_iterator position, It first, Sentinel last) {
    const size_type index = position_index(position, true);
    StaticVector inserted;
    if (!inserted.try_append_range(first, last))
      return false;
    if (inserted.size() > N - size())
      return false;
    insert_prepared(index, inserted);
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires std::constructible_from<T, std::iter_reference_t<It>> &&
             std::is_nothrow_move_constructible_v<T>
  iterator insert(const_iterator position, It first, Sentinel last) {
    const size_type index = position_index(position, true);
    HYLE_FATAL_ASSERT(
        try_insert(pointer_at(index), std::move(first), std::move(last)));
    return pointer_at(index);
  }

  bool try_insert(const_iterator position, std::initializer_list<T> init)
    requires std::is_copy_constructible_v<T> &&
             std::is_nothrow_move_constructible_v<T>
  {
    return try_insert(position, init.begin(), init.end());
  }
  iterator insert(const_iterator position, std::initializer_list<T> init)
    requires std::is_copy_constructible_v<T> &&
             std::is_nothrow_move_constructible_v<T>
  {
    return insert(position, init.begin(), init.end());
  }

  bool try_assign(size_type count, const T &value)
    requires std::is_copy_constructible_v<T>
  {
    if (count > N)
      return false;
    StaticVector replacement(count, value);
    replace_with(replacement);
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires std::constructible_from<T, std::iter_reference_t<It>> &&
             (std::is_move_constructible_v<T> ||
              std::is_copy_constructible_v<T>)
  bool try_assign(It first, Sentinel last) {
    StaticVector replacement;
    if (!replacement.try_append_range(first, last))
      return false;
    replace_with(replacement);
    return true;
  }

  bool try_assign(std::initializer_list<T> init)
    requires std::is_copy_constructible_v<T>
  {
    return try_assign(init.begin(), init.end());
  }

  void assign(size_type count, const T &value)
    requires std::is_copy_constructible_v<T>
  {
    HYLE_FATAL_ASSERT(try_assign(count, value));
  }
  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires std::constructible_from<T, std::iter_reference_t<It>> &&
             (std::is_move_constructible_v<T> ||
              std::is_copy_constructible_v<T>)
  void assign(It first, Sentinel last) {
    HYLE_FATAL_ASSERT(try_assign(first, last));
  }
  void assign(std::initializer_list<T> init)
    requires std::is_copy_constructible_v<T>
  {
    HYLE_FATAL_ASSERT(try_assign(init));
  }

  bool try_resize(size_type count)
    requires std::is_default_constructible_v<T>
  {
    if (count > N)
      return false;
    if (count < size()) {
      destroy_range(data() + count, size() - count);
      set_size(count);
      return true;
    }
    const size_type original_size = size();
    try {
      while (size() < count) {
        ::new (static_cast<void *>(data() + size())) T();
        increment_size();
      }
    } catch (...) {
      destroy_range(data() + original_size, size() - original_size);
      set_size(original_size);
      throw;
    }
    return true;
  }

  bool try_resize(size_type count, const T &value)
    requires std::is_copy_constructible_v<T>
  {
    if (count > N)
      return false;
    if (count <= size())
      return resize_down(count);
    const size_type original_size = size();
    try {
      while (size() < count) {
        ::new (static_cast<void *>(data() + size())) T(value);
        increment_size();
      }
    } catch (...) {
      destroy_range(data() + original_size, size() - original_size);
      set_size(original_size);
      throw;
    }
    return true;
  }

  void resize(size_type count)
    requires std::is_default_constructible_v<T>
  {
    HYLE_FATAL_ASSERT(try_resize(count));
  }
  void resize(size_type count, const T &value)
    requires std::is_copy_constructible_v<T>
  {
    HYLE_FATAL_ASSERT(try_resize(count, value));
  }

  void pop_back() noexcept {
    HYLE_FATAL_ASSERT(!empty());
    const size_type index = size() - 1;
    data()[index].~T();
    set_size(index);
  }

  void clear() noexcept {
    destroy_range(data(), size());
    set_size(0);
  }

  iterator erase(const_iterator position) noexcept
    requires std::is_nothrow_move_constructible_v<T>
  {
    const size_type index = position_index(position, false);
    return erase(pointer_at(index), pointer_at(index + 1));
  }

  iterator erase(const_iterator first, const_iterator last) noexcept
    requires std::is_nothrow_move_constructible_v<T>
  {
    const size_type first_index = position_index(first, true);
    const size_type last_index = position_index(last, true);
    HYLE_FATAL_ASSERT(first_index <= last_index);
    const size_type count = last_index - first_index;
    if (count == 0)
      return pointer_at(first_index);

    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memmove(data() + first_index, data() + last_index,
                   (size() - last_index) * sizeof(T));
    } else {
      destroy_range(data() + first_index, count);
      for (size_type i = first_index; i + count < size(); ++i) {
        ::new (static_cast<void *>(data() + i)) T(std::move(data()[i + count]));
        data()[i + count].~T();
      }
    }
    set_size(size() - count);
    return pointer_at(first_index);
  }

  void
  swap(StaticVector &other) noexcept(std::is_nothrow_move_constructible_v<T>)
    requires std::is_move_constructible_v<T>
  {
    if (this == &other)
      return;
    StaticVector temporary(std::move(*this));
    move_from(other);
    other.move_from(temporary);
  }

  friend void
  swap(StaticVector &left,
       StaticVector &right) noexcept(std::is_nothrow_move_constructible_v<T>)
    requires std::is_move_constructible_v<T>
  {
    left.swap(right);
  }

  friend bool operator==(const StaticVector &left, const StaticVector &right)
    requires std::equality_comparable<T>
  {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
  }

  friend auto operator<=>(const StaticVector &left, const StaticVector &right)
    requires std::three_way_comparable<T>
  {
    return std::lexicographical_compare_three_way(left.begin(), left.end(),
                                                  right.begin(), right.end());
  }

private:
  using stored_size_type = detail::static_vector_size_t<N>;

  void increment_size() noexcept { set_size(size() + 1); }
  void set_size(size_type value) noexcept {
    HYLE_FATAL_ASSERT(value <= N);
    if constexpr (N != 0)
      storage_.size = static_cast<stored_size_type>(value);
  }

  pointer pointer_at(size_type index) noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    if constexpr (N == 0) {
      return nullptr;
    } else {
      return data() + index;
    }
  }
  const_pointer pointer_at(size_type index) const noexcept {
    HYLE_FATAL_ASSERT(index <= size());
    if constexpr (N == 0) {
      return nullptr;
    } else {
      return data() + index;
    }
  }

  size_type position_index(const_iterator position,
                           bool allow_end) const noexcept {
    if constexpr (N == 0) {
      HYLE_FATAL_ASSERT(position == nullptr && allow_end);
      return 0;
    } else {
      HYLE_FATAL_ASSERT(position >= cbegin() &&
                        (allow_end ? position <= cend() : position < cend()));
      return static_cast<size_type>(position - cbegin());
    }
  }

  static void destroy_range(pointer first, size_type count) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (size_type i = count; i != 0; --i)
        first[i - 1].~T();
    }
  }

  bool resize_down(size_type count) noexcept {
    if (count < size()) {
      destroy_range(data() + count, size() - count);
      set_size(count);
    }
    return true;
  }

  template <class Fn> void construct_with_cleanup(Fn &&fn) {
    try {
      std::forward<Fn>(fn)();
    } catch (...) {
      clear();
      throw;
    }
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires std::constructible_from<T, std::iter_reference_t<It>>
  bool try_append_range(It first, Sentinel last) {
    for (; first != last; ++first) {
      if (!try_emplace_back(*first))
        return false;
    }
    return true;
  }

  template <std::input_iterator It, std::sentinel_for<It> Sentinel>
    requires std::constructible_from<T, std::iter_reference_t<It>>
  void append_range(It first, Sentinel last) {
    HYLE_FATAL_ASSERT(try_append_range(first, last));
  }

  void open_gap(size_type index, size_type count) noexcept {
    HYLE_FATAL_ASSERT(index <= size() && count <= N - size());
    if (count == 0)
      return;
    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memmove(data() + index + count, data() + index,
                   (size() - index) * sizeof(T));
    } else {
      for (size_type i = size(); i > index; --i) {
        ::new (static_cast<void *>(data() + i - 1 + count))
            T(std::move(data()[i - 1]));
        data()[i - 1].~T();
      }
    }
  }

  void insert_prepared(size_type index, StaticVector &inserted) noexcept {
    const size_type count = inserted.size();
    HYLE_FATAL_ASSERT(index <= size() && count <= N - size());
    if (count == 0)
      return;
    open_gap(index, count);
    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memcpy(data() + index, inserted.data(), count * sizeof(T));
    } else {
      for (size_type i = 0; i < count; ++i) {
        ::new (static_cast<void *>(data() + index + i))
            T(std::move(inserted.data()[i]));
      }
      inserted.clear();
    }
    set_size(size() + count);
  }

  void move_from(StaticVector &other) noexcept(
      std::is_nothrow_move_constructible_v<T>)
    requires std::is_move_constructible_v<T>
  {
    HYLE_FATAL_ASSERT(empty());
    if constexpr (std::is_trivially_copyable_v<T>) {
      if (!other.empty())
        std::memcpy(data(), other.data(), other.size() * sizeof(T));
      set_size(other.size());
      other.set_size(0);
    } else {
      for (size_type i = 0; i < other.size(); ++i) {
        ::new (static_cast<void *>(data() + i)) T(std::move(other.data()[i]));
        increment_size();
      }
      other.clear();
    }
  }

  void replace_with(StaticVector &replacement) {
    clear();
    if constexpr (std::is_move_constructible_v<T>) {
      move_from(replacement);
    } else {
      append_range(replacement.begin(), replacement.end());
    }
  }

  // Element bytes are deliberately uninitialized: default construction is
  // O(1), independent of N. Only [0, size()) contains live T objects.
  HYLE_NO_UNIQUE_ADDRESS detail::StaticVectorStorage<T, N> storage_;
};

// C++26 names the standard facility inplace_vector. HYLE exposes the explicit
// storage contract as StaticVector while keeping the familiar spelling as a
// source-level alias, not a second implementation.
template <class T, std::size_t N> using InplaceVector = StaticVector<T, N>;

} // namespace hyle::container
