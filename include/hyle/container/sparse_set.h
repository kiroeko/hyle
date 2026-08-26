#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "hyle/container/pod_vector.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Dense/sparse set for IDs in [0, universe_size). Membership, insertion,
// swap-erase and clear are O(1); iteration is a contiguous dense scan. Erase
// deliberately does not preserve order. The universe is allocated up front so
// insert() cannot allocate in a hot graph/ECS loop.
template <std::unsigned_integral Id = std::uint32_t>
class SparseSet {
 public:
  using value_type = Id;
  using size_type = std::size_t;
  using iterator = const Id*;
  using const_iterator = const Id*;

  SparseSet() noexcept : SparseSet(mem::system_allocator()) {}
  explicit SparseSet(mem::Allocator alloc) noexcept
      : sparse_(alloc), dense_(alloc) {}
  explicit SparseSet(size_type universe_size) : SparseSet() {
    reset_universe(universe_size);
  }
  SparseSet(mem::Allocator alloc, size_type universe_size) : SparseSet(alloc) {
    reset_universe(universe_size);
  }

  mem::Allocator allocator() const noexcept { return sparse_.allocator(); }
  size_type universe_size() const noexcept {
    return static_cast<size_type>(sparse_.size());
  }
  size_type size() const noexcept { return static_cast<size_type>(dense_.size()); }
  bool empty() const noexcept { return dense_.empty(); }
  const Id* data() const noexcept { return dense_.data(); }
  const_iterator begin() const noexcept { return dense_.begin(); }
  const_iterator end() const noexcept { return dense_.end(); }

  bool try_reset_universe(size_type universe_size) noexcept {
    if constexpr (sizeof(Id) < sizeof(size_type)) {
      if (universe_size >
          static_cast<size_type>(std::numeric_limits<Id>::max()) + 1)
        return false;
    }
    SparseSet replacement(allocator());
    if (!replacement.sparse_.try_resize(universe_size, Id{0}) ||
        !replacement.dense_.try_reserve(universe_size))
      return false;
    swap(replacement);
    return true;
  }
  void reset_universe(size_type universe_size) {
    HYLE_FATAL_ASSERT(try_reset_universe(universe_size));
  }

  bool contains(Id id) const noexcept {
    const size_type key = static_cast<size_type>(id);
    if (key >= universe_size()) return false;
    const size_type position = static_cast<size_type>(sparse_[key]);
    return position < size() && dense_[position] == id;
  }

  bool insert(Id id) noexcept {
    const size_type key = static_cast<size_type>(id);
    HYLE_FATAL_ASSERT(key < universe_size());
    if (contains(id)) return false;
    const Id position = static_cast<Id>(dense_.size());
    // dense_ reserved the full universe in reset_universe(), so this cannot
    // allocate or fail. Keep the assertion release-surviving as an invariant.
    HYLE_FATAL_ASSERT(dense_.try_push_back(id));
    sparse_[key] = position;
    return true;
  }

  bool erase(Id id) noexcept {
    if (!contains(id)) return false;
    const size_type key = static_cast<size_type>(id);
    const size_type position = static_cast<size_type>(sparse_[key]);
    const Id moved = dense_.back();
    dense_[position] = moved;
    sparse_[static_cast<size_type>(moved)] = static_cast<Id>(position);
    dense_.pop_back();
    return true;
  }

  void clear() noexcept { dense_.clear(); }

  void swap(SparseSet& other) noexcept {
    sparse_.swap(other.sparse_);
    dense_.swap(other.dense_);
  }

 private:
  PodVector<Id> sparse_;
  PodVector<Id> dense_;
};

template <std::unsigned_integral Id>
void swap(SparseSet<Id>& left, SparseSet<Id>& right) noexcept {
  left.swap(right);
}

}  // namespace hyle::container
