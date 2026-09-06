#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

#include "hyle/container/concurrent_hash_map.h"

namespace hyle::container {

namespace detail {
struct ConcurrentFlatSetValue {};
}  // namespace detail

// Compact sharded concurrent membership. contains() is a snapshot; callbacks
// hold one shard's shared lock and must not re-enter this set. Destruction and
// management operations require all callers to be quiescent.
template <class K, class Hasher = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
class ConcurrentFlatHashSet {
  using Value = detail::ConcurrentFlatSetValue;
  using Map = ConcurrentFlatHashMap<K, Value, Hasher, Eq, Alloc>;
  using Storage = FlatHashMap<K, Value, Hasher, Eq, Alloc>;
  static_assert(sizeof(typename Storage::value_type) == sizeof(K),
                "ConcurrentFlatHashSet must not pay mapped-value storage");

 public:
  using InsertResult = typename Map::InsertResult;
  static constexpr std::size_t kDefaultShardCount = Map::kDefaultShardCount;

  ConcurrentFlatHashSet()
    requires(std::is_default_constructible_v<Alloc> &&
             std::is_default_constructible_v<Hasher> &&
             std::is_default_constructible_v<Eq>)
      = default;

  explicit ConcurrentFlatHashSet(std::size_t shard_count,
                                 Alloc alloc = Alloc{},
                                 Hasher hash = Hasher{}, Eq eq = Eq{})
      : map_(shard_count, std::move(alloc), std::move(hash), std::move(eq)) {}

  ConcurrentFlatHashSet(const ConcurrentFlatHashSet&) = delete;
  ConcurrentFlatHashSet& operator=(const ConcurrentFlatHashSet&) = delete;
  ConcurrentFlatHashSet(ConcurrentFlatHashSet&&) = delete;
  ConcurrentFlatHashSet& operator=(ConcurrentFlatHashSet&&) = delete;

  [[nodiscard]] bool valid() const noexcept { return map_.valid(); }
  [[nodiscard]] std::size_t shard_count() const noexcept {
    return map_.shard_count();
  }
  [[nodiscard]] std::size_t size() const noexcept { return map_.size(); }
  [[nodiscard]] bool empty() const noexcept { return map_.empty(); }

  template <class Q> [[nodiscard]] bool contains(const Q& key) const noexcept {
    return map_.contains(key);
  }

  InsertResult try_insert(K key) {
    typename Map::Accessor accessor;
    const InsertResult result =
        map_.try_emplace(accessor, std::move(key), Value{});
    return result;
  }

  bool insert(K key) {
    const InsertResult result = try_insert(std::move(key));
    HYLE_FATAL_ASSERT(result != InsertResult::kOutOfMemory);
    return result == InsertResult::kInserted;
  }

  template <class Q> bool erase(const Q& key) noexcept {
    return map_.erase(key);
  }

  bool try_reserve(std::size_t count) { return map_.try_reserve(count); }
  void reserve(std::size_t count) { map_.reserve(count); }
  void clear() noexcept { map_.clear(); }

  template <class Fn> void for_each(Fn&& fn) const {
    map_.for_each([&](const K& key, const Value&) { std::invoke(fn, key); });
  }

 private:
  Map map_;
};

}  // namespace hyle::container
