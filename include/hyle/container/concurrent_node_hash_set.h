#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "hyle/container/concurrent_node_hash_map.h"

namespace hyle::container {

namespace detail {
struct ConcurrentNodeSetValue {};
}  // namespace detail

// Stable-node concurrent membership. A successful find() keeps the bucket
// protected through ConstAccessor; release the accessor before re-entering the
// set or calling management operations.
template <class K, class Hasher = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator,
          RetiredTableLifetime TableLifetime =
              RetiredTableLifetime::kMapDestruction>
class ConcurrentNodeHashSet {
  using Value = detail::ConcurrentNodeSetValue;
  using Map = ConcurrentNodeHashMap<K, Value, Hasher, Eq, Alloc, TableLifetime>;

 public:
  using InsertResult = typename Map::InsertResult;
  static constexpr std::size_t kDefaultBucketCount = Map::kDefaultBucketCount;
  static constexpr RetiredTableLifetime kRetiredTableLifetime =
      Map::kRetiredTableLifetime;

  class ConstAccessor {
    friend class ConcurrentNodeHashSet;

   public:
    ConstAccessor() noexcept = default;
    ConstAccessor(const ConstAccessor&) = delete;
    ConstAccessor& operator=(const ConstAccessor&) = delete;
    ConstAccessor(ConstAccessor&&) noexcept = default;
    ConstAccessor& operator=(ConstAccessor&&) noexcept = default;

    explicit operator bool() const noexcept {
      return static_cast<bool>(accessor_);
    }
    const K& key() const noexcept { return accessor_.key(); }
    const K& operator*() const noexcept { return key(); }
    const K* operator->() const noexcept { return &key(); }
    void release() noexcept { accessor_.release(); }

   private:
    typename Map::ConstAccessor accessor_;
  };

  ConcurrentNodeHashSet()
    requires(std::is_default_constructible_v<Alloc> &&
             std::is_default_constructible_v<Hasher> &&
             std::is_default_constructible_v<Eq>)
      = default;

  explicit ConcurrentNodeHashSet(std::size_t bucket_count,
                                 Alloc alloc = Alloc{},
                                 Hasher hash = Hasher{}, Eq eq = Eq{})
      : map_(bucket_count, std::move(alloc), std::move(hash), std::move(eq)) {}

  explicit ConcurrentNodeHashSet(
      mem::ReclamationDomain& domain,
      std::size_t bucket_count = kDefaultBucketCount,
      Alloc alloc = Alloc{}, Hasher hash = Hasher{}, Eq eq = Eq{})
    requires(TableLifetime == RetiredTableLifetime::kAfterReaders)
      : map_(domain, bucket_count, std::move(alloc), std::move(hash),
             std::move(eq)) {}

  ConcurrentNodeHashSet(const ConcurrentNodeHashSet&) = delete;
  ConcurrentNodeHashSet& operator=(const ConcurrentNodeHashSet&) = delete;
  ConcurrentNodeHashSet(ConcurrentNodeHashSet&&) = delete;
  ConcurrentNodeHashSet& operator=(ConcurrentNodeHashSet&&) = delete;

  [[nodiscard]] bool valid() const noexcept { return map_.valid(); }
  [[nodiscard]] std::size_t size() const noexcept { return map_.size(); }
  [[nodiscard]] bool empty() const noexcept { return map_.empty(); }
  [[nodiscard]] std::size_t bucket_count() const noexcept {
    return map_.bucket_count();
  }

  template <class Q>
  bool find(ConstAccessor& result, const Q& key) const noexcept {
    return map_.find(result.accessor_, key);
  }

  template <class Q> [[nodiscard]] bool contains(const Q& key) const noexcept {
    return map_.contains(key);
  }

  InsertResult try_insert(K key) {
    typename Map::Accessor accessor;
    return map_.try_emplace(accessor, std::move(key), Value{});
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

template <class K, class Hasher = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
using RetainedTableConcurrentNodeHashSet =
    ConcurrentNodeHashSet<K, Hasher, Eq, Alloc,
                          RetiredTableLifetime::kMapDestruction>;

template <class K, class Hasher = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
using ReclaimingTableConcurrentNodeHashSet =
    ConcurrentNodeHashSet<K, Hasher, Eq, Alloc,
                          RetiredTableLifetime::kAfterReaders>;

}  // namespace hyle::container
