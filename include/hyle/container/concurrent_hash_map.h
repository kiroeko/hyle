#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/detail/adaptive_rw_lock.h"
#include "hyle/container/hash.h"
#include "hyle/container/flat_hash_map.h"
#include "hyle/core/cache.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Sharded concurrent flat hash map for shared registries, memo tables and
// caches. A key maps to one independently locked FlatHashMap shard, so unrelated
// writers proceed in parallel and reads share a shard. Accessors keep the shard
// locked while a reference is live; no pointer can silently outlive an erase or
// rehash as it could in a naive mutex-wrapped flat map.
//
// Alloc/Hash/Eq calls must be safe to invoke concurrently: independent shards
// share copies of those policies and may allocate, hash or compare at the same
// time. The default SystemAllocator and stateless Hash/Eq satisfy this contract.
// Destruction requires all operations and accessors to be quiescent. Re-entering
// this same map from a for_each callback deadlocks and is unsupported. Keep at
// most one live accessor from this map per thread; release it before
// size/reserve/clear/for_each or another lookup (lookup automatically releases
// a reused accessor).
template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
class ConcurrentHashMap {
  static_assert(std::is_copy_constructible_v<Alloc> &&
                    std::is_copy_constructible_v<Hash> &&
                    std::is_copy_constructible_v<Eq>,
                "ConcurrentHashMap copies allocator/hash/eq into its shards");

  using Table = FlatHashMap<K, V, Hash, Eq, Alloc>;

  struct alignas(core::kDestructiveInterferenceSize) Shard {
    Shard(Alloc alloc, Hash hash, Eq eq)
        : table(std::move(alloc), std::move(hash), std::move(eq)) {}

    mutable detail::AdaptiveRwLock lock;
    Table table;
  };

public:
  static constexpr std::size_t kDefaultShardCount = 64;

  enum class InsertResult : std::uint8_t {
    kInserted,
    kAlreadyPresent,
    kAssigned,
    kOutOfMemory,
  };

  class ConstAccessor {
    friend class ConcurrentHashMap;

  public:
    ConstAccessor() noexcept = default;
    ConstAccessor(const ConstAccessor&) = delete;
    ConstAccessor& operator=(const ConstAccessor&) = delete;
    ConstAccessor(ConstAccessor &&other) noexcept { move_from(other); }
    ConstAccessor &operator=(ConstAccessor &&other) noexcept {
      if (this != &other) {
        release();
        move_from(other);
      }
      return *this;
    }
    ~ConstAccessor() { release(); }

    explicit operator bool() const noexcept { return value_ != nullptr; }
    const K &key() const noexcept {
      HYLE_FATAL_ASSERT(key_ != nullptr);
      return *key_;
    }
    const V &value() const noexcept {
      HYLE_FATAL_ASSERT(value_ != nullptr);
      return *value_;
    }
    const V &operator*() const noexcept { return value(); }
    const V *operator->() const noexcept { return &value(); }

    void release() noexcept {
      if (shard_ != nullptr)
        shard_->lock.unlock_shared();
      shard_ = nullptr;
      key_ = nullptr;
      value_ = nullptr;
    }

  private:
    void bind(Shard *shard, const typename Table::Entry *entry) noexcept {
      shard_ = shard;
      key_ = &entry->key();
      value_ = &entry->value();
    }
    void move_from(ConstAccessor &other) noexcept {
      shard_ = std::exchange(other.shard_, nullptr);
      key_ = std::exchange(other.key_, nullptr);
      value_ = std::exchange(other.value_, nullptr);
    }

    Shard *shard_ = nullptr;
    const K *key_ = nullptr;
    const V *value_ = nullptr;
  };

  class Accessor {
    friend class ConcurrentHashMap;

  public:
    Accessor() noexcept = default;
    Accessor(const Accessor&) = delete;
    Accessor& operator=(const Accessor&) = delete;
    Accessor(Accessor &&other) noexcept { move_from(other); }
    Accessor &operator=(Accessor &&other) noexcept {
      if (this != &other) {
        release();
        move_from(other);
      }
      return *this;
    }
    ~Accessor() { release(); }

    explicit operator bool() const noexcept { return value_ != nullptr; }
    const K &key() const noexcept {
      HYLE_FATAL_ASSERT(key_ != nullptr);
      return *key_;
    }
    V &value() const noexcept {
      HYLE_FATAL_ASSERT(value_ != nullptr);
      return *value_;
    }
    V &operator*() const noexcept { return value(); }
    V *operator->() const noexcept { return &value(); }

    void release() noexcept {
      if (shard_ != nullptr)
        shard_->lock.unlock();
      shard_ = nullptr;
      key_ = nullptr;
      value_ = nullptr;
    }

  private:
    void bind(Shard *shard, typename Table::Entry *entry) noexcept {
      shard_ = shard;
      key_ = &entry->key();
      value_ = &entry->value();
    }
    void move_from(Accessor &other) noexcept {
      shard_ = std::exchange(other.shard_, nullptr);
      key_ = std::exchange(other.key_, nullptr);
      value_ = std::exchange(other.value_, nullptr);
    }

    Shard *shard_ = nullptr;
    const K *key_ = nullptr;
    V *value_ = nullptr;
  };

  ConcurrentHashMap()
    requires(std::is_default_constructible_v<Alloc> &&
             std::is_default_constructible_v<Hash> &&
             std::is_default_constructible_v<Eq>)
      : ConcurrentHashMap(kDefaultShardCount, Alloc{}, Hash{}, Eq{}) {}

  explicit ConcurrentHashMap(std::size_t shard_count, Alloc alloc = Alloc{},
                             Hash hash = Hash{}, Eq eq = Eq{})
      : alloc_(std::move(alloc)), hash_(std::move(hash)), eq_(std::move(eq)) {
    initialize(shard_count);
  }

  ~ConcurrentHashMap() { destroy_shards(); }
  ConcurrentHashMap(const ConcurrentHashMap&) = delete;
  ConcurrentHashMap& operator=(const ConcurrentHashMap&) = delete;
  ConcurrentHashMap(ConcurrentHashMap&&) = delete;
  ConcurrentHashMap& operator=(ConcurrentHashMap&&) = delete;

  bool valid() const noexcept { return shards_ != nullptr; }
  std::size_t shard_count() const noexcept { return shard_count_; }

  template <class Q>
    requires(
        std::is_nothrow_invocable_r_v<std::uint64_t, const Hash &, const Q &> &&
        std::is_nothrow_invocable_r_v<bool, const Eq &, const K &, const Q &>)
  bool find(ConstAccessor &result, const Q &key) const noexcept {
    result.release();
    HYLE_FATAL_ASSERT(valid());
    const std::uint64_t hash = hash_(key);
    Shard &shard = shard_for(hash);
    shard.lock.lock_shared();
    const typename Table::Slot *slot =
        shard.table.find_slot_prehashed(key, hash);
    if (slot == nullptr) {
      shard.lock.unlock_shared();
      return false;
    }
    result.bind(&shard, slot);
    return true;
  }

  template <class Q>
    requires(
        std::is_nothrow_invocable_r_v<std::uint64_t, const Hash &, const Q &> &&
        std::is_nothrow_invocable_r_v<bool, const Eq &, const K &, const Q &>)
  bool find(Accessor &result, const Q &key) noexcept {
    result.release();
    HYLE_FATAL_ASSERT(valid());
    const std::uint64_t hash = hash_(key);
    Shard &shard = shard_for(hash);
    shard.lock.lock();
    typename Table::Slot *slot = shard.table.find_slot_prehashed(key, hash);
    if (slot == nullptr) {
      shard.lock.unlock();
      return false;
    }
    result.bind(&shard, slot);
    return true;
  }

  // Snapshot only: another thread may erase immediately after this returns.
  template <class Q> bool contains(const Q &key) const noexcept {
    ConstAccessor accessor;
    return find(accessor, key);
  }

  template <class... Args>
    requires std::is_nothrow_constructible_v<V, Args...>
  InsertResult try_emplace(Accessor &result, K key, Args &&...args) {
    result.release();
    HYLE_FATAL_ASSERT(valid());
    const std::uint64_t hash = hash_(key);
    Shard &shard = shard_for(hash);
    shard.lock.lock();
    if (typename Table::Slot *slot =
            shard.table.find_slot_prehashed(key, hash)) {
      result.bind(&shard, slot);
      return InsertResult::kAlreadyPresent;
    }
    if (!shard.table.try_prepare_insert()) {
      shard.lock.unlock();
      return InsertResult::kOutOfMemory;
    }
    typename Table::Slot *slot = shard.table.emplace_absent_prehashed(
        std::move(key), hash, std::forward<Args>(args)...);
    result.bind(&shard, slot);
    return InsertResult::kInserted;
  }

  template <class... Args>
    requires std::is_nothrow_constructible_v<V, Args...>
  bool emplace(Accessor &result, K key, Args &&...args) {
    const InsertResult inserted =
        try_emplace(result, std::move(key), std::forward<Args>(args)...);
    HYLE_FATAL_ASSERT(inserted != InsertResult::kOutOfMemory);
    return inserted == InsertResult::kInserted;
  }

  InsertResult try_insert_or_assign(Accessor &result, K key, V value) {
    result.release();
    HYLE_FATAL_ASSERT(valid());
    const std::uint64_t hash = hash_(key);
    Shard &shard = shard_for(hash);
    shard.lock.lock();
    if (typename Table::Slot *slot =
            shard.table.find_slot_prehashed(key, hash)) {
      slot->value() = std::move(value);
      result.bind(&shard, slot);
      return InsertResult::kAssigned;
    }
    if (!shard.table.try_prepare_insert()) {
      shard.lock.unlock();
      return InsertResult::kOutOfMemory;
    }
    typename Table::Slot *slot = shard.table.emplace_absent_prehashed(
        std::move(key), hash, std::move(value));
    result.bind(&shard, slot);
    return InsertResult::kInserted;
  }

  bool insert_or_assign(Accessor &result, K key, V value) {
    const InsertResult inserted =
        try_insert_or_assign(result, std::move(key), std::move(value));
    HYLE_FATAL_ASSERT(inserted != InsertResult::kOutOfMemory);
    return inserted == InsertResult::kInserted;
  }

  template <class Q> bool erase(const Q &key) noexcept {
    HYLE_FATAL_ASSERT(valid());
    const std::uint64_t hash = hash_(key);
    Shard &shard = shard_for(hash);
    shard.lock.lock();
    const bool erased = shard.table.erase(key);
    shard.lock.unlock();
    return erased;
  }

  // Reserves approximately count/shards entries in every shard. With HYLE's
  // mixed 64-bit hash this is the useful total-size reserve contract; a custom
  // hasher that maps most keys to one shard can still force that shard to grow.
  // The try_ form reports allocation failure; the ordinary form is fail-fast.
  bool try_reserve(std::size_t count) {
    HYLE_FATAL_ASSERT(valid());
    if (count > static_cast<std::size_t>(-1) - (shard_count_ - 1))
      return false;
    const std::size_t per_shard = (count + shard_count_ - 1) / shard_count_;
    for (std::size_t i = 0; i < shard_count_; ++i) {
      Shard &shard = shards_[i];
      shard.lock.lock();
      const bool ok = shard.table.try_reserve(per_shard);
      shard.lock.unlock();
      if (!ok)
        return false;
    }
    return true;
  }

  void reserve(std::size_t count) {
    HYLE_FATAL_ASSERT(try_reserve(count));
  }

  std::size_t size() const noexcept {
    HYLE_FATAL_ASSERT(valid());
    std::size_t result = 0;
    for (std::size_t i = 0; i < shard_count_; ++i) {
      Shard &shard = shards_[i];
      shard.lock.lock_shared();
      result += shard.table.size();
      shard.lock.unlock_shared();
    }
    return result;
  }

  bool empty() const noexcept { return size() == 0; }

  void clear() noexcept {
    HYLE_FATAL_ASSERT(valid());
    for (std::size_t i = 0; i < shard_count_; ++i) {
      Shard &shard = shards_[i];
      shard.lock.lock();
      shard.table.clear();
      shard.lock.unlock();
    }
  }

  // Weakly consistent across shards: each shard is stable while visited, but
  // other shards can change between callbacks. fn must not re-enter this map.
  template <class Fn> void for_each(Fn &&fn) const {
    HYLE_FATAL_ASSERT(valid());
    for (std::size_t i = 0; i < shard_count_; ++i) {
      Shard &shard = shards_[i];
      shard.lock.lock_shared();
      try {
        static_cast<const Table &>(shard.table).for_each(fn);
      } catch (...) {
        shard.lock.unlock_shared();
        throw;
      }
      shard.lock.unlock_shared();
    }
  }

private:
  void initialize(std::size_t count) {
    if (count == 0 || (count & (count - 1)) != 0 ||
        count > static_cast<std::size_t>(-1) / sizeof(Shard))
      return;
    void *storage = alloc_.allocate(sizeof(Shard) * count, alignof(Shard));
    if (storage == nullptr)
      return;
    shards_ = static_cast<Shard *>(storage);
    std::size_t built = 0;
    try {
      for (; built < count; ++built)
        ::new (static_cast<void *>(shards_ + built)) Shard(alloc_, hash_, eq_);
    } catch (...) {
      while (built != 0)
        shards_[--built].~Shard();
      alloc_.deallocate(shards_, sizeof(Shard) * count, alignof(Shard));
      shards_ = nullptr;
      return;
    }
    shard_count_ = count;
    shard_mask_ = count - 1;
  }

  void destroy_shards() noexcept {
    if (shards_ == nullptr)
      return;
    for (std::size_t i = shard_count_; i != 0; --i)
      shards_[i - 1].~Shard();
    alloc_.deallocate(shards_, sizeof(Shard) * shard_count_, alignof(Shard));
    shards_ = nullptr;
    shard_count_ = 0;
    shard_mask_ = 0;
  }

  Shard &shard_for(std::uint64_t hash) const noexcept {
    // Swiss h2 consumes low seven bits and h1 starts above them. Taking the
    // high bits for sharding preserves both entropy sources, like Folly's
    // sharded map.
    return shards_[(hash >> 32) & shard_mask_];
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  HYLE_NO_UNIQUE_ADDRESS Hash hash_{};
  HYLE_NO_UNIQUE_ADDRESS Eq eq_{};
  Shard *shards_ = nullptr;
  std::size_t shard_count_ = 0;
  std::size_t shard_mask_ = 0;
};

// Prefer this explicit name in new code when the distinction from
// ConcurrentNodeHashMap matters. The shorter spelling remains the same type.
template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
using ConcurrentFlatHashMap = ConcurrentHashMap<K, V, Hash, Eq, Alloc>;

} // namespace hyle::container
