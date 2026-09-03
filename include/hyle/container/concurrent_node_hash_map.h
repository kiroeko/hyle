#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/detail/adaptive_rw_lock.h"
#include "hyle/container/hash.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/reclamation.h"
#include "hyle/mem/system_inline.h"

namespace hyle::container {

// Old bucket tables may either be deliberately retained until map destruction
// or returned after every pre-publication reader leaves. Retention is the
// zero-read-tax default; reclamation is an explicit memory/latency trade-off
// for long-lived registries that continue to grow during service.
enum class RetiredTableLifetime : std::uint8_t {
  kAfterReaders,
  kMapDestruction,
};

// Fine-grained node hash map for read-heavy shared registries and long-lived
// accessors. Each bucket owns an independent reader/writer lock; values remain
// at stable addresses across rehash. This is the oneTBB concurrent_hash_map
// family, complementary to ConcurrentHashMap's compact sharded-flat family.
//
// Rehash is a cold stop-the-table phase: it drains each bucket lock, relinks
// stable nodes, then publishes a new bucket array. The default retains old
// arrays until destruction; geometric growth bounds their total capacity to
// less than twice the current table. The explicit reclaiming-table alias uses
// an allocation-free read-side guard and returns arrays after preceding readers
// leave. Alloc/Hash/Eq must be safe for concurrent calls. Destruction requires
// quiescence. Management operations
// reserve/clear/for_each must not be called while the same thread holds an
// accessor. Keep at most one live accessor from this map per thread and release
// it before another lookup (lookup automatically releases a reused accessor).
template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator,
          RetiredTableLifetime TableLifetime =
              RetiredTableLifetime::kMapDestruction>
class ConcurrentNodeHashMap {
  static constexpr bool kReclaimAfterReaders =
      TableLifetime == RetiredTableLifetime::kAfterReaders;
  static_assert(std::is_nothrow_move_constructible_v<K> &&
                    std::is_nothrow_move_constructible_v<V> &&
                    std::is_nothrow_destructible_v<K> &&
                    std::is_nothrow_destructible_v<V> &&
                    std::is_nothrow_move_assignable_v<V>,
                "ConcurrentNodeHashMap requires nothrow key/value moves");
  static_assert(
      std::is_nothrow_invocable_r_v<std::uint64_t, Hash, const K &> &&
          std::is_nothrow_invocable_r_v<bool, Eq, const K &, const K &>,
      "ConcurrentNodeHashMap requires noexcept hash/equality");

  struct Node {
    template <class... Args>
      requires std::is_nothrow_constructible_v<V, Args...>
    Node(std::uint64_t node_hash, K node_key, Node *node_next,
         Args &&...args) noexcept
        : hash(node_hash), key(std::move(node_key)),
          value(std::forward<Args>(args)...), next(node_next) {}

    std::uint64_t hash;
    K key;
    HYLE_NO_UNIQUE_ADDRESS V value;
    Node *next;
  };

  struct Bucket {
    detail::AdaptiveRwLock lock;
    Node *head = nullptr;
  };

  struct Table;

  struct ReclaimedTableState {
    mem::RetireRecord record;
    ConcurrentNodeHashMap *owner = nullptr;
  };

  struct RetainedTableState {
    Table *next = nullptr;
  };

  using TableRetirementState =
      std::conditional_t<kReclaimAfterReaders, ReclaimedTableState,
                         RetainedTableState>;

  struct Table {
    Bucket *buckets = nullptr;
    std::size_t bucket_count = 0;
    std::size_t mask = 0;
    std::atomic<std::size_t> size{0};
    HYLE_NO_UNIQUE_ADDRESS TableRetirementState retirement;
  };

  struct ReclamationLifetimeState {
    mem::ReclamationDomain *domain = nullptr;
  };

  struct RetainedLifetimeState {
    Table *retired = nullptr;
  };

  using LifetimeState =
      std::conditional_t<kReclaimAfterReaders, ReclamationLifetimeState,
                         RetainedLifetimeState>;

  class TableReadGuard final {
  public:
    explicit TableReadGuard(const ConcurrentNodeHashMap &owner) noexcept {
      if constexpr (kReclaimAfterReaders)
        guard_ = owner.reclamation_domain().read();
    }

    [[nodiscard]] Table *
    protect(const std::atomic<Table *> &published) const noexcept {
      if constexpr (kReclaimAfterReaders)
        return guard_.protect(published);
      return published.load(std::memory_order_acquire);
    }

    void reset() noexcept {
      if constexpr (kReclaimAfterReaders)
        guard_.reset();
    }

  private:
    struct EmptyGuard {};
    HYLE_NO_UNIQUE_ADDRESS
    std::conditional_t<kReclaimAfterReaders, mem::ReclamationGuard, EmptyGuard>
        guard_;
  };

public:
  static constexpr std::size_t kDefaultBucketCount = 64;
  static constexpr RetiredTableLifetime kRetiredTableLifetime = TableLifetime;

  enum class InsertResult : std::uint8_t {
    kInserted,
    kAlreadyPresent,
    kAssigned,
    kOutOfMemory,
  };

  class ConstAccessor {
    friend class ConcurrentNodeHashMap;

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

    explicit operator bool() const noexcept { return node_ != nullptr; }
    const K &key() const noexcept {
      HYLE_FATAL_ASSERT(node_ != nullptr);
      return node_->key;
    }
    const V &value() const noexcept {
      HYLE_FATAL_ASSERT(node_ != nullptr);
      return node_->value;
    }
    const V &operator*() const noexcept { return value(); }
    const V *operator->() const noexcept { return &value(); }

    void release() noexcept {
      if (bucket_ != nullptr)
        bucket_->lock.unlock_shared();
      bucket_ = nullptr;
      node_ = nullptr;
    }

  private:
    void bind(Bucket *bucket, const Node *node) noexcept {
      bucket_ = bucket;
      node_ = node;
    }
    void move_from(ConstAccessor &other) noexcept {
      bucket_ = std::exchange(other.bucket_, nullptr);
      node_ = std::exchange(other.node_, nullptr);
    }

    Bucket *bucket_ = nullptr;
    const Node *node_ = nullptr;
  };

  class Accessor {
    friend class ConcurrentNodeHashMap;

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

    explicit operator bool() const noexcept { return node_ != nullptr; }
    const K &key() const noexcept {
      HYLE_FATAL_ASSERT(node_ != nullptr);
      return node_->key;
    }
    V &value() const noexcept {
      HYLE_FATAL_ASSERT(node_ != nullptr);
      return node_->value;
    }
    V &operator*() const noexcept { return value(); }
    V *operator->() const noexcept { return &value(); }

    void release() noexcept {
      if (bucket_ != nullptr)
        bucket_->lock.unlock();
      bucket_ = nullptr;
      node_ = nullptr;
    }

  private:
    void bind(Bucket *bucket, Node *node) noexcept {
      bucket_ = bucket;
      node_ = node;
    }
    void move_from(Accessor &other) noexcept {
      bucket_ = std::exchange(other.bucket_, nullptr);
      node_ = std::exchange(other.node_, nullptr);
    }

    Bucket *bucket_ = nullptr;
    Node *node_ = nullptr;
  };

  ConcurrentNodeHashMap()
    requires(std::is_default_constructible_v<Alloc> &&
             std::is_default_constructible_v<Hash> &&
             std::is_default_constructible_v<Eq>)
      : ConcurrentNodeHashMap(kDefaultBucketCount, Alloc{}, Hash{}, Eq{}) {}

  explicit ConcurrentNodeHashMap(std::size_t bucket_count,
                                 Alloc alloc = Alloc{}, Hash hash = Hash{},
                                 Eq eq = Eq{})
      : alloc_(std::move(alloc)), hash_(std::move(hash)), eq_(std::move(eq)),
        lifetime_(default_lifetime_state()) {
    current_.store(allocate_table(normalize_bucket_count(bucket_count)),
                   std::memory_order_relaxed);
  }

  explicit ConcurrentNodeHashMap(mem::ReclamationDomain &domain,
                                 std::size_t bucket_count = kDefaultBucketCount,
                                 Alloc alloc = Alloc{}, Hash hash = Hash{},
                                 Eq eq = Eq{})
    requires(kReclaimAfterReaders)
      : alloc_(std::move(alloc)), hash_(std::move(hash)), eq_(std::move(eq)),
        lifetime_(ReclamationLifetimeState{&domain}) {
    current_.store(allocate_table(normalize_bucket_count(bucket_count)),
                   std::memory_order_relaxed);
  }

  ~ConcurrentNodeHashMap() { destroy_all(); }
  ConcurrentNodeHashMap(const ConcurrentNodeHashMap&) = delete;
  ConcurrentNodeHashMap& operator=(const ConcurrentNodeHashMap&) = delete;
  ConcurrentNodeHashMap(ConcurrentNodeHashMap&&) = delete;
  ConcurrentNodeHashMap& operator=(ConcurrentNodeHashMap&&) = delete;

  bool valid() const noexcept {
    return current_.load(std::memory_order_relaxed) != nullptr;
  }

  std::size_t size() const noexcept {
    TableReadGuard guard(*this);
    Table *table = guard.protect(current_);
    return table == nullptr ? 0 : table->size.load(std::memory_order_relaxed);
  }
  bool empty() const noexcept { return size() == 0; }
  std::size_t bucket_count() const noexcept {
    TableReadGuard guard(*this);
    Table *table = guard.protect(current_);
    return table == nullptr ? 0 : table->bucket_count;
  }

#if defined(HYLE_TESTING)
  using TableLoadHook = void (*)(void *) noexcept;

  // TEST-ONLY: pause a lookup after it protects the current table but before
  // it acquires that table's bucket lock, making the reclamation race
  // deterministic without exposing table internals in the shipped API.
  void set_table_load_hook_for_testing(TableLoadHook hook,
                                       void *state = nullptr) noexcept {
    table_load_hook_state_.store(state, std::memory_order_relaxed);
    table_load_hook_.store(hook, std::memory_order_release);
  }
#endif

  template <class Q>
    requires(
        std::is_nothrow_invocable_r_v<std::uint64_t, const Hash &, const Q &> &&
        std::is_nothrow_invocable_r_v<bool, const Eq &, const K &, const Q &>)
  bool find(ConstAccessor &result, const Q &key) const noexcept {
    result.release();
    HYLE_FATAL_ASSERT(valid());
    const std::uint64_t hash = hash_(key);
    for (;;) {
      TableReadGuard table_guard(*this);
      Table *table = table_guard.protect(current_);
      notify_table_loaded_for_testing();
      Bucket &bucket = table->buckets[hash & table->mask];
      bucket.lock.lock_shared();
      if (current_.load(std::memory_order_acquire) != table) {
        bucket.lock.unlock_shared();
        continue;
      }
      table_guard.reset();
      const Node *node = find_node(bucket, key, hash);
      if (node == nullptr) {
        bucket.lock.unlock_shared();
        return false;
      }
      result.bind(&bucket, node);
      return true;
    }
  }

  template <class Q>
    requires(
        std::is_nothrow_invocable_r_v<std::uint64_t, const Hash &, const Q &> &&
        std::is_nothrow_invocable_r_v<bool, const Eq &, const K &, const Q &>)
  bool find(Accessor &result, const Q &key) noexcept {
    result.release();
    HYLE_FATAL_ASSERT(valid());
    const std::uint64_t hash = hash_(key);
    for (;;) {
      TableReadGuard table_guard(*this);
      Table *table = table_guard.protect(current_);
      notify_table_loaded_for_testing();
      Bucket &bucket = table->buckets[hash & table->mask];
      bucket.lock.lock();
      if (current_.load(std::memory_order_acquire) != table) {
        bucket.lock.unlock();
        continue;
      }
      table_guard.reset();
      Node *node = find_node(bucket, key, hash);
      if (node == nullptr) {
        bucket.lock.unlock();
        return false;
      }
      result.bind(&bucket, node);
      return true;
    }
  }

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
    bool growth_checked = false;
    for (;;) {
      TableReadGuard table_guard(*this);
      Table *table = table_guard.protect(current_);
      notify_table_loaded_for_testing();
      Bucket &bucket = table->buckets[hash & table->mask];
      bucket.lock.lock();
      if (current_.load(std::memory_order_acquire) != table) {
        bucket.lock.unlock();
        continue;
      }
      table_guard.reset();
      if (Node *existing = find_node(bucket, key, hash)) {
        result.bind(&bucket, existing);
        return InsertResult::kAlreadyPresent;
      }
      if (!growth_checked &&
          table->size.load(std::memory_order_relaxed) >= table->bucket_count) {
        bucket.lock.unlock();
        maybe_grow();
        growth_checked = true;
        continue;
      }
      Node *node = allocate_node(hash, std::move(key), bucket.head,
                                 std::forward<Args>(args)...);
      if (node == nullptr) {
        bucket.lock.unlock();
        return InsertResult::kOutOfMemory;
      }
      bucket.head = node;
      table->size.fetch_add(1, std::memory_order_relaxed);
      result.bind(&bucket, node);
      return InsertResult::kInserted;
    }
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
    bool growth_checked = false;
    for (;;) {
      TableReadGuard table_guard(*this);
      Table *table = table_guard.protect(current_);
      notify_table_loaded_for_testing();
      Bucket &bucket = table->buckets[hash & table->mask];
      bucket.lock.lock();
      if (current_.load(std::memory_order_acquire) != table) {
        bucket.lock.unlock();
        continue;
      }
      table_guard.reset();
      if (Node *existing = find_node(bucket, key, hash)) {
        existing->value = std::move(value);
        result.bind(&bucket, existing);
        return InsertResult::kAssigned;
      }
      if (!growth_checked &&
          table->size.load(std::memory_order_relaxed) >= table->bucket_count) {
        bucket.lock.unlock();
        maybe_grow();
        growth_checked = true;
        continue;
      }
      Node *node =
          allocate_node(hash, std::move(key), bucket.head, std::move(value));
      if (node == nullptr) {
        bucket.lock.unlock();
        return InsertResult::kOutOfMemory;
      }
      bucket.head = node;
      table->size.fetch_add(1, std::memory_order_relaxed);
      result.bind(&bucket, node);
      return InsertResult::kInserted;
    }
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
    for (;;) {
      TableReadGuard table_guard(*this);
      Table *table = table_guard.protect(current_);
      notify_table_loaded_for_testing();
      Bucket &bucket = table->buckets[hash & table->mask];
      bucket.lock.lock();
      if (current_.load(std::memory_order_acquire) != table) {
        bucket.lock.unlock();
        continue;
      }
      table_guard.reset();
      Node **link = &bucket.head;
      while (*link != nullptr &&
             ((*link)->hash != hash || !eq_((*link)->key, key)))
        link = &(*link)->next;
      if (*link == nullptr) {
        bucket.lock.unlock();
        return false;
      }
      Node *erased = *link;
      *link = erased->next;
      table->size.fetch_sub(1, std::memory_order_relaxed);
      destroy_node(erased);
      bucket.lock.unlock();
      return true;
    }
  }

  // The try_ form reports table-allocation failure without moving any node;
  // the ordinary form follows HYLE's fail-fast allocation convention.
  bool try_reserve(std::size_t count) {
    HYLE_FATAL_ASSERT(valid());
    const std::size_t target = normalize_bucket_count(count);
    if (target == 0)
      return false;
    std::lock_guard lock(rehash_mutex_);
    Table *table = current_.load(std::memory_order_acquire);
    if (target <= table->bucket_count)
      return true;
    return rehash_locked(table, target);
  }

  void reserve(std::size_t count) {
    HYLE_FATAL_ASSERT(try_reserve(count));
  }

  void clear() noexcept {
    HYLE_FATAL_ASSERT(valid());
    std::lock_guard lock(rehash_mutex_);
    Table *table = current_.load(std::memory_order_acquire);
    close_and_lock_all(table);
    for (std::size_t i = 0; i < table->bucket_count; ++i) {
      Node *node = table->buckets[i].head;
      while (node != nullptr) {
        Node *next = node->next;
        destroy_node(node);
        node = next;
      }
      table->buckets[i].head = nullptr;
    }
    table->size.store(0, std::memory_order_relaxed);
    reopen_and_unlock_all(table);
  }

  template <class Fn> void for_each(Fn &&fn) const {
    HYLE_FATAL_ASSERT(valid());
    std::lock_guard management_lock(rehash_mutex_);
    Table *table = current_.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < table->bucket_count; ++i) {
      Bucket &bucket = table->buckets[i];
      bucket.lock.lock_shared();
      try {
        for (const Node *node = bucket.head; node != nullptr; node = node->next)
          fn(node->key, node->value);
      } catch (...) {
        bucket.lock.unlock_shared();
        throw;
      }
      bucket.lock.unlock_shared();
    }
  }

private:
  [[nodiscard]] static LifetimeState default_lifetime_state() noexcept {
    if constexpr (kReclaimAfterReaders) {
      return ReclamationLifetimeState{&mem::default_reclamation_domain()};
    } else {
      return RetainedLifetimeState{};
    }
  }

  [[nodiscard]] mem::ReclamationDomain &reclamation_domain() const noexcept
    requires(kReclaimAfterReaders)
  {
    HYLE_FATAL_ASSERT(lifetime_.domain != nullptr);
    return *lifetime_.domain;
  }

  void notify_table_loaded_for_testing() const noexcept {
#if defined(HYLE_TESTING)
    if (TableLoadHook hook = table_load_hook_.load(std::memory_order_acquire)) {
      hook(table_load_hook_state_.load(std::memory_order_relaxed));
    }
#endif
  }

  static std::size_t normalize_bucket_count(std::size_t requested) noexcept {
    if (requested == 0)
      requested = kDefaultBucketCount;
    std::size_t count = 16;
    while (count < requested) {
      if (count > static_cast<std::size_t>(-1) / 2)
        return 0;
      count *= 2;
    }
    return count;
  }

  Table *allocate_table(std::size_t count) noexcept {
    if (count == 0 || count > static_cast<std::size_t>(-1) / sizeof(Bucket))
      return nullptr;
    void *table_storage = alloc_.allocate(sizeof(Table), alignof(Table));
    if (table_storage == nullptr)
      return nullptr;
    Table *table = ::new (table_storage) Table;
    void *bucket_storage =
        alloc_.allocate(sizeof(Bucket) * count, alignof(Bucket));
    if (bucket_storage == nullptr) {
      table->~Table();
      alloc_.deallocate(table, sizeof(Table), alignof(Table));
      return nullptr;
    }
    table->buckets = static_cast<Bucket *>(bucket_storage);
    for (std::size_t i = 0; i < count; ++i)
      ::new (static_cast<void *>(table->buckets + i)) Bucket;
    table->bucket_count = count;
    table->mask = count - 1;
    if constexpr (kReclaimAfterReaders)
      table->retirement.owner = this;
    return table;
  }

  void free_table(Table *table, bool destroy_nodes) noexcept {
    if (table == nullptr)
      return;
    for (std::size_t i = 0; i < table->bucket_count; ++i) {
      if (destroy_nodes) {
        Node *node = table->buckets[i].head;
        while (node != nullptr) {
          Node *next = node->next;
          destroy_node(node);
          node = next;
        }
      }
      table->buckets[i].~Bucket();
    }
    alloc_.deallocate(table->buckets, sizeof(Bucket) * table->bucket_count,
                      alignof(Bucket));
    table->~Table();
    alloc_.deallocate(table, sizeof(Table), alignof(Table));
  }

  void destroy_all() noexcept {
    Table *current = current_.load(std::memory_order_relaxed);
    if (current == nullptr)
      return;
    if constexpr (kReclaimAfterReaders)
      static_cast<void>(reclamation_domain().synchronize());
    free_table(current, true);
    if constexpr (!kReclaimAfterReaders) {
      Table *retired = lifetime_.retired;
      while (retired != nullptr) {
        Table *next = retired->retirement.next;
        free_table(retired, false);
        retired = next;
      }
      lifetime_.retired = nullptr;
    }
    current_.store(nullptr, std::memory_order_relaxed);
  }

  static void reclaim_retired_table(void *raw) noexcept
    requires(kReclaimAfterReaders)
  {
    Table *table = static_cast<Table *>(raw);
    HYLE_FATAL_ASSERT(table != nullptr);
    HYLE_FATAL_ASSERT(table->retirement.owner != nullptr);
    table->retirement.owner->free_table(table, false);
  }

  template <class... Args>
  Node *allocate_node(std::uint64_t hash, K key, Node *next,
                      Args &&...args) noexcept {
    void *storage = alloc_.allocate(sizeof(Node), alignof(Node));
    if (storage == nullptr)
      return nullptr;
    return ::new (storage)
        Node(hash, std::move(key), next, std::forward<Args>(args)...);
  }

  void destroy_node(Node *node) noexcept {
    node->~Node();
    alloc_.deallocate(node, sizeof(Node), alignof(Node));
  }

  template <class Q>
  Node *find_node(Bucket &bucket, const Q &key,
                  std::uint64_t hash) const noexcept {
    for (Node *node = bucket.head; node != nullptr; node = node->next)
      if (node->hash == hash && HYLE_LIKELY(eq_(node->key, key)))
        return node;
    return nullptr;
  }

  void maybe_grow() {
    {
      TableReadGuard guard(*this);
      Table *observed = guard.protect(current_);
      if (observed->size.load(std::memory_order_relaxed) <
          observed->bucket_count)
        return;
    }
    std::lock_guard lock(rehash_mutex_);
    Table *table = current_.load(std::memory_order_acquire);
    if (table->size.load(std::memory_order_relaxed) < table->bucket_count ||
        table->bucket_count > static_cast<std::size_t>(-1) / 2)
      return;
    (void)rehash_locked(table, table->bucket_count * 2);
  }

  void close_and_lock_all(Table *table) const noexcept {
    for (std::size_t i = 0; i < table->bucket_count; ++i)
      table->buckets[i].lock.lock();
  }

  void reopen_and_unlock_all(Table *table) const noexcept {
    for (std::size_t i = table->bucket_count; i != 0; --i)
      table->buckets[i - 1].lock.unlock();
  }

  bool rehash_locked(Table *old_table, std::size_t new_count) {
    Table *replacement = allocate_table(new_count);
    if (replacement == nullptr)
      return false;
    close_and_lock_all(old_table);
    for (std::size_t i = 0; i < old_table->bucket_count; ++i) {
      Node *node = old_table->buckets[i].head;
      while (node != nullptr) {
        Node *next = node->next;
        Bucket &target = replacement->buckets[node->hash & replacement->mask];
        node->next = target.head;
        target.head = node;
        node = next;
      }
    }
    replacement->size.store(old_table->size.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
    current_.store(replacement, std::memory_order_release);
    for (std::size_t i = old_table->bucket_count; i != 0; --i)
      old_table->buckets[i - 1].lock.unlock();
    if constexpr (kReclaimAfterReaders) {
      mem::ReclamationDomain &domain = reclamation_domain();
      domain.retire(old_table->retirement.record, old_table,
                    &ConcurrentNodeHashMap::reclaim_retired_table);
      static_cast<void>(domain.try_reclaim());
    } else {
      old_table->retirement.next = lifetime_.retired;
      lifetime_.retired = old_table;
    }
    return true;
  }

  HYLE_NO_UNIQUE_ADDRESS Alloc alloc_{};
  HYLE_NO_UNIQUE_ADDRESS Hash hash_{};
  HYLE_NO_UNIQUE_ADDRESS Eq eq_{};
  HYLE_NO_UNIQUE_ADDRESS LifetimeState lifetime_{};
  mutable std::mutex rehash_mutex_;
  std::atomic<Table *> current_{nullptr};
#if defined(HYLE_TESTING)
  std::atomic<TableLoadHook> table_load_hook_{nullptr};
  std::atomic<void *> table_load_hook_state_{nullptr};
#endif
};

template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
using RetainedTableConcurrentNodeHashMap =
    ConcurrentNodeHashMap<K, V, Hash, Eq, Alloc,
                          RetiredTableLifetime::kMapDestruction>;

template <class K, class V, class Hash = Hash<K>, class Eq = std::equal_to<>,
          class Alloc = mem::SystemAllocator>
using ReclaimingTableConcurrentNodeHashMap =
    ConcurrentNodeHashMap<K, V, Hash, Eq, Alloc,
                          RetiredTableLifetime::kAfterReaders>;

} // namespace hyle::container
